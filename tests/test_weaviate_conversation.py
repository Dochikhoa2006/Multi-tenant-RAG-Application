from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from weaviate.classes.config import DataType
from weaviate.classes.query import HybridFusion

from backend.config import (
    WEAVIATE_GRPC_PORT,
    WEAVIATE_GRPC_SECURE,
    WEAVIATE_URL,
    get_collection_name,
)
from backend.model_config import CONVERSATION_SEARCH, HYBRID_SEARCH
from backend.weaviate_client.client import WeaviateManager
from backend.weaviate_client.conversation import ConversationCollection
from backend.weaviate_client.models import (
    DeletionReport,
    IncompatibleCollectionSchemaError,
    IncompleteDeletionError,
    SearchResult,
    UserIsolationError,
    WeaviateResponseError,
)


USER_ID = "usr_abc123"
CONVERSATION_ID = "00000000-0000-0000-0000-000000000001"
SECOND_CONVERSATION_ID = "00000000-0000-0000-0000-000000000002"
COLLECTION_NAMES = (
    get_collection_name(USER_ID, "conversations"),
    get_collection_name(USER_ID, "knowledge_facts"),
    get_collection_name(USER_ID, "policy"),
)
EXISTING_COLLECTION_SUBSETS = [
    frozenset(
        name for index, name in enumerate(COLLECTION_NAMES) if mask & (1 << index)
    )
    for mask in range(8)
]


def _property_config(
    name: str,
    data_type: DataType,
    index_filterable: bool,
    index_searchable: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        data_type=data_type,
        index_filterable=index_filterable,
        index_searchable=index_searchable,
        index_range_filters=False,
    )


def _compatible_config(name: str) -> SimpleNamespace:
    if name.endswith("_Conversations"):
        properties = [
            _property_config("user_id", DataType.TEXT, True, False),
            _property_config("conversation_id", DataType.UUID, True, False),
            _property_config("raw_text", DataType.TEXT, False, True),
        ]
    else:
        properties = [
            _property_config("user_id", DataType.TEXT, True, False),
            _property_config("document_id", DataType.UUID, True, False),
            _property_config("paragraph_id", DataType.INT, True, False),
            _property_config("chunk_id", DataType.UUID, True, False),
            _property_config("raw_text", DataType.TEXT, False, True),
        ]
    return SimpleNamespace(
        name=name,
        properties=properties,
        references=[],
        vector_config={
            "default": SimpleNamespace(
                vectorizer=SimpleNamespace(
                    vectorizer="none",
                    source_properties=None,
                )
            )
        },
    )


def _configure_existing_schema_reads(client: MagicMock) -> None:
    def use(name: str) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(get=MagicMock(return_value=_compatible_config(name)))
        )

    client.collections.use.side_effect = use


def _manager_and_collection() -> tuple[WeaviateManager, MagicMock, MagicMock]:
    client = MagicMock()
    collection = MagicMock()
    client.collections.use.return_value = collection
    manager = WeaviateManager(client=client)
    manager.connect()
    client.connect.reset_mock()
    return manager, client, collection


def _delete_result(
    object_ids: list[str],
    statuses: list[bool],
    *,
    successful: int,
    failed: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        matches=len(object_ids),
        successful=successful,
        failed=failed,
        objects=[
            SimpleNamespace(uuid=UUID(object_id), successful=item_status)
            for object_id, item_status in zip(object_ids, statuses, strict=True)
        ],
    )


def _dry_run_result(object_ids: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        matches=len(object_ids),
        successful=0,
        failed=0,
        objects=[SimpleNamespace(uuid=UUID(object_id)) for object_id in object_ids],
    )


def test_manager_connection_lifecycle_is_idempotent() -> None:
    client = MagicMock()
    manager = WeaviateManager(client=client)

    with pytest.raises(RuntimeError, match="connect"):
        _ = manager.client

    assert manager.connect() is client
    assert manager.connect() is client
    client.connect.assert_called_once_with()

    manager.disconnect()
    manager.disconnect()
    client.close.assert_called_once_with()

    with pytest.raises(RuntimeError, match="connect"):
        _ = manager.client


def test_connection_failure_is_propagated_and_manager_stays_disconnected() -> None:
    client = MagicMock()
    client.connect.side_effect = RuntimeError("connection failed")
    manager = WeaviateManager(client=client)

    with pytest.raises(RuntimeError, match="connection failed"):
        manager.connect()
    with pytest.raises(RuntimeError, match="connect"):
        _ = manager.client


def test_manager_builds_default_client_from_connection_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection_params = object()
    sdk_client = MagicMock()
    from_url = MagicMock(return_value=connection_params)
    client_constructor = MagicMock(return_value=sdk_client)
    monkeypatch.setattr(
        "backend.weaviate_client.client.ConnectionParams.from_url",
        from_url,
    )
    monkeypatch.setattr(
        "backend.weaviate_client.client.weaviate.WeaviateClient",
        client_constructor,
    )

    manager = WeaviateManager()

    assert manager.connect() is sdk_client
    from_url.assert_called_once_with(
        WEAVIATE_URL,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=WEAVIATE_GRPC_SECURE,
    )
    client_constructor.assert_called_once_with(connection_params=connection_params)
    sdk_client.connect.assert_called_once_with()


def test_ensure_user_collections_creates_exact_schemas_once() -> None:
    client = MagicMock()
    manager = WeaviateManager(client=client)
    manager.connect()
    created: set[str] = set()
    client.collections.exists.side_effect = lambda name: name in created
    _configure_existing_schema_reads(client)

    def record_create(**kwargs: object) -> None:
        created.add(str(kwargs["name"]))

    client.collections.create.side_effect = record_create

    manager.ensure_user_collections(USER_ID)
    manager.ensure_user_collections(USER_ID)

    assert created == {
        COLLECTION_NAMES[0],
        COLLECTION_NAMES[1],
        COLLECTION_NAMES[2],
    }
    assert client.collections.create.call_count == 3
    calls = {
        call.kwargs["name"]: call.kwargs
        for call in client.collections.create.call_args_list
    }
    conversation_properties = calls[COLLECTION_NAMES[0]]["properties"]
    chunk_properties = calls[COLLECTION_NAMES[1]]["properties"]
    assert [item.name for item in conversation_properties] == [
        "user_id",
        "conversation_id",
        "raw_text",
    ]
    assert [item.name for item in chunk_properties] == [
        "user_id",
        "document_id",
        "paragraph_id",
        "chunk_id",
        "raw_text",
    ]
    assert [item.dataType for item in conversation_properties] == [
        DataType.TEXT,
        DataType.UUID,
        DataType.TEXT,
    ]
    assert [item.dataType for item in chunk_properties] == [
        DataType.TEXT,
        DataType.UUID,
        DataType.INT,
        DataType.UUID,
        DataType.TEXT,
    ]
    assert all(item.name != "vector" for item in conversation_properties)
    assert all(item.name != "vector" for item in chunk_properties)
    assert [item.indexFilterable for item in conversation_properties] == [
        True,
        True,
        False,
    ]
    assert [item.indexSearchable for item in conversation_properties] == [
        False,
        False,
        True,
    ]
    assert [item.indexFilterable for item in chunk_properties] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert [item.indexSearchable for item in chunk_properties] == [
        False,
        False,
        False,
        False,
        True,
    ]
    for collection_name in COLLECTION_NAMES:
        assert calls[collection_name]["vector_config"].model_dump()["vectorizer"] == {}
    assert [item.name for item in calls[COLLECTION_NAMES[2]]["properties"]] == [
        item.name for item in chunk_properties
    ]


@pytest.mark.parametrize("existing", EXISTING_COLLECTION_SUBSETS)
def test_collection_creation_is_idempotent_for_every_existing_subset(
    existing: frozenset[str],
) -> None:
    client = MagicMock()
    manager = WeaviateManager(client=client)
    manager.connect()
    client.collections.exists.side_effect = lambda name: name in existing
    _configure_existing_schema_reads(client)

    manager.ensure_user_collections(USER_ID)

    assert [call.args[0] for call in client.collections.exists.call_args_list] == list(
        COLLECTION_NAMES
    )
    assert [
        call.kwargs["name"] for call in client.collections.create.call_args_list
    ] == [name for name in COLLECTION_NAMES if name not in existing]


@pytest.mark.parametrize(
    "incompatibility",
    [
        "missing_property",
        "extra_property",
        "wrong_type",
        "wrong_filter_index",
        "wrong_search_index",
        "reference",
        "missing_vector",
        "multiple_vectors",
        "provider_vectorizer",
        "vector_source_property",
        "mismatched_name",
    ],
)
def test_existing_incompatible_schema_fails_before_any_creation(
    incompatibility: str,
) -> None:
    client = MagicMock()
    manager = WeaviateManager(client=client)
    manager.connect()
    existing_name = COLLECTION_NAMES[1]
    config = _compatible_config(existing_name)

    if incompatibility == "missing_property":
        config.properties.pop()
    elif incompatibility == "extra_property":
        config.properties.append(_property_config("extra", DataType.TEXT, True, False))
    elif incompatibility == "wrong_type":
        config.properties[1].data_type = DataType.TEXT
    elif incompatibility == "wrong_filter_index":
        config.properties[0].index_filterable = False
    elif incompatibility == "wrong_search_index":
        config.properties[-1].index_searchable = False
    elif incompatibility == "reference":
        config.references = [SimpleNamespace(name="parent")]
    elif incompatibility == "missing_vector":
        config.vector_config = {}
    elif incompatibility == "multiple_vectors":
        config.vector_config["second"] = config.vector_config["default"]
    elif incompatibility == "provider_vectorizer":
        config.vector_config["default"].vectorizer.vectorizer = "text2vec-openai"
    elif incompatibility == "vector_source_property":
        config.vector_config["default"].vectorizer.source_properties = ["raw_text"]
    elif incompatibility == "mismatched_name":
        config.name = COLLECTION_NAMES[0]

    client.collections.exists.side_effect = lambda name: name == existing_name
    client.collections.use.return_value.config.get.return_value = config

    with pytest.raises(IncompatibleCollectionSchemaError, match="incompatible"):
        manager.ensure_user_collections(USER_ID)

    client.collections.create.assert_not_called()


def test_compatible_existing_schema_is_inspected_without_recreation() -> None:
    client = MagicMock()
    manager = WeaviateManager(client=client)
    manager.connect()
    client.collections.exists.return_value = True
    _configure_existing_schema_reads(client)

    manager.ensure_user_collections(USER_ID)

    assert [call.args[0] for call in client.collections.use.call_args_list] == list(
        COLLECTION_NAMES
    )
    client.collections.create.assert_not_called()


def test_collection_create_failure_is_propagated_and_retry_fills_missing() -> None:
    client = MagicMock()
    manager = WeaviateManager(client=client)
    manager.connect()
    created: set[str] = set()
    failed_once = False
    client.collections.exists.side_effect = lambda name: name in created
    _configure_existing_schema_reads(client)

    def create(**kwargs: object) -> None:
        nonlocal failed_once
        name = str(kwargs["name"])
        if name == COLLECTION_NAMES[1] and not failed_once:
            failed_once = True
            raise RuntimeError("create failed")
        created.add(name)

    client.collections.create.side_effect = create

    with pytest.raises(RuntimeError, match="create failed"):
        manager.ensure_user_collections(USER_ID)
    assert created == {COLLECTION_NAMES[0]}

    manager.ensure_user_collections(USER_ID)
    assert created == set(COLLECTION_NAMES)


def test_insert_uses_conversation_id_as_object_uuid_and_native_vector() -> None:
    manager, client, collection = _manager_and_collection()
    collection.data.insert.return_value = UUID(CONVERSATION_ID)
    conversations = ConversationCollection(manager, USER_ID)

    inserted = conversations.insert(
        CONVERSATION_ID,
        "Question\nAnswer",
        [0.1, 0.2],
    )

    assert inserted == CONVERSATION_ID
    client.collections.use.assert_called_once_with(COLLECTION_NAMES[0])
    collection.data.insert.assert_called_once_with(
        properties={
            "user_id": USER_ID,
            "conversation_id": CONVERSATION_ID,
            "raw_text": "Question\nAnswer",
        },
        uuid=CONVERSATION_ID,
        vector=[0.1, 0.2],
    )


def test_delete_and_batch_delete_use_object_ids() -> None:
    manager, _, collection = _manager_and_collection()
    conversations = ConversationCollection(manager, USER_ID)

    conversations.delete(CONVERSATION_ID)
    conversations.delete_batch(
        [CONVERSATION_ID, SECOND_CONVERSATION_ID, CONVERSATION_ID]
    )

    collection.data.delete_by_id.assert_called_once_with(CONVERSATION_ID)
    where = collection.data.delete_many.call_args.kwargs["where"]
    assert where.target == "_id"
    assert where.value == [CONVERSATION_ID, SECOND_CONVERSATION_ID]


def test_empty_batch_is_a_noop() -> None:
    manager, client, collection = _manager_and_collection()
    conversations = ConversationCollection(manager, USER_ID)

    conversations.delete_batch([])

    client.collections.use.assert_not_called()
    collection.data.delete_many.assert_not_called()


def test_verified_batch_delete_requires_empty_dry_run_postcondition() -> None:
    manager, _, collection = _manager_and_collection()
    collection.data.delete_many.side_effect = [
        _delete_result(
            [CONVERSATION_ID, SECOND_CONVERSATION_ID],
            [True, True],
            successful=2,
            failed=0,
        ),
        _dry_run_result([]),
    ]
    conversations = ConversationCollection(manager, USER_ID)

    report = conversations.delete_batch_verified(
        [CONVERSATION_ID, SECOND_CONVERSATION_ID]
    )

    assert report == DeletionReport(
        matched=2,
        successful=2,
        failed=0,
        deleted_ids=(CONVERSATION_ID, SECOND_CONVERSATION_ID),
        remaining_ids=(),
    )
    first, second = collection.data.delete_many.call_args_list
    assert first.kwargs["verbose"] is True
    assert "dry_run" not in first.kwargs
    assert second.kwargs["verbose"] is True
    assert second.kwargs["dry_run"] is True


def test_partial_verified_batch_delete_is_retryable() -> None:
    manager, _, collection = _manager_and_collection()
    collection.data.delete_many.side_effect = [
        _delete_result(
            [CONVERSATION_ID, SECOND_CONVERSATION_ID],
            [True, False],
            successful=1,
            failed=1,
        ),
        _dry_run_result([SECOND_CONVERSATION_ID]),
        _delete_result(
            [SECOND_CONVERSATION_ID],
            [True],
            successful=1,
            failed=0,
        ),
        _dry_run_result([]),
    ]
    conversations = ConversationCollection(manager, USER_ID)

    with pytest.raises(IncompleteDeletionError) as error:
        conversations.delete_batch_verified(
            [CONVERSATION_ID, SECOND_CONVERSATION_ID]
        )
    assert error.value.report.remaining_ids == (SECOND_CONVERSATION_ID,)

    retry = conversations.delete_batch_verified(
        [CONVERSATION_ID, SECOND_CONVERSATION_ID]
    )
    assert retry.confirmed
    assert retry.deleted_ids == (SECOND_CONVERSATION_ID,)


def test_verified_batch_delete_rejects_malformed_or_cross_scope_results() -> None:
    manager, _, collection = _manager_and_collection()
    conversations = ConversationCollection(manager, USER_ID)
    collection.data.delete_many.side_effect = [
        _delete_result(
            [CONVERSATION_ID],
            [True],
            successful=1,
            failed=0,
        ),
        SimpleNamespace(matches=1, successful=0, failed=0, objects=[]),
    ]
    with pytest.raises(WeaviateResponseError, match="one object per match"):
        conversations.delete_batch_verified([CONVERSATION_ID])

    outside_id = "00000000-0000-0000-0000-000000000099"
    collection.data.delete_many.side_effect = [
        _delete_result([outside_id], [True], successful=1, failed=0),
        _dry_run_result([]),
    ]
    with pytest.raises(WeaviateResponseError, match="scope"):
        conversations.delete_batch_verified([CONVERSATION_ID])


def test_verified_batch_delete_accepts_already_absent_ids_and_empty_input() -> None:
    manager, client, collection = _manager_and_collection()
    collection.data.delete_many.side_effect = [
        _delete_result([], [], successful=0, failed=0),
        _dry_run_result([]),
    ]
    conversations = ConversationCollection(manager, USER_ID)

    assert conversations.delete_batch_verified([CONVERSATION_ID]).confirmed
    assert conversations.delete_batch_verified([]) == DeletionReport(
        0, 0, 0, (), ()
    )
    assert collection.data.delete_many.call_count == 2


def test_hybrid_search_returns_typed_results_and_expected_query() -> None:
    manager, _, collection = _manager_and_collection()
    collection.query.hybrid.return_value = SimpleNamespace(
        objects=[
            SimpleNamespace(
                uuid=UUID(CONVERSATION_ID),
                properties={
                    "user_id": USER_ID,
                    "conversation_id": CONVERSATION_ID,
                    "raw_text": "Question\nAnswer",
                },
                metadata=SimpleNamespace(score=0.91),
            )
        ]
    )
    conversations = ConversationCollection(manager, USER_ID)

    results = conversations.hybrid_search("database indexing", [0.3, 0.7], 20)

    assert results == [
        SearchResult(
            object_id=CONVERSATION_ID,
            properties={
                "user_id": USER_ID,
                "conversation_id": CONVERSATION_ID,
                "raw_text": "Question\nAnswer",
            },
            score=0.91,
        )
    ]
    kwargs = collection.query.hybrid.call_args.kwargs
    assert kwargs["query"] == "database indexing"
    assert kwargs["vector"] == [0.3, 0.7]
    assert HYBRID_SEARCH.alpha == 0.70
    assert kwargs["alpha"] == 0.70
    assert kwargs["query_properties"] == ["raw_text"]
    assert kwargs["fusion_type"] is HybridFusion.RELATIVE_SCORE
    assert kwargs["limit"] == 20
    assert kwargs["include_vector"] is False
    assert kwargs["diversity_selection"].limit == 5
    assert kwargs["diversity_selection"].balance == 0.70
    assert CONVERSATION_SEARCH.candidate_count == 20
    assert CONVERSATION_SEARCH.final_count == 5
    assert CONVERSATION_SEARCH.mmr_lambda == 0.70
    assert kwargs["return_metadata"].score is True


def test_sdk_errors_are_not_swallowed() -> None:
    manager, _, collection = _manager_and_collection()
    collection.data.insert.side_effect = RuntimeError("write failed")
    conversations = ConversationCollection(manager, USER_ID)

    with pytest.raises(RuntimeError, match="write failed"):
        conversations.insert(CONVERSATION_ID, "content", [0.1])


def test_delete_failure_is_not_swallowed() -> None:
    manager, _, collection = _manager_and_collection()
    collection.data.delete_by_id.side_effect = RuntimeError("delete failed")
    conversations = ConversationCollection(manager, USER_ID)

    with pytest.raises(RuntimeError, match="delete failed"):
        conversations.delete(CONVERSATION_ID)


def test_delete_uuid_validation_prevents_single_and_batch_mutation() -> None:
    manager, _, collection = _manager_and_collection()
    conversations = ConversationCollection(manager, USER_ID)

    with pytest.raises(ValueError, match="UUID"):
        conversations.delete("not-a-uuid")
    with pytest.raises(ValueError, match="UUID"):
        conversations.delete_batch([CONVERSATION_ID, "not-a-uuid"])

    collection.data.delete_by_id.assert_not_called()
    collection.data.delete_many.assert_not_called()


@pytest.mark.parametrize("conversation_id", ["", "not-a-uuid", None, 42])
def test_conversation_uuid_validation_prevents_insert(
    conversation_id: object,
) -> None:
    manager, _, collection = _manager_and_collection()
    conversations = ConversationCollection(manager, USER_ID)

    with pytest.raises((TypeError, ValueError)):
        conversations.insert(conversation_id, "content", [0.1])  # type: ignore[arg-type]
    collection.data.insert.assert_not_called()


@pytest.mark.parametrize("raw_text", ["", "   \n\t"])
def test_conversation_text_validation_prevents_insert(raw_text: str) -> None:
    manager, _, collection = _manager_and_collection()
    conversations = ConversationCollection(manager, USER_ID)

    with pytest.raises(ValueError, match="raw_text"):
        conversations.insert(CONVERSATION_ID, raw_text, [0.1])
    collection.data.insert.assert_not_called()


@pytest.mark.parametrize(
    "vector",
    [[], [float("nan")], [float("inf")], [True], ["not-a-number"]],
)
def test_conversation_vector_validation_prevents_insert(vector: list[object]) -> None:
    manager, _, collection = _manager_and_collection()
    conversations = ConversationCollection(manager, USER_ID)

    with pytest.raises((TypeError, ValueError)):
        conversations.insert(CONVERSATION_ID, "content", vector)  # type: ignore[arg-type]
    collection.data.insert.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_search_top_k_validation_prevents_query(top_k: object) -> None:
    manager, _, collection = _manager_and_collection()
    conversations = ConversationCollection(manager, USER_ID)

    with pytest.raises((TypeError, ValueError)):
        conversations.hybrid_search("query", [0.1], top_k)  # type: ignore[arg-type]
    collection.query.hybrid.assert_not_called()


def test_native_mmr_final_limit_does_not_exceed_direct_candidate_limit() -> None:
    manager, _, collection = _manager_and_collection()
    collection.query.hybrid.return_value = SimpleNamespace(objects=[])
    conversations = ConversationCollection(manager, USER_ID)

    assert conversations.hybrid_search("query", [0.1], 3) == []

    kwargs = collection.query.hybrid.call_args.kwargs
    assert kwargs["limit"] == 3
    assert kwargs["diversity_selection"].limit == 3
    assert kwargs["diversity_selection"].balance == 0.70


@pytest.mark.parametrize(
    ("query_text", "query_vector"),
    [
        ("", [0.1]),
        (" \n", [0.1]),
        ("query", []),
        ("query", [float("nan")]),
        ("query", [float("-inf")]),
    ],
)
def test_search_query_validation_prevents_sdk_call(
    query_text: str,
    query_vector: list[float],
) -> None:
    manager, _, collection = _manager_and_collection()
    conversations = ConversationCollection(manager, USER_ID)

    with pytest.raises((TypeError, ValueError)):
        conversations.hybrid_search(query_text, query_vector, 20)
    collection.query.hybrid.assert_not_called()


def test_search_ignores_unrequested_response_vector_data() -> None:
    manager, _, collection = _manager_and_collection()
    collection.query.hybrid.return_value = SimpleNamespace(
        objects=[
            SimpleNamespace(
                uuid=UUID(CONVERSATION_ID),
                properties={
                    "user_id": USER_ID,
                    "conversation_id": CONVERSATION_ID,
                    "raw_text": "content",
                },
                vector={"unexpected": [float("nan")]},
                metadata=SimpleNamespace(score=0.8),
            )
        ]
    )
    conversations = ConversationCollection(manager, USER_ID)

    results = conversations.hybrid_search("query", [0.1], 20)

    assert results[0].vector is None
    assert collection.query.hybrid.call_args.kwargs["include_vector"] is False


@pytest.mark.parametrize("bad_score", [None, float("nan"), float("inf"), "bad"])
def test_missing_or_malformed_response_scores_are_rejected(bad_score: object) -> None:
    manager, _, collection = _manager_and_collection()
    collection.query.hybrid.return_value = SimpleNamespace(
        objects=[
            SimpleNamespace(
                uuid=UUID(CONVERSATION_ID),
                properties={
                    "user_id": USER_ID,
                    "conversation_id": CONVERSATION_ID,
                    "raw_text": "content",
                },
                vector=[0.1],
                metadata=SimpleNamespace(score=bad_score),
            )
        ]
    )
    conversations = ConversationCollection(manager, USER_ID)

    with pytest.raises(WeaviateResponseError, match="score"):
        conversations.hybrid_search("query", [0.1], 20)


def test_cross_user_search_result_is_rejected() -> None:
    manager, _, collection = _manager_and_collection()
    collection.query.hybrid.return_value = SimpleNamespace(
        objects=[
            SimpleNamespace(
                uuid=UUID(CONVERSATION_ID),
                properties={
                    "user_id": "usr_other",
                    "conversation_id": CONVERSATION_ID,
                    "raw_text": "content",
                },
                vector=[0.1],
                metadata=SimpleNamespace(score=0.8),
            )
        ]
    )
    conversations = ConversationCollection(manager, USER_ID)

    with pytest.raises(UserIsolationError, match="user_id"):
        conversations.hybrid_search("query", [0.1], 20)


@pytest.mark.parametrize(
    ("object_id", "business_id"),
    [("not-a-uuid", CONVERSATION_ID), (CONVERSATION_ID, "not-a-uuid"),
     (CONVERSATION_ID, SECOND_CONVERSATION_ID)],
)
def test_malformed_or_mismatched_result_ids_are_rejected(
    object_id: str,
    business_id: str,
) -> None:
    manager, _, collection = _manager_and_collection()
    collection.query.hybrid.return_value = SimpleNamespace(
        objects=[
            SimpleNamespace(
                uuid=object_id,
                properties={
                    "user_id": USER_ID,
                    "conversation_id": business_id,
                    "raw_text": "content",
                },
                vector=[0.1],
                metadata=SimpleNamespace(score=0.8),
            )
        ]
    )
    conversations = ConversationCollection(manager, USER_ID)

    with pytest.raises(WeaviateResponseError, match="UUID|conversation_id"):
        conversations.hybrid_search("query", [0.1], 20)


@pytest.mark.parametrize("properties", [None, [], "not-a-mapping"])
def test_malformed_result_properties_are_rejected(properties: object) -> None:
    manager, _, collection = _manager_and_collection()
    collection.query.hybrid.return_value = SimpleNamespace(
        objects=[
            SimpleNamespace(
                uuid=UUID(CONVERSATION_ID),
                properties=properties,
                vector=[0.1],
                metadata=SimpleNamespace(score=0.8),
            )
        ]
    )
    conversations = ConversationCollection(manager, USER_ID)

    with pytest.raises(WeaviateResponseError, match="properties"):
        conversations.hybrid_search("query", [0.1], 20)


@pytest.mark.parametrize(
    ("user_id", "expected_name"),
    [
        (USER_ID, get_collection_name(USER_ID, "conversations")),
        ("usr_other", get_collection_name("usr_other", "conversations")),
    ],
)
def test_every_conversation_operation_uses_only_its_bound_collection(
    user_id: str,
    expected_name: str,
) -> None:
    client = MagicMock()
    collection = MagicMock()
    collection.data.insert.return_value = UUID(CONVERSATION_ID)
    collection.query.hybrid.return_value = SimpleNamespace(
        objects=[
            SimpleNamespace(
                uuid=UUID(CONVERSATION_ID),
                properties={
                    "user_id": user_id,
                    "conversation_id": CONVERSATION_ID,
                    "raw_text": "content",
                },
                vector=[0.1],
                metadata=SimpleNamespace(score=0.8),
            )
        ]
    )
    client.collections.use.return_value = collection
    manager = WeaviateManager(client=client)
    manager.connect()
    client.collections.use.reset_mock()
    conversations = ConversationCollection(manager, user_id)

    conversations.insert(CONVERSATION_ID, "content", [0.1])
    conversations.delete(CONVERSATION_ID)
    conversations.delete_batch([CONVERSATION_ID])
    conversations.hybrid_search("query", [0.1], 20)

    assert [call.args[0] for call in client.collections.use.call_args_list] == [
        expected_name,
        expected_name,
        expected_name,
        expected_name,
    ]
    assert collection.data.insert.call_args.kwargs["properties"]["user_id"] == user_id
