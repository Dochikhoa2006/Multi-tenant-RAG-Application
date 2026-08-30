from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from backend.config import get_collection_name
from backend.weaviate_client.client import WeaviateManager
from backend.weaviate_client.knowledge import KnowledgeCollection
from backend.weaviate_client.models import (
    ChunkRecord,
    IncompleteDeletionError,
    PartialParagraphUpdateError,
    WeaviateResponseError,
)
from backend.weaviate_client.policy import PolicyCollection


USER_ID = "usr_abc123"
DOCUMENT_ID = "10000000-0000-0000-0000-000000000001"
CHUNK_ID = "20000000-0000-0000-0000-000000000001"
SECOND_CHUNK_ID = "20000000-0000-0000-0000-000000000002"
THIRD_CHUNK_ID = "20000000-0000-0000-0000-000000000003"


def _manager_and_collection() -> tuple[WeaviateManager, MagicMock, MagicMock]:
    client = MagicMock()
    collection = MagicMock()
    client.collections.use.return_value = collection
    manager = WeaviateManager(client=client)
    manager.connect()
    client.connect.reset_mock()
    return manager, client, collection


def _delete_result(
    ids: tuple[str, ...] = (), *, failed_ids: tuple[str, ...] = ()
) -> SimpleNamespace:
    failed = set(failed_ids)
    return SimpleNamespace(
        matches=len(ids),
        successful=len(ids) - len(failed),
        failed=len(failed),
        objects=[
            SimpleNamespace(uuid=UUID(item), successful=item not in failed)
            for item in ids
        ],
    )


def _dry_run_result(ids: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(
        matches=len(ids),
        successful=0,
        failed=0,
        objects=[
            SimpleNamespace(uuid=UUID(item), successful=False) for item in ids
        ],
    )


def _successful_delete(collection: MagicMock, ids: tuple[str, ...] = ()) -> None:
    collection.data.delete_many.side_effect = [
        _delete_result(ids),
        _dry_run_result(),
    ]


def test_insert_chunk_uses_chunk_id_as_uuid() -> None:
    manager, client, collection = _manager_and_collection()
    collection.data.insert.return_value = UUID(CHUNK_ID)
    knowledge = KnowledgeCollection(manager, USER_ID)

    inserted = knowledge.insert_chunk(
        DOCUMENT_ID,
        2,
        CHUNK_ID,
        "  Exact chunk text.\n",
        [0.1, 0.9],
    )

    assert inserted == CHUNK_ID
    client.collections.use.assert_called_once_with(
        get_collection_name(USER_ID, "knowledge_facts")
    )
    collection.data.insert.assert_called_once_with(
        properties={
            "user_id": USER_ID,
            "document_id": DOCUMENT_ID,
            "paragraph_id": 2,
            "chunk_id": CHUNK_ID,
            "raw_text": "  Exact chunk text.\n",
        },
        uuid=CHUNK_ID,
        vector=[0.1, 0.9],
    )


def test_delete_by_document_uses_document_filter() -> None:
    manager, _, collection = _manager_and_collection()
    _successful_delete(collection)
    knowledge = KnowledgeCollection(manager, USER_ID)

    knowledge.delete_by_document(DOCUMENT_ID)

    where = collection.data.delete_many.call_args_list[0].kwargs["where"]
    assert where.target == "document_id"
    assert where.value == DOCUMENT_ID


def test_delete_by_paragraphs_is_scoped_to_document() -> None:
    manager, _, collection = _manager_and_collection()
    _successful_delete(collection)
    knowledge = KnowledgeCollection(manager, USER_ID)

    knowledge.delete_by_paragraphs(DOCUMENT_ID, [2, 3, 2])

    where = collection.data.delete_many.call_args_list[0].kwargs["where"]
    assert len(where.filters) == 2
    document_filter, paragraph_filter = where.filters
    assert document_filter.target == "document_id"
    assert document_filter.value == DOCUMENT_ID
    assert paragraph_filter.target == "paragraph_id"
    assert paragraph_filter.value == [2, 3]


def test_empty_paragraph_delete_is_a_noop() -> None:
    manager, client, collection = _manager_and_collection()
    knowledge = KnowledgeCollection(manager, USER_ID)

    knowledge.delete_by_paragraphs(DOCUMENT_ID, [])

    client.collections.use.assert_not_called()
    collection.data.delete_many.assert_not_called()


def test_update_paragraph_ids_validates_then_updates_each_chunk() -> None:
    manager, _, collection = _manager_and_collection()
    knowledge = KnowledgeCollection(manager, USER_ID)

    knowledge.update_paragraph_ids({CHUNK_ID: 4, SECOND_CHUNK_ID: 5})

    assert collection.data.update.call_args_list[0].kwargs == {
        "uuid": CHUNK_ID,
        "properties": {"paragraph_id": 4},
    }
    assert collection.data.update.call_args_list[1].kwargs == {
        "uuid": SECOND_CHUNK_ID,
        "properties": {"paragraph_id": 5},
    }


def test_invalid_update_mapping_causes_no_partial_mutation() -> None:
    manager, _, collection = _manager_and_collection()
    knowledge = KnowledgeCollection(manager, USER_ID)

    with pytest.raises(ValueError, match="greater than zero"):
        knowledge.update_paragraph_ids({CHUNK_ID: 4, SECOND_CHUNK_ID: 0})

    collection.data.update.assert_not_called()


def test_partial_update_failure_reports_completed_and_failed_chunks() -> None:
    manager, _, collection = _manager_and_collection()
    collection.data.update.side_effect = [None, RuntimeError("update failed")]
    knowledge = KnowledgeCollection(manager, USER_ID)

    with pytest.raises(PartialParagraphUpdateError) as error:
        knowledge.update_paragraph_ids({CHUNK_ID: 4, SECOND_CHUNK_ID: 5})

    assert error.value.completed_chunk_ids == (CHUNK_ID,)
    assert error.value.failed_chunk_id == SECOND_CHUNK_ID
    assert isinstance(error.value.__cause__, RuntimeError)
    assert collection.data.update.call_count == 2


def test_policy_uses_separate_collection_with_same_interface() -> None:
    manager, client, collection = _manager_and_collection()
    collection.data.insert.return_value = UUID(CHUNK_ID)
    policy = PolicyCollection(manager, USER_ID)

    policy.insert_chunk(DOCUMENT_ID, 1, CHUNK_ID, "Policy text", [0.2, 0.8])

    client.collections.use.assert_called_once_with(get_collection_name(USER_ID, "policy"))
    assert collection.data.insert.call_args.kwargs["properties"]["raw_text"] == (
        "Policy text"
    )


def test_knowledge_hybrid_search_keeps_collection_result_order() -> None:
    manager, _, collection = _manager_and_collection()
    collection.query.hybrid.return_value = SimpleNamespace(
        objects=[
            SimpleNamespace(
                uuid=UUID(CHUNK_ID),
                properties={
                    "user_id": USER_ID,
                    "document_id": DOCUMENT_ID,
                    "paragraph_id": 1,
                    "chunk_id": CHUNK_ID,
                    "raw_text": "first",
                },
                vector=[0.1, 0.2],
                metadata=SimpleNamespace(score=0.9),
            ),
            SimpleNamespace(
                uuid=UUID(SECOND_CHUNK_ID),
                properties={
                    "user_id": USER_ID,
                    "document_id": DOCUMENT_ID,
                    "paragraph_id": 2,
                    "chunk_id": SECOND_CHUNK_ID,
                    "raw_text": "second",
                },
                vector=[0.2, 0.1],
                metadata=SimpleNamespace(score=0.8),
            ),
        ]
    )
    knowledge = KnowledgeCollection(manager, USER_ID)

    results = knowledge.hybrid_search("query", [0.5, 0.5], 30)

    assert [item.object_id for item in results] == [CHUNK_ID, SECOND_CHUNK_ID]
    assert collection.query.hybrid.call_args.kwargs["return_properties"] == [
        "user_id",
        "document_id",
        "paragraph_id",
        "chunk_id",
        "raw_text",
    ]


def test_chunk_delete_failure_is_not_swallowed() -> None:
    manager, _, collection = _manager_and_collection()
    collection.data.delete_many.side_effect = RuntimeError("delete failed")
    knowledge = KnowledgeCollection(manager, USER_ID)

    with pytest.raises(RuntimeError, match="delete failed"):
        knowledge.delete_by_document(DOCUMENT_ID)


def test_delete_requires_verbose_success_and_empty_postcondition() -> None:
    manager, _, collection = _manager_and_collection()
    collection.data.delete_many.side_effect = [
        _delete_result((CHUNK_ID,)),
        _dry_run_result((CHUNK_ID,)),
    ]
    knowledge = KnowledgeCollection(manager, USER_ID)

    with pytest.raises(IncompleteDeletionError) as error:
        knowledge.delete_by_document(DOCUMENT_ID)

    assert error.value.report.remaining_ids == (CHUNK_ID,)


def test_delete_rejects_malformed_verbose_response() -> None:
    manager, _, collection = _manager_and_collection()
    collection.data.delete_many.return_value = SimpleNamespace(
        matches=1, successful=1, failed=0, objects=[]
    )
    knowledge = KnowledgeCollection(manager, USER_ID)

    with pytest.raises(WeaviateResponseError, match="one object per match"):
        knowledge.delete_by_document(DOCUMENT_ID)


def test_actual_delete_and_real_shape_empty_dry_run_are_verified() -> None:
    ids = (CHUNK_ID, SECOND_CHUNK_ID, THIRD_CHUNK_ID)
    manager, _, collection = _manager_and_collection()
    collection.data.delete_many.side_effect = [
        _delete_result(ids),
        _dry_run_result(),
    ]
    knowledge = KnowledgeCollection(manager, USER_ID)

    report = knowledge.delete_by_document(DOCUMENT_ID)

    assert report.confirmed
    assert report.matched == 3
    assert report.successful == 3
    assert report.deleted_ids == ids


def test_real_shape_dry_run_reports_all_remaining_ids() -> None:
    ids = (CHUNK_ID, SECOND_CHUNK_ID, THIRD_CHUNK_ID)
    manager, _, collection = _manager_and_collection()
    collection.data.delete_many.side_effect = [
        _delete_result(),
        _dry_run_result(ids),
    ]
    knowledge = KnowledgeCollection(manager, USER_ID)

    with pytest.raises(IncompleteDeletionError) as error:
        knowledge.delete_by_document(DOCUMENT_ID)

    assert error.value.report.remaining_ids == ids


def _snapshot_object(chunk_id: str = CHUNK_ID, paragraph_id: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        uuid=UUID(chunk_id),
        properties={
            "user_id": USER_ID,
            "document_id": DOCUMENT_ID,
            "paragraph_id": paragraph_id,
            "chunk_id": chunk_id,
            "raw_text": "exact text",
        },
        vector=[0.2, 0.8],
    )


def test_snapshot_by_paragraphs_returns_complete_recoverable_records() -> None:
    manager, _, collection = _manager_and_collection()
    collection.data.delete_many.return_value = _dry_run_result((CHUNK_ID,))
    collection.query.fetch_objects_by_ids.return_value = SimpleNamespace(
        objects=[_snapshot_object()]
    )
    knowledge = KnowledgeCollection(manager, USER_ID)

    records = knowledge.snapshot_by_paragraphs(DOCUMENT_ID, [2])

    assert records == (
        ChunkRecord(
            CHUNK_ID, USER_ID, DOCUMENT_ID, 2, CHUNK_ID, "exact text", (0.2, 0.8)
        ),
    )
    assert collection.data.delete_many.call_args.kwargs["dry_run"] is True
    assert collection.query.fetch_objects_by_ids.call_args.kwargs["include_vector"] is True


def test_snapshot_discovery_accepts_real_shape_three_object_dry_run() -> None:
    ids = (CHUNK_ID, SECOND_CHUNK_ID, THIRD_CHUNK_ID)
    manager, _, collection = _manager_and_collection()
    collection.data.delete_many.return_value = _dry_run_result(ids)
    collection.query.fetch_objects_by_ids.return_value = SimpleNamespace(
        objects=[_snapshot_object(chunk_id=item) for item in ids]
    )
    knowledge = KnowledgeCollection(manager, USER_ID)

    records = knowledge.snapshot_by_paragraphs(DOCUMENT_ID, [2])

    assert tuple(record.chunk_id for record in records) == ids


def test_restore_chunks_inserts_missing_records_then_verifies_exact_state() -> None:
    manager, _, collection = _manager_and_collection()
    record = ChunkRecord(
        CHUNK_ID, USER_ID, DOCUMENT_ID, 2, CHUNK_ID, "exact text", (0.2, 0.8)
    )
    collection.query.fetch_objects_by_ids.side_effect = [
        SimpleNamespace(objects=[]),
        SimpleNamespace(objects=[_snapshot_object()]),
    ]
    collection.data.insert.return_value = UUID(CHUNK_ID)
    knowledge = KnowledgeCollection(manager, USER_ID)

    restored = knowledge.restore_chunks((record,))

    assert restored == (CHUNK_ID,)
    collection.data.insert.assert_called_once()


def test_delete_chunks_and_paragraph_verification_use_exact_chunk_ids() -> None:
    manager, _, collection = _manager_and_collection()
    _successful_delete(collection, (CHUNK_ID,))
    knowledge = KnowledgeCollection(manager, USER_ID)

    report = knowledge.delete_chunks([CHUNK_ID])

    assert report.confirmed
    assert report.deleted_ids == (CHUNK_ID,)

    collection.query.fetch_objects_by_ids.return_value = SimpleNamespace(
        objects=[_snapshot_object(paragraph_id=2)]
    )
    knowledge.verify_paragraph_ids({CHUNK_ID: 2})
    with pytest.raises(WeaviateResponseError, match="verification failed"):
        knowledge.verify_paragraph_ids({CHUNK_ID: 3})


@pytest.mark.parametrize("paragraph_id", [0, -1, True, 1.5, "1"])
def test_chunk_paragraph_validation_prevents_insert(paragraph_id: object) -> None:
    manager, _, collection = _manager_and_collection()
    knowledge = KnowledgeCollection(manager, USER_ID)

    with pytest.raises((TypeError, ValueError)):
        knowledge.insert_chunk(
            DOCUMENT_ID,
            paragraph_id,  # type: ignore[arg-type]
            CHUNK_ID,
            "content",
            [0.1],
        )
    collection.data.insert.assert_not_called()


@pytest.mark.parametrize(
    ("document_id", "chunk_id"),
    [("not-a-uuid", CHUNK_ID), (DOCUMENT_ID, "not-a-uuid")],
)
def test_chunk_uuid_validation_prevents_insert(
    document_id: str,
    chunk_id: str,
) -> None:
    manager, _, collection = _manager_and_collection()
    knowledge = KnowledgeCollection(manager, USER_ID)

    with pytest.raises(ValueError, match="UUID"):
        knowledge.insert_chunk(document_id, 1, chunk_id, "content", [0.1])
    collection.data.insert.assert_not_called()


@pytest.mark.parametrize(
    ("raw_text", "vector"),
    [("", [0.1]), (" \n", [0.1]), ("content", []),
     ("content", [float("nan")]), ("content", [float("inf")])],
)
def test_chunk_content_validation_prevents_insert(
    raw_text: str,
    vector: list[float],
) -> None:
    manager, _, collection = _manager_and_collection()
    knowledge = KnowledgeCollection(manager, USER_ID)

    with pytest.raises((TypeError, ValueError)):
        knowledge.insert_chunk(DOCUMENT_ID, 1, CHUNK_ID, raw_text, vector)
    collection.data.insert.assert_not_called()


def test_delete_paragraphs_validates_entire_batch_before_mutation() -> None:
    manager, _, collection = _manager_and_collection()
    knowledge = KnowledgeCollection(manager, USER_ID)

    with pytest.raises(ValueError, match="greater than zero"):
        knowledge.delete_by_paragraphs(DOCUMENT_ID, [1, 0, 2])
    collection.data.delete_many.assert_not_called()


def test_chunk_delete_and_update_uuid_validation_prevents_mutation() -> None:
    manager, _, collection = _manager_and_collection()
    knowledge = KnowledgeCollection(manager, USER_ID)

    with pytest.raises(ValueError, match="UUID"):
        knowledge.delete_by_document("not-a-uuid")
    with pytest.raises(ValueError, match="UUID"):
        knowledge.delete_by_paragraphs("not-a-uuid", [1])
    with pytest.raises(ValueError, match="UUID"):
        knowledge.update_paragraph_ids({"not-a-uuid": 1})

    collection.data.delete_many.assert_not_called()
    collection.data.update.assert_not_called()


@pytest.mark.parametrize(
    ("collection_class", "user_id", "expected_name"),
    [
        (
            KnowledgeCollection,
            USER_ID,
            get_collection_name(USER_ID, "knowledge_facts"),
        ),
        (PolicyCollection, USER_ID, get_collection_name(USER_ID, "policy")),
        (
            KnowledgeCollection,
            "usr_other",
            get_collection_name("usr_other", "knowledge_facts"),
        ),
        (
            PolicyCollection,
            "usr_other",
            get_collection_name("usr_other", "policy"),
        ),
    ],
)
def test_every_chunk_operation_uses_only_its_bound_collection(
    collection_class: type[KnowledgeCollection] | type[PolicyCollection],
    user_id: str,
    expected_name: str,
) -> None:
    client = MagicMock()
    collection = MagicMock()
    collection.data.delete_many.side_effect = [
        _delete_result(),
        _delete_result(),
        _delete_result(),
        _delete_result(),
    ]
    collection.data.insert.return_value = UUID(CHUNK_ID)
    collection.query.hybrid.return_value = SimpleNamespace(
        objects=[
            SimpleNamespace(
                uuid=UUID(CHUNK_ID),
                properties={
                    "user_id": user_id,
                    "document_id": DOCUMENT_ID,
                    "paragraph_id": 1,
                    "chunk_id": CHUNK_ID,
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
    chunks = collection_class(manager, user_id)

    chunks.insert_chunk(DOCUMENT_ID, 1, CHUNK_ID, "content", [0.1])
    chunks.delete_by_document(DOCUMENT_ID)
    chunks.delete_by_paragraphs(DOCUMENT_ID, [1])
    chunks.update_paragraph_ids({CHUNK_ID: 2})
    chunks.hybrid_search("query", [0.1], 20)

    assert [call.args[0] for call in client.collections.use.call_args_list] == [
        expected_name,
        expected_name,
        expected_name,
        expected_name,
        expected_name,
    ]
    assert collection.data.insert.call_args.kwargs["properties"]["user_id"] == user_id
