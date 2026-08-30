from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from weaviate.classes.config import DataType

from backend.config import get_collection_name
from backend.model_config import EMBEDDING_MODEL
from backend.weaviate_client.client import WeaviateManager, _collection_description
from scripts.migrate_onnx_vectors import (
    VectorMigrationError,
    export_collections,
    rebuild_collections,
    verify_collections,
)


USER_ID = "usr_migration"
DOCUMENT_ID = "91000000-0000-0000-0000-000000000001"


def _property(
    name: str, data_type: DataType, filterable: bool, searchable: bool
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        data_type=data_type,
        index_filterable=filterable,
        index_searchable=searchable,
        index_range_filters=False,
    )


def _config(name: str, description: str = "legacy") -> SimpleNamespace:
    if name.endswith("_Conversations"):
        properties = [
            _property("user_id", DataType.TEXT, True, False),
            _property("conversation_id", DataType.UUID, True, False),
            _property("raw_text", DataType.TEXT, False, True),
        ]
    else:
        properties = [
            _property("user_id", DataType.TEXT, True, False),
            _property("document_id", DataType.UUID, True, False),
            _property("paragraph_id", DataType.INT, True, False),
            _property("chunk_id", DataType.UUID, True, False),
            _property("raw_text", DataType.TEXT, False, True),
        ]
    return SimpleNamespace(
        name=name,
        description=description,
        properties=properties,
        references=[],
        vector_config={
            "default": SimpleNamespace(
                vectorizer=SimpleNamespace(vectorizer="none", source_properties=None)
            )
        },
    )


class FakeConfig:
    def __init__(self, value: SimpleNamespace) -> None:
        self.value = value
        self.updates: list[str] = []

    def get(self) -> SimpleNamespace:
        return self.value

    def update(self, *, description: str) -> None:
        self.value.description = description
        self.updates.append(description)


class FakeData:
    def __init__(self, collection: "FakeCollection") -> None:
        self.collection = collection

    def insert_many(self, objects: list[object]) -> SimpleNamespace:
        for item in objects:
            self.collection.items.append(
                SimpleNamespace(
                    uuid=item.uuid,
                    properties=dict(item.properties),
                    vector=list(item.vector),
                )
            )
        return SimpleNamespace(has_errors=False)


class FakeCollection:
    def __init__(
        self,
        name: str,
        items: list[SimpleNamespace] | None = None,
        *,
        description: str = "legacy",
    ) -> None:
        self.name = name
        self.items = list(items or [])
        self.config = FakeConfig(_config(name, description))
        self.data = FakeData(self)
        self.iterator_calls: list[dict[str, object]] = []

    def iterator(self, **kwargs: object) -> list[SimpleNamespace]:
        self.iterator_calls.append(dict(kwargs))
        return list(self.items)


class FakeCollections:
    def __init__(self, values: dict[str, FakeCollection]) -> None:
        self.values = values
        self.deleted: list[str] = []
        self.created: list[str] = []

    def list_all(self, *, simple: bool) -> list[str]:
        assert simple is True
        return list(self.values)

    def exists(self, name: str) -> bool:
        return name in self.values

    def use(self, name: str) -> FakeCollection:
        return self.values[name]

    def delete(self, name: str) -> None:
        self.deleted.append(name)
        del self.values[name]

    def create(self, *, name: str, description: str, **kwargs: object) -> FakeCollection:
        assert kwargs["properties"]
        assert kwargs["vector_config"] is not None
        collection = FakeCollection(name, description=description)
        self.values[name] = collection
        self.created.append(name)
        return collection


class FakeClient:
    def __init__(self, collections: FakeCollections) -> None:
        self.collections = collections

    def connect(self) -> None:
        return None

    def close(self) -> None:
        return None


def _record(collection_type: str, index: int) -> SimpleNamespace:
    object_id = f"92000000-0000-0000-0000-{index:012d}"
    if collection_type == "conversations":
        properties = {
            "user_id": USER_ID,
            "conversation_id": object_id,
            "raw_text": "Question and answer",
        }
    else:
        properties = {
            "user_id": USER_ID,
            "document_id": DOCUMENT_ID,
            "paragraph_id": 1,
            "chunk_id": object_id,
            "raw_text": f"{collection_type} chunk",
        }
    return SimpleNamespace(uuid=object_id, properties=properties, vector=[0.5, 0.5])


def _manager() -> tuple[WeaviateManager, FakeCollections]:
    values: dict[str, FakeCollection] = {}
    for index, collection_type in enumerate(
        ("conversations", "knowledge_facts", "policy"), start=1
    ):
        name = get_collection_name(USER_ID, collection_type)
        values[name] = FakeCollection(name, [_record(collection_type, index)])
    # A noncanonical collection must never be exported or mutated.
    values["Unrelated"] = FakeCollection("Unrelated")
    collections = FakeCollections(values)
    manager = WeaviateManager(FakeClient(collections))
    manager.connect()
    return manager, collections


class UnitEmbedder:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_after = fail_after

    def embed_many(self, texts: list[str], *, model: str) -> list[list[float]]:
        assert model == EMBEDDING_MODEL
        self.calls.append(list(texts))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("injected embedding failure")
        return [[1.0] + [0.0] * 767 for _ in texts]


def test_export_and_rebuild_all_canonical_collections_losslessly(tmp_path: Path) -> None:
    manager, collections = _manager()
    source_collections = {
        name: collection
        for name, collection in collections.values.items()
        if name != "Unrelated"
    }

    manifest = export_collections(manager, tmp_path)

    assert manifest.is_file()
    assert len(list(tmp_path.glob("*.jsonl"))) == 3
    assert all(
        collection.iterator_calls == [
            {
                "include_vector": False,
                "return_properties": list(collection.items[0].properties),
            }
        ]
        for collection in source_collections.values()
    )
    embedder = UnitEmbedder()
    rebuild_collections(manager, tmp_path, embedder)
    verify_collections(manager, tmp_path)

    canonical_names = set(source_collections)
    assert set(collections.deleted) == canonical_names
    assert set(collections.created) == canonical_names
    assert "Unrelated" in collections.values
    assert all(
        collections.values[name].config.value.description == _collection_description()
        for name in canonical_names
    )
    assert all(
        len(collections.values[name].items[0].vector) == 768
        and collections.values[name].items[0].vector[0] == 1.0
        for name in canonical_names
    )
    assert all(
        any(call["include_vector"] is True for call in collections.values[name].iterator_calls)
        for name in canonical_names
    )


def test_rebuild_preflight_failure_deletes_nothing(tmp_path: Path) -> None:
    manager, collections = _manager()
    export_collections(manager, tmp_path)
    embedder = UnitEmbedder(fail_after=0)

    with pytest.raises(RuntimeError, match="injected embedding failure"):
        rebuild_collections(manager, tmp_path, embedder)

    assert collections.deleted == []
    assert collections.created == []
    assert all(
        collection.config.updates == []
        for name, collection in collections.values.items()
        if name != "Unrelated"
    )


def test_partial_rebuild_never_marks_any_collection_ready(tmp_path: Path) -> None:
    manager, collections = _manager()
    export_collections(manager, tmp_path)
    # First call is the preflight, second rebuilds one collection, third fails.
    embedder = UnitEmbedder(fail_after=2)

    with pytest.raises(RuntimeError, match="injected embedding failure"):
        rebuild_collections(manager, tmp_path, embedder)

    canonical = [
        collection
        for name, collection in collections.values.items()
        if name != "Unrelated"
    ]
    assert canonical
    assert all(
        collection.config.value.description == _collection_description("rebuilding")
        for collection in canonical
    )

    # A maintenance retry starts solely from the checksummed raw export,
    # recreates incomplete collections, and reaches one consistent new profile.
    rebuild_collections(manager, tmp_path, UnitEmbedder())
    verify_collections(manager, tmp_path)
    assert all(
        collection.config.value.description == _collection_description()
        for name, collection in collections.values.items()
        if name != "Unrelated"
    )


def test_export_requires_complete_three_collection_user_set(tmp_path: Path) -> None:
    manager, collections = _manager()
    collections.delete(get_collection_name(USER_ID, "policy"))

    with pytest.raises(VectorMigrationError, match="exactly the three"):
        export_collections(manager, tmp_path)
