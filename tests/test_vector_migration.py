from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from weaviate.classes.config import DataType

from backend.config import get_collection_name
from backend.model_config import EMBEDDING_MODEL, EMBEDDING_VECTOR_PROFILE
from backend.weaviate_client.client import WeaviateManager, _collection_description
from scripts.migrate_onnx_vectors import (
    VectorMigrationError,
    export_collections,
    rebuild_collections,
    verify_collections,
)


USER_ID = "usr_migration"
COLLECTION_TYPES = ("conversations", "knowledge_facts", "policy")


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


class FakeAggregate:
    def __init__(self, collection: "FakeCollection") -> None:
        self.collection = collection

    def over_all(self, *, total_count: bool) -> SimpleNamespace:
        assert total_count is True
        return SimpleNamespace(total_count=len(self.collection.items))


class FakeCollection:
    def __init__(
        self,
        name: str,
        items: list[object] | None = None,
        *,
        description: str = "legacy",
    ) -> None:
        self.name = name
        self.items = list(items or [])
        self.config = FakeConfig(_config(name, description))
        self.aggregate = FakeAggregate(self)


class FakeCollections:
    def __init__(self, values: dict[str, FakeCollection]) -> None:
        self.values = values
        self.deleted: list[str] = []
        self.created: list[str] = []
        self.fail_create_name_once: str | None = None

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
        if self.fail_create_name_once == name:
            self.fail_create_name_once = None
            raise RuntimeError("injected collection creation failure")
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


def _manager(
    *, populated_collection_type: str | None = None
) -> tuple[WeaviateManager, FakeCollections]:
    values: dict[str, FakeCollection] = {}
    for collection_type in COLLECTION_TYPES:
        name = get_collection_name(USER_ID, collection_type)
        items = [object()] if collection_type == populated_collection_type else []
        values[name] = FakeCollection(name, items)
    values["Unrelated"] = FakeCollection("Unrelated")
    collections = FakeCollections(values)
    manager = WeaviateManager(FakeClient(collections))
    manager.connect()
    return manager, collections


def _export(manager: WeaviateManager, directory: Path) -> Path:
    return export_collections(
        manager,
        directory,
        confirm_disposable_process_state=True,
    )


def _rebuild(manager: WeaviateManager, directory: Path) -> None:
    rebuild_collections(
        manager,
        directory,
        confirm_disposable_process_state=True,
    )


def test_export_requires_explicit_disposable_state_confirmation(tmp_path: Path) -> None:
    manager, collections = _manager()

    with pytest.raises(VectorMigrationError, match="explicit confirmation"):
        export_collections(manager, tmp_path)

    assert not (tmp_path / "vector-migration-manifest.json").exists()
    assert collections.deleted == []
    assert all(not collection.config.updates for collection in collections.values.values())


@pytest.mark.parametrize("collection_type", COLLECTION_TYPES)
def test_populated_collection_blocks_export_before_manifest_or_mutation(
    tmp_path: Path, collection_type: str
) -> None:
    manager, collections = _manager(populated_collection_type=collection_type)

    with pytest.raises(VectorMigrationError, match="populated state is unsupported"):
        _export(manager, tmp_path)

    assert not (tmp_path / "vector-migration-manifest.json").exists()
    assert collections.deleted == []
    assert all(not collection.config.updates for collection in collections.values.values())


def test_empty_canonical_triplets_rebuild_and_verify(tmp_path: Path) -> None:
    manager, collections = _manager()
    canonical_names = {
        get_collection_name(USER_ID, collection_type)
        for collection_type in COLLECTION_TYPES
    }

    manifest_path = _export(manager, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest == {
        "schema_version": "2.0",
        "migration_scope": "disposable-empty-prototype-state",
        "mapping_state_preserved": False,
        "embedding_model": EMBEDDING_MODEL,
        "vector_profile": EMBEDDING_VECTOR_PROFILE,
        "collections": [
            {
                "name": name,
                "user_id": USER_ID,
                "collection_type": collection_type,
                "count": 0,
            }
            for name, collection_type in sorted(
                (get_collection_name(USER_ID, item), item)
                for item in COLLECTION_TYPES
            )
        ],
    }
    assert list(tmp_path.glob("*.jsonl")) == []

    _rebuild(manager, tmp_path)
    verify_collections(manager, tmp_path)

    assert set(collections.deleted) == canonical_names
    assert set(collections.created) == canonical_names
    assert "Unrelated" in collections.values
    assert all(
        collections.values[name].config.value.description == _collection_description()
        and not collections.values[name].items
        for name in canonical_names
    )


def test_objects_added_after_export_block_rebuild_before_mutation(tmp_path: Path) -> None:
    manager, collections = _manager()
    _export(manager, tmp_path)
    name = get_collection_name(USER_ID, "knowledge_facts")
    collections.values[name].items.append(object())

    with pytest.raises(VectorMigrationError, match="populated state is unsupported"):
        _rebuild(manager, tmp_path)

    assert collections.deleted == []
    assert collections.created == []
    assert all(not collection.config.updates for collection in collections.values.values())


def test_old_populated_manifest_is_rejected_before_mutation(tmp_path: Path) -> None:
    manager, collections = _manager()
    (tmp_path / "vector-migration-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "embedding_model": EMBEDDING_MODEL,
                "vector_profile": EMBEDDING_VECTOR_PROFILE,
                "collections": [{"count": 1}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(VectorMigrationError, match="not an empty-state manifest"):
        _rebuild(manager, tmp_path)

    assert collections.deleted == []
    assert collections.created == []


def test_partial_empty_rebuild_stays_rebuilding_and_retries(tmp_path: Path) -> None:
    manager, collections = _manager()
    _export(manager, tmp_path)
    failure_name = get_collection_name(USER_ID, "knowledge_facts")
    collections.fail_create_name_once = failure_name

    with pytest.raises(RuntimeError, match="injected collection creation failure"):
        _rebuild(manager, tmp_path)

    canonical_names = {
        get_collection_name(USER_ID, collection_type)
        for collection_type in COLLECTION_TYPES
    }
    assert all(
        collection.config.value.description == _collection_description("rebuilding")
        for name, collection in collections.values.items()
        if name in canonical_names
    )
    assert all(
        collection.config.value.description != _collection_description()
        for name, collection in collections.values.items()
        if name in canonical_names
    )

    _rebuild(manager, tmp_path)
    verify_collections(manager, tmp_path)
    assert all(
        collections.values[name].config.value.description == _collection_description()
        for name in canonical_names
    )


def test_export_rejects_incomplete_canonical_triplet(tmp_path: Path) -> None:
    manager, collections = _manager()
    del collections.values[get_collection_name(USER_ID, "policy")]

    with pytest.raises(VectorMigrationError, match="exactly the three"):
        _export(manager, tmp_path)


def test_rebuild_requires_disposable_confirmation_even_with_manifest(
    tmp_path: Path,
) -> None:
    manager, collections = _manager()
    _export(manager, tmp_path)

    with pytest.raises(VectorMigrationError, match="explicit confirmation"):
        rebuild_collections(manager, tmp_path)

    assert collections.deleted == []
    assert collections.created == []
