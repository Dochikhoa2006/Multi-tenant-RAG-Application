from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

import backend.wizard as wizard_package
from backend.wizard.crud import create_wizard, delete_wizard
from backend.weaviate_client.models import ChunkRecord, DeletionReport
from backend.wizard.errors import WizardDeleteError, WizardDeleteRecoveryError
from backend.wizard.runtime import WizardRuntime


USER_ID = "usr_abc123"
DOCUMENT_ID = "40000000-0000-0000-0000-000000000001"
POLICY_DOCUMENT_ID = "40000000-0000-0000-0000-000000000002"
CHUNK_ID = "50000000-0000-0000-0000-000000000001"


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> Sequence[float]:
        self.calls.append(text)
        return [1.0]

    def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self.embed(text) for text in texts]


class UUIDSequence:
    def __init__(self, *values: str) -> None:
        self._values = iter(values)

    def __call__(self) -> str:
        return next(self._values)


class FakeCollection:
    def __init__(self) -> None:
        self.deleted_documents: list[str] = []
        self.fail_delete = False
        self.partial_delete = False
        self.fail_restore = False
        self.records: dict[str, ChunkRecord] = {}

    def snapshot_by_document(self, document_id: str) -> tuple[ChunkRecord, ...]:
        return tuple(
            record for record in self.records.values()
            if record.document_id == document_id
        )

    def delete_by_document(self, document_id: str) -> DeletionReport:
        self.deleted_documents.append(document_id)
        if self.fail_delete:
            raise RuntimeError("delete failed")
        ids = tuple(
            chunk_id for chunk_id, record in self.records.items()
            if record.document_id == document_id
        )
        for index, chunk_id in enumerate(ids):
            del self.records[chunk_id]
            if self.partial_delete and index == 0:
                raise RuntimeError("partial delete failed")
        return DeletionReport(len(ids), len(ids), 0, ids, ())

    def restore_chunks(self, records: Sequence[ChunkRecord]) -> tuple[str, ...]:
        if self.fail_restore:
            raise RuntimeError("restore failed")
        for record in records:
            self.records[record.chunk_id] = record
        return tuple(record.chunk_id for record in records)


def _runtime(
    collection: FakeCollection,
    embedder: FakeEmbedder,
    *uuid_values: str,
) -> tuple[WizardRuntime, MagicMock]:
    factory = MagicMock(return_value=collection)
    runtime = WizardRuntime(
        MagicMock(),
        embedder,
        uuid_factory=UUIDSequence(*uuid_values),
        collection_factory=factory,
    )
    return runtime, factory


def test_create_wizard_initializes_empty_mappings_without_storage_work() -> None:
    collection = FakeCollection()
    embedder = FakeEmbedder()
    runtime, collection_factory = _runtime(
        collection,
        embedder,
        DOCUMENT_ID,
        POLICY_DOCUMENT_ID,
    )

    knowledge_id = create_wizard(USER_ID, "knowledge_facts", runtime=runtime)
    policy_id = create_wizard(USER_ID, "policy", runtime=runtime)

    assert knowledge_id == DOCUMENT_ID
    assert policy_id == POLICY_DOCUMENT_ID
    assert runtime.document_map(USER_ID, "knowledge_facts").get_paragraph_data(
        knowledge_id
    ) == {1: ""}
    assert runtime.paragraph_map(USER_ID, "knowledge_facts").get_document_chunks(
        knowledge_id
    ) == {1: []}
    assert runtime.document_map(USER_ID, "policy").get_paragraph_data(policy_id) == {
        1: ""
    }
    assert runtime.paragraph_map(USER_ID, "policy").get_document_chunks(policy_id) == {
        1: []
    }
    collection_factory.assert_not_called()
    assert embedder.calls == []
    assert collection.deleted_documents == []


def test_create_wizard_rolls_back_both_maps_after_partial_initialization() -> None:
    runtime, collection_factory = _runtime(
        FakeCollection(), FakeEmbedder(), DOCUMENT_ID
    )
    documents = runtime.document_map(USER_ID, "knowledge_facts")
    paragraphs = runtime.paragraph_map(USER_ID, "knowledge_facts")
    original_replace = paragraphs.replace_document

    def mutate_then_fail(
        document_id: str,
        paragraph_chunks: dict[int, list[str]],
    ) -> None:
        original_replace(document_id, paragraph_chunks)
        raise RuntimeError("paragraph initialization failed")

    paragraphs.replace_document = mutate_then_fail  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="paragraph initialization failed"):
        create_wizard(USER_ID, "knowledge_facts", runtime=runtime)

    with pytest.raises(KeyError):
        documents.get_paragraph_data(DOCUMENT_ID)
    assert paragraphs.get_document_chunks(DOCUMENT_ID) == {}
    collection_factory.assert_not_called()


@pytest.mark.parametrize("collision_map", ["document", "paragraph"])
def test_create_wizard_rejects_generated_document_id_collision(
    collision_map: str,
) -> None:
    runtime, collection_factory = _runtime(
        FakeCollection(), FakeEmbedder(), DOCUMENT_ID
    )
    documents = runtime.document_map(USER_ID, "knowledge_facts")
    paragraphs = runtime.paragraph_map(USER_ID, "knowledge_facts")
    if collision_map == "document":
        documents.create_document(DOCUMENT_ID)
    else:
        paragraphs.replace_document(DOCUMENT_ID, {1: []})

    with pytest.raises(ValueError, match="already exists"):
        create_wizard(USER_ID, "knowledge_facts", runtime=runtime)

    if collision_map == "document":
        assert documents.get_paragraph_data(DOCUMENT_ID) == {1: ""}
        assert paragraphs.get_document_chunks(DOCUMENT_ID) == {}
    else:
        with pytest.raises(KeyError):
            documents.get_paragraph_data(DOCUMENT_ID)
        assert paragraphs.get_document_chunks(DOCUMENT_ID) == {1: []}
    collection_factory.assert_not_called()


def test_delete_wizard_removes_storage_before_local_mappings() -> None:
    collection = FakeCollection()
    embedder = FakeEmbedder()
    runtime, factory = _runtime(collection, embedder, DOCUMENT_ID)
    create_wizard(USER_ID, "knowledge_facts", runtime=runtime)
    documents = runtime.document_map(USER_ID, "knowledge_facts")
    paragraphs = runtime.paragraph_map(USER_ID, "knowledge_facts")
    documents.update_paragraphs(DOCUMENT_ID, {1: "saved text"})
    paragraphs.replace_document(DOCUMENT_ID, {1: [CHUNK_ID]})

    delete_wizard(
        USER_ID,
        DOCUMENT_ID,
        "knowledge_facts",
        runtime=runtime,
    )

    assert collection.deleted_documents == [DOCUMENT_ID]
    factory.assert_called_once_with(runtime.manager, USER_ID, "knowledge_facts")
    with pytest.raises(KeyError):
        documents.get_paragraph_data(DOCUMENT_ID)
    assert paragraphs.get_document_chunks(DOCUMENT_ID) == {}


def test_delete_failure_preserves_document_and_paragraph_mappings() -> None:
    collection = FakeCollection()
    collection.fail_delete = True
    runtime, _ = _runtime(collection, FakeEmbedder(), DOCUMENT_ID)
    create_wizard(USER_ID, "policy", runtime=runtime)
    documents = runtime.document_map(USER_ID, "policy")
    paragraphs = runtime.paragraph_map(USER_ID, "policy")
    documents.update_paragraphs(DOCUMENT_ID, {1: "policy"})
    paragraphs.replace_document(DOCUMENT_ID, {1: [CHUNK_ID]})

    with pytest.raises(WizardDeleteError) as error:
        delete_wizard(USER_ID, DOCUMENT_ID, "policy", runtime=runtime)

    assert error.value.failed_stage == "storage_delete"

    assert documents.get_paragraph_data(DOCUMENT_ID) == {1: "policy"}
    assert paragraphs.get_document_chunks(DOCUMENT_ID) == {1: [CHUNK_ID]}


def test_crud_rejects_invalid_scope_and_missing_runtime() -> None:
    runtime, factory = _runtime(FakeCollection(), FakeEmbedder(), DOCUMENT_ID)

    with pytest.raises(ValueError, match="collection_type"):
        create_wizard(USER_ID, "conversations", runtime=runtime)
    with pytest.raises(RuntimeError, match="configure_default_runtime"):
        create_wizard(USER_ID, "knowledge_facts")
    with pytest.raises(ValueError, match="UUID"):
        delete_wizard(
            USER_ID,
            "not-a-uuid",
            "knowledge_facts",
            runtime=runtime,
        )

    factory.assert_not_called()


def test_partial_wizard_delete_restores_storage_and_both_mappings() -> None:
    collection = FakeCollection()
    collection.partial_delete = True
    runtime, _ = _runtime(collection, FakeEmbedder(), DOCUMENT_ID)
    create_wizard(USER_ID, "knowledge_facts", runtime=runtime)
    documents = runtime.document_map(USER_ID, "knowledge_facts")
    paragraphs = runtime.paragraph_map(USER_ID, "knowledge_facts")
    documents.update_paragraphs(DOCUMENT_ID, {1: "saved"})
    paragraphs.replace_document(DOCUMENT_ID, {1: [CHUNK_ID]})
    collection.records[CHUNK_ID] = ChunkRecord(
        CHUNK_ID, USER_ID, DOCUMENT_ID, 1, CHUNK_ID, "saved", (0.1,)
    )

    with pytest.raises(WizardDeleteError) as error:
        delete_wizard(
            USER_ID, DOCUMENT_ID, "knowledge_facts", runtime=runtime
        )

    assert error.value.failed_stage == "storage_delete"
    assert set(collection.records) == {CHUNK_ID}
    assert documents.get_paragraph_data(DOCUMENT_ID) == {1: "saved"}
    assert paragraphs.get_document_chunks(DOCUMENT_ID) == {1: [CHUNK_ID]}


def test_wizard_delete_recovery_failure_is_explicit() -> None:
    collection = FakeCollection()
    collection.partial_delete = True
    collection.fail_restore = True
    runtime, _ = _runtime(collection, FakeEmbedder(), DOCUMENT_ID)
    create_wizard(USER_ID, "policy", runtime=runtime)
    documents = runtime.document_map(USER_ID, "policy")
    paragraphs = runtime.paragraph_map(USER_ID, "policy")
    documents.update_paragraphs(DOCUMENT_ID, {1: "saved"})
    paragraphs.replace_document(DOCUMENT_ID, {1: [CHUNK_ID]})
    collection.records[CHUNK_ID] = ChunkRecord(
        CHUNK_ID, USER_ID, DOCUMENT_ID, 1, CHUNK_ID, "saved", (0.1,)
    )

    with pytest.raises(WizardDeleteRecoveryError) as error:
        delete_wizard(USER_ID, DOCUMENT_ID, "policy", runtime=runtime)

    assert error.value.recovery_errors


def test_wizard_package_exports_complete_public_interface() -> None:
    expected = [
        "ChunkEmbedder",
        "WizardRuntime",
        "configure_default_runtime",
        "create_wizard",
        "delete_wizard",
        "save_wizard",
        "WizardDeleteError",
        "WizardDeleteRecoveryError",
        "WizardSaveError",
        "WizardSaveRecoveryError",
    ]

    assert wizard_package.__all__ == expected
    assert all(hasattr(wizard_package, name) for name in expected)
