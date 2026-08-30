from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from backend.weaviate_client.models import (
    ChunkRecord,
    DeletionReport,
    PartialParagraphUpdateError,
)
from backend.wizard.errors import WizardSaveError, WizardSaveRecoveryError
from backend.wizard.runtime import WizardRuntime
from backend.wizard.save import save_wizard


USER_ID = "usr_abc123"
DOCUMENT_ID = "60000000-0000-0000-0000-000000000001"


def _uuid(index: int, *, prefix: int = 7) -> str:
    return f"{prefix}0000000-0000-0000-0000-{index:012d}"


class UUIDSequence:
    def __init__(self, values: Sequence[str]) -> None:
        self._values = iter(values)

    def __call__(self) -> str:
        return next(self._values)


class RecordingEmbedder:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        vector: Sequence[float] = (0.25, 0.75),
    ) -> None:
        self.events = events
        self.vector = vector

    def embed(self, text: str) -> Sequence[float]:
        self.events.append(("embed", text))
        return self.vector

    def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.events.append(("embed_many", tuple(texts)))
        return [self.vector for _ in texts]


class RecordingCollection:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events
        self.deleted_unions: list[tuple[int, ...]] = []
        self.inserted: list[tuple[str, int, str, str, tuple[float, ...]]] = []
        self.updated: list[dict[str, int]] = []
        self.fail_insert = False
        self.fail_insert_after: int | None = None
        self.ambiguous_insert_at: int | None = None
        self.insert_attempts = 0
        self.fail_update = False
        self.partial_update_after: int | None = None
        self.mutate_failed_update = False
        self.fail_restore = False
        self.verified_updates: list[dict[str, int]] = []
        self.records: dict[str, ChunkRecord] = {}

    def seed(
        self,
        paragraph_data: dict[int, str],
        chunk_data: dict[int, list[str]],
    ) -> None:
        for paragraph_id, chunk_ids in chunk_data.items():
            for chunk_id in chunk_ids:
                self.records[chunk_id] = ChunkRecord(
                    chunk_id,
                    USER_ID,
                    DOCUMENT_ID,
                    paragraph_id,
                    chunk_id,
                    paragraph_data[paragraph_id],
                    (0.1, 0.2),
                )

    def snapshot_by_document(self, document_id: str) -> tuple[ChunkRecord, ...]:
        return tuple(
            record for record in self.records.values()
            if record.document_id == document_id
        )

    def snapshot_by_paragraphs(
        self, document_id: str, paragraph_ids: list[int]
    ) -> tuple[ChunkRecord, ...]:
        wanted = set(paragraph_ids)
        return tuple(
            record for record in self.records.values()
            if record.document_id == document_id and record.paragraph_id in wanted
        )

    def delete_by_document(self, document_id: str) -> DeletionReport:
        self.events.append(("delete_document", document_id))
        ids = [
            chunk_id for chunk_id, record in self.records.items()
            if record.document_id == document_id
        ]
        for chunk_id in ids:
            del self.records[chunk_id]
        return DeletionReport(len(ids), len(ids), 0, tuple(ids), ())

    def delete_by_paragraphs(
        self,
        document_id: str,
        paragraph_ids: list[int],
    ) -> DeletionReport:
        union = tuple(paragraph_ids)
        self.deleted_unions.append(union)
        self.events.append(("delete", union))
        wanted = set(paragraph_ids)
        ids = [
            chunk_id for chunk_id, record in self.records.items()
            if record.document_id == document_id and record.paragraph_id in wanted
        ]
        for chunk_id in ids:
            del self.records[chunk_id]
        return DeletionReport(len(ids), len(ids), 0, tuple(ids), ())

    def delete_chunks(self, chunk_ids: list[str]) -> DeletionReport:
        deleted = tuple(chunk_id for chunk_id in chunk_ids if chunk_id in self.records)
        for chunk_id in deleted:
            del self.records[chunk_id]
        self.events.append(("delete_chunks", tuple(chunk_ids)))
        return DeletionReport(len(deleted), len(deleted), 0, deleted, ())

    def restore_chunks(self, records: Sequence[ChunkRecord]) -> tuple[str, ...]:
        self.events.append(("restore", tuple(record.chunk_id for record in records)))
        if self.fail_restore:
            raise RuntimeError("restore failed")
        for record in records:
            self.records[record.chunk_id] = record
        return tuple(record.chunk_id for record in records)

    def verify_paragraph_ids(self, expected: dict[str, int]) -> None:
        self.verified_updates.append(dict(expected))
        if any(
            chunk_id not in self.records
            or self.records[chunk_id].paragraph_id != paragraph_id
            for chunk_id, paragraph_id in expected.items()
        ):
            raise RuntimeError("paragraph verification failed")

    def insert_chunk(
        self,
        document_id: str,
        paragraph_id: int,
        chunk_id: str,
        raw_text: str,
        vector: Sequence[float],
    ) -> str:
        record = (
            document_id,
            paragraph_id,
            chunk_id,
            raw_text,
            tuple(vector),
        )
        self.events.append(("insert", paragraph_id, chunk_id, raw_text))
        self.insert_attempts += 1
        if self.fail_insert or (
            self.fail_insert_after is not None
            and len(self.inserted) >= self.fail_insert_after
        ):
            raise RuntimeError("insert failed")
        self.inserted.append(record)
        self.records[chunk_id] = ChunkRecord(
            chunk_id,
            USER_ID,
            document_id,
            paragraph_id,
            chunk_id,
            raw_text,
            tuple(vector),
        )
        if self.ambiguous_insert_at == self.insert_attempts:
            self.ambiguous_insert_at = None
            raise RuntimeError("insert response lost after storage mutation")
        return chunk_id

    def update_paragraph_ids(
        self,
        chunk_id_to_new_paragraph_id: dict[str, int],
    ) -> None:
        update = dict(chunk_id_to_new_paragraph_id)
        self.events.append(("update", update))
        if self.fail_update:
            raise RuntimeError("update failed")
        if self.partial_update_after is not None:
            completed: list[str] = []
            limit = self.partial_update_after
            self.partial_update_after = None
            for chunk_id, paragraph_id in update.items():
                if len(completed) >= limit:
                    if self.mutate_failed_update:
                        record = self.records[chunk_id]
                        self.records[chunk_id] = ChunkRecord(
                            record.object_id,
                            record.user_id,
                            record.document_id,
                            paragraph_id,
                            record.chunk_id,
                            record.raw_text,
                            record.vector,
                        )
                        self.mutate_failed_update = False
                    raise PartialParagraphUpdateError(tuple(completed), chunk_id)
                record = self.records[chunk_id]
                self.records[chunk_id] = ChunkRecord(
                    record.object_id, record.user_id, record.document_id,
                    paragraph_id, record.chunk_id, record.raw_text, record.vector,
                )
                completed.append(chunk_id)
            return
        self.updated.append(update)
        for chunk_id, paragraph_id in update.items():
            record = self.records[chunk_id]
            self.records[chunk_id] = ChunkRecord(
                record.object_id,
                record.user_id,
                record.document_id,
                paragraph_id,
                record.chunk_id,
                record.raw_text,
                record.vector,
            )


def _runtime(
    *,
    split_outputs: dict[str, list[str]] | None = None,
    chunk_outputs: dict[str, list[str]] | None = None,
    embedding_vector: Sequence[float] = (0.25, 0.75),
    new_chunk_ids: Sequence[str] = (),
) -> tuple[WizardRuntime, RecordingCollection, list[tuple[object, ...]]]:
    events: list[tuple[object, ...]] = []
    split_map = split_outputs or {}
    chunk_map = chunk_outputs or {}

    def splitter(text: str) -> list[str]:
        events.append(("split", text))
        return list(split_map.get(text, [] if not text else [text]))

    def chunker(text: str) -> list[str]:
        events.append(("chunk", text))
        return list(chunk_map.get(text, [] if not text else [text]))

    collection = RecordingCollection(events)
    runtime = WizardRuntime(
        MagicMock(),
        RecordingEmbedder(events, embedding_vector),
        paragraph_splitter=splitter,
        paragraph_chunker=chunker,
        uuid_factory=UUIDSequence(new_chunk_ids),
        collection_factory=lambda manager, user_id, collection_type: collection,
    )
    return runtime, collection, events


def _seed_document(
    runtime: WizardRuntime,
    paragraph_data: dict[int, str],
    chunk_data: dict[int, list[str]] | None = None,
    *,
    collection_type: str = "knowledge_facts",
) -> None:
    documents = runtime.document_map(USER_ID, collection_type)
    paragraphs = runtime.paragraph_map(USER_ID, collection_type)
    documents.create_document(DOCUMENT_ID)
    documents.update_paragraphs(DOCUMENT_ID, paragraph_data)
    paragraphs.replace_document(
        DOCUMENT_ID,
        chunk_data or {paragraph_id: [] for paragraph_id in paragraph_data},
    )
    collection = runtime.collection(USER_ID, collection_type)
    collection.seed(
        paragraph_data,
        chunk_data or {paragraph_id: [] for paragraph_id in paragraph_data},
    )


def test_single_modified_paragraph_runs_all_steps_and_commits() -> None:
    new_chunk_id = _uuid(101)
    runtime, collection, events = _runtime(new_chunk_ids=[new_chunk_id])
    old_chunk_id = _uuid(1, prefix=8)
    _seed_document(runtime, {1: "old text"}, {1: [old_chunk_id]})

    save_wizard(
        USER_ID,
        DOCUMENT_ID,
        "knowledge_facts",
        "new text",
        [1],
        runtime=runtime,
    )

    assert collection.deleted_unions == [(1,)]
    assert [event[0] for event in events] == [
        "delete",
        "split",
        "chunk",
        "embed_many",
        "insert",
    ]
    assert collection.inserted == [
        (DOCUMENT_ID, 1, new_chunk_id, "new text", (0.25, 0.75))
    ]
    assert collection.updated == []
    assert runtime.document_map(
        USER_ID, "knowledge_facts"
    ).get_paragraph_data(DOCUMENT_ID) == {1: "new text"}
    assert runtime.paragraph_map(
        USER_ID, "knowledge_facts"
    ).get_document_chunks(DOCUMENT_ID) == {1: [new_chunk_id]}


def test_consecutive_modified_paragraphs_form_one_union_and_renumber_tail() -> None:
    current_union = "new-a|new-b|"
    new_ids = [_uuid(101), _uuid(102)]
    runtime, collection, _ = _runtime(
        split_outputs={current_union: ["new-a|", "new-b|"]},
        new_chunk_ids=new_ids,
    )
    old_chunks = {paragraph_id: [_uuid(paragraph_id, prefix=8)] for paragraph_id in range(1, 5)}
    _seed_document(
        runtime,
        {1: "old-a|", 2: "old-b|", 3: "old-c|", 4: "tail|"},
        old_chunks,
    )

    save_wizard(
        USER_ID,
        DOCUMENT_ID,
        "knowledge_facts",
        current_union + "tail|",
        [1, 2, 3],
        runtime=runtime,
    )

    assert collection.deleted_unions == [(1, 2, 3)]
    assert [record[1] for record in collection.inserted] == [1, 2]
    assert collection.updated == [{old_chunks[4][0]: 3}]
    assert runtime.document_map(
        USER_ID, "knowledge_facts"
    ).get_paragraph_data(DOCUMENT_ID) == {
        1: "new-a|",
        2: "new-b|",
        3: "tail|",
    }


def test_nonconsecutive_unions_delete_before_split_and_update_unmodified_chunks() -> None:
    first_union = "p2-new|p3-new|"
    second_union = "p7-new-a|p7-new-b|"
    new_ids = [_uuid(101), _uuid(102), _uuid(103)]
    runtime, collection, events = _runtime(
        split_outputs={
            first_union: [first_union],
            second_union: ["p7-new-a|", "p7-new-b|"],
        },
        new_chunk_ids=new_ids,
    )
    saved = {paragraph_id: f"p{paragraph_id}|" for paragraph_id in range(1, 9)}
    old_chunks = {paragraph_id: [_uuid(paragraph_id, prefix=8)] for paragraph_id in saved}
    _seed_document(runtime, saved, old_chunks)
    current_text = (
        "p1|"
        + first_union
        + "p4|p5|p6|"
        + second_union
        + "p8|"
    )

    save_wizard(
        USER_ID,
        DOCUMENT_ID,
        "knowledge_facts",
        current_text,
        [2, 3, 7],
        runtime=runtime,
    )

    assert collection.deleted_unions == [(2, 3), (7,)]
    first_split_index = next(
        index for index, event in enumerate(events) if event[0] == "split"
    )
    assert [event[0] for event in events[:first_split_index]] == ["delete", "delete"]
    first_insert_index = next(
        index for index, event in enumerate(events) if event[0] == "insert"
    )
    assert all(event[0] != "embed_many" for event in events[first_insert_index:])
    assert [record[1] for record in collection.inserted] == [2, 6, 7]
    assert collection.updated == [
        {
            old_chunks[4][0]: 3,
            old_chunks[5][0]: 4,
            old_chunks[6][0]: 5,
        }
    ]
    documents = runtime.document_map(USER_ID, "knowledge_facts")
    assert documents.get_paragraphs(DOCUMENT_ID) == list(range(1, 9))
    assert documents.get_full_text(DOCUMENT_ID) == current_text
    paragraph_chunks = runtime.paragraph_map(
        USER_ID, "knowledge_facts"
    ).get_document_chunks(DOCUMENT_ID)
    assert paragraph_chunks[1] == old_chunks[1]
    assert paragraph_chunks[2] == [new_ids[0]]
    assert paragraph_chunks[3] == old_chunks[4]
    assert paragraph_chunks[6] == [new_ids[1]]
    assert paragraph_chunks[7] == [new_ids[2]]
    assert paragraph_chunks[8] == old_chunks[8]


@pytest.mark.parametrize(
    ("saved", "current", "hinted"),
    [
        ({1: "old"}, "replacement", []),
        ({1: "first|", 2: "final"}, "first|final appended", [2]),
    ],
)
def test_diff_derives_changes_and_appended_text_marks_final_paragraph(
    saved: dict[int, str],
    current: str,
    hinted: list[int],
) -> None:
    runtime, collection, _ = _runtime(new_chunk_ids=[_uuid(101)])
    _seed_document(runtime, saved)

    save_wizard(
        USER_ID,
        DOCUMENT_ID,
        "knowledge_facts",
        current,
        hinted,
        runtime=runtime,
    )

    expected_unions = [(max(saved),)] if hinted else [(1,)]
    assert collection.deleted_unions == expected_unions
    assert runtime.document_map(
        USER_ID, "knowledge_facts"
    ).get_full_text(DOCUMENT_ID) == current


def test_unchanged_text_is_a_validated_noop() -> None:
    runtime, collection, events = _runtime()
    old_chunk = _uuid(1, prefix=8)
    _seed_document(runtime, {1: "saved"}, {1: [old_chunk]})

    save_wizard(
        USER_ID,
        DOCUMENT_ID,
        "knowledge_facts",
        "saved",
        [1],
        runtime=runtime,
    )

    assert events == []
    assert collection.deleted_unions == []
    assert runtime.paragraph_map(
        USER_ID, "knowledge_facts"
    ).get_document_chunks(DOCUMENT_ID) == {1: [old_chunk]}


def test_empty_wizard_can_be_saved_and_cleared_back_to_one_empty_paragraph() -> None:
    new_chunk_id = _uuid(101)
    runtime, collection, _ = _runtime(new_chunk_ids=[new_chunk_id])
    _seed_document(runtime, {1: ""}, {1: []})

    save_wizard(
        USER_ID,
        DOCUMENT_ID,
        "knowledge_facts",
        "content",
        [1],
        runtime=runtime,
    )
    save_wizard(
        USER_ID,
        DOCUMENT_ID,
        "knowledge_facts",
        "",
        [1],
        runtime=runtime,
    )

    assert collection.deleted_unions == [(1,), (1,)]
    assert runtime.document_map(
        USER_ID, "knowledge_facts"
    ).get_paragraph_data(DOCUMENT_ID) == {1: ""}
    assert runtime.paragraph_map(
        USER_ID, "knowledge_facts"
    ).get_document_chunks(DOCUMENT_ID) == {1: []}


@pytest.mark.parametrize("failure", ["vector", "insert", "update"])
def test_failure_after_deletion_does_not_commit_mapping_baseline(failure: str) -> None:
    embedding_vector: Sequence[float] = (
        [float("nan")] if failure == "vector" else [0.5]
    )
    runtime, collection, _ = _runtime(
        split_outputs={"new-a|new-b|": ["new-a|", "new-b|"]},
        embedding_vector=embedding_vector,
        new_chunk_ids=[_uuid(101), _uuid(102)],
    )
    old_chunks = {1: [_uuid(1, prefix=8)], 2: [_uuid(2, prefix=8)]}
    _seed_document(runtime, {1: "old|", 2: "tail|"}, old_chunks)
    if failure == "insert":
        collection.fail_insert = True
    if failure == "update":
        collection.fail_update = True

    current_text = "new-a|new-b|tail|"
    modified = [1]
    with pytest.raises(WizardSaveError) as error:
        save_wizard(
            USER_ID,
            DOCUMENT_ID,
            "knowledge_facts",
            current_text,
            modified,
            runtime=runtime,
        )
    expected_stage = {"vector": "step_5", "insert": "step_7a", "update": "step_7b"}
    assert error.value.failed_stage == expected_stage[failure]

    assert runtime.document_map(
        USER_ID, "knowledge_facts"
    ).get_paragraph_data(DOCUMENT_ID) == {1: "old|", 2: "tail|"}
    assert runtime.paragraph_map(
        USER_ID, "knowledge_facts"
    ).get_document_chunks(DOCUMENT_ID) == old_chunks


def test_save_validates_ids_and_processor_losslessness() -> None:
    runtime, collection, _ = _runtime(
        split_outputs={"new": ["normalized"]},
    )
    _seed_document(runtime, {1: "old"})

    with pytest.raises(ValueError, match="does not exist"):
        save_wizard(
            USER_ID,
            DOCUMENT_ID,
            "knowledge_facts",
            "new",
            [2],
            runtime=runtime,
        )
    assert collection.deleted_unions == []

    with pytest.raises(WizardSaveError) as error:
        save_wizard(
            USER_ID,
            DOCUMENT_ID,
            "knowledge_facts",
            "new",
            [1],
            runtime=runtime,
        )
    assert error.value.failed_stage == "step_4"
    assert isinstance(error.value.__cause__, ValueError)
    assert collection.deleted_unions == [(1,)]


def test_incomplete_step3_delete_restores_snapshot_and_never_splits() -> None:
    runtime, collection, events = _runtime(new_chunk_ids=[_uuid(101)])
    old_chunk = _uuid(1, prefix=8)
    _seed_document(runtime, {1: "old"}, {1: [old_chunk]})

    original = collection.delete_by_paragraphs

    def incomplete(document_id: str, paragraph_ids: list[int]) -> DeletionReport:
        original(document_id, paragraph_ids)
        raise RuntimeError("post-delete verification failed")

    collection.delete_by_paragraphs = incomplete  # type: ignore[method-assign]
    with pytest.raises(WizardSaveError) as error:
        save_wizard(
            USER_ID, DOCUMENT_ID, "knowledge_facts", "new", [1], runtime=runtime
        )

    assert error.value.failed_stage == "step_3"
    assert all(event[0] != "split" for event in events)
    assert set(collection.records) == {old_chunk}


@pytest.mark.parametrize("failure", ["splitter", "embedder"])
def test_processing_failure_after_delete_restores_old_chunks(failure: str) -> None:
    runtime, collection, _ = _runtime(new_chunk_ids=[_uuid(101)])
    old_chunk = _uuid(1, prefix=8)
    _seed_document(runtime, {1: "old"}, {1: [old_chunk]})
    if failure == "splitter":
        runtime.paragraph_splitter = lambda text: (_ for _ in ()).throw(
            RuntimeError("split failed")
        )
    else:
        runtime.embedder.embed_many = lambda texts: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeError("embed failed")
        )

    with pytest.raises(WizardSaveError) as error:
        save_wizard(
            USER_ID, DOCUMENT_ID, "knowledge_facts", "new", [1], runtime=runtime
        )

    assert error.value.failed_stage == ("step_4" if failure == "splitter" else "step_5")
    assert set(collection.records) == {old_chunk}


@pytest.mark.parametrize(
    "vectors",
    [[], [[1.0], [2.0]], [[float("nan")]]],
)
def test_batch_embedding_response_is_fully_validated_before_insert(
    vectors: list[list[float]],
) -> None:
    runtime, collection, _ = _runtime(new_chunk_ids=[_uuid(101)])
    _seed_document(runtime, {1: "old"}, {1: []})
    runtime.embedder.embed_many = lambda texts: vectors  # type: ignore[method-assign]

    with pytest.raises(WizardSaveError) as error:
        save_wizard(
            USER_ID, DOCUMENT_ID, "knowledge_facts", "new", [1], runtime=runtime
        )

    assert error.value.failed_stage == "step_5"
    assert collection.inserted == []


def test_batch_embedding_dimensions_must_match() -> None:
    runtime, collection, _ = _runtime(
        split_outputs={"new-a|new-b|": ["new-a|", "new-b|"]},
        new_chunk_ids=[_uuid(101), _uuid(102)],
    )
    _seed_document(runtime, {1: "old"}, {1: []})
    runtime.embedder.embed_many = lambda texts: [[1.0], [1.0, 2.0]]  # type: ignore[method-assign]

    with pytest.raises(WizardSaveError) as error:
        save_wizard(
            USER_ID,
            DOCUMENT_ID,
            "knowledge_facts",
            "new-a|new-b|",
            [1],
            runtime=runtime,
        )

    assert error.value.failed_stage == "step_5"
    assert collection.inserted == []


def test_embedder_single_and_batch_apis_share_implementation() -> None:
    class SharedEmbedder:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def embed(self, text: str) -> Sequence[float]:
            self.calls.append(text)
            return [0.4, 0.6]

        def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            return [self.embed(text) for text in texts]

    embedder = SharedEmbedder()
    runtime = WizardRuntime(MagicMock(), embedder)

    assert runtime.embedder.embed("one") == [0.4, 0.6]
    assert runtime.embedder.embed_many(["two", "three"]) == [
        [0.4, 0.6],
        [0.4, 0.6],
    ]
    assert embedder.calls == ["one", "two", "three"]


@pytest.mark.parametrize("missing_method", ["embed", "embed_many"])
def test_runtime_rejects_incomplete_embedder_interface(
    missing_method: str,
) -> None:
    class IncompleteEmbedder:
        pass

    embedder = IncompleteEmbedder()
    if missing_method == "embed":
        embedder.embed_many = lambda texts: [[0.1] for _ in texts]  # type: ignore[attr-defined]
    else:
        embedder.embed = lambda text: [0.1]  # type: ignore[attr-defined]

    with pytest.raises(TypeError, match="both embed"):
        WizardRuntime(MagicMock(), embedder)  # type: ignore[arg-type]


def test_partial_step7a_is_compensated_and_retry_has_no_orphans() -> None:
    new_ids = [_uuid(101), _uuid(102), _uuid(103), _uuid(104)]
    runtime, collection, _ = _runtime(
        split_outputs={"new-a|new-b|": ["new-a|", "new-b|"]},
        new_chunk_ids=new_ids,
    )
    old_chunk = _uuid(1, prefix=8)
    _seed_document(runtime, {1: "old|"}, {1: [old_chunk]})
    collection.fail_insert_after = 1

    with pytest.raises(WizardSaveError) as error:
        save_wizard(
            USER_ID,
            DOCUMENT_ID,
            "knowledge_facts",
            "new-a|new-b|",
            [1],
            runtime=runtime,
        )
    assert error.value.failed_stage == "step_7a"
    assert set(collection.records) == {old_chunk}

    collection.fail_insert_after = None
    save_wizard(
        USER_ID,
        DOCUMENT_ID,
        "knowledge_facts",
        "new-a|new-b|",
        [1],
        runtime=runtime,
    )
    assert set(collection.records) == {new_ids[2], new_ids[3]}
    assert len(collection.records) == 2


def test_ambiguous_step7a_insert_is_deleted_during_compensation() -> None:
    new_chunk = _uuid(101)
    runtime, collection, events = _runtime(new_chunk_ids=[new_chunk])
    old_chunk = _uuid(1, prefix=8)
    _seed_document(runtime, {1: "old"}, {1: [old_chunk]})
    collection.ambiguous_insert_at = 1

    with pytest.raises(WizardSaveError) as error:
        save_wizard(
            USER_ID, DOCUMENT_ID, "knowledge_facts", "new", [1], runtime=runtime
        )

    assert error.value.failed_stage == "step_7a"
    assert error.value.inserted_chunk_ids == ()
    assert ("delete_chunks", (new_chunk,)) in events
    assert set(collection.records) == {old_chunk}


def test_partial_step7b_is_restored_to_previous_paragraph_ids() -> None:
    runtime, collection, _ = _runtime(
        split_outputs={"new-a|new-b|": ["new-a|", "new-b|"]},
        new_chunk_ids=[_uuid(101), _uuid(102)],
    )
    old_chunks = {1: [_uuid(1, prefix=8)], 2: [_uuid(2, prefix=8)], 3: [_uuid(3, prefix=8)]}
    _seed_document(runtime, {1: "old|", 2: "tail2|", 3: "tail3|"}, old_chunks)
    collection.partial_update_after = 1
    collection.mutate_failed_update = True

    with pytest.raises(WizardSaveError) as error:
        save_wizard(
            USER_ID,
            DOCUMENT_ID,
            "knowledge_facts",
            "new-a|new-b|tail2|tail3|",
            [1],
            runtime=runtime,
        )

    assert error.value.failed_stage == "step_7b"
    assert error.value.updated_chunk_ids == (old_chunks[2][0],)
    assert {
        chunk_id: record.paragraph_id for chunk_id, record in collection.records.items()
    } == {old_chunks[1][0]: 1, old_chunks[2][0]: 2, old_chunks[3][0]: 3}
    assert collection.verified_updates[-1] == {
        old_chunks[2][0]: 2,
        old_chunks[3][0]: 3,
    }


def test_compensation_failure_reports_unresolved_ids() -> None:
    runtime, collection, _ = _runtime(new_chunk_ids=[_uuid(101)])
    old_chunk = _uuid(1, prefix=8)
    _seed_document(runtime, {1: "old"}, {1: [old_chunk]})
    collection.fail_restore = True
    runtime.embedder.embed_many = lambda texts: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("embed failed")
    )

    with pytest.raises(WizardSaveRecoveryError) as error:
        save_wizard(
            USER_ID, DOCUMENT_ID, "knowledge_facts", "new", [1], runtime=runtime
        )

    assert error.value.failed_stage == "step_5"
    assert error.value.unresolved_chunk_ids == (old_chunk,)
    assert error.value.recovery_errors


def test_mapping_only_recovery_failure_has_no_unresolved_storage_ids() -> None:
    runtime, collection, _ = _runtime(new_chunk_ids=[_uuid(101)])
    old_chunk = _uuid(1, prefix=8)
    _seed_document(runtime, {1: "old"}, {1: [old_chunk]})
    documents = runtime.document_map(USER_ID, "knowledge_facts")

    def fail_commit(document_id: str, paragraph_data: dict[int, str]) -> None:
        raise RuntimeError("mapping commit failed")

    def fail_mapping_recovery(
        document_id: str,
        paragraph_data: dict[int, str],
    ) -> None:
        raise RuntimeError("mapping recovery failed")

    documents.update_paragraphs = fail_commit  # type: ignore[method-assign]
    documents.restore_document = fail_mapping_recovery  # type: ignore[method-assign]

    with pytest.raises(WizardSaveRecoveryError) as error:
        save_wizard(
            USER_ID, DOCUMENT_ID, "knowledge_facts", "new", [1], runtime=runtime
        )

    assert error.value.failed_stage == "step_8"
    assert error.value.unresolved_chunk_ids == ()
    assert len(error.value.recovery_errors) == 1
    assert str(error.value.recovery_errors[0]) == "mapping recovery failed"
    assert set(collection.records) == {old_chunk}


def test_step8_mapping_failure_rolls_back_maps_and_storage() -> None:
    runtime, collection, _ = _runtime(new_chunk_ids=[_uuid(101)])
    old_chunk = _uuid(1, prefix=8)
    _seed_document(runtime, {1: "old"}, {1: [old_chunk]})
    documents = runtime.document_map(USER_ID, "knowledge_facts")

    def fail_commit(document_id: str, paragraph_data: dict[int, str]) -> None:
        raise RuntimeError("mapping commit failed")

    documents.update_paragraphs = fail_commit  # type: ignore[method-assign]
    with pytest.raises(WizardSaveError) as error:
        save_wizard(
            USER_ID, DOCUMENT_ID, "knowledge_facts", "new", [1], runtime=runtime
        )

    assert error.value.failed_stage == "step_8"
    assert documents.get_paragraph_data(DOCUMENT_ID) == {1: "old"}
    assert runtime.paragraph_map(
        USER_ID, "knowledge_facts"
    ).get_document_chunks(DOCUMENT_ID) == {1: [old_chunk]}
    assert set(collection.records) == {old_chunk}
