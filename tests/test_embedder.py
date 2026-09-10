from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import pytest

from backend.model_config import EMBEDDING_MODEL, LATEON_EMBEDDING_DIMENSION
from backend.rag.embedder import (
    conversation_segments,
    encode_documents,
    encode_query,
    embed_chunks,
    embed_conversation_background,
    embed_text,
    embed_conversation,
)
from backend.rag.runtime import RAGRuntime, RerankResult
from backend.wizard.diagnostics import (
    DiagnosticTraceRegistry,
    activate_operation,
    install_registry,
    uninstall_registry,
)
from uuid import uuid4


CONVERSATION_ID = "81000000-0000-0000-0000-000000000001"


def _lateon_row(value: float) -> list[float]:
    return [value, *([0.0] * (LATEON_EMBEDDING_DIMENSION - 1))]


class DummyLLM:
    def complete(self, prompt: str, **kwargs: object) -> str:
        raise AssertionError("LLM must not be used by embeddings")

    def stream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        raise AssertionError("LLM must not be used by embeddings")


class DummyReranker:
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        **kwargs: object,
    ) -> Sequence[RerankResult]:
        return []


class FakeEmbeddings:
    def __init__(
        self,
        *,
        single: object = (0.1, 0.2),
        batch: object = ((0.1, 0.2), (0.3, 0.4)),
        failure: Exception | None = None,
    ) -> None:
        self.single = single
        self.batch = batch
        self.failure = failure
        self.single_calls: list[tuple[str, str]] = []
        self.batch_calls: list[tuple[list[str], str]] = []

    def embed(self, text: str, *, model: str) -> Sequence[float]:
        self.single_calls.append((text, model))
        if self.failure is not None:
            raise self.failure
        return self.single  # type: ignore[return-value]

    def embed_many(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> Sequence[Sequence[float]]:
        self.batch_calls.append((list(texts), model))
        if self.failure is not None:
            raise self.failure
        return self.batch  # type: ignore[return-value]


class FakeMultiVectors:
    def __init__(self, documents: object | None = None, query: object | None = None) -> None:
        self.documents = documents
        self.query = query if query is not None else [_lateon_row(0.1)]
        self.document_calls: list[list[str]] = []

    def encode_query(self, text: str) -> object:
        return self.query

    def encode_documents(self, texts: Sequence[str]) -> object:
        values = list(texts)
        self.document_calls.append(values)
        if self.documents is not None:
            return self.documents
        return [[_lateon_row(float(index + 1))] for index in range(len(values))]


class LosslessSegmenter:
    def __init__(self, segments: Sequence[str] | None = None) -> None:
        self.segments = segments

    def segment_document(self, text: str) -> Sequence[str]:
        return self.segments if self.segments is not None else [text]


@dataclass
class FakeConversationCollection:
    failure: Exception | None = None

    def __post_init__(self) -> None:
        self.inserts: list[tuple[object, ...]] = []

    def insert(
        self,
        conversation_id: str,
        raw_text: str,
        segment_texts: Sequence[str],
        segment_vectors: Sequence[Sequence[Sequence[float]]],
        diversity_vector: Sequence[float],
    ) -> str:
        self.inserts.append(
            (
                conversation_id,
                raw_text,
                list(segment_texts),
                [[list(row) for row in matrix] for matrix in segment_vectors],
                list(diversity_vector),
            )
        )
        if self.failure is not None:
            raise self.failure
        return conversation_id


def _runtime(
    embeddings: FakeEmbeddings,
    collection: FakeConversationCollection | None = None,
    *,
    multi_vectors: FakeMultiVectors | None = None,
    segmenter: LosslessSegmenter | None = None,
) -> tuple[RAGRuntime, list[str]]:
    users: list[str] = []
    target = collection or FakeConversationCollection()

    def factory(user_id: str) -> FakeConversationCollection:
        users.append(user_id)
        return target

    return (
        RAGRuntime(
            DummyLLM(),
            embeddings,
            DummyReranker(),
            factory,
            multi_vectors=multi_vectors or FakeMultiVectors(),
            conversation_segmenter=segmenter or LosslessSegmenter(),
        ),
        users,
    )


def test_embed_text_uses_configured_model_and_returns_copied_floats() -> None:
    embeddings = FakeEmbeddings(single=[1, 2.5])
    runtime, _ = _runtime(embeddings)

    assert embed_text("source text", runtime=runtime) == [1.0, 2.5]
    assert embeddings.single_calls == [("source text", EMBEDDING_MODEL)]


def test_embed_chunks_batches_once_with_configured_model() -> None:
    embeddings = FakeEmbeddings(batch=[[1, 2], [3.5, 4]])
    runtime, _ = _runtime(embeddings)

    assert embed_chunks(["one", "two"], runtime=runtime) == [
        [1.0, 2.0],
        [3.5, 4.0],
    ]
    assert embeddings.batch_calls == [(["one", "two"], EMBEDDING_MODEL)]


def test_embed_chunks_empty_input_does_not_resolve_or_call_provider() -> None:
    assert embed_chunks([]) == []


@pytest.mark.parametrize(
    ("batch", "error"),
    [
        ([[1.0]], "wrong number"),
        ([[1.0], []], "must not be empty"),
        ([[1.0], [float("nan")]], "must be finite"),
        ([[1.0], [1.0, 2.0]], "consistent dimensions"),
        ([1.0, 2.0], "sequence of numbers"),
    ],
)
def test_embed_chunks_rejects_malformed_provider_output(
    batch: object,
    error: str,
) -> None:
    runtime, _ = _runtime(FakeEmbeddings(batch=batch))
    with pytest.raises((TypeError, ValueError), match=error):
        embed_chunks(["one", "two"], runtime=runtime)


def test_background_embedding_returns_observable_task_and_inserts_labeled_text() -> None:
    embeddings = FakeEmbeddings(single=[0.4, 0.6])
    collection = FakeConversationCollection()
    runtime, users = _runtime(embeddings, collection)

    async def scenario() -> asyncio.Task[None]:
        task = embed_conversation_background(
            "usr_test",
            CONVERSATION_ID,
            "What is MMR?",
            "It balances relevance and diversity.",
            runtime=runtime,
        )
        assert isinstance(task, asyncio.Task)
        assert task.get_name() == f"embed-conversation-{CONVERSATION_ID}"
        await task
        return task

    task = asyncio.run(scenario())

    raw_text = (
        "Question:\nWhat is MMR?\n\n"
        "Answer:\nIt balances relevance and diversity."
    )
    assert task.done() and task.exception() is None
    assert users == ["usr_test"]
    assert embeddings.single_calls == [(raw_text, EMBEDDING_MODEL)]
    assert collection.inserts == [
        (CONVERSATION_ID, raw_text, [raw_text], [[_lateon_row(1.0)]], [0.4, 0.6])
    ]


def test_conversation_persistence_trace_observes_existing_concurrent_work_once() -> None:
    embeddings = FakeEmbeddings(single=[0.25] * 768)
    collection = FakeConversationCollection()
    raw_text = "Question:\nQuestion\n\nAnswer:\nAnswer"
    segmenter = LosslessSegmenter(["Question:\nQuestion\n\n", "Answer:\nAnswer"])
    multi_vectors = FakeMultiVectors()
    runtime, users = _runtime(
        embeddings,
        collection,
        multi_vectors=multi_vectors,
        segmenter=segmenter,
    )
    registry = DiagnosticTraceRegistry("usr_test")
    session_id = str(uuid4())
    operation_id = str(uuid4())
    registry.start("usr_test", session_id, "run-persistence")
    handle = registry.begin_operation(
        user_id="usr_test",
        session_id=session_id,
        operation_id=operation_id,
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    )
    assert handle is not None
    install_registry(registry)
    try:
        with activate_operation(handle):
            asyncio.run(
                embed_conversation(
                    "usr_test",
                    CONVERSATION_ID,
                    "Question",
                    "Answer",
                    runtime=runtime,
                )
            )
    finally:
        uninstall_registry(registry)

    operation = registry.snapshot("usr_test", session_id, operation_id)["operations"][0]
    stages = operation["stages"]
    counts = operation["counts"]
    assert stages["chat.persistence_segmentation"]["call_count"] == 1
    assert stages["chat.persistence_embeddings_fork_join"]["call_count"] == 1
    assert stages["chat.persistence_lateon"]["call_count"] == 1
    assert stages["chat.persistence_gte"]["call_count"] == 1
    assert stages["chat.persistence_storage_total"]["call_count"] == 1
    assert counts["persistence_segment_count"] == 2
    assert counts["persistence_lateon_document_count"] == 2
    assert counts["persistence_gte_vector_count"] == 1
    assert users == ["usr_test"]
    assert multi_vectors.document_calls == [segmenter.segments]
    assert embeddings.single_calls == [(raw_text, EMBEDDING_MODEL)]
    assert len(collection.inserts) == 1


def test_multi_vector_contract_validates_shape_finiteness_and_dimensions() -> None:
    runtime, _ = _runtime(
        FakeEmbeddings(),
        multi_vectors=FakeMultiVectors(
            documents=[[_lateon_row(1.0)], [_lateon_row(3.0), _lateon_row(5.0)]],
            query=[_lateon_row(7.0)],
        ),
    )
    assert encode_query("query", runtime=runtime) == [_lateon_row(7.0)]
    assert encode_documents(["a", "b"], runtime=runtime) == [
        [_lateon_row(1.0)],
        [_lateon_row(3.0), _lateon_row(5.0)],
    ]

    malformed, _ = _runtime(
        FakeEmbeddings(),
        multi_vectors=FakeMultiVectors(
            documents=[[_lateon_row(1.0)], [_lateon_row(float("nan"))]]
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        encode_documents(["a", "b"], runtime=malformed)


def test_conversation_segmenter_contract_is_ordered_nonempty_and_lossless() -> None:
    runtime, _ = _runtime(
        FakeEmbeddings(), segmenter=LosslessSegmenter(["first ", "second"])
    )
    assert conversation_segments("first second", runtime=runtime) == [
        "first ",
        "second",
    ]

    invalid, _ = _runtime(
        FakeEmbeddings(), segmenter=LosslessSegmenter(["first", "changed"])
    )
    with pytest.raises(ValueError, match="reconstruct canonical text exactly"):
        conversation_segments("first second", runtime=invalid)


def test_background_embedding_failure_propagates_through_returned_task() -> None:
    embeddings = FakeEmbeddings(failure=RuntimeError("embedding failed"))
    runtime, _ = _runtime(embeddings)

    async def scenario() -> None:
        task = embed_conversation_background(
            "usr_test",
            CONVERSATION_ID,
            "Question",
            "Answer",
            runtime=runtime,
        )
        with pytest.raises(RuntimeError, match="embedding failed"):
            await task

    asyncio.run(scenario())


def test_background_insert_failure_propagates_through_returned_task() -> None:
    collection = FakeConversationCollection(RuntimeError("insert failed"))
    runtime, _ = _runtime(FakeEmbeddings(), collection)

    async def scenario() -> None:
        task = embed_conversation_background(
            "usr_test",
            CONVERSATION_ID,
            "Question",
            "Answer",
            runtime=runtime,
        )
        with pytest.raises(RuntimeError, match="insert failed"):
            await task

    asyncio.run(scenario())
