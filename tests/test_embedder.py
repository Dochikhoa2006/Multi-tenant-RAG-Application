from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import pytest

from backend.model_config import EMBEDDING_MODEL
from backend.rag.embedder import (
    embed_chunks,
    embed_conversation_background,
    embed_text,
)
from backend.rag.runtime import RAGRuntime, RerankResult


CONVERSATION_ID = "81000000-0000-0000-0000-000000000001"


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


@dataclass
class FakeConversationCollection:
    failure: Exception | None = None

    def __post_init__(self) -> None:
        self.inserts: list[tuple[str, str, list[float]]] = []

    def insert(
        self,
        conversation_id: str,
        raw_text: str,
        vector: Sequence[float],
    ) -> str:
        self.inserts.append((conversation_id, raw_text, list(vector)))
        if self.failure is not None:
            raise self.failure
        return conversation_id


def _runtime(
    embeddings: FakeEmbeddings,
    collection: FakeConversationCollection | None = None,
) -> tuple[RAGRuntime, list[str]]:
    users: list[str] = []
    target = collection or FakeConversationCollection()

    def factory(user_id: str) -> FakeConversationCollection:
        users.append(user_id)
        return target

    return (
        RAGRuntime(DummyLLM(), embeddings, DummyReranker(), factory),
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
    assert collection.inserts == [(CONVERSATION_ID, raw_text, [0.4, 0.6])]


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
