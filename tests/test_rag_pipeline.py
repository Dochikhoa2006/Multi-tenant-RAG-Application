from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
import threading
from typing import Any

import pytest

from backend.model_config import (
    CONVERSATION_SEARCH,
    EMBEDDING_MODEL,
    KNOWLEDGE_SEARCH,
    LATEON_EMBEDDING_DIMENSION,
    POLICY_SEARCH,
    RERANKER_MODEL,
)
from backend.rag.pipeline import UserRetrievalCollections, run_rag_pipeline
from backend.rag.runtime import RAGRuntime, RerankResult
from backend.weaviate_client.models import HydratedSearchResult, SearchResult


USER_ID = "usr_pipeline"
CONVERSATION_ID = "70000000-0000-0000-0000-000000000001"
ORIGINAL_QUERY = "How does that retrieval method work?"
REWRITTEN_QUERY = "Explain MMR retrieval in this RAG architecture"


def _lateon_row(first: float, second: float = 0.0) -> list[float]:
    return [first, second, *([0.0] * (LATEON_EMBEDDING_DIMENSION - 2))]


def _uuid(index: int) -> str:
    return f"71000000-0000-0000-0000-{index:012d}"


@dataclass(frozen=True)
class _FixtureResult:
    search: SearchResult
    raw_text: str
    vector: tuple[float, ...]


def _result(
    index: int,
    text: str,
    vector: Sequence[float],
    score: float,
    collection_type: str,
) -> _FixtureResult:
    object_id = _uuid(index)
    canonical_id = (
        _uuid(index + 100) if collection_type == "conversations" else object_id
    )
    return _FixtureResult(
        search=SearchResult(
            object_id=object_id,
            canonical_id=canonical_id,
            retrieval_text=text,
            segment_index=0 if collection_type == "conversations" else None,
        ),
        raw_text=text,
        vector=tuple(vector),
    )


class WordTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class RecordingEmbeddings:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[str, str, int]] = []

    def embed(self, text: str, *, model: str) -> Sequence[float]:
        self.calls.append((text, model, threading.get_ident()))
        self.events.append(f"embed:{text}")
        if text == ORIGINAL_QUERY:
            return [1.0, 0.0]
        if text == REWRITTEN_QUERY:
            return [0.0, 1.0]
        return [0.5, 0.5]

    def embed_many(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> Sequence[Sequence[float]]:
        return [[0.5, 0.5] for _ in texts]


class RecordingMultiVectors:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[str, str, int]] = []

    def encode_query(self, text: str) -> Sequence[Sequence[float]]:
        self.calls.append(("query", text, threading.get_ident()))
        self.events.append(f"multi:{text}")
        return (
            [_lateon_row(1.0, 0.0)]
            if text == ORIGINAL_QUERY
            else [_lateon_row(0.0, 1.0)]
        )

    def encode_documents(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[Sequence[float]]]:
        self.calls.append(("documents", "\n".join(texts), threading.get_ident()))
        return [[_lateon_row(0.2, 0.8)] for _ in texts]


class OneSegment:
    def segment_document(self, text: str) -> Sequence[str]:
        return [text]


class RecordingLLM:
    def __init__(
        self,
        events: list[str],
        *,
        chunks: Sequence[str] = ("MMR ", "balances diversity."),
        stream_failure: Exception | None = None,
    ) -> None:
        self.events = events
        self.chunks = list(chunks)
        self.stream_failure = stream_failure
        self.complete_calls: list[tuple[str, dict[str, object], int]] = []
        self.stream_calls: list[tuple[str, dict[str, object], int]] = []

    def complete(self, prompt: str, **kwargs: object) -> str:
        self.complete_calls.append((prompt, dict(kwargs), threading.get_ident()))
        self.events.append("rewrite")
        return REWRITTEN_QUERY

    def stream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        self.stream_calls.append((prompt, dict(kwargs), threading.get_ident()))
        self.events.append("stream_start")

        async def iterator() -> AsyncIterator[str]:
            for chunk in self.chunks:
                self.events.append(f"stream:{chunk}")
                yield chunk
                await asyncio.sleep(0)
            if self.stream_failure is not None:
                raise self.stream_failure
            self.events.append("stream_complete")

        return iterator()


class RecordingReranker:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[dict[str, object]] = []

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        model: str,
        top_n: int,
    ) -> Sequence[RerankResult]:
        self.calls.append(
            {
                "query": query,
                "documents": list(documents),
                "model": model,
                "top_n": top_n,
                "thread": threading.get_ident(),
            }
        )
        self.events.append(f"rerank:{documents[0]}")
        return [RerankResult(index, 1.0 - index / 10) for index in range(len(documents))]


class RecordingCollection:
    def __init__(
        self,
        name: str,
        results: list[_FixtureResult],
        events: list[str],
        *,
        rendezvous: threading.Barrier | None = None,
    ) -> None:
        self.user_id = USER_ID
        self.collection_type = name
        self.name = name
        self.results = results
        self.events = events
        self.rendezvous = rendezvous
        self.calls: list[tuple[str, list[list[float]], int, int]] = []

    def hybrid_search(
        self,
        query_text: str,
        query_vector: Sequence[Sequence[float]],
        top_k: int,
    ) -> list[SearchResult]:
        self.calls.append(
            (
                query_text,
                [list(row) for row in query_vector],
                top_k,
                threading.get_ident(),
            )
        )
        self.events.append(f"search:{self.name}")
        if self.rendezvous is not None:
            self.rendezvous.wait(timeout=2)
        return [item.search for item in self.results]

    def hydrate_mmr_head(
        self,
        candidates: Sequence[SearchResult],
    ) -> list[HydratedSearchResult]:
        by_id = {item.search.object_id: item for item in self.results}
        return [
            HydratedSearchResult(
                candidate.object_id,
                candidate.canonical_id,
                by_id[candidate.object_id].vector,
                by_id[candidate.object_id].raw_text,
                candidate.segment_index,
            )
            for candidate in candidates
        ]


@dataclass
class RecordingConversationWriter:
    events: list[str]

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
        self.events.append("conversation_insert")
        self.inserts.append(
            (
                conversation_id,
                raw_text,
                list(segment_texts),
                [[list(row) for row in matrix] for matrix in segment_vectors],
                list(diversity_vector),
                threading.get_ident(),
            )
        )
        return conversation_id


class RecordingQueue:
    def __init__(self, events: list[str], failure: Exception | None = None) -> None:
        self.events = events
        self.failure = failure
        self.jobs: list[tuple[str, str, Callable[[], Awaitable[None]]]] = []

    async def enqueue(
        self,
        user_id: str,
        operation: str,
        work_factory: Callable[[], Awaitable[None]],
    ) -> object:
        self.events.append("queue_accept")
        if self.failure is not None:
            raise self.failure
        self.jobs.append((user_id, operation, work_factory))
        return f"job-{len(self.jobs)}"

    async def run_next(self) -> None:
        _, _, work_factory = self.jobs.pop(0)
        await work_factory()


@dataclass
class Harness:
    runtime: RAGRuntime
    collections: UserRetrievalCollections
    events: list[str]
    embeddings: RecordingEmbeddings
    llm: RecordingLLM
    reranker: RecordingReranker
    queue: RecordingQueue
    writer: RecordingConversationWriter
    conversation: RecordingCollection
    knowledge: RecordingCollection
    policy: RecordingCollection
    factory_threads: list[int]


def _harness(
    *,
    chunks: Sequence[str] = ("MMR ", "balances diversity."),
    stream_failure: Exception | None = None,
    queue_failure: Exception | None = None,
    rendezvous: threading.Barrier | None = None,
    include_queue: bool = True,
) -> Harness:
    events: list[str] = []
    embeddings = RecordingEmbeddings(events)
    multi_vectors = RecordingMultiVectors(events)
    llm = RecordingLLM(events, chunks=chunks, stream_failure=stream_failure)
    reranker = RecordingReranker(events)
    queue = RecordingQueue(events, queue_failure)
    writer = RecordingConversationWriter(events)
    factory_threads: list[int] = []

    def factory(user_id: str) -> RecordingConversationWriter:
        assert user_id == USER_ID
        factory_threads.append(threading.get_ident())
        return writer

    runtime = RAGRuntime(
        llm,
        embeddings,
        reranker,
        factory,
        multi_vectors=multi_vectors,
        conversation_segmenter=OneSegment(),
        tokenizer=WordTokenizer(),
        background_queue=queue if include_queue else None,
    )
    conversation = RecordingCollection(
        "conversations",
        [
            _result(1, "Earlier discussion about MMR.", [1.0, 0.0], 0.9, "conversations"),
            _result(2, "Different retrieval concern.", [0.0, 1.0], 0.8, "conversations"),
        ],
        events,
    )
    knowledge = RecordingCollection(
        "knowledge_facts",
        [_result(3, "MMR combines relevance and novelty.", [1.0, 0.0], 0.9, "knowledge_facts")],
        events,
        rendezvous=rendezvous,
    )
    policy = RecordingCollection(
        "policy",
        [_result(4, "Explain tradeoffs explicitly.", [0.0, 1.0], 0.8, "policy")],
        events,
        rendezvous=rendezvous,
    )
    collections = UserRetrievalCollections(
        USER_ID,
        conversation,
        knowledge,
        policy,
    )
    return Harness(
        runtime,
        collections,
        events,
        embeddings,
        llm,
        reranker,
        queue,
        writer,
        conversation,
        knowledge,
        policy,
        factory_threads,
    )


async def _collect(harness: Harness) -> list[str]:
    return [
        chunk
        async for chunk in run_rag_pipeline(
            USER_ID,
            CONVERSATION_ID,
            ORIGINAL_QUERY,
            harness.collections,
            runtime=harness.runtime,
        )
    ]


def test_pipeline_uses_official_rewritten_query_and_enqueues_complete_answer() -> None:
    harness = _harness()
    main_thread = threading.get_ident()

    chunks = asyncio.run(_collect(harness))

    assert chunks == ["MMR ", "balances diversity."]
    assert harness.embeddings.calls == []
    assert harness.conversation.calls[0][:3] == (
        ORIGINAL_QUERY,
        [_lateon_row(1.0, 0.0)],
        CONVERSATION_SEARCH.candidate_count,
    )
    assert harness.knowledge.calls[0][:3] == (
        REWRITTEN_QUERY,
        [_lateon_row(0.0, 1.0)],
        KNOWLEDGE_SEARCH.candidate_count,
    )
    assert harness.policy.calls[0][:3] == (
        REWRITTEN_QUERY,
        [_lateon_row(0.0, 1.0)],
        POLICY_SEARCH.candidate_count,
    )
    assert {call["query"] for call in harness.reranker.calls} == {
        ORIGINAL_QUERY,
        REWRITTEN_QUERY,
    }
    assert {call["model"] for call in harness.reranker.calls} == {RERANKER_MODEL}
    assert {call["top_n"] for call in harness.reranker.calls} == {1, 2}
    prompt = harness.llm.stream_calls[0][0]
    assert REWRITTEN_QUERY in prompt
    assert ORIGINAL_QUERY not in prompt
    assert harness.events.index("stream_complete") < harness.events.index(
        "queue_accept"
    )
    assert len(harness.queue.jobs) == 1
    assert harness.writer.inserts == []

    asyncio.run(harness.queue.run_next())

    raw_text = (
        f"Question:\n{ORIGINAL_QUERY}\n\n"
        "Answer:\nMMR balances diversity."
    )
    assert harness.embeddings.calls[-1][:2] == (raw_text, EMBEDDING_MODEL)
    assert harness.writer.inserts[0][:3] == (
        CONVERSATION_ID,
        raw_text,
        [raw_text],
    )
    provider_threads = [call[2] for call in harness.embeddings.calls]
    provider_threads += [call[2] for call in harness.llm.complete_calls]
    provider_threads += [call[2] for call in harness.llm.stream_calls]
    provider_threads += [call[3] for call in harness.conversation.calls]
    provider_threads += [call[3] for call in harness.knowledge.calls]
    provider_threads += [call[3] for call in harness.policy.calls]
    provider_threads += [int(call["thread"]) for call in harness.reranker.calls]
    provider_threads += harness.factory_threads
    provider_threads += [
        call[2] for call in harness.runtime.multi_vectors.calls  # type: ignore[attr-defined]
    ]
    provider_threads.append(harness.writer.inserts[0][5])
    assert provider_threads and all(thread != main_thread for thread in provider_threads)


def test_knowledge_and_policy_retrieval_run_concurrently() -> None:
    harness = _harness(rendezvous=threading.Barrier(2))

    assert asyncio.run(_collect(harness)) == ["MMR ", "balances diversity."]


def test_partial_generation_failure_never_enqueues_conversation() -> None:
    harness = _harness(
        chunks=["partial answer"],
        stream_failure=RuntimeError("generation failed"),
    )

    async def scenario() -> list[str]:
        seen: list[str] = []
        with pytest.raises(RuntimeError, match="generation failed"):
            async for chunk in run_rag_pipeline(
                USER_ID,
                CONVERSATION_ID,
                ORIGINAL_QUERY,
                harness.collections,
                runtime=harness.runtime,
            ):
                seen.append(chunk)
        return seen

    assert asyncio.run(scenario()) == ["partial answer"]
    assert harness.queue.jobs == []


def test_cancelled_consumer_never_enqueues_partial_conversation() -> None:
    harness = _harness(chunks=["first", "second"])

    async def scenario() -> str:
        stream = run_rag_pipeline(
            USER_ID,
            CONVERSATION_ID,
            ORIGINAL_QUERY,
            harness.collections,
            runtime=harness.runtime,
        )
        first = await anext(stream)
        await stream.aclose()
        return first

    assert asyncio.run(scenario()) == "first"
    assert harness.queue.jobs == []


def test_empty_successful_stream_is_rejected_without_enqueue() -> None:
    harness = _harness(chunks=[])

    with pytest.raises(ValueError, match="non-empty answer"):
        asyncio.run(_collect(harness))

    assert harness.queue.jobs == []


def test_queue_acceptance_failure_propagates_after_complete_stream() -> None:
    harness = _harness(queue_failure=RuntimeError("queue unavailable"))

    async def scenario() -> list[str]:
        seen: list[str] = []
        with pytest.raises(RuntimeError, match="queue unavailable"):
            async for chunk in run_rag_pipeline(
                USER_ID,
                CONVERSATION_ID,
                ORIGINAL_QUERY,
                harness.collections,
                runtime=harness.runtime,
            ):
                seen.append(chunk)
        return seen

    assert asyncio.run(scenario()) == ["MMR ", "balances diversity."]
    assert harness.queue.jobs == []
    assert harness.writer.inserts == []


def test_pipeline_requires_queue_before_external_work() -> None:
    harness = _harness(include_queue=False)

    with pytest.raises(RuntimeError, match="observable background queue"):
        asyncio.run(_collect(harness))

    assert harness.embeddings.calls == []
    assert harness.conversation.calls == []


def test_collection_bundle_rejects_cross_user_and_missing_search_interface() -> None:
    harness = _harness()
    harness.policy.user_id = "usr_other"
    with pytest.raises(ValueError, match="must be bound"):
        UserRetrievalCollections(
            USER_ID,
            harness.conversation,
            harness.knowledge,
            harness.policy,
        )

    class MissingSearch:
        user_id = USER_ID
        collection_type = "policy"

    with pytest.raises(TypeError, match="hybrid_search"):
        UserRetrievalCollections(
            USER_ID,
            harness.conversation,
            harness.knowledge,
            MissingSearch(),
        )


def test_collection_bundle_rejects_swapped_collection_roles_before_external_work() -> None:
    harness = _harness()

    with pytest.raises(ValueError, match="knowledge_facts collection_type"):
        UserRetrievalCollections(
            USER_ID,
            harness.conversation,
            harness.policy,
            harness.knowledge,
        )

    assert harness.embeddings.calls == []
    assert harness.conversation.calls == []
    assert harness.knowledge.calls == []
    assert harness.policy.calls == []


@pytest.mark.parametrize("collection_type", [None, 7, "knowledge_facts"])
def test_collection_bundle_rejects_missing_non_string_or_mismatched_type(
    collection_type: object,
) -> None:
    harness = _harness()
    harness.conversation.collection_type = collection_type

    with pytest.raises(
        ValueError,
        match="conversations collection_type must be 'conversations'",
    ):
        UserRetrievalCollections(
            USER_ID,
            harness.conversation,
            harness.knowledge,
            harness.policy,
        )

    assert harness.events == []
