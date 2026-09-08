from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from backend.model_config import (
    CONVERSATION_SEARCH,
    KNOWLEDGE_SEARCH,
    POLICY_SEARCH,
    RERANKER_MODEL,
    TOKEN_BUDGETS,
    RetrievalConfig,
)
from backend.rag.retrieval import _adaptive_k, _mmr, retrieve
from backend.rag.runtime import RAGRuntime, RerankResult
from backend.weaviate_client.models import HydratedSearchResult, SearchResult


def _uuid(index: int) -> str:
    return f"90000000-0000-0000-0000-{index:012d}"


@dataclass(frozen=True)
class _FixtureResult:
    search: SearchResult
    raw_text: str
    vector: tuple[float, ...]


def _chunk(
    index: int,
    text: str,
    vector: Sequence[float],
    score: float,
) -> _FixtureResult:
    object_id = _uuid(index)
    return _FixtureResult(
        search=SearchResult(object_id, object_id, text),
        raw_text=text,
        vector=tuple(vector),
    )


def _conversation_segment(
    segment_id: int,
    conversation: int,
    segment_index: int,
    *,
    raw_text: str | None = None,
    vector: Sequence[float] = (1.0, 0.0),
    score: float = 0.5,
) -> _FixtureResult:
    return _FixtureResult(
        search=SearchResult(
            object_id=_uuid(segment_id),
            canonical_id=_uuid(conversation),
            retrieval_text=f"segment-{segment_id}",
            segment_index=segment_index,
        ),
        raw_text=raw_text or f"canonical-{conversation}",
        vector=tuple(vector),
    )


class FakeCollection:
    def __init__(
        self,
        results: list[_FixtureResult] | Callable[[int], list[_FixtureResult]],
    ) -> None:
        self.results = results
        self.calls: list[tuple[str, list[list[float]], int]] = []
        self.hydration_calls: list[list[SearchResult]] = []

    def hybrid_search(
        self, query: str, vector: list[list[float]], top_k: int
    ) -> list[SearchResult]:
        self.calls.append((query, vector, top_k))
        if callable(self.results):
            return [item.search for item in self.results(top_k)]
        return [item.search for item in self.results[:top_k]]

    def hydrate_mmr_head(
        self,
        candidates: Sequence[SearchResult],
    ) -> list[HydratedSearchResult]:
        self.hydration_calls.append(list(candidates))
        fixtures = self.results(160) if callable(self.results) else self.results
        by_id = {item.search.object_id: item for item in fixtures}
        return [
            HydratedSearchResult(
                object_id=candidate.object_id,
                canonical_id=candidate.canonical_id,
                diversity_vector=by_id[candidate.object_id].vector,
                raw_text=by_id[candidate.object_id].raw_text,
                segment_index=candidate.segment_index,
            )
            for candidate in candidates
        ]


class FakeReranker:
    def __init__(self, results: Sequence[RerankResult] | None = None) -> None:
        self.results = results
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
            {"query": query, "documents": list(documents), "model": model, "top_n": top_n}
        )
        if self.results is not None:
            return self.results
        return [RerankResult(index, float(len(documents) - index)) for index in range(len(documents))]


class UnusedLLM:
    def complete(self, prompt: str, **kwargs: object) -> str:
        raise AssertionError("LLM must not be used by retrieval")

    def stream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        raise AssertionError("LLM must not be used by retrieval")


class UnusedEmbeddings:
    def embed(self, text: str, *, model: str) -> Sequence[float]:
        raise AssertionError("GTE must not be recomputed during retrieval")

    def embed_many(self, texts: Sequence[str], *, model: str) -> Sequence[Sequence[float]]:
        raise AssertionError("GTE must not be recomputed during retrieval")


class UnusedMultiVectors:
    def encode_query(self, text: str) -> Sequence[Sequence[float]]:
        raise AssertionError("query vector is encoded before retrieval")

    def encode_documents(self, texts: Sequence[str]) -> Sequence[Sequence[Sequence[float]]]:
        raise AssertionError("documents are not encoded during retrieval")


class OneSegment:
    def segment_document(self, text: str) -> Sequence[str]:
        return [text]


class WordTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def _runtime(reranker: FakeReranker) -> RAGRuntime:
    return RAGRuntime(
        UnusedLLM(),
        UnusedEmbeddings(),
        reranker,
        lambda user_id: SimpleNamespace(),
        multi_vectors=UnusedMultiVectors(),
        conversation_segmenter=OneSegment(),
        tokenizer=WordTokenizer(),
    )


def _candidate(index: int, score: float, vector: Sequence[float]) -> dict[str, object]:
    return {
        "object_id": _uuid(index),
        "rerank_score": score,
        "diversity_vector": tuple(vector),
    }


def test_collection_candidate_ceilings_and_final_maxima_are_exact() -> None:
    assert (CONVERSATION_SEARCH.candidate_count, CONVERSATION_SEARCH.candidate_ceiling) == (40, 40)
    assert (KNOWLEDGE_SEARCH.candidate_count, KNOWLEDGE_SEARCH.candidate_ceiling) == (50, 50)
    assert (POLICY_SEARCH.candidate_count, POLICY_SEARCH.candidate_ceiling) == (40, 40)
    assert (
        CONVERSATION_SEARCH.final_count,
        KNOWLEDGE_SEARCH.final_count,
        POLICY_SEARCH.final_count,
    ) == (5, 8, 5)


def test_knowledge_runs_bge_then_adaptive_k_then_bounded_mmr() -> None:
    collection = FakeCollection(
        [
            _chunk(1, "first", [1.0, 0.0], 0.9),
            _chunk(2, "second", [0.99, 0.01], 0.8),
            _chunk(3, "diverse", [0.0, 1.0], 0.7),
        ]
    )
    reranker = FakeReranker(
        [RerankResult(0, 3.0), RerankResult(1, 2.9), RerankResult(2, -0.9)]
    )
    results = retrieve(
        collection,
        "rewritten",
        [[0.2, 0.8], [0.1, 0.9]],
        "knowledge_facts",
        runtime=_runtime(reranker),
    )
    assert collection.calls == [("rewritten", [[0.2, 0.8], [0.1, 0.9]], 50)]
    assert reranker.calls == [
        {
            "query": "rewritten",
            "documents": ["first", "second", "diverse"],
            "model": RERANKER_MODEL,
            "top_n": 3,
        }
    ]
    assert len(results) == 2
    assert results == [
        {"object_id": _uuid(1), "raw_text": "first", "rerank_score": 3.0},
        {"object_id": _uuid(2), "raw_text": "second", "rerank_score": 2.9},
    ]
    assert len(collection.hydration_calls) == 1
    assert len(collection.hydration_calls[0]) == 3


def test_adaptive_k_uses_exact_decision_window_and_can_return_zero() -> None:
    config = RetrievalConfig(40, 40, 3, -1.0, 0.15, 0.70)
    eligible = [
        _candidate(1, 10.0, [1.0]),
        _candidate(2, 9.9, [1.0]),
        _candidate(3, 1.0, [1.0]),
        _candidate(4, -100.0, [1.0]),
    ]
    assert _adaptive_k([], config) == 0
    assert _adaptive_k(eligible, config) == 2
    assert _adaptive_k([_candidate(index, 2.0, [1.0]) for index in range(3)], config) == 3


def test_adaptive_k_does_not_artificially_cut_nearly_equal_logits() -> None:
    config = RetrievalConfig(40, 40, 5, -1.0, 0.15, 0.70)
    eligible = [
        _candidate(index, score, [1.0])
        for index, score in enumerate((0.04, 0.03, 0.02, 0.01, 0.0), start=1)
    ]

    assert _adaptive_k(eligible, config) == 5


def test_adaptive_k_sigmoid_is_stable_for_extreme_logits() -> None:
    config = RetrievalConfig(40, 40, 3, -1001.0, 0.15, 0.70)
    eligible = [
        _candidate(1, 1000.0, [1.0]),
        _candidate(2, 999.0, [1.0]),
        _candidate(3, -1000.0, [1.0]),
    ]

    assert _adaptive_k(eligible, config) == 2


def test_mmr_uses_normalized_bge_relevance_and_exact_two_k_head() -> None:
    config = RetrievalConfig(40, 40, 5, -1.0, 0.15, 0.50)
    eligible = [
        _candidate(1, 10.0, [1.0, 0.0]),
        _candidate(2, 9.0, [1.0, 0.0]),
        _candidate(3, 8.0, [0.0, 1.0]),
        _candidate(4, 7.0, [0.0, 1.0]),
        _candidate(5, 1000.0, [1.0, 1.0]),
    ]
    selected = _mmr(eligible, 2, config)
    assert [item["object_id"] for item in selected] == [_uuid(1), _uuid(3)]
    assert _uuid(5) not in {item["object_id"] for item in selected}
    assert all("normalized_rerank_score" not in item for item in selected)


def test_mmr_lambda_is_independently_configured_per_collection() -> None:
    assert CONVERSATION_SEARCH.mmr_lambda == 0.70
    assert KNOWLEDGE_SEARCH.mmr_lambda == 0.70
    assert POLICY_SEARCH.mmr_lambda == 0.70
    assert CONVERSATION_SEARCH is not KNOWLEDGE_SEARCH
    assert KNOWLEDGE_SEARCH is not POLICY_SEARCH


def test_conversation_overfetches_unique_identities_and_runs_bge_once() -> None:
    all_hits = [
        _conversation_segment(
            index + 1,
            (index % 20) + 1,
            index // 20,
            vector=(1.0, float((index % 20) % 2)),
        )
        for index in range(160)
    ]
    collection = FakeCollection(lambda limit: all_hits[:limit])
    reranker = FakeReranker()
    results = retrieve(
        collection,
        "question",
        [[0.3, 0.7]],
        "conversations",
        runtime=_runtime(reranker),
    )
    assert [call[2] for call in collection.calls] == [40, 80, 160]
    assert len(reranker.calls) == 1
    assert reranker.calls[0]["top_n"] == 160
    assert len(set(item["object_id"] for item in results)) == len(results)
    assert len(results) <= CONVERSATION_SEARCH.final_count
    assert len(collection.hydration_calls) == 1
    assert len(collection.hydration_calls[0]) <= 2 * CONVERSATION_SEARCH.final_count


def test_conversation_collapses_segments_by_highest_bge_and_keeps_canonical_text() -> None:
    conversation_id = _uuid(10)
    hits = [
        _conversation_segment(1, 10, 0, raw_text="Question: Q\n\nAnswer: A"),
        _conversation_segment(2, 10, 1, raw_text="Question: Q\n\nAnswer: A"),
    ]
    results = retrieve(
        FakeCollection(hits),
        "question",
        [[1.0, 0.0]],
        "conversations",
        runtime=_runtime(FakeReranker([RerankResult(0, 0.2), RerankResult(1, 0.9)])),
    )
    assert len(results) == 1
    assert results[0]["object_id"] == conversation_id
    assert results[0]["raw_text"] == "Question: Q\n\nAnswer: A"


def test_conversation_defers_integrity_checks_to_winning_head_representative() -> None:
    hits = [
        _conversation_segment(1, 10, 0, vector=(1.0, 0.0)),
        _conversation_segment(2, 10, 1, vector=(0.0, 1.0)),
    ]
    collection = FakeCollection(hits)
    results = retrieve(
        collection,
        "question",
        [[1.0]],
        "conversations",
        runtime=_runtime(FakeReranker()),
    )
    assert results == [{"object_id": _uuid(10), "raw_text": "canonical-10"}]
    assert [item.object_id for item in collection.hydration_calls[0]] == [_uuid(1)]


def test_policy_context_budget_keeps_only_complete_mmr_results() -> None:
    texts = [(f"policy-{index} " * 600).strip() for index in range(5)]
    collection = FakeCollection(
        [_chunk(index + 1, text, [1.0, float(index)], 1.0) for index, text in enumerate(texts)]
    )
    reranker = FakeReranker(
        [RerankResult(index, 1.0 - index / 100) for index in range(5)]
    )
    results = retrieve(
        collection,
        "query",
        [[1.0, 0.0]],
        "policy",
        runtime=_runtime(reranker),
    )
    context = "\n\n".join(item["raw_text"] for item in results)
    assert len(WordTokenizer().encode(context)) <= TOKEN_BUDGETS.policy_tokens
    assert 0 < len(results) < len(texts)


def test_relevance_floor_can_produce_no_context() -> None:
    collection = FakeCollection([_chunk(1, "text", [1.0], 0.8)])
    results = retrieve(
        collection,
        "query",
        [[1.0]],
        "policy",
        runtime=_runtime(FakeReranker([RerankResult(0, -2.0)])),
    )
    assert results == []
    assert collection.hydration_calls == []


@pytest.mark.parametrize("collection_type", ["unknown", "knowledge", ""])
def test_retrieve_rejects_unknown_collection_types(collection_type: str) -> None:
    with pytest.raises(ValueError, match="unsupported collection_type"):
        retrieve(FakeCollection([]), "query", [[1.0]], collection_type)


@pytest.mark.parametrize(
    ("query", "vector"),
    [
        ("", [[1.0]]),
        ("query", []),
        ("query", [[float("inf")]]),
        ("query", [[True]]),
        ("query", [[1.0], [1.0, 2.0]]),
    ],
)
def test_retrieve_validates_multi_vector_before_collection_access(
    query: str,
    vector: list[object],
) -> None:
    collection = FakeCollection([])
    with pytest.raises((TypeError, ValueError)):
        retrieve(collection, query, vector, "conversations")  # type: ignore[arg-type]
    assert collection.calls == []


@pytest.mark.parametrize(
    "rerank_results",
    [
        [RerankResult(3, 0.5)],
        [RerankResult(0, 0.5), RerankResult(0, 0.4)],
        [RerankResult(0, float("nan"))],
        [],
    ],
)
def test_cross_encoder_rejects_malformed_provider_results(
    rerank_results: list[RerankResult],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        retrieve(
            FakeCollection([_chunk(1, "text", [1.0], 0.8)]),
            "query",
            [[1.0]],
            "knowledge_facts",
            runtime=_runtime(FakeReranker(rerank_results)),
        )
