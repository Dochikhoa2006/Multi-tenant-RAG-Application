from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace

import pytest
from backend.model_config import (
    KNOWLEDGE_SEARCH,
    POLICY_SEARCH,
    RERANKER_MODEL,
    TOKEN_BUDGETS,
)
from backend.rag.retrieval import retrieve
from backend.rag.runtime import RAGRuntime, RerankResult
from backend.weaviate_client.models import SearchResult


def _uuid(index: int) -> str:
    return f"90000000-0000-0000-0000-{index:012d}"


def _result(index: int, text: str, _vector: Sequence[float], score: float) -> SearchResult:
    object_id = _uuid(index)
    return SearchResult(
        object_id=object_id,
        properties={"user_id": "usr_test", "chunk_id": object_id, "raw_text": text},
        score=score,
    )


class FakeCollection:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, list[float], int]] = []

    def hybrid_search(
        self, query: str, vector: list[float], top_k: int
    ) -> list[SearchResult]:
        self.calls.append((query, vector, top_k))
        return list(self.results)


class FakeReranker:
    def __init__(self, results: Sequence[RerankResult] = ()) -> None:
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
        return self.results


class UnusedLLM:
    def complete(self, prompt: str, **kwargs: object) -> str:
        raise AssertionError("LLM must not be used by retrieval")

    def stream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        raise AssertionError("LLM must not be used by retrieval")


class UnusedEmbeddings:
    def embed(self, text: str, *, model: str) -> Sequence[float]:
        raise AssertionError("embedding provider must not be used by retrieval")

    def embed_many(self, texts: Sequence[str], *, model: str) -> Sequence[Sequence[float]]:
        raise AssertionError("embedding provider must not be used by retrieval")


class WordTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def _runtime(reranker: FakeReranker) -> RAGRuntime:
    return RAGRuntime(
        UnusedLLM(),
        UnusedEmbeddings(),
        reranker,
        lambda user_id: SimpleNamespace(),
        tokenizer=WordTokenizer(),
    )


def test_conversation_preserves_native_mmr_order_without_python_reranking() -> None:
    collection = FakeCollection(
        [
            _result(1, "highest", [1.0, 0.0], 1.0),
            _result(2, "redundant", [0.99, 0.01], 0.95),
            _result(3, "diverse", [0.0, 1.0], 0.8),
        ]
    )

    timings: dict[str, float] = {}
    results = retrieve(
        collection,
        "question",
        [0.3, 0.7],
        "conversations",
        timing_observer=timings.__setitem__,
    )

    assert [result["raw_text"] for result in results] == [
        "highest",
        "redundant",
        "diverse",
    ]
    assert collection.calls == [("question", [0.3, 0.7], 20)]
    assert timings["conversation_mmr_rerank"] == 0.0
    assert all(
        set(result) == {
            "object_id",
            "properties",
            "raw_text",
            "hybrid_score",
            "rerank_score",
        }
        for result in results
    )
    assert [result["rerank_score"] for result in results] == [1.0, 0.95, 0.8]


def test_conversation_native_mmr_order_is_unchanged_for_equal_scores() -> None:
    collection = FakeCollection(
        [
            _result(1, "first", [1.0, 0.0], 0.8),
            _result(2, "second", [0.0, 1.0], 0.8),
        ]
    )

    results = retrieve(collection, "question", [1.0, 0.0], "conversations")

    assert [result["raw_text"] for result in results] == ["first", "second"]


def test_conversation_rejects_more_than_native_final_count() -> None:
    collection = FakeCollection(
        [_result(index, f"result-{index}", [1.0], 1.0) for index in range(1, 7)]
    )

    with pytest.raises(ValueError, match="native conversation MMR"):
        retrieve(collection, "question", [1.0], "conversations")


def test_knowledge_cross_encoder_uses_configured_model_and_order() -> None:
    collection = FakeCollection(
        [
            _result(1, "first", [1.0, 0.0], 0.9),
            _result(2, "second", [0.0, 1.0], 0.8),
        ]
    )
    reranker = FakeReranker([RerankResult(1, 0.99), RerankResult(0, 0.5)])

    results = retrieve(
        collection,
        "rewritten",
        [0.2, 0.8],
        "knowledge_facts",
        runtime=_runtime(reranker),
    )

    assert [result["raw_text"] for result in results] == ["second", "first"]
    assert [result["rerank_score"] for result in results] == [0.99, 0.5]
    assert collection.calls[0][2] == 30
    assert KNOWLEDGE_SEARCH.candidate_count == 30
    assert KNOWLEDGE_SEARCH.final_count == 8
    assert reranker.calls == [
        {
            "query": "rewritten",
            "documents": ["first", "second"],
            "model": RERANKER_MODEL,
            "top_n": 8,
        }
    ]


def test_policy_context_budget_keeps_only_complete_reranked_results() -> None:
    texts = [(f"policy-{index} " * 600).strip() for index in range(5)]
    collection = FakeCollection(
        [_result(index + 1, text, [1.0, float(index)], 1.0 - index / 10) for index, text in enumerate(texts)]
    )
    reranker = FakeReranker([RerankResult(index, 1.0 - index / 10) for index in range(5)])

    results = retrieve(
        collection,
        "query",
        [1.0, 0.0],
        "policy",
        runtime=_runtime(reranker),
    )

    tokenizer = WordTokenizer()
    context = "\n\n".join(result["raw_text"] for result in results)
    assert len(tokenizer.encode(context)) <= TOKEN_BUDGETS.policy_tokens
    assert 0 < len(results) < len(texts)
    assert [result["raw_text"] for result in results] == texts[: len(results)]
    assert collection.calls[0][2] == 20
    assert POLICY_SEARCH.candidate_count == 20
    assert POLICY_SEARCH.final_count == 5
    assert reranker.calls[0]["top_n"] == 5


def test_context_budget_drops_only_a_lowest_ranked_suffix() -> None:
    texts = [
        ("first " * 450).strip(),
        ("second " * 450).strip(),
        ("small " * 100).strip(),
    ]
    collection = FakeCollection(
        [_result(index + 1, text, [1.0, float(index)], 0.9 - index / 10) for index, text in enumerate(texts)]
    )
    reranker = FakeReranker(
        [RerankResult(index, 0.9 - index / 10) for index in range(3)]
    )

    results = retrieve(
        collection,
        "query",
        [1.0, 0.0],
        "policy",
        runtime=_runtime(reranker),
    )

    assert [result["raw_text"] for result in results] == [texts[0]]


@pytest.mark.parametrize("collection_type", ["unknown", "knowledge", ""])
def test_retrieve_rejects_unknown_collection_types(collection_type: str) -> None:
    with pytest.raises(ValueError, match="unsupported collection_type"):
        retrieve(FakeCollection([]), "query", [1.0], collection_type)


@pytest.mark.parametrize(
    ("query", "vector"),
    [
        ("", [1.0]),
        ("query", []),
        ("query", [float("inf")]),
        ("query", [True]),
    ],
)
def test_retrieve_validates_query_before_collection_access(
    query: str,
    vector: list[object],
) -> None:
    collection = FakeCollection([])

    with pytest.raises((TypeError, ValueError)):
        retrieve(collection, query, vector, "conversations")

    assert collection.calls == []


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        [object()],
        [
            SearchResult(
                object_id=_uuid(1),
                properties={"raw_text": ""},
                score=0.5,
            )
        ],
        [
            SearchResult(
                object_id=_uuid(1),
                properties={"raw_text": "text"},
                score=float("nan"),
            )
        ],
    ],
)
def test_retrieve_rejects_malformed_collection_response(response: object) -> None:
    class MalformedCollection:
        def hybrid_search(
            self,
            query: str,
            vector: list[float],
            top_k: int,
        ) -> object:
            return response

    with pytest.raises((TypeError, ValueError)):
        retrieve(MalformedCollection(), "query", [1.0], "conversations")


@pytest.mark.parametrize(
    "rerank_results",
    [
        [RerankResult(3, 0.5)],
        [RerankResult(0, 0.5), RerankResult(0, 0.4)],
        [RerankResult(0, float("nan"))],
    ],
)
def test_cross_encoder_rejects_malformed_provider_results(
    rerank_results: list[RerankResult],
) -> None:
    collection = FakeCollection([_result(1, "text", [1.0], 0.8)])
    with pytest.raises((TypeError, ValueError)):
        retrieve(
            collection,
            "query",
            [1.0],
            "knowledge_facts",
            runtime=_runtime(FakeReranker(rerank_results)),
        )
