from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from uuid import UUID, uuid4, uuid5

import pytest

from backend.model_config import (
    CONVERSATION_SEARCH,
    FIXED_CONVERSATION_CANDIDATE_COUNT,
    FIXED_KNOWLEDGE_CANDIDATE_COUNT,
    FIXED_POLICY_CANDIDATE_COUNT,
    KNOWLEDGE_SEARCH,
    MMR_DIVERSITY_VECTOR_DIMENSION,
    POLICY_SEARCH,
    RERANKER_MODEL,
    TOKEN_BUDGETS,
    RetrievalConfig,
)
from backend.rag.retrieval import _adaptive_k, _eligible_candidates, _mmr, retrieve
from backend.rag.runtime import RAGRuntime, RerankResult
from backend.weaviate_client.models import HydratedSearchResult, SearchResult
from backend.weaviate_client.models import IncompatibleCollectionSchemaError
from backend.wizard.diagnostics import (
    DiagnosticTraceRegistry,
    activate_operation,
    install_registry,
    uninstall_registry,
)


def _uuid(index: int) -> str:
    return f"90000000-0000-0000-0000-{index:012d}"


def _stored_gte_vector(values: Sequence[float]) -> tuple[float, ...]:
    supplied = tuple(values)
    assert len(supplied) <= MMR_DIVERSITY_VECTOR_DIMENSION
    return supplied + (0.0,) * (MMR_DIVERSITY_VECTOR_DIMENSION - len(supplied))


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
        vector=_stored_gte_vector(vector),
    )


def _conversation_segment(
    segment_id: int,
    conversation: int,
    segment_index: int,
    *,
    raw_text: str | None = None,
    retrieval_text: str | None = None,
    vector: Sequence[float] = (1.0, 0.0),
    score: float = 0.5,
) -> _FixtureResult:
    canonical_id = _uuid(conversation)
    segment = retrieval_text or f"segment-{segment_id}"
    return _FixtureResult(
        search=SearchResult(
            object_id=str(
                uuid5(UUID(canonical_id), f"retrieval-segment:{segment_index}")
            ),
            canonical_id=canonical_id,
            retrieval_text=segment,
            segment_index=segment_index,
        ),
        raw_text=raw_text or f"canonical-{conversation}\n{segment}",
        vector=_stored_gte_vector(vector),
    )


class FakeCollection:
    def __init__(
        self,
        results: list[_FixtureResult] | Callable[[int], list[_FixtureResult]],
    ) -> None:
        self.results = results
        self.calls: list[tuple[str, list[list[float]], int]] = []
        self.hydration_calls: list[list[SearchResult]] = []
        self._returned_fixtures: dict[str, _FixtureResult] = {}

    def hybrid_search(
        self, query: str, vector: list[list[float]], top_k: int
    ) -> list[SearchResult]:
        self.calls.append((query, vector, top_k))
        if callable(self.results):
            fixtures = list(self.results(top_k))
        else:
            fixtures = list(self.results[:top_k])
        self._returned_fixtures.update(
            {item.search.object_id: item for item in fixtures}
        )
        return [item.search for item in fixtures]

    def hydrate_mmr_head(
        self,
        candidates: Sequence[SearchResult],
    ) -> list[HydratedSearchResult]:
        self.hydration_calls.append(list(candidates))
        return [
            HydratedSearchResult(
                object_id=candidate.object_id,
                canonical_id=candidate.canonical_id,
                diversity_vector=self._returned_fixtures[candidate.object_id].vector,
                raw_text=self._returned_fixtures[candidate.object_id].raw_text,
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
    assert (
        FIXED_CONVERSATION_CANDIDATE_COUNT,
        FIXED_KNOWLEDGE_CANDIDATE_COUNT,
        FIXED_POLICY_CANDIDATE_COUNT,
    ) == (50, 50, 40)
    assert (
        CONVERSATION_SEARCH.candidate_count,
        CONVERSATION_SEARCH.candidate_ceiling,
    ) == (FIXED_CONVERSATION_CANDIDATE_COUNT,) * 2
    assert (
        KNOWLEDGE_SEARCH.candidate_count,
        KNOWLEDGE_SEARCH.candidate_ceiling,
    ) == (FIXED_KNOWLEDGE_CANDIDATE_COUNT,) * 2
    assert (
        POLICY_SEARCH.candidate_count,
        POLICY_SEARCH.candidate_ceiling,
    ) == (FIXED_POLICY_CANDIDATE_COUNT,) * 2
    assert (
        CONVERSATION_SEARCH.final_count,
        KNOWLEDGE_SEARCH.final_count,
        POLICY_SEARCH.final_count,
    ) == (5, 8, 5)


def test_knowledge_runs_bge_then_adaptive_k_then_exact_pool_mmr() -> None:
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
    assert len(collection.hydration_calls[0]) == 2


def test_adaptive_k_uses_all_eligible_candidates_and_can_return_zero() -> None:
    config = RetrievalConfig(50, 50, 3, -1.0, 0.15, 0.70)
    eligible = [
        _candidate(1, 5.0, [1.0]),
        _candidate(2, 4.9, [1.0]),
        _candidate(3, 4.8, [1.0]),
        _candidate(4, 4.7, [1.0]),
        _candidate(5, 0.0, [1.0]),
    ]
    assert _adaptive_k([], config) == 0
    assert _adaptive_k(eligible, config) == 4
    assert _adaptive_k([_candidate(index, 2.0, [1.0]) for index in range(3)], config) == 3


def test_adaptive_k_analyzes_all_fifty_candidates_when_no_gap_qualifies() -> None:
    config = RetrievalConfig(50, 50, 5, -1.0, 0.15, 0.70)
    eligible = [
        _candidate(index, 0.5, [1.0])
        for index in range(1, 51)
    ]

    assert _adaptive_k(eligible, config) == 50


def test_adaptive_k_does_not_artificially_cut_nearly_equal_logits() -> None:
    config = RetrievalConfig(50, 50, 5, -1.0, 0.15, 0.70)
    eligible = [
        _candidate(index, score, [1.0])
        for index, score in enumerate((0.04, 0.03, 0.02, 0.01, 0.0), start=1)
    ]

    assert _adaptive_k(eligible, config) == 5


def test_adaptive_k_sigmoid_is_stable_for_extreme_logits() -> None:
    config = RetrievalConfig(50, 50, 3, -1001.0, 0.15, 0.70)
    eligible = [
        _candidate(1, 1000.0, [1.0]),
        _candidate(2, 999.0, [1.0]),
        _candidate(3, -1000.0, [1.0]),
    ]

    assert _adaptive_k(eligible, config) == 2


def test_mmr_uses_normalized_bge_relevance_across_exact_adaptive_pool() -> None:
    config = RetrievalConfig(50, 50, 5, -1.0, 0.15, 0.50)
    adaptive_pool = [
        _candidate(1, 10.0, [1.0, 0.0]),
        _candidate(2, 9.0, [1.0, 0.0]),
        _candidate(3, 8.0, [1.0, 0.0]),
        _candidate(4, 7.0, [1.0, 0.0]),
        _candidate(5, 6.0, [0.0, 1.0]),
    ]
    selected = _mmr(adaptive_pool, 2, config)
    assert [item["object_id"] for item in selected] == [_uuid(1), _uuid(5)]
    assert all("normalized_rerank_score" not in item for item in selected)


def test_mmr_greedily_selects_one_unique_item_per_round_up_to_final_limit() -> None:
    config = RetrievalConfig(50, 50, 5, -1.0, 0.15, 0.70)
    adaptive_pool = [
        _candidate(index, float(20 - index), [1.0, float(index)])
        for index in range(1, 11)
    ]

    selected = _mmr(adaptive_pool, config.final_count, config)
    assert len(selected) == config.final_count
    assert len({item["object_id"] for item in selected}) == config.final_count
    assert len(_mmr(adaptive_pool[:3], config.final_count, config)) == 3


def test_retrieve_hydrates_exact_dynamic_adaptive_pool_then_caps_final_count() -> None:
    fixtures = [
        _chunk(index, f"chunk-{index}", [1.0, float(index)], 1.0)
        for index in range(1, 13)
    ]
    scores = [5.0 - index / 10 for index in range(9)] + [0.0, -0.1, -0.2]
    collection = FakeCollection(fixtures)

    results = retrieve(
        collection,
        "query",
        [[1.0]],
        "knowledge_facts",
        runtime=_runtime(
            FakeReranker(
                [RerankResult(index, score) for index, score in enumerate(scores)]
            )
        ),
    )

    assert len(collection.hydration_calls) == 1
    assert len(collection.hydration_calls[0]) == 9
    assert len(results) == KNOWLEDGE_SEARCH.final_count


def test_retrieve_hydrates_all_fifty_when_adaptive_k_keeps_all() -> None:
    fixtures = [
        _chunk(index, f"chunk-{index}", [1.0, float(index)], 1.0)
        for index in range(1, 51)
    ]
    collection = FakeCollection(fixtures)

    results = retrieve(
        collection,
        "query",
        [[1.0]],
        "knowledge_facts",
        runtime=_runtime(
            FakeReranker([RerankResult(index, 0.5) for index in range(50)])
        ),
    )

    assert len(collection.hydration_calls) == 1
    assert len(collection.hydration_calls[0]) == 50
    assert len(results) == KNOWLEDGE_SEARCH.final_count


def test_vector_recovery_stays_in_adaptive_pool_and_obeys_final_limit() -> None:
    fixtures = [
        _chunk(index, f"chunk-{index}", [1.0, float(index)], 1.0)
        for index in range(1, 13)
    ]
    scores = [5.0 - index / 10 for index in range(9)] + [0.0, -0.1, -0.2]

    class CorruptAdaptivePoolVector(FakeCollection):
        def hydrate_mmr_head(
            self,
            candidates: Sequence[SearchResult],
        ) -> list[HydratedSearchResult]:
            hydrated = super().hydrate_mmr_head(candidates)
            first = hydrated[0]
            hydrated[0] = HydratedSearchResult(
                first.object_id,
                first.canonical_id,
                None,
                first.raw_text,
                first.segment_index,
            )
            return hydrated

    collection = CorruptAdaptivePoolVector(fixtures)
    results = retrieve(
        collection,
        "query",
        [[1.0]],
        "knowledge_facts",
        runtime=_runtime(
            FakeReranker(
                [RerankResult(index, score) for index, score in enumerate(scores)]
            )
        ),
    )

    assert [item.object_id for item in collection.hydration_calls[0]] == [
        _uuid(index) for index in range(1, 10)
    ]
    assert [item["object_id"] for item in results] == [
        _uuid(index) for index in range(1, 9)
    ]


def test_mmr_lambda_is_independently_configured_per_collection() -> None:
    assert CONVERSATION_SEARCH.mmr_lambda == 0.70
    assert KNOWLEDGE_SEARCH.mmr_lambda == 0.70
    assert POLICY_SEARCH.mmr_lambda == 0.70
    assert CONVERSATION_SEARCH is not KNOWLEDGE_SEARCH
    assert KNOWLEDGE_SEARCH is not POLICY_SEARCH


def test_conversation_searches_fifty_segment_hits_once_and_runs_bge_once() -> None:
    all_hits = [
        _conversation_segment(
            index + 1,
            (index % 17) + 1,
            index // 17,
            vector=(1.0, float((index % 17) % 2)),
        )
        for index in range(50)
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
    assert [call[2] for call in collection.calls] == [50]
    assert len(reranker.calls) == 1
    assert reranker.calls[0]["top_n"] == 50
    assert len(set(item["object_id"] for item in results)) == len(results)
    assert len(results) <= CONVERSATION_SEARCH.final_count
    assert len(collection.hydration_calls) == 1
    assert len(collection.hydration_calls[0]) == 17


def test_conversation_deep_trace_observes_existing_calls_without_duplicates() -> None:
    hits = [
        _conversation_segment(index, index, 0)
        for index in range(1, 4)
    ]
    collection = FakeCollection(hits)
    reranker = FakeReranker()
    registry = DiagnosticTraceRegistry("diagnostic_user")
    trace_session = str(uuid4())
    operation_id = str(uuid4())
    chat_session = str(uuid4())
    registry.start("diagnostic_user", trace_session, "run")
    handle = registry.begin_operation(
        user_id="diagnostic_user",
        session_id=trace_session,
        operation_id=operation_id,
        kind="chat_query",
        collection_type="conversations",
        wizard_id=chat_session,
    )
    assert handle is not None
    install_registry(registry)
    try:
        with activate_operation(handle):
            results = retrieve(
                collection,
                "question",
                [[0.3, 0.7]],
                "conversations",
                runtime=_runtime(reranker),
            )
    finally:
        uninstall_registry(registry)

    operation = registry.snapshot(
        "diagnostic_user", trace_session, operation_id
    )["operations"][0]
    assert len(collection.calls) == 1
    assert len(reranker.calls) == 1
    assert len(collection.hydration_calls) == 1
    assert set(operation["stages"]) == {
        "chat.conversation_hybrid",
        "chat.conversation_bge",
        "chat.conversation_collapse",
        "chat.conversation_relevance_floor",
        "chat.conversation_adaptive_k",
        "chat.conversation_hydration",
        "chat.conversation_mmr",
        "chat.conversation_finalize",
    }
    assert operation["counts"]["conversation_hybrid_storage_call_count"] == 1
    assert operation["counts"]["conversation_bge_model_call_count"] == 1
    assert operation["counts"]["conversation_hydration_storage_call_count"] == 1
    assert operation["counts"]["conversation_final_count"] == len(results)
    assert operation["samples"]["conversation_final_ids"]["truncated"] is False
    assert set(operation["digests"]) == {"conversation_final_context_sha256"}


@pytest.mark.parametrize(
    ("collection_type", "prefix"),
    [("knowledge_facts", "knowledge"), ("policy", "policy")],
)
def test_kp_deep_trace_observes_existing_calls_without_duplicates(
    collection_type: str,
    prefix: str,
) -> None:
    collection = FakeCollection(
        [
            _chunk(1, f"{prefix} one", (1.0, 0.0), 0.9),
            _chunk(2, f"{prefix} two", (0.0, 1.0), 0.8),
        ]
    )
    reranker = FakeReranker()
    registry = DiagnosticTraceRegistry("diagnostic_user")
    trace_session = str(uuid4())
    operation_id = str(uuid4())
    registry.start("diagnostic_user", trace_session, "run")
    handle = registry.begin_operation(
        user_id="diagnostic_user",
        session_id=trace_session,
        operation_id=operation_id,
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    )
    assert handle is not None
    install_registry(registry)
    try:
        with activate_operation(handle):
            results = retrieve(
                collection,
                "rewritten question",
                [[0.3, 0.7]],
                collection_type,
                runtime=_runtime(reranker),
            )
    finally:
        uninstall_registry(registry)

    operation = registry.snapshot(
        "diagnostic_user", trace_session, operation_id
    )["operations"][0]
    assert len(collection.calls) == 1
    assert len(reranker.calls) == 1
    assert len(collection.hydration_calls) == 1
    assert {
        f"chat.{prefix}_hybrid",
        f"chat.{prefix}_bge",
        f"chat.{prefix}_relevance_floor",
        f"chat.{prefix}_adaptive_k",
        f"chat.{prefix}_hydration",
        f"chat.{prefix}_mmr",
        f"chat.{prefix}_finalize",
        f"chat.{prefix}_budgeting",
    } <= set(operation["stages"])
    assert operation["counts"][f"{prefix}_hybrid_storage_call_count"] == 1
    assert operation["counts"][f"{prefix}_bge_model_call_count"] == 1
    assert operation["counts"][f"{prefix}_hydration_storage_call_count"] == 1
    assert operation["counts"][f"{prefix}_final_count"] == len(results)
    assert operation["samples"][f"{prefix}_final_ids"]["truncated"] is False
    assert (
        operation["samples"][f"{prefix}_final_item_fingerprints"]["exact_count"]
        == len(results)
    )


def _run_traced_retrieval(
    collection: FakeCollection,
    reranker: FakeReranker,
    collection_type: str,
    *,
    capture_mode: str = "deep",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    registry = DiagnosticTraceRegistry("diagnostic_user", capture_mode=capture_mode)
    trace_session = str(uuid4())
    operation_id = str(uuid4())
    registry.start("diagnostic_user", trace_session, "run")
    handle = registry.begin_operation(
        user_id="diagnostic_user",
        session_id=trace_session,
        operation_id=operation_id,
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    )
    assert handle is not None
    install_registry(registry)
    try:
        with activate_operation(handle):
            results = retrieve(
                collection,
                "rewritten question",
                [[0.3, 0.7]],
                collection_type,
                runtime=_runtime(reranker),
            )
    finally:
        uninstall_registry(registry)
    if capture_mode == "evaluation":
        registry.finish(handle, "succeeded")
    operation = registry.snapshot(
        "diagnostic_user", trace_session, operation_id
    )["operations"][0]
    return results, operation


def _candidate_observations(
    operation: dict[str, object], prefix: str
) -> list[dict[str, str]]:
    samples = operation["samples"]
    texts = operation["texts"]
    assert isinstance(samples, dict)
    assert isinstance(texts, dict)
    fields = texts[f"{prefix}_candidate_decision_fields"].split("|")
    rows: list[str] = []
    for page in (1, 2):
        rows.extend(samples[f"{prefix}_candidate_decisions_page_{page}"]["items"])
    return [dict(zip(fields, row.split("|"), strict=True)) for row in rows]


@pytest.mark.parametrize(
    ("collection_type", "prefix", "candidate_count"),
    [
        ("knowledge_facts", "knowledge", 50),
        ("policy", "policy", 40),
    ],
)
def test_candidate_observation_is_complete_and_does_not_change_work(
    collection_type: str,
    prefix: str,
    candidate_count: int,
) -> None:
    fixtures = [
        _chunk(index, "duplicate text", (1.0, 0.0), 0.5)
        for index in range(1, candidate_count + 1)
    ]
    plain_collection = FakeCollection(fixtures)
    plain_reranker = FakeReranker(
        [RerankResult(index, 0.5) for index in range(candidate_count)]
    )
    plain = retrieve(
        plain_collection,
        "rewritten question",
        [[0.3, 0.7]],
        collection_type,
        runtime=_runtime(plain_reranker),
    )
    traced_collection = FakeCollection(fixtures)
    traced_reranker = FakeReranker(
        [RerankResult(index, 0.5) for index in range(candidate_count)]
    )
    traced, operation = _run_traced_retrieval(
        traced_collection,
        traced_reranker,
        collection_type,
    )

    assert traced == plain
    assert traced_collection.calls == plain_collection.calls
    assert traced_collection.hydration_calls == plain_collection.hydration_calls
    assert traced_reranker.calls == plain_reranker.calls
    rows = _candidate_observations(operation, prefix)
    assert len(rows) == candidate_count
    assert [int(row["hybrid_rank"]) for row in rows] == list(
        range(1, candidate_count + 1)
    )
    assert [int(row["bge_rank"]) for row in rows] == list(
        range(1, candidate_count + 1)
    )
    assert operation["samples"][f"{prefix}_candidate_decisions_page_1"][
        "exact_count"
    ] == 32
    assert operation["samples"][f"{prefix}_candidate_decisions_page_2"][
        "exact_count"
    ] == candidate_count - 32
    digest_rows = []
    for page in (1, 2):
        digest_rows.extend(
            operation["samples"][f"{prefix}_candidate_text_digests_page_{page}"][
                "items"
            ]
        )
    assert len(digest_rows) == candidate_count
    assert len({row.split(":", 1)[1] for row in digest_rows}) == 1
    assert "mmr_limit" in {
        row["first_exclusion_reason"] for row in rows
    }


def test_candidate_observation_distinguishes_floor_and_adaptive_rejection() -> None:
    collection = FakeCollection(
        [
            _chunk(1, "one", (1.0, 0.0), 0.5),
            _chunk(2, "two", (0.0, 1.0), 0.5),
            _chunk(3, "three", (1.0, 1.0), 0.5),
            _chunk(4, "four", (1.0, -1.0), 0.5),
        ]
    )
    reranker = FakeReranker(
        [
            RerankResult(0, 3.0),
            RerankResult(1, 2.9),
            RerankResult(2, -0.9),
            RerankResult(3, -1.1),
        ]
    )

    _, operation = _run_traced_retrieval(
        collection, reranker, "knowledge_facts"
    )
    rows = _candidate_observations(operation, "knowledge")

    assert [row["first_exclusion_reason"] for row in rows] == [
        "selected",
        "selected",
        "adaptive_gap",
        "floor",
    ]
    assert rows[2]["floor_pass"] == "1"
    assert rows[2]["adaptive_inclusion"] == "0"
    assert rows[3]["floor_pass"] == "0"
    assert float.fromhex(rows[3]["raw_score_hex"]) == -1.1


def test_candidate_observation_floor_equality_and_all_negative_scores() -> None:
    collection = FakeCollection(
        [
            _chunk(1, "boundary", (1.0, 0.0), 0.5),
            _chunk(2, "below", (0.0, 1.0), 0.5),
        ]
    )
    reranker = FakeReranker(
        [RerankResult(0, -1.0), RerankResult(1, -1.0001)]
    )

    _, operation = _run_traced_retrieval(
        collection, reranker, "knowledge_facts"
    )
    rows = _candidate_observations(operation, "knowledge")

    assert rows[0]["floor_pass"] == "1"
    assert rows[0]["first_exclusion_reason"] == "selected"
    assert rows[1]["floor_pass"] == "0"
    assert rows[1]["first_exclusion_reason"] == "floor"
    assert float.fromhex(rows[0]["floor_hex"]) == -1.0

    all_rejected, rejected_operation = _run_traced_retrieval(
        FakeCollection(
            [
                _chunk(1, "first", (1.0, 0.0), 0.5),
                _chunk(2, "second", (0.0, 1.0), 0.5),
            ]
        ),
        FakeReranker(
            [RerankResult(0, -1.01), RerankResult(1, -2.0)]
        ),
        "knowledge_facts",
    )
    assert all_rejected == []
    assert {
        row["first_exclusion_reason"]
        for row in _candidate_observations(rejected_operation, "knowledge")
    } == {"floor"}


def test_candidate_observation_records_hydration_mmr_fallback_and_budget() -> None:
    class MixedCollection(FakeCollection):
        def hydrate_mmr_head(
            self,
            candidates: Sequence[SearchResult],
        ) -> list[HydratedSearchResult]:
            hydrated = super().hydrate_mmr_head(candidates)
            first = hydrated[0]
            hydrated[0] = HydratedSearchResult(
                first.object_id,
                first.canonical_id,
                None,
                quarantine_reason="malformed_candidate",
            )
            second = hydrated[1]
            hydrated[1] = HydratedSearchResult(
                second.object_id,
                second.canonical_id,
                None,
                second.raw_text,
                second.segment_index,
            )
            return hydrated

    long_texts = [f"item-{index} " + "word " * 400 for index in range(1, 11)]
    fixtures = [
        _chunk(index, text, (1.0, 0.0), 0.5)
        for index, text in enumerate(long_texts, start=1)
    ]
    collection = MixedCollection(fixtures)
    reranker = FakeReranker(
        [RerankResult(index, 0.5) for index in range(len(fixtures))]
    )

    _, operation = _run_traced_retrieval(
        collection, reranker, "knowledge_facts"
    )
    rows = _candidate_observations(operation, "knowledge")

    assert rows[0]["hydration_outcome"] == "f"
    assert rows[0]["first_exclusion_reason"] == "hydration"
    assert rows[1]["hydration_outcome"] == "y"
    assert rows[9]["first_exclusion_reason"] == "fallback_limit"
    assert "budget" in {row["first_exclusion_reason"] for row in rows}


def test_candidate_observer_failure_cannot_change_retrieval() -> None:
    class FaultingRegistry(DiagnosticTraceRegistry):
        def set_text(
            self, handle: object, name: str, value: str
        ) -> None:
            if name == "knowledge_candidate_decision_fields":
                raise RuntimeError("observer fault")
            super().set_text(handle, name, value)  # type: ignore[arg-type]

    fixtures = [
        _chunk(1, "first", (1.0, 0.0), 0.5),
        _chunk(2, "second", (0.0, 1.0), 0.5),
    ]
    plain_collection = FakeCollection(fixtures)
    plain_reranker = FakeReranker()
    expected = retrieve(
        plain_collection,
        "rewritten question",
        [[0.3, 0.7]],
        "knowledge_facts",
        runtime=_runtime(plain_reranker),
    )
    registry = FaultingRegistry("diagnostic_user")
    trace_session = str(uuid4())
    registry.start("diagnostic_user", trace_session, "run")
    handle = registry.begin_operation(
        user_id="diagnostic_user",
        session_id=trace_session,
        operation_id=str(uuid4()),
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    )
    assert handle is not None
    traced_collection = FakeCollection(fixtures)
    traced_reranker = FakeReranker()
    install_registry(registry)
    try:
        with activate_operation(handle):
            actual = retrieve(
                traced_collection,
                "rewritten question",
                [[0.3, 0.7]],
                "knowledge_facts",
                runtime=_runtime(traced_reranker),
            )
    finally:
        uninstall_registry(registry)

    snapshot = registry.snapshot("diagnostic_user", trace_session)
    assert actual == expected
    assert traced_collection.calls == plain_collection.calls
    assert traced_collection.hydration_calls == plain_collection.hydration_calls
    assert traced_reranker.calls == plain_reranker.calls
    assert snapshot["trace_faulted"] is True


def test_candidate_derivation_runs_only_in_deep_trace_and_fault_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.rag import retrieval

    observations: list[object] = []

    def failed_observer(**kwargs: object) -> None:
        observations.append(kwargs)
        raise RuntimeError("derivation failed before sample recording")

    monkeypatch.setattr(retrieval, "_observe_candidate_decisions", failed_observer)
    fixtures = [_chunk(1, "first", (1.0, 0.0), 0.5)]
    plain_collection, plain_reranker = FakeCollection(fixtures), FakeReranker()
    plain = retrieve(
        plain_collection, "rewritten question", [[0.3, 0.7]],
        "knowledge_facts", runtime=_runtime(plain_reranker),
    )
    assert observations == []
    for mode in ("evaluation", "deep"):
        collection, reranker = FakeCollection(fixtures), FakeReranker()
        actual, _ = _run_traced_retrieval(
            collection, reranker, "knowledge_facts", capture_mode=mode,
        )
        assert actual == plain
        assert collection.calls == plain_collection.calls
        assert collection.hydration_calls == plain_collection.hydration_calls
        assert reranker.calls == plain_reranker.calls
        assert len(observations) == (1 if mode == "deep" else 0)


def test_kp_trace_records_empty_pool_without_extra_model_or_storage_calls() -> None:
    collection = FakeCollection([])
    reranker = FakeReranker()

    results, operation = _run_traced_retrieval(
        collection, reranker, "knowledge_facts"
    )

    assert results == []
    assert len(collection.calls) == 1
    assert collection.hydration_calls == []
    assert reranker.calls == []
    assert operation["counts"]["knowledge_bge_model_call_count"] == 0
    assert operation["counts"]["knowledge_hydration_storage_call_count"] == 0
    assert operation["counts"]["knowledge_final_count"] == 0
    assert operation["samples"]["knowledge_final_ids"] == {
        "exact_count": 0,
        "items": [],
        "truncated": False,
    }


def test_conversation_short_page_does_not_retry() -> None:
    hits = [_conversation_segment(index, index, 0) for index in range(1, 18)]
    collection = FakeCollection(hits)
    reranker = FakeReranker()

    retrieve(
        collection,
        "question",
        [[0.3, 0.7]],
        "conversations",
        runtime=_runtime(reranker),
    )

    assert [call[2] for call in collection.calls] == [50]
    assert len(reranker.calls) == 1
    assert reranker.calls[0]["top_n"] == 17


def test_conversation_collapses_segments_by_highest_bge_and_keeps_canonical_text() -> None:
    conversation_id = _uuid(10)
    hits = [
        _conversation_segment(
            1,
            10,
            0,
            raw_text="Question: Q\n\nAnswer: A",
            retrieval_text="Question: Q",
        ),
        _conversation_segment(
            2,
            10,
            1,
            raw_text="Question: Q\n\nAnswer: A",
            retrieval_text="Answer: A",
        ),
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
    canonical = "segment-1 segment-2"
    hits = [
        _conversation_segment(1, 10, 0, raw_text=canonical, vector=(1.0, 0.0)),
        _conversation_segment(2, 10, 1, raw_text=canonical, vector=(0.0, 1.0)),
    ]
    collection = FakeCollection(hits)
    results = retrieve(
        collection,
        "question",
        [[1.0]],
        "conversations",
        runtime=_runtime(FakeReranker()),
    )
    assert results == [{"object_id": _uuid(10), "raw_text": canonical}]
    assert [item.object_id for item in collection.hydration_calls[0]] == [
        hits[0].search.object_id
    ]


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
        runtime=_runtime(FakeReranker([RerankResult(0, -3.01)])),
    )
    assert results == []
    assert collection.hydration_calls == []


@pytest.mark.parametrize(
    ("scores", "expected"),
    [([], []), ([-3.0], [1]), ([math.nextafter(-3.0, -math.inf)], []),
     ([-1.0, -2.0], [1]), ([-0.5, -0.6, -2.0], [1, 2]),
     ([-2.0, -2.0, -2.1], [1])],
)
def test_policy_rescue_boundaries_and_nonempty_pool(scores, expected) -> None:
    candidates = [_candidate(i, s, [1.0]) for i, s in enumerate(scores, 1)]
    selected = _eligible_candidates(candidates, POLICY_SEARCH, collection_type="policy")
    assert [c["object_id"] for c in selected] == [_uuid(i) for i in expected]
    assert all(any(c is original for original in candidates) for c in selected)
    assert _adaptive_k(selected, POLICY_SEARCH) == len(selected)


@pytest.mark.parametrize("floor", [-4.0, -2.5, -1.0, 0.0])
def test_policy_rescue_preserves_configured_floor(floor) -> None:
    config = replace(POLICY_SEARCH, adaptive_relevance_floor=floor)
    candidates = [_candidate(1, -2.0, [1.0]), _candidate(2, -3.5, [1.0])]
    ordinary = [c for c in candidates if c["rerank_score"] >= floor]
    assert _eligible_candidates(candidates, config, collection_type="policy") == (
        ordinary or candidates[:1]
    )


@pytest.mark.parametrize("collection_type", ["knowledge_facts", "conversations"])
def test_policy_rescue_does_not_apply_to_other_collections(collection_type) -> None:
    fixture = (_conversation_segment(1, 10, 0) if collection_type == "conversations"
               else _chunk(1, "rule", [1.0], 0.0))
    collection = FakeCollection([fixture])
    reranker = FakeReranker([RerankResult(0, -2.0)])
    assert retrieve(collection, "query", [[1.0]], collection_type, runtime=_runtime(reranker)) == []
    assert len(collection.calls) == len(reranker.calls) == 1
    assert collection.hydration_calls == []


@pytest.mark.parametrize("observer_fault", [False, True])
def test_policy_rescue_trace_tie_order_and_call_parity(monkeypatch, observer_fault) -> None:
    import backend.rag.retrieval as module
    fixtures = [_chunk(2, "second", [0.0, 1.0], 0.0), _chunk(1, "first", [1.0, 0.0], 0.0)]
    scores = [RerankResult(0, -2.0), RerankResult(1, -2.0)]
    plain_collection = FakeCollection(fixtures)
    plain_reranker = FakeReranker(scores)
    plain = retrieve(plain_collection, "rewritten question", [[0.3, 0.7]], "policy", runtime=_runtime(plain_reranker))
    if observer_fault:
        def fail(**kwargs):
            raise RuntimeError("observation only")
        monkeypatch.setattr(module, "_observe_candidate_decisions", fail)
    collection = FakeCollection(fixtures)
    reranker = FakeReranker(scores)
    traced, operation = _run_traced_retrieval(collection, reranker, "policy")
    assert traced == plain
    assert [r["object_id"] for r in traced] == [_uuid(1)]
    assert collection.calls == plain_collection.calls
    assert collection.hydration_calls == plain_collection.hydration_calls
    assert reranker.calls == plain_reranker.calls
    assert len(collection.calls) == len(collection.hydration_calls) == len(reranker.calls) == 1
    if not observer_fault:
        rows = _candidate_observations(operation, "policy")
        assert float.fromhex(rows[0]["floor_hex"]) == -3.0
        assert rows[0]["first_exclusion_reason"] == "selected"
        assert float.fromhex(rows[1]["floor_hex"]) == -1.0
        assert rows[1]["first_exclusion_reason"] == "floor"


@pytest.mark.parametrize("outcome", ["valid", "missing", "malformed", "vector", "budget"])
def test_policy_rescue_uses_unchanged_hydration_and_budget(outcome) -> None:
    class Collection(FakeCollection):
        def hydrate_mmr_head(self, candidates):
            hydrated = super().hydrate_mmr_head(candidates)
            if outcome == "missing":
                return []
            if outcome == "malformed":
                return [replace(hydrated[0], quarantine_reason="malformed_candidate")]
            if outcome == "vector":
                return [replace(hydrated[0], diversity_vector=None)]
            return hydrated
    text = "word " * (TOKEN_BUDGETS.policy_tokens + 1) if outcome == "budget" else "support"
    collection = Collection([_chunk(1, text, [1.0], 0.0)])
    reranker = FakeReranker([RerankResult(0, -2.697265625)])
    if outcome == "missing":
        with pytest.raises(TypeError, match="one ordered result"):
            retrieve(collection, "query", [[1.0]], "policy", runtime=_runtime(reranker))
    else:
        results, operation = _run_traced_retrieval(collection, reranker, "policy")
        assert bool(results) == (outcome in {"valid", "vector"})
        row = _candidate_observations(operation, "policy")[0]
        assert row["first_exclusion_reason"] == {"valid": "selected", "vector": "selected", "malformed": "hydration", "budget": "budget"}[outcome]
        assert operation["flags"]["policy_mmr_fallback"] == (outcome == "vector")
    assert len(collection.calls) == len(collection.hydration_calls) == len(reranker.calls) == 1


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


def test_malformed_candidate_is_quarantined_without_reordering_valid_work(
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = _chunk(1, "first", [1.0, 0.0], 0.8)
    second = _chunk(2, "second", [0.0, 1.0], 0.7)
    secret_text = "sensitive malformed object contents"

    class MixedCollection(FakeCollection):
        def hybrid_search(
            self,
            query: str,
            vector: list[list[float]],
            top_k: int,
        ) -> list[object]:
            valid = super().hybrid_search(query, vector, top_k)
            return [valid[0], {"raw_text": secret_text}, valid[1]]

    collection = MixedCollection([first, second])
    results = retrieve(
        collection,
        "query",
        [[1.0]],
        "knowledge_facts",
        runtime=_runtime(FakeReranker()),
    )

    assert [item["object_id"] for item in results] == [_uuid(1), _uuid(2)]
    assert any(
        getattr(record, "reason", None) == "invalid_result_type"
        for record in caplog.records
    )
    assert secret_text not in caplog.text


def test_conversation_segment_identity_conflict_quarantines_whole_group() -> None:
    corrupt = _conversation_segment(1, 10, 0)
    valid = _conversation_segment(2, 20, 0)
    conflicting = SearchResult(
        object_id=_uuid(999),
        canonical_id=corrupt.search.canonical_id,
        retrieval_text="conflicting segment",
        segment_index=0,
    )

    class ConflictingCollection(FakeCollection):
        def hybrid_search(
            self,
            query: str,
            vector: list[list[float]],
            top_k: int,
        ) -> list[SearchResult]:
            super().hybrid_search(query, vector, top_k)
            return [corrupt.search, conflicting, valid.search]

    reranker = FakeReranker()
    results = retrieve(
        ConflictingCollection([corrupt, valid]),
        "query",
        [[1.0]],
        "conversations",
        runtime=_runtime(reranker),
    )

    assert [item["object_id"] for item in results] == [valid.search.canonical_id]
    assert reranker.calls[0]["documents"] == [valid.search.retrieval_text]


def test_isolated_unusable_mmr_vector_falls_back_to_bge_order() -> None:
    fixtures = [
        _chunk(1, "first", [1.0, 0.0], 0.9),
        _chunk(2, "similar", [1.0, 0.0], 0.8),
        _chunk(3, "diverse", [0.0, 1.0], 0.7),
    ]

    class CorruptVectorCollection(FakeCollection):
        def hydrate_mmr_head(
            self,
            candidates: Sequence[SearchResult],
        ) -> list[HydratedSearchResult]:
            hydrated = super().hydrate_mmr_head(candidates)
            first = hydrated[0]
            hydrated[0] = HydratedSearchResult(
                first.object_id,
                first.canonical_id,
                None,
                first.raw_text,
                first.segment_index,
            )
            return hydrated

    collection = CorruptVectorCollection(fixtures)
    reranker = FakeReranker(
        [
            RerankResult(0, 3.0),
            RerankResult(1, 2.9),
            RerankResult(2, -0.9),
        ]
    )
    results, operation = _run_traced_retrieval(
        collection,
        reranker,
        "knowledge_facts",
    )

    assert [item["object_id"] for item in results] == [_uuid(1), _uuid(2)]
    assert len(collection.calls) == 1
    assert len(collection.hydration_calls) == 1
    assert len(reranker.calls) == 1
    assert operation["flags"]["knowledge_mmr_usable"] is False
    assert operation["flags"]["knowledge_mmr_fallback"] is True
    assert operation["flags"]["knowledge_hydration_values_finite"] is False


def test_corrupt_hydrated_candidate_is_removed_without_extra_fetch() -> None:
    fixtures = [
        _chunk(1, "first", [1.0, 0.0], 0.9),
        _chunk(2, "second", [0.0, 1.0], 0.8),
    ]

    class QuarantineCollection(FakeCollection):
        def hydrate_mmr_head(
            self,
            candidates: Sequence[SearchResult],
        ) -> list[HydratedSearchResult]:
            hydrated = super().hydrate_mmr_head(candidates)
            first = hydrated[0]
            hydrated[0] = HydratedSearchResult(
                first.object_id,
                first.canonical_id,
                None,
                quarantine_reason="malformed_candidate",
            )
            return hydrated

    collection = QuarantineCollection(fixtures)
    results = retrieve(
        collection,
        "query",
        [[1.0]],
        "knowledge_facts",
        runtime=_runtime(FakeReranker()),
    )

    assert [item["object_id"] for item in results] == [_uuid(2)]
    assert len(collection.hydration_calls) == 1


def test_corrupt_conversation_representative_is_quarantined() -> None:
    collection = FakeCollection(
        [
            _conversation_segment(
                1,
                10,
                0,
                raw_text="canonical text without the selected segment",
                retrieval_text="different segment",
            )
        ]
    )

    assert retrieve(
        collection,
        "query",
        [[1.0]],
        "conversations",
        runtime=_runtime(FakeReranker()),
    ) == []


def test_systemic_schema_mismatch_remains_fatal() -> None:
    class StaleCollection(FakeCollection):
        def hybrid_search(
            self,
            query: str,
            vector: list[list[float]],
            top_k: int,
        ) -> list[SearchResult]:
            raise IncompatibleCollectionSchemaError("KnowledgeFacts_test", "stale")

    with pytest.raises(IncompatibleCollectionSchemaError, match="stale"):
        retrieve(
            StaleCollection([]),
            "query",
            [[1.0]],
            "knowledge_facts",
            runtime=_runtime(FakeReranker()),
        )


def test_bge_failure_remains_fatal_without_hydration_or_hybrid_fallback() -> None:
    class FailedReranker(FakeReranker):
        def rerank(self, *args: object, **kwargs: object) -> Sequence[RerankResult]:
            raise RuntimeError("BGE unavailable")

    collection = FakeCollection([_chunk(1, "first", [1.0], 0.8)])
    with pytest.raises(RuntimeError, match="BGE unavailable"):
        retrieve(
            collection,
            "query",
            [[1.0]],
            "knowledge_facts",
            runtime=_runtime(FailedReranker()),
        )
    assert collection.hydration_calls == []
