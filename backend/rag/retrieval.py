"""Hybrid candidate retrieval, MMR/cross-encoder reranking, and budgeting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
import math
from time import perf_counter
from typing import Any

import tiktoken

from backend.model_config import (
    CONVERSATION_SEARCH,
    KNOWLEDGE_SEARCH,
    POLICY_SEARCH,
    RERANKER_MODEL,
    TEXT_PROCESSING,
    TOKEN_BUDGETS,
    RetrievalConfig,
)
from backend.rag.runtime import (
    RAGRuntime,
    RerankResult,
    TimingObserver,
    Tokenizer,
    resolve_runtime,
)
from backend.weaviate_client.models import SearchResult


_COLLECTION_CONFIGS: dict[str, RetrievalConfig] = {
    "conversations": CONVERSATION_SEARCH,
    "knowledge_facts": KNOWLEDGE_SEARCH,
    "policy": POLICY_SEARCH,
}
_CONTEXT_BUDGETS = {
    "knowledge_facts": TOKEN_BUDGETS.knowledge_tokens,
    "policy": TOKEN_BUDGETS.policy_tokens,
}
_CONTEXT_SEPARATOR = "\n\n"
_HYBRID_TIMING_PHASES = {
    "conversations": "conversation_hybrid_search",
    "knowledge_facts": "knowledge_hybrid_search",
    "policy": "policy_hybrid_search",
}
_RERANK_TIMING_PHASES = {
    "knowledge_facts": "knowledge_cross_encoder_rerank",
    "policy": "policy_cross_encoder_rerank",
}
_CONVERSATION_OVERFETCH_MULTIPLIERS = (1, 2, 4)
_LOGGER = logging.getLogger(__name__)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _vector(value: object, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise TypeError(f"{name} must contain only numbers")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain only numbers") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} values must be finite")
        result.append(number)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return tuple(result)


def _multi_vector(value: object, name: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of vectors")
    rows = tuple(_vector(row, f"{name} row") for row in value)
    if not rows:
        raise ValueError(f"{name} must not be empty")
    if len({len(row) for row in rows}) != 1:
        raise ValueError(f"{name} rows must have consistent dimensions")
    return rows


def _normalized_candidates(
    results: object,
    collection_type: str,
) -> list[dict[str, Any]]:
    if not isinstance(results, list):
        raise TypeError("hybrid_search must return a list")
    normalized: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, SearchResult):
            raise TypeError("hybrid_search results must be SearchResult values")
        properties = result.properties
        if not isinstance(properties, Mapping):
            raise TypeError("search result properties must be a mapping")
        copied_properties = dict(properties)
        raw_text = _required_text(copied_properties.get("raw_text"), "raw_text")
        retrieval_text = (
            _required_text(copied_properties.get("segment_text"), "segment_text")
            if collection_type == "conversations"
            else raw_text
        )
        score = float(result.score)
        if not math.isfinite(score):
            raise ValueError("hybrid score must be finite")
        normalized.append(
            {
                "object_id": result.object_id,
                "properties": copied_properties,
                "raw_text": raw_text,
                "retrieval_text": retrieval_text,
                "hybrid_score": score,
                "diversity_vector": _vector(result.vector, "diversity vector"),
            }
        )
    return normalized


def _cross_encoder(
    candidates: list[dict[str, Any]],
    query: str,
    config: RetrievalConfig,
    runtime: RAGRuntime,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    raw_results = runtime.reranker.rerank(
        query,
        [candidate["retrieval_text"] for candidate in candidates],
        model=RERANKER_MODEL,
        top_n=len(candidates),
    )
    if isinstance(raw_results, (str, bytes)) or not isinstance(raw_results, Sequence):
        raise TypeError("reranker must return a sequence of RerankResult values")
    if len(raw_results) != len(candidates):
        raise ValueError("reranker must return one score for every candidate")
    selected: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for item in raw_results:
        if not isinstance(item, RerankResult):
            raise TypeError("reranker results must be RerankResult values")
        if isinstance(item.index, bool) or not isinstance(item.index, int):
            raise TypeError("reranker result index must be an integer")
        if not 0 <= item.index < len(candidates):
            raise ValueError("reranker result index is out of range")
        if item.index in used_indices:
            raise ValueError("reranker result indices must be unique")
        if isinstance(item.score, bool):
            raise TypeError("reranker score must be numeric")
        score = float(item.score)
        if not math.isfinite(score):
            raise ValueError("reranker score must be finite")
        used_indices.add(item.index)
        candidate = dict(candidates[item.index])
        candidate["rerank_score"] = score
        selected.append(candidate)
    return sorted(
        selected,
        key=lambda candidate: (
            -candidate["rerank_score"],
            str(candidate["object_id"]),
        ),
    )


def _conversation_id(candidate: Mapping[str, Any]) -> str:
    properties = candidate.get("properties")
    if not isinstance(properties, Mapping):
        raise TypeError("Conversation candidate properties must be a mapping")
    value = properties.get("conversation_id")
    return _required_text(value, "conversation_id")


def _segment_index(candidate: Mapping[str, Any]) -> int:
    properties = candidate.get("properties")
    if not isinstance(properties, Mapping):
        raise TypeError("Conversation candidate properties must be a mapping")
    value = properties.get("segment_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("segment_index must be a non-negative integer")
    return value


def _collapse_conversations(
    candidates: Sequence[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    canonical: dict[str, tuple[str, tuple[float, ...]]] = {}
    for candidate in candidates:
        conversation_id = _conversation_id(candidate)
        identity = (candidate["raw_text"], candidate["diversity_vector"])
        if conversation_id in canonical and canonical[conversation_id] != identity:
            raise ValueError(
                "Conversation segments disagree on canonical text or GTE vector"
            )
        canonical[conversation_id] = identity
        current = selected.get(conversation_id)
        if current is None or (
            -candidate["rerank_score"],
            _segment_index(candidate),
            str(candidate["object_id"]),
        ) < (
            -current["rerank_score"],
            _segment_index(current),
            str(current["object_id"]),
        ):
            selected[conversation_id] = candidate
    collapsed: list[dict[str, Any]] = []
    for conversation_id, candidate in selected.items():
        copied = dict(candidate)
        copied["segment_id"] = copied["object_id"]
        copied["object_id"] = conversation_id
        collapsed.append(copied)
    collapsed.sort(key=lambda item: (-item["rerank_score"], item["object_id"]))
    return collapsed[:limit]


def _normalize_scores(candidates: Sequence[Mapping[str, Any]]) -> list[float]:
    if not candidates:
        return []
    scores = [float(candidate["rerank_score"]) for candidate in candidates]
    low = min(scores)
    high = max(scores)
    if high == low:
        return [1.0] * len(scores)
    scale = high - low
    return [(score - low) / scale for score in scores]


def _sigmoid_score(score: float) -> float:
    """Map one raw BGE logit to an absolute, window-independent scale."""

    if score >= 0.0:
        return 1.0 / (1.0 + math.exp(-score))
    exponent = math.exp(score)
    return exponent / (1.0 + exponent)


def _adaptive_k(
    eligible: Sequence[Mapping[str, Any]],
    config: RetrievalConfig,
) -> int:
    decision = list(eligible[: config.final_count])
    if not decision:
        return 0
    normalized = [
        _sigmoid_score(float(candidate["rerank_score"]))
        for candidate in decision
    ]
    gaps = [
        normalized[index] - normalized[index + 1]
        for index in range(len(normalized) - 1)
    ]
    if not gaps:
        return 1
    largest = max(gaps)
    if largest < config.adaptive_gap_threshold:
        return len(decision)
    return gaps.index(largest) + 1


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("MMR vectors must be non-empty and have equal dimensions")
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("MMR vectors must have positive norms")
    cosine = sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return min(1.0, max(0.0, cosine))


def _mmr(
    eligible: Sequence[dict[str, Any]],
    k: int,
    config: RetrievalConfig,
) -> list[dict[str, Any]]:
    if k == 0:
        return []
    head = list(eligible[: min(2 * k, len(eligible))])
    normalized = _normalize_scores(head)
    remaining = list(range(len(head)))
    selected_indices: list[int] = []
    while remaining and len(selected_indices) < k:
        best_index = remaining[0]
        best_objective = -math.inf
        for index in remaining:
            redundancy = (
                max(
                    _cosine_similarity(
                        head[index]["diversity_vector"],
                        head[chosen]["diversity_vector"],
                    )
                    for chosen in selected_indices
                )
                if selected_indices
                else 0.0
            )
            objective = (
                config.mmr_lambda * normalized[index]
                - (1.0 - config.mmr_lambda) * redundancy
            )
            if objective > best_objective:
                best_index = index
                best_objective = objective
        selected_indices.append(best_index)
        remaining.remove(best_index)
    results: list[dict[str, Any]] = []
    for index in selected_indices:
        copied = dict(head[index])
        copied["normalized_rerank_score"] = normalized[index]
        results.append(copied)
    return results


def _hybrid_candidates(
    hybrid_search: Any,
    query: str,
    vector: tuple[tuple[float, ...], ...],
    collection_type: str,
    config: RetrievalConfig,
) -> list[dict[str, Any]]:
    limits = (
        tuple(config.candidate_count * item for item in _CONVERSATION_OVERFETCH_MULTIPLIERS)
        if collection_type == "conversations"
        else (config.candidate_count,)
    )
    accumulated: dict[str, dict[str, Any]] = {}
    for limit in limits:
        page = _normalized_candidates(
            hybrid_search(query, [list(row) for row in vector], limit),
            collection_type,
        )
        for candidate in page:
            accumulated.setdefault(str(candidate["object_id"]), candidate)
        if collection_type != "conversations":
            break
        unique_conversations = {
            _conversation_id(candidate) for candidate in accumulated.values()
        }
        if len(unique_conversations) >= config.candidate_count or len(page) < limit:
            break
    return list(accumulated.values())


def _budget_results(
    results: Sequence[Mapping[str, Any]],
    budget: int,
    *,
    tokenizer: Tokenizer | None = None,
) -> list[dict[str, Any]]:
    active_tokenizer = tokenizer or tiktoken.get_encoding(
        TEXT_PROCESSING.tokenizer_encoding
    )
    selected: list[dict[str, Any]] = []
    for result in results:
        raw_text = _required_text(result.get("raw_text"), "raw_text")
        copied = dict(result)
        copied["raw_text"] = raw_text
        selected.append(copied)

    while selected:
        context = _CONTEXT_SEPARATOR.join(item["raw_text"] for item in selected)
        if len(active_tokenizer.encode(context)) <= budget:
            break
        selected.pop()
    return selected


def _render_context(
    results: Sequence[Mapping[str, Any]],
    budget: int,
    *,
    tokenizer: Tokenizer | None = None,
) -> str:
    selected = _budget_results(results, budget, tokenizer=tokenizer)
    return _CONTEXT_SEPARATOR.join(result["raw_text"] for result in selected)


def retrieve(
    collection_client: object,
    query_text: str,
    query_vector: Sequence[Sequence[float]],
    collection_type: str,
    *,
    runtime: RAGRuntime | None = None,
    timing_observer: TimingObserver | None = None,
) -> list[dict[str, Any]]:
    """Run the unified hybrid, BGE, Adaptive-K, MMR, and budget flow."""

    query = _required_text(query_text, "query_text")
    vector = _multi_vector(query_vector, "query_vector")
    if collection_type not in _COLLECTION_CONFIGS:
        raise ValueError(f"unsupported collection_type {collection_type!r}")
    hybrid_search = getattr(collection_client, "hybrid_search", None)
    if not callable(hybrid_search):
        raise TypeError("collection_client must provide hybrid_search()")
    config = _COLLECTION_CONFIGS[collection_type]
    search_started = perf_counter()
    candidates = _hybrid_candidates(
        hybrid_search,
        query,
        vector,
        collection_type,
        config,
    )
    search_elapsed = (perf_counter() - search_started) * 1000.0
    if timing_observer is not None:
        timing_observer(_HYBRID_TIMING_PHASES[collection_type], search_elapsed)
    active_runtime = resolve_runtime(runtime)
    rerank_started = perf_counter()
    reranked = _cross_encoder(candidates, query, config, active_runtime)
    rerank_elapsed = (perf_counter() - rerank_started) * 1000.0
    if timing_observer is not None and collection_type in _RERANK_TIMING_PHASES:
        timing_observer(_RERANK_TIMING_PHASES[collection_type], rerank_elapsed)

    if collection_type == "conversations":
        reranked = _collapse_conversations(reranked, config.candidate_count)
    eligible = [
        candidate
        for candidate in reranked
        if candidate["rerank_score"] >= config.adaptive_relevance_floor
    ]
    k = _adaptive_k(eligible, config)
    mmr_started = perf_counter()
    selected = _mmr(eligible, k, config)
    mmr_elapsed = (perf_counter() - mmr_started) * 1000.0
    if timing_observer is not None and collection_type == "conversations":
        timing_observer("conversation_mmr_rerank", mmr_elapsed)
    _LOGGER.debug(
        "Unified retrieval selected adaptive context",
        extra={
            "collection_type": collection_type,
            "hybrid_candidates": len(candidates),
            "eligible_candidates": len(eligible),
            "adaptive_k": k,
            "mmr_head": min(2 * k, len(eligible)),
        },
    )
    if collection_type in _CONTEXT_BUDGETS:
        selected = _budget_results(
            selected,
            _CONTEXT_BUDGETS[collection_type],
            tokenizer=active_runtime.tokenizer,
        )
    return selected


__all__ = ["retrieve"]
