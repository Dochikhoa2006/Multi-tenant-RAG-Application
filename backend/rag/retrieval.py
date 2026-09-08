"""Hybrid candidate retrieval, MMR/cross-encoder reranking, and budgeting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
import math
from time import perf_counter
from typing import Any
from uuid import UUID, uuid5

import tiktoken

from backend.model_config import (
    CONVERSATION_SEARCH,
    KNOWLEDGE_SEARCH,
    MMR_DIVERSITY_VECTOR_DIMENSION,
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
from backend.weaviate_client.models import HydratedSearchResult, SearchResult


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
) -> list[SearchResult]:
    if not isinstance(results, list):
        raise TypeError("hybrid_search must return a list")
    normalized: list[SearchResult] = []
    seen_object_ids: set[str] = set()
    quarantined_object_ids: set[str] = set()
    quarantined_conversations: set[str] = set()
    conversation_indices: dict[str, dict[int, str]] = {}
    for result in results:
        if not isinstance(result, SearchResult):
            _LOGGER.warning(
                "Quarantined malformed retrieval candidate",
                extra={
                    "collection_type": collection_type,
                    "object_id": "unavailable",
                    "reason": "invalid_result_type",
                },
            )
            continue
        object_id = result.object_id
        canonical_id = result.canonical_id
        try:
            object_id = str(UUID(_required_text(object_id, "object_id")))
            canonical_id = str(UUID(_required_text(canonical_id, "canonical_id")))
            _required_text(result.retrieval_text, "retrieval_text")
        except (TypeError, ValueError):
            try:
                safe_id = str(UUID(str(result.object_id)))
            except (TypeError, ValueError, AttributeError):
                safe_id = "unavailable"
            try:
                recoverable_conversation = str(UUID(str(result.canonical_id)))
            except (TypeError, ValueError, AttributeError):
                recoverable_conversation = None
            if collection_type == "conversations" and recoverable_conversation:
                quarantined_conversations.add(recoverable_conversation)
            _LOGGER.warning(
                "Quarantined malformed retrieval candidate",
                extra={
                    "collection_type": collection_type,
                    "object_id": safe_id,
                    "reason": "invalid_candidate_fields",
                },
            )
            continue
        if object_id in seen_object_ids:
            quarantined_object_ids.add(object_id)
            if collection_type == "conversations":
                quarantined_conversations.add(canonical_id)
                quarantined_conversations.update(
                    item.canonical_id
                    for item in normalized
                    if item.object_id == object_id
                )
            _LOGGER.warning(
                "Quarantined malformed retrieval candidate",
                extra={
                    "collection_type": collection_type,
                    "object_id": object_id,
                    "reason": "duplicate_object_uuid",
                },
            )
            continue
        seen_object_ids.add(object_id)
        if collection_type != "conversations" and canonical_id != object_id:
            quarantined_object_ids.add(object_id)
            _LOGGER.warning(
                "Quarantined malformed retrieval candidate",
                extra={
                    "collection_type": collection_type,
                    "object_id": object_id,
                    "reason": "canonical_uuid_mismatch",
                },
            )
            continue
        if collection_type == "conversations":
            segment_index = result.segment_index
            if (
                isinstance(segment_index, bool)
                or not isinstance(segment_index, int)
                or segment_index < 0
                or object_id
                != str(
                    uuid5(
                        UUID(canonical_id),
                        f"retrieval-segment:{segment_index}",
                    )
                )
            ):
                quarantined_conversations.add(canonical_id)
                _LOGGER.warning(
                    "Quarantined malformed Conversation group",
                    extra={
                        "collection_type": collection_type,
                        "object_id": object_id,
                        "conversation_id": canonical_id,
                        "reason": "invalid_segment_identity",
                    },
                )
                continue
            indices = conversation_indices.setdefault(canonical_id, {})
            existing = indices.get(segment_index)
            if existing is not None and existing != object_id:
                quarantined_conversations.add(canonical_id)
                _LOGGER.warning(
                    "Quarantined malformed Conversation group",
                    extra={
                        "collection_type": collection_type,
                        "object_id": object_id,
                        "conversation_id": canonical_id,
                        "reason": "conflicting_segment_index",
                    },
                )
                continue
            indices[segment_index] = object_id
        normalized.append(
            SearchResult(
                object_id=object_id,
                canonical_id=canonical_id,
                retrieval_text=result.retrieval_text,
                segment_index=result.segment_index,
            )
        )
    return [
        item
        for item in normalized
        if item.object_id not in quarantined_object_ids
        and item.canonical_id not in quarantined_conversations
    ]


def _cross_encoder(
    candidates: list[SearchResult],
    query: str,
    config: RetrievalConfig,
    runtime: RAGRuntime,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    raw_results = runtime.reranker.rerank(
        query,
        [candidate.retrieval_text for candidate in candidates],
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
        candidate = candidates[item.index]
        selected.append(
            {
                "object_id": candidate.object_id,
                "canonical_id": candidate.canonical_id,
                "retrieval_text": candidate.retrieval_text,
                "segment_index": candidate.segment_index,
                "rerank_score": score,
            }
        )
    return sorted(
        selected,
        key=lambda candidate: (
            -candidate["rerank_score"],
            str(candidate["object_id"]),
        ),
    )


def _conversation_id(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("canonical_id")
    return _required_text(value, "conversation_id")


def _segment_index(candidate: Mapping[str, Any]) -> int:
    value = candidate.get("segment_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("segment_index must be a non-negative integer")
    return value


def _collapse_conversations(
    candidates: Sequence[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        conversation_id = _conversation_id(candidate)
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
        results.append(head[index])
    return results


def _hybrid_candidates(
    hybrid_search: Any,
    query: str,
    vector: tuple[tuple[float, ...], ...],
    config: RetrievalConfig,
    collection_type: str,
) -> list[SearchResult]:
    return _normalized_candidates(
        hybrid_search(
            query,
            [list(row) for row in vector],
            config.candidate_count,
        ),
        collection_type,
    )


def _hydration_requests(
    head: Sequence[Mapping[str, Any]],
    collection_type: str,
) -> list[SearchResult]:
    requests: list[SearchResult] = []
    for candidate in head:
        object_id = (
            _required_text(candidate.get("segment_id"), "segment_id")
            if collection_type == "conversations"
            else _required_text(candidate.get("object_id"), "object_id")
        )
        canonical_id = (
            _required_text(candidate.get("object_id"), "conversation_id")
            if collection_type == "conversations"
            else object_id
        )
        requests.append(
            SearchResult(
                object_id=object_id,
                canonical_id=canonical_id,
                retrieval_text=_required_text(
                    candidate.get("retrieval_text"),
                    "retrieval_text",
                ),
                segment_index=(
                    _segment_index(candidate)
                    if collection_type == "conversations"
                    else None
                ),
            )
        )
    return requests


def _hydrate_mmr_head(
    collection_client: object,
    eligible: Sequence[dict[str, Any]],
    k: int,
    collection_type: str,
) -> tuple[list[dict[str, Any]], bool]:
    if k == 0:
        return [], True
    head = list(eligible[: min(2 * k, len(eligible))])
    hydrate = getattr(collection_client, "hydrate_mmr_head", None)
    if not callable(hydrate):
        raise TypeError("collection_client must provide hydrate_mmr_head()")
    requested = _hydration_requests(head, collection_type)
    raw_hydrated = hydrate(requested)
    if not isinstance(raw_hydrated, list) or len(raw_hydrated) != len(head):
        raise TypeError("hydrate_mmr_head must return one ordered result per candidate")
    hydrated_head: list[dict[str, Any]] = []
    mmr_usable = True
    for candidate, request, hydrated in zip(
        head,
        requested,
        raw_hydrated,
        strict=True,
    ):
        if not isinstance(hydrated, HydratedSearchResult):
            _LOGGER.warning(
                "Quarantined malformed hydrated candidate",
                extra={
                    "collection_type": collection_type,
                    "object_id": request.object_id,
                    "reason": "invalid_hydration_result_type",
                },
            )
            continue
        if (
            hydrated.object_id != request.object_id
            or hydrated.canonical_id != request.canonical_id
            or hydrated.segment_index != request.segment_index
        ):
            _LOGGER.warning(
                "Quarantined malformed hydrated candidate",
                extra={
                    "collection_type": collection_type,
                    "object_id": request.object_id,
                    "reason": "hydration_identity_mismatch",
                },
            )
            continue
        if hydrated.quarantine_reason is not None:
            _LOGGER.warning(
                "Quarantined malformed hydrated candidate",
                extra={
                    "collection_type": collection_type,
                    "object_id": request.object_id,
                    "reason": hydrated.quarantine_reason,
                },
            )
            continue
        copied = dict(candidate)
        if collection_type == "conversations":
            try:
                raw_text = _required_text(hydrated.raw_text, "raw_text")
            except (TypeError, ValueError):
                _LOGGER.warning(
                    "Quarantined malformed Conversation group",
                    extra={
                        "collection_type": collection_type,
                        "object_id": request.object_id,
                        "conversation_id": request.canonical_id,
                        "reason": "malformed_canonical_text",
                    },
                )
                continue
            if request.retrieval_text not in raw_text:
                _LOGGER.warning(
                    "Quarantined malformed Conversation group",
                    extra={
                        "collection_type": collection_type,
                        "object_id": request.object_id,
                        "conversation_id": request.canonical_id,
                        "reason": "segment_not_in_canonical_text",
                    },
                )
                continue
            copied["raw_text"] = raw_text
        else:
            copied["raw_text"] = copied["retrieval_text"]
        try:
            vector = _vector(hydrated.diversity_vector, "diversity vector")
            if len(vector) != MMR_DIVERSITY_VECTOR_DIMENSION:
                raise ValueError("diversity vector has the wrong dimension")
            if math.sqrt(sum(item * item for item in vector)) <= 0.0:
                raise ValueError("diversity vector has a non-positive norm")
        except (TypeError, ValueError):
            mmr_usable = False
            _LOGGER.warning(
                "Using BGE order because a stored MMR vector is unusable",
                extra={
                    "collection_type": collection_type,
                    "object_id": request.object_id,
                    "reason": "unusable_mmr_vector",
                },
            )
        else:
            copied["diversity_vector"] = vector
        hydrated_head.append(copied)
    return hydrated_head, mmr_usable


def _final_results(
    selected: Sequence[Mapping[str, Any]],
    collection_type: str,
) -> list[dict[str, Any]]:
    if collection_type == "conversations":
        return [
            {
                "object_id": _required_text(item.get("object_id"), "object_id"),
                "raw_text": _required_text(item.get("raw_text"), "raw_text"),
            }
            for item in selected
        ]
    return [
        {
            "object_id": _required_text(item.get("object_id"), "object_id"),
            "raw_text": _required_text(item.get("raw_text"), "raw_text"),
            "rerank_score": float(item["rerank_score"]),
        }
        for item in selected
    ]


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
    if not callable(getattr(collection_client, "hydrate_mmr_head", None)):
        raise TypeError("collection_client must provide hydrate_mmr_head()")
    config = _COLLECTION_CONFIGS[collection_type]
    search_started = perf_counter()
    candidates = _hybrid_candidates(
        hybrid_search,
        query,
        vector,
        config,
        collection_type,
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
    hydrated_head, mmr_usable = _hydrate_mmr_head(
        collection_client,
        eligible,
        k,
        collection_type,
    )
    mmr_started = perf_counter()
    selected = (
        _mmr(hydrated_head, min(k, len(hydrated_head)), config)
        if mmr_usable
        else hydrated_head[:k]
    )
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
            "mmr_head": len(hydrated_head),
            "mmr_fallback": not mmr_usable,
        },
    )
    selected = _final_results(selected, collection_type)
    if collection_type in _CONTEXT_BUDGETS:
        selected = _budget_results(
            selected,
            _CONTEXT_BUDGETS[collection_type],
            tokenizer=active_runtime.tokenizer,
        )
    return selected


__all__ = ["retrieve"]
