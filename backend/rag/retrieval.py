"""Hybrid candidate retrieval, MMR/cross-encoder reranking, and budgeting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
    ONNX_RERANKER,
    POLICY_SEARCH,
    RERANKER_MODEL,
    RERANKER_MODEL_REVISION,
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
from backend.wizard.diagnostics import (
    TRACE_SAMPLE_LIMIT,
    add_count,
    capture_retrieval_candidate_pages,
    framed_content_digest,
    observe_elapsed,
    observe_stage,
    observe_trace_metadata,
    set_flag,
    set_framed_digest,
    set_sample,
    trace_operation_active,
)


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
_TRACE_COLLECTION_PREFIXES = {
    "knowledge_facts": "knowledge",
    "policy": "policy",
}
_LOGGER = logging.getLogger(__name__)
_POLICY_EMPTY_POOL_FLOOR = -3.0


def _eligible_candidates(
    reranked: Sequence[dict[str, Any]],
    config: RetrievalConfig,
    *,
    collection_type: str,
) -> list[dict[str, Any]]:
    eligible = [
        candidate
        for candidate in reranked
        if candidate["rerank_score"] >= config.adaptive_relevance_floor
    ]
    # Rescue only a bounded, rank-strong Policy result; never expand a
    # nonempty pool or change the already-established BGE/UUID ordering.
    if (
        not eligible
        and collection_type == "policy"
        and reranked
        and reranked[0]["rerank_score"] >= _POLICY_EMPTY_POOL_FLOOR
    ):
        return [reranked[0]]
    return eligible


def _context_item_fingerprints(
    items: Sequence[Mapping[str, Any]],
) -> Iterable[str]:
    if not trace_operation_active():
        return ()

    def fingerprints() -> Iterable[str]:
        for item in items:
            object_id = str(item["object_id"])
            digest = framed_content_digest(
                "chat-kp-item-v1",
                (
                    object_id.encode("utf-8"),
                    str(item["raw_text"]).encode("utf-8"),
                ),
            )
            yield f"{object_id}:{digest}"

    return fingerprints()


def _observe_candidate_decisions(
    *,
    prefix: str,
    hybrid_candidates: Sequence[SearchResult],
    reranked: Sequence[Mapping[str, Any]],
    eligible: Sequence[Mapping[str, Any]],
    adaptive_pool: Sequence[Mapping[str, Any]],
    hydrated_pool: Sequence[Mapping[str, Any]],
    mmr_selected: Sequence[Mapping[str, Any]],
    final_selected: Sequence[Mapping[str, Any]],
    mmr_usable: bool,
    config: RetrievalConfig,
) -> None:
    """Derive bounded evidence only inside ``observe_trace_metadata``."""

    ceiling = 50 if prefix == "knowledge" else 40
    if len(hybrid_candidates) > ceiling or len(reranked) > ceiling:
        raise ValueError("retrieval candidate observation exceeds its ceiling")
    hybrid_ranks = {
        candidate.object_id: index
        for index, candidate in enumerate(hybrid_candidates, start=1)
    }
    eligible_ids = {str(candidate["object_id"]) for candidate in eligible}
    adaptive_ids = {str(candidate["object_id"]) for candidate in adaptive_pool}
    hydrated_ids = {str(candidate["object_id"]) for candidate in hydrated_pool}
    mmr_ids = {str(candidate["object_id"]) for candidate in mmr_selected}
    final_ids = {str(candidate["object_id"]) for candidate in final_selected}
    decisions: list[str] = []
    text_digests: list[str] = []
    collection_code = "k" if prefix == "knowledge" else "p"
    for bge_rank, candidate in enumerate(reranked, start=1):
        object_id = str(candidate["object_id"])
        score = float(candidate["rerank_score"])
        floor_pass = object_id in eligible_ids
        effective_floor = config.adaptive_relevance_floor
        if prefix == "policy" and floor_pass and score < effective_floor:
            effective_floor = _POLICY_EMPTY_POOL_FLOOR
        adaptive_inclusion = object_id in adaptive_ids
        if not adaptive_inclusion:
            hydration_outcome = "n"
        elif object_id in hydrated_ids:
            hydration_outcome = "y"
        else:
            hydration_outcome = "f"
        mmr_inclusion = object_id in mmr_ids
        final_inclusion = object_id in final_ids
        if not floor_pass:
            reason = "floor"
        elif not adaptive_inclusion:
            reason = "adaptive_gap"
        elif hydration_outcome == "f":
            reason = "hydration"
        elif not mmr_inclusion:
            reason = "mmr_limit" if mmr_usable else "fallback_limit"
        elif not final_inclusion:
            reason = "budget"
        else:
            reason = "selected"
        decisions.append(
            "|".join(
                (
                    object_id,
                    collection_code,
                    str(hybrid_ranks[object_id]),
                    str(bge_rank),
                    score.hex(),
                    float(effective_floor).hex(),
                    "1" if floor_pass else "0",
                    "1" if adaptive_inclusion else "0",
                    hydration_outcome,
                    "1" if mmr_inclusion else "0",
                    "1" if final_inclusion else "0",
                    reason,
                )
            )
        )
        digest = framed_content_digest(
            "chat-kp-candidate-text-v1",
            (str(candidate["retrieval_text"]).encode("utf-8"),),
        )
        text_digests.append(f"{object_id}:{digest}")

    selected_boundary = (
        float(adaptive_pool[-1]["rerank_score"]).hex()
        if adaptive_pool
        else "none"
    )
    first_excluded = (
        float(eligible[len(adaptive_pool)]["rerank_score"]).hex()
        if len(adaptive_pool) < len(eligible)
        else "none"
    )
    capture_retrieval_candidate_pages(
        prefix,
        decisions,
        text_digests,
        adaptive_metadata=(
            f"gap_threshold_hex={float(config.adaptive_gap_threshold).hex()};"
            f"eligible_count={len(eligible)};"
            f"adaptive_count={len(adaptive_pool)};"
            f"last_selected_raw_score_hex={selected_boundary};"
            f"first_adaptive_excluded_raw_score_hex={first_excluded}"
        ),
        reranker_identity=(
            f"model={RERANKER_MODEL};revision={RERANKER_MODEL_REVISION};"
            f"onnx={ONNX_RERANKER.onnx_filename};"
            f"manifest={ONNX_RERANKER.manifest_filename};"
            f"output={ONNX_RERANKER.output_name};"
            f"max_tokens={ONNX_RERANKER.max_tokens}"
        ),
    )


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
    decision = list(eligible)
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
    pool = list(eligible)
    normalized = _normalize_scores(pool)
    remaining = list(range(len(pool)))
    selected_indices: list[int] = []
    selection_limit = min(k, len(pool))
    while remaining and len(selected_indices) < selection_limit:
        best_index = remaining[0]
        best_objective = -math.inf
        for index in remaining:
            redundancy = (
                max(
                    _cosine_similarity(
                        pool[index]["diversity_vector"],
                        pool[chosen]["diversity_vector"],
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
        results.append(pool[index])
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
    adaptive_pool: Sequence[dict[str, Any]],
    collection_type: str,
) -> tuple[list[dict[str, Any]], bool]:
    if not adaptive_pool:
        return [], True
    pool = list(adaptive_pool)
    hydrate = getattr(collection_client, "hydrate_mmr_head", None)
    if not callable(hydrate):
        raise TypeError("collection_client must provide hydrate_mmr_head()")
    requested = _hydration_requests(pool, collection_type)
    raw_hydrated = hydrate(requested)
    if not isinstance(raw_hydrated, list) or len(raw_hydrated) != len(pool):
        raise TypeError("hydrate_mmr_head must return one ordered result per candidate")
    hydrated_pool: list[dict[str, Any]] = []
    mmr_usable = True
    for candidate, request, hydrated in zip(
        pool,
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
        hydrated_pool.append(copied)
    return hydrated_pool, mmr_usable


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
    trace_prefix = _TRACE_COLLECTION_PREFIXES.get(collection_type)
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
    if collection_type == "conversations":
        observe_elapsed("chat.conversation_hybrid", search_elapsed)
        add_count("conversation_hybrid_candidate_count", len(candidates))
        add_count("conversation_hybrid_storage_call_count", 1)
        set_sample(
            "conversation_hybrid_segment_ids",
            (candidate.object_id for candidate in candidates),
            exact_count=len(candidates),
        )
        set_sample(
            "conversation_hybrid_conversation_ids",
            (candidate.canonical_id for candidate in candidates),
            exact_count=len(candidates),
        )
    elif trace_prefix is not None:
        observe_elapsed(f"chat.{trace_prefix}_hybrid", search_elapsed)
        add_count(f"{trace_prefix}_hybrid_candidate_count", len(candidates))
        add_count(f"{trace_prefix}_hybrid_storage_call_count", 1)
        set_sample(
            f"{trace_prefix}_hybrid_ids",
            (candidate.object_id for candidate in candidates),
            exact_count=len(candidates),
        )
    active_runtime = resolve_runtime(runtime)
    rerank_started = perf_counter()
    reranked = _cross_encoder(candidates, query, config, active_runtime)
    rerank_elapsed = (perf_counter() - rerank_started) * 1000.0
    if timing_observer is not None and collection_type in _RERANK_TIMING_PHASES:
        timing_observer(_RERANK_TIMING_PHASES[collection_type], rerank_elapsed)

    if collection_type == "conversations":
        observe_elapsed("chat.conversation_bge", rerank_elapsed)
        add_count("conversation_bge_input_count", len(candidates))
        add_count("conversation_bge_result_count", len(reranked))
        add_count("conversation_bge_model_call_count", 1 if candidates else 0)
        set_flag("conversation_bge_scores_finite", True)
        set_sample(
            "conversation_bge_segment_ids",
            (candidate["object_id"] for candidate in reranked),
            exact_count=len(reranked),
        )
        reranked_segment_count = len(reranked)
        with observe_stage("chat.conversation_collapse"):
            reranked = _collapse_conversations(reranked, config.candidate_count)
        add_count("conversation_collapse_input_count", reranked_segment_count)
        add_count("conversation_collapse_result_count", len(reranked))
        set_sample(
            "conversation_collapsed_ids",
            (candidate["object_id"] for candidate in reranked),
            exact_count=len(reranked),
        )
        with observe_stage("chat.conversation_relevance_floor"):
            eligible = [
                candidate
                for candidate in reranked
                if candidate["rerank_score"] >= config.adaptive_relevance_floor
            ]
        add_count("conversation_relevance_input_count", len(reranked))
        add_count("conversation_relevance_eligible_count", len(eligible))
        set_sample(
            "conversation_eligible_ids",
            (candidate["object_id"] for candidate in eligible),
            exact_count=len(eligible),
        )
        with observe_stage("chat.conversation_adaptive_k"):
            adaptive_count = _adaptive_k(eligible, config)
    else:
        if trace_prefix is not None:
            observe_elapsed(f"chat.{trace_prefix}_bge", rerank_elapsed)
            add_count(f"{trace_prefix}_bge_input_count", len(candidates))
            add_count(f"{trace_prefix}_bge_result_count", len(reranked))
            add_count(
                f"{trace_prefix}_bge_model_call_count",
                1 if candidates else 0,
            )
            set_flag(f"{trace_prefix}_bge_scores_finite", True)
            set_sample(
                f"{trace_prefix}_bge_ids",
                (candidate["object_id"] for candidate in reranked),
                exact_count=len(reranked),
            )
            with observe_stage(f"chat.{trace_prefix}_relevance_floor"):
                eligible = _eligible_candidates(
                    reranked, config, collection_type=collection_type
                )
            add_count(f"{trace_prefix}_relevance_input_count", len(reranked))
            add_count(f"{trace_prefix}_relevance_eligible_count", len(eligible))
            set_sample(
                f"{trace_prefix}_eligible_ids",
                (candidate["object_id"] for candidate in eligible),
                exact_count=len(eligible),
            )
            with observe_stage(f"chat.{trace_prefix}_adaptive_k"):
                adaptive_count = _adaptive_k(eligible, config)
        else:  # pragma: no cover - supported collection types are exhaustive.
            eligible = [
                candidate
                for candidate in reranked
                if candidate["rerank_score"] >= config.adaptive_relevance_floor
            ]
            adaptive_count = _adaptive_k(eligible, config)
    adaptive_pool = eligible[:adaptive_count]
    if collection_type == "conversations":
        add_count("conversation_adaptive_eligible_count", len(eligible))
        add_count("conversation_adaptive_selected_count", adaptive_count)
        set_sample(
            "conversation_adaptive_pool_ids",
            (candidate["object_id"] for candidate in adaptive_pool),
            exact_count=len(adaptive_pool),
        )
        with observe_stage("chat.conversation_hydration"):
            hydrated_pool, mmr_usable = _hydrate_mmr_head(
                collection_client,
                adaptive_pool,
                collection_type,
            )
        add_count("conversation_hydration_requested_count", len(adaptive_pool))
        add_count("conversation_hydration_result_count", len(hydrated_pool))
        add_count(
            "conversation_hydration_quarantined_count",
            len(adaptive_pool) - len(hydrated_pool),
        )
        add_count(
            "conversation_hydration_storage_call_count",
            1 if adaptive_pool else 0,
        )
        set_flag("conversation_mmr_usable", mmr_usable)
        set_flag("conversation_hydration_values_finite", mmr_usable)
        set_sample(
            "conversation_hydrated_ids",
            (candidate["object_id"] for candidate in hydrated_pool),
            exact_count=len(hydrated_pool),
        )
    else:
        if trace_prefix is not None:
            add_count(f"{trace_prefix}_adaptive_eligible_count", len(eligible))
            add_count(f"{trace_prefix}_adaptive_selected_count", adaptive_count)
            set_sample(
                f"{trace_prefix}_adaptive_pool_ids",
                (candidate["object_id"] for candidate in adaptive_pool),
                exact_count=len(adaptive_pool),
            )
            with observe_stage(f"chat.{trace_prefix}_hydration"):
                hydrated_pool, mmr_usable = _hydrate_mmr_head(
                    collection_client,
                    adaptive_pool,
                    collection_type,
                )
            add_count(
                f"{trace_prefix}_hydration_requested_count",
                len(adaptive_pool),
            )
            add_count(
                f"{trace_prefix}_hydration_result_count",
                len(hydrated_pool),
            )
            add_count(
                f"{trace_prefix}_hydration_quarantined_count",
                len(adaptive_pool) - len(hydrated_pool),
            )
            add_count(
                f"{trace_prefix}_hydration_storage_call_count",
                1 if adaptive_pool else 0,
            )
            set_flag(f"{trace_prefix}_mmr_usable", mmr_usable)
            set_flag(f"{trace_prefix}_hydration_values_finite", mmr_usable)
            set_sample(
                f"{trace_prefix}_hydrated_ids",
                (candidate["object_id"] for candidate in hydrated_pool),
                exact_count=len(hydrated_pool),
            )
        else:  # pragma: no cover - supported collection types are exhaustive.
            hydrated_pool, mmr_usable = _hydrate_mmr_head(
                collection_client,
                adaptive_pool,
                collection_type,
            )
    mmr_started = perf_counter()
    selected = (
        _mmr(hydrated_pool, config.final_count, config)
        if mmr_usable
        else hydrated_pool[: config.final_count]
    )
    mmr_selected = selected
    mmr_elapsed = (perf_counter() - mmr_started) * 1000.0
    if timing_observer is not None and collection_type == "conversations":
        timing_observer("conversation_mmr_rerank", mmr_elapsed)
    if collection_type == "conversations":
        observe_elapsed("chat.conversation_mmr", mmr_elapsed)
        add_count("conversation_mmr_pool_count", len(hydrated_pool))
        add_count("conversation_mmr_selected_count", len(selected))
        set_flag("conversation_mmr_fallback", not mmr_usable)
        set_sample(
            "conversation_mmr_selected_ids",
            (candidate["object_id"] for candidate in selected),
            exact_count=len(selected),
        )
    elif trace_prefix is not None:
        observe_elapsed(f"chat.{trace_prefix}_mmr", mmr_elapsed)
        add_count(f"{trace_prefix}_mmr_pool_count", len(hydrated_pool))
        add_count(f"{trace_prefix}_mmr_selected_count", len(selected))
        set_flag(f"{trace_prefix}_mmr_fallback", not mmr_usable)
        set_sample(
            f"{trace_prefix}_mmr_selected_ids",
            (candidate["object_id"] for candidate in selected),
            exact_count=len(selected),
        )
    _LOGGER.debug(
        "Unified retrieval selected adaptive context",
        extra={
            "collection_type": collection_type,
            "hybrid_candidates": len(candidates),
            "eligible_candidates": len(eligible),
            "adaptive_k": adaptive_count,
            "mmr_pool": len(hydrated_pool),
            "mmr_fallback": not mmr_usable,
        },
    )
    if collection_type == "conversations":
        with observe_stage("chat.conversation_finalize"):
            selected = _final_results(selected, collection_type)
        add_count("conversation_final_count", len(selected))
        set_sample(
            "conversation_final_ids",
            (item["object_id"] for item in selected),
            exact_count=len(selected),
        )
        set_framed_digest(
            "conversation_final_context_sha256",
            "chat-conversation-final-v1",
            (
                value
                for item in selected
                for value in (
                    item["object_id"].encode("utf-8"),
                    item["raw_text"].encode("utf-8"),
                )
            ),
        )
    else:
        if trace_prefix is not None:
            with observe_stage(f"chat.{trace_prefix}_finalize"):
                selected = _final_results(selected, collection_type)
        else:  # pragma: no cover - supported collection types are exhaustive.
            selected = _final_results(selected, collection_type)
    if collection_type in _CONTEXT_BUDGETS:
        prebudget_count = len(selected)
        if trace_prefix is not None:
            with observe_stage(f"chat.{trace_prefix}_budgeting"):
                selected = _budget_results(
                    selected,
                    _CONTEXT_BUDGETS[collection_type],
                    tokenizer=active_runtime.tokenizer,
                )
        else:  # pragma: no cover - supported collection types are exhaustive.
            selected = _budget_results(
                selected,
                _CONTEXT_BUDGETS[collection_type],
                tokenizer=active_runtime.tokenizer,
            )
        if trace_prefix is not None:
            add_count(f"{trace_prefix}_prebudget_count", prebudget_count)
            add_count(f"{trace_prefix}_final_count", len(selected))
            add_count(
                f"{trace_prefix}_budget_dropped_count",
                prebudget_count - len(selected),
            )
            set_sample(
                f"{trace_prefix}_final_ids",
                (item["object_id"] for item in selected),
                exact_count=len(selected),
            )
            set_sample(
                f"{trace_prefix}_final_item_fingerprints",
                _context_item_fingerprints(selected),
                exact_count=len(selected),
            )
            set_flag(
                f"{trace_prefix}_final_proof_truncated",
                len(selected) > TRACE_SAMPLE_LIMIT,
            )
            set_framed_digest(
                f"{trace_prefix}_final_context_sha256",
                f"chat-{trace_prefix}-final-v1",
                (
                    value
                    for item in selected
                    for value in (
                        item["object_id"].encode("utf-8"),
                        item["raw_text"].encode("utf-8"),
                    )
                ),
            )
    if trace_prefix is not None:
        observe_trace_metadata(
            _observe_candidate_decisions,
            prefix=trace_prefix,
            hybrid_candidates=candidates,
            reranked=reranked,
            eligible=eligible,
            adaptive_pool=adaptive_pool,
            hydrated_pool=hydrated_pool,
            mmr_selected=mmr_selected,
            final_selected=selected,
            mmr_usable=mmr_usable,
            config=config,
        )
    return selected


__all__ = ["retrieve"]
