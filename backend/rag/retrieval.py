"""Hybrid candidate retrieval, MMR/cross-encoder reranking, and budgeting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    "conversations": "conversation_mmr_rerank",
    "knowledge_facts": "knowledge_cross_encoder_rerank",
    "policy": "policy_cross_encoder_rerank",
}


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


def _normalized_candidates(results: object) -> list[dict[str, Any]]:
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
        score = float(result.score)
        if not math.isfinite(score):
            raise ValueError("hybrid score must be finite")
        normalized.append(
            {
                "object_id": result.object_id,
                "properties": copied_properties,
                "raw_text": raw_text,
                "hybrid_score": score,
                "rerank_score": score,
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
        [candidate["raw_text"] for candidate in candidates],
        model=RERANKER_MODEL,
        top_n=config.final_count,
    )
    if isinstance(raw_results, (str, bytes)) or not isinstance(raw_results, Sequence):
        raise TypeError("reranker must return a sequence of RerankResult values")
    if len(raw_results) > config.final_count:
        raise ValueError("reranker returned more than top_n results")
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
    return selected


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
    query_vector: Sequence[float],
    collection_type: str,
    *,
    runtime: RAGRuntime | None = None,
    timing_observer: TimingObserver | None = None,
) -> list[dict[str, Any]]:
    """Retrieve configured candidates and apply the collection's Stage 2 strategy."""

    query = _required_text(query_text, "query_text")
    vector = _vector(query_vector, "query_vector")
    if collection_type not in _COLLECTION_CONFIGS:
        raise ValueError(f"unsupported collection_type {collection_type!r}")
    hybrid_search = getattr(collection_client, "hybrid_search", None)
    if not callable(hybrid_search):
        raise TypeError("collection_client must provide hybrid_search()")
    config = _COLLECTION_CONFIGS[collection_type]
    search_started = perf_counter()
    raw_candidates = hybrid_search(query, list(vector), config.candidate_count)
    search_elapsed = (perf_counter() - search_started) * 1000.0
    if timing_observer is not None:
        timing_observer(_HYBRID_TIMING_PHASES[collection_type], search_elapsed)
    candidates = _normalized_candidates(raw_candidates)
    if collection_type == "conversations":
        if len(candidates) > config.final_count:
            raise ValueError("native conversation MMR returned too many results")
        reranked = candidates
        if timing_observer is not None:
            timing_observer(_RERANK_TIMING_PHASES[collection_type], 0.0)
    else:
        rerank_started = perf_counter()
        active_runtime = resolve_runtime(runtime)
        reranked = _cross_encoder(candidates, query, config, active_runtime)
        reranked = _budget_results(
            reranked,
            _CONTEXT_BUDGETS[collection_type],
            tokenizer=active_runtime.tokenizer,
        )
        if timing_observer is not None:
            timing_observer(
                _RERANK_TIMING_PHASES[collection_type],
                (perf_counter() - rerank_started) * 1000.0,
            )
    return reranked


__all__ = ["retrieve"]
