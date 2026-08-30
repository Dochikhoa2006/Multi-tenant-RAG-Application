"""Grounded answer prompt composition and provider-neutral token streaming."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from functools import partial
import inspect
import math
from time import perf_counter
from typing import Any

import tiktoken

from backend.model_config import PRIMARY_GENERATOR, TEXT_PROCESSING, TOKEN_BUDGETS
from backend.prompts import ANSWER_GENERATION_PROMPT
from backend.rag.retrieval import _budget_results
from backend.rag.runtime import RAGRuntime, TimingObserver, Tokenizer, resolve_runtime


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _validated_context(
    value: object,
    name: str,
) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of mappings")
    results: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"{name} must contain mappings")
        copied = dict(item)
        copied["raw_text"] = _required_text(copied.get("raw_text"), "raw_text")
        results.append(copied)
    return results


def _render_context_pair(
    knowledge_facts: object,
    policy_guidelines: object,
    *,
    tokenizer: Tokenizer | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active_tokenizer = tokenizer or tiktoken.get_encoding(
        TEXT_PROCESSING.tokenizer_encoding
    )
    knowledge = _budget_results(
        _validated_context(knowledge_facts, "knowledge_facts"),
        TOKEN_BUDGETS.knowledge_tokens,
        tokenizer=active_tokenizer,
    )
    policy = _budget_results(
        _validated_context(policy_guidelines, "policy_guidelines"),
        TOKEN_BUDGETS.policy_tokens,
        tokenizer=active_tokenizer,
    )

    return knowledge, policy


def _render(items: Sequence[Mapping[str, Any]]) -> str:
    return "\n\n".join(item["raw_text"] for item in items)


def _answer_prompt(
    query: str,
    knowledge: Sequence[Mapping[str, Any]],
    policy: Sequence[Mapping[str, Any]],
) -> str:
    return ANSWER_GENERATION_PROMPT.format(
        rewritten_query=query,
        knowledge_facts=_render(knowledge) or "No knowledge facts were retrieved.",
        policy_guidelines=_render(policy) or "No policy guidelines were retrieved.",
    )


def _rerank_score(item: Mapping[str, Any]) -> float | None:
    value = item.get("rerank_score")
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _drop_total_budget_tail(
    knowledge: list[dict[str, Any]],
    policy: list[dict[str, Any]],
    tokenizer: Tokenizer,
) -> None:
    if not knowledge and not policy:
        raise ValueError(
            "rewritten query and fixed answer prompt exceed total token budget"
        )
    if not knowledge:
        policy.pop()
        return
    if not policy:
        knowledge.pop()
        return

    knowledge_score = _rerank_score(knowledge[-1])
    policy_score = _rerank_score(policy[-1])
    if knowledge_score is None and policy_score is not None:
        knowledge.pop()
        return
    if policy_score is None and knowledge_score is not None:
        policy.pop()
        return
    if knowledge_score is not None and policy_score is not None:
        if knowledge_score < policy_score:
            knowledge.pop()
            return
        if policy_score < knowledge_score:
            policy.pop()
            return

    knowledge_size = len(tokenizer.encode(knowledge[-1]["raw_text"]))
    policy_size = len(tokenizer.encode(policy[-1]["raw_text"]))
    if knowledge_size > policy_size:
        knowledge.pop()
    else:
        policy.pop()


def _build_budgeted_prompt(
    query: str,
    knowledge_facts: object,
    policy_guidelines: object,
    tokenizer: Tokenizer,
) -> str:
    knowledge, policy = _render_context_pair(
        knowledge_facts,
        policy_guidelines,
        tokenizer=tokenizer,
    )
    while True:
        prompt = _answer_prompt(query, knowledge, policy)
        if len(tokenizer.encode(prompt)) <= TOKEN_BUDGETS.total_context_tokens:
            return prompt
        _drop_total_budget_tail(knowledge, policy, tokenizer)


async def generate_answer_stream(
    rewritten_query: str,
    knowledge_facts: Sequence[Mapping[str, Any]],
    policy_guidelines: Sequence[Mapping[str, Any]],
    *,
    runtime: RAGRuntime | None = None,
    timing_observer: TimingObserver | None = None,
) -> AsyncGenerator[str, None]:
    query = _required_text(rewritten_query, "rewritten_query")
    active_runtime = resolve_runtime(runtime)
    tokenizer = active_runtime.tokenizer or tiktoken.get_encoding(
        TEXT_PROCESSING.tokenizer_encoding
    )
    prompt_started = perf_counter()
    prompt = await asyncio.to_thread(
        _build_budgeted_prompt,
        query,
        knowledge_facts,
        policy_guidelines,
        tokenizer,
    )
    if timing_observer is not None:
        timing_observer(
            "prompt_construction",
            (perf_counter() - prompt_started) * 1000.0,
        )
    stream = await asyncio.to_thread(
        partial(
            active_runtime.llm.stream,
            prompt,
            model=PRIMARY_GENERATOR.model,
            reasoning=PRIMARY_GENERATOR.reasoning,
            max_output_tokens=PRIMARY_GENERATOR.max_output_tokens,
        )
    )
    if inspect.isawaitable(stream):
        stream = await stream
    if not hasattr(stream, "__aiter__"):
        raise TypeError("LLM stream() must return an async iterator")
    async for chunk in stream:
        if not isinstance(chunk, str):
            raise TypeError("LLM stream chunks must be strings")
        if chunk:
            yield chunk


__all__ = ["generate_answer_stream"]
