"""Conversation-aware query rewriting through an injected LLM client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from backend.model_config import QUERY_REWRITER
from backend.prompts import QUERY_REWRITE_PROMPT
from backend.rag.runtime import RAGRuntime, resolve_runtime


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _conversation_context(results: object) -> str:
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise TypeError("conversation_results must be a sequence of mappings")
    rendered: list[str] = []
    for index, result in enumerate(results, start=1):
        if not isinstance(result, Mapping):
            raise TypeError("conversation_results must contain mappings")
        raw_text = result.get("raw_text")
        if raw_text is None:
            properties = result.get("properties")
            if isinstance(properties, Mapping):
                raw_text = properties.get("raw_text")
        rendered.append(
            f"[Conversation {index}]\n{_required_text(raw_text, 'raw_text')}"
        )
    return "\n\n".join(rendered) if rendered else "No prior conversation context."


def rewrite_query(
    original_query: str,
    conversation_results: Sequence[Mapping[str, Any]],
    *,
    runtime: RAGRuntime | None = None,
) -> str:
    query = _required_text(original_query, "original_query")
    prompt = QUERY_REWRITE_PROMPT.format(
        original_query=query,
        conversation_context=_conversation_context(conversation_results),
    )
    response = resolve_runtime(runtime).llm.complete(
        prompt,
        model=QUERY_REWRITER.model,
    )
    if not isinstance(response, str):
        raise TypeError("LLM query rewrite response must be a string")
    rewritten = response.strip()
    if not rewritten:
        raise ValueError("LLM query rewrite response must not be empty")
    return rewritten


__all__ = ["rewrite_query"]
