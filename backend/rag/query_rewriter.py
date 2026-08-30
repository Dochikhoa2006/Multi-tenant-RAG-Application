"""Conversation-aware query rewriting through an injected LLM client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from typing import Any

from backend.model_config import QUERY_REWRITER
from backend.prompts import QUERY_REWRITE_PROMPT
from backend.rag.query_rewrite_contract import ConversationPair, QueryRewritePrompt
from backend.rag.runtime import RAGRuntime, resolve_runtime


logger = logging.getLogger(__name__)
_QUESTION_PREFIX = "Question:\n"
_ANSWER_SEPARATOR = "\n\nAnswer:\n"


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _object_id(result: Mapping[object, object], index: int) -> str:
    value = result.get("object_id")
    return value if isinstance(value, str) and value.strip() else f"result-{index}"


def _structured_pair(
    raw_text: str,
    *,
    object_id: str,
) -> ConversationPair | None:
    if not raw_text.startswith(_QUESTION_PREFIX):
        return None
    body = raw_text[len(_QUESTION_PREFIX) :]
    question, separator, answer = body.partition(_ANSWER_SEPARATOR)
    if not separator or not question.strip() or not answer.strip():
        return None
    return ConversationPair(
        object_id=object_id,
        question=question,
        answer=answer,
    )


def _conversation_context(
    results: object,
) -> tuple[str, tuple[ConversationPair, ...]]:
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise TypeError("conversation_results must be a sequence of mappings")
    rendered: list[str] = []
    pairs: list[ConversationPair] = []
    skipped_ids: list[str] = []
    for index, result in enumerate(results, start=1):
        if not isinstance(result, Mapping):
            skipped_ids.append(f"result-{index}")
            continue
        raw_text = result.get("raw_text")
        if raw_text is None:
            properties = result.get("properties")
            if isinstance(properties, Mapping):
                raw_text = properties.get("raw_text")
        object_id = _object_id(result, index)
        if not isinstance(raw_text, str) or not raw_text.strip():
            skipped_ids.append(object_id)
            continue
        valid_text = raw_text
        rendered.append(f"[Conversation {index}]\n{valid_text}")
        pair = _structured_pair(valid_text, object_id=object_id)
        if pair is None:
            skipped_ids.append(object_id)
        else:
            pairs.append(pair)
    if skipped_ids:
        logger.warning(
            "Skipped malformed query-rewrite conversation records",
            extra={
                "skipped_count": len(skipped_ids),
                "object_ids": tuple(skipped_ids),
            },
        )
    context = "\n\n".join(rendered) if rendered else "No prior conversation context."
    return context, tuple(pairs)


def rewrite_query(
    original_query: str,
    conversation_results: Sequence[Mapping[str, Any]],
    *,
    runtime: RAGRuntime | None = None,
) -> str:
    query = _required_text(original_query, "original_query")
    conversation_context, pairs = _conversation_context(conversation_results)
    prompt_text = QUERY_REWRITE_PROMPT.format(
        original_query=query,
        conversation_context=conversation_context,
    )
    prompt = QueryRewritePrompt(
        prompt_text,
        original_query=query,
        conversation_pairs=pairs,
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
