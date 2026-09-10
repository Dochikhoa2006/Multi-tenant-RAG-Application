"""Validated asynchronous P3 session-title generation."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from functools import partial
import re

from backend.model_config import SESSION_TITLE_GENERATOR
from backend.prompts import SESSION_TITLE_PROMPT
from backend.rag.runtime import RAGRuntime, resolve_runtime
from backend.wizard.diagnostics import (
    activate_title_provider,
    add_count,
    observe_stage,
    set_framed_digest,
    set_text,
    trace_utf8_bytes,
)


_TITLE_WORD = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _conversation_list(value: object) -> str:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("conversation_list must be a sequence of strings")
    rendered = [
        f"[Conversation {index}]\n{_required_text(item, 'conversation')}"
        for index, item in enumerate(value, start=1)
    ]
    if not rendered:
        raise ValueError("conversation_list must not be empty")
    return "\n\n".join(rendered)


def validate_session_title(value: object) -> str:
    """Validate and normalize the public 3--6 word session-title contract."""

    if not isinstance(value, str):
        raise TypeError("session title response must be a string")
    if "\n" in value or "\r" in value:
        raise ValueError("session title must be a single line")
    words = value.strip().split()
    if not 3 <= len(words) <= 6:
        raise ValueError("session title must contain 3 to 6 words")
    if any(_TITLE_WORD.fullmatch(word) is None for word in words):
        raise ValueError("session title contains unsupported punctuation")
    return " ".join(words)


async def generate_session_title(
    conversation_list: Sequence[str],
    *,
    runtime: RAGRuntime | None = None,
) -> str:
    with observe_stage("chat.title_context_build"):
        conversations = _conversation_list(conversation_list)
    add_count("title_conversation_count", len(conversation_list))
    context_bytes = trace_utf8_bytes(conversations)
    if context_bytes is not None:
        add_count("title_context_utf8_bytes", len(context_bytes))
        set_framed_digest(
            "title_context_sha256",
            "chat-title-context-v1",
            (context_bytes,),
        )
    with observe_stage("chat.title_prompt_build"):
        prompt = SESSION_TITLE_PROMPT.format(conversation_list=conversations)
    prompt_bytes = trace_utf8_bytes(prompt)
    if prompt_bytes is not None:
        add_count("title_prompt_utf8_bytes", len(prompt_bytes))
        set_framed_digest(
            "title_prompt_sha256",
            "chat-title-prompt-v1",
            (prompt_bytes,),
        )
    with observe_stage("chat.title_runtime_setup"):
        active_runtime = resolve_runtime(runtime)
    with activate_title_provider():
        with observe_stage("chat.title_generation"):
            response = await asyncio.to_thread(
                partial(
                    active_runtime.llm.complete,
                    prompt,
                    model=SESSION_TITLE_GENERATOR.model,
                )
            )
    with observe_stage("chat.title_validation"):
        title = validate_session_title(response)
    set_text("title", title)
    return title


__all__ = ["generate_session_title", "validate_session_title"]
