"""Structured query-rewrite metadata carried alongside the legacy string prompt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ConversationPair:
    """One validated historical user/assistant exchange in retrieval order."""

    object_id: str
    question: str
    answer: str

    def __post_init__(self) -> None:
        for name, value in (
            ("object_id", self.object_id),
            ("question", self.question),
            ("answer", self.answer),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must not be empty")


class QueryRewritePrompt(str):
    """P1-compatible string with structured data for the local Granite adapter."""

    original_query: str
    conversation_pairs: tuple[ConversationPair, ...]

    def __new__(
        cls,
        prompt: str,
        *,
        original_query: str,
        conversation_pairs: Iterable[ConversationPair],
    ) -> QueryRewritePrompt:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a nonempty string")
        if not isinstance(original_query, str) or not original_query.strip():
            raise ValueError("original_query must be a nonempty string")
        pairs = tuple(conversation_pairs)
        if any(not isinstance(pair, ConversationPair) for pair in pairs):
            raise TypeError("conversation_pairs must contain ConversationPair values")
        instance = super().__new__(cls, prompt)
        instance.original_query = original_query
        instance.conversation_pairs = pairs
        return instance


__all__ = ["ConversationPair", "QueryRewritePrompt"]
