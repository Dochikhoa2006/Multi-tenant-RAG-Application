"""Configured embedding helpers and observable background conversation writes."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import math

from backend.mappings._common import required_uuid, validated_user_id
from backend.model_config import EMBEDDING_MODEL
from backend.rag.runtime import RAGRuntime, resolve_runtime


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _validated_vector(value: object) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("embedding provider must return a sequence of numbers")
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise TypeError("embedding vectors must contain only numbers")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise TypeError("embedding vectors must contain only numbers") from exc
        if not math.isfinite(number):
            raise ValueError("embedding vector values must be finite")
        vector.append(number)
    if not vector:
        raise ValueError("embedding vectors must not be empty")
    return vector


def embed_text(
    text: str,
    *,
    runtime: RAGRuntime | None = None,
) -> list[float]:
    value = _required_text(text, "text")
    active_runtime = resolve_runtime(runtime)
    return _validated_vector(
        active_runtime.embeddings.embed(value, model=EMBEDDING_MODEL)
    )


def embed_chunks(
    chunks: list[str],
    *,
    runtime: RAGRuntime | None = None,
) -> list[list[float]]:
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")
    texts = [_required_text(chunk, "chunk") for chunk in chunks]
    if not texts:
        return []
    active_runtime = resolve_runtime(runtime)
    raw_vectors = active_runtime.embeddings.embed_many(
        texts,
        model=EMBEDDING_MODEL,
    )
    if isinstance(raw_vectors, (str, bytes)) or not isinstance(raw_vectors, Sequence):
        raise TypeError("embedding provider must return a sequence of vectors")
    if len(raw_vectors) != len(texts):
        raise ValueError("embedding provider returned the wrong number of vectors")
    vectors = [_validated_vector(vector) for vector in raw_vectors]
    if len({len(vector) for vector in vectors}) != 1:
        raise ValueError("embedding vectors must have consistent dimensions")
    return vectors


def embed_conversation_background(
    user_id: str,
    conversation_id: str,
    question: str,
    answer: str,
    *,
    runtime: RAGRuntime | None = None,
) -> asyncio.Task[None]:
    """Compatibility wrapper returning a task the caller must retain and observe."""

    conversation = required_uuid(conversation_id, "conversation_id")
    loop = asyncio.get_running_loop()
    return loop.create_task(
        embed_conversation(
            user_id,
            conversation,
            question,
            answer,
            runtime=runtime,
        ),
        name=f"embed-conversation-{conversation}",
    )


async def embed_conversation(
    user_id: str,
    conversation_id: str,
    question: str,
    answer: str,
    *,
    runtime: RAGRuntime | None = None,
) -> None:
    """Embed and insert one complete Q+A pair without creating a task."""

    user = validated_user_id(user_id)
    conversation = required_uuid(conversation_id, "conversation_id")
    question_text = _required_text(question, "question")
    answer_text = _required_text(answer, "answer")
    active_runtime = resolve_runtime(runtime)
    raw_text = f"Question:\n{question_text}\n\nAnswer:\n{answer_text}"

    vector = await asyncio.to_thread(
        embed_text,
        raw_text,
        runtime=active_runtime,
    )

    def insert() -> None:
        collection = active_runtime.conversation_collection_factory(user)
        collection.insert(
            conversation,
            raw_text,
            vector,
        )

    await asyncio.to_thread(insert)


__all__ = [
    "embed_chunks",
    "embed_conversation",
    "embed_conversation_background",
    "embed_text",
]
