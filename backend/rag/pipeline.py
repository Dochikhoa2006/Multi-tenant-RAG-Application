"""Asynchronous composition of the documented seven-step RAG flow."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable
from dataclasses import dataclass
from functools import partial
import inspect
from typing import Any

from backend.mappings._common import required_uuid, validated_user_id
from backend.rag.embedder import embed_conversation, embed_text
from backend.rag.generator import generate_answer_stream
from backend.rag.query_rewriter import rewrite_query
from backend.rag.retrieval import retrieve
from backend.rag.runtime import RAGRuntime, resolve_runtime


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _validated_collection(
    value: object,
    name: str,
    user_id: str,
    collection_type: str,
) -> object:
    if getattr(value, "user_id", None) != user_id:
        raise ValueError(f"{name} must be bound to user_id {user_id!r}")
    actual_collection_type = getattr(value, "collection_type", None)
    if (
        not isinstance(actual_collection_type, str)
        or actual_collection_type != collection_type
    ):
        raise ValueError(
            f"{name} collection_type must be {collection_type!r}, "
            f"got {actual_collection_type!r}"
        )
    if not callable(getattr(value, "hybrid_search", None)):
        raise TypeError(f"{name} must provide hybrid_search()")
    return value


@dataclass(frozen=True)
class UserRetrievalCollections:
    """The three existing Stage 2 collection wrappers for one user."""

    user_id: str
    conversations: object
    knowledge_facts: object
    policy: object

    def __post_init__(self) -> None:
        user = validated_user_id(self.user_id)
        object.__setattr__(self, "user_id", user)
        for name, collection_type in (
            ("conversations", "conversations"),
            ("knowledge_facts", "knowledge_facts"),
            ("policy", "policy"),
        ):
            _validated_collection(
                getattr(self, name),
                name,
                user,
                collection_type,
            )


async def _sync_call(function: object, /, *args: object, **kwargs: object) -> Any:
    if not callable(function):
        raise TypeError("function must be callable")
    return await asyncio.to_thread(partial(function, *args, **kwargs))


async def run_rag_pipeline(
    user_id: str,
    conversation_id: str,
    original_query: str,
    collections: UserRetrievalCollections,
    *,
    runtime: RAGRuntime | None = None,
) -> AsyncGenerator[str, None]:
    """Stream an answer and enqueue complete Q+A persistence on success only."""

    user = validated_user_id(user_id)
    conversation = required_uuid(conversation_id, "conversation_id")
    query = _required_text(original_query, "original_query")
    if not isinstance(collections, UserRetrievalCollections):
        raise TypeError("collections must be a UserRetrievalCollections")
    if collections.user_id != user:
        raise ValueError("collections user_id does not match user_id")
    active_runtime = resolve_runtime(runtime)
    queue = active_runtime.background_queue
    if queue is None:
        raise RuntimeError("RAG pipeline requires an observable background queue")

    original_vector = await _sync_call(embed_text, query, runtime=active_runtime)
    conversations = await _sync_call(
        retrieve,
        collections.conversations,
        query,
        original_vector,
        "conversations",
        runtime=active_runtime,
    )
    rewritten_query = await _sync_call(
        rewrite_query,
        query,
        conversations,
        runtime=active_runtime,
    )
    rewritten_vector = await _sync_call(
        embed_text,
        rewritten_query,
        runtime=active_runtime,
    )

    knowledge_task = _sync_call(
        retrieve,
        collections.knowledge_facts,
        rewritten_query,
        rewritten_vector,
        "knowledge_facts",
        runtime=active_runtime,
    )
    policy_task = _sync_call(
        retrieve,
        collections.policy,
        rewritten_query,
        rewritten_vector,
        "policy",
        runtime=active_runtime,
    )
    knowledge, policy = await asyncio.gather(knowledge_task, policy_task)

    answer_chunks: list[str] = []
    async for chunk in generate_answer_stream(
        rewritten_query,
        knowledge,
        policy,
        runtime=active_runtime,
    ):
        answer_chunks.append(chunk)
        yield chunk

    answer = "".join(answer_chunks)
    if not answer.strip():
        raise ValueError("answer stream completed without a non-empty answer")

    def work_factory() -> Awaitable[None]:
        return embed_conversation(
            user,
            conversation,
            query,
            answer,
            runtime=active_runtime,
        )

    accepted = queue.enqueue(user, "embed_conversation", work_factory)
    if not inspect.isawaitable(accepted):
        raise TypeError("background queue enqueue() must return an awaitable")
    await accepted


__all__ = ["UserRetrievalCollections", "run_rag_pipeline"]
