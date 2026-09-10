"""Asynchronous composition of the documented seven-step RAG flow."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable
from dataclasses import dataclass
from functools import partial
import inspect
from time import perf_counter
from typing import Any

from backend.mappings._common import required_uuid, validated_user_id
from backend.rag.embedder import embed_conversation, encode_query
from backend.rag.generator import generate_answer_stream
from backend.rag.query_rewriter import rewrite_query
from backend.rag.retrieval import retrieve
from backend.rag.runtime import RAGRuntime, TimingObserver, resolve_runtime
from backend.wizard.diagnostics import (
    activate_operation,
    active_trace_handle,
    add_count,
    attach_related_task,
    observe_stage,
    set_flag,
    set_text,
)


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
    if not callable(getattr(value, "hydrate_mmr_head", None)):
        raise TypeError(f"{name} must provide hydrate_mmr_head()")
    return value


@dataclass(frozen=True)
class UserRetrievalCollections:
    """The three collection wrappers for one user."""

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


async def _timed_sync_call(
    phase: str,
    observer: TimingObserver | None,
    function: object,
    /,
    *args: object,
    **kwargs: object,
) -> Any:
    started = perf_counter()
    result = await _sync_call(function, *args, **kwargs)
    if observer is not None:
        observer(phase, (perf_counter() - started) * 1000.0)
    return result


async def run_rag_pipeline(
    user_id: str,
    conversation_id: str,
    original_query: str,
    collections: UserRetrievalCollections,
    *,
    runtime: RAGRuntime | None = None,
    timing_observer: TimingObserver | None = None,
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

    with observe_stage("chat.original_query_lateon"):
        original_vector = await _timed_sync_call(
            "original_query_embedding",
            timing_observer,
            encode_query,
            query,
            runtime=active_runtime,
        )
    add_count("original_query_lateon_rows", len(original_vector))
    add_count("original_query_lateon_dimension", len(original_vector[0]))
    set_flag("original_query_lateon_values_finite", True)
    conversations = await _sync_call(
        retrieve,
        collections.conversations,
        query,
        original_vector,
        "conversations",
        runtime=active_runtime,
        timing_observer=timing_observer,
    )
    rewritten_query = await _timed_sync_call(
        "query_rewrite",
        timing_observer,
        rewrite_query,
        query,
        conversations,
        runtime=active_runtime,
    )
    with observe_stage("chat.rewritten_query_lateon"):
        rewritten_vector = await _timed_sync_call(
            "rewritten_query_embedding",
            timing_observer,
            encode_query,
            rewritten_query,
            runtime=active_runtime,
        )
    add_count("rewritten_query_lateon_rows", len(rewritten_vector))
    add_count("rewritten_query_lateon_dimension", len(rewritten_vector[0]))
    set_flag("rewritten_query_lateon_values_finite", True)

    with observe_stage("chat.knowledge_policy_fork_join"):
        knowledge_task = _sync_call(
            retrieve,
            collections.knowledge_facts,
            rewritten_query,
            rewritten_vector,
            "knowledge_facts",
            runtime=active_runtime,
            timing_observer=timing_observer,
        )
        policy_task = _sync_call(
            retrieve,
            collections.policy,
            rewritten_query,
            rewritten_vector,
            "policy",
            runtime=active_runtime,
            timing_observer=timing_observer,
        )
        knowledge, policy = await asyncio.gather(knowledge_task, policy_task)

    answer_chunks: list[str] = []
    generation_started: float | None = None
    async for chunk in generate_answer_stream(
        rewritten_query,
        knowledge,
        policy,
        runtime=active_runtime,
        timing_observer=timing_observer,
    ):
        if generation_started is None:
            generation_started = perf_counter()
        answer_chunks.append(chunk)
        yield chunk

    if timing_observer is not None and generation_started is not None:
        timing_observer(
            "generation",
            (perf_counter() - generation_started) * 1000.0,
        )

    answer = "".join(answer_chunks)
    if not answer.strip():
        raise ValueError("answer stream completed without a non-empty answer")

    trace_handle = active_trace_handle()

    async def traced_persistence() -> None:
        with activate_operation(trace_handle):
            add_count("persistence_task_started_count", 1)
            try:
                with observe_stage("chat.persistence_task_execution"):
                    await embed_conversation(
                        user,
                        conversation,
                        query,
                        answer,
                        runtime=active_runtime,
                    )
            except BaseException:
                set_flag("persistence_task_succeeded", False)
                raise
            else:
                set_flag("persistence_task_succeeded", True)
                add_count("persistence_task_completed_count", 1)

    def work_factory() -> Awaitable[None]:
        if trace_handle is not None:
            return traced_persistence()
        return embed_conversation(
            user,
            conversation,
            query,
            answer,
            runtime=active_runtime,
        )

    with observe_stage("chat.persistence_enqueue"):
        accepted = queue.enqueue(user, "embed_conversation", work_factory)
        if not inspect.isawaitable(accepted):
            raise TypeError("background queue enqueue() must return an awaitable")
        task_id = await accepted
    add_count("persistence_enqueue_count", 1)
    set_flag("persistence_enqueue_accepted", True)
    set_text("persistence_conversation_id", conversation)
    attach_related_task(trace_handle, "conversation_persistence", task_id)


__all__ = ["UserRetrievalCollections", "run_rag_pipeline"]
