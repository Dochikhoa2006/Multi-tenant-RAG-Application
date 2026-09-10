"""Configured embedding helpers and observable background conversation writes."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import math

from backend.mappings._common import required_uuid, validated_user_id
from backend.model_config import EMBEDDING_MODEL, LATEON_EMBEDDING_DIMENSION
from backend.rag.runtime import RAGRuntime, resolve_runtime
from backend.wizard.diagnostics import (
    add_count,
    observe_stage,
    set_flag,
    set_framed_digest,
    set_sample,
    trace_operation_active,
    trace_utf8_bytes,
)


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


def _validated_multi_vector(value: object) -> list[list[float]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("multi-vector provider must return a sequence of vectors")
    rows = [_validated_vector(row) for row in value]
    if not rows:
        raise ValueError("multi-vector provider must not return an empty matrix")
    if len({len(row) for row in rows}) != 1:
        raise ValueError("multi-vector rows must have consistent dimensions")
    if len(rows[0]) != LATEON_EMBEDDING_DIMENSION:
        raise ValueError("LateOn multi-vector rows must have exactly 128 dimensions")
    return rows


def encode_query(
    text: str,
    *,
    runtime: RAGRuntime | None = None,
) -> list[list[float]]:
    value = _required_text(text, "text")
    active_runtime = resolve_runtime(runtime)
    return _validated_multi_vector(active_runtime.multi_vectors.encode_query(value))


def encode_documents(
    texts: Sequence[str],
    *,
    runtime: RAGRuntime | None = None,
) -> list[list[list[float]]]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of strings")
    validated = [_required_text(text, "document") for text in texts]
    if not validated:
        return []
    active_runtime = resolve_runtime(runtime)
    with observe_stage("chat.persistence_lateon"):
        raw = active_runtime.multi_vectors.encode_documents(validated)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("multi-vector provider must return a sequence of matrices")
    if len(raw) != len(validated):
        raise ValueError("multi-vector provider returned the wrong document count")
    matrices = [_validated_multi_vector(matrix) for matrix in raw]
    dimensions = {len(row) for matrix in matrices for row in matrix}
    if len(dimensions) != 1:
        raise ValueError("document multi-vectors must have consistent dimensions")
    if trace_operation_active():
        row_counts = [len(matrix) for matrix in matrices]
        add_count("persistence_lateon_document_count", len(matrices))
        add_count("persistence_lateon_total_rows", sum(row_counts))
        add_count("persistence_lateon_min_rows", min(row_counts))
        add_count("persistence_lateon_max_rows", max(row_counts))
        add_count("persistence_lateon_dimension", next(iter(dimensions)))
        set_flag("persistence_lateon_values_finite", True)
    return matrices


def conversation_segments(
    text: str,
    *,
    runtime: RAGRuntime | None = None,
) -> list[str]:
    canonical = _required_text(text, "text")
    active_runtime = resolve_runtime(runtime)
    raw = active_runtime.conversation_segmenter.segment_document(canonical)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("conversation segmenter must return a sequence of strings")
    segments = [_required_text(item, "conversation segment") for item in raw]
    if not segments:
        raise ValueError("conversation segmenter must return at least one segment")
    if "".join(segments) != canonical:
        raise ValueError("conversation segments must reconstruct canonical text exactly")
    return segments


def embed_text(
    text: str,
    *,
    runtime: RAGRuntime | None = None,
) -> list[float]:
    value = _required_text(text, "text")
    active_runtime = resolve_runtime(runtime)
    with observe_stage("chat.persistence_gte"):
        vector = _validated_vector(
            active_runtime.embeddings.embed(value, model=EMBEDDING_MODEL)
        )
    add_count("persistence_gte_vector_count", 1)
    add_count("persistence_gte_dimension", len(vector))
    set_flag("persistence_gte_values_finite", True)
    return vector


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
    raw_bytes = trace_utf8_bytes(raw_text)
    if raw_bytes is not None:
        add_count("persistence_payload_utf8_bytes", len(raw_bytes))
        set_framed_digest(
            "persistence_payload_sha256",
            "chat-conversation-persistence-payload-v1",
            (raw_bytes,),
        )

    with observe_stage("chat.persistence_segmentation"):
        segments = await asyncio.to_thread(
            conversation_segments, raw_text, runtime=active_runtime
        )
    add_count("persistence_segment_count", len(segments))
    set_sample(
        "persistence_segment_sizes",
        (f"{index}:{len(segment.encode('utf-8'))}" for index, segment in enumerate(segments)),
        exact_count=len(segments),
    )
    set_framed_digest(
        "persistence_segments_sha256",
        "chat-conversation-persistence-segments-v1",
        (segment.encode("utf-8") for segment in segments),
    )
    with observe_stage("chat.persistence_embeddings_fork_join"):
        segment_vectors, diversity_vector = await asyncio.gather(
            asyncio.to_thread(
                encode_documents,
                segments,
                runtime=active_runtime,
            ),
            asyncio.to_thread(
                embed_text,
                raw_text,
                runtime=active_runtime,
            ),
        )

    def insert() -> None:
        with observe_stage("chat.persistence_collection_factory"):
            collection = active_runtime.conversation_collection_factory(user)
        with observe_stage("chat.persistence_storage_total"):
            inserted_id = collection.insert(
                conversation,
                raw_text,
                segments,
                segment_vectors,
                diversity_vector,
            )
        if trace_operation_active():
            set_flag(
                "persistence_returned_conversation_id_matches",
                inserted_id == conversation,
            )

    await asyncio.to_thread(insert)


__all__ = [
    "embed_chunks",
    "encode_documents",
    "encode_query",
    "conversation_segments",
    "embed_conversation",
    "embed_conversation_background",
    "embed_text",
]
