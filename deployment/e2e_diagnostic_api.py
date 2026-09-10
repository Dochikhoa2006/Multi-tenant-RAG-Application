"""Real HTTP executor, corpus checks, and deep tracing for Phase 2D."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from time import perf_counter, sleep
from typing import Any
from uuid import UUID, uuid4

import httpx

from backend.api.telemetry import TELEMETRY_SCHEMA_VERSION
from backend.model_config import (
    GTE_EMBEDDING_DIMENSION,
    LATEON_EMBEDDING_DIMENSION,
    PRIMARY_GENERATOR,
    SESSION_TITLE_GENERATOR,
)
from backend.wizard.diagnostics import (
    TRACE_OPERATION_HEADER,
    TRACE_SCHEMA_VERSION,
    TRACE_SESSION_HEADER,
    framed_content_digest,
)
from deployment.e2e_diagnostic import (
    E2EDiagnosticError,
    QuerySelection,
    RequestRecorder,
    SelectedQuery,
    utc_timestamp,
)
from deployment.wizard_diagnostic import CorpusGeneration, CorpusState
from deployment.wizard_diagnostic_api import CorpusStorage, TRACE_PATH


class E2EBatchFailed(E2EDiagnosticError):
    """Raised after every selected request was attempted and at least one failed."""


class StreamContractError(E2EDiagnosticError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DeepTraceContractError(E2EDiagnosticError):
    """The hidden trace did not prove the measured real query path."""


class PostGenerationContractError(E2EDiagnosticError):
    """One query's real post-generation result violated its contract."""


TRACE_FINALIZATION_TIMEOUT_SECONDS = 30.0
TRACE_POLL_SECONDS = 0.1
TASK_TIMEOUT_SECONDS = 900.0
TASK_POLL_SECONDS = 1.0


@dataclass(frozen=True)
class ValidatedStream:
    answer: str
    request_id: str
    conversation_id: str
    token_event_count: int
    answer_utf8_bytes: int
    answer_chunks_sha256: str
    telemetry_schema_version: str
    timings_ms: Mapping[str, object]


@dataclass(frozen=True)
class TaskObservation:
    task_id: str
    operation: str
    status: str
    error_code: str | None
    created_at: str | None
    started_at: str | None
    finished_at: str | None
    poll_count: int
    queue_wait_ms: float | None
    execution_ms: float | None
    total_ms: float | None

    def artifact(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "operation": self.operation,
            "status": self.status,
            "error_code": self.error_code,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "poll_count": self.poll_count,
            "queue_wait_ms": self.queue_wait_ms,
            "execution_ms": self.execution_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True)
class QueryAttemptResult:
    query: SelectedQuery
    status: str
    failure_scope: str | None
    failure_stage: str | None
    failure_code: str | None
    question: str
    answer: str
    answer_complete: bool
    session_id: str | None
    request_id: str | None
    conversation_id: str | None
    diagnostic_operation_id: str | None
    http_status: int | None
    query_http_attempted: bool
    token_event_count: int
    telemetry_schema_version: str | None
    timings_ms: Mapping[str, object] | None
    client_timings_ms: Mapping[str, object] | None
    trace_polling: Mapping[str, object] | None
    post_generation: Mapping[str, object] | None
    registry_verification: Mapping[str, object] | None
    deep_trace: Mapping[str, object] | None
    duration_ms: float
    started_at: str


_CHAT_TRACE_STAGES = (
    "chat.endpoint_pre_pipeline_total",
    "chat.api_validation_session",
    "chat.runtime_setup",
    "chat.collection_factory",
    "chat.session_stream_reservation",
    "chat.collection_ensure",
    "chat.original_query_lateon",
    "chat.conversation_hybrid",
    "chat.conversation_bge",
    "chat.conversation_collapse",
    "chat.conversation_relevance_floor",
    "chat.conversation_adaptive_k",
    "chat.conversation_hydration",
    "chat.conversation_mmr",
    "chat.conversation_finalize",
    "chat.granite_budgeting",
    "chat.granite_http",
    "chat.rewritten_query_lateon",
    "chat.knowledge_policy_fork_join",
    "chat.knowledge_hybrid",
    "chat.knowledge_bge",
    "chat.knowledge_relevance_floor",
    "chat.knowledge_adaptive_k",
    "chat.knowledge_hydration",
    "chat.knowledge_mmr",
    "chat.knowledge_finalize",
    "chat.knowledge_budgeting",
    "chat.policy_hybrid",
    "chat.policy_bge",
    "chat.policy_relevance_floor",
    "chat.policy_adaptive_k",
    "chat.policy_hydration",
    "chat.policy_mmr",
    "chat.policy_finalize",
    "chat.policy_budgeting",
    "chat.qwen_prompt_construction",
    "chat.qwen_http",
    "chat.qwen_ttft",
    "chat.qwen_generation",
    "chat.qwen_stream_total",
)

_POST_GENERATION_STAGES = (
    "chat.conversation_registry_update",
    "chat.persistence_enqueue",
    "chat.persistence_task_execution",
    "chat.persistence_segmentation",
    "chat.persistence_embeddings_fork_join",
    "chat.persistence_lateon",
    "chat.persistence_gte",
    "chat.persistence_collection_factory",
    "chat.persistence_storage_total",
    "chat.persistence_weaviate_insert",
    "chat.title_enqueue",
    "chat.title_task_execution",
    "chat.title_snapshot",
    "chat.title_transcript_render",
    "chat.title_context_build",
    "chat.title_prompt_build",
    "chat.title_runtime_setup",
    "chat.title_generation",
    "chat.title_qwen_http",
    "chat.title_qwen_total",
    "chat.title_validation",
    "chat.title_registry_update",
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeepTraceContractError(f"{name} must be an object")
    return value


def _nonnegative_int(mapping: Mapping[str, Any], name: str) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeepTraceContractError(f"Deep trace count {name} is invalid")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DeepTraceContractError(f"Deep trace digest {name} is invalid")
    try:
        int(value, 16)
    except ValueError as exc:
        raise DeepTraceContractError(
            f"Deep trace digest {name} is invalid"
        ) from exc
    return value.lower()


def _complete_sample(
    samples: Mapping[str, Any], name: str
) -> tuple[str, ...]:
    sample = _mapping(samples.get(name), f"deep trace sample {name}")
    items = sample.get("items")
    exact_count = sample.get("exact_count")
    if (
        not isinstance(items, list)
        or any(not isinstance(item, str) or not item for item in items)
        or isinstance(exact_count, bool)
        or not isinstance(exact_count, int)
        or exact_count != len(items)
        or sample.get("truncated") is not False
    ):
        raise DeepTraceContractError(
            f"Deep trace sample {name} is incomplete or truncated"
        )
    return tuple(items)


def _item_fingerprint(value: str, name: str) -> tuple[str, str]:
    object_id, separator, digest = value.partition(":")
    if not separator:
        raise DeepTraceContractError(
            f"Deep trace item fingerprint {name} is malformed"
        )
    try:
        object_id = str(UUID(object_id))
    except ValueError as exc:
        raise DeepTraceContractError(
            f"Deep trace item fingerprint {name} has an invalid ID"
        ) from exc
    return object_id, _sha256(digest, name)


def _complete_item_fingerprints(
    samples: Mapping[str, Any], name: str
) -> tuple[tuple[str, str], ...]:
    return tuple(
        _item_fingerprint(value, name) for value in _complete_sample(samples, name)
    )


def _validate_stage(stages: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    stage = _mapping(stages.get(name), f"deep trace stage {name}")
    calls = stage.get("call_count")
    failures = stage.get("failure_count")
    durations = (stage.get("total_ms"), stage.get("min_ms"), stage.get("max_ms"))
    if (
        isinstance(calls, bool)
        or not isinstance(calls, int)
        or calls <= 0
        or isinstance(failures, bool)
        or not isinstance(failures, int)
        or failures < 0
        or failures > calls
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in durations
        )
        or float(stage["min_ms"]) > float(stage["max_ms"])
        or float(stage["max_ms"]) > float(stage["total_ms"]) + 0.001
    ):
        raise DeepTraceContractError(f"Deep trace stage {name} is invalid")
    return stage


def _validate_kp_evidence(
    prefix: str,
    counts: Mapping[str, Any],
    flags: Mapping[str, Any],
    digests: Mapping[str, Any],
    samples: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    hybrid = _nonnegative_int(counts, f"{prefix}_hybrid_candidate_count")
    hybrid_calls = _nonnegative_int(counts, f"{prefix}_hybrid_storage_call_count")
    bge_input = _nonnegative_int(counts, f"{prefix}_bge_input_count")
    bge_result = _nonnegative_int(counts, f"{prefix}_bge_result_count")
    bge_calls = _nonnegative_int(counts, f"{prefix}_bge_model_call_count")
    relevance_input = _nonnegative_int(counts, f"{prefix}_relevance_input_count")
    eligible = _nonnegative_int(counts, f"{prefix}_relevance_eligible_count")
    adaptive_eligible = _nonnegative_int(
        counts, f"{prefix}_adaptive_eligible_count"
    )
    adaptive = _nonnegative_int(counts, f"{prefix}_adaptive_selected_count")
    hydrated_requested = _nonnegative_int(
        counts, f"{prefix}_hydration_requested_count"
    )
    hydrated = _nonnegative_int(counts, f"{prefix}_hydration_result_count")
    quarantined = _nonnegative_int(
        counts, f"{prefix}_hydration_quarantined_count"
    )
    hydration_calls = _nonnegative_int(
        counts, f"{prefix}_hydration_storage_call_count"
    )
    mmr_pool = _nonnegative_int(counts, f"{prefix}_mmr_pool_count")
    mmr_selected = _nonnegative_int(counts, f"{prefix}_mmr_selected_count")
    prebudget = _nonnegative_int(counts, f"{prefix}_prebudget_count")
    final_count = _nonnegative_int(counts, f"{prefix}_final_count")
    dropped = _nonnegative_int(counts, f"{prefix}_budget_dropped_count")
    mmr_usable = flags.get(f"{prefix}_mmr_usable")
    hydration_values_finite = flags.get(
        f"{prefix}_hydration_values_finite"
    )
    mmr_fallback = flags.get(f"{prefix}_mmr_fallback")
    if (
        hybrid_calls != 1
        or bge_input != hybrid
        or bge_result != bge_input
        or bge_calls != (1 if bge_input else 0)
        or flags.get(f"{prefix}_bge_scores_finite") is not True
        or relevance_input != bge_result
        or eligible > relevance_input
        or adaptive_eligible != eligible
        or adaptive > eligible
        or hydrated_requested != adaptive
        or hydrated + quarantined != hydrated_requested
        or hydration_calls != (1 if hydrated_requested else 0)
        or mmr_usable not in {True, False}
        or hydration_values_finite is not mmr_usable
        or mmr_fallback is mmr_usable
        or mmr_pool != hydrated
        or mmr_selected > mmr_pool
        or prebudget != mmr_selected
        or final_count > prebudget
        or dropped != prebudget - final_count
        or flags.get(f"{prefix}_final_proof_truncated") is not False
    ):
        raise DeepTraceContractError(
            f"{prefix.title()} retrieval evidence is inconsistent"
        )
    final_ids = _complete_sample(samples, f"{prefix}_final_ids")
    final_fingerprints = _complete_item_fingerprints(
        samples, f"{prefix}_final_item_fingerprints"
    )
    if (
        len(final_ids) != final_count
        or len(final_fingerprints) != final_count
        or tuple(item[0] for item in final_fingerprints) != final_ids
    ):
        raise DeepTraceContractError(
            f"{prefix.title()} final-context evidence is inconsistent"
        )
    _sha256(digests.get(f"{prefix}_final_context_sha256"), f"{prefix} context")
    return final_ids, final_fingerprints


def validate_chat_trace_evidence(
    payload: Mapping[str, Any],
    *,
    user_id: str,
    trace_session_id: str,
    operation_id: str,
    chat_session_id: str,
    stream: ValidatedStream,
    http_status: int,
) -> Mapping[str, Any]:
    """Validate one completed, bounded trace without touching the real pipeline."""

    if not isinstance(stream, ValidatedStream) or http_status != 200:
        raise DeepTraceContractError("Deep trace is not paired with a successful stream")
    if (
        payload.get("schema_version") != TRACE_SCHEMA_VERSION
        or payload.get("session_id") != trace_session_id
        or payload.get("user_id") != user_id
        or payload.get("overflowed") is not False
        or payload.get("trace_faulted") is not False
        or payload.get("missing_evidence") is not False
    ):
        raise DeepTraceContractError("Deep trace session is invalid or incomplete")
    operations = payload.get("operations")
    if not isinstance(operations, list) or len(operations) != 1:
        raise DeepTraceContractError("Deep trace did not return one operation")
    operation = _mapping(operations[0], "deep trace operation")
    if (
        operation.get("operation_id") != operation_id
        or operation.get("kind") != "chat_query"
        or operation.get("user_id") != user_id
        or operation.get("collection_type") != "conversations"
        or operation.get("wizard_id") != chat_session_id
        or operation.get("task_id") is not None
        or operation.get("outcome") != "succeeded"
        or not isinstance(operation.get("finished_at"), str)
    ):
        raise DeepTraceContractError("Deep trace operation correlation is invalid")

    stages = _mapping(operation.get("stages"), "deep trace stages")
    validated_stages = {
        name: _validate_stage(stages, name) for name in _CHAT_TRACE_STAGES
    }
    if any(
        validated_stages[name].get("failure_count") != 0
        for name in _CHAT_TRACE_STAGES
        if name not in {"chat.granite_http", "chat.qwen_http"}
    ):
        raise DeepTraceContractError("A required deep trace stage failed")

    counts = _mapping(operation.get("counts"), "deep trace counts")
    flags = _mapping(operation.get("flags"), "deep trace flags")
    digests = _mapping(operation.get("digests"), "deep trace digests")
    samples = _mapping(operation.get("samples"), "deep trace samples")
    texts = _mapping(operation.get("texts"), "deep trace texts")

    lateon_rows = _nonnegative_int(counts, "original_query_lateon_rows")
    if (
        lateon_rows <= 0
        or _nonnegative_int(counts, "original_query_lateon_dimension")
        != LATEON_EMBEDDING_DIMENSION
        or flags.get("original_query_lateon_values_finite") is not True
    ):
        raise DeepTraceContractError("Original-query LateOn evidence is invalid")
    rewritten_lateon_rows = _nonnegative_int(
        counts, "rewritten_query_lateon_rows"
    )
    if (
        rewritten_lateon_rows <= 0
        or _nonnegative_int(counts, "rewritten_query_lateon_dimension")
        != LATEON_EMBEDDING_DIMENSION
        or flags.get("rewritten_query_lateon_values_finite") is not True
    ):
        raise DeepTraceContractError("Rewritten-query LateOn evidence is invalid")

    hybrid = _nonnegative_int(counts, "conversation_hybrid_candidate_count")
    if _nonnegative_int(counts, "conversation_hybrid_storage_call_count") != 1:
        raise DeepTraceContractError("Conversation hybrid call count is invalid")
    bge_input = _nonnegative_int(counts, "conversation_bge_input_count")
    bge_result = _nonnegative_int(counts, "conversation_bge_result_count")
    bge_calls = _nonnegative_int(counts, "conversation_bge_model_call_count")
    collapsed_input = _nonnegative_int(counts, "conversation_collapse_input_count")
    collapsed = _nonnegative_int(counts, "conversation_collapse_result_count")
    relevance_input = _nonnegative_int(counts, "conversation_relevance_input_count")
    eligible = _nonnegative_int(counts, "conversation_relevance_eligible_count")
    adaptive_eligible = _nonnegative_int(counts, "conversation_adaptive_eligible_count")
    adaptive = _nonnegative_int(counts, "conversation_adaptive_selected_count")
    hydrated_requested = _nonnegative_int(
        counts, "conversation_hydration_requested_count"
    )
    hydrated = _nonnegative_int(counts, "conversation_hydration_result_count")
    quarantined = _nonnegative_int(
        counts, "conversation_hydration_quarantined_count"
    )
    hydration_calls = _nonnegative_int(
        counts, "conversation_hydration_storage_call_count"
    )
    mmr_pool = _nonnegative_int(counts, "conversation_mmr_pool_count")
    mmr_selected = _nonnegative_int(counts, "conversation_mmr_selected_count")
    final_count = _nonnegative_int(counts, "conversation_final_count")
    if (
        bge_input != hybrid
        or bge_result != bge_input
        or bge_calls != (1 if bge_input else 0)
        or flags.get("conversation_bge_scores_finite") is not True
        or collapsed_input != bge_result
        or collapsed > collapsed_input
        or relevance_input != collapsed
        or eligible > relevance_input
        or adaptive_eligible != eligible
        or adaptive > eligible
        or hydrated_requested != adaptive
        or hydrated + quarantined != hydrated_requested
        or hydration_calls != (1 if hydrated_requested else 0)
        or mmr_pool != hydrated
        or mmr_selected > mmr_pool
        or final_count != mmr_selected
    ):
        raise DeepTraceContractError("Conversation retrieval counts are inconsistent")

    final_ids = _complete_sample(samples, "conversation_final_ids")
    input_pair_ids = _complete_sample(samples, "granite_input_pair_ids")
    retained_pair_ids = _complete_sample(samples, "granite_retained_pair_ids")
    input_pairs = _nonnegative_int(counts, "granite_input_pair_count")
    retained_pairs = _nonnegative_int(counts, "granite_retained_pair_count")
    dropped_pairs = _nonnegative_int(counts, "granite_dropped_pair_count")
    if (
        len(final_ids) != final_count
        or tuple(final_ids) != tuple(input_pair_ids)
        or input_pairs != len(input_pair_ids)
        or retained_pairs != len(retained_pair_ids)
        or dropped_pairs != input_pairs - retained_pairs
        or retained_pair_ids != input_pair_ids[:retained_pairs]
        or flags.get("granite_input_pair_ids_proof_truncated") is not False
        or flags.get("granite_retained_pair_ids_proof_truncated") is not False
    ):
        raise DeepTraceContractError("Granite Conversation context proof is invalid")
    _sha256(digests.get("conversation_final_context_sha256"), "final context")
    _sha256(digests.get("granite_conversation_context_sha256"), "Granite context")
    _sha256(digests.get("granite_request_messages_sha256"), "Granite messages")

    attempts = _nonnegative_int(counts, "granite_attempt_count")
    transient_failures = _nonnegative_int(
        counts, "granite_transient_failure_count"
    )
    transient_retries = _nonnegative_int(counts, "granite_transient_retry_count")
    format_failures = _nonnegative_int(counts, "granite_format_failure_count")
    format_retries = _nonnegative_int(counts, "granite_format_retry_count")
    usage_responses = _nonnegative_int(counts, "granite_usage_response_count")
    rendered_tokens = _nonnegative_int(counts, "granite_rendered_input_tokens")
    final_prompt_tokens = _nonnegative_int(counts, "granite_final_prompt_tokens")
    final_completion_tokens = _nonnegative_int(
        counts, "granite_final_completion_tokens"
    )
    prompt_tokens_total = _nonnegative_int(counts, "granite_prompt_tokens_total")
    completion_tokens_total = _nonnegative_int(
        counts, "granite_completion_tokens_total"
    )
    if (
        attempts not in {1, 2}
        or transient_retries + format_retries != attempts - 1
        or transient_retries > transient_failures
        or format_retries > format_failures
        or usage_responses <= 0
        or usage_responses > attempts
        or validated_stages["chat.granite_http"].get("call_count") != attempts
        or validated_stages["chat.granite_http"].get("failure_count")
        != transient_failures
        or rendered_tokens <= 0
        or final_prompt_tokens != rendered_tokens
        or final_completion_tokens <= 0
        or prompt_tokens_total < final_prompt_tokens
        or completion_tokens_total < final_completion_tokens
        or (flags.get("granite_strict_json") is True)
        == (flags.get("granite_repair_applied") is True)
    ):
        raise DeepTraceContractError("Granite request evidence is inconsistent")
    cached_available = flags.get("granite_cached_tokens_available")
    if cached_available not in {True, False}:
        raise DeepTraceContractError("Granite cache-usage evidence is invalid")
    if cached_available:
        cached = _nonnegative_int(counts, "granite_final_cached_prompt_tokens")
        if cached > final_prompt_tokens:
            raise DeepTraceContractError("Granite cache-usage evidence is invalid")
    rewritten = texts.get("granite_rewritten_query")
    if (
        not isinstance(rewritten, str)
        or not rewritten.strip()
        or flags.get("granite_rewritten_query_truncated") is not False
        or _nonnegative_int(counts, "granite_rewritten_query_utf8_bytes")
        != len(rewritten.encode("utf-8"))
    ):
        raise DeepTraceContractError("Granite rewritten query is incomplete")
    expected_rewrite_digest = hashlib.sha256()
    expected_rewrite_digest.update(b"chat-granite-rewritten-query-v1")
    encoded = rewritten.encode("utf-8")
    expected_rewrite_digest.update(len(encoded).to_bytes(8, "big"))
    expected_rewrite_digest.update(encoded)
    if _sha256(
        digests.get("granite_rewritten_query_sha256"), "rewritten query"
    ) != expected_rewrite_digest.hexdigest():
        raise DeepTraceContractError("Granite rewritten-query digest is inconsistent")

    knowledge_ids, knowledge_fingerprints = _validate_kp_evidence(
        "knowledge", counts, flags, digests, samples
    )
    policy_ids, policy_fingerprints = _validate_kp_evidence(
        "policy", counts, flags, digests, samples
    )
    qwen_context: dict[
        str, tuple[tuple[str, ...], tuple[tuple[str, str], ...]]
    ] = {}
    for prefix, final_ids, final_fingerprints in (
        ("knowledge", knowledge_ids, knowledge_fingerprints),
        ("policy", policy_ids, policy_fingerprints),
    ):
        used_count = _nonnegative_int(counts, f"qwen_{prefix}_used_count")
        used_ids = _complete_sample(samples, f"qwen_{prefix}_used_ids")
        used_fingerprints = _complete_item_fingerprints(
            samples, f"qwen_{prefix}_used_item_fingerprints"
        )
        if (
            used_count != len(used_ids)
            or used_count != len(used_fingerprints)
            or used_ids != final_ids[:used_count]
            or used_fingerprints != final_fingerprints[:used_count]
            or flags.get(f"qwen_{prefix}_used_proof_truncated") is not False
        ):
            raise DeepTraceContractError(
                f"Qwen {prefix} context is not an exact retained prefix"
            )
        _nonnegative_int(counts, f"qwen_{prefix}_context_utf8_bytes")
        _sha256(
            digests.get(f"qwen_{prefix}_context_sha256"),
            f"Qwen {prefix} context",
        )
        qwen_context[prefix] = used_ids, used_fingerprints

    constructed_prompt = _sha256(
        digests.get("qwen_constructed_prompt_sha256"), "constructed Qwen prompt"
    )
    stream_prompt = _sha256(
        digests.get("qwen_stream_prompt_sha256"), "stream Qwen prompt"
    )
    request_prompt = _sha256(
        digests.get("qwen_request_prompt_sha256"), "request Qwen prompt"
    )
    _sha256(digests.get("qwen_request_messages_sha256"), "Qwen messages")
    constructed_bytes = _nonnegative_int(
        counts, "qwen_constructed_prompt_utf8_bytes"
    )
    stream_bytes = _nonnegative_int(counts, "qwen_stream_prompt_utf8_bytes")
    request_bytes = _nonnegative_int(counts, "qwen_request_prompt_utf8_bytes")
    if (
        not qwen_context
        or constructed_prompt != stream_prompt
        or stream_prompt != request_prompt
        or constructed_bytes <= 0
        or constructed_bytes != stream_bytes
        or stream_bytes != request_bytes
        or _nonnegative_int(counts, "qwen_request_message_count") != 1
        or _nonnegative_int(counts, "qwen_request_max_tokens")
        != PRIMARY_GENERATOR.max_output_tokens
        or texts.get("qwen_requested_model") != PRIMARY_GENERATOR.model
        or texts.get("qwen_observed_model") != PRIMARY_GENERATOR.model
    ):
        raise DeepTraceContractError("Qwen prompt/request evidence is inconsistent")

    qwen_attempts = _nonnegative_int(counts, "qwen_attempt_count")
    qwen_failures = _nonnegative_int(counts, "qwen_transient_failure_count")
    qwen_retries = _nonnegative_int(counts, "qwen_retry_count")
    if (
        qwen_attempts not in {1, 2}
        or qwen_failures != qwen_retries
        or qwen_retries != qwen_attempts - 1
        or validated_stages["chat.qwen_http"].get("call_count")
        != qwen_attempts
        or validated_stages["chat.qwen_http"].get("failure_count")
        != qwen_failures
        or validated_stages["chat.qwen_ttft"].get("call_count") != 1
        or validated_stages["chat.qwen_generation"].get("call_count") != 1
        or validated_stages["chat.qwen_stream_total"].get("call_count") != 1
        or flags.get("qwen_finish_seen") is not True
        or flags.get("qwen_done_seen") is not True
        or texts.get("qwen_finish_reason") not in {"stop", "length"}
    ):
        raise DeepTraceContractError("Qwen stream evidence is inconsistent")

    usage_events = _nonnegative_int(counts, "qwen_usage_event_count")
    usage_available = flags.get("qwen_usage_available")
    if (
        flags.get("qwen_usage_valid") is not True
        or usage_available is not (usage_events > 0)
    ):
        raise DeepTraceContractError("Qwen usage evidence is invalid")
    if usage_events:
        qwen_prompt_tokens = _nonnegative_int(counts, "qwen_final_prompt_tokens")
        qwen_completion_tokens = _nonnegative_int(
            counts, "qwen_final_completion_tokens"
        )
        qwen_total_tokens = _nonnegative_int(counts, "qwen_final_total_tokens")
        cached_available = flags.get("qwen_cached_tokens_available")
        if (
            qwen_completion_tokens <= 0
            or qwen_total_tokens != qwen_prompt_tokens + qwen_completion_tokens
            or cached_available not in {True, False}
        ):
            raise DeepTraceContractError("Qwen token usage is inconsistent")
        if cached_available and _nonnegative_int(
            counts, "qwen_final_cached_prompt_tokens"
        ) > qwen_prompt_tokens:
            raise DeepTraceContractError("Qwen cached-token usage is inconsistent")

    qwen_chunks = _nonnegative_int(counts, "qwen_answer_chunk_count")
    qwen_answer_bytes = _nonnegative_int(counts, "qwen_answer_utf8_bytes")
    qwen_answer_digest = _sha256(
        digests.get("qwen_answer_chunks_sha256"), "Qwen answer"
    )
    sse_tokens = _nonnegative_int(counts, "sse_token_event_count")
    sse_answer_bytes = _nonnegative_int(counts, "sse_answer_utf8_bytes")
    sse_answer_digest = _sha256(
        digests.get("sse_answer_chunks_sha256"), "SSE answer"
    )
    telemetry_events = _nonnegative_int(counts, "sse_telemetry_event_count")
    done_events = _nonnegative_int(counts, "sse_done_event_count")
    error_events = _nonnegative_int(counts, "sse_error_event_count")
    total_events = _nonnegative_int(counts, "sse_total_event_count")
    if (
        qwen_chunks <= 0
        or qwen_chunks != sse_tokens
        or sse_tokens != stream.token_event_count
        or qwen_answer_bytes != sse_answer_bytes
        or sse_answer_bytes != stream.answer_utf8_bytes
        or qwen_answer_digest != sse_answer_digest
        or sse_answer_digest != stream.answer_chunks_sha256
        or telemetry_events != 1
        or done_events != 1
        or error_events != 0
        or total_events != sse_tokens + 2
        or _nonnegative_int(counts, "sse_first_token_ordinal") != 1
        or _nonnegative_int(counts, "sse_last_token_ordinal") != sse_tokens
        or _nonnegative_int(counts, "sse_telemetry_ordinal") != sse_tokens + 1
        or _nonnegative_int(counts, "sse_done_ordinal") != sse_tokens + 2
        or flags.get("sse_success_order_valid") is not True
        or texts.get("sse_terminal_event") != "done"
    ):
        raise DeepTraceContractError("Qwen and public SSE evidence do not agree")
    return operation


def _related_task_ids(operation: Mapping[str, Any]) -> dict[str, str]:
    related = _mapping(operation.get("related_tasks"), "deep trace related tasks")
    if set(related) != {"conversation_persistence", "session_title"}:
        raise DeepTraceContractError(
            "Deep trace did not correlate both post-generation tasks"
        )
    task_ids: dict[str, str] = {}
    for role, value in related.items():
        try:
            task_ids[str(role)] = str(UUID(str(value)))
        except (TypeError, ValueError) as exc:
            raise DeepTraceContractError(
                "Deep trace related task ID is invalid"
            ) from exc
    if len(set(task_ids.values())) != 2:
        raise DeepTraceContractError("Post-generation task IDs must be distinct")
    return task_ids


def validate_post_generation_evidence(
    operation: Mapping[str, Any],
    *,
    conversation_id: str,
    persistence_task: TaskObservation,
    title_task: TaskObservation,
) -> str:
    """Validate successful background work using only existing-call evidence."""

    stages = _mapping(operation.get("stages"), "deep trace stages")
    validated = {
        name: _validate_stage(stages, name) for name in _POST_GENERATION_STAGES
    }
    if any(stage.get("failure_count") != 0 for stage in validated.values()):
        raise PostGenerationContractError("A post-generation trace stage failed")

    related = _related_task_ids(operation)
    if (
        related["conversation_persistence"] != persistence_task.task_id
        or related["session_title"] != title_task.task_id
        or persistence_task.operation != "embed_conversation"
        or title_task.operation != "generate_session_title"
        or persistence_task.status != "succeeded"
        or title_task.status != "succeeded"
    ):
        raise PostGenerationContractError(
            "Post-generation task correlation or status is invalid"
        )

    counts = _mapping(operation.get("counts"), "deep trace counts")
    flags = _mapping(operation.get("flags"), "deep trace flags")
    digests = _mapping(operation.get("digests"), "deep trace digests")
    texts = _mapping(operation.get("texts"), "deep trace texts")

    segment_count = _nonnegative_int(counts, "persistence_segment_count")
    lateon_documents = _nonnegative_int(
        counts, "persistence_lateon_document_count"
    )
    lateon_total_rows = _nonnegative_int(
        counts, "persistence_lateon_total_rows"
    )
    lateon_min_rows = _nonnegative_int(counts, "persistence_lateon_min_rows")
    lateon_max_rows = _nonnegative_int(counts, "persistence_lateon_max_rows")
    expected_inserts = _nonnegative_int(
        counts, "persistence_weaviate_expected_insert_count"
    )
    insert_attempts = _nonnegative_int(
        counts, "persistence_weaviate_insert_attempt_count"
    )
    insert_successes = _nonnegative_int(
        counts, "persistence_weaviate_insert_success_count"
    )
    if (
        _nonnegative_int(counts, "conversation_registry_update_count") != 1
        or _nonnegative_int(counts, "persistence_enqueue_count") != 1
        or flags.get("persistence_enqueue_accepted") is not True
        or _nonnegative_int(counts, "persistence_task_started_count") != 1
        or _nonnegative_int(counts, "persistence_task_completed_count") != 1
        or flags.get("persistence_task_succeeded") is not True
        or segment_count <= 0
        or lateon_documents != segment_count
        or lateon_total_rows < lateon_documents
        or lateon_min_rows <= 0
        or lateon_min_rows > lateon_max_rows
        or lateon_max_rows > lateon_total_rows
        or _nonnegative_int(counts, "persistence_lateon_dimension")
        != LATEON_EMBEDDING_DIMENSION
        or flags.get("persistence_lateon_values_finite") is not True
        or _nonnegative_int(counts, "persistence_gte_vector_count") != 1
        or _nonnegative_int(counts, "persistence_gte_dimension")
        != GTE_EMBEDDING_DIMENSION
        or flags.get("persistence_gte_values_finite") is not True
        or expected_inserts != segment_count
        or insert_attempts != expected_inserts
        or insert_successes != expected_inserts
        or flags.get("persistence_returned_conversation_id_matches") is not True
        or texts.get("persistence_conversation_id") != conversation_id
    ):
        raise PostGenerationContractError(
            "Conversation persistence evidence is inconsistent"
        )
    _sha256(digests.get("persistence_payload_sha256"), "persistence payload")
    _sha256(digests.get("persistence_segments_sha256"), "persistence segments")
    _sha256(
        digests.get("persistence_segment_ids_sha256"),
        "persistence segment IDs",
    )

    title_attempts = _nonnegative_int(counts, "title_qwen_attempt_count")
    title_retries = _nonnegative_int(counts, "title_qwen_retry_count")
    title_transient_failures = _nonnegative_int(
        counts, "title_qwen_transient_failure_count"
    )
    if (
        _nonnegative_int(counts, "title_enqueue_count") != 1
        or flags.get("title_enqueue_accepted") is not True
        or _nonnegative_int(counts, "title_task_started_count") != 1
        or _nonnegative_int(counts, "title_task_completed_count") != 1
        or flags.get("title_task_succeeded") is not True
        or _nonnegative_int(counts, "title_registry_update_count") != 1
        or _nonnegative_int(counts, "title_snapshot_conversation_count") <= 0
        or flags.get("title_snapshot_trigger_matches") is not True
        or texts.get("title_snapshot_last_conversation_id") != conversation_id
        or title_attempts not in {1, 2}
        or title_retries != title_attempts - 1
        or title_transient_failures != title_retries
        or validated["chat.title_qwen_http"].get("call_count") != title_attempts
        or validated["chat.title_qwen_http"].get("failure_count")
        != title_transient_failures
        or texts.get("title_qwen_requested_model")
        != SESSION_TITLE_GENERATOR.model
        or texts.get("title_qwen_observed_model")
        != SESSION_TITLE_GENERATOR.model
        or texts.get("title_qwen_finish_reason") != "stop"
        or flags.get("title_qwen_usage_valid") is not True
    ):
        raise PostGenerationContractError("Session-title evidence is inconsistent")
    title = texts.get("title")
    if not isinstance(title, str) or not 3 <= len(title.split()) <= 6:
        raise PostGenerationContractError("Session title is invalid")
    if (
        _sha256(digests.get("title_prompt_sha256"), "title prompt")
        != _sha256(digests.get("title_qwen_prompt_sha256"), "title Qwen prompt")
    ):
        raise PostGenerationContractError(
            "The title prompt does not match the Qwen request"
        )
    usage_available = flags.get("title_qwen_usage_available")
    if usage_available not in {True, False}:
        raise PostGenerationContractError("Title usage availability is invalid")
    usage_responses = _nonnegative_int(counts, "title_qwen_usage_response_count")
    if usage_responses != (1 if usage_available else 0):
        raise PostGenerationContractError("Title usage count is invalid")
    if usage_available:
        prompt_tokens = _nonnegative_int(counts, "title_qwen_prompt_tokens")
        completion_tokens = _nonnegative_int(
            counts, "title_qwen_completion_tokens"
        )
        total_tokens = _nonnegative_int(counts, "title_qwen_total_tokens")
        if total_tokens != prompt_tokens + completion_tokens:
            raise PostGenerationContractError("Title token usage is inconsistent")
    return title


def _trace_artifact(
    payload: Mapping[str, Any], operation: Mapping[str, Any]
) -> dict[str, object]:
    return {
        "schema_version": payload.get("schema_version"),
        "session_id": payload.get("session_id"),
        "operation": {
            key: operation.get(key)
            for key in (
                "operation_id",
                "kind",
                "user_id",
                "collection_type",
                "wizard_id",
                "task_id",
                "related_tasks",
                "started_at",
                "finished_at",
                "outcome",
                "stages",
                "counts",
                "flags",
                "digests",
                "samples",
                "texts",
            )
        },
    }


class _DeepTraceSession:
    def __init__(
        self,
        client: httpx.Client,
        runtime_url: str,
        user_id: str,
        run_id: str,
    ) -> None:
        self.client = client
        self.runtime_url = runtime_url
        self.user_id = user_id
        self.run_id = run_id
        self.session_id = str(uuid4())
        self.started = False

    def start(self) -> None:
        response = self.client.post(
            f"{self.runtime_url}{TRACE_PATH}",
            json={
                "user_id": self.user_id,
                "session_id": self.session_id,
                "run_id": self.run_id,
            },
        )
        if response.status_code != 201:
            raise DeepTraceContractError("Deep trace session could not be started")
        self.started = True
        payload = _mapping(response.json(), "deep trace start")
        if (
            payload.get("schema_version") != TRACE_SCHEMA_VERSION
            or payload.get("session_id") != self.session_id
            or payload.get("run_id") != self.run_id
            or payload.get("user_id") != self.user_id
        ):
            raise DeepTraceContractError("Deep trace start contract is invalid")

    def operation(
        self,
        operation_id: str,
        chat_session_id: str,
        stream: ValidatedStream,
        http_status: int,
    ) -> tuple[Mapping[str, Any], dict[str, object], dict[str, object]]:
        payload, _, polling = self.wait_operation(operation_id)
        operation = validate_chat_trace_evidence(
            payload,
            user_id=self.user_id,
            trace_session_id=self.session_id,
            operation_id=operation_id,
            chat_session_id=chat_session_id,
            stream=stream,
            http_status=http_status,
        )
        return operation, _trace_artifact(payload, operation), polling

    def wait_operation(
        self,
        operation_id: str,
        *,
        timeout_seconds: float = TRACE_FINALIZATION_TIMEOUT_SECONDS,
        poll_seconds: float = TRACE_POLL_SECONDS,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, object]]:
        started = perf_counter()
        polls = 0
        while True:
            polls += 1
            response = self.client.get(
                f"{self.runtime_url}{TRACE_PATH}",
                params={
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "operation_id": operation_id,
                },
            )
            if response.status_code == 200:
                try:
                    payload = _mapping(response.json(), "deep trace GET")
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise DeepTraceContractError(
                        "Deep trace response is malformed"
                    ) from exc
                if (
                    payload.get("schema_version") != TRACE_SCHEMA_VERSION
                    or payload.get("session_id") != self.session_id
                    or payload.get("run_id") != self.run_id
                    or payload.get("user_id") != self.user_id
                    or payload.get("overflowed") is not False
                    or payload.get("trace_faulted") is not False
                    or payload.get("missing_evidence") is not False
                ):
                    raise DeepTraceContractError(
                        "Deep trace session is invalid or incomplete"
                    )
                operations = payload.get("operations")
                if not isinstance(operations, list) or len(operations) != 1:
                    raise DeepTraceContractError(
                        "Deep trace did not return one operation"
                    )
                operation = _mapping(operations[0], "deep trace operation")
                if operation.get("operation_id") != operation_id:
                    raise DeepTraceContractError(
                        "Deep trace operation correlation is invalid"
                    )
                if operation.get("outcome") in {"succeeded", "failed"} and isinstance(
                    operation.get("finished_at"), str
                ):
                    return payload, operation, {
                        "poll_count": polls,
                        "wait_ms": round((perf_counter() - started) * 1000.0, 3),
                    }
            elif response.status_code not in {404}:
                raise DeepTraceContractError("Deep trace operation is unavailable")
            if perf_counter() - started >= timeout_seconds:
                raise DeepTraceContractError(
                    "Deep trace operation did not reach a terminal state"
                )
            sleep(poll_seconds)

    def fetch_operation(
        self, operation_id: str
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        response = self.client.get(
            f"{self.runtime_url}{TRACE_PATH}",
            params={
                "user_id": self.user_id,
                "session_id": self.session_id,
                "operation_id": operation_id,
            },
        )
        if response.status_code != 200:
            raise DeepTraceContractError("Deep trace operation is unavailable")
        try:
            payload = _mapping(response.json(), "deep trace GET")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DeepTraceContractError("Deep trace response is malformed") from exc
        if (
            payload.get("schema_version") != TRACE_SCHEMA_VERSION
            or payload.get("session_id") != self.session_id
            or payload.get("run_id") != self.run_id
            or payload.get("user_id") != self.user_id
            or payload.get("overflowed") is not False
            or payload.get("trace_faulted") is not False
            or payload.get("missing_evidence") is not False
        ):
            raise DeepTraceContractError("Deep trace session is invalid or incomplete")
        operations = payload.get("operations")
        if not isinstance(operations, list) or len(operations) != 1:
            raise DeepTraceContractError("Deep trace did not return one operation")
        operation = _mapping(operations[0], "deep trace operation")
        if operation.get("operation_id") != operation_id:
            raise DeepTraceContractError("Deep trace operation correlation is invalid")
        return payload, operation

    def delete(self) -> None:
        if not self.started:
            return
        response = self.client.delete(
            f"{self.runtime_url}{TRACE_PATH}",
            params={"user_id": self.user_id, "session_id": self.session_id},
        )
        if response.status_code != 204:
            raise DeepTraceContractError("Deep trace session could not be deleted")
        self.started = False


def _uuid(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise StreamContractError("INVALID_RESPONSE", f"{name} is missing")
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise StreamContractError("INVALID_RESPONSE", f"{name} is invalid") from exc


def _partial_answer(events: Sequence[tuple[str, object]]) -> str:
    parts: list[str] = []
    for name, payload in events:
        if name != "token" or not isinstance(payload, Mapping):
            continue
        text = payload.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


def validate_e2e_stream(
    events: Sequence[tuple[str, object]],
    response_request_id: object,
) -> ValidatedStream:
    """Validate the public SSE contract without evaluating answer quality."""

    from deployment.ragctl import validate_chat_result

    header_request_id = _uuid(response_request_id, "X-Request-ID")
    event_names: list[str] = []
    answer_parts: list[str] = []
    telemetry: Mapping[str, object] | None = None
    done: Mapping[str, object] | None = None
    for event_name, payload in events:
        event_names.append(event_name)
        if not isinstance(payload, Mapping):
            raise StreamContractError(
                "INVALID_SSE_CONTRACT", "SSE payload must be an object"
            )
        if payload.get("request_id") != header_request_id:
            raise StreamContractError(
                "REQUEST_ID_MISMATCH", "SSE request IDs do not match the response"
            )
        if event_name == "token":
            text = payload.get("text")
            if not isinstance(text, str) or not text:
                raise StreamContractError(
                    "INVALID_SSE_CONTRACT", "Token event is malformed"
                )
            answer_parts.append(text)
        elif event_name == "telemetry":
            if telemetry is not None:
                raise StreamContractError(
                    "INVALID_SSE_CONTRACT", "Telemetry event is duplicated"
                )
            telemetry = payload
        elif event_name == "done":
            if done is not None:
                raise StreamContractError(
                    "INVALID_SSE_CONTRACT", "Done event is duplicated"
                )
            done = payload
        elif event_name == "error":
            code = payload.get("code")
            raise StreamContractError(
                code if isinstance(code, str) and code else "CHAT_PROCESSING_FAILED",
                "RAG stream returned an error event",
            )
        else:
            raise StreamContractError(
                "INVALID_SSE_CONTRACT", "RAG stream returned an unknown event"
            )
    try:
        answer, timings = validate_chat_result(
            event_names,
            answer_parts,
            telemetry,
            done,
            verify_atlas_grounding=False,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise StreamContractError(
            "INVALID_SSE_CONTRACT", "RAG stream violated its public contract"
        ) from exc
    if telemetry is None or telemetry.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
        raise StreamContractError(
            "INVALID_TELEMETRY_SCHEMA", "RAG telemetry schema is invalid"
        )
    if done is None:
        raise StreamContractError("INVALID_SSE_CONTRACT", "Done event is missing")
    conversation_id = _uuid(done.get("conversation_id"), "conversation_id")
    return ValidatedStream(
        answer=answer,
        request_id=header_request_id,
        conversation_id=conversation_id,
        token_event_count=event_names.count("token"),
        answer_utf8_bytes=len(answer.encode("utf-8")),
        answer_chunks_sha256=framed_content_digest(
            "chat-qwen-answer-chunks-v1",
            (part.encode("utf-8") for part in answer_parts),
        ),
        telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
        timings_ms=timings,
    )


def verify_physical_corpus(
    config: Mapping[str, str],
    state: CorpusState,
    active: CorpusGeneration,
) -> dict[str, int]:
    """Read and compare the exact Phase 1 corpus without mutating or recovering it."""

    counts: dict[str, int] = {}
    with CorpusStorage(config, state.diagnostic_user_id) as storage:
        storage.validate_membership(state)
        for document in active.documents:
            integrity = storage.inspect_document(
                state.diagnostic_user_id,
                document.collection,
                document.wizard_id,
                require_nonempty=True,
            )
            if (
                integrity.chunk_ids != document.chunk_ids
                or integrity.fingerprint != document.storage_fingerprint
            ):
                raise E2EDiagnosticError(
                    f"Phase 1 {document.collection} corpus integrity "
                    "does not match state"
                )
            counts[document.collection] = integrity.chunk_count
    return counts


def _created_session(response: httpx.Response, user_id: str) -> str:
    if response.status_code != 201:
        raise StreamContractError(
            "SESSION_CREATE_FAILED", "Chat session creation did not return 201"
        )
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise StreamContractError(
            "INVALID_SESSION_RESPONSE", "Chat session response is not JSON"
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("user_id") != user_id:
        raise StreamContractError(
            "INVALID_SESSION_RESPONSE", "Chat session response has wrong ownership"
        )
    return _uuid(payload.get("session_id"), "session_id")


def _parsed_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{name} is not timezone-aware")
    return parsed


def _task_observation(
    payload: Mapping[str, Any],
    *,
    task_id: str,
    user_id: str,
    expected_operation: str,
    poll_count: int,
) -> TaskObservation:
    if (
        payload.get("task_id") != task_id
        or payload.get("user_id") != user_id
        or payload.get("operation") != expected_operation
        or payload.get("status") not in {"succeeded", "failed"}
    ):
        raise ValueError("task response correlation is invalid")
    created = _parsed_datetime(payload.get("created_at"), "created_at")
    started = _parsed_datetime(payload.get("started_at"), "started_at")
    finished = _parsed_datetime(payload.get("finished_at"), "finished_at")
    if not created <= started <= finished:
        raise ValueError("task timestamps are out of order")
    status_value = str(payload["status"])
    error_code = payload.get("error_code")
    if status_value == "succeeded":
        if error_code is not None:
            raise ValueError("successful task returned an error code")
    elif error_code not in {"TASK_FAILED", "TASK_CANCELLED"}:
        raise ValueError("failed task returned an invalid error code")
    return TaskObservation(
        task_id=task_id,
        operation=expected_operation,
        status=status_value,
        error_code=None if error_code is None else str(error_code),
        created_at=created.isoformat(),
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        poll_count=poll_count,
        queue_wait_ms=round((started - created).total_seconds() * 1000.0, 3),
        execution_ms=round((finished - started).total_seconds() * 1000.0, 3),
        total_ms=round((finished - created).total_seconds() * 1000.0, 3),
    )


def poll_task(
    client: httpx.Client,
    runtime_url: str,
    user_id: str,
    task_id: str,
    expected_operation: str,
    *,
    timeout_seconds: float = TASK_TIMEOUT_SECONDS,
    poll_seconds: float = TASK_POLL_SECONDS,
) -> TaskObservation:
    started = perf_counter()
    polls = 0
    while True:
        polls += 1
        try:
            response = client.get(
                f"{runtime_url}/api/tasks/{task_id}",
                params={"user_id": user_id},
            )
        except httpx.HTTPError:
            return TaskObservation(
                task_id,
                expected_operation,
                "unavailable",
                "TASK_STATUS_TRANSPORT_ERROR",
                None,
                None,
                None,
                polls,
                None,
                None,
                None,
            )
        if response.status_code in {401, 403}:
            raise DeepTraceContractError("Task status authentication failed")
        if response.status_code == 200:
            try:
                payload = response.json()
                if not isinstance(payload, Mapping):
                    raise ValueError("task response must be an object")
                status_value = payload.get("status")
                if status_value in {"succeeded", "failed"}:
                    return _task_observation(
                        payload,
                        task_id=task_id,
                        user_id=user_id,
                        expected_operation=expected_operation,
                        poll_count=polls,
                    )
                if status_value not in {"queued", "running"}:
                    raise ValueError("task status is invalid")
            except (TypeError, ValueError, json.JSONDecodeError):
                return TaskObservation(
                    task_id,
                    expected_operation,
                    "invalid",
                    "INVALID_TASK_RESPONSE",
                    None,
                    None,
                    None,
                    polls,
                    None,
                    None,
                    None,
                )
        elif response.status_code != 404:
            return TaskObservation(
                task_id,
                expected_operation,
                "unavailable",
                "TASK_STATUS_HTTP_ERROR",
                None,
                None,
                None,
                polls,
                None,
                None,
                None,
            )
        if perf_counter() - started >= timeout_seconds:
            return TaskObservation(
                task_id,
                expected_operation,
                "timeout",
                "TASK_STATUS_TIMEOUT",
                None,
                None,
                None,
                polls,
                None,
                None,
                None,
            )
        sleep(poll_seconds)


def verify_session_postcondition(
    client: httpx.Client,
    runtime_url: str,
    user_id: str,
    session_id: str,
    conversation_id: str,
    question: str,
    answer: str,
    title: str,
) -> dict[str, object]:
    started = perf_counter()
    response = client.get(
        f"{runtime_url}/api/chat/sessions/{session_id}",
        params={"user_id": user_id},
    )
    if response.status_code in {401, 403}:
        raise DeepTraceContractError("Session verification authentication failed")
    if response.status_code != 200:
        raise PostGenerationContractError("Session verification request failed")
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise PostGenerationContractError(
            "Session verification response is invalid"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("session_id") != session_id
        or payload.get("user_id") != user_id
        or payload.get("title") != title
    ):
        raise PostGenerationContractError("Session postcondition is inconsistent")
    conversations = payload.get("conversations")
    if not isinstance(conversations, list):
        raise PostGenerationContractError("Session conversations are invalid")
    matching = [
        item
        for item in conversations
        if isinstance(item, Mapping)
        and item.get("conversation_id") == conversation_id
    ]
    if (
        len(matching) != 1
        or matching[0].get("question") != question
        or matching[0].get("answer") != answer
    ):
        raise PostGenerationContractError(
            "Conversation registry postcondition is inconsistent"
        )
    return {
        "status": "succeeded",
        "conversation_present": True,
        "question_matches": True,
        "answer_matches": True,
        "title": title,
        "conversation_count": len(conversations),
        "verification_latency_ms": round(
            (perf_counter() - started) * 1000.0, 3
        ),
    }


def _record_attempt(recorder: RequestRecorder, result: QueryAttemptResult) -> None:
    recorder.record(
        source_index=result.query.source_index,
        status=result.status,
        question=result.question,
        answer=result.answer,
        answer_complete=result.answer_complete,
        session_id=result.session_id,
        request_id=result.request_id,
        conversation_id=result.conversation_id,
        http_status=result.http_status,
        token_event_count=result.token_event_count,
        telemetry_schema_version=result.telemetry_schema_version,
        timings_ms=result.timings_ms,
        duration_ms=result.duration_ms,
        started_at=result.started_at,
        failure_stage=result.failure_stage,
        failure_code=result.failure_code,
        diagnostic_operation_id=result.diagnostic_operation_id,
        deep_trace=result.deep_trace,
        failure_scope=result.failure_scope,
        query_http_attempted=result.query_http_attempted,
        client_timings_ms=result.client_timings_ms,
        trace_polling=result.trace_polling,
        post_generation=result.post_generation,
        registry_verification=result.registry_verification,
    )


def _stage_total(operation: Mapping[str, Any], name: str) -> object:
    stages = operation.get("stages")
    if not isinstance(stages, Mapping):
        return "unavailable"
    stage = stages.get(name)
    if not isinstance(stage, Mapping):
        return "unavailable"
    return stage.get("total_ms", "unavailable")


def format_query_terminal(result: QueryAttemptResult) -> str:
    operation: Mapping[str, Any] = {}
    if isinstance(result.deep_trace, Mapping):
        candidate = result.deep_trace.get("operation")
        if isinstance(candidate, Mapping):
            operation = candidate
    counts = operation.get("counts")
    if not isinstance(counts, Mapping):
        counts = {}
    texts = operation.get("texts")
    if not isinstance(texts, Mapping):
        texts = {}
    post = result.post_generation if isinstance(result.post_generation, Mapping) else {}
    persistence = post.get("conversation_persistence")
    if not isinstance(persistence, Mapping):
        persistence = {}
    title_task = post.get("session_title")
    if not isinstance(title_task, Mapping):
        title_task = {}
    client = result.client_timings_ms or {}
    qwen_tokens = counts.get("qwen_final_total_tokens", "unavailable")
    granite_tokens = counts.get("granite_final_total_tokens", "unavailable")
    title_tokens = counts.get("title_qwen_total_tokens", "unavailable")
    title = (
        result.registry_verification.get("title", "unavailable")
        if isinstance(result.registry_verification, Mapping)
        else "unavailable"
    )
    return "\n".join(
        (
            f"Query [{result.query.source_index}]: {result.question}",
            f"Rewrite: {texts.get('granite_rewritten_query', 'unavailable')}",
            "Retrieval: "
            f"conversation={counts.get('conversation_final_count', 'unavailable')} "
            f"knowledge={counts.get('knowledge_final_count', 'unavailable')} "
            f"policy={counts.get('policy_final_count', 'unavailable')} "
            f"kp_fork_join_ms={_stage_total(operation, 'chat.knowledge_policy_fork_join')}",
            "Models: "
            f"granite_ms={_stage_total(operation, 'chat.granite_http')} "
            f"granite_tokens={granite_tokens} "
            f"qwen_ms={_stage_total(operation, 'chat.qwen_stream_total')} "
            f"qwen_ttft_ms={_stage_total(operation, 'chat.qwen_ttft')} "
            f"qwen_tokens={qwen_tokens}",
            "Client: "
            f"ttft_ms={client.get('ttft_ms', 'unavailable')} "
            f"stream_total_ms={client.get('stream_total_ms', 'unavailable')} "
            f"attempt_total_ms={round(result.duration_ms, 3)}",
            "Persistence: "
            f"status={persistence.get('status', 'unavailable')} "
            f"segments={counts.get('persistence_segment_count', 'unavailable')} "
            f"segmentation_ms={_stage_total(operation, 'chat.persistence_segmentation')} "
            f"lateon_ms={_stage_total(operation, 'chat.persistence_lateon')} "
            f"gte_ms={_stage_total(operation, 'chat.persistence_gte')} "
            f"weaviate_ms={_stage_total(operation, 'chat.persistence_weaviate_insert')} "
            f"inserted={counts.get('persistence_weaviate_insert_success_count', 'unavailable')} "
            f"queue_wait_ms={persistence.get('queue_wait_ms', 'unavailable')} "
            f"execution_ms={persistence.get('execution_ms', 'unavailable')} "
            f"total_ms={persistence.get('total_ms', 'unavailable')}",
            "Title: "
            f"status={title_task.get('status', 'unavailable')} "
            f"qwen_ms={_stage_total(operation, 'chat.title_qwen_total')} "
            f"qwen_tokens={title_tokens} "
            f"queue_wait_ms={title_task.get('queue_wait_ms', 'unavailable')} "
            f"execution_ms={title_task.get('execution_ms', 'unavailable')} "
            f"total_ms={title_task.get('total_ms', 'unavailable')} value={title}",
            f"Result: {result.status}"
            + (
                ""
                if result.failure_code is None
                else f" ({result.failure_scope}:{result.failure_code})"
            ),
            "Answer:",
            result.answer,
        )
    )


def _execute_query(
    client: httpx.Client,
    runtime_url: str,
    user_id: str,
    session_id: str,
    query: SelectedQuery,
    trace: _DeepTraceSession,
    *,
    started_at: str,
    started_clock: float,
) -> QueryAttemptResult:
    from deployment.ragctl import iter_sse

    events: list[tuple[str, object]] = []
    response_status: int | None = None
    response_request_id: str | None = None
    operation_id = str(uuid4())
    result: ValidatedStream | None = None
    deep_trace: Mapping[str, object] | None = None
    trace_polling: Mapping[str, object] | None = None
    post_generation: Mapping[str, object] | None = None
    registry_verification: Mapping[str, object] | None = None
    code: str | None = None
    failure_scope: str | None = None
    failure_stage: str | None = None
    query_started = perf_counter()
    first_token_ms: float | None = None
    stream_total_ms: float | None = None
    try:
        with client.stream(
            "POST",
            f"{runtime_url}/api/chat/query",
            headers={
                "Accept": "text/event-stream",
                TRACE_SESSION_HEADER: trace.session_id,
                TRACE_OPERATION_HEADER: operation_id,
            },
            json={
                "user_id": user_id,
                "session_id": session_id,
                "question": query.question,
            },
        ) as response:
            response_status = response.status_code
            raw_request_id = response.headers.get("X-Request-ID")
            try:
                response_request_id = _uuid(raw_request_id, "X-Request-ID")
            except StreamContractError:
                response_request_id = None
            if response.status_code != 200:
                if response.status_code in {401, 403}:
                    raise StreamContractError(
                        "AUTHENTICATION_FAILED",
                        "Chat query authentication failed",
                    )
                raise StreamContractError(
                    "QUERY_HTTP_ERROR", "Chat query did not return HTTP 200"
                )
            content_type = response.headers.get("Content-Type", "")
            if content_type.partition(";")[0].strip().lower() != "text/event-stream":
                raise StreamContractError(
                    "INVALID_CONTENT_TYPE", "Chat query did not return an SSE stream"
                )
            for event in iter_sse(response.iter_lines()):
                events.append(event)
                if event[0] == "token" and first_token_ms is None:
                    first_token_ms = (perf_counter() - query_started) * 1000.0
                if event[0] in {"done", "error"}:
                    stream_total_ms = (perf_counter() - query_started) * 1000.0
            result = validate_e2e_stream(events, raw_request_id)
    except KeyboardInterrupt:
        raise
    except httpx.HTTPError:
        code = "TRANSPORT_ERROR"
    except (json.JSONDecodeError, UnicodeError):
        code = "INVALID_SSE"
    except StreamContractError as exc:
        code = exc.code
    except (TypeError, ValueError, RuntimeError):
        code = "INVALID_RESPONSE"
    if stream_total_ms is None:
        stream_total_ms = (perf_counter() - query_started) * 1000.0
    client_timings = {
        "ttft_ms": None if first_token_ms is None else round(first_token_ms, 3),
        "stream_total_ms": round(stream_total_ms, 3),
    }
    if result is not None:
        try:
            payload, operation, trace_polling = trace.wait_operation(operation_id)
            operation = validate_chat_trace_evidence(
                payload,
                user_id=user_id,
                trace_session_id=trace.session_id,
                operation_id=operation_id,
                chat_session_id=session_id,
                stream=result,
                http_status=response_status or 0,
            )
            flags = _mapping(operation.get("flags"), "deep trace flags")
            if flags.get("title_enqueue_accepted") is False:
                raise PostGenerationContractError("Title enqueue was not accepted")
            related = _related_task_ids(operation)
            persistence_task = poll_task(
                client,
                runtime_url,
                user_id,
                related["conversation_persistence"],
                "embed_conversation",
            )
            title_task = poll_task(
                client,
                runtime_url,
                user_id,
                related["session_title"],
                "generate_session_title",
            )
            post_generation = {
                "conversation_persistence": persistence_task.artifact(),
                "session_title": title_task.artifact(),
            }
            final_payload, final_operation = trace.fetch_operation(operation_id)
            deep_trace = _trace_artifact(final_payload, final_operation)
            if persistence_task.status != "succeeded":
                raise PostGenerationContractError(
                    "Conversation persistence did not succeed"
                )
            if title_task.status != "succeeded":
                raise PostGenerationContractError("Session title did not succeed")
            persistence_finished = _parsed_datetime(
                persistence_task.finished_at, "persistence finished_at"
            )
            title_started = _parsed_datetime(
                title_task.started_at, "title started_at"
            )
            if persistence_finished > title_started:
                raise PostGenerationContractError(
                    "Post-generation tasks violated same-user FIFO"
                )
            title = validate_post_generation_evidence(
                final_operation,
                conversation_id=result.conversation_id,
                persistence_task=persistence_task,
                title_task=title_task,
            )
            registry_verification = verify_session_postcondition(
                client,
                runtime_url,
                user_id,
                session_id,
                result.conversation_id,
                query.question,
                result.answer,
                title,
            )
        except PostGenerationContractError:
            code = "POST_GENERATION_FAILED"
            failure_scope = "individual"
            failure_stage = "post_generation"
        except DeepTraceContractError:
            code = "DEEP_TRACE_INVALID"
            failure_scope = "systemic"
            failure_stage = "trace"
        except (httpx.HTTPError, TypeError, ValueError, RuntimeError):
            code = "POST_GENERATION_INVALID"
            failure_scope = "individual"
            failure_stage = "post_generation"
    elif response_status == 200:
        try:
            payload, operation, trace_polling = trace.wait_operation(operation_id)
            related_value = operation.get("related_tasks")
            if isinstance(related_value, Mapping) and related_value:
                related = _related_task_ids(operation)
                persistence_task = poll_task(
                    client,
                    runtime_url,
                    user_id,
                    related["conversation_persistence"],
                    "embed_conversation",
                )
                title_task = poll_task(
                    client,
                    runtime_url,
                    user_id,
                    related["session_title"],
                    "generate_session_title",
                )
                post_generation = {
                    "conversation_persistence": persistence_task.artifact(),
                    "session_title": title_task.artifact(),
                }
                payload, operation = trace.fetch_operation(operation_id)
            deep_trace = _trace_artifact(payload, operation)
        except DeepTraceContractError:
            code = "DEEP_TRACE_INVALID"
            failure_scope = "systemic"
            failure_stage = "trace"

    if code is not None and failure_scope is None:
        if code == "AUTHENTICATION_FAILED":
            failure_scope = "systemic"
            failure_stage = "authentication"
        else:
            failure_scope = "individual"
            failure_stage = "query"
    status = "succeeded" if code is None and result is not None else "failed"
    return QueryAttemptResult(
        query=query,
        status=status,
        failure_scope=failure_scope,
        failure_stage=failure_stage,
        failure_code=code,
        question=query.question,
        answer=result.answer if result is not None else _partial_answer(events),
        answer_complete=result is not None,
        session_id=session_id,
        request_id=(
            result.request_id if result is not None else response_request_id
        ),
        conversation_id=None if result is None else result.conversation_id,
        diagnostic_operation_id=operation_id,
        http_status=response_status,
        query_http_attempted=True,
        token_event_count=sum(name == "token" for name, _ in events),
        telemetry_schema_version=(
            None if result is None else result.telemetry_schema_version
        ),
        timings_ms=None if result is None else result.timings_ms,
        client_timings_ms=client_timings,
        trace_polling=trace_polling,
        post_generation=post_generation,
        registry_verification=registry_verification,
        deep_trace=deep_trace,
        duration_ms=(perf_counter() - started_clock) * 1000.0,
        started_at=started_at,
    )


def _session_failure_result(
    query: SelectedQuery,
    *,
    started_at: str,
    started_clock: float,
    code: str,
    http_status: int | None,
    scope: str,
) -> QueryAttemptResult:
    return QueryAttemptResult(
        query=query,
        status="failed",
        failure_scope=scope,
        failure_stage=(
            "interruption"
            if code == "INTERRUPTED"
            else "authentication"
            if scope == "systemic"
            else "session_create"
        ),
        failure_code=code,
        question=query.question,
        answer="",
        answer_complete=False,
        session_id=None,
        request_id=None,
        conversation_id=None,
        diagnostic_operation_id=None,
        http_status=http_status,
        query_http_attempted=False,
        token_event_count=0,
        telemetry_schema_version=None,
        timings_ms=None,
        client_timings_ms=None,
        trace_polling=None,
        post_generation=None,
        registry_verification=None,
        deep_trace=None,
        duration_ms=(perf_counter() - started_clock) * 1000.0,
        started_at=started_at,
    )


def run_e2e_phase_2d(
    runtime_url: str,
    runtime_headers: Mapping[str, str],
    config: Mapping[str, str],
    state: CorpusState,
    active: CorpusGeneration,
    selection: QuerySelection,
    recorder: RequestRecorder,
    progress: dict[str, object],
    *,
    continuous: bool,
    run_id: str,
) -> dict[str, int]:
    """Verify the retained corpus and trace real public chat requests serially."""

    try:
        physical_counts = verify_physical_corpus(config, state, active)
    except BaseException:
        progress["physical_corpus_status"] = "failed"
        raise
    progress["physical_corpus_status"] = "succeeded"
    timeout = httpx.Timeout(connect=30, read=900, write=120, pool=30)
    shared_session_id: str | None = None
    with httpx.Client(headers=dict(runtime_headers), timeout=timeout) as client:
        health = client.get(f"{runtime_url}/health")
        if health.status_code != 200:
            raise E2EDiagnosticError("RAG runtime is not ready")
        trace = _DeepTraceSession(
            client,
            runtime_url,
            state.diagnostic_user_id,
            run_id,
        )
        phase_error: BaseException | None = None
        try:
            trace.start()
            progress["trace_status"] = "active"
            if continuous:
                try:
                    shared_session_id = _created_session(
                        client.post(
                            f"{runtime_url}/api/chat/sessions",
                            json={"user_id": state.diagnostic_user_id},
                        ),
                        state.diagnostic_user_id,
                    )
                except (httpx.HTTPError, StreamContractError) as exc:
                    raise E2EDiagnosticError(
                        "Continuous chat session could not be created"
                    ) from exc
            for query in selection.selected:
                started_at = utc_timestamp()
                started_clock = perf_counter()
                session_id = shared_session_id
                if session_id is None:
                    response_status: int | None = None
                    try:
                        response = client.post(
                            f"{runtime_url}/api/chat/sessions",
                            json={"user_id": state.diagnostic_user_id},
                        )
                        response_status = response.status_code
                        session_id = _created_session(
                            response, state.diagnostic_user_id
                        )
                    except KeyboardInterrupt:
                        attempt = _session_failure_result(
                            query,
                            started_at=started_at,
                            started_clock=started_clock,
                            code="INTERRUPTED",
                            http_status=response_status,
                            scope="systemic",
                        )
                        _record_attempt(recorder, attempt)
                        print(format_query_terminal(attempt), flush=True)
                        raise
                    except httpx.HTTPError:
                        attempt = _session_failure_result(
                            query,
                            started_at=started_at,
                            started_clock=started_clock,
                            code="TRANSPORT_ERROR",
                            http_status=response_status,
                            scope="individual",
                        )
                        _record_attempt(recorder, attempt)
                        print(format_query_terminal(attempt), flush=True)
                        continue
                    except StreamContractError as exc:
                        scope = (
                            "systemic"
                            if response_status in {401, 403}
                            else "individual"
                        )
                        attempt = _session_failure_result(
                            query,
                            started_at=started_at,
                            started_clock=started_clock,
                            code=exc.code,
                            http_status=response_status,
                            scope=scope,
                        )
                        _record_attempt(recorder, attempt)
                        print(format_query_terminal(attempt), flush=True)
                        if scope == "systemic":
                            raise E2EDiagnosticError(
                                "E2E authentication failed during session creation"
                            )
                        continue
                try:
                    attempt = _execute_query(
                        client,
                        runtime_url,
                        state.diagnostic_user_id,
                        session_id,
                        query,
                        trace,
                        started_at=started_at,
                        started_clock=started_clock,
                    )
                except KeyboardInterrupt:
                    attempt = QueryAttemptResult(
                        query=query,
                        status="failed",
                        failure_scope="systemic",
                        failure_stage="interruption",
                        failure_code="INTERRUPTED",
                        question=query.question,
                        answer="",
                        answer_complete=False,
                        session_id=session_id,
                        request_id=None,
                        conversation_id=None,
                        diagnostic_operation_id=None,
                        http_status=None,
                        query_http_attempted=True,
                        token_event_count=0,
                        telemetry_schema_version=None,
                        timings_ms=None,
                        client_timings_ms=None,
                        trace_polling=None,
                        post_generation=None,
                        registry_verification=None,
                        deep_trace=None,
                        duration_ms=(perf_counter() - started_clock) * 1000.0,
                        started_at=started_at,
                    )
                    _record_attempt(recorder, attempt)
                    print(format_query_terminal(attempt), flush=True)
                    raise
                _record_attempt(recorder, attempt)
                print(format_query_terminal(attempt), flush=True)
                if attempt.failure_scope == "systemic":
                    raise E2EDiagnosticError(
                        f"Systemic E2E failure: {attempt.failure_code}"
                    )
        except BaseException as exc:
            phase_error = exc
            progress["trace_status"] = "failed"
        finally:
            session_was_started = trace.started
            try:
                trace.delete()
            except BaseException as cleanup_error:
                progress["trace_deleted"] = False
                if phase_error is not None:
                    cleanup_error.add_note(
                        "Phase 2D also failed before trace cleanup: "
                        + repr(phase_error)
                    )
                raise
            else:
                progress["trace_deleted"] = session_was_started
                if progress.get("trace_status") != "failed":
                    progress["trace_status"] = "succeeded"
        if phase_error is not None:
            raise phase_error
    if recorder.failed:
        raise E2EBatchFailed(
            f"{recorder.failed} of {recorder.attempted} E2E request(s) failed"
        )
    return physical_counts


__all__ = [
    "E2EBatchFailed",
    "DeepTraceContractError",
    "PostGenerationContractError",
    "QueryAttemptResult",
    "StreamContractError",
    "TaskObservation",
    "ValidatedStream",
    "format_query_terminal",
    "poll_task",
    "run_e2e_phase_2d",
    "validate_chat_trace_evidence",
    "validate_e2e_stream",
    "validate_post_generation_evidence",
    "verify_session_postcondition",
    "verify_physical_corpus",
]
