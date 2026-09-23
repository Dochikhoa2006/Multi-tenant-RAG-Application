"""Offline contract preflight and opt-in live-evidence acceptance checks."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any
from uuid import UUID, uuid4


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "evaluation" / "protected_contracts.json"
LATENCY_METRICS = ("ttft", "generation", "total_request")
MANDATORY_METRICS = frozenset(
    {"faithfulness", "response_relevancy", "context_utilization"}
)
DEFAULT_FIXTURES = ROOT / "diagnostics" / "fixtures" / "wizard"
DEFAULT_QUERIES = ROOT / "diagnostics" / "queries.py"
BEHAVIORAL_BASELINE_COMMIT = "fba3c3481d5dc438b6081ab85afcbaa99dbc2987"
APPROVED_INTEGRATION_COMMIT = "d37c82710a8bd166af4c68f5f57d1068a7cdc880"
APPROVED_EVIDENCE_HOOK_FILES = frozenset(
    {
        "backend/api/tasks.py",
        "backend/api/wizard_diagnostics.py",
        "backend/config.py",
        "backend/providers/sglang_query_rewriter.py",
        "backend/rag/generator.py",
        "backend/rag/retrieval.py",
        "backend/runtime_app.py",
        "backend/wizard/diagnostics.py",
        "deployment/e2e_diagnostic.py",
        "deployment/e2e_diagnostic_api.py",
        "deployment/evaluation_bridge.py",
        "deployment/modal_runtime.py",
        "deployment/ragctl.py",
    }
)
REQUIRED_PROTECTED_FILES = APPROVED_EVIDENCE_HOOK_FILES | frozenset({
    "backend/api/chat.py", "backend/api/models.py", "backend/api/telemetry.py",
    "backend/model_config.py", "backend/providers/sglang_qwen_llm.py",
    "backend/rag/embedder.py", "backend/rag/pipeline.py", "backend/rag/query_rewriter.py",
    "backend/rag/retrieval.py", "backend/rag/session_title.py", "backend/requirements.txt",
    "backend/task_queue.py", "backend/weaviate_client/_base.py",
    "backend/weaviate_client/_chunk_collection.py", "backend/weaviate_client/conversation.py",
    "backend/weaviate_client/knowledge.py", "backend/weaviate_client/policy.py",
    "deployment/compose.weaviate-secure.yaml", "deployment/evaluation_bridge_worker.py",
})
EVALUATION_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "source",
        "request_id",
        "conversation_id",
        "original_query",
        "rewritten_query",
        "response",
        "knowledge_contexts",
        "knowledge_context_ids",
        "policy_contexts",
        "policy_context_ids",
        "retrieved_contexts",
        "context_roles",
        "telemetry",
        "reference",
        "reference_context_ids",
        "captured_at",
    }
)


class GateError(RuntimeError):
    """A safe non-regression-gate failure."""


def _load_object(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise GateError(f"{name} must be one regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"{name} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise GateError(f"{name} must contain one JSON object")
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protected_contracts(
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, object]:
    """Fail before live work when protected source or literals drift."""

    manifest = _load_object(manifest_path, "protected-contract manifest")
    if manifest.get("schema_version") != "1.0":
        raise GateError("protected-contract manifest schema is unsupported")
    if (
        manifest.get("behavioral_baseline_commit") != BEHAVIORAL_BASELINE_COMMIT
        or manifest.get("approved_integration_commit") != APPROVED_INTEGRATION_COMMIT
    ):
        raise GateError("protected commit identity drifted")
    files = manifest.get("files")
    contracts = manifest.get("contracts")
    hooks = manifest.get("approved_evidence_hook_files")
    baseline_files = manifest.get("behavioral_baseline_files")
    if (
        not isinstance(files, Mapping)
        or frozenset(files) != REQUIRED_PROTECTED_FILES
        or not isinstance(contracts, Mapping)
        or not isinstance(hooks, Mapping)
        or frozenset(hooks) != APPROVED_EVIDENCE_HOOK_FILES
        or not all(isinstance(value, str) and value for value in hooks.values())
        or not isinstance(baseline_files, Mapping)
    ):
        raise GateError("protected-contract manifest is incomplete")
    for relative, expected in baseline_files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise GateError("behavioral-baseline file entry is malformed")
    checked: list[str] = []
    for relative, expected in sorted(files.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise GateError("protected file entry is malformed")
        unresolved = ROOT / relative
        if unresolved.is_symlink():
            raise GateError(f"protected source is a symlink: {relative}")
        path = unresolved.resolve()
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise GateError("protected file escapes the repository") from exc
        if not path.is_file() or path.is_symlink() or _digest(path) != expected:
            raise GateError(f"protected contract drifted: {relative}")
        checked.append(relative)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend.api.models import QueryRequest, TaskResource
    from backend.api.telemetry import TELEMETRY_SCHEMA_VERSION, TIMING_KEYS
    from backend.model_config import (
        FIXED_CONVERSATION_CANDIDATE_COUNT,
        FIXED_KNOWLEDGE_CANDIDATE_COUNT,
        FIXED_POLICY_CANDIDATE_COUNT,
        GRANITE_QUERY_REWRITE,
        QWEN_SGLANG,
        SGLANG_QUERY_REWRITE,
        TOKEN_BUDGETS,
    )
    from backend.providers.sglang_query_rewriter import SGLangGraniteQueryRewriter
    from backend.providers.sglang_qwen_llm import SGLangQwenLLMClient
    from deployment.ragctl import CHAT_TIMING_KEYS, iter_sse, validate_chat_result

    expected_keys = tuple(contracts.get("timing_keys", ()))
    if TELEMETRY_SCHEMA_VERSION != contracts.get("telemetry_schema_version"):
        raise GateError("telemetry schema drifted")
    if TIMING_KEYS != expected_keys:
        raise GateError("telemetry timing keys drifted")
    if sorted(QueryRequest.model_fields) != contracts.get("query_request_fields"):
        raise GateError("QueryRequest fields drifted")
    if sorted(TaskResource.model_fields) != contracts.get("task_resource_fields"):
        raise GateError("TaskResource fields drifted")
    ceilings = contracts.get("candidate_ceilings")
    if ceilings != {
        "conversation": FIXED_CONVERSATION_CANDIDATE_COUNT,
        "knowledge": FIXED_KNOWLEDGE_CANDIDATE_COUNT,
        "policy": FIXED_POLICY_CANDIDATE_COUNT,
    }:
        raise GateError("retrieval candidate ceilings drifted")
    budgets = contracts.get("context_budgets")
    if budgets != {
        "knowledge": TOKEN_BUDGETS.knowledge_tokens,
        "policy": TOKEN_BUDGETS.policy_tokens,
        "total": TOKEN_BUDGETS.total_context_tokens,
    }:
        raise GateError("context budgets drifted")
    qwen = object.__new__(SGLangQwenLLMClient)
    qwen.config = QWEN_SGLANG
    qwen_payload = qwen._payload(
        "protected-contract",
        model=QWEN_SGLANG.served_model,
        reasoning="low",
        max_output_tokens=QWEN_SGLANG.answer_max_output_tokens,
        stream=True,
    )
    if contracts.get("qwen") != {
        "temperature": qwen_payload["temperature"],
        "top_p": qwen_payload["top_p"],
        "top_k": qwen_payload["top_k"],
        "min_p": qwen_payload["min_p"],
        "presence_penalty": qwen_payload["presence_penalty"],
        "thinking": qwen_payload["chat_template_kwargs"]["enable_thinking"],
    }:
        raise GateError("Qwen generation settings drifted")
    granite = object.__new__(SGLangGraniteQueryRewriter)
    granite.granite_config = GRANITE_QUERY_REWRITE
    granite.sglang_config = SGLANG_QUERY_REWRITE
    granite_body = granite._request_body([{"role": "user", "content": "probe"}])
    if contracts.get("granite") != {
        key: granite_body[key]
        for key in ("temperature", "n", "stream", "continue_final_message")
    }:
        raise GateError("Granite request settings drifted")
    request_id = "00000000-0000-4000-8000-000000000001"
    conversation_id = "00000000-0000-4000-8000-000000000002"
    telemetry = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "request_id": request_id,
        "timings_ms": {name: 0.0 for name in CHAT_TIMING_KEYS},
    }
    lines = [
        "event: token", json.dumps({"request_id": request_id, "text": "x"}), "",
        "event: telemetry", json.dumps(telemetry), "",
        "event: done", json.dumps(
            {"request_id": request_id, "conversation_id": conversation_id}
        ), "",
    ]
    frames = list(iter_sse(
        line if not line.startswith("{") else "data: " + line for line in lines
    ))
    events = [name for name, _ in frames]
    validate_chat_result(
        events,
        [str(frames[0][1]["text"])],
        frames[1][1],
        frames[2][1],
        verify_atlas_grounding=False,
    )
    if events != ["token", "telemetry", "done"] or contracts.get(
        "sse_success_order"
    ) != "token* -> telemetry -> done":
        raise GateError("SSE success ordering drifted")
    return {
        "status": "passed",
        "manifest_schema_version": "1.0",
        "behavioral_baseline_commit": manifest.get("behavioral_baseline_commit"),
        "approved_integration_commit": manifest.get("approved_integration_commit"),
        "checked_file_count": len(checked),
    }


def validate_live_prerequisites(
    *,
    env_path: Path,
    fixtures_path: Path,
    queries_path: Path,
    corpus_state_path: Path,
) -> dict[str, object]:
    """Read-only preflight for a later, separately authorized real run."""

    protected = validate_protected_contracts()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from deployment.e2e_diagnostic import (
        load_query_selection,
        validate_reusable_corpus_state,
    )
    from deployment.ragctl import load_dotenv, validate_config
    from deployment.wizard_diagnostic import (
        load_corpus_state,
    )

    try:
        config = load_dotenv(env_path)
        validate_config(config)
        rag_user = config["RAG_USER_ID"]
        selection = load_query_selection(queries_path, ROOT, start=0, limit=None)
        state = load_corpus_state(corpus_state_path, rag_user)
        active = validate_reusable_corpus_state(state, rag_user)
    except Exception as exc:
        raise GateError("live configuration, query, or corpus prerequisite failed") from exc
    if state is None or state.diagnostic_user_id != rag_user:
        raise GateError("settled corpus owner must match RAG_USER_ID")
    fixture_counts: dict[str, int] = {}
    for collection in ("knowledge", "policy"):
        folder = fixtures_path / collection
        if not folder.is_dir() or folder.is_symlink():
            raise GateError(f"live {collection} fixture folder is unavailable")
        files = [
            path
            for path in sorted(folder.iterdir(), key=lambda item: item.name)
            if path.is_file() and not path.is_symlink() and not path.name.startswith(".")
        ]
        if not files:
            raise GateError(f"live {collection} fixtures are empty")
        fixture_counts[collection] = len(files)
    evaluator = ROOT / "evaluation" / ".venv" / "bin" / "rag-evaluate"
    if not evaluator.is_file() or not os.access(evaluator, os.X_OK):
        raise GateError("isolated evaluator environment is unavailable")
    return {
        "status": "passed",
        "protected_contracts": protected,
        "corpus_owner_is_rag_user": True,
        "query_count": len(selection.selected),
        "fixture_counts": fixture_counts,
        "active_generation": active.generation_id,
        "evaluator_available": True,
    }


def _canonical_uuid(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise GateError(f"{name} is missing")
    try:
        canonical = str(UUID(value))
    except ValueError as exc:
        raise GateError(f"{name} is malformed") from exc
    if canonical != value:
        raise GateError(f"{name} is not canonical")
    return canonical


def validate_evaluation_success(
    observation: Mapping[str, object],
    *,
    expected_request: Mapping[str, object] | None = None,
    expected_source: str | None = None,
) -> dict[str, object]:
    """Validate a real successful reference-free Ragas result and its record."""

    if observation.get("status") != "succeeded":
        raise GateError("live evaluation did not succeed")
    record_value = observation.get("record_path")
    result_value = observation.get("result_path")
    expected_hash = observation.get("record_sha256")
    if not all(isinstance(value, str) and value for value in (record_value, result_value)):
        raise GateError("live evaluation artifact paths are missing")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise GateError("live evaluation record hash is malformed")
    record_path = Path(str(record_value))
    result_path = Path(str(result_value))
    record = _load_object(record_path, "evaluation record")
    result = _load_object(result_path, "evaluation result")
    if _digest(record_path) != expected_hash:
        raise GateError("evaluation record hash does not match bytes")
    if result.get("schema_version") != "1.0" or result.get("status") != "succeeded":
        raise GateError("evaluation result is not a successful schema-1.0 result")
    if result.get("record_sha256") != expected_hash:
        raise GateError("evaluation result record hash is inconsistent")
    if (
        frozenset(record) != EVALUATION_RECORD_FIELDS
        or record.get("schema_version") != "1.0"
        or record.get("source") not in {"e2e", "rag_ask"}
        or (expected_source is not None and record.get("source") != expected_source)
    ):
        raise GateError("evaluation record contract is invalid")
    captured_at = record.get("captured_at")
    if not isinstance(captured_at, str):
        raise GateError("evaluation captured_at is invalid")
    try:
        parsed_captured_at = datetime.fromisoformat(
            captured_at[:-1] + "+00:00" if captured_at.endswith("Z") else captured_at
        )
    except ValueError as exc:
        raise GateError("evaluation captured_at is invalid") from exc
    if parsed_captured_at.tzinfo is None or parsed_captured_at.utcoffset() is None:
        raise GateError("evaluation captured_at is invalid")
    for key in ("source", "request_id", "conversation_id"):
        if result.get(key) != record.get(key):
            raise GateError(f"evaluation {key} correlation is inconsistent")
    _canonical_uuid(record.get("request_id"), "evaluation request ID")
    _canonical_uuid(record.get("conversation_id"), "evaluation conversation ID")
    for key in ("original_query", "rewritten_query", "response"):
        if not isinstance(record.get(key), str) or not str(record[key]).strip():
            raise GateError(f"evaluation record {key} is invalid")
    knowledge = record.get("knowledge_contexts")
    policy = record.get("policy_contexts")
    knowledge_ids = record.get("knowledge_context_ids")
    policy_ids = record.get("policy_context_ids")
    retrieved = record.get("retrieved_contexts")
    roles = record.get("context_roles")
    if not all(
        isinstance(value, list)
        for value in (knowledge, policy, knowledge_ids, policy_ids, retrieved, roles)
    ):
        raise GateError("evaluation context evidence is malformed")
    if (
        len(knowledge) != len(knowledge_ids)
        or len(policy) != len(policy_ids)
        or retrieved != knowledge + policy
        or roles != ["knowledge"] * len(knowledge) + ["policy"] * len(policy)
    ):
        raise GateError("evaluation context order is inconsistent")
    for identifier in knowledge_ids + policy_ids:
        _canonical_uuid(identifier, "evaluation context ID")
    telemetry = record.get("telemetry")
    if (
        not isinstance(telemetry, Mapping)
        or telemetry.get("schema_version") != "1.0"
        or not isinstance(telemetry.get("timings_ms"), Mapping)
    ):
        raise GateError("evaluation telemetry is invalid")
    if expected_request is not None:
        expected = {
            "original_query": expected_request.get("question"),
            "response": expected_request.get("answer"),
            "request_id": expected_request.get("request_id"),
            "conversation_id": expected_request.get("conversation_id"),
            "telemetry": expected_request.get("telemetry"),
        }
        for key, value in expected.items():
            if value is not None and record.get(key) != value:
                raise GateError(f"evaluation record does not match SSE {key}")
        _validate_record_evidence(record, expected_request)
    if not isinstance(result.get("ragas_version"), str):
        raise GateError("Ragas version is missing")
    judge = result.get("judge")
    if not isinstance(judge, Mapping):
        raise GateError("judge identity is missing")
    judge_model = judge.get("model")
    normalized_judge_model = (
        ""
        if not isinstance(judge_model, str)
        else judge_model.strip().casefold().rsplit("/", 1)[-1].split(":", 1)[0]
    )
    if (
        judge.get("provider") != "ollama"
        or not isinstance(judge_model, str)
        or not judge_model.strip()
        or normalized_judge_model == "qwen3-4b-awq"
        or not isinstance(judge.get("model_digest"), str)
        or not str(judge["model_digest"]).strip()
        or not isinstance(judge.get("ollama_version"), str)
        or not str(judge["ollama_version"]).strip()
    ):
        raise GateError("local Ollama judge identity is invalid")
    metrics = result.get("metrics")
    if not isinstance(metrics, list):
        raise GateError("evaluation metrics are missing")
    seen: set[str] = set()
    for metric in metrics:
        if not isinstance(metric, Mapping) or not isinstance(metric.get("name"), str):
            raise GateError("evaluation metric is malformed")
        name = str(metric["name"])
        if name in seen:
            raise GateError("evaluation metric is duplicated")
        seen.add(name)
        if name in MANDATORY_METRICS:
            score = metric.get("score")
            if (
                metric.get("status") != "succeeded"
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise GateError(f"mandatory metric failed: {name}")
    if not MANDATORY_METRICS.issubset(seen):
        raise GateError("mandatory reference-free metrics are incomplete")
    return {
        "status": "passed",
        "request_id": record["request_id"],
        "conversation_id": record["conversation_id"],
        "record_sha256": expected_hash,
        "mandatory_metrics": sorted(MANDATORY_METRICS),
    }


def _validate_record_evidence(
    record: Mapping[str, object], expected_request: Mapping[str, object]
) -> None:
    trace_value = expected_request.get("deep_trace")
    if trace_value is None:
        trace_value = expected_request.get("evaluation_evidence")
    trace = cast_mapping(trace_value)
    operation = cast_mapping(trace.get("operation"))
    session_id = trace.get("session_id")
    user_id = operation.get("user_id")
    operation_id = operation.get("operation_id")
    chat_session_id = expected_request.get("session_id")
    request_id = expected_request.get("request_id")
    if not all(
        isinstance(value, str)
        for value in (session_id, user_id, operation_id, chat_session_id, request_id)
    ):
        raise GateError("live trace correlation is incomplete")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend.wizard.diagnostics import framed_content_digest
    from deployment.evaluation_bridge import parse_request_evidence

    try:
        evidence = parse_request_evidence(
            operation,
            user_id=str(user_id),
            trace_session_id=str(session_id),
            operation_id=str(operation_id),
            chat_session_id=str(chat_session_id),
            request_id=str(request_id),
        )
    except Exception as exc:
        raise GateError("live trace evidence is invalid") from exc
    if record.get("rewritten_query") != evidence.rewritten_query:
        raise GateError("live rewritten-query evidence mismatch")

    for name, identity in (("knowledge", evidence.knowledge), ("policy", evidence.policy)):
        ids = record.get(f"{name}_context_ids")
        contexts = record.get(f"{name}_contexts")
        if not isinstance(ids, list) or not isinstance(contexts, list):
            raise GateError(f"live {name} record evidence is malformed")
        if tuple(ids) != identity.ids or len(contexts) != len(ids):
            raise GateError(f"live {name} context identity/order mismatch")
        rendered_bytes = max(0, len(contexts) - 1) * 2
        digest_parts: list[bytes] = []
        for identifier, text, expected_fingerprint in zip(
            ids, contexts, identity.fingerprints, strict=True
        ):
            if not isinstance(identifier, str) or not isinstance(text, str):
                raise GateError(f"live {name} context is malformed")
            identifier_bytes = identifier.encode("utf-8")
            text_bytes = text.encode("utf-8")
            fingerprint = framed_content_digest(
                "chat-kp-item-v1", (identifier_bytes, text_bytes)
            )
            if expected_fingerprint != f"{identifier}:{fingerprint}":
                raise GateError(f"live {name} fingerprint mismatch")
            rendered_bytes += len(text_bytes)
            digest_parts.extend((identifier_bytes, text_bytes))
        digest = framed_content_digest(
            f"chat-qwen-{name}-context-v1", digest_parts
        )
        if digest != identity.digest or rendered_bytes != identity.rendered_utf8_bytes:
            raise GateError(f"live {name} digest mismatch")


def _finite_positive(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise GateError(f"{name} must be finite and positive")
    return float(value)


def paired_latency_gate(
    samples: Sequence[Mapping[str, object]],
    *,
    baseline_mode: str = "off",
    comparison_mode: str = "on",
    minimum_pairs: int = 30,
    resamples: int = 10_000,
    seed: int = 20_260_911,
    threshold_percent: float = 5.0,
) -> dict[str, object]:
    """Apply the deterministic paired-median one-sided latency gate."""

    if minimum_pairs < 1 or resamples < 1:
        raise GateError("latency pair and bootstrap counts must be positive")
    indexed: dict[str, dict[str, Mapping[str, object]]] = {}
    for sample in samples:
        pair_key = sample.get("pair_key")
        mode = sample.get("mode")
        if not isinstance(pair_key, str) or not pair_key:
            raise GateError("latency sample pair_key is invalid")
        if mode not in {baseline_mode, comparison_mode}:
            raise GateError("latency sample mode is invalid")
        bucket = indexed.setdefault(pair_key, {})
        if str(mode) in bucket:
            raise GateError("duplicate latency sample for one pair and mode")
        bucket[str(mode)] = sample
    complete = [
        pair for pair in indexed.values()
        if baseline_mode in pair and comparison_mode in pair
    ]
    if len(complete) != len(indexed):
        raise GateError("unpaired latency observations cannot be discarded")
    if len(complete) < minimum_pairs:
        raise GateError("insufficient complete paired latency observations")
    for pair in complete:
        if _pair_identity(pair[baseline_mode]) != _pair_identity(
            pair[comparison_mode]
        ):
            raise GateError("paired latency observations identify different queries")
    rng = random.Random(seed)
    decisions: dict[str, object] = {}
    for metric in LATENCY_METRICS:
        changes: list[float] = []
        for pair in complete:
            before = _finite_positive(
                cast_mapping(pair[baseline_mode].get("timings_ms")).get(metric),
                f"{baseline_mode} {metric}",
            )
            after = _finite_positive(
                cast_mapping(pair[comparison_mode].get("timings_ms")).get(metric),
                f"{comparison_mode} {metric}",
            )
            changes.append(((after - before) / before) * 100.0)
        bootstrap = [
            statistics.median(rng.choice(changes) for _ in changes)
            for _ in range(resamples)
        ]
        bootstrap.sort()
        upper = bootstrap[math.ceil(0.95 * len(bootstrap)) - 1]
        decisions[metric] = {
            "paired_median_change_percent": round(statistics.median(changes), 6),
            "one_sided_95pct_upper_bound_percent": round(upper, 6),
            "status": "passed" if upper < threshold_percent else "failed",
        }
    return {
        "schema_version": "1.0",
        "status": (
            "passed"
            if all(value["status"] == "passed" for value in decisions.values())
            else "failed"
        ),
        "complete_pair_count": len(complete),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "threshold_percent": threshold_percent,
        "metrics": decisions,
    }


def cast_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GateError("latency timings_ms is malformed")
    return value


def _pair_identity(sample: Mapping[str, object]) -> tuple[object, ...]:
    question = sample.get("question")
    query_identity = sample.get("query_identity")
    values = tuple(
        sample.get(name)
        for name in ("query_source_index", "query_schedule_index", "query_repetition")
    )
    if (
        not isinstance(question, str)
        or not question.strip()
        or not isinstance(query_identity, str)
        or len(query_identity) != 64
        or any(type(value) is not int or value < 0 for value in values)
    ):
        raise GateError("paired query identity is incomplete")
    return (question, query_identity, *values)


def validate_contention_sample(sample: Mapping[str, object]) -> None:
    """Require the RAG request to start during an actual judge request."""

    activity = cast_mapping(sample.get("judge_activity"))
    donor = cast_mapping(sample.get("overlapping_evaluation_request"))
    evaluation = cast_mapping(donor.get("evaluation"))
    for key in ("evaluation_job_id", "request_id", "record_sha256"):
        expected = evaluation.get(key) if key != "request_id" else donor.get(key)
        if activity.get(key) != expected:
            raise GateError("judge activity does not belong to the donor evaluation")
    if (
        activity.get("success") is not True
        or type(activity.get("sequence")) is not int
        or activity["sequence"] < 1
        or not isinstance(activity.get("judge_id"), str)
        or not activity["judge_id"]
    ):
        raise GateError("judge activity evidence is incomplete")
    judge_started = _finite_positive(activity.get("started_monotonic"), "judge start")
    request_started = _finite_positive(sample.get("request_started_monotonic"), "request start")
    request_done = _finite_positive(sample.get("request_done_monotonic"), "request done")
    judge_finished = _finite_positive(activity.get("finished_monotonic"), "judge finish")
    if not judge_started < request_started < min(request_done, judge_finished):
        raise GateError("local judge did not overlap the RAG request")


def _validate_post_generation(row: Mapping[str, object]) -> None:
    post = cast_mapping(row.get("post_generation"))
    tasks = []
    for name in ("conversation_persistence", "session_title"):
        task = cast_mapping(post.get(name))
        if task.get("status") != "succeeded":
            raise GateError("live post-generation work failed")
        _canonical_uuid(task.get("task_id"), "post-generation task ID")
        dates = []
        for key in ("created_at", "started_at", "finished_at"):
            value = task.get(key)
            if not isinstance(value, str):
                raise GateError("post-generation timestamps are missing")
            try:
                date = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise GateError("post-generation timestamp is malformed") from exc
            if date.tzinfo is None:
                raise GateError("post-generation timestamp lacks timezone")
            dates.append(date)
        if not dates[0] <= dates[1] <= dates[2]:
            raise GateError("post-generation timestamps are inconsistent")
        tasks.append(dates)
    if tasks[0][2] > tasks[1][1]:
        raise GateError("post-generation tasks violated FIFO")


def _validate_live_sample(sample: Mapping[str, object], *, evaluation: bool) -> None:
    if sample.get("status") != "succeeded" or sample.get("answer_complete") is not True:
        raise GateError("live measured RAG request did not succeed")
    _canonical_uuid(sample.get("request_id"), "live sample request ID")
    _canonical_uuid(sample.get("conversation_id"), "live sample conversation ID")
    answer = sample.get("answer")
    if not isinstance(answer, str) or not answer:
        raise GateError("live answer is missing")
    if sample.get("answer_sha256") != hashlib.sha256(answer.encode("utf-8")).hexdigest():
        raise GateError("live answer digest is inconsistent")
    telemetry = cast_mapping(sample.get("telemetry"))
    if (
        telemetry.get("schema_version") != "1.0"
        or telemetry.get("request_id") != sample.get("request_id")
        or telemetry.get("timings_ms") != sample.get("timings_ms")
    ):
        raise GateError("live timing sample differs from authoritative telemetry")
    if sample.get("acceptance_experiment_sha256") != sample.get(
        "configuration_sha256"
    ):
        raise GateError("runtime acceptance experiment identity is wrong")
    _validate_post_generation(sample)
    if evaluation:
        validate_evaluation_success(
            cast_mapping(sample.get("evaluation")), expected_request=sample
        )


def validate_latency_schedule(
    samples: Sequence[Mapping[str, object]], *, contention: bool = False
) -> list[Mapping[str, object]]:
    """Validate run provenance/schedule before calculating any performance PASS.

    The local authorized collector must add this evidence to its raw samples;
    plain timing triples cannot stand in for a completed live acceptance run.
    """
    if not samples:
        raise GateError("live latency samples are missing")
    identities = set()
    seen_requests = set()
    blocks: dict[int, list[Mapping[str, object]]] = {}
    for sample in samples:
        identity = []
        for key in ("protected_contract_sha256", "configuration_sha256", "corpus_sha256", "models_sha256"):
            value = sample.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise GateError("live runtime/configuration identity is missing")
            identity.append(value)
        if sample["protected_contract_sha256"] != _digest(MANIFEST_PATH):
            raise GateError("live sample used different protected contracts")
        identities.add(tuple(identity))
        if sample.get("request_id") in seen_requests:
            raise GateError("live request is duplicated")
        seen_requests.add(sample.get("request_id"))
        mode = sample.get("mode")
        observed = sample.get("evaluation_evidence_enabled")
        expected_enabled = mode != "off"
        if type(observed) is not bool or observed is not expected_enabled:
            raise GateError("authoritative runtime evidence-toggle state is wrong")
        if not isinstance(sample.get("runtime_instance_id"), str) or not sample.get("runtime_instance_id"):
            raise GateError("runtime instance identity is missing")
        if not isinstance(sample.get("runtime_worker_id"), str) or not sample.get("runtime_worker_id"):
            raise GateError("runtime worker identity is missing")
        if contention:
            if mode not in {"uncontended", "contended"}:
                raise GateError("contention mode is invalid")
            if mode == "contended":
                if sample.get("judge_activity_source") != "ollama_http":
                    raise GateError("actual Ollama request activity is required")
                validate_contention_sample(sample)
                other = cast_mapping(sample.get("overlapping_evaluation_request"))
                if other.get("request_id") == sample.get("request_id"):
                    raise GateError("contention must involve another request")
                _validate_live_sample(other, evaluation=True)
            else:
                idle_order = tuple(_finite_positive(sample.get(key), key) for key in (
                    "idle_reservation_acquired_monotonic",
                    "request_started_monotonic", "done_validated_monotonic",
                    "idle_reservation_released_monotonic",
                ))
                if sample.get("evaluator_idle_reserved") is not True or list(
                    idle_order
                ) != sorted(idle_order):
                    raise GateError("uncontended request lacked an idle evaluator reservation")
            _validate_live_sample(sample, evaluation=True)
        else:
            block = sample.get("block_sequence")
            if type(block) is not int or block not in range(4):
                raise GateError("OFF/ON block sequence is invalid")
            if mode != ("off", "on", "on", "off")[block]:
                raise GateError("OFF/ON schedule must be OFF ON ON OFF")
            blocks.setdefault(block, []).append(sample)
            _validate_live_sample(sample, evaluation=mode == "on")
    if len(identities) != 1:
        raise GateError("live runtime/configuration drifted between samples")
    if contention:
        pairs: dict[str, dict[str, Mapping[str, object]]] = {}
        for sample in samples:
            key, mode = sample.get("pair_key"), str(sample.get("mode"))
            if not isinstance(key, str) or mode in pairs.setdefault(key, {}):
                raise GateError("contention pair is missing or duplicated")
            pairs[key][mode] = sample
        for pair in pairs.values():
            if set(pair) != {"uncontended", "contended"} or _pair_identity(
                pair["uncontended"]
            ) != _pair_identity(pair["contended"]):
                raise GateError("contention pair identifies different queries")
        return list(samples)
    if set(blocks) != set(range(4)):
        raise GateError("all four fresh-runtime blocks are required")
    if [sample["block_sequence"] for sample in samples] != sorted(sample["block_sequence"] for sample in samples):
        raise GateError("live blocks are not in execution order")
    runtime_ids = []
    selected = []
    for block in range(4):
        group = blocks[block]
        runtime_id = group[0].get("runtime_instance_id")
        if not isinstance(runtime_id, str) or not runtime_id or any(item.get("runtime_instance_id") != runtime_id for item in group):
            raise GateError("fresh-runtime identity is missing or inconsistent")
        runtime_ids.append(runtime_id)
        if len(group) <= 3:
            raise GateError("three warmups plus measured observations are required")
        for index, sample in enumerate(group):
            if sample.get("block_request_index") != index or sample.get("warmup") is not (index < 3):
                raise GateError("live warmup schedule is inconsistent")
            if index >= 3:
                selected.append(sample)
    if len(set(runtime_ids)) != 4:
        raise GateError("runtime blocks must be independently started")
    paired_questions: dict[str, str] = {}
    for sample in selected:
        question = sample.get("question")
        key = sample.get("pair_key")
        if not isinstance(question, str) or not question.strip() or not isinstance(key, str):
            raise GateError("paired live query schedule is missing")
        if key in paired_questions and paired_questions[key] != question:
            raise GateError("paired query text differs")
        paired_questions[key] = question
    measured_schedules = []
    for block in range(4):
        measured_schedules.append(
            [_pair_identity(item) for item in blocks[block] if item.get("warmup") is False]
        )
    if any(schedule != measured_schedules[0] for schedule in measured_schedules[1:]):
        raise GateError("fresh-runtime blocks used different ordered query schedules")
    return selected


def verify_live_artifacts(
    *,
    ask_observation: Mapping[str, object],
    e2e_rows: Sequence[Mapping[str, object]],
    latency_samples: Sequence[Mapping[str, object]],
    contention_samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate authorized real-run artifacts without making service calls."""

    protected = validate_protected_contracts()
    for name in ("question", "answer", "request_id", "conversation_id", "telemetry"):
        if ask_observation.get(name) is None:
            raise GateError(f"live ask {name} evidence is missing")
    ask_evaluation = ask_observation.get("evaluation", ask_observation)
    if not isinstance(ask_evaluation, Mapping):
        raise GateError("live ask evaluation evidence is missing")
    ask = validate_evaluation_success(
        ask_evaluation,
        expected_request=ask_observation,
        expected_source="rag_ask",
    )
    _validate_post_generation(ask_observation)
    if not e2e_rows:
        raise GateError("live E2E evidence is missing")
    for row in e2e_rows:
        if row.get("status") != "succeeded":
            raise GateError("live E2E request failed")
        _validate_post_generation(row)
        evaluation = row.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise GateError("live E2E evaluation evidence is missing")
        validate_evaluation_success(evaluation, expected_request=row, expected_source="e2e")
    off_on = paired_latency_gate(validate_latency_schedule(latency_samples))
    contention = paired_latency_gate(
        validate_latency_schedule(contention_samples, contention=True),
        baseline_mode="uncontended",
        comparison_mode="contended",
    )
    statuses = {
        "protected_contracts": protected["status"],
        "live_ask": "passed",
        "live_e2e": "passed",
        "off_on_latency": off_on["status"],
        "evaluator_overlap_latency": contention["status"],
        "real_ragas": "passed",
    }
    return {
        "schema_version": "1.0",
        "status": "passed" if set(statuses.values()) == {"passed"} else "failed",
        "statuses": statuses,
        "ask": ask,
        "e2e_request_count": len(e2e_rows),
        "off_on_latency": off_on,
        "evaluator_overlap_latency": contention,
    }


def _load_jsonl(path: Path) -> list[Mapping[str, object]]:
    if not path.is_file() or path.is_symlink():
        raise GateError("JSONL evidence must be one regular file")
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateError(f"invalid JSONL row {line_number}") from exc
        if not isinstance(value, Mapping):
            raise GateError(f"JSONL row {line_number} is not an object")
        rows.append(value)
    return rows


def _write_private(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _live_inputs(args: argparse.Namespace) -> dict[str, object]:
    validate_live_prerequisites(
        env_path=args.env, fixtures_path=args.fixtures,
        queries_path=args.queries, corpus_state_path=args.corpus_state,
    )
    from deployment.e2e_diagnostic import load_query_selection
    from deployment.ragctl import load_dotenv
    from deployment.wizard_diagnostic import load_corpus_state

    config = load_dotenv(args.env)
    user = config["RAG_USER_ID"].strip()
    selection = load_query_selection(args.queries, ROOT)
    state = load_corpus_state(args.corpus_state, user)
    stable = {key: value for key, value in config.items() if key != "RAG_EVALUATION_EVIDENCE_ENABLED"}
    contracts = _load_object(MANIFEST_PATH, "protected-contract manifest")["contracts"]
    return {
        "config": config, "user": user, "selection": selection,
        "configuration_sha256": _canonical_digest(stable),
        "corpus_sha256": _digest(args.corpus_state),
        "models_sha256": _canonical_digest(
            {"qwen": contracts["qwen"], "granite": contracts["granite"]}
        ),
        "state": state,
    }


def _runtime_instance(runner: object) -> str:
    from deployment.ragctl import RUNTIME_APP, target_containers
    matches = [
        item for item in target_containers(runner)
        if item.get("app_name") == RUNTIME_APP and item.get("start_time") != "Pending"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("container_id"), str):
        raise GateError("one authoritative runtime container is required")
    return str(matches[0]["container_id"])


def _post_generation(config: Mapping[str, str], row: Mapping[str, object]) -> dict[str, object]:
    import httpx
    from deployment.e2e_diagnostic_api import poll_task
    from deployment.ragctl import _runtime_headers, read_runtime_url

    runtime = read_runtime_url(config)
    params = {key: row[key] for key in ("user_id", "session_id", "conversation_id")}
    deadline = time.monotonic() + 900
    with httpx.Client(headers=_runtime_headers(config), timeout=120) as client:
        while True:
            response = client.get(
                f"{runtime}/api/tasks/_acceptance/post-generation", params=params
            )
            if response.status_code in {401, 403}:
                raise GateError("task-proof authentication failed")
            if response.status_code == 200:
                payload = response.json()
                tasks = payload.get("tasks") if isinstance(payload, Mapping) else None
                if isinstance(tasks, list) and len(tasks) == 2 and all(
                    isinstance(item, Mapping) and item.get("status") in {"succeeded", "failed"}
                    for item in tasks
                ):
                    break
            if time.monotonic() >= deadline:
                raise GateError("post-generation task proof timed out")
            time.sleep(1)
        by_operation = {str(item["operation"]): dict(item) for item in tasks}
        if (
            payload.get("schema_version") != "1.0"
            or not isinstance(payload.get("acceptance_experiment_sha256"), str)
            or len(payload["acceptance_experiment_sha256"]) != 64
        ):
            raise GateError("post-generation runtime provenance is malformed")
        deleted = client.delete(
            f"{runtime}/api/chat/sessions/{row['session_id']}",
            params={"user_id": row["user_id"]},
        )
        if deleted.status_code != 202:
            raise GateError("known-session cleanup was not accepted")
        deletion = poll_task(
            client, runtime, str(row["user_id"]),
            str(deleted.json()["task_id"]), "delete_session",
        )
        if deletion.status != "succeeded":
            raise GateError("known-session cleanup failed")
    return {
        "post_generation": {
            "conversation_persistence": by_operation["embed_conversation"],
            "session_title": by_operation["generate_session_title"],
        },
        "runtime_worker_id": payload["runtime_worker_id"],
        "evaluation_evidence_enabled": payload["evaluation_evidence_enabled"],
        "acceptance_experiment_sha256": payload["acceptance_experiment_sha256"],
        "session_cleanup": deletion.artifact(),
    }


def _ask_process(
    config: Mapping[str, str],
    question: str,
    activity: str | None,
    done_validated_hook: Callable[[], None] | None = None,
) -> Mapping[str, object]:
    from deployment.ragctl import ask
    return ask(
        config, question,
        acceptance_activity_path=None if activity is None else Path(activity),
        _acceptance_done_validated_hook=done_validated_hook,
    )


def _ask_sample(
    config: Mapping[str, str], user: str, question: str,
    *, activity: Path | None = None, reserve_evaluator_idle: bool = False,
) -> dict[str, object]:
    descriptor: int | None = None
    acquired: float | None = None
    released: list[float] = []
    hook: Callable[[], None] | None = None
    if reserve_evaluator_idle:
        from deployment.evaluation_bridge import LOCAL_EVALUATION_LOCK_PATH
        LOCAL_EVALUATION_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(LOCAL_EVALUATION_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        acquired = time.monotonic()
        def release() -> None:
            if not released:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                released.append(time.monotonic())
        hook = release
    try:
        row = dict(_ask_process(config, question, None if activity is None else str(activity), hook))
    finally:
        if descriptor is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    if reserve_evaluator_idle:
        if acquired is None or not released:
            raise GateError("idle evaluator reservation was not released after SSE done")
        row.update({"evaluator_idle_reserved": True,
                    "idle_reservation_acquired_monotonic": acquired,
                    "idle_reservation_released_monotonic": released[0]})
    row["user_id"] = user
    row.update(_post_generation(config, row))
    return row


def _activity_intervals(path: Path) -> list[dict[str, object]]:
    events = _load_jsonl(path)
    starts: dict[tuple[object, object], Mapping[str, object]] = {}
    intervals = []
    for event in events:
        key = (event.get("evaluation_job_id"), event.get("sequence"))
        if event.get("event") == "start":
            starts[key] = event
        elif event.get("event") == "end" and key in starts:
            start = starts.pop(key)
            intervals.append({
                "evaluation_job_id": key[0], "sequence": key[1],
                "request_id": event.get("request_id"),
                "record_sha256": event.get("record_sha256"),
                "judge_id": event.get("judge_id"),
                "started_monotonic": start.get("monotonic"),
                "finished_monotonic": event.get("monotonic"),
                "success": event.get("success"),
            })
    return intervals


def _identity_fields(inputs: Mapping[str, object]) -> dict[str, object]:
    return {
        "protected_contract_sha256": _digest(MANIFEST_PATH),
        "configuration_sha256": inputs["configuration_sha256"],
        "corpus_sha256": inputs["corpus_sha256"],
        "models_sha256": inputs["models_sha256"],
    }


def _validate_functional_dependency(path: Path, inputs: Mapping[str, object]) -> None:
    evidence = _load_object(path, "functional acceptance evidence")
    if evidence.get("status") != "passed" or evidence.get("identities") != _identity_fields(inputs):
        raise GateError("Phase 1 functional acceptance evidence is missing or mismatched")


def _query_fields(selection: object, index: int) -> dict[str, object]:
    selected = selection.selected[index % len(selection.selected)]
    identity = _canonical_digest(
        {"source": selection.source_sha256, "index": selected.source_index, "text": selected.question}
    )
    return {
        "question": selected.question, "query_identity": identity,
        "query_source_index": selected.source_index,
        "query_schedule_index": index,
        "query_repetition": index // len(selection.selected),
    }


def _collect_functional(args: argparse.Namespace) -> dict[str, object]:
    from deployment import ragctl

    inputs = _live_inputs(args)
    config, user, selection = inputs["config"], inputs["user"], inputs["selection"]
    output = args.output.resolve()
    output.mkdir(parents=True, mode=0o700)
    runner = ragctl.CommandRunner(config)
    try:
        ragctl.down(config, runner)
        ragctl.up(config, runner, acceptance_observer_enabled=True,
                  acceptance_experiment_sha256=str(inputs["configuration_sha256"]))
        ask_row = _ask_sample(config, user, selection.selected[0].question)
        ask_row.update(_identity_fields(inputs))
        ask_row["runtime_instance_id"] = _runtime_instance(runner)
        _validate_live_sample(ask_row, evaluation=True)
    finally:
        ragctl.down(config, runner)
    before = set(ragctl.E2E_DIAGNOSTICS_PATH.iterdir()) if ragctl.E2E_DIAGNOSTICS_PATH.exists() else set()
    acceptance_e2e_config = dict(config)
    acceptance_e2e_config["RAG_DIAGNOSTIC_USER_ID"] = user
    ragctl.diagnose_e2e(
        acceptance_e2e_config, runner, args.queries, start=0, limit=args.e2e_limit
    )
    created = set(ragctl.E2E_DIAGNOSTICS_PATH.iterdir()) - before
    if len(created) != 1:
        raise GateError("E2E diagnostic artifact correlation failed")
    e2e_path = created.pop()
    rows = _load_jsonl(e2e_path / "requests.jsonl")
    for row in rows:
        if row.get("status") != "succeeded":
            raise GateError("functional E2E request failed")
        validate_evaluation_success(cast_mapping(row.get("evaluation")), expected_request=row, expected_source="e2e")
    result = {
        "schema_version": "1.0", "status": "passed", "ask": ask_row,
        "e2e_directory": str(e2e_path), "e2e_request_count": len(rows),
        "identities": _identity_fields(inputs),
    }
    _write_private(output / "functional.json", result)
    return result


def _collect_performance(args: argparse.Namespace) -> dict[str, object]:
    from deployment import ragctl

    inputs = _live_inputs(args)
    _validate_functional_dependency(args.functional_evidence, inputs)
    config, user, selection = inputs["config"], inputs["user"], inputs["selection"]
    output = args.output.resolve()
    output.mkdir(parents=True, mode=0o700)
    runner = ragctl.CommandRunner(config)
    rows: list[dict[str, object]] = []
    pairs_per_crossover = args.pairs // 2
    for block, mode in enumerate(("off", "on", "on", "off")):
        try:
            ragctl.down(config, runner)
            ragctl.up(config, runner, evaluation_evidence_enabled=mode == "on",
                      acceptance_observer_enabled=True,
                      acceptance_experiment_sha256=str(inputs["configuration_sha256"]))
            runtime_id = _runtime_instance(runner)
            for ordinal in range(pairs_per_crossover + 3):
                schedule = max(0, ordinal - 3)
                fields = _query_fields(selection, schedule)
                row = _ask_sample(config, user, str(fields["question"]))
                row.update(fields | _identity_fields(inputs))
                row.update({
                    "mode": mode, "block_sequence": block,
                    "block_request_index": ordinal, "warmup": ordinal < 3,
                    "pair_key": f"crossover-{0 if block < 2 else 1}-{schedule}",
                    "runtime_instance_id": runtime_id,
                })
                rows.append(row)
        finally:
            ragctl.down(config, runner)
    measured = validate_latency_schedule(rows)
    off_on = paired_latency_gate(measured)

    activity = output / "judge-activity.jsonl"
    descriptor = os.open(activity, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    contention: list[dict[str, object]] = []
    try:
        ragctl.up(config, runner, evaluation_evidence_enabled=True,
                  acceptance_observer_enabled=True,
                  acceptance_experiment_sha256=str(inputs["configuration_sha256"]))
        runtime_id = _runtime_instance(runner)
        for index in range(args.pairs):
            fields = _query_fields(selection, index)
            base = _ask_sample(config, user, str(fields["question"]),
                               reserve_evaluator_idle=True)
            base.update(fields | _identity_fields(inputs) | {
                "mode": "uncontended", "pair_key": f"contention-{index}",
                "runtime_instance_id": runtime_id,
            })
            contention.append(base)
            donor_fields = _query_fields(selection, index + 1)
            prior = len(_load_jsonl(activity))
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _ask_process, config, str(donor_fields["question"]), str(activity)
                )
                deadline = time.monotonic() + 1800
                active_key: tuple[object, object] | None = None
                while True:
                    events = _load_jsonl(activity)
                    recent = events[prior:]
                    starts = {
                        (item.get("evaluation_job_id"), item.get("sequence"))
                        for item in recent if item.get("event") == "start"
                    }
                    ends = {
                        (item.get("evaluation_job_id"), item.get("sequence"))
                        for item in recent if item.get("event") == "end"
                    }
                    if starts - ends:
                        active_key = next(iter(starts - ends))
                        break
                    if future.done() or time.monotonic() >= deadline:
                        raise GateError("correlated donor judge activity did not start")
                    time.sleep(0.05)
                compared = _ask_sample(config, user, str(fields["question"]), activity=activity)
            donor = dict(future.result())
            donor["user_id"] = user
            donor.update(_post_generation(config, donor) | _identity_fields(inputs))
            intervals = _activity_intervals(activity)
            job_id = cast_mapping(donor.get("evaluation")).get("evaluation_job_id")
            matches = [
                item for item in intervals
                if item.get("evaluation_job_id") == job_id
                and active_key == (item.get("evaluation_job_id"), item.get("sequence"))
            ]
            if not matches:
                raise GateError("donor evaluation has no correlated judge interval")
            compared.update(fields | _identity_fields(inputs) | {
                "mode": "contended", "pair_key": f"contention-{index}",
                "runtime_instance_id": runtime_id,
                "judge_activity_source": "ollama_http",
                "judge_activity": matches[0],
                "overlapping_evaluation_request": donor,
                "request_done_monotonic": compared["done_received_monotonic"],
            })
            contention.append(compared)
    finally:
        ragctl.down(config, runner)
    overlap = paired_latency_gate(
        validate_latency_schedule(contention, contention=True),
        baseline_mode="uncontended", comparison_mode="contended",
    )
    result = {"schema_version": "1.0", "status": "passed" if off_on["status"] == overlap["status"] == "passed" else "failed", "off_on": off_on, "contention": overlap}
    _write_private(output / "performance.json", result)
    with (output / "latency-samples.jsonl").open("x", encoding="utf-8") as target:
        for row in rows + contention:
            target.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.chmod(output / "latency-samples.jsonl", 0o600)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="validate protected contracts offline")
    live_preflight = commands.add_parser(
        "live-preflight", help="validate real-run prerequisites without services"
    )
    live_preflight.add_argument("--env", type=Path, default=ROOT / ".env")
    live_preflight.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    live_preflight.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    live_preflight.add_argument(
        "--corpus-state",
        type=Path,
        default=ROOT / ".local" / "diagnostics" / "wizard" / "corpus-state.json",
    )
    verify = commands.add_parser("verify", help="verify authorized live-run artifacts")
    verify.add_argument("--ask-status", type=Path, required=True)
    verify.add_argument("--e2e-requests", type=Path, required=True)
    verify.add_argument("--latency-samples", type=Path, required=True)
    verify.add_argument("--contention-samples", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    for name in ("collect-functional", "collect-performance"):
        collect = commands.add_parser(name, help=f"run authorized real {name[8:]} collection")
        collect.add_argument("--authorize-live", action="store_true")
        collect.add_argument("--env", type=Path, default=ROOT / ".env")
        collect.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
        collect.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
        collect.add_argument(
            "--corpus-state", type=Path,
            default=ROOT / ".local" / "diagnostics" / "wizard" / "corpus-state.json",
        )
        collect.add_argument("--output", type=Path, required=True)
        if name == "collect-functional":
            collect.add_argument("--e2e-limit", type=int)
        else:
            collect.add_argument("--pairs", type=int, default=30)
            collect.add_argument("--functional-evidence", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = validate_protected_contracts()
        elif args.command == "live-preflight":
            result = validate_live_prerequisites(
                env_path=args.env,
                fixtures_path=args.fixtures,
                queries_path=args.queries,
                corpus_state_path=args.corpus_state,
            )
        elif args.command == "verify":
            # This must be the first gate step.  Real-run artifacts are not
            # accepted when protected production contracts have drifted.
            validate_protected_contracts()
            result = verify_live_artifacts(
                ask_observation=_load_object(args.ask_status, "ask status"),
                e2e_rows=_load_jsonl(args.e2e_requests),
                latency_samples=_load_jsonl(args.latency_samples),
                contention_samples=_load_jsonl(args.contention_samples),
            )
            output = args.output.resolve()
            if output.exists() or output.is_symlink():
                raise GateError("gate output already exists")
            _write_private(output, result)
        else:
            if not args.authorize_live:
                raise GateError("live collection requires --authorize-live")
            if args.output.exists() or args.output.is_symlink():
                raise GateError("live collection output already exists")
            if args.command == "collect-performance" and (
                args.pairs < 30 or args.pairs % 2
            ):
                raise GateError("performance collection requires an even total of at least 30 pairs")
            result = (
                _collect_functional(args)
                if args.command == "collect-functional"
                else _collect_performance(args)
            )
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    except (GateError, OSError, UnicodeError) as exc:
        print(f"non-regression gate failed: {type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
