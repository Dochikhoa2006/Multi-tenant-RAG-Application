"""One-record, post-generation evaluator orchestration."""

from __future__ import annotations

import importlib.metadata
import math
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4
from datetime import datetime, timezone

from .metrics import (
    EXPECTED_RAGAS_VERSION,
    METRIC_PLANS,
    EvaluationSettings,
    MetricBackend,
    RagasMetricBackend,
    metric_skip_code,
    safe_exception_type,
)
from .models import (
    RESULT_SCHEMA_VERSION,
    EvaluationRecord,
    EvaluationResult,
    FrozenDict,
    MetricOutcome,
)


class InvalidMetricScore(ValueError):
    """Raised when a metric returns a value outside its public score contract."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _duration_ms(start_ns: int) -> float:
    value = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 3)
    return 0.0 if value == 0 else value


def _validate_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidMetricScore("metric score is not numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise InvalidMetricScore("metric score is outside [0, 1]")
    return score


def _default_failed_metadata(
    settings: EvaluationSettings,
) -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "provider": "ollama",
            "model": settings.judge_model,
            "model_digest": None,
            "ollama_version": None,
        },
        {
            "provider": "huggingface_local",
            "model": settings.embedding_model_path.name,
            "configuration_sha256": None,
            "dimension": None,
            "sentence_transformers_version": None,
        },
    )


async def evaluate_record(
    record: EvaluationRecord,
    *,
    settings: EvaluationSettings | None = None,
    include_noise_sensitivity: bool = False,
    backend: MetricBackend | None = None,
) -> EvaluationResult:
    """Evaluate one immutable record without contacting the RAG application."""

    started_at = _utc_now()
    resolved_settings = settings or EvaluationSettings.from_environment()
    owns_backend = backend is None
    setup_error_code: str | None = None
    setup_error_type: str | None = None

    if backend is None:
        try:
            backend = await RagasMetricBackend.create(resolved_settings)
        except Exception as exc:
            setup_error_code = "EVALUATOR_SETUP_FAILED"
            setup_error_type = safe_exception_type(exc)

    outcomes: list[MetricOutcome] = []
    if backend is None:
        judge_metadata, embedding_metadata = _default_failed_metadata(
            resolved_settings
        )
        try:
            ragas_version: str | None = importlib.metadata.version("ragas")
        except importlib.metadata.PackageNotFoundError:
            ragas_version = None
        for plan in METRIC_PLANS:
            skip_code = metric_skip_code(
                plan,
                record,
                include_noise_sensitivity=include_noise_sensitivity,
            )
            if skip_code is not None:
                outcomes.append(
                    MetricOutcome(
                        name=plan.name,
                        ragas_metric=plan.ragas_metric,
                        status="skipped",
                        query_basis=plan.query_basis,
                        error_code=skip_code,
                    )
                )
            else:
                outcomes.append(
                    MetricOutcome(
                        name=plan.name,
                        ragas_metric=plan.ragas_metric,
                        status="failed",
                        query_basis=plan.query_basis,
                        error_code="EVALUATOR_SETUP_FAILED",
                        error_type=setup_error_type,
                    )
                )
    else:
        judge_metadata = dict(backend.judge_metadata)
        embedding_metadata = dict(backend.embedding_metadata)
        ragas_version = backend.ragas_version
        try:
            for plan in METRIC_PLANS:
                skip_code = metric_skip_code(
                    plan,
                    record,
                    include_noise_sensitivity=include_noise_sensitivity,
                )
                if skip_code is not None:
                    outcomes.append(
                        MetricOutcome(
                            name=plan.name,
                            ragas_metric=plan.ragas_metric,
                            status="skipped",
                            query_basis=plan.query_basis,
                            error_code=skip_code,
                        )
                    )
                    continue

                metric_started = time.perf_counter_ns()
                try:
                    score = _validate_score(await backend.score(plan, record))
                    outcomes.append(
                        MetricOutcome(
                            name=plan.name,
                            ragas_metric=plan.ragas_metric,
                            status="succeeded",
                            query_basis=plan.query_basis,
                            score=score,
                            duration_ms=_duration_ms(metric_started),
                        )
                    )
                except InvalidMetricScore as exc:
                    outcomes.append(
                        MetricOutcome(
                            name=plan.name,
                            ragas_metric=plan.ragas_metric,
                            status="failed",
                            query_basis=plan.query_basis,
                            duration_ms=_duration_ms(metric_started),
                            error_code="INVALID_METRIC_SCORE",
                            error_type=safe_exception_type(exc),
                        )
                    )
                except Exception as exc:
                    outcomes.append(
                        MetricOutcome(
                            name=plan.name,
                            ragas_metric=plan.ragas_metric,
                            status="failed",
                            query_basis=plan.query_basis,
                            duration_ms=_duration_ms(metric_started),
                            error_code="METRIC_EVALUATION_FAILED",
                            error_type=safe_exception_type(exc),
                        )
                    )
        finally:
            if owns_backend:
                try:
                    await backend.close()
                except Exception:
                    setup_error_code = "EVALUATOR_CLOSE_FAILED"
                    setup_error_type = "EvaluationCleanupError"

    applicable = [outcome for outcome in outcomes if outcome.status != "skipped"]
    succeeded = sum(outcome.status == "succeeded" for outcome in applicable)
    failed = sum(outcome.status == "failed" for outcome in applicable)
    if setup_error_code is not None or succeeded == 0:
        status = "failed"
    elif failed:
        status = "partial"
    else:
        status = "succeeded"

    if ragas_version != EXPECTED_RAGAS_VERSION and setup_error_code is None:
        setup_error_code = "RAGAS_VERSION_MISMATCH"
        setup_error_type = "EvaluationSetupError"
        status = "failed"

    return EvaluationResult(
        schema_version=RESULT_SCHEMA_VERSION,
        evaluation_id=str(uuid4()),
        record_sha256=record.sha256(),
        source=record.source,
        request_id=record.request_id,
        conversation_id=record.conversation_id,
        started_at=started_at,
        completed_at=_utc_now(),
        status=status,
        ragas_version=ragas_version,
        judge=FrozenDict(judge_metadata),
        embeddings=FrozenDict(embedding_metadata),
        metrics=tuple(outcomes),
        setup_error_code=setup_error_code,
        setup_error_type=setup_error_type,
    )


def write_result_atomic(result: EvaluationResult, output_path: Path) -> Path:
    """Atomically create a private result file without overwriting data."""

    target = output_path.expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"evaluation result already exists: {target}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(result.to_json())
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        temporary.unlink()
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return target
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
