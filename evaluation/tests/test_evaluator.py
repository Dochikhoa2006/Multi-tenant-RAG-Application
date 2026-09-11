from __future__ import annotations

import asyncio
from dataclasses import replace
import math

from rag_evaluation.evaluator import evaluate_record
from rag_evaluation.metrics import EXPECTED_RAGAS_VERSION, RagasMetricBackend
from rag_evaluation.models import EvaluationRecord


class SyntheticBackend:
    ragas_version = EXPECTED_RAGAS_VERSION
    judge_metadata = {
        "provider": "ollama",
        "model": "synthetic-local-judge",
        "model_digest": "a" * 64,
        "ollama_version": "test",
    }
    embedding_metadata = {
        "provider": "huggingface_local",
        "model": "synthetic-embedding",
        "configuration_sha256": "b" * 64,
        "dimension": 384,
        "sentence_transformers_version": "test",
    }

    def __init__(self, outcomes=None) -> None:
        self.outcomes = outcomes or {}
        self.calls: list[str] = []
        self.closed = False

    async def score(self, plan, record) -> float:
        self.calls.append(plan.name)
        outcome = self.outcomes.get(plan.name, 0.8)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.closed = True


def _evaluate(record, backend, *, noise=False):
    return asyncio.run(
        evaluate_record(
            record,
            backend=backend,
            include_noise_sensitivity=noise,
        )
    )


def test_reference_free_evaluation_calls_only_required_metrics(record) -> None:
    backend = SyntheticBackend()
    result = _evaluate(record, backend)

    assert result.status == "succeeded"
    assert backend.calls == [
        "faithfulness",
        "response_relevancy",
        "context_utilization",
    ]
    outcomes = {item.name: item for item in result.metrics}
    assert outcomes["context_recall"].status == "skipped"
    assert outcomes["context_recall"].error_code == "REFERENCE_NOT_PROVIDED"
    assert outcomes["noise_sensitivity"].error_code == "NOT_REQUESTED"
    assert result.record_sha256 == record.sha256()


def test_reference_evaluation_and_noise_are_explicit(record_mapping) -> None:
    record_mapping["reference"] = "A human-authored answer."
    record_mapping["reference_context_ids"] = ["gold-1"]
    record = EvaluationRecord.from_mapping(record_mapping)
    backend = SyntheticBackend()

    result = _evaluate(record, backend, noise=True)

    assert result.status == "succeeded"
    assert backend.calls == [
        "faithfulness",
        "response_relevancy",
        "context_utilization",
        "context_recall",
        "context_precision_with_reference",
        "factual_correctness",
        "noise_sensitivity",
    ]


def test_metric_failure_is_independent_and_sanitized(record) -> None:
    secret = "raw context and provider response must not leak"
    backend = SyntheticBackend(
        {"response_relevancy": RuntimeError(secret)}
    )

    result = _evaluate(record, backend)

    assert result.status == "partial"
    assert backend.calls == [
        "faithfulness",
        "response_relevancy",
        "context_utilization",
    ]
    failed = next(item for item in result.metrics if item.status == "failed")
    assert failed.error_code == "METRIC_EVALUATION_FAILED"
    assert failed.error_type == "RuntimeError"
    assert secret not in result.to_json()


def test_invalid_metric_values_fail_only_their_metric(record) -> None:
    for invalid in (True, -0.01, 1.01, math.nan, math.inf):
        backend = SyntheticBackend({"faithfulness": invalid})
        result = _evaluate(record, backend)
        failed = result.metrics[0]
        assert failed.status == "failed"
        assert failed.error_code == "INVALID_METRIC_SCORE"
        assert result.status == "partial"


def test_all_applicable_metric_failures_make_result_failed(record) -> None:
    backend = SyntheticBackend(
        {
            "faithfulness": RuntimeError("one"),
            "response_relevancy": RuntimeError("two"),
            "context_utilization": RuntimeError("three"),
        }
    )
    result = _evaluate(record, backend)
    assert result.status == "failed"


def test_injected_backend_is_not_owned_or_closed(record) -> None:
    backend = SyntheticBackend()
    _evaluate(record, backend)
    assert backend.closed is False


def test_result_contains_no_raw_record_content(record) -> None:
    result = _evaluate(record, SyntheticBackend())
    payload = result.to_json()
    assert record.original_query not in payload
    assert record.response not in payload
    assert record.knowledge_contexts[0] not in payload
    assert record.policy_contexts[0] not in payload


def test_ragas_import_or_ollama_setup_failure_is_independent(
    record, monkeypatch
) -> None:
    for failure in (
        ModuleNotFoundError("ragas is unavailable"),
        ConnectionError("local Ollama is unavailable"),
        TimeoutError("local judge timed out"),
    ):
        async def fail_create(*_args, _failure=failure, **_kwargs):
            raise _failure

        monkeypatch.setattr(RagasMetricBackend, "create", fail_create)
        result = asyncio.run(evaluate_record(record))
        assert result.status == "failed"
        assert all(
            metric.status in {"failed", "skipped"} for metric in result.metrics
        )
        assert result.setup_error_code == "EVALUATOR_SETUP_FAILED"
        assert str(failure) not in result.to_json()
