from __future__ import annotations

import asyncio
import ast
import inspect
import json
import os
from pathlib import Path
import textwrap
from types import SimpleNamespace
from uuid import uuid4

import pytest

from rag_evaluation.metrics import (
    EXPECTED_RAGAS_VERSION,
    METRIC_PLANS,
    EvaluationSettings,
    EvaluationSetupError,
    RagasMetricBackend,
    assert_collections_api_compatible,
    metric_skip_code,
    _judge_activity_hooks,
)
from rag_evaluation.models import EvaluationRecord


class RecordingMetric:
    def __init__(self, value: float = 0.75) -> None:
        self.value = value
        self.calls: list[dict[str, object]] = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value)


class DummyClient:
    async def close(self) -> None:
        return None


def test_acceptance_judge_activity_is_correlated_and_content_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "activity.jsonl"
    path.touch(mode=0o600)
    job_id, request_id = str(uuid4()), str(uuid4())
    monkeypatch.setenv("RAG_EVAL_ACTIVITY_PATH", str(path))
    monkeypatch.setenv("RAG_EVAL_JOB_ID", job_id)
    monkeypatch.setenv("RAG_EVAL_REQUEST_ID", request_id)
    monkeypatch.setenv("RAG_EVAL_RECORD_SHA256", "a" * 64)
    hooks, descriptor = _judge_activity_hooks(
        {"model": "local-judge", "model_digest": "digest"}
    )
    request = SimpleNamespace(extensions={})
    response = SimpleNamespace(request=request, status_code=200)
    asyncio.run(hooks["request"][0](request))
    asyncio.run(hooks["response"][0](response))
    os.close(descriptor)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["start", "end"]
    assert all(event["evaluation_job_id"] == job_id for event in events)
    assert all(event["request_id"] == request_id for event in events)
    assert not any("prompt" in json.dumps(event).lower() for event in events)


def _backend() -> tuple[RagasMetricBackend, dict[str, RecordingMetric]]:
    metrics = {plan.name: RecordingMetric() for plan in METRIC_PLANS}
    return (
        RagasMetricBackend(
            client=DummyClient(),
            metrics=metrics,
            embeddings=object(),
            ragas_version=EXPECTED_RAGAS_VERSION,
            judge_metadata={"provider": "ollama", "model": "qwen3.5:4b"},
            embedding_metadata={"provider": "huggingface_local", "dimension": 384},
        ),
        metrics,
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com",
        "http://192.168.1.2:11434",
        "http://user:password@127.0.0.1:11434",
        "http://127.0.0.1:11434/custom",
    ],
)
def test_settings_reject_nonlocal_or_credentialed_endpoints(url, tmp_path) -> None:
    with pytest.raises(EvaluationSetupError):
        EvaluationSettings(
            ollama_url=url,
            embedding_model_path=tmp_path,
        ).validate()


def test_settings_normalize_loopback_openai_url(tmp_path) -> None:
    settings = EvaluationSettings(
        ollama_url="http://localhost:11434/v1/",
        embedding_model_path=tmp_path,
    )
    settings.validate()
    assert settings.ollama_origin == "http://localhost:11434"
    assert settings.openai_base_url == "http://localhost:11434/v1"


@pytest.mark.parametrize(
    "model",
    [
        "qwen3-4b-awq",
        "QWEN3-4B-AWQ:latest",
        "merged-granite-4.1-3b-query-rewrite",
    ],
)
def test_settings_reject_production_inference_model_identifiers(
    tmp_path, model
) -> None:
    with pytest.raises(EvaluationSetupError, match="production inference model"):
        EvaluationSettings(
            judge_model=model,
            embedding_model_path=tmp_path,
        ).validate()


def test_backend_uses_bounded_judge_output_without_changing_client_safety() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(RagasMetricBackend.create)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    def named_call(name: str) -> ast.Call:
        return next(
            call
            for call in calls
            if isinstance(call.func, ast.Name) and call.func.id == name
        )

    def keywords(call: ast.Call) -> dict[str, ast.expr]:
        return {item.arg: item.value for item in call.keywords if item.arg is not None}

    llm = keywords(named_call("llm_factory"))
    assert ast.literal_eval(llm["provider"]) == "openai"
    assert ast.unparse(llm["client"]) == "client"
    assert ast.literal_eval(llm["temperature"]) == 0.0
    assert ast.literal_eval(llm["max_tokens"]) == 4096

    client = keywords(named_call("AsyncOpenAI"))
    assert ast.unparse(client["base_url"]) == "settings.openai_base_url"
    assert ast.literal_eval(client["max_retries"]) == 0
    assert ast.unparse(client["timeout"]) == "settings.request_timeout_seconds"
    http_client = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "AsyncClient"
    )
    assert ast.literal_eval(keywords(http_client)["trust_env"]) is False


def test_locked_collections_api_is_compatible() -> None:
    assert assert_collections_api_compatible() == "0.4.3"


def test_reference_and_opt_in_metric_selection(record) -> None:
    plans = {plan.name: plan for plan in METRIC_PLANS}
    assert metric_skip_code(
        plans["faithfulness"], record, include_noise_sensitivity=False
    ) is None
    assert (
        metric_skip_code(
            plans["context_recall"], record, include_noise_sensitivity=False
        )
        == "REFERENCE_NOT_PROVIDED"
    )
    assert (
        metric_skip_code(
            plans["noise_sensitivity"], record, include_noise_sensitivity=False
        )
        == "NOT_REQUESTED"
    )


def test_metric_backend_routes_original_and_rewritten_queries(record_mapping) -> None:
    record_mapping["reference"] = "Expected answer."
    record = EvaluationRecord.from_mapping(record_mapping)
    backend, metrics = _backend()

    for plan in METRIC_PLANS:
        assert asyncio.run(backend.score(plan, record)) == 0.75

    assert metrics["faithfulness"].calls[0]["user_input"] == record.original_query
    assert metrics["response_relevancy"].calls[0]["user_input"] == record.original_query
    assert metrics["context_utilization"].calls[0]["user_input"] == record.rewritten_query
    assert metrics["context_recall"].calls[0]["user_input"] == record.rewritten_query
    assert (
        metrics["context_precision_with_reference"].calls[0]["user_input"]
        == record.rewritten_query
    )
    assert metrics["noise_sensitivity"].calls[0]["user_input"] == record.original_query
    assert metrics["factual_correctness"].calls[0] == {
        "response": record.response,
        "reference": record.reference,
    }
    for name in (
        "faithfulness",
        "context_utilization",
        "context_recall",
        "context_precision_with_reference",
        "noise_sensitivity",
    ):
        assert metrics[name].calls[0]["retrieved_contexts"] == list(
            record.retrieved_contexts
        )


def test_tracking_and_huggingface_network_are_forced_off(monkeypatch) -> None:
    import os
    import rag_evaluation.metrics as metrics_module

    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert Path(metrics_module.__file__).is_file()
