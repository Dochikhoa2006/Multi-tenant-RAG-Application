from __future__ import annotations

import json
from pathlib import Path
import stat
from dataclasses import replace

import pytest

import rag_evaluation.cli as cli
from rag_evaluation.cli import _load_record, build_parser
from rag_evaluation.evaluator import write_result_atomic
from rag_evaluation.models import (
    RESULT_SCHEMA_VERSION,
    EvaluationResult,
    FrozenDict,
    MetricOutcome,
)
from datetime import datetime, timezone


def _result(record) -> EvaluationResult:
    now = datetime.now(timezone.utc)
    return EvaluationResult(
        schema_version=RESULT_SCHEMA_VERSION,
        evaluation_id="33333333-3333-4333-8333-333333333333",
        record_sha256=record.sha256(),
        source=record.source,
        request_id=record.request_id,
        conversation_id=record.conversation_id,
        started_at=now,
        completed_at=now,
        status="succeeded",
        ragas_version="0.4.3",
        judge=FrozenDict({"provider": "ollama", "model": "qwen3.5:4b"}),
        embeddings=FrozenDict({"provider": "huggingface_local", "dimension": 384}),
        metrics=(
            MetricOutcome(
                name="faithfulness",
                ragas_metric="Faithfulness",
                status="succeeded",
                query_basis="original_query",
                score=0.75,
                duration_ms=1.25,
            ),
        ),
    )


def test_parser_exposes_smoke_and_single_record_run(tmp_path) -> None:
    smoke = build_parser().parse_args(["smoke"])
    assert smoke.command == "smoke"

    run = build_parser().parse_args(
        ["run", str(tmp_path / "record.json"), "--noise-sensitivity"]
    )
    assert run.command == "run"
    assert run.noise_sensitivity is True


def test_record_loader_rejects_symlinks(record, tmp_path) -> None:
    source = tmp_path / "record.json"
    source.write_text(record.canonical_json(), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="non-symlink"):
        _load_record(link)


def test_record_loader_reads_one_utf8_record(record, tmp_path) -> None:
    source = tmp_path / "record.json"
    source.write_text(record.canonical_json(), encoding="utf-8")
    assert _load_record(source) == record


def test_result_write_is_private_atomic_and_never_overwrites(record, tmp_path) -> None:
    output = tmp_path / "nested" / "result.json"
    result = _result(record)

    assert write_result_atomic(result, output) == output.resolve()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert not list(output.parent.glob("*.tmp"))

    with pytest.raises(FileExistsError):
        write_result_atomic(result, output)


def test_result_json_has_no_judge_reason_or_error_message(record) -> None:
    payload = _result(record).to_mapping()
    metric = payload["metrics"][0]
    assert "reason" not in metric
    assert "error_message" not in metric


@pytest.mark.parametrize(
    ("status", "setup_error_code", "expected_exit"),
    [
        ("succeeded", None, 0),
        ("partial", None, 1),
        ("failed", None, 1),
        ("failed", "EVALUATOR_SETUP_FAILED", 2),
    ],
)
def test_cli_writes_one_result_and_returns_contract_exit_code(
    record,
    tmp_path,
    monkeypatch,
    capsys,
    status,
    setup_error_code,
    expected_exit,
) -> None:
    source = tmp_path / "record.json"
    source.write_text(record.canonical_json(), encoding="utf-8")
    output = tmp_path / "result.json"
    result = replace(
        _result(record),
        status=status,
        setup_error_code=setup_error_code,
        setup_error_type="EvaluationSetupError" if setup_error_code else None,
    )

    async def synthetic_evaluate(*args, **kwargs):
        return result

    monkeypatch.setattr(cli, "evaluate_record", synthetic_evaluate)
    exit_code = cli.main(["run", str(source), "--output", str(output)])

    assert exit_code == expected_exit
    assert output.is_file()
    terminal = capsys.readouterr().out
    assert f"evaluation_status={status}" in terminal
    assert record.response not in terminal
    assert record.knowledge_contexts[0] not in terminal


def test_cli_invalid_input_returns_two_without_creating_result(tmp_path, capsys) -> None:
    output = tmp_path / "result.json"
    exit_code = cli.main(
        ["run", str(tmp_path / "missing.json"), "--output", str(output)]
    )
    assert exit_code == 2
    assert not output.exists()
    assert "evaluation_status=invalid" in capsys.readouterr().out
