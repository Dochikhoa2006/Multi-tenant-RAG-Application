"""Offline tests of the separately authorized live-artifact gate."""

from copy import deepcopy
import fcntl
import json
import os
from uuid import uuid4

import pytest

from evaluation import non_regression_gate as gate


def _schedule():
    rows = []
    for block, mode in enumerate(("off", "on", "on", "off")):
        for index in range(18):
            pair = index - 3
            question = f"question-{pair}"
            rows.append({
                "block_sequence": block, "block_request_index": index,
                "runtime_instance_id": f"runtime-{block}", "warmup": index < 3,
                "runtime_worker_id": f"worker-{block}",
                "evaluation_evidence_enabled": mode == "on",
                "mode": mode, "pair_key": f"{block // 2}-{pair}", "question": question,
                "query_identity": gate._canonical_digest(question),
                "query_source_index": max(pair, 0),
                "query_schedule_index": max(pair, 0), "query_repetition": 0,
                "request_id": str(uuid4()),
                "protected_contract_sha256": gate._digest(gate.MANIFEST_PATH),
                "configuration_sha256": "a" * 64, "corpus_sha256": "b" * 64,
                "models_sha256": "c" * 64,
                "acceptance_experiment_sha256": "a" * 64,
                "timings_ms": {name: 100.0 for name in gate.LATENCY_METRICS},
            })
    return rows


def test_four_runtime_schedule_excludes_exactly_three_warmups_per_block(monkeypatch):
    checked = []
    monkeypatch.setattr(gate, "_validate_live_sample", lambda row, *, evaluation: checked.append((row, evaluation)))
    selected = gate.validate_latency_schedule(_schedule())
    assert len(checked) == 72
    assert sum(enabled for _, enabled in checked) == 36
    assert len(selected) == 60
    assert gate.paired_latency_gate(selected)["complete_pair_count"] == 30


@pytest.mark.parametrize("defect", ["config", "blocks", "warmup", "runtime", "question"])
def test_live_schedule_rejects_drift_and_invalid_pairing(monkeypatch, defect):
    monkeypatch.setattr(gate, "_validate_live_sample", lambda *_args, **_kwargs: None)
    rows = _schedule()
    if defect == "config":
        rows[-1]["configuration_sha256"] = "d" * 64
    elif defect == "blocks":
        rows[0]["mode"] = "on"
    elif defect == "warmup":
        rows[2]["warmup"] = False
    elif defect == "runtime":
        for row in rows:
            row["runtime_instance_id"] = "same-runtime"
    else:
        rows[-1]["question"] = "different query"
    with pytest.raises(gate.GateError):
        gate.validate_latency_schedule(rows)


def test_contention_baseline_is_uncontended_and_other_judge_must_be_verified(monkeypatch):
    checked = []
    monkeypatch.setattr(gate, "_validate_live_sample", lambda row, *, evaluation: checked.append(row["request_id"]))
    baseline = _schedule()[0]
    baseline["mode"] = "uncontended"
    baseline["evaluation_evidence_enabled"] = True
    baseline.update({
        "evaluator_idle_reserved": True,
        "idle_reservation_acquired_monotonic": 0.5,
        "request_started_monotonic": 1.0,
        "done_validated_monotonic": 2.0,
        "idle_reservation_released_monotonic": 2.5,
    })
    contended = deepcopy(baseline)
    donor_id = str(uuid4())
    job_id = str(uuid4())
    contended.update({
        "mode": "contended", "request_id": str(uuid4()),
        "judge_activity_source": "ollama_http",
        "judge_activity": {
            "evaluation_job_id": job_id, "request_id": donor_id,
            "record_sha256": "d" * 64, "sequence": 1, "judge_id": "local",
            "started_monotonic": 1.0, "finished_monotonic": 3.0,
            "success": True,
        },
        "request_started_monotonic": 2.0,
        "request_done_monotonic": 4.0,
        "overlapping_evaluation_request": {
            "request_id": donor_id,
            "evaluation": {"evaluation_job_id": job_id, "record_sha256": "d" * 64},
        },
    })
    assert gate.validate_latency_schedule([baseline, contended], contention=True)
    assert len(checked) == 3  # baseline, actual overlapping job, measured request
    contended["judge_activity_source"] = "process_alive"
    with pytest.raises(gate.GateError, match="actual Ollama"):
        gate.validate_latency_schedule([baseline, contended], contention=True)


def test_exact_five_percent_upper_bound_does_not_pass():
    rows = []
    for index in range(30):
        for mode, value in (("off", 100.0), ("on", 105.0)):
            rows.append({"pair_key": str(index), "mode": mode,
                         "question": f"q-{index}", "query_identity": "a" * 64,
                         "query_source_index": index, "query_schedule_index": index,
                         "query_repetition": 0,
                         "timings_ms": {name: value for name in gate.LATENCY_METRICS}})
    assert gate.paired_latency_gate(rows)["status"] == "failed"


def test_performance_requires_matching_functional_acceptance(tmp_path):
    inputs = {
        "protected_contract_sha256": "a" * 64,
        "configuration_sha256": "b" * 64,
        "corpus_sha256": "c" * 64,
        "models_sha256": "d" * 64,
    }
    evidence = tmp_path / "functional.json"
    evidence.write_text(json.dumps({
        "status": "passed", "identities": gate._identity_fields(inputs),
    }))
    gate._validate_functional_dependency(evidence, inputs)
    changed = json.loads(evidence.read_text())
    changed["identities"]["corpus_sha256"] = "e" * 64
    evidence.write_text(json.dumps(changed))
    with pytest.raises(gate.GateError, match="missing or mismatched"):
        gate._validate_functional_dependency(evidence, inputs)


def test_idle_reservation_is_released_at_done_before_evaluation(
    monkeypatch, tmp_path,
):
    from deployment import evaluation_bridge

    lock_path = tmp_path / "execution.lock"
    monkeypatch.setattr(evaluation_bridge, "LOCAL_EVALUATION_LOCK_PATH", lock_path)

    def ask_process(_config, _question, _activity, hook):
        probe = os.open(lock_path, os.O_RDWR)
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hook()
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe, fcntl.LOCK_UN)
        os.close(probe)
        return {"request_started_monotonic": 1.0, "done_validated_monotonic": 2.0}

    monkeypatch.setattr(gate, "_ask_process", ask_process)
    monkeypatch.setattr(gate, "_post_generation", lambda *_: {})
    row = gate._ask_sample({}, "user", "question", reserve_evaluator_idle=True)
    assert row["evaluator_idle_reserved"] is True
    assert row["idle_reservation_released_monotonic"] >= row["idle_reservation_acquired_monotonic"]


def test_performance_parser_requires_functional_evidence_and_even_pair_total():
    parser = gate.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["collect-performance", "--authorize-live", "--output", "out"])
    parsed = parser.parse_args([
        "collect-performance", "--authorize-live", "--output", "out",
        "--functional-evidence", "functional.json", "--pairs", "30",
    ])
    assert parsed.pairs == 30
    assert gate.main([
        "collect-performance", "--authorize-live", "--output", "out",
        "--functional-evidence", "functional.json", "--pairs", "31",
    ]) == 2
