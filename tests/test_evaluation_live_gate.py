"""Offline tests of the separately authorized live-artifact gate."""

from copy import deepcopy
from uuid import uuid4

import pytest

from evaluation import non_regression_gate as gate


def _schedule():
    rows = []
    for block, mode in enumerate(("off", "on", "on", "off")):
        for index in range(18):
            pair = (index - 3) + (15 if block >= 2 else 0)
            rows.append({
                "block_sequence": block, "block_request_index": index,
                "runtime_instance_id": f"runtime-{block}", "warmup": index < 3,
                "mode": mode, "pair_key": str(pair), "question": f"question-{pair}",
                "request_id": str(uuid4()),
                "protected_contract_sha256": gate._digest(gate.MANIFEST_PATH),
                "configuration_sha256": "a" * 64, "corpus_sha256": "b" * 64,
                "models_sha256": "c" * 64,
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
    contended = deepcopy(baseline)
    contended.update({
        "mode": "contended", "request_id": str(uuid4()),
        "judge_activity_source": "ollama_http", "judge_started_monotonic": 1.0,
        "request_started_monotonic": 2.0, "judge_finished_monotonic": 3.0,
        "request_done_monotonic": 4.0,
        "overlapping_evaluation_request": {"request_id": str(uuid4())},
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
                         "timings_ms": {name: value for name in gate.LATENCY_METRICS}})
    assert gate.paired_latency_gate(rows)["status"] == "failed"
