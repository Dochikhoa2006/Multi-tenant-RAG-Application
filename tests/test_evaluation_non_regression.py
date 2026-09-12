from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
from uuid import uuid4

import pytest

from backend.rag import generator
from backend.wizard.diagnostics import (
    DiagnosticTraceRegistry,
    activate_operation,
    capture_evaluation_contexts,
    capture_evaluation_rewrite,
    install_registry,
    uninstall_registry,
)
from deployment import evaluation_bridge as bridge
from evaluation import non_regression_gate as gate


USER = "evaluation_user"


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, value: str) -> list[int]:
        self.calls.append(value)
        return list(range(len(value.split())))


def _identity(collection: str, identifier: str, text: str) -> bridge.ContextIdentity:
    fingerprint = bridge.framed_content_digest(
        "chat-kp-item-v1", (identifier.encode(), text.encode())
    )
    digest = bridge.framed_content_digest(
        f"chat-qwen-{collection}-context-v1",
        (identifier.encode(), text.encode()),
    )
    return bridge.ContextIdentity(
        collection, (identifier,), (f"{identifier}:{fingerprint}",), digest, len(text)
    )


def _evidence() -> bridge.RequestEvidence:
    return bridge.RequestEvidence(
        USER,
        str(uuid4()),
        str(uuid4()),
        str(uuid4()),
        str(uuid4()),
        "official rewrite",
        _identity("knowledge", str(uuid4()), "knowledge"),
        _identity("policy", str(uuid4()), "policy"),
    )


def test_protected_contract_manifest_is_a_mandatory_exact_preflight(tmp_path: Path) -> None:
    assert gate.validate_protected_contracts()["status"] == "passed"
    manifest = json.loads(gate.MANIFEST_PATH.read_text())
    manifest["files"]["backend/api/models.py"] = "0" * 64
    drifted = tmp_path / "manifest.json"
    drifted.write_text(json.dumps(manifest))
    with pytest.raises(gate.GateError, match="drifted"):
        gate.validate_protected_contracts(drifted)
    manifest["files"] = {}
    drifted.write_text(json.dumps(manifest))
    with pytest.raises(gate.GateError, match="incomplete"):
        gate.validate_protected_contracts(drifted)


def test_only_the_two_literal_observation_hooks_differ_in_generation_sources() -> None:
    expected = {
        "backend/rag/generator.py": [
            "+    capture_evaluation_contexts,",
            "+            capture_evaluation_contexts(knowledge, policy)",
        ],
        "backend/providers/sglang_query_rewriter.py": [
            "+    capture_evaluation_rewrite,",
            "+        capture_evaluation_rewrite(result)",
        ],
    }
    for relative, allowed in expected.items():
        diff = subprocess.check_output(
            ["git", "diff", "--no-ext-diff", gate.BEHAVIORAL_BASELINE_COMMIT, "--", relative],
            cwd=gate.ROOT, text=True,
        )
        changes = [line for line in diff.splitlines()
                   if line[:1] in {"+", "-"} and not line.startswith(("+++", "---"))]
        assert changes == allowed


def test_all_other_protected_rag_sources_are_identical_to_behavioral_baseline() -> None:
    manifest = json.loads(gate.MANIFEST_PATH.read_text())
    exceptions = gate.APPROVED_EVIDENCE_HOOK_FILES | {"deployment/evaluation_bridge_worker.py"}
    for relative in manifest["files"]:
        if relative in exceptions:
            continue
        original = subprocess.check_output(
            ["git", "show", f"{gate.BEHAVIORAL_BASELINE_COMMIT}:{relative}"], cwd=gate.ROOT
        )
        assert (gate.ROOT / relative).read_bytes() == original, relative


def test_deep_malformed_correlation_faults_deep_session_only() -> None:
    session = str(uuid4())
    registry = DiagnosticTraceRegistry(USER, capture_mode="deep")
    registry.start(USER, session, "deep-run")
    assert registry.begin_operation(
        user_id=USER,
        session_id="not-a-uuid",
        operation_id=str(uuid4()),
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    ) is None
    assert registry.snapshot(USER, session)["trace_faulted"] is True

    unrelated = DiagnosticTraceRegistry(USER, capture_mode="deep")
    unrelated.start(USER, session, "deep-run")
    assert unrelated.begin_operation(
        user_id="somebody_else",
        session_id="not-a-uuid",
        operation_id="bad",
        kind="chat_query",
        collection_type="conversations",
        wizard_id="bad",
    ) is None
    assert unrelated.snapshot(USER, session)["trace_faulted"] is False


def test_evaluation_fault_is_scoped_to_valid_owner() -> None:
    first, second = str(uuid4()), str(uuid4())
    registry = DiagnosticTraceRegistry(USER, capture_mode="evaluation")
    registry.start(USER, first, "ask-1")
    registry.start(USER, second, "ask-2")
    assert registry.begin_operation(
        user_id=USER,
        session_id=first,
        operation_id="bad",
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    ) is None
    assert registry.snapshot(USER, first)["trace_faulted"] is True
    assert registry.snapshot(USER, second)["trace_faulted"] is False
    assert registry.begin_operation(
        user_id=USER,
        session_id="bad",
        operation_id="bad",
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    ) is None
    assert registry.snapshot(USER, second)["trace_faulted"] is False


def test_evaluator_child_environment_is_allowlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "secret")
    monkeypatch.setenv("WEAVIATE_API_KEY", "secret")
    monkeypatch.setenv("SGLANG_QUERY_REWRITE_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "local-judge")
    environment = bridge._sanitized_child_environment(evaluator=True)
    assert environment["RAGAS_DO_NOT_TRACK"] == "true"
    assert environment["RAG_EVAL_JUDGE_MODEL"] == "local-judge"
    for name in (
        "MODAL_TOKEN_SECRET",
        "WEAVIATE_API_KEY",
        "SGLANG_QUERY_REWRITE_API_KEY",
        "OPENAI_API_KEY",
        "PYTHONPATH",
    ):
        assert name not in environment


def test_evaluation_timing_fields_are_separate() -> None:
    artifact = bridge.EvaluationObservation(
        "succeeded", None, 30.0, None, None, "a" * 64, 10.0, 20.0
    ).artifact()
    assert artifact["queue_wait_ms"] == 10.0
    assert artifact["execution_ms"] == 20.0
    assert artifact["evaluation_ms"] == 30.0


def test_admitted_worker_timeout_is_observational(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bridge, "_run_supervised_worker", lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError())
    )
    evidence = _evidence()
    job = bridge.submit_local_evaluation(
        config={},
        user_id=USER,
        evidence=evidence,
        allowed_document_ids=None,
        source="rag_ask",
        request_id=evidence.request_id,
        conversation_id=str(uuid4()),
        original_query="question",
        response="answer",
        telemetry={"schema_version": "1.0", "timings_ms": {}},
        captured_at="2026-09-11T00:00:00Z",
        directory=tmp_path,
        stem="evaluation",
        execution_lock_path=tmp_path / "execution.lock",
    )
    result = bridge.finish_evaluation_job(job)
    assert result.status == "failed"
    assert result.error_code == "EVALUATION_TIMEOUT"
    assert result.execution_ms >= 0


def test_supervised_worker_uses_private_ipc_and_returns_safe_failure(tmp_path: Path) -> None:
    lock_path = tmp_path / "execution.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        result = bridge._run_supervised_worker(
            b"{}",
            descriptor=descriptor,
            timeout_seconds=5.0,
            state=bridge._EvaluationJobState(),
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert result == {
        "status": "failed",
        "error_code": "EVALUATION_WORKER_PAYLOAD_INVALID",
        "record_path": None,
        "result_path": None,
        "record_sha256": None,
    }


def test_evidence_observation_does_not_change_final_prompt_or_call_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge = [
        {"object_id": str(uuid4()), "raw_text": "kept knowledge", "rerank_score": 1.0},
        {"object_id": str(uuid4()), "raw_text": "decoy knowledge", "rerank_score": 0.1},
    ]
    policy = [
        {"object_id": str(uuid4()), "raw_text": "kept policy", "rerank_score": 1.0}
    ]
    off_tokenizer = _Tokenizer()
    monkeypatch.setattr(generator, "capture_evaluation_contexts", lambda *_args: None)
    off = generator._build_budgeted_prompt(
        "official rewrite", knowledge, policy, off_tokenizer
    )
    captured: list[tuple[object, object]] = []
    on_tokenizer = _Tokenizer()
    monkeypatch.setattr(
        generator,
        "capture_evaluation_contexts",
        lambda k, p: captured.append((tuple(k), tuple(p))),
    )
    on = generator._build_budgeted_prompt(
        "official rewrite", knowledge, policy, on_tokenizer
    )
    assert on.encode() == off.encode()
    assert on_tokenizer.calls == off_tokenizer.calls
    assert len(captured) == 1


def _write_successful_evaluation(
    tmp_path: Path,
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
    knowledge_id: str | None = None,
    policy_id: str | None = None,
    knowledge_text: str = "knowledge",
    policy_text: str = "policy",
) -> dict[str, object]:
    request_id = request_id or str(uuid4())
    conversation_id = conversation_id or str(uuid4())
    knowledge_id = knowledge_id or str(uuid4())
    policy_id = policy_id or str(uuid4())
    record = {
        "schema_version": "1.0",
        "source": "e2e",
        "request_id": request_id,
        "conversation_id": conversation_id,
        "original_query": "question",
        "rewritten_query": "official rewrite",
        "response": "answer",
        "knowledge_contexts": [knowledge_text],
        "knowledge_context_ids": [knowledge_id],
        "policy_contexts": [policy_text],
        "policy_context_ids": [policy_id],
        "retrieved_contexts": [knowledge_text, policy_text],
        "context_roles": ["knowledge", "policy"],
        "telemetry": {"schema_version": "1.0", "timings_ms": {}},
        "reference": None,
        "reference_context_ids": [],
        "captured_at": "2026-09-11T00:00:00Z",
    }
    encoded = json.dumps(
        record, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    record_path = tmp_path / "record.json"
    record_path.write_bytes(encoded)
    record_hash = hashlib.sha256(encoded).hexdigest()
    result = {
        "schema_version": "1.0",
        "status": "succeeded",
        "source": "e2e",
        "request_id": request_id,
        "conversation_id": conversation_id,
        "record_sha256": record_hash,
        "ragas_version": "0.4.3",
        "judge": {
            "provider": "ollama",
            "model": "local",
            "model_digest": "local-digest",
            "ollama_version": "test",
        },
        "metrics": [
            {"name": name, "status": "succeeded", "score": 0.5}
            for name in sorted(gate.MANDATORY_METRICS)
        ],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))
    return {
        "status": "succeeded",
        "record_path": str(record_path),
        "result_path": str(result_path),
        "record_sha256": record_hash,
    }


def _terminal_evidence_request(
    *,
    request_id: str,
    conversation_id: str,
    knowledge_id: str,
    policy_id: str,
    knowledge_text: str = "knowledge",
    policy_text: str = "policy",
) -> dict[str, object]:
    trace_session_id = str(uuid4())
    operation_id = str(uuid4())
    chat_session_id = str(uuid4())
    registry = DiagnosticTraceRegistry(USER, capture_mode="evaluation")
    install_registry(registry)
    try:
        registry.start(USER, trace_session_id, "live-gate")
        handle = registry.begin_operation(
            user_id=USER,
            session_id=trace_session_id,
            operation_id=operation_id,
            kind="chat_query",
            collection_type="conversations",
            wizard_id=chat_session_id,
        )
        assert handle is not None
        with activate_operation(handle):
            capture_evaluation_rewrite("official rewrite")
            capture_evaluation_contexts(
                ({"object_id": knowledge_id, "raw_text": knowledge_text},),
                ({"object_id": policy_id, "raw_text": policy_text},),
            )
        registry.finish(handle, "succeeded")
        payload = registry.snapshot(USER, trace_session_id, operation_id)
    finally:
        uninstall_registry(registry)
    return {
        "question": "question",
        "answer": "answer",
        "request_id": request_id,
        "conversation_id": conversation_id,
        "session_id": chat_session_id,
        "telemetry": {"schema_version": "1.0", "timings_ms": {}},
        "evaluation_evidence": {
            "schema_version": payload["schema_version"],
            "session_id": trace_session_id,
            "operation": payload["operations"][0],
        },
    }


def test_live_ragas_success_requires_all_mandatory_metrics(tmp_path: Path) -> None:
    observation = _write_successful_evaluation(tmp_path)
    assert gate.validate_evaluation_success(observation)["status"] == "passed"
    with pytest.raises(gate.GateError, match="record contract"):
        gate.validate_evaluation_success(observation, expected_source="rag_ask")
    result_path = Path(str(observation["result_path"]))
    result = json.loads(result_path.read_text())
    result["metrics"][0]["status"] = "failed"
    result_path.write_text(json.dumps(result))
    with pytest.raises(gate.GateError, match="mandatory metric failed"):
        gate.validate_evaluation_success(observation)


def test_live_evaluation_requires_exact_final_context_evidence(tmp_path: Path) -> None:
    request_id, conversation_id = str(uuid4()), str(uuid4())
    knowledge_id, policy_id = str(uuid4()), str(uuid4())
    observation = _write_successful_evaluation(
        tmp_path,
        request_id=request_id,
        conversation_id=conversation_id,
        knowledge_id=knowledge_id,
        policy_id=policy_id,
        knowledge_text="Knowledge ünicode",
        policy_text="Policy exact",
    )
    expected = _terminal_evidence_request(
        request_id=request_id,
        conversation_id=conversation_id,
        knowledge_id=knowledge_id,
        policy_id=policy_id,
        knowledge_text="Knowledge ünicode",
        policy_text="Policy exact",
    )
    assert gate.validate_evaluation_success(
        observation, expected_request=expected
    )["status"] == "passed"


def test_live_evaluation_rejects_context_not_in_final_qwen_evidence(
    tmp_path: Path,
) -> None:
    request_id, conversation_id = str(uuid4()), str(uuid4())
    knowledge_id, policy_id = str(uuid4()), str(uuid4())
    observation = _write_successful_evaluation(
        tmp_path,
        request_id=request_id,
        conversation_id=conversation_id,
        knowledge_id=knowledge_id,
        policy_id=policy_id,
    )
    expected = _terminal_evidence_request(
        request_id=request_id,
        conversation_id=conversation_id,
        knowledge_id=str(uuid4()),  # a pre-budget/decoy ID must never substitute
        policy_id=policy_id,
    )
    with pytest.raises(gate.GateError, match="identity/order mismatch"):
        gate.validate_evaluation_success(observation, expected_request=expected)


def test_live_evaluation_rejects_hydrated_context_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    request_id, conversation_id = str(uuid4()), str(uuid4())
    knowledge_id, policy_id = str(uuid4()), str(uuid4())
    observation = _write_successful_evaluation(
        tmp_path,
        request_id=request_id,
        conversation_id=conversation_id,
        knowledge_id=knowledge_id,
        policy_id=policy_id,
        knowledge_text="substituted text",
    )
    expected = _terminal_evidence_request(
        request_id=request_id,
        conversation_id=conversation_id,
        knowledge_id=knowledge_id,
        policy_id=policy_id,
        knowledge_text="original exact text",
    )
    with pytest.raises(gate.GateError, match="fingerprint mismatch"):
        gate.validate_evaluation_success(observation, expected_request=expected)


def test_live_evaluation_rejects_rewritten_query_not_from_accepted_trace(
    tmp_path: Path,
) -> None:
    request_id, conversation_id = str(uuid4()), str(uuid4())
    knowledge_id, policy_id = str(uuid4()), str(uuid4())
    observation = _write_successful_evaluation(
        tmp_path,
        request_id=request_id,
        conversation_id=conversation_id,
        knowledge_id=knowledge_id,
        policy_id=policy_id,
    )
    record_path = Path(str(observation["record_path"]))
    record = json.loads(record_path.read_text())
    record["rewritten_query"] = "a different rewrite"
    encoded = json.dumps(
        record, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    record_path.write_bytes(encoded)
    record_hash = hashlib.sha256(encoded).hexdigest()
    observation["record_sha256"] = record_hash
    result_path = Path(str(observation["result_path"]))
    result = json.loads(result_path.read_text())
    result["record_sha256"] = record_hash
    result_path.write_text(json.dumps(result))
    expected = _terminal_evidence_request(
        request_id=request_id,
        conversation_id=conversation_id,
        knowledge_id=knowledge_id,
        policy_id=policy_id,
    )
    with pytest.raises(gate.GateError, match="rewritten-query evidence mismatch"):
        gate.validate_evaluation_success(observation, expected_request=expected)


def test_latency_gate_and_contention_overlap_contract() -> None:
    samples = []
    for index in range(30):
        samples.extend(
            [
                {"pair_key": str(index), "mode": "off", "question": f"q-{index}",
                 "query_identity": "a" * 64, "query_source_index": index,
                 "query_schedule_index": index, "query_repetition": 0,
                 "timings_ms": {name: 100.0 for name in gate.LATENCY_METRICS}},
                {"pair_key": str(index), "mode": "on", "question": f"q-{index}",
                 "query_identity": "a" * 64, "query_source_index": index,
                 "query_schedule_index": index, "query_repetition": 0,
                 "timings_ms": {name: 101.0 for name in gate.LATENCY_METRICS}},
            ]
        )
    assert gate.paired_latency_gate(samples)["status"] == "passed"
    donor_id, job_id = str(uuid4()), str(uuid4())
    contention = {
        "judge_activity": {"evaluation_job_id": job_id, "request_id": donor_id,
                           "record_sha256": "d" * 64, "sequence": 1,
                           "judge_id": "local", "started_monotonic": 1.0,
                           "finished_monotonic": 4.0, "success": True},
        "overlapping_evaluation_request": {
            "request_id": donor_id,
            "evaluation": {"evaluation_job_id": job_id, "record_sha256": "d" * 64},
        },
        "request_started_monotonic": 2.0, "request_done_monotonic": 3.0,
    }
    gate.validate_contention_sample(contention)
    with pytest.raises(gate.GateError, match="did not overlap"):
        contention["judge_activity"]["finished_monotonic"] = 1.5
        gate.validate_contention_sample(contention)


def test_latency_rejects_unpaired_data_instead_of_silently_dropping_it() -> None:
    rows = [
        {"pair_key": "paired", "mode": mode, "timings_ms": {name: 100 for name in gate.LATENCY_METRICS}}
        for mode in ("off", "on")
    ]
    rows.append({"pair_key": "missing-peer", "mode": "off"})
    with pytest.raises(gate.GateError, match="unpaired"):
        gate.paired_latency_gate(rows, minimum_pairs=1)


def test_live_timing_triples_cannot_pass_without_runtime_provenance() -> None:
    with pytest.raises(gate.GateError, match="identity is missing"):
        gate.validate_latency_schedule([
            {"pair_key": "one", "mode": "on", "timings_ms": {name: 1 for name in gate.LATENCY_METRICS}}
        ])


def test_contention_requires_actual_overlap_but_not_full_request_containment() -> None:
    donor_id, job_id = str(uuid4()), str(uuid4())
    gate.validate_contention_sample({
        "judge_activity": {"evaluation_job_id": job_id, "request_id": donor_id,
                           "record_sha256": "d" * 64, "sequence": 1,
                           "judge_id": "local", "started_monotonic": 1.0,
                           "finished_monotonic": 3.0, "success": True},
        "overlapping_evaluation_request": {
            "request_id": donor_id,
            "evaluation": {"evaluation_job_id": job_id, "record_sha256": "d" * 64},
        },
        "request_started_monotonic": 2.0, "request_done_monotonic": 4.0,
    })


def test_production_dependency_and_import_isolation() -> None:
    for directory in (gate.ROOT / "backend", gate.ROOT / "deployment"):
        for path in directory.rglob("*.py"):
            source = path.read_text()
            assert "import ragas" not in source
            assert "from ragas" not in source
            assert "import rag_evaluation" not in source
            assert "from rag_evaluation" not in source
    assert "ragas" not in (gate.ROOT / "backend/requirements.txt").read_text().casefold()
    assert "ragas" not in (gate.ROOT / "deployment/modal_runtime.py").read_text().casefold()
