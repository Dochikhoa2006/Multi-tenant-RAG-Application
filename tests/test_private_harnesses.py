from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from deployment import e2e_diagnostic as artifacts
from deployment import e2e_diagnostic_api as api
from deployment import evaluation_bridge as bridge
from deployment import ragctl, wizard_diagnostic as wizard
from test_e2e_diagnostic import _complete_state
from test_evaluation_bridge import _evidence_operation, USER, SESSION, OPERATION, CHAT_SESSION, REQUEST


def dataset(tmp_path: Path, count: int = 1):
    path = tmp_path / "external.jsonl"
    path.write_text("".join(json.dumps({"query_id": f"id-{i}", "question": f"PRIVATE QUERY {i}"}) + "\n" for i in range(count)))
    return artifacts.load_query_selection(path, tmp_path)


@pytest.mark.parametrize("extra", ["\n", '{"query_id":"id-0","question":"duplicate ID"}\n',
    '{"query_id":"other","question":"x","unknown":1}\n',
    '{"query_id":"other","question":"x","reference_context_ids":["bad"]}\n',
    '{"query_id":"other","question":"x","question":"y"}\n'])
def test_dataset_rejects_incomplete_or_ambiguous_records(tmp_path, extra):
    selection = dataset(tmp_path)
    with selection.source_path.open("a") as target:
        target.write(extra)
    with pytest.raises(artifacts.E2EDiagnosticError):
        artifacts.load_query_selection(selection.source_path, tmp_path)


def test_external_references_and_unicode_are_preserved(tmp_path):
    path = tmp_path / "private.jsonl"
    reference_id = str(uuid4())
    record = {"query_id": "q-1", "question": "  café 漢字?\n", "reference": "Exact gold", "reference_context_ids": [reference_id]}
    path.write_text(json.dumps(record) + "\n")
    selected = artifacts.load_query_selection(path, tmp_path).selected[0]
    assert selected.question == record["question"]
    assert selected.reference == "Exact gold"
    assert selected.reference_context_ids == (reference_id,)
    evidence = bridge.parse_request_evidence(_evidence_operation(), user_id=USER,
        trace_session_id=SESSION, operation_id=OPERATION, chat_session_id=CHAT_SESSION, request_id=REQUEST)
    payload = json.loads(bridge._worker_payload(config={}, user_id=USER, evidence=evidence,
        allowed_document_ids=None, source="e2e", request_id=REQUEST, conversation_id=str(uuid4()),
        original_query=selected.question, response="answer", telemetry={}, captured_at="2026-09-13T00:00:00Z",
        directory=tmp_path, stem="test", exact_names=False, evaluation_job_id=str(uuid4()),
        acceptance_activity_path=None, reference=selected.reference, reference_context_ids=selected.reference_context_ids))
    assert payload["reference"] == record["reference"]
    assert payload["reference_context_ids"] == [reference_id]


def _runner(monkeypatch, tmp_path, count, *, continuous=False, fault=None):
    state, _ = _complete_state(tmp_path)
    selection = dataset(tmp_path, count)
    run = artifacts.create_e2e_run(tmp_path / "outputs", state.diagnostic_user_id, selection, state.active, continuous=continuous)
    recorder = artifacts.RequestRecorder(run.requests_path)
    recorder.dataset = (selection, state.active, {"synthetic": True})
    calls = []
    active = set()
    created = []

    class Trace:
        def __init__(self, client, url, user, run_id):
            self.session_id = str(uuid4())
            self.started = False

        def start(self):
            assert not active
            active.add(self.session_id)
            self.started = True
            calls.append(("start", self.session_id))
            if fault == "ambiguous_start":
                raise api.DeepTraceContractError("ambiguous")

        def delete(self, **kwargs):
            if not self.started:
                return
            calls.append(("delete", self.session_id))
            if fault == "delete":
                raise api.DeepTraceContractError("delete failed")
            active.remove(self.session_id)
            self.started = False

    def transport(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.method == "POST" and request.url.path == "/api/chat/sessions":
            identifier = str(uuid4())
            created.append(identifier)
            return httpx.Response(201, json={"session_id": identifier, "user_id": state.diagnostic_user_id})
        raise AssertionError(str(request.url))

    real_client = httpx.Client
    monkeypatch.setattr(api.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(transport), **kwargs))
    monkeypatch.setattr(api, "_DeepTraceSession", Trace)
    monkeypatch.setattr(api, "verify_physical_corpus", lambda *args: {"knowledge": 1, "policy": 1})
    monkeypatch.setattr(api, "_delete_chat_session", lambda c, u, owner, sid: calls.append(("cleanup", sid)) or {"status": "succeeded"})

    def execute(client, url, user, sid, query, trace, config, directory, allowed, **kwargs):
        assert user == state.diagnostic_user_id and trace.session_id in active
        assert kwargs["operation_id"] is not None
        calls.append(("query", query.source_index, trace.session_id, kwargs["operation_id"], sid))
        if fault == "interrupt" and query.source_index == 1:
            raise KeyboardInterrupt
        base = api._session_failure_result(query, started_at=artifacts.utc_timestamp(),
            started_clock=api.perf_counter(), code="QUERY_ERROR", http_status=200, scope="individual")
        failed = fault in {"individual", "systemic"} and query.source_index == 1
        return replace(base, status="failed" if failed else "succeeded",
            failure_code="QUERY_ERROR" if failed else None,
            failure_stage="query" if failed else None,
            failure_scope=("systemic" if fault == "systemic" else "individual") if failed else None,
            session_id=sid, request_id=str(uuid4()), conversation_id=str(uuid4()),
            diagnostic_operation_id=kwargs["operation_id"], query_http_attempted=True,
            answer="PRIVATE ANSWER", answer_complete=True, duration_ms=10.0,
            timings_ms={"ttft": 1.0, "generation": 2.0, "total_request": 3.0},
            deep_trace={"session_id": trace.session_id, "operation": {"texts": {"granite_rewritten_query": "PRIVATE REWRITE"}, "stages": {}}},
            registry_verification={"title": "PRIVATE TITLE"},
            evaluation={"status": "failed" if fault == "evaluation" else "succeeded", "evaluation_ms": 100.0})

    monkeypatch.setattr(api, "_execute_query", execute)
    progress = {}
    def invoke():
        return api.run_e2e_phase_2d("https://runtime", {}, {}, state, state.active,
            selection, recorder, progress, continuous=continuous, run_id=run.run_id, dataset_evaluation=True)
    return invoke, calls, recorder, run, selection, created, active


@pytest.mark.parametrize("count", [1, 64, 65, 256, 500, 512])
@pytest.mark.parametrize("continuous", [False, True])
def test_every_query_once_with_bounded_rotation(monkeypatch, tmp_path, capsys, count, continuous):
    invoke, calls, recorder, run, selection, created, active = _runner(monkeypatch, tmp_path, count, continuous=continuous)
    invoke()
    queries = [call for call in calls if call[0] == "query"]
    assert [call[1] for call in queries] == list(range(count))
    assert len({call[3] for call in queries}) == count
    traces = [call for call in calls if call[0] in {"start", "delete"}]
    assert [call[0] for call in traces] == ["start", "delete"] * ((count + 63) // 64)
    assert all(sum(q[2] == trace[1] for q in queries) <= 64 for trace in traces[::2])
    assert len(created) == (1 if continuous else count)
    assert {call[1] for call in calls if call[0] == "cleanup"} == set(created)
    assert not active
    rows = [json.loads(line) for line in run.requests_path.read_text().splitlines()]
    assert len(rows) == count and [r["query_id"] for r in rows] == [q.query_id for q in selection.selected]
    assert all(r["diagnostic_trace_session_id"] == q[2] for r, q in zip(rows, queries))
    assert "PRIVATE" not in capsys.readouterr().out
    assert "PRIVATE" not in run.requests_path.read_text()
    assert run.directory.stat().st_mode & 0o777 == 0o700
    assert run.requests_path.stat().st_mode & 0o777 == 0o600
    assert run.summary_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("fault,attempted", [("individual", 65), ("systemic", 2),
    ("ambiguous_start", 0), ("delete", 64), ("interrupt", 2), ("evaluation", 65)])
def test_rotation_failure_accounting_and_cleanup(monkeypatch, tmp_path, fault, attempted):
    invoke, calls, recorder, run, selection, created, active = _runner(monkeypatch, tmp_path, 65, fault=fault)
    with pytest.raises((api.E2EDiagnosticError, KeyboardInterrupt)):
        invoke()
    rows = [json.loads(line) for line in run.requests_path.read_text().splitlines()]
    assert sum(row["query_http_attempted"] for row in rows) == attempted
    assert len(rows) == 65
    assert len({row["source_index"] for row in rows}) == 65
    assert {c[1] for c in calls if c[0] == "cleanup"} == set(created)
    if fault == "delete":
        assert sum(c[0] == "start" for c in calls) == 1
    else:
        assert not active
    if fault == "evaluation":
        assert all(row["status"] == "succeeded" for row in rows)


def test_statistics_are_reproducible_and_reject_invalid_values():
    assert artifacts._finite_samples([], 2)["mean"] is None
    one = artifacts._finite_samples([3.5], 2)
    assert one["p99"] == 3.5 and one["population_variance"] == 0.0 and one["null_count"] == 1
    values = artifacts._finite_samples([1.0, 2.0, 3.0, None], 4)
    assert values["p50"] == 2.0 and values["p95"] == 3.0
    assert values["population_variance"] == 0.666667
    for invalid in (True, -1, float("nan"), float("inf")):
        with pytest.raises(artifacts.E2EDiagnosticError):
            artifacts._finite_samples([invalid], 1)


def test_finished_dataset_fills_rows_after_startup_failure(tmp_path):
    state, _ = _complete_state(tmp_path)
    selection = dataset(tmp_path, 3)
    run = artifacts.create_e2e_run(tmp_path / "output", state.diagnostic_user_id, selection, state.active, continuous=False)
    recorder = artifacts.RequestRecorder(run.requests_path)
    recorder.dataset = (selection, state.active, {})
    artifacts.update_e2e_summary(run, recorder, status="failed", pre_down_status="succeeded",
        up_status="failed", down_status="managed_by_up_failure_cleanup", physical_corpus_status="pending", finished=True)
    report = json.loads(run.dataset_report_path.read_text())
    assert report["accounting"] == {"intended": 3, "attempted": 0, "succeeded": 0,
        "failed": 0, "not_attempted": 3, "row_count": 3, "silently_dropped": 0, "identity_complete": True}
    assert "PRIVATE" not in run.dataset_report_path.read_text()
    assert run.dataset_report_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("entry", [".hidden", "nested", "file.pdf", "duplicate", "empty", "symlink", "root_extra"])
def test_strict_private_fixture_inventory(monkeypatch, tmp_path, entry):
    _complete_state(tmp_path)
    root = tmp_path / "fixtures"
    (root / "knowledge" / ".gitkeep").touch()
    wizard.preflight_wizard_fixtures(root, tmp_path, strict_private=True)
    if entry == "nested":
        (root / "policy" / entry).mkdir()
    elif entry == "symlink":
        (root / "policy" / "link.txt").symlink_to(root / "knowledge" / "fact.txt")
    elif entry == "duplicate":
        (root / "policy" / "same.txt").write_text("fact")
    elif entry == "empty":
        (root / "policy" / "empty.txt").write_text("   ")
    elif entry == "root_extra":
        (root / "unexpected.txt").write_text("extra")
    else:
        (root / "policy" / entry).write_text("data")
    with pytest.raises(wizard.WizardDiagnosticError):
        wizard.preflight_wizard_fixtures(root, tmp_path, strict_private=True)


def test_consolidated_ingestion_report_reuses_exact_existing_evidence(tmp_path):
    state, _ = _complete_state(tmp_path)
    fixtures = wizard.preflight_wizard_fixtures(tmp_path / "fixtures", tmp_path, strict_private=True)
    run = wizard.create_diagnostic_run(tmp_path / "outputs", state.diagnostic_user_id, fixtures)
    active = replace(state.active, generation_id=run.run_id)
    state = replace(state, active=active)
    state_path = tmp_path / "outputs" / "corpus-state.json"
    wizard.write_corpus_state(state_path, state)
    recorder = wizard.OperationRecorder(run.operations_path)
    for doc in active.documents:
        recorder.record(f"corpus.{doc.collection}.save.postcondition", "passed",
            selected_ids={"wizard_id": doc.wizard_id}, lateon_dimension=128, gte_dimension=768,
            trace={"counts": {"final_paragraph_count": 1, "final_chunk_count": 1},
                   "stages": {"save.stage": {"total_ms": 12.0}}, "texts": {"raw": "PRIVATE DOC"}})
        recorder.record("corpus.verify.final", "passed", collection=doc.collection)
    wizard.update_run_summary(run, status="succeeded", pre_down_status="succeeded", up_status="succeeded",
        down_status="succeeded", finished=True, scratch_status="passed", corpus_action="created")
    wizard.write_ingestion_report(run, fixtures, state_path)
    report = json.loads(run.ingestion_report_path.read_text())
    assert report["status"] == "succeeded" and report["source"]["verified_files"] == 2
    assert report["collections"]["knowledge"]["chunk_ids"] == list(active.document("knowledge").chunk_ids)
    assert report["collections"]["policy"]["paragraph_count"] == 1
    assert "PRIVATE DOC" not in run.ingestion_report_path.read_text()
    assert run.directory.stat().st_mode & 0o777 == 0o700
    for path in (run.ingestion_report_path, run.operations_path, run.summary_path, state_path):
        assert path.stat().st_mode & 0o777 == 0o600
    reused = wizard.create_diagnostic_run(run.directory.parent, state.diagnostic_user_id, fixtures)
    reuse_ops = wizard.OperationRecorder(reused.operations_path)
    for collection in ("knowledge", "policy"):
        reuse_ops.record("corpus.verify.active", "passed", collection=collection)
    wizard.update_run_summary(reused, status="succeeded", pre_down_status="succeeded", up_status="succeeded", down_status="succeeded", finished=True)
    wizard.write_ingestion_report(reused, fixtures, state_path)
    assert json.loads(reused.ingestion_report_path.read_text())["collections"]["knowledge"]["paragraph_count"] == 1


def test_dataset_cli_rejects_slicing_before_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(ragctl, "load_dotenv", lambda _: {})
    monkeypatch.setattr(ragctl, "diagnose_e2e", lambda *a, **k: pytest.fail("must fail before lifecycle"))
    assert ragctl.main(["diagnose", "e2e", "--dataset-evaluation", "--queries", str(tmp_path / "x.jsonl"), "--start", "0"]) == 1


@pytest.mark.parametrize("empty_role", [None, "knowledge", "policy"])
@pytest.mark.parametrize("gold", [False, True])
def test_dataset_requires_exact_final_evidence_and_applicable_metrics(monkeypatch, tmp_path, empty_role, gold):
    import test_evaluation_bridge as fixtures
    capture = fixtures.capture_evaluation_contexts
    monkeypatch.setattr(fixtures, "capture_evaluation_contexts", lambda k, p: capture(
        () if empty_role == "knowledge" else k, () if empty_role == "policy" else p))
    operation = _evidence_operation()
    evidence = bridge.parse_request_evidence(operation, user_id=USER, trace_session_id=SESSION,
        operation_id=OPERATION, chat_session_id=CHAT_SESSION, request_id=REQUEST)
    query = artifacts.SelectedQuery(0, "Exact question ü", "stable-id",
        "Genuine supplied reference" if gold else None, (str(uuid4()),) if gold else ())
    contexts = bridge.ResolvedContexts(
        () if empty_role == "knowledge" else ("Knowledge ünicode", "Knowledge second"),
        () if empty_role == "policy" else ("Policy exact text",))
    record = bridge.build_evaluation_record(source="e2e", request_id=REQUEST,
        conversation_id=str(uuid4()), original_query=query.question, response="Exact SSE answer ü",
        telemetry={"schema_version": "1.0", "timings_ms": {"ttft": 2.0}}, evidence=evidence,
        contexts=contexts, captured_at="2026-09-13T00:00:00Z", reference=query.reference,
        reference_context_ids=query.reference_context_ids)
    digest = hashlib.sha256(bridge._canonical_bytes(record)).hexdigest()
    mandatory = ["faithfulness", "response_relevancy", "context_utilization"]
    if gold:
        mandatory += ["context_recall", "context_precision_with_reference", "factual_correctness"]
    result = {"schema_version": "1.0", "status": "succeeded", "source": "e2e",
        "request_id": REQUEST, "conversation_id": record["conversation_id"], "record_sha256": digest,
        "ragas_version": "0.4.3", "judge": {"model": "qwen3.5:4b"}, "embeddings": {"dimension": 384},
        "metrics": [{"name": name, "status": "succeeded", "score": 0.75, "duration_ms": 10.0} for name in mandatory]}
    record_path, result_path = tmp_path / "record.json", tmp_path / "result.json"
    record_path.write_text(json.dumps(record))
    result_path.write_text(json.dumps(result))
    observation = {"status": "succeeded", "record_path": str(record_path), "result_path": str(result_path), "record_sha256": digest}
    expected = {"request_id": REQUEST, "conversation_id": record["conversation_id"],
        "session_id": CHAT_SESSION, "answer": record["response"], "telemetry": record["telemetry"],
        "deep_trace": {"session_id": SESSION, "operation": operation}}
    validate = lambda: api._validate_dataset_evaluation(observation, query, expected, tmp_path)
    assert validate() == observation
    for mutation in ("metric_missing", "metric_duplicate", "bad_score", "reference", "answer", "context", "decoy"):
        changed_record, changed_result = json.loads(json.dumps(record)), json.loads(json.dumps(result))
        if mutation == "metric_missing":
            changed_result["metrics"].pop()
        elif mutation == "metric_duplicate":
            changed_result["metrics"].append(changed_result["metrics"][0])
        elif mutation == "bad_score":
            changed_result["metrics"][0]["score"] = True
        elif mutation == "reference":
            changed_record["reference"] = "Substituted reference"
        elif mutation == "answer":
            changed_record["response"] = "Different SSE answer"
        elif mutation == "context":
            changed_record["retrieved_contexts"] = ["pre-budget context"]
        else:
            changed_record["knowledge_context_ids"] = [str(uuid4())]
        changed_digest = hashlib.sha256(bridge._canonical_bytes(changed_record)).hexdigest()
        observation["record_sha256"] = changed_digest
        changed_result["record_sha256"] = changed_digest
        record_path.write_text(json.dumps(changed_record))
        result_path.write_text(json.dumps(changed_result))
        assert validate()["error_code"] == "MANDATORY_EVALUATION_FAILED"


def test_dataset_report_has_safe_metrics_accounting_and_separate_timing(monkeypatch, tmp_path):
    invoke, calls, recorder, run, selection, created, active = _runner(monkeypatch, tmp_path, 2)
    invoke()
    rows = [json.loads(line) for line in run.requests_path.read_text().splitlines()]
    for i, row in enumerate(rows):
        result_path = run.directory / f"result-{i}.json"
        result_path.write_text(json.dumps({"schema_version": "1.0", "status": "succeeded",
            "request_id": row["request_id"], "conversation_id": row["conversation_id"], "record_sha256": "abc",
            "metrics": [{"name": name, "status": "succeeded", "score": i, "duration_ms": 50.0}
                        for name in ("faithfulness", "response_relevancy", "context_utilization")]}))
        row["evaluation"].update(result_path=str(result_path), record_sha256="abc",
                                 queue_wait_ms=300.0, execution_ms=600.0, evaluation_ms=900.0)
    run.requests_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {"status": "succeeded", "lifecycle": {}, "started_at": "now", "finished_at": "later"}
    artifacts.write_dataset_report(run, recorder, summary)
    report = json.loads(run.dataset_report_path.read_text())
    assert report["status"] == "succeeded" and report["accounting"]["succeeded"] == 2
    assert report["ragas_statistics"]["faithfulness"]["scores"]["mean"] == 0.5
    assert report["evaluation_statistics"]["evaluation_ms"]["mean"] == 900.0
    assert report["latency_statistics"]["all_attempts"]["diagnostic.duration_ms"]["mean"] == 10.0
    assert "evaluation" not in json.dumps(report["latency_statistics"])
    assert "PRIVATE" not in run.dataset_report_path.read_text()
    result_path.write_text(result_path.read_text().replace('"score": 1', '"score": true'))
    artifacts.write_dataset_report(run, recorder, summary)
    invalid = json.loads(run.dataset_report_path.read_text())
    assert invalid["status"] == "failed" and invalid["accounting"]["failed"] == 1
    result_path.write_text('{"metrics": null}')
    artifacts.write_dataset_report(run, recorder, summary)
    invalid = json.loads(run.dataset_report_path.read_text())
    assert invalid["status"] == "failed" and invalid["accounting"]["row_count"] == 2
    assert invalid["ragas_statistics"]["faithfulness"]["applicable"] == 2
    assert invalid["ragas_statistics"]["faithfulness"]["failed"] == 1
    rows[0]["query_id"] = "wrong-query"
    run.requests_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(artifacts.E2EDiagnosticError, match="identity"):
        artifacts.write_dataset_report(run, recorder, summary)


def test_bootstrap_cli_override_is_temporary_and_standalone_is_unchanged(monkeypatch, tmp_path):
    config = {"RAG_USER_ID": "primary", "RAG_DIAGNOSTIC_USER_ID": "separate_diagnostic"}
    monkeypatch.setattr(ragctl, "load_dotenv", lambda _: config)
    captured = []
    monkeypatch.setattr(ragctl, "diagnose_wizard", lambda c, r, f, **k: captured.append((c, k)))
    base = ["diagnose", "wizard", "--fixtures", str(tmp_path)]
    assert ragctl.main(base + ["--bootstrap-primary-user-collections"]) == 0
    assert captured[-1][0]["RAG_DIAGNOSTIC_USER_ID"] == "primary"
    assert captured[-1][1]["bootstrap_primary_user_collections"] is True
    assert config["RAG_DIAGNOSTIC_USER_ID"] == "separate_diagnostic"
    assert ragctl.main(base) == 0
    assert captured[-1][0] == config and captured[-1][1]["bootstrap_primary_user_collections"] is False


def test_dataset_identity_inspection_is_offline_and_secret_free(monkeypatch, tmp_path):
    import subprocess
    original = subprocess.run
    def inspect(command, **kwargs):
        if command[-1] == "preflight":
            return SimpleNamespace(returncode=0)
        return original(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", inspect)
    state, _ = _complete_state(tmp_path)
    identity = artifacts.dataset_identities({"MODAL_TOKEN_SECRET": "NEVER-PRINT-SECRET"}, state)
    assert "NEVER-PRINT-SECRET" not in json.dumps(identity)
    assert identity["corpus_owner"] == state.diagnostic_user_id
    assert identity["evaluator_configuration"]["reasoning_effort"] == "none"
    assert len(identity["effective_configuration_sha256"]) == 64


@pytest.mark.parametrize("remaining_status", [200, 404])
def test_trace_rotation_confirms_deletion_before_releasing_session(remaining_status):
    calls = []
    def transport(request):
        calls.append(request.method)
        return httpx.Response(204 if request.method == "DELETE" else remaining_status)
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        trace = api._DeepTraceSession(client, "https://runtime", USER, "run")
        trace.started = True
        if remaining_status == 200:
            with pytest.raises(api.DeepTraceContractError, match="confirmed"):
                trace.delete(confirm_absent=True)
            assert trace.started
        else:
            trace.delete(confirm_absent=True)
            assert not trace.started
    assert calls == ["DELETE", "GET"]


def test_known_chat_session_cleanup_is_polled_and_verified(monkeypatch):
    task_id = str(uuid4())
    calls = []
    def transport(request):
        assert request.url.params["user_id"] == USER
        assert request.url.path == f"/api/chat/sessions/{CHAT_SESSION}"
        calls.append(request.method)
        return httpx.Response(202, json={"task_id": task_id}) if request.method == "DELETE" else httpx.Response(404)
    def poll(client, url, user, identifier, operation):
        assert (user, identifier, operation) == (USER, task_id, "delete_session")
        calls.append("poll")
        return SimpleNamespace(status="succeeded", artifact=lambda: {"status": "succeeded", "task_id": identifier})
    monkeypatch.setattr(api, "poll_task", poll)
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        assert api._delete_chat_session(client, "https://runtime", USER, CHAT_SESSION)["status"] == "succeeded"
    assert calls == ["DELETE", "poll", "GET"]


def test_public_launcher_help_works_without_pythonpath_or_repository_cwd(tmp_path):
    import os
    import subprocess
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    for subcommand, flag in (("wizard", "--bootstrap-primary-user-collections"), ("e2e", "--dataset-evaluation")):
        result = subprocess.run([str(ragctl.PROJECT_ROOT / "rag"), "diagnose", subcommand, "--help"],
            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert flag in result.stdout
