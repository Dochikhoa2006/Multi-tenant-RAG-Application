from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import pytest

from backend.wizard.diagnostics import (
    DiagnosticTraceRegistry,
    EVALUATION_SESSION_CAPACITY,
    TRACE_SESSION_MAX_OPERATIONS,
    active_trace_handle,
    activate_operation,
    add_count,
    attach_related_task,
    capture_evaluation_contexts,
    capture_evaluation_rewrite,
    finish_operation,
    install_registry,
    observe_stage,
    uninstall_registry,
)
from deployment import evaluation_bridge as bridge


USER = "evaluation_user"
SESSION = "10000000-0000-4000-8000-000000000001"
OPERATION = "10000000-0000-4000-8000-000000000002"
CHAT_SESSION = "10000000-0000-4000-8000-000000000003"
REQUEST = "10000000-0000-4000-8000-000000000004"
KNOWLEDGE_ID = "20000000-0000-4000-8000-000000000001"
KNOWLEDGE_ID_2 = "20000000-0000-4000-8000-000000000003"
POLICY_ID = "20000000-0000-4000-8000-000000000002"
KNOWLEDGE_DOCUMENT = "30000000-0000-4000-8000-000000000001"
POLICY_DOCUMENT = "30000000-0000-4000-8000-000000000002"


def _evidence_operation() -> dict[str, object]:
    registry = DiagnosticTraceRegistry(USER, capture_mode="evaluation")
    install_registry(registry)
    try:
        registry.start(USER, SESSION, "ask-run")
        handle = registry.begin_operation(
            user_id=USER,
            session_id=SESSION,
            operation_id=OPERATION,
            kind="chat_query",
            collection_type="conversations",
            wizard_id=CHAT_SESSION,
        )
        assert handle is not None
        with activate_operation(handle):
            assert active_trace_handle() is None
            with observe_stage("chat.qwen_http"):
                pass
            add_count("qwen_attempt_count", 1)
            attach_related_task(handle, "conversation_persistence", str(uuid4()))
            capture_evaluation_rewrite("official rewritten query")
            capture_evaluation_contexts(
                (
                    {"object_id": KNOWLEDGE_ID, "raw_text": "Knowledge ünicode"},
                    {"object_id": KNOWLEDGE_ID_2, "raw_text": "Knowledge second"},
                ),
                ({"object_id": POLICY_ID, "raw_text": "Policy exact text"},),
            )
        with pytest.raises(KeyError):
            registry.snapshot(USER, SESSION, OPERATION)
        session_snapshot = registry.snapshot(USER, SESSION)
        assert session_snapshot["operation_count"] == 1
        assert session_snapshot["operations"] == []
        finish_operation(handle, "succeeded")
        payload = registry.snapshot(USER, SESSION, OPERATION)
        assert payload["capture_mode"] == "evaluation"
        operation = payload["operations"][0]
        assert operation["stages"] == {}
        assert operation["related_tasks"] == {}
        assert "qwen_attempt_count" not in operation["counts"]
        rendered = json.dumps(payload, ensure_ascii=False)
        assert "Knowledge ünicode" not in rendered
        assert "Policy exact text" not in rendered
        return operation
    finally:
        uninstall_registry(registry)


def test_evaluation_capture_is_terminal_bounded_and_metadata_only() -> None:
    operation = _evidence_operation()
    evidence = bridge.parse_request_evidence(
        operation,
        user_id=USER,
        trace_session_id=SESSION,
        operation_id=OPERATION,
        chat_session_id=CHAT_SESSION,
        request_id=REQUEST,
    )
    assert evidence.rewritten_query == "official rewritten query"
    assert evidence.knowledge.ids == (KNOWLEDGE_ID, KNOWLEDGE_ID_2)
    assert evidence.policy.ids == (POLICY_ID,)


def test_evaluation_profile_rejects_non_chat_operations() -> None:
    registry = DiagnosticTraceRegistry(USER, capture_mode="evaluation")
    registry.start(USER, SESSION, "ask-run")
    assert (
        registry.begin_operation(
            user_id=USER,
            session_id=SESSION,
            operation_id=OPERATION,
            kind="save",
            collection_type="knowledge_facts",
            wizard_id=CHAT_SESSION,
        )
        is None
    )


def test_evaluation_sessions_are_concurrent_scoped_and_independently_bounded() -> None:
    assert EVALUATION_SESSION_CAPACITY == 256
    assert TRACE_SESSION_MAX_OPERATIONS == 256
    registry = DiagnosticTraceRegistry(USER, capture_mode="evaluation")
    first_session = str(uuid4())
    second_session = str(uuid4())
    malformed_session = str(uuid4())
    registry.start(USER, first_session, "ask-1")
    registry.start(USER, second_session, "ask-2")
    registry.start(USER, malformed_session, "ask-malformed")
    first_operation = str(uuid4())
    second_operation = str(uuid4())
    first = registry.begin_operation(
        user_id=USER,
        session_id=first_session,
        operation_id=first_operation,
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    )
    second = registry.begin_operation(
        user_id=USER,
        session_id=second_session,
        operation_id=second_operation,
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    )
    assert first is not None and second is not None
    registry.mark_fault(first)
    registry.finish(first, "failed")
    registry.finish(second, "succeeded")
    assert registry.begin_operation(
        user_id=USER,
        session_id=malformed_session,
        operation_id="not-a-uuid",
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    ) is None
    assert registry.snapshot(USER, first_session)["trace_faulted"] is True
    assert registry.snapshot(USER, second_session)["trace_faulted"] is False
    assert registry.snapshot(USER, malformed_session)["trace_faulted"] is True
    registry.delete(USER, first_session)
    assert registry.snapshot(USER, second_session, second_operation)["operations"]


def test_evaluation_session_capacity_is_separate_from_deep_operation_limit() -> None:
    registry = DiagnosticTraceRegistry(USER, capture_mode="evaluation")
    sessions = [str(uuid4()) for _ in range(EVALUATION_SESSION_CAPACITY)]
    for sequence, session_id in enumerate(sessions):
        registry.start(USER, session_id, f"ask-{sequence}")
    with pytest.raises(RuntimeError, match="capacity"):
        registry.start(USER, str(uuid4()), "overflow")

    registry.delete(USER, sessions[0])
    replacement = str(uuid4())
    registry.start(USER, replacement, "replacement")
    first_operation = registry.begin_operation(
        user_id=USER,
        session_id=replacement,
        operation_id=str(uuid4()),
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    )
    assert first_operation is not None
    assert registry.begin_operation(
        user_id=USER,
        session_id=replacement,
        operation_id=str(uuid4()),
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    ) is None
    assert registry.snapshot(USER, replacement)["overflowed"] is True


class _Collections:
    def __init__(self, objects: dict[str, list[object]]) -> None:
        self.objects = objects
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def exists(self, name: str) -> bool:
        return name in self.objects

    def use(self, name: str) -> object:
        owner = self

        class Query:
            def fetch_objects_by_ids(self, *args: object, **kwargs: object) -> object:
                owner.calls.append((name, args, dict(kwargs)))
                return SimpleNamespace(objects=owner.objects[name])

        return SimpleNamespace(query=Query())


class _Storage:
    collections: _Collections

    def __init__(self, _config: object, _user: str) -> None:
        self.manager = SimpleNamespace(client=SimpleNamespace(collections=self.collections))

    def __enter__(self) -> "_Storage":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _stored(chunk_id: str, document_id: str, text: str) -> object:
    return SimpleNamespace(
        uuid=chunk_id,
        properties={
            "user_id": USER,
            "document_id": document_id,
            "paragraph_id": 1,
            "chunk_id": chunk_id,
            "raw_text": text,
        },
    )


def test_exact_context_resolution_is_ordered_vector_free_and_document_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = bridge.parse_request_evidence(
        _evidence_operation(),
        user_id=USER,
        trace_session_id=SESSION,
        operation_id=OPERATION,
        chat_session_id=CHAT_SESSION,
        request_id=REQUEST,
    )
    from backend.config import get_collection_name

    collections = _Collections(
        {
            get_collection_name(USER, "knowledge_facts"): [
                _stored(KNOWLEDGE_ID_2, KNOWLEDGE_DOCUMENT, "Knowledge second"),
                _stored(KNOWLEDGE_ID, KNOWLEDGE_DOCUMENT, "Knowledge ünicode")
            ],
            get_collection_name(USER, "policy"): [
                _stored(POLICY_ID, POLICY_DOCUMENT, "Policy exact text")
            ],
        }
    )
    _Storage.collections = collections
    monkeypatch.setattr(bridge, "CorpusStorage", _Storage)

    resolved = bridge.resolve_exact_contexts(
        {},
        user_id=USER,
        evidence=evidence,
        allowed_document_ids={
            "knowledge": frozenset({KNOWLEDGE_DOCUMENT}),
            "policy": frozenset({POLICY_DOCUMENT}),
        },
    )

    assert resolved.knowledge == ("Knowledge ünicode", "Knowledge second")
    assert resolved.policy == ("Policy exact text",)
    assert len(collections.calls) == 2
    for _, args, kwargs in collections.calls:
        assert args and len(args[0]) in {1, 2}
        assert kwargs["include_vector"] is False
        assert kwargs["return_properties"] == list(bridge._PROPERTIES)


def test_context_resolution_fails_closed_on_wrong_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = bridge.parse_request_evidence(
        _evidence_operation(),
        user_id=USER,
        trace_session_id=SESSION,
        operation_id=OPERATION,
        chat_session_id=CHAT_SESSION,
        request_id=REQUEST,
    )
    from backend.config import get_collection_name

    wrong = _stored(KNOWLEDGE_ID, KNOWLEDGE_DOCUMENT, "Knowledge ünicode")
    wrong.properties["user_id"] = "another_user"
    collections = _Collections(
        {
            get_collection_name(USER, "knowledge_facts"): [wrong],
            get_collection_name(USER, "policy"): [
                _stored(POLICY_ID, POLICY_DOCUMENT, "Policy exact text")
            ],
        }
    )
    _Storage.collections = collections
    monkeypatch.setattr(bridge, "CorpusStorage", _Storage)
    with pytest.raises(bridge.EvaluationBridgeError, match="storage identity"):
        bridge.resolve_exact_contexts({}, user_id=USER, evidence=evidence)


def test_record_preserves_raw_context_and_public_telemetry() -> None:
    evidence = bridge.parse_request_evidence(
        _evidence_operation(),
        user_id=USER,
        trace_session_id=SESSION,
        operation_id=OPERATION,
        chat_session_id=CHAT_SESSION,
        request_id=REQUEST,
    )
    telemetry = {"schema_version": "1.0", "timings_ms": {"ttft": 1.25}}
    record = bridge.build_evaluation_record(
        source="e2e",
        request_id=REQUEST,
        conversation_id="40000000-0000-4000-8000-000000000002",
        original_query="Question?",
        response="Answer.",
        telemetry=telemetry,
        evidence=evidence,
        contexts=bridge.ResolvedContexts(
            ("Knowledge ünicode", "Knowledge second"), ("Policy exact text",)
        ),
        captured_at="2026-09-11T00:00:00Z",
    )
    assert record["retrieved_contexts"] == [
        "Knowledge ünicode",
        "Knowledge second",
        "Policy exact text",
    ]
    assert record["context_roles"] == ["knowledge", "knowledge", "policy"]
    assert record["telemetry"] == telemetry
    assert record["reference"] is None

    with pytest.raises(bridge.EvaluationBridgeError, match="request correlation"):
        bridge.build_evaluation_record(
            source="e2e",
            request_id="40000000-0000-4000-8000-000000000009",
            conversation_id="40000000-0000-4000-8000-000000000002",
            original_query="Question?",
            response="Answer.",
            telemetry=telemetry,
            evidence=evidence,
            contexts=bridge.ResolvedContexts(
                ("Knowledge ünicode", "Knowledge second"),
                ("Policy exact text",),
            ),
            captured_at="2026-09-11T00:00:00Z",
        )


def test_private_write_is_restrictive_and_refuses_collision(tmp_path: Path) -> None:
    path = tmp_path / "private" / "record.json"
    digest = bridge.write_private_json(path, {"value": "ünicode"})
    assert len(digest) == 64
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    with pytest.raises(FileExistsError):
        bridge.write_private_json(path, {"value": "changed"})


def test_finished_evaluator_result_is_correlated_without_importing_ragas(
    tmp_path: Path,
) -> None:
    record_path = tmp_path / "record.json"
    result_path = tmp_path / "result.json"
    record_sha = "a" * 64
    request_id = "50000000-0000-4000-8000-000000000001"
    conversation_id = "50000000-0000-4000-8000-000000000002"
    bridge.write_private_json(
        result_path,
        {
            "schema_version": "1.0",
            "status": "succeeded",
            "record_sha256": record_sha,
            "source": "e2e",
            "request_id": request_id,
            "conversation_id": conversation_id,
        },
    )
    process = subprocess.Popen(["/usr/bin/true"])
    launch = bridge.EvaluationLaunch(
        started=bridge.perf_counter(),
        process=process,
        record_path=record_path,
        result_path=result_path,
        record_sha256=record_sha,
        source="e2e",
        request_id=request_id,
        conversation_id=conversation_id,
    )
    observation = bridge.finish_local_evaluation(launch)
    assert observation.status == "succeeded"
    assert observation.error_code is None


@pytest.mark.parametrize("status", ["partial", "failed"])
def test_finished_evaluator_preserves_non_success_result_status(
    tmp_path: Path, status: str
) -> None:
    record_sha = "b" * 64
    request_id = "50000000-0000-4000-8000-000000000003"
    conversation_id = "50000000-0000-4000-8000-000000000004"
    result_path = tmp_path / "result.json"
    bridge.write_private_json(
        result_path,
        {
            "schema_version": "1.0",
            "status": status,
            "record_sha256": record_sha,
            "source": "rag_ask",
            "request_id": request_id,
            "conversation_id": conversation_id,
        },
    )
    launch = bridge.EvaluationLaunch(
        started=bridge.perf_counter(),
        process=subprocess.Popen(["/usr/bin/true"]),
        record_path=tmp_path / "record.json",
        result_path=result_path,
        record_sha256=record_sha,
        source="rag_ask",
        request_id=request_id,
        conversation_id=conversation_id,
    )
    observation = bridge.finish_local_evaluation(launch)
    assert observation.status == status
    assert observation.error_code == "EVALUATION_INCOMPLETE"


def test_evaluator_crash_and_malformed_result_are_safe(tmp_path: Path) -> None:
    launch = bridge.EvaluationLaunch(
        started=bridge.perf_counter(),
        process=subprocess.Popen(["/usr/bin/false"]),
        record_path=tmp_path / "record.json",
        result_path=tmp_path / "missing.json",
        record_sha256="c" * 64,
        source="e2e",
        request_id="50000000-0000-4000-8000-000000000005",
        conversation_id="50000000-0000-4000-8000-000000000006",
    )
    assert bridge.finish_local_evaluation(launch).error_code == "EVALUATION_RESULT_INVALID"

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    launch = bridge.EvaluationLaunch(
        started=bridge.perf_counter(),
        process=subprocess.Popen(["/usr/bin/true"]),
        record_path=tmp_path / "record.json",
        result_path=malformed,
        record_sha256="d" * 64,
        source="e2e",
        request_id="50000000-0000-4000-8000-000000000007",
        conversation_id="50000000-0000-4000-8000-000000000008",
    )
    assert bridge.finish_local_evaluation(launch).error_code == "EVALUATION_RESULT_INVALID"


def test_evaluator_timeout_and_explicit_cancellation_reap_children(
    tmp_path: Path,
) -> None:
    def sleeping_launch() -> bridge.EvaluationLaunch:
        return bridge.EvaluationLaunch(
            started=bridge.perf_counter(),
            process=subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"]
            ),
            record_path=tmp_path / "record.json",
            result_path=tmp_path / "result.json",
            record_sha256="e" * 64,
            source="e2e",
            request_id="50000000-0000-4000-8000-000000000009",
            conversation_id="50000000-0000-4000-8000-000000000010",
        )

    timed = sleeping_launch()
    assert bridge.finish_local_evaluation(timed, timeout_seconds=0.001).error_code == (
        "EVALUATION_TIMEOUT"
    )
    assert timed.process is not None and timed.process.poll() is not None

    cancelled = sleeping_launch()
    bridge.cancel_local_evaluation(cancelled)
    assert cancelled.process is not None and cancelled.process.poll() is not None


def test_missing_evaluator_environment_and_exact_ask_names_are_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge, "EVALUATOR", tmp_path / "missing-rag-evaluate")
    record = {
        "source": "rag_ask",
        "request_id": "50000000-0000-4000-8000-000000000011",
        "conversation_id": "50000000-0000-4000-8000-000000000012",
    }
    launch = bridge.start_local_evaluation(
        record,
        tmp_path / "ask-run",
        "ignored",
        exact_names=True,
    )
    assert launch.process is None
    assert launch.error_code == "EVALUATION_START_FAILED"
    assert launch.record_path == tmp_path / "ask-run" / "record.json"
    assert launch.result_path == tmp_path / "ask-run" / "result.json"
    assert stat.S_IMODE(launch.record_path.stat().st_mode) == 0o600


def _external_lock_holder(
    lock_path: Path, ready: Path, release: Path
) -> subprocess.Popen[bytes]:
    script = (
        "import fcntl, os, pathlib, sys, time; "
        "path=pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True); "
        "fd=os.open(path, os.O_RDWR|os.O_CREAT, 0o600); "
        "fcntl.flock(fd, fcntl.LOCK_EX); pathlib.Path(sys.argv[2]).touch(); "
        "release=pathlib.Path(sys.argv[3]); "
        "exec('while not release.exists():\\n time.sleep(0.01)')"
    )
    return subprocess.Popen(
        [sys.executable, "-c", script, str(lock_path), str(ready), str(release)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_path(path: Path) -> None:
    deadline = time.monotonic() + 5.0
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"Timed out waiting for {path}")
        time.sleep(0.01)


def _submit_test_job(
    tmp_path: Path,
    evidence: bridge.RequestEvidence,
    lock_path: Path,
) -> bridge.EvaluationJob:
    return bridge.submit_local_evaluation(
        config={},
        user_id=USER,
        evidence=evidence,
        allowed_document_ids=None,
        source="e2e",
        request_id=REQUEST,
        conversation_id="50000000-0000-4000-8000-000000000013",
        original_query="Question?",
        response="Answer.",
        telemetry={"schema_version": "1.0", "timings_ms": {}},
        captured_at="2026-09-11T00:00:00Z",
        directory=tmp_path / "evaluation",
        stem="request",
        execution_lock_path=lock_path,
    )


def test_local_job_waits_cross_process_before_hydration_and_passes_lock_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = bridge.parse_request_evidence(
        _evidence_operation(),
        user_id=USER,
        trace_session_id=SESSION,
        operation_id=OPERATION,
        chat_session_id=CHAT_SESSION,
        request_id=REQUEST,
    )
    lock_path = tmp_path / "execution.lock"
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    locker = _external_lock_holder(lock_path, ready, release)
    _wait_for_path(ready)
    hydrated = Event()
    inherited_descriptors: list[int] = []

    def resolve(*_args: object, **_kwargs: object) -> bridge.ResolvedContexts:
        hydrated.set()
        return bridge.ResolvedContexts(("Knowledge",), ("Policy",))

    def build(**_kwargs: object) -> dict[str, object]:
        return {
            "source": "e2e",
            "request_id": REQUEST,
            "conversation_id": "50000000-0000-4000-8000-000000000013",
        }

    def start(
        _record: object,
        _directory: Path,
        _stem: str,
        *,
        exact_names: bool = False,
        execution_lock_fd: int | None = None,
    ) -> bridge.EvaluationLaunch:
        assert exact_names is False
        assert execution_lock_fd is not None
        inherited_descriptors.append(execution_lock_fd)
        return bridge.EvaluationLaunch(
            bridge.perf_counter(),
            None,
            None,
            None,
            "a" * 64,
            "e2e",
            REQUEST,
            "50000000-0000-4000-8000-000000000013",
        )

    monkeypatch.setattr(bridge, "resolve_exact_contexts", resolve)
    monkeypatch.setattr(bridge, "build_evaluation_record", build)
    monkeypatch.setattr(bridge, "start_local_evaluation", start)
    monkeypatch.setattr(
        bridge,
        "finish_local_evaluation",
        lambda _launch: bridge.EvaluationObservation(
            "succeeded", None, 1.0, None, None, "a" * 64
        ),
    )
    def supervised(
        _payload: bytes, *, descriptor: int, timeout_seconds: float, state: object
    ) -> dict[str, object]:
        assert timeout_seconds > 0
        hydrated.set()
        inherited_descriptors.append(descriptor)
        return {
            "status": "succeeded",
            "error_code": None,
            "record_path": None,
            "result_path": None,
            "record_sha256": "a" * 64,
        }

    monkeypatch.setattr(bridge, "_run_supervised_worker", supervised)

    job = _submit_test_job(tmp_path, evidence, lock_path)
    time.sleep(0.05)
    assert hydrated.is_set() is False
    competing = _submit_test_job(tmp_path, evidence, lock_path)
    assert bridge.finish_evaluation_job(competing).error_code == (
        "EVALUATION_LOCAL_JOB_BUSY"
    )
    release.touch()
    observation = bridge.finish_evaluation_job(job)
    locker.wait(timeout=5)
    assert observation.status == "succeeded"
    assert observation.queue_wait_ms > 0
    assert observation.duration_ms >= observation.queue_wait_ms
    assert hydrated.is_set() is True
    assert len(inherited_descriptors) == 1


def test_waiting_local_job_is_cancellable_before_hydration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = bridge.parse_request_evidence(
        _evidence_operation(),
        user_id=USER,
        trace_session_id=SESSION,
        operation_id=OPERATION,
        chat_session_id=CHAT_SESSION,
        request_id=REQUEST,
    )
    lock_path = tmp_path / "execution.lock"
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    locker = _external_lock_holder(lock_path, ready, release)
    _wait_for_path(ready)
    hydrated = Event()
    monkeypatch.setattr(
        bridge,
        "resolve_exact_contexts",
        lambda *_args, **_kwargs: hydrated.set(),
    )
    job = _submit_test_job(tmp_path, evidence, lock_path)
    bridge.cancel_evaluation_job(job)
    observation = bridge.finish_evaluation_job(job)
    release.touch()
    locker.wait(timeout=5)
    assert observation.error_code == "EVALUATION_CANCELLED"
    assert hydrated.is_set() is False


def test_waiting_local_job_recovers_lock_after_owner_process_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = bridge.parse_request_evidence(
        _evidence_operation(),
        user_id=USER,
        trace_session_id=SESSION,
        operation_id=OPERATION,
        chat_session_id=CHAT_SESSION,
        request_id=REQUEST,
    )
    lock_path = tmp_path / "execution.lock"
    ready = tmp_path / "ready"
    release = tmp_path / "never-released"
    locker = _external_lock_holder(lock_path, ready, release)
    _wait_for_path(ready)
    hydrated = Event()
    monkeypatch.setattr(
        bridge,
        "resolve_exact_contexts",
        lambda *_args, **_kwargs: (
            hydrated.set() or bridge.ResolvedContexts(("Knowledge",), ("Policy",))
        ),
    )
    monkeypatch.setattr(
        bridge,
        "build_evaluation_record",
        lambda **_kwargs: {
            "source": "e2e",
            "request_id": REQUEST,
            "conversation_id": "50000000-0000-4000-8000-000000000013",
        },
    )
    monkeypatch.setattr(
        bridge,
        "start_local_evaluation",
        lambda *_args, **_kwargs: bridge.EvaluationLaunch(
            bridge.perf_counter(),
            None,
            None,
            None,
            "a" * 64,
            "e2e",
            REQUEST,
            "50000000-0000-4000-8000-000000000013",
        ),
    )
    monkeypatch.setattr(
        bridge,
        "finish_local_evaluation",
        lambda _launch: bridge.EvaluationObservation(
            "succeeded", None, 1.0, None, None, "a" * 64
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_run_supervised_worker",
        lambda *_args, **_kwargs: (
            hydrated.set()
            or {
                "status": "succeeded",
                "error_code": None,
                "record_path": None,
                "result_path": None,
                "record_sha256": "a" * 64,
            }
        ),
    )

    job = _submit_test_job(tmp_path, evidence, lock_path)
    time.sleep(0.05)
    assert hydrated.is_set() is False
    locker.terminate()
    locker.wait(timeout=5)
    observation = bridge.finish_evaluation_job(job)

    assert observation.status == "succeeded"
    assert hydrated.is_set() is True


def test_failed_hydration_releases_local_slot_for_next_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = bridge.parse_request_evidence(
        _evidence_operation(),
        user_id=USER,
        trace_session_id=SESSION,
        operation_id=OPERATION,
        chat_session_id=CHAT_SESSION,
        request_id=REQUEST,
    )
    calls = 0

    def resolve(*_args: object, **_kwargs: object) -> bridge.ResolvedContexts:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise bridge.EvaluationBridgeError("synthetic hydration failure")
        return bridge.ResolvedContexts(("Knowledge",), ("Policy",))

    monkeypatch.setattr(bridge, "resolve_exact_contexts", resolve)
    monkeypatch.setattr(
        bridge,
        "build_evaluation_record",
        lambda **_kwargs: {
            "source": "e2e",
            "request_id": REQUEST,
            "conversation_id": "50000000-0000-4000-8000-000000000013",
        },
    )
    monkeypatch.setattr(
        bridge,
        "start_local_evaluation",
        lambda *_args, **_kwargs: bridge.EvaluationLaunch(
            bridge.perf_counter(),
            None,
            None,
            None,
            "a" * 64,
            "e2e",
            REQUEST,
            "50000000-0000-4000-8000-000000000013",
        ),
    )
    monkeypatch.setattr(
        bridge,
        "finish_local_evaluation",
        lambda _launch: bridge.EvaluationObservation(
            "succeeded", None, 1.0, None, None, "a" * 64
        ),
    )
    def supervised_failure_then_success(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "status": "failed" if calls == 1 else "succeeded",
            "error_code": "EVALUATION_EVIDENCE_INVALID" if calls == 1 else None,
            "record_path": None,
            "result_path": None,
            "record_sha256": None if calls == 1 else "a" * 64,
        }

    calls = 0
    monkeypatch.setattr(
        bridge, "_run_supervised_worker", supervised_failure_then_success
    )
    lock_path = tmp_path / "execution.lock"

    first = bridge.finish_evaluation_job(
        _submit_test_job(tmp_path, evidence, lock_path)
    )
    second = bridge.finish_evaluation_job(
        _submit_test_job(tmp_path, evidence, lock_path)
    )

    assert first.error_code == "EVALUATION_EVIDENCE_INVALID"
    assert second.status == "succeeded"
    assert calls == 2


def test_cancellation_during_hydration_does_not_spawn_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = bridge.parse_request_evidence(
        _evidence_operation(),
        user_id=USER,
        trace_session_id=SESSION,
        operation_id=OPERATION,
        chat_session_id=CHAT_SESSION,
        request_id=REQUEST,
    )
    hydration_started = Event()
    hydration_release = Event()
    evaluator_started = Event()

    def resolve(*_args: object, **_kwargs: object) -> bridge.ResolvedContexts:
        hydration_started.set()
        assert hydration_release.wait(timeout=5)
        return bridge.ResolvedContexts(("Knowledge",), ("Policy",))

    monkeypatch.setattr(bridge, "resolve_exact_contexts", resolve)
    monkeypatch.setattr(
        bridge,
        "start_local_evaluation",
        lambda *_args, **_kwargs: evaluator_started.set(),
    )
    def supervised_cancel(*_args: object, **_kwargs: object) -> dict[str, object]:
        hydration_started.set()
        assert hydration_release.wait(timeout=5)
        return {
            "status": "succeeded",
            "error_code": None,
            "record_path": None,
            "result_path": None,
            "record_sha256": "a" * 64,
        }

    monkeypatch.setattr(bridge, "_run_supervised_worker", supervised_cancel)
    job = _submit_test_job(tmp_path, evidence, tmp_path / "execution.lock")
    assert hydration_started.wait(timeout=5)
    bridge.cancel_evaluation_job(job)
    hydration_release.set()
    observation = bridge.finish_evaluation_job(job)

    assert observation.error_code == "EVALUATION_CANCELLED"
    assert evaluator_started.is_set() is False


def test_production_modules_do_not_import_ragas_or_evaluation_package() -> None:
    root = Path(__file__).resolve().parents[1]
    for directory in (root / "backend", root / "deployment"):
        for path in directory.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "import ragas" not in source
            assert "from ragas" not in source
            assert "import rag_evaluation" not in source
            assert "from rag_evaluation" not in source


def test_architecture_docs_describe_automatic_bounded_evaluation() -> None:
    root = Path(__file__).resolve().parents[1]
    audit = (root / "evaluation/ARCHITECTURE_AUDIT.md").read_text(encoding="utf-8")
    readme = (root / "evaluation/README.md").read_text(encoding="utf-8")
    combined = audit + readme
    assert "--evaluate" not in combined
    assert "TRACE_SESSION_MAX_OPERATIONS" in audit
    assert "EVALUATION_SESSION_CAPACITY" in audit
    assert "LOCAL_EVALUATION_CONCURRENCY" in audit
    assert "Every successful ordinary ask" in audit
    assert "Phase 2D boundary" in combined


def test_e2e_submits_before_remote_polling_and_joins_after_duration() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "deployment/e2e_diagnostic_api.py").read_text(encoding="utf-8")
    function = source[
        source.index("def _execute_query(") : source.index(
            "def _session_failure_result("
        )
    ]
    assert function.index("evaluation_job = submit_local_evaluation(") < function.index(
        "persistence_task = poll_task("
    )
    duration = function.index("duration_ms = (perf_counter() - started_clock)")
    joined = function.index("evaluation = finish_evaluation_job(")
    assert duration < joined


def test_e2e_duration_freezes_before_delayed_evaluation_join(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from deployment import e2e_diagnostic_api as e2e
    from deployment.e2e_diagnostic import SelectedQuery
    from deployment.ragctl import CHAT_TIMING_KEYS

    request_id = "60000000-0000-4000-8000-000000000001"
    conversation_id = "60000000-0000-4000-8000-000000000002"
    persistence_task_id = "60000000-0000-4000-8000-000000000003"
    title_task_id = "60000000-0000-4000-8000-000000000004"
    operation = {
        "flags": {"title_enqueue_accepted": True},
        "related_tasks": {
            "conversation_persistence": persistence_task_id,
            "session_title": title_task_id,
        }
    }
    clock = [0.0]

    class Response:
        status_code = 200
        headers = {
            "X-Request-ID": request_id,
            "Content-Type": "text/event-stream",
        }

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def iter_lines(self) -> object:
            timings = {name: 0.0 for name in CHAT_TIMING_KEYS}
            return iter(
                [
                    "event: token",
                    "data: " + json.dumps({"request_id": request_id, "text": "answer"}),
                    "",
                    "event: telemetry",
                    "data: "
                    + json.dumps(
                        {
                            "request_id": request_id,
                            "schema_version": "1.0",
                            "timings_ms": timings,
                        }
                    ),
                    "",
                    "event: done",
                    "data: "
                    + json.dumps(
                        {
                            "request_id": request_id,
                            "conversation_id": conversation_id,
                        }
                    ),
                    "",
                ]
            )

    class Client:
        def stream(self, *_args: object, **_kwargs: object) -> Response:
            return Response()

    class Trace:
        session_id = SESSION

        def wait_operation(self, _operation_id: str) -> tuple[object, object, object]:
            return {}, operation, {}

        def fetch_operation(self, _operation_id: str) -> tuple[object, object]:
            return {}, operation

    def poll_task(
        _client: object,
        _url: str,
        _user: str,
        task_id: str,
        task_operation: str,
    ) -> e2e.TaskObservation:
        clock[0] += 1.0
        if task_operation == "embed_conversation":
            started = "2026-09-11T00:00:00.100000+00:00"
            finished = "2026-09-11T00:00:01+00:00"
        else:
            started = "2026-09-11T00:00:01+00:00"
            finished = "2026-09-11T00:00:02+00:00"
        return e2e.TaskObservation(
            task_id,
            task_operation,
            "succeeded",
            None,
            "2026-09-11T00:00:00+00:00",
            started,
            finished,
            1,
            100.0,
            900.0,
            1000.0,
        )

    def verify(*_args: object, **_kwargs: object) -> dict[str, object]:
        clock[0] += 1.0
        return {"status": "succeeded", "title": "A Useful Session Title"}

    def finish(_job: object) -> bridge.EvaluationObservation:
        clock[0] += 100.0
        return bridge.EvaluationObservation(
            "succeeded", None, 123.0, None, None, "a" * 64, 17.0
        )

    monkeypatch.setattr(e2e, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        e2e, "validate_chat_trace_evidence", lambda *_a, **_k: operation
    )
    monkeypatch.setattr(e2e, "parse_request_evidence", lambda *_a, **_k: object())
    monkeypatch.setattr(e2e, "submit_local_evaluation", lambda **_kwargs: object())
    monkeypatch.setattr(e2e, "poll_task", poll_task)
    monkeypatch.setattr(e2e, "_trace_artifact", lambda *_args: {"operation": operation})
    monkeypatch.setattr(
        e2e,
        "validate_post_generation_evidence",
        lambda *_args, **_kwargs: "A Useful Session Title",
    )
    monkeypatch.setattr(e2e, "verify_session_postcondition", verify)
    monkeypatch.setattr(e2e, "finish_evaluation_job", finish)

    result = e2e._execute_query(
        Client(),  # type: ignore[arg-type]
        "https://runtime",
        USER,
        CHAT_SESSION,
        SelectedQuery(0, "Question?"),
        Trace(),  # type: ignore[arg-type]
        {},
        tmp_path,
        {},
        started_at="2026-09-11T00:00:00Z",
        started_clock=0.0,
    )

    assert result.status == "succeeded", (
        result.failure_code,
        result.failure_scope,
        result.failure_stage,
    )
    assert result.duration_ms == 3000.0
    assert result.evaluation == {
        "status": "succeeded",
        "error_code": None,
        "evaluation_ms": 123.0,
        "queue_wait_ms": 17.0,
        "execution_ms": 0.0,
        "record_path": None,
        "result_path": None,
        "record_sha256": "a" * 64,
    }
    assert clock[0] == 103.0
