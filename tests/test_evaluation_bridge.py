from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from backend.wizard.diagnostics import (
    DiagnosticTraceRegistry,
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


def test_production_modules_do_not_import_ragas_or_evaluation_package() -> None:
    root = Path(__file__).resolve().parents[1]
    for directory in (root / "backend", root / "deployment"):
        for path in directory.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "import ragas" not in source
            assert "from ragas" not in source
            assert "import rag_evaluation" not in source
            assert "from rag_evaluation" not in source


def test_e2e_launches_local_evaluation_before_remote_task_polling() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "deployment/e2e_diagnostic_api.py").read_text(encoding="utf-8")
    function = source[source.index("def _execute_query(") : source.index("def _session_failure_result(")]
    assert function.index("evaluation_launch = start_local_evaluation(") < function.index(
        "persistence_task = poll_task("
    )
    assert "evaluation = finish_local_evaluation(" in function
    assert ").artifact()" in function
