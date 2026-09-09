from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
import pytest

from backend.api.wizard_diagnostics import build_wizard_diagnostic_router
from backend.mappings.document_map import DocumentMap
from backend.mappings.paragraph_map import ParagraphMap
from backend.wizard.diagnostics import (
    DiagnosticTraceRegistry,
    TRACE_SAMPLE_LIMIT,
    TRACE_SESSION_MAX_OPERATIONS,
    activate_operation,
    add_count,
    install_registry,
    mapping_checkpoint,
    mapping_digest,
    mapping_membership_digest,
    observe_stage,
    set_sample,
    uninstall_registry,
)
from deployment.wizard_diagnostic import OperationRecorder, WizardDiagnosticError
from deployment.wizard_diagnostic_api import (
    WizardApi,
    validate_trace_evidence,
    validate_upload_zero_side_effects,
    verify_partial_resave_mappings,
)


def _operation_payload(*, kind: str = "upload", outcome: str = "succeeded"):
    stage = {
        "call_count": 1,
        "total_ms": 1.0,
        "min_ms": 1.0,
        "max_ms": 1.0,
        "failure_count": 0,
    }
    return {
        "operation_id": str(uuid4()),
        "kind": kind,
        "outcome": outcome,
        "task_id": None,
        "finished_at": "2026-01-01T00:00:00Z",
        "stages": {
            "api.lookup_validation": stage,
            "upload.multipart_copy": stage,
            "upload.file_decode_read": stage,
            "upload.draft_merge": stage,
            "upload.server_total": stage,
        },
        "counts": {},
        "flags": {
            "task_enqueued": False,
            "task_id_present": False,
            "save_delete_executed": False,
            "lateon_executed": False,
            "gte_executed": False,
            "weaviate_mutation_executed": False,
        },
        "samples": {},
    }


def test_hidden_router_has_exactly_three_non_openapi_routes() -> None:
    registry = DiagnosticTraceRegistry("wizard_diagnostic")
    router = build_wizard_diagnostic_router(registry)
    paths_and_methods = {
        (route.path, tuple(sorted(route.methods or ()))) for route in router.routes
    }

    assert paths_and_methods == {
        ("/api/_diagnostics/wizard/trace", ("POST",)),
        ("/api/_diagnostics/wizard/trace", ("GET",)),
        ("/api/_diagnostics/wizard/trace", ("DELETE",)),
    }
    app = FastAPI()
    app.include_router(router)
    assert "/api/_diagnostics/wizard/trace" not in app.openapi()["paths"]


def test_providerless_app_stays_generic_and_runtime_owns_conditional_mount() -> None:
    root = Path(__file__).resolve().parents[1]
    main_source = (root / "backend/main.py").read_text(encoding="utf-8")
    runtime_source = (root / "backend/runtime_app.py").read_text(encoding="utf-8")

    assert "wizard_diagnostic" not in main_source
    assert "if WIZARD_DIAGNOSTICS_ENABLED:" in runtime_source
    assert "build_wizard_diagnostic_router" in runtime_source
    assert runtime_source.index("application = create_app(") < runtime_source.index(
        "if WIZARD_DIAGNOSTICS_ENABLED:"
    )


def test_registry_is_single_session_bounded_and_get_is_non_destructive() -> None:
    registry = DiagnosticTraceRegistry("wizard_diagnostic")
    session_id = str(uuid4())
    registry.start("wizard_diagnostic", session_id, "run-1")
    with pytest.raises(RuntimeError, match="already active"):
        registry.start("wizard_diagnostic", str(uuid4()), "run-2")
    with pytest.raises(KeyError):
        registry.snapshot("another_user", session_id)

    first_operation = str(uuid4())
    first = registry.begin_operation(
        user_id="wizard_diagnostic",
        session_id=session_id,
        operation_id=first_operation,
        kind="save",
        collection_type="knowledge_facts",
        wizard_id=str(uuid4()),
    )
    assert first is not None
    registry.observe_stage(first, "save.chunking", 2.0, False)
    registry.observe_stage(first, "save.chunking", 3.0, False)
    registry.set_sample(first, "ids", range(40), exact_count=40)
    registry.finish(first, "succeeded")

    first_read = registry.snapshot("wizard_diagnostic", session_id, first_operation)
    second_read = registry.snapshot("wizard_diagnostic", session_id, first_operation)
    assert first_read == second_read
    operation = first_read["operations"][0]
    assert operation["stages"]["save.chunking"] == {
        "call_count": 2,
        "total_ms": 5.0,
        "min_ms": 2.0,
        "max_ms": 3.0,
        "failure_count": 0,
    }
    assert operation["samples"]["ids"]["exact_count"] == 40
    assert len(operation["samples"]["ids"]["items"]) == TRACE_SAMPLE_LIMIT
    assert operation["samples"]["ids"]["truncated"] is True

    for _ in range(TRACE_SESSION_MAX_OPERATIONS - 1):
        assert registry.begin_operation(
            user_id="wizard_diagnostic",
            session_id=session_id,
            operation_id=str(uuid4()),
            kind="upload",
            collection_type="knowledge_facts",
            wizard_id=str(uuid4()),
        ) is not None
    assert registry.begin_operation(
        user_id="wizard_diagnostic",
        session_id=session_id,
        operation_id=str(uuid4()),
        kind="upload",
        collection_type="knowledge_facts",
        wizard_id=str(uuid4()),
    ) is None
    assert registry.snapshot("wizard_diagnostic", session_id)["overflowed"] is True

    registry.delete("wizard_diagnostic", session_id)
    with pytest.raises(KeyError):
        registry.snapshot("wizard_diagnostic", session_id)


def test_observation_faults_are_contained_and_mark_the_active_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = DiagnosticTraceRegistry("wizard_diagnostic")
    session_id = str(uuid4())
    registry.start("wizard_diagnostic", session_id, "run-1")
    handle = registry.begin_operation(
        user_id="wizard_diagnostic",
        session_id=session_id,
        operation_id=str(uuid4()),
        kind="save",
        collection_type="knowledge_facts",
        wizard_id=str(uuid4()),
    )
    assert handle is not None
    install_registry(registry)
    try:
        monkeypatch.setattr(
            registry,
            "add_count",
            lambda *_: (_ for _ in ()).throw(RuntimeError("telemetry")),
        )
        with activate_operation(handle):
            add_count("count", 1)
            with observe_stage("save.diff"):
                pass
    finally:
        uninstall_registry(registry)
    assert registry.snapshot("wizard_diagnostic", session_id)["trace_faulted"] is True


def test_mapping_checkpoint_is_bounded_and_has_ordered_and_membership_digests() -> None:
    user_id = "wizard_diagnostic"
    collection = "knowledge_facts"
    wizard_id = str(uuid4())
    chunks = [str(uuid4()), str(uuid4())]
    documents = DocumentMap(user_id, collection)
    paragraphs = ParagraphMap(user_id, collection)
    documents.create_document(wizard_id)
    documents.update_paragraphs(wizard_id, {1: "one\n\n", 2: "two"})
    paragraphs.replace_document(wizard_id, {1: chunks, 2: []})
    runtime = SimpleNamespace(
        document_map=lambda *_: documents,
        paragraph_map=lambda *_: paragraphs,
    )

    checkpoint = mapping_checkpoint(
        runtime, user_id, collection, wizard_id, [chunks[0]]
    )

    assert checkpoint["document"]["full_text_sha256"]
    assert checkpoint["paragraph_map"]["mapping_sha256"] == mapping_digest(
        {1: chunks, 2: []}
    )
    assert checkpoint["paragraph_map"]["membership_sha256"] == (
        mapping_membership_digest({1: list(reversed(chunks)), 2: []})
    )
    assert checkpoint["probe"]["owned_count"] == 1
    assert checkpoint["probe"]["owners"][chunks[0]]["paragraph_id"] == 1


def test_trace_validator_rejects_missing_and_proof_truncated_evidence() -> None:
    operation = _operation_payload()
    validate_upload_zero_side_effects(operation)

    faulty = {**operation, "flags": {**operation["flags"], "task_enqueued": True}}
    with pytest.raises(WizardDiagnosticError, match="forbidden side effect"):
        validate_upload_zero_side_effects(faulty)

    operation["samples"] = {
        "ids": {"exact_count": 2, "items": ["one"], "truncated": True}
    }
    with pytest.raises(WizardDiagnosticError, match="proof-critically truncated"):
        validate_trace_evidence(
            operation,
            kind="upload",
            outcome="succeeded",
            proof_samples=("ids",),
        )
    operation["stages"]["upload.server_total"]["total_ms"] = math.inf
    with pytest.raises(WizardDiagnosticError, match="malformed"):
        validate_trace_evidence(
            operation,
            kind="upload",
            outcome="succeeded",
            required_stages=("upload.server_total",),
        )


def test_partial_resave_proves_retained_and_replaced_chunk_ids() -> None:
    retained = str(uuid4())
    old_changed = str(uuid4())
    new_changed = str(uuid4())
    result = verify_partial_resave_mappings(
        {1: [retained], 2: [old_changed]},
        {1: [retained], 2: [new_changed]},
        2,
        probed_old_owner_count=0,
        physical_chunk_ids=(retained, new_changed),
    )
    assert result == {
        "unchanged_paragraph_count": 1,
        "old_changed_chunk_count": 1,
        "new_changed_chunk_count": 1,
    }

    with pytest.raises(WizardDiagnosticError, match="unchanged paragraph"):
        verify_partial_resave_mappings(
            {1: [retained], 2: [old_changed]},
            {1: [str(uuid4())], 2: [new_changed]},
            2,
            probed_old_owner_count=0,
            physical_chunk_ids=(),
        )
    with pytest.raises(WizardDiagnosticError, match="remains in Weaviate"):
        verify_partial_resave_mappings(
            {1: [retained], 2: [old_changed]},
            {1: [retained], 2: [new_changed]},
            2,
            probed_old_owner_count=0,
            physical_chunk_ids=(old_changed,),
        )


def test_anonymous_trace_client_is_credential_free(tmp_path: Path) -> None:
    recorder = OperationRecorder(tmp_path / "operations.jsonl")
    recorder.path.touch()
    api = WizardApi(
        "https://runtime.example",
        {"Modal-Key": "private-id", "Modal-Secret": "private-secret"},
        recorder,
        "wizard_diagnostic",
        "run-1",
    )
    try:
        assert "modal-key" in api.client.headers
        assert "modal-secret" in api.client.headers
        assert "modal-key" not in api.anonymous_client.headers
        assert "modal-secret" not in api.anonymous_client.headers
    finally:
        api.close()
