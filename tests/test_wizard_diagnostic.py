from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from uuid import uuid4

import httpx
import pytest

from deployment import ragctl, wizard_diagnostic_api
from deployment.wizard_diagnostic import (
    CorpusDocument,
    OperationRecorder,
    WizardDiagnosticError,
    begin_replacement,
    corpus_state_lock,
    create_diagnostic_run,
    discard_pending_replacement,
    load_corpus_state,
    mark_replacement_verified,
    new_corpus_state,
    preflight_wizard_fixtures,
    promote_replacement,
    record_active_ingestion_policy,
    record_replacement_document,
    remove_pending_cleanup,
    require_empty_first_seed,
    update_run_summary,
    validate_corpus_preflight,
    validate_diagnostic_user,
    validate_physical_membership,
    write_corpus_state,
)
from deployment.wizard_diagnostic_api import (
    diagnostic_edit_markers,
    marker_chunk_count,
    save_request_payload,
    storage_integrity,
    upload_continuation,
    upload_form_fields,
    wizard_saved_fingerprint,
)
from backend.weaviate_client.models import ChunkRecord


def _fixture_tree(root: Path) -> Path:
    fixtures = root / "fixtures"
    knowledge = fixtures / "knowledge"
    policy = fixtures / "policy"
    knowledge.mkdir(parents=True)
    policy.mkdir()
    (knowledge / "knowledge.txt").write_text("Knowledge fact.", encoding="utf-8")
    (policy / "policy.md").write_text("Policy rule.", encoding="utf-8")
    return fixtures


def _add_saved_document(state, collection: str):
    created = CorpusDocument(collection, str(uuid4()))
    state = record_replacement_document(state, created)
    submitting = replace(created, status="save_submitting")
    state = record_replacement_document(state, submitting)
    submitted = replace(
        submitting,
        status="save_submitted",
        task_id=str(uuid4()),
    )
    state = record_replacement_document(state, submitted)
    chunk_id = str(uuid4())
    saved = replace(
        submitted,
        status="saved",
        chunk_ids=(chunk_id,),
        storage_fingerprint=hashlib.sha256(chunk_id.encode()).hexdigest(),
    )
    return record_replacement_document(state, saved), saved


def _complete_state(fixtures_path: Path, project_root: Path, run_id: str = "run-1"):
    fixtures = preflight_wizard_fixtures(fixtures_path, project_root)
    state = begin_replacement(
        new_corpus_state("wizard_diagnostic"), run_id, fixtures
    )
    for collection in ("knowledge", "policy"):
        state, _ = _add_saved_document(state, collection)
    state = mark_replacement_verified(state)
    return promote_replacement(state), fixtures


class _IsolationPhysical:
    def __init__(self, items: list[object] | None = None) -> None:
        self.items = items or []
        self.iterator_calls: list[dict[str, object]] = []

    def iterator(self, **kwargs: object):
        self.iterator_calls.append(kwargs)
        return iter(self.items)


class _IsolationCollections:
    def __init__(self, present: set[str], physical: dict[str, _IsolationPhysical]):
        self.present = present
        self.physical = physical

    def exists(self, name: str) -> bool:
        return name in self.present

    def use(self, name: str) -> _IsolationPhysical:
        return self.physical[name]


class _IsolationManager:
    def __init__(self, user_id: str, present_count: int) -> None:
        self.user_id = user_id
        self.names = tuple(
            wizard_diagnostic_api.get_collection_name(user_id, collection_type)
            for collection_type in ("conversations", "knowledge_facts", "policy")
        )
        self.physical = {name: _IsolationPhysical() for name in self.names}
        self.collections = _IsolationCollections(
            set(self.names[:present_count]), self.physical
        )
        self.client = type("Client", (), {"collections": self.collections})()
        self.ensure_calls: list[str] = []

    def ensure_user_collections(self, user_id: str) -> None:
        self.ensure_calls.append(user_id)
        self.collections.present.update(self.names)


def _isolation_storage(manager: _IsolationManager):
    storage = wizard_diagnostic_api.CorpusStorage({}, "primary")
    storage.manager = manager
    return storage


@pytest.mark.parametrize("present_count", [0, 3])
def test_isolation_bootstrap_creates_or_reuses_three_empty_collections(
    present_count: int,
) -> None:
    user_id = "primary_other"
    manager = _IsolationManager(user_id, present_count)

    _isolation_storage(manager).ensure_empty_user_collections(user_id)

    assert manager.collections.present == set(manager.names)
    assert manager.ensure_calls == [user_id]
    assert all(
        physical.iterator_calls
        == [{"include_vector": False, "return_properties": []}]
        for physical in manager.physical.values()
    )


@pytest.mark.parametrize("present_count", [1, 2])
def test_isolation_bootstrap_rejects_partial_collection_state(
    present_count: int,
) -> None:
    user_id = "primary_other"
    manager = _IsolationManager(user_id, present_count)

    with pytest.raises(WizardDiagnosticError, match="zero or all three"):
        _isolation_storage(manager).ensure_empty_user_collections(user_id)

    assert manager.ensure_calls == []


def test_isolation_bootstrap_rejects_unexpected_existing_data() -> None:
    user_id = "primary_other"
    manager = _IsolationManager(user_id, 3)
    manager.physical[manager.names[1]].items.append(object())

    with pytest.raises(WizardDiagnosticError, match="unexpected data"):
        _isolation_storage(manager).ensure_empty_user_collections(user_id)


def test_parser_accepts_reseed_and_preserves_existing_commands() -> None:
    cli = ragctl.parser()

    diagnostic = cli.parse_args(
        ["diagnose", "wizard", "--fixtures", "diagnostics/fixtures/wizard"]
    )
    assert diagnostic.command == "diagnose"
    assert diagnostic.diagnostic == "wizard"
    assert diagnostic.fixtures == Path("diagnostics/fixtures/wizard")
    assert diagnostic.reseed_corpus is False

    reseed = cli.parse_args(
        [
            "diagnose",
            "wizard",
            "--fixtures",
            "diagnostics/fixtures/wizard",
            "--reseed-corpus",
        ]
    )
    assert reseed.reseed_corpus is True

    assert cli.parse_args(["up"]).command == "up"
    assert cli.parse_args(["ask", "question"]).command == "ask"
    assert cli.parse_args(["status"]).command == "status"
    assert cli.parse_args(["down"]).command == "down"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "Missing required"),
        ({"RAG_DIAGNOSTIC_USER_ID": ""}, "must not be empty"),
        ({"RAG_DIAGNOSTIC_USER_ID": "   "}, "whitespace"),
        ({"RAG_DIAGNOSTIC_USER_ID": " wizard_diagnostic"}, "whitespace"),
        ({"RAG_DIAGNOSTIC_USER_ID": "wizard/diagnostic"}, "is invalid"),
    ],
)
def test_diagnostic_user_and_derived_user_use_backend_contract(
    config: dict[str, str], message: str
) -> None:
    with pytest.raises(WizardDiagnosticError, match=message):
        validate_diagnostic_user(config)

    assert validate_diagnostic_user(
        {"RAG_DIAGNOSTIC_USER_ID": "wizard_diagnostic"}
    ) == ("wizard_diagnostic", "wizard_diagnostic_other")


def test_fixture_preflight_requires_root_and_both_collections(tmp_path: Path) -> None:
    with pytest.raises(WizardDiagnosticError, match="root does not exist"):
        preflight_wizard_fixtures(Path("missing"), tmp_path)

    fixtures = tmp_path / "fixtures"
    (fixtures / "knowledge").mkdir(parents=True)
    (fixtures / "knowledge" / "fact.txt").write_text("fact", encoding="utf-8")
    with pytest.raises(WizardDiagnosticError, match="Missing fixture collection"):
        preflight_wizard_fixtures(fixtures, tmp_path)


def test_fixture_preflight_rejects_empty_collection_and_ignores_markers(
    tmp_path: Path,
) -> None:
    fixtures = _fixture_tree(tmp_path)
    (fixtures / "knowledge" / "knowledge.txt").unlink()
    (fixtures / "knowledge" / ".gitkeep").touch()

    with pytest.raises(WizardDiagnosticError, match="contains no fixture files"):
        preflight_wizard_fixtures(fixtures, tmp_path)


def test_fixture_manifest_is_deterministic_hashed_and_project_relative(
    tmp_path: Path,
) -> None:
    fixtures = _fixture_tree(tmp_path)
    knowledge = fixtures / "knowledge"
    (knowledge / "knowledge.txt").unlink()
    (knowledge / "z-last.txt").write_text("last", encoding="utf-8")
    (knowledge / "a-first.md").write_text("first", encoding="utf-8")
    (knowledge / ".gitkeep").touch()

    result = preflight_wizard_fixtures(Path("fixtures"), tmp_path)

    assert result.root == fixtures.resolve()
    assert [item.path.name for item in result.knowledge] == [
        "a-first.md",
        "z-last.txt",
    ]
    assert result.inventory()["knowledge"] == [
        {
            "path": "knowledge/a-first.md",
            "size_bytes": 5,
            "sha256": hashlib.sha256(b"first").hexdigest(),
        },
        {
            "path": "knowledge/z-last.txt",
            "size_bytes": 4,
            "sha256": hashlib.sha256(b"last").hexdigest(),
        },
    ]
    assert len(result.digest) == 64
    assert result.digest == preflight_wizard_fixtures(fixtures, tmp_path).digest

    (knowledge / "z-last.txt").write_text("changed", encoding="utf-8")
    assert preflight_wizard_fixtures(fixtures, tmp_path).digest != result.digest


def test_fixture_preflight_rejects_unsupported_nested_and_symlink_entries(
    tmp_path: Path,
) -> None:
    unsupported_root = tmp_path / "unsupported"
    fixtures = _fixture_tree(unsupported_root)
    (fixtures / "knowledge" / "bad.bin").write_bytes(b"content")
    with pytest.raises(WizardDiagnosticError, match="Unsupported file extension"):
        preflight_wizard_fixtures(fixtures, unsupported_root)

    nested_root = tmp_path / "nested"
    fixtures = _fixture_tree(nested_root)
    (fixtures / "knowledge" / "nested").mkdir()
    with pytest.raises(WizardDiagnosticError, match="only top-level files"):
        preflight_wizard_fixtures(fixtures, nested_root)

    symlink_root = tmp_path / "symlink"
    fixtures = _fixture_tree(symlink_root)
    (fixtures / "knowledge" / "linked.txt").symlink_to(
        fixtures / "knowledge" / "knowledge.txt"
    )
    with pytest.raises(WizardDiagnosticError, match="must not be a symlink"):
        preflight_wizard_fixtures(fixtures, symlink_root)


def test_fixture_preflight_rejects_invalid_encoding_and_empty_text(
    tmp_path: Path,
) -> None:
    invalid_root = tmp_path / "invalid"
    fixtures = _fixture_tree(invalid_root)
    (fixtures / "knowledge" / "knowledge.txt").write_bytes(b"\xff")
    with pytest.raises(WizardDiagnosticError, match="Invalid fixture"):
        preflight_wizard_fixtures(fixtures, invalid_root)

    empty_root = tmp_path / "empty"
    fixtures = _fixture_tree(empty_root)
    (fixtures / "knowledge" / "knowledge.txt").write_text("", encoding="utf-8")
    with pytest.raises(WizardDiagnosticError, match="non-empty text"):
        preflight_wizard_fixtures(fixtures, empty_root)


def test_fixture_preflight_rejects_unreadable_file(tmp_path: Path) -> None:
    fixtures = _fixture_tree(tmp_path)
    fixture = fixtures / "knowledge" / "knowledge.txt"
    fixture.chmod(0)
    try:
        with pytest.raises(WizardDiagnosticError, match="Could not inspect|Invalid fixture"):
            preflight_wizard_fixtures(fixtures, tmp_path)
    finally:
        fixture.chmod(0o600)


def test_fixture_preflight_enforces_file_limit_and_batches_collection_total(
    tmp_path: Path,
) -> None:
    per_file_root = tmp_path / "per-file"
    fixtures = _fixture_tree(per_file_root)
    (fixtures / "knowledge" / "knowledge.txt").write_text("12345", encoding="utf-8")
    with pytest.raises(WizardDiagnosticError, match="UPLOAD_MAX_FILE_BYTES"):
        preflight_wizard_fixtures(
            fixtures,
            per_file_root,
            {
                "UPLOAD_MAX_FILE_BYTES": "4",
                "UPLOAD_MAX_TOTAL_BYTES": "10",
                "UPLOAD_READ_CHUNK_BYTES": "4",
            },
        )

    total_root = tmp_path / "total"
    fixtures = _fixture_tree(total_root)
    knowledge = fixtures / "knowledge"
    (knowledge / "knowledge.txt").unlink()
    (knowledge / "a.txt").write_text("12", encoding="utf-8")
    (knowledge / "b.txt").write_text("345", encoding="utf-8")
    (knowledge / "c.txt").write_text("67", encoding="utf-8")
    (fixtures / "policy" / "policy.md").write_text("1234", encoding="utf-8")
    result = preflight_wizard_fixtures(
        fixtures,
        total_root,
        {
            "UPLOAD_MAX_FILE_BYTES": "4",
            "UPLOAD_MAX_TOTAL_BYTES": "5",
            "UPLOAD_READ_CHUNK_BYTES": "4",
        },
    )

    batches = result.batches("knowledge")
    assert [[item.path.name for item in batch.files] for batch in batches] == [
        ["a.txt", "b.txt"],
        ["c.txt"],
    ]
    assert [batch.size_bytes for batch in batches] == [5, 2]
    assert all(batch.size_bytes <= 5 for batch in batches)
    assert result.combined_text("knowledge") == "12\n\n345\n\n67"
    reconstructed = result.rules.text_join_separator.join(
        result.rules.text_join_separator.join(
            item.path.read_text(encoding=result.rules.text_encoding)
            for item in batch.files
        )
        for batch in batches
    )
    assert reconstructed == result.combined_text("knowledge")


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"SUPPORTED_FILE_EXTENSIONS": ", ,"}, "at least one"),
        ({"TEXT_FILE_ENCODING": "not-a-codec"}, "supported codec"),
        ({"UPLOAD_MAX_FILE_BYTES": "zero"}, "integer"),
        ({"UPLOAD_MAX_FILE_BYTES": "0"}, "greater than zero"),
        (
            {
                "UPLOAD_MAX_FILE_BYTES": "11",
                "UPLOAD_MAX_TOTAL_BYTES": "10",
            },
            "must not exceed",
        ),
        (
            {
                "UPLOAD_MAX_FILE_BYTES": "10",
                "UPLOAD_MAX_TOTAL_BYTES": "10",
                "UPLOAD_READ_CHUNK_BYTES": "11",
            },
            "must not exceed",
        ),
    ],
)
def test_fixture_preflight_rejects_invalid_ingestion_configuration(
    tmp_path: Path, config: dict[str, str], message: str
) -> None:
    fixtures = _fixture_tree(tmp_path)

    with pytest.raises(WizardDiagnosticError, match=message):
        preflight_wizard_fixtures(fixtures, tmp_path, config)


def test_fixture_preflight_uses_custom_extension_encoding_and_separator(
    tmp_path: Path,
) -> None:
    fixtures = _fixture_tree(tmp_path)
    knowledge = fixtures / "knowledge" / "knowledge.txt"
    policy = fixtures / "policy" / "policy.md"
    knowledge.rename(knowledge.with_suffix(".note"))
    policy.rename(policy.with_suffix(".note"))
    (fixtures / "knowledge" / "knowledge.note").write_text(
        "Knowledge fact.", encoding="utf-16"
    )
    (fixtures / "policy" / "policy.note").write_text(
        "Policy rule.", encoding="utf-16"
    )
    config = {
        "SUPPORTED_FILE_EXTENSIONS": "NOTE",
        "TEXT_FILE_ENCODING": "UTF_16",
        "TEXT_FILE_JOIN_SEPARATOR": "\r\n--\r\n",
    }

    result = preflight_wizard_fixtures(fixtures, tmp_path, config)

    assert result.rules.supported_extensions == (".note",)
    assert result.rules.text_encoding == "utf-16"
    assert result.rules.text_join_separator == "\r\n--\r\n"
    assert result.combined_text("knowledge") == "Knowledge fact."


def test_corpus_digest_excludes_operational_policy(tmp_path: Path) -> None:
    fixtures = _fixture_tree(tmp_path)
    baseline = preflight_wizard_fixtures(
        fixtures, tmp_path, {"TEXT_FILE_ENCODING": "utf8"}
    )
    policy_change = preflight_wizard_fixtures(
        fixtures,
        tmp_path,
        {
            "TEXT_FILE_ENCODING": "utf-8",
            "SUPPORTED_FILE_EXTENSIONS": ".txt,.md,.extra",
            "UPLOAD_MAX_FILE_BYTES": "1000",
            "UPLOAD_MAX_TOTAL_BYTES": "2000",
            "UPLOAD_READ_CHUNK_BYTES": "500",
        },
    )
    separator_change = preflight_wizard_fixtures(
        fixtures, tmp_path, {"TEXT_FILE_JOIN_SEPARATOR": "\n"}
    )

    assert baseline.rules.text_encoding == "utf-8"
    assert baseline.digest == policy_change.digest
    assert baseline.digest != separator_change.digest

    state = begin_replacement(
        new_corpus_state("wizard_diagnostic"), "policy", baseline
    )
    for collection in ("knowledge", "policy"):
        state, _ = _add_saved_document(state, collection)
    state = promote_replacement(mark_replacement_verified(state))
    updated = record_active_ingestion_policy(state, policy_change)
    assert updated.active is not None
    assert updated.active.fixture_digest == state.active.fixture_digest  # type: ignore[union-attr]
    assert updated.active.ingestion_policy == policy_change.rules.ingestion_policy()


def _chunk_record(**changes: object) -> ChunkRecord:
    chunk_id = str(uuid4())
    values: dict[str, object] = {
        "object_id": chunk_id,
        "user_id": "wizard_diagnostic",
        "document_id": str(uuid4()),
        "paragraph_id": 1,
        "chunk_id": chunk_id,
        "raw_text": "sanitized fingerprint text",
        "late_interaction": ((0.25, -0.5), (1.0, 2.0)),
        "mmr_diversity": (0.75, -1.25, 3.0),
    }
    values.update(changes)
    return ChunkRecord(**values)  # type: ignore[arg-type]


def test_storage_fingerprint_is_deterministic_and_sanitized() -> None:
    first = _chunk_record()
    second = _chunk_record(document_id=first.document_id)

    forward = storage_integrity((first, second))
    reverse = storage_integrity((second, first))

    assert forward == reverse
    assert forward.chunk_ids == tuple(sorted((first.chunk_id, second.chunk_id)))
    assert forward.chunk_count == 2
    assert forward.lateon_dimension == 2
    assert forward.gte_dimension == 3
    assert "sanitized fingerprint text" not in forward.fingerprint


@pytest.mark.parametrize(
    "mutation",
    [
        {"user_id": "wizard_diagnostic_other"},
        {"document_id": str(uuid4())},
        {"paragraph_id": 2},
        {"raw_text": "changed"},
        {"late_interaction": ((0.25, -0.4), (1.0, 2.0))},
        {"mmr_diversity": (0.75, -1.2, 3.0)},
    ],
)
def test_storage_fingerprint_detects_persisted_mutations(
    mutation: dict[str, object],
) -> None:
    record = _chunk_record()
    changed = replace(record, **mutation)

    assert storage_integrity((record,)).fingerprint != storage_integrity(
        (changed,)
    ).fingerprint


def test_storage_fingerprint_detects_chunk_identity_changes() -> None:
    record = _chunk_record()
    changed_id = str(uuid4())
    changed = replace(record, object_id=changed_id, chunk_id=changed_id)

    assert storage_integrity((record,)).fingerprint != storage_integrity(
        (changed,)
    ).fingerprint

    with pytest.raises(WizardDiagnosticError, match="object and chunk IDs"):
        storage_integrity((replace(record, object_id=str(uuid4())),))


def test_request_builders_preserve_negative_case_wire_values() -> None:
    assert upload_form_fields(
        "wizard_diagnostic",
        current_text="draft",
        modified_paragraph_ids=[1, "bad", 3],
    ) == {
        "user_id": "wizard_diagnostic",
        "current_text": "draft",
        "modified_paragraph_ids": ["1", "bad", "3"],
    }
    assert save_request_payload("wizard_diagnostic", "text", ["bad"]) == {
        "user_id": "wizard_diagnostic",
        "current_text": "text",
        "modified_paragraph_ids": ["bad"],
    }
    assert wizard_saved_fingerprint(
        {"full_text": "saved", "paragraph_ids": [1, 2]}
    ) != wizard_saved_fingerprint(
        {"full_text": "saved", "paragraph_ids": [2, 1]}
    )

    request = httpx.Request(
        "POST",
        "https://diagnostic.invalid/upload",
        data=upload_form_fields(
            "wizard_diagnostic", modified_paragraph_ids=[1, 2]
        ),
        files=[("files", ("fixture.txt", b"text", "text/plain"))],
    )
    encoded = request.read()
    assert request.headers["content-type"].startswith("multipart/form-data;")
    assert encoded.count(b'name="modified_paragraph_ids"') == 2


def test_upload_continuation_carries_real_draft_fields() -> None:
    assert upload_continuation(None) == (None, None)
    first = {
        "full_text": "first batch",
        "modified_paragraph_ids": [1],
    }
    second = {
        "full_text": "first batch\n\nsecond batch",
        "modified_paragraph_ids": [1, 2],
    }

    assert upload_continuation(first) == ("first batch", [1])
    assert upload_continuation(second) == (
        "first batch\n\nsecond batch",
        [1, 2],
    )

    with pytest.raises(WizardDiagnosticError, match="modified paragraph IDs"):
        upload_continuation(
            {"full_text": "draft", "modified_paragraph_ids": ["1"]}
        )


def test_persistent_batch_orchestration_submits_one_final_save() -> None:
    tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(wizard_diagnostic_api.ensure_persistent_corpus)
        )
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]

    assert sum(
        isinstance(call.func, ast.Attribute) and call.func.attr == "submit_save"
        for call in calls
    ) == 1
    assert sum(
        isinstance(call.func, ast.Name)
        and call.func.id == "_upload_fixture_batches"
        for call in calls
    ) == 1


def test_destructive_edit_markers_and_storage_checks_are_sanitized() -> None:
    old_marker, new_marker = diagnostic_edit_markers("run-123")
    assert old_marker != new_marker
    assert "run-123" not in old_marker
    assert diagnostic_edit_markers("run-123") == (old_marker, new_marker)
    record = _chunk_record(raw_text=f"before {old_marker} after")

    assert marker_chunk_count((record,), old_marker) == 1
    assert marker_chunk_count((record,), new_marker) == 0


def test_operation_log_has_stable_sequences_and_totals(tmp_path: Path) -> None:
    path = tmp_path / "operations.jsonl"
    path.touch()
    recorder = OperationRecorder(path)

    recorder.record(
        "wizard.create",
        "passed",
        method="POST",
        path="/api/knowledge/wizards",
        expected_status=201,
        actual_status=201,
        duration_ms=1.25,
        selected_ids={"wizard_id": str(uuid4())},
    )
    recorder.record("wizard.check", "failed")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["sequence"] for row in rows] == [1, 2]
    assert rows[0]["timestamp"].endswith("Z")
    assert rows[0]["method"] == "POST"
    assert rows[1]["method"] is None
    assert rows[1]["poll_count"] is None
    assert recorder.totals == {"total": 2, "passed": 1, "failed": 1}
    with pytest.raises(WizardDiagnosticError, match="outcome"):
        recorder.record("invalid", "unknown")


def test_diagnostic_run_creates_unique_phase_1c_artifacts(tmp_path: Path) -> None:
    fixtures_path = _fixture_tree(tmp_path)
    fixtures = preflight_wizard_fixtures(fixtures_path, tmp_path)
    output_root = tmp_path / "output"

    first = create_diagnostic_run(output_root, "wizard_diagnostic", fixtures)
    second = create_diagnostic_run(output_root, "wizard_diagnostic", fixtures)

    assert first.run_id != second.run_id
    assert first.operations_path.read_bytes() == b""
    summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "1.3"
    assert summary["phase"] == "1C"
    assert summary["run_id"] == first.run_id
    assert summary["diagnostic_user_id"] == "wizard_diagnostic"
    assert summary["fixture_digest"] == fixtures.digest
    assert summary["operation_totals"] == {
        "total": 0,
        "passed": 0,
        "failed": 0,
    }
    assert summary["scratch_status"] == "pending"
    assert summary["corpus_action"] == "pending"
    assert summary["telemetry_status"] == "pending"
    assert summary["security_status"] == "pending"
    assert summary["trace_deleted"] is False
    assert summary["compensation_status"] == (
        "Compensation path not acceptance-tested because no safe deterministic "
        "natural failure seam exists."
    )
    assert summary["started_at"].endswith("Z")

    update_run_summary(
        first,
        status="succeeded",
        pre_down_status="succeeded",
        up_status="succeeded",
        down_status="succeeded",
        finished=True,
        operations_count=3,
        operation_totals={"total": 3, "passed": 3, "failed": 0},
        scratch_status="passed",
        corpus_action="created",
        telemetry_status="passed",
        security_status="passed",
        trace_deleted=True,
    )
    completed = json.loads(first.summary_path.read_text(encoding="utf-8"))
    assert completed["status"] == "succeeded"
    assert completed["lifecycle"] == {
        "pre_down": "succeeded",
        "up": "succeeded",
        "down": "succeeded",
    }
    assert completed["operations_count"] == 3
    assert completed["operation_totals"]["passed"] == 3
    assert completed["telemetry_status"] == "passed"
    assert completed["security_status"] == "passed"
    assert completed["trace_deleted"] is True
    assert completed["finished_at"].endswith("Z")
    assert not first.summary_path.with_suffix(".json.tmp").exists()


def test_corpus_state_round_trip_and_first_generation_promotion(tmp_path: Path) -> None:
    fixtures_path = _fixture_tree(tmp_path)
    fixtures = preflight_wizard_fixtures(fixtures_path, tmp_path)
    state = begin_replacement(
        new_corpus_state("wizard_diagnostic"), "first-generation", fixtures
    )

    knowledge = CorpusDocument("knowledge", str(uuid4()))
    state = record_replacement_document(state, knowledge)
    state = record_replacement_document(
        state,
        CorpusDocument(
            "knowledge",
            knowledge.wizard_id,
            status="save_submitting",
        ),
    )
    state = record_replacement_document(
        state,
        CorpusDocument(
            "knowledge",
            knowledge.wizard_id,
            status="save_submitted",
            task_id=str(uuid4()),
        ),
    )
    submitted = state.pending_replacement.document("knowledge")
    assert submitted is not None
    chunk_id = str(uuid4())
    state = record_replacement_document(
        state,
        replace(
            submitted,
            status="saved",
            chunk_ids=(chunk_id,),
            storage_fingerprint=hashlib.sha256(chunk_id.encode()).hexdigest(),
        ),
    )
    state, _ = _add_saved_document(state, "policy")
    state = mark_replacement_verified(state)
    state = promote_replacement(state)

    assert state.active is not None
    assert state.active.completed_at is not None
    assert state.pending_replacement is None
    assert state.pending_cleanup == ()

    path = tmp_path / "corpus-state.json"
    write_corpus_state(path, state)
    assert load_corpus_state(path, "wizard_diagnostic") == state
    assert not path.with_suffix(".json.tmp").exists()


def test_save_submitting_state_is_atomic_recoverable_and_strict(
    tmp_path: Path,
) -> None:
    fixtures = preflight_wizard_fixtures(_fixture_tree(tmp_path), tmp_path)
    state = begin_replacement(
        new_corpus_state("wizard_diagnostic"), "submitting", fixtures
    )
    created = CorpusDocument("knowledge", str(uuid4()))
    state = record_replacement_document(state, created)

    with pytest.raises(WizardDiagnosticError, match="transition"):
        record_replacement_document(
            state,
            replace(created, status="save_submitted", task_id=str(uuid4())),
        )
    with pytest.raises(WizardDiagnosticError, match="must not have task_id"):
        replace(created, status="save_submitting", task_id=str(uuid4()))
    with pytest.raises(WizardDiagnosticError, match="integrity data"):
        replace(created, status="save_submitting", chunk_ids=(str(uuid4()),))

    submitting = replace(created, status="save_submitting")
    state = record_replacement_document(state, submitting)
    chunk_id = str(uuid4())
    with pytest.raises(WizardDiagnosticError, match="transition"):
        record_replacement_document(
            state,
            replace(
                submitting,
                status="saved",
                task_id=str(uuid4()),
                chunk_ids=(chunk_id,),
                storage_fingerprint="a" * 64,
            ),
        )

    path = tmp_path / "corpus-state.json"
    write_corpus_state(path, state)
    loaded = load_corpus_state(path, "wizard_diagnostic")
    assert loaded == state
    assert loaded.pending_replacement is not None
    assert loaded.pending_replacement.document("knowledge") == submitting


def test_reseed_promotes_before_document_scoped_cleanup(tmp_path: Path) -> None:
    fixtures_path = _fixture_tree(tmp_path)
    original, fixtures = _complete_state(fixtures_path, tmp_path)
    old_documents = original.active.documents if original.active else ()

    state = begin_replacement(original, "replacement", fixtures)
    replacement_documents = []
    for collection in ("knowledge", "policy"):
        state, document = _add_saved_document(state, collection)
        replacement_documents.append(document)
    replacement_documents = tuple(replacement_documents)
    state = mark_replacement_verified(state)
    promoted = promote_replacement(state)

    assert promoted.active is not None
    assert promoted.active.generation_id == "replacement"
    assert promoted.active.documents == replacement_documents
    assert promoted.pending_cleanup == old_documents

    for document in old_documents:
        promoted = remove_pending_cleanup(promoted, document)
    assert promoted.pending_cleanup == ()


def _membership_for_state(state) -> dict[str, dict[str, str]]:
    membership = {"knowledge": {}, "policy": {}}
    generations = tuple(
        generation
        for generation in (state.active, state.pending_replacement)
        if generation is not None
    )
    for generation in generations:
        for document in generation.documents:
            for chunk_id in document.chunk_ids:
                membership[document.collection][chunk_id] = document.wizard_id
    for document in state.pending_cleanup:
        for chunk_id in document.chunk_ids:
            membership[document.collection][chunk_id] = document.wizard_id
    return membership


def test_physical_membership_requires_exact_active_chunks(tmp_path: Path) -> None:
    state, _ = _complete_state(_fixture_tree(tmp_path), tmp_path)
    observed = _membership_for_state(state)

    assert validate_physical_membership(state, observed) == {
        "knowledge": 1,
        "policy": 1,
    }

    missing = {name: dict(items) for name, items in observed.items()}
    missing["knowledge"].pop(next(iter(missing["knowledge"])))
    with pytest.raises(WizardDiagnosticError, match="missing recorded chunks"):
        validate_physical_membership(state, missing)

    unknown_document = {name: dict(items) for name, items in observed.items()}
    unknown_document["knowledge"][str(uuid4())] = str(uuid4())
    with pytest.raises(WizardDiagnosticError, match="unknown documents"):
        validate_physical_membership(state, unknown_document)

    active_knowledge = state.active.document("knowledge")  # type: ignore[union-attr]
    unknown_chunk = {name: dict(items) for name, items in observed.items()}
    unknown_chunk["knowledge"][str(uuid4())] = active_knowledge.wizard_id
    with pytest.raises(WizardDiagnosticError, match="unknown chunks"):
        validate_physical_membership(state, unknown_chunk)


def test_physical_membership_allows_recorded_interrupted_save_only(
    tmp_path: Path,
) -> None:
    original, fixtures = _complete_state(_fixture_tree(tmp_path), tmp_path)
    state = begin_replacement(original, "interrupted", fixtures)
    created = CorpusDocument("knowledge", str(uuid4()))
    state = record_replacement_document(state, created)
    submitting = replace(created, status="save_submitting")
    state = record_replacement_document(state, submitting)
    observed = _membership_for_state(state)
    observed["knowledge"][str(uuid4())] = submitting.wizard_id

    validate_physical_membership(state, observed)

    submitted = replace(
        submitting, status="save_submitted", task_id=str(uuid4())
    )
    state = record_replacement_document(state, submitted)

    validate_physical_membership(state, observed)

    observed["knowledge"][str(uuid4())] = str(uuid4())
    with pytest.raises(WizardDiagnosticError, match="unknown documents"):
        validate_physical_membership(state, observed)

    created_state = begin_replacement(original, "created-only", fixtures)
    created_document = CorpusDocument("knowledge", str(uuid4()))
    created_state = record_replacement_document(created_state, created_document)
    created_membership = _membership_for_state(created_state)
    created_membership["knowledge"][str(uuid4())] = created_document.wizard_id
    with pytest.raises(WizardDiagnosticError, match="unexpectedly has chunks"):
        validate_physical_membership(created_state, created_membership)


def test_physical_membership_allows_partial_recorded_pending_cleanup(
    tmp_path: Path,
) -> None:
    original, fixtures = _complete_state(_fixture_tree(tmp_path), tmp_path)
    state = begin_replacement(original, "replacement", fixtures)
    for collection in ("knowledge", "policy"):
        state, _ = _add_saved_document(state, collection)
    promoted = promote_replacement(mark_replacement_verified(state))
    observed = _membership_for_state(promoted)
    old_knowledge = next(
        item for item in promoted.pending_cleanup if item.collection == "knowledge"
    )
    observed["knowledge"].pop(old_knowledge.chunk_ids[0])

    validate_physical_membership(promoted, observed)


def test_promotion_requires_complete_verified_generation(tmp_path: Path) -> None:
    fixtures = preflight_wizard_fixtures(_fixture_tree(tmp_path), tmp_path)
    state = begin_replacement(
        new_corpus_state("wizard_diagnostic"), "replacement", fixtures
    )
    with pytest.raises(WizardDiagnosticError, match="not complete"):
        mark_replacement_verified(state)

    for collection in ("knowledge", "policy"):
        state, _ = _add_saved_document(state, collection)
    with pytest.raises(WizardDiagnosticError, match="not been verified"):
        promote_replacement(state)

    verified = mark_replacement_verified(state)
    assert verified.pending_replacement is not None
    assert verified.pending_replacement.verified_at is not None
    assert promote_replacement(verified).active is not None


def test_interrupted_replacement_is_recoverable_without_losing_active(
    tmp_path: Path,
) -> None:
    fixtures_path = _fixture_tree(tmp_path)
    original, fixtures = _complete_state(fixtures_path, tmp_path)
    state = begin_replacement(original, "interrupted", fixtures)
    interrupted = CorpusDocument("knowledge", str(uuid4()))
    state = record_replacement_document(state, interrupted)

    recovered = discard_pending_replacement(state)

    assert recovered.active == original.active
    assert recovered.pending_replacement is None


def test_replacement_document_transitions_preserve_recorded_identity(
    tmp_path: Path,
) -> None:
    fixtures = preflight_wizard_fixtures(_fixture_tree(tmp_path), tmp_path)
    state = begin_replacement(
        new_corpus_state("wizard_diagnostic"), "transition", fixtures
    )
    created = CorpusDocument("knowledge", str(uuid4()))

    with pytest.raises(WizardDiagnosticError, match="recorded at Create"):
        record_replacement_document(
            state,
            CorpusDocument(
                "knowledge",
                str(uuid4()),
                status="saved",
                task_id=str(uuid4()),
                chunk_ids=(str(uuid4()),),
                storage_fingerprint="a" * 64,
            ),
        )

    state = record_replacement_document(state, created)
    with pytest.raises(WizardDiagnosticError, match="identity cannot change"):
        record_replacement_document(
            state,
            replace(
                created,
                wizard_id=str(uuid4()),
                status="save_submitted",
                task_id=str(uuid4()),
            ),
        )


def test_corpus_preflight_decides_create_reuse_and_reseed(tmp_path: Path) -> None:
    fixtures_path = _fixture_tree(tmp_path)
    active, fixtures = _complete_state(fixtures_path, tmp_path)

    assert validate_corpus_preflight(None, fixtures, reseed=False) == "create"
    assert validate_corpus_preflight(active, fixtures, reseed=False) == "reuse"
    assert validate_corpus_preflight(active, fixtures, reseed=True) == "reseed"

    (fixtures_path / "knowledge" / "knowledge.txt").write_text(
        "Changed fact.", encoding="utf-8"
    )
    changed = preflight_wizard_fixtures(fixtures_path, tmp_path)
    with pytest.raises(WizardDiagnosticError, match="--reseed-corpus"):
        validate_corpus_preflight(active, changed, reseed=False)
    assert validate_corpus_preflight(active, changed, reseed=True) == "reseed"


@pytest.mark.parametrize(
    "counts",
    [
        {"knowledge": 1, "policy": 0},
        {"knowledge": 0, "policy": 2},
    ],
)
def test_first_seed_fails_closed_when_untracked_objects_exist(
    counts: dict[str, int],
) -> None:
    with pytest.raises(WizardDiagnosticError, match="state is missing"):
        require_empty_first_seed(counts)

    require_empty_first_seed({"knowledge": 0, "policy": 0})


def test_corpus_state_rejects_wrong_user_corruption_and_unknown_ids(
    tmp_path: Path,
) -> None:
    state, _ = _complete_state(_fixture_tree(tmp_path), tmp_path)
    path = tmp_path / "corpus-state.json"
    write_corpus_state(path, state)

    with pytest.raises(WizardDiagnosticError, match="different diagnostic user"):
        load_corpus_state(path, "someone_else")

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["active"]["fixture_digest"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WizardDiagnosticError, match="fixture digest"):
        load_corpus_state(path, "wizard_diagnostic")

    write_corpus_state(path, state)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pending_cleanup"] = [payload["active"]["documents"][0]]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WizardDiagnosticError, match="duplicate document"):
        load_corpus_state(path, "wizard_diagnostic")

    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(WizardDiagnosticError, match="unreadable"):
        load_corpus_state(path, "wizard_diagnostic")


def test_corpus_state_lock_is_nonblocking(tmp_path: Path) -> None:
    lock_path = tmp_path / "corpus-state.lock"
    with corpus_state_lock(lock_path):
        with pytest.raises(WizardDiagnosticError, match="already running"):
            with corpus_state_lock(lock_path):
                pass

    with corpus_state_lock(lock_path):
        assert lock_path.is_file()


def test_corpus_state_and_output_reject_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    state_link = tmp_path / "corpus-state.json"
    state_link.symlink_to(target)
    with pytest.raises(WizardDiagnosticError, match="must not be a symlink"):
        load_corpus_state(state_link, "wizard_diagnostic")

    fixtures = preflight_wizard_fixtures(_fixture_tree(tmp_path), tmp_path)
    output_target = tmp_path / "real-output"
    output_target.mkdir()
    output_link = tmp_path / "output"
    output_link.symlink_to(output_target, target_is_directory=True)
    with pytest.raises(WizardDiagnosticError, match="must not be a symlink"):
        create_diagnostic_run(output_link, "wizard_diagnostic", fixtures)


def test_diagnostic_run_fails_when_output_root_cannot_be_created(
    tmp_path: Path,
) -> None:
    fixtures_path = _fixture_tree(tmp_path)
    fixtures = preflight_wizard_fixtures(fixtures_path, tmp_path)
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")

    with pytest.raises(WizardDiagnosticError, match="Could not create"):
        create_diagnostic_run(blocked, "wizard_diagnostic", fixtures)


def test_repository_tracks_phase_1a_scaffold_and_ignores_outputs() -> None:
    assert (
        ragctl.PROJECT_ROOT
        / "diagnostics/fixtures/wizard/knowledge/.gitkeep"
    ).is_file()
    assert (
        ragctl.PROJECT_ROOT / "diagnostics/fixtures/wizard/policy/.gitkeep"
    ).is_file()
    assert "/.local/diagnostics/" in (
        ragctl.PROJECT_ROOT / ".gitignore"
    ).read_text(encoding="utf-8")
    assert "RAG_DIAGNOSTIC_USER_ID=wizard_diagnostic" in (
        ragctl.PROJECT_ROOT / ".env.example"
    ).read_text(encoding="utf-8")
