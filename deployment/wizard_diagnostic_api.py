"""Real API, storage, and bounded telemetry for the Wizard diagnostic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import math
from pathlib import Path
import struct
import time
from typing import Any
from uuid import UUID, uuid4

import httpx

from backend.config import get_collection_name
from backend.model_config import GTE_EMBEDDING_DIMENSION, LATEON_EMBEDDING_DIMENSION
from backend.wizard.diagnostics import (
    TRACE_OPERATION_HEADER,
    TRACE_SAMPLE_LIMIT,
    TRACE_SCHEMA_VERSION,
    TRACE_SESSION_HEADER,
    mapping_membership_digest,
)
from deployment.wizard_diagnostic import (
    STORAGE_FINGERPRINT_VERSION,
    CorpusDocument,
    CorpusGeneration,
    CorpusState,
    OperationRecorder,
    WizardDiagnosticError,
    WizardFixtures,
    begin_replacement,
    discard_pending_replacement,
    mark_replacement_verified,
    new_corpus_state,
    promote_replacement,
    record_active_ingestion_policy,
    record_replacement_document,
    remove_pending_cleanup,
    validate_physical_membership,
    write_corpus_state,
)


COLLECTION_PREFIXES = {"knowledge": "/api/knowledge", "policy": "/api/policy"}
TASK_TIMEOUT_SECONDS = 900.0
TASK_POLL_SECONDS = 1.0
TRACE_PATH = "/api/_diagnostics/wizard/trace"
COMPENSATION_STATUS = (
    "Compensation path not acceptance-tested because no safe deterministic "
    "natural failure seam exists."
)


def _collection_type(collection: str) -> str:
    if collection == "knowledge":
        return "knowledge_facts"
    if collection == "policy":
        return "policy"
    raise WizardDiagnosticError(f"Unknown Wizard collection: {collection}")


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WizardDiagnosticError(f"{context} returned malformed JSON")
    return value


def _mapping_list(value: object, context: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise WizardDiagnosticError(f"{context} returned malformed JSON")
    return value


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise WizardDiagnosticError(f"Response is missing {name}")
    return value


def _wizard_id(payload: Mapping[str, Any]) -> str:
    return _required_string(payload.get("wizard_id"), "wizard_id")


def _task_id(payload: Mapping[str, Any]) -> str:
    return _required_string(payload.get("task_id"), "task_id")


def _paragraph_ids(payload: Mapping[str, Any]) -> tuple[int, ...]:
    values = payload.get("paragraph_ids")
    if not isinstance(values, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0
        for item in values
    ):
        raise WizardDiagnosticError("Wizard returned invalid paragraph IDs")
    return tuple(values)


def _parse_timestamp(value: object, name: str) -> datetime:
    raw = _required_string(value, name)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WizardDiagnosticError(f"Task has invalid {name}") from exc


def _frame(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def validate_trace_evidence(
    operation: Mapping[str, Any],
    *,
    kind: str,
    outcome: str,
    required_stages: Sequence[str] = (),
    proof_samples: Sequence[str] = (),
    expected_task_id: str | None = None,
) -> None:
    """Fail closed on incomplete or malformed bounded trace evidence."""

    if operation.get("kind") != kind or operation.get("outcome") != outcome:
        raise WizardDiagnosticError("Wizard trace outcome is inconsistent")
    if not isinstance(operation.get("operation_id"), str):
        raise WizardDiagnosticError("Wizard trace has no operation correlation")
    if expected_task_id is not None and operation.get("task_id") != expected_task_id:
        raise WizardDiagnosticError("Wizard trace task correlation is inconsistent")
    if outcome in {"succeeded", "failed"} and not isinstance(
        operation.get("finished_at"), str
    ):
        raise WizardDiagnosticError("Wizard trace has no completion timestamp")

    stages = _mapping(operation.get("stages"), "trace stages")
    for name in required_stages:
        stage = _mapping(stages.get(name), f"trace stage {name}")
        call_count = stage.get("call_count")
        total_ms = stage.get("total_ms")
        minimum = stage.get("min_ms")
        maximum = stage.get("max_ms")
        failure_count = stage.get("failure_count")
        if (
            isinstance(call_count, bool)
            or not isinstance(call_count, int)
            or call_count <= 0
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0
                for value in (total_ms, minimum, maximum)
            )
            or float(minimum) > float(maximum)
            or float(maximum) > float(total_ms) + 0.001
            or isinstance(failure_count, bool)
            or not isinstance(failure_count, int)
            or failure_count < 0
            or failure_count > call_count
            or (outcome == "succeeded" and failure_count != 0)
        ):
            raise WizardDiagnosticError(f"Wizard trace stage {name} is malformed")

    samples = _mapping(operation.get("samples"), "trace samples")
    for name in proof_samples:
        sample = _mapping(samples.get(name), f"trace sample {name}")
        items = sample.get("items")
        exact_count = sample.get("exact_count")
        if (
            not isinstance(items, list)
            or isinstance(exact_count, bool)
            or not isinstance(exact_count, int)
            or exact_count != len(items)
            or sample.get("truncated") is not False
        ):
            raise WizardDiagnosticError(
                f"Wizard trace sample {name} is proof-critically truncated"
            )


def validate_upload_zero_side_effects(operation: Mapping[str, Any]) -> None:
    validate_trace_evidence(
        operation,
        kind="upload",
        outcome="succeeded",
        required_stages=(
            "api.lookup_validation",
            "upload.multipart_copy",
            "upload.file_decode_read",
            "upload.draft_merge",
            "upload.server_total",
        ),
    )
    flags = _mapping(operation.get("flags"), "upload trace flags")
    required_false = (
        "task_enqueued",
        "task_id_present",
        "save_delete_executed",
        "lateon_executed",
        "gte_executed",
        "weaviate_mutation_executed",
    )
    if operation.get("task_id") is not None or any(
        flags.get(name) is not False for name in required_false
    ):
        raise WizardDiagnosticError("Upload trace reports a forbidden side effect")
    forbidden_prefixes = ("save.", "delete.", "compensation.")
    if any(
        name.startswith(forbidden_prefixes)
        or name.startswith("weaviate.delete_")
        for name in _mapping(operation.get("stages"), "upload trace stages")
    ):
        raise WizardDiagnosticError("Upload trace contains pipeline mutation stages")


def _validate_successful_save_trace(
    operation: Mapping[str, Any], task_id: str
) -> None:
    validate_trace_evidence(
        operation,
        kind="save",
        outcome="succeeded",
        expected_task_id=task_id,
        required_stages=(
            "api.lookup_validation",
            "collection.preparation",
            "enqueue",
            "save.diff",
            "save.recovery_snapshot",
            "save.old_chunk_delete",
            "save.semantic_split",
            "save.chunking",
            "save.lateon",
            "save.gte",
            "save.renumbering",
            "save.weaviate_insert",
            "save.paragraph_map_commit",
            "save.document_map_commit",
        ),
    )
    counts = _mapping(operation.get("counts"), "save trace counts")
    flags = _mapping(operation.get("flags"), "save trace flags")
    required_counts: dict[str, int] = {}
    for name in (
        "enqueue_count",
        "new_chunk_count",
        "weaviate_insert_count",
        "lateon_document_count",
        "lateon_total_rows",
        "lateon_min_rows",
        "lateon_max_rows",
        "gte_vector_count",
        "lateon_dimension",
        "gte_dimension",
    ):
        value = counts.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise WizardDiagnosticError(
                f"Save trace count {name} is incomplete"
            )
        required_counts[name] = value
    if (
        required_counts["enqueue_count"] != 1
        or required_counts["lateon_dimension"] != LATEON_EMBEDDING_DIMENSION
        or required_counts["gte_dimension"] != GTE_EMBEDDING_DIMENSION
        or required_counts["lateon_document_count"]
        != required_counts["gte_vector_count"]
        or required_counts["lateon_document_count"]
        != required_counts["new_chunk_count"]
        or required_counts["new_chunk_count"]
        != required_counts["weaviate_insert_count"]
        or required_counts["lateon_min_rows"]
        > required_counts["lateon_max_rows"]
        or required_counts["lateon_max_rows"]
        > required_counts["lateon_total_rows"]
        or flags.get("lateon_values_finite") is not True
        or flags.get("gte_values_finite") is not True
        or flags.get("save_delete_executed") is not True
    ):
        raise WizardDiagnosticError("Save trace model evidence is incomplete")


def _validate_successful_delete_trace(
    operation: Mapping[str, Any], task_id: str
) -> None:
    validate_trace_evidence(
        operation,
        kind="delete",
        outcome="succeeded",
        expected_task_id=task_id,
        required_stages=(
            "api.lookup_validation",
            "collection.preparation",
            "enqueue",
            "delete.recovery_snapshot",
            "delete.storage_delete",
            "weaviate.delete_mutation",
            "weaviate.delete_verification",
            "delete.paragraph_map_remove",
            "delete.document_map_remove",
        ),
    )
    counts = _mapping(operation.get("counts"), "delete trace counts")
    required: dict[str, int] = {}
    for name in (
        "enqueue_count",
        "recovery_snapshot_chunk_count",
        "weaviate_deleted_match_count",
        "weaviate_deleted_success_count",
        "weaviate_delete_remaining_count",
    ):
        value = counts.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WizardDiagnosticError(f"Delete trace count {name} is incomplete")
        required[name] = value
    if (
        required["enqueue_count"] != 1
        or required["recovery_snapshot_chunk_count"]
        != required["weaviate_deleted_match_count"]
        or required["weaviate_deleted_match_count"]
        != required["weaviate_deleted_success_count"]
        or required["weaviate_delete_remaining_count"] != 0
    ):
        raise WizardDiagnosticError("Delete trace storage counts are inconsistent")


def wizard_saved_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash only saved mapping state, excluding draft-only Upload fields."""

    full_text = payload.get("full_text")
    if not isinstance(full_text, str):
        raise WizardDiagnosticError("Wizard returned invalid full_text")
    digest = hashlib.sha256()
    _frame(digest, b"wizard-map-v1")
    _frame(digest, full_text.encode("utf-8"))
    for paragraph_id in _paragraph_ids(payload):
        _frame(digest, str(paragraph_id).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class UploadPart:
    filename: str
    content: bytes
    media_type: str = "text/plain"


def upload_form_fields(
    user_id: str,
    *,
    current_text: str | None = None,
    modified_paragraph_ids: Sequence[object] | object | None = None,
) -> dict[str, str | list[str]]:
    """Build the production multipart form shape without validating its values."""

    fields: dict[str, str | list[str]] = {"user_id": user_id}
    if current_text is not None:
        fields["current_text"] = current_text
    if modified_paragraph_ids is not None:
        values = (
            list(modified_paragraph_ids)
            if isinstance(modified_paragraph_ids, Sequence)
            and not isinstance(modified_paragraph_ids, (str, bytes))
            else [modified_paragraph_ids]
        )
        fields["modified_paragraph_ids"] = [str(value) for value in values]
    return fields


def save_request_payload(
    user_id: str, current_text: str, modified_paragraph_ids: object
) -> dict[str, object]:
    """Build the production Save JSON shape without diagnostic pre-validation."""

    return {
        "user_id": user_id,
        "current_text": current_text,
        "modified_paragraph_ids": modified_paragraph_ids,
    }


def upload_continuation(
    previous: Mapping[str, Any] | None,
) -> tuple[str | None, list[int] | None]:
    """Return the exact draft fields to carry into the next Upload batch."""

    if previous is None:
        return None, None
    full_text = previous.get("full_text")
    modified = previous.get("modified_paragraph_ids")
    if not isinstance(full_text, str):
        raise WizardDiagnosticError("Upload returned invalid full_text")
    if not isinstance(modified, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0
        for item in modified
    ):
        raise WizardDiagnosticError(
            "Upload returned invalid modified paragraph IDs"
        )
    return full_text, modified


def diagnostic_edit_markers(run_id: str) -> tuple[str, str]:
    """Create short, unique, non-secret tokens for destructive edit checks."""

    token = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16].upper()
    return f"WIZARDOLD{token}", f"WIZARDNEW{token}"


def marker_chunk_count(records: Sequence[object], marker: str) -> int:
    """Count matching chunks without returning or recording their raw text."""

    if not isinstance(marker, str) or not marker:
        raise WizardDiagnosticError("Diagnostic marker must not be empty")
    count = 0
    for record in records:
        raw_text = getattr(record, "raw_text", None)
        if not isinstance(raw_text, str):
            raise WizardDiagnosticError("Persisted chunk text is malformed")
        if marker in raw_text:
            count += 1
    return count


def verify_partial_resave_mappings(
    before: Mapping[int, Sequence[str]],
    after: Mapping[int, Sequence[str]],
    changed_paragraph_id: int,
    *,
    probed_old_owner_count: int,
    physical_chunk_ids: Sequence[str],
) -> dict[str, int]:
    """Prove retained versus replaced chunk IDs without inspecting text."""

    if changed_paragraph_id not in before or changed_paragraph_id not in after:
        raise WizardDiagnosticError("Changed paragraph is absent from mapping proof")
    unchanged = [item for item in before if item != changed_paragraph_id]
    if not unchanged:
        raise WizardDiagnosticError("Partial re-Save has no unchanged paragraph")
    if any(list(after.get(item, ())) != list(before[item]) for item in unchanged):
        raise WizardDiagnosticError("An unchanged paragraph changed chunk IDs")
    old_changed = tuple(before[changed_paragraph_id])
    new_changed = tuple(after[changed_paragraph_id])
    if not old_changed or not new_changed:
        raise WizardDiagnosticError("Changed paragraph has no before/after chunks")
    if len(old_changed) > TRACE_SAMPLE_LIMIT:
        raise WizardDiagnosticError(
            "Changed paragraph exceeds the bounded chunk-ID proof"
        )
    if set(old_changed).intersection(new_changed):
        raise WizardDiagnosticError("Changed paragraph retained an old chunk ID")
    mapped_after = {
        chunk_id for chunk_ids in after.values() for chunk_id in chunk_ids
    }
    if any(chunk_id in mapped_after for chunk_id in old_changed):
        raise WizardDiagnosticError("An old changed chunk remains in ParagraphMap")
    if probed_old_owner_count != 0:
        raise WizardDiagnosticError("An old changed chunk has a mapping owner")
    physical = set(physical_chunk_ids)
    if any(chunk_id in physical for chunk_id in old_changed):
        raise WizardDiagnosticError("An old changed chunk remains in Weaviate")
    return {
        "unchanged_paragraph_count": len(unchanged),
        "old_changed_chunk_count": len(old_changed),
        "new_changed_chunk_count": len(new_changed),
    }


class WizardApi:
    """Strict authenticated client for the existing Wizard and task APIs."""

    def __init__(
        self,
        base_url: str,
        headers: Mapping[str, str],
        recorder: OperationRecorder,
        diagnostic_user_id: str,
        run_id: str,
    ) -> None:
        self.recorder = recorder
        self.diagnostic_user_id = diagnostic_user_id
        self.run_id = run_id
        self.session_id = str(uuid4())
        self._operation_ids: dict[str, str] = {}
        self._task_operation_ids: dict[str, str] = {}
        self.trace_started = False
        self.trace_deleted = False
        timeout = httpx.Timeout(connect=30, read=120, write=120, pool=30)
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=dict(headers),
            timeout=timeout,
        )
        self.anonymous_client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )

    def close(self) -> None:
        try:
            self.anonymous_client.close()
        finally:
            self.client.close()

    @staticmethod
    def _trace_kind(method: str, path: str) -> str | None:
        if "/wizards/" not in path:
            return None
        if method == "POST" and path.endswith("/upload"):
            return "upload"
        if method == "PUT":
            return "save"
        if method == "DELETE":
            return "delete"
        return None

    def _request(
        self,
        name: str,
        method: str,
        path: str,
        expected_status: int,
        *,
        selected_ids: Mapping[str, str] | None = None,
        **kwargs: object,
    ) -> httpx.Response:
        started = time.monotonic()
        response: httpx.Response | None = None
        operation_id: str | None = None
        trace_kind = self._trace_kind(method, path)
        if (
            trace_kind is not None
            and selected_ids is not None
            and selected_ids.get("user_id") == self.diagnostic_user_id
        ):
            if not self.trace_started:
                raise WizardDiagnosticError(
                    "A Wizard trace session must be started before diagnostic work"
                )
            operation_id = str(uuid4())
            request_headers = dict(kwargs.pop("headers", {}) or {})
            request_headers[TRACE_SESSION_HEADER] = self.session_id
            request_headers[TRACE_OPERATION_HEADER] = operation_id
            kwargs["headers"] = request_headers
            self._operation_ids[name] = operation_id
        try:
            response = self.client.request(method, path, **kwargs)
            if response.status_code != expected_status:
                raise WizardDiagnosticError(
                    f"{name} returned HTTP {response.status_code}; "
                    f"expected {expected_status}"
                )
        except BaseException:
            self.recorder.record(
                name,
                "failed",
                method=method,
                path=path,
                expected_status=expected_status,
                actual_status=response.status_code if response is not None else None,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
                selected_ids=dict(selected_ids) if selected_ids else None,
                diagnostic_operation_id=operation_id,
            )
            raise
        self.recorder.record(
            name,
            "passed",
            method=method,
            path=path,
            expected_status=expected_status,
            actual_status=response.status_code,
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            selected_ids=dict(selected_ids) if selected_ids else None,
            diagnostic_operation_id=operation_id,
        )
        return response

    def start_trace(self) -> None:
        response = self._request(
            "telemetry.trace.start",
            "POST",
            TRACE_PATH,
            201,
            selected_ids={"user_id": self.diagnostic_user_id},
            json={
                "user_id": self.diagnostic_user_id,
                "session_id": self.session_id,
                "run_id": self.run_id,
            },
        )
        # HTTP 201 means the server allocated the session; remember that before
        # validating its response so every later failure still deletes it.
        self.trace_started = True
        payload = _mapping(response.json(), "trace start")
        if (
            payload.get("schema_version") != TRACE_SCHEMA_VERSION
            or payload.get("session_id") != self.session_id
            or payload.get("user_id") != self.diagnostic_user_id
            or payload.get("run_id") != self.run_id
        ):
            raise WizardDiagnosticError("Trace start returned an invalid contract")

        started = time.monotonic()
        anonymous = self.anonymous_client.get(
            TRACE_PATH,
            params={
                "user_id": self.diagnostic_user_id,
                "session_id": self.session_id,
            },
        )
        passed = anonymous.status_code == 401
        self.recorder.record(
            "telemetry.security.anonymous",
            "passed" if passed else "failed",
            method="GET",
            path=TRACE_PATH,
            expected_status=401,
            actual_status=anonymous.status_code,
            duration_ms=round((time.monotonic() - started) * 1000, 3),
        )
        if not passed:
            raise WizardDiagnosticError(
                "Anonymous Wizard diagnostic access did not return HTTP 401"
            )

    def trace_snapshot(
        self,
        operation_name: str | None = None,
        *,
        collection: str | None = None,
        wizard_id: str | None = None,
        probe_chunk_ids: Sequence[str] = (),
    ) -> Mapping[str, Any]:
        params: list[tuple[str, str]] = [
            ("user_id", self.diagnostic_user_id),
            ("session_id", self.session_id),
        ]
        if operation_name is not None:
            operation_id = self._operation_ids.get(operation_name)
            if operation_id is None:
                raise WizardDiagnosticError(
                    f"Required trace correlation is missing for {operation_name}"
                )
            params.append(("operation_id", operation_id))
        if (collection is None) != (wizard_id is None):
            raise WizardDiagnosticError(
                "Trace mapping collection and wizard must be supplied together"
            )
        if collection is not None and wizard_id is not None:
            params.extend(
                (
                    ("collection_type", _collection_type(collection)),
                    ("wizard_id", wizard_id),
                )
            )
        if len(probe_chunk_ids) > TRACE_SAMPLE_LIMIT:
            raise WizardDiagnosticError("Trace chunk probe exceeds the proof bound")
        params.extend(("probe_chunk_ids", value) for value in probe_chunk_ids)
        response = self._request(
            "telemetry.trace.get",
            "GET",
            TRACE_PATH,
            200,
            selected_ids={"user_id": self.diagnostic_user_id},
            params=params,
        )
        payload = _mapping(response.json(), "trace GET")
        if (
            payload.get("schema_version") != TRACE_SCHEMA_VERSION
            or payload.get("session_id") != self.session_id
            or payload.get("user_id") != self.diagnostic_user_id
            or payload.get("run_id") != self.run_id
            or payload.get("overflowed") is not False
            or payload.get("trace_faulted") is not False
            or payload.get("missing_evidence") is not False
        ):
            raise WizardDiagnosticError(
                "Wizard trace is missing, overflowed, or faulted"
            )
        return payload

    def trace_operation(
        self,
        operation_name: str,
        *,
        collection: str | None = None,
        wizard_id: str | None = None,
        probe_chunk_ids: Sequence[str] = (),
    ) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
        payload = self.trace_snapshot(
            operation_name,
            collection=collection,
            wizard_id=wizard_id,
            probe_chunk_ids=probe_chunk_ids,
        )
        operations = payload.get("operations")
        if not isinstance(operations, list) or len(operations) != 1:
            raise WizardDiagnosticError("Trace GET did not return one operation")
        operation = _mapping(operations[0], "trace operation")
        expected_id = self._operation_ids[operation_name]
        if operation.get("operation_id") != expected_id:
            raise WizardDiagnosticError("Trace operation correlation is invalid")
        checkpoint_value = payload.get("mapping_checkpoint")
        checkpoint = (
            _mapping(checkpoint_value, "mapping checkpoint")
            if checkpoint_value is not None
            else None
        )
        return operation, checkpoint

    def mapping_checkpoint(
        self, collection: str, user_id: str, wizard_id: str
    ) -> Mapping[str, Any] | None:
        if user_id != self.diagnostic_user_id:
            return None
        payload = self.trace_snapshot(collection=collection, wizard_id=wizard_id)
        return _mapping(payload.get("mapping_checkpoint"), "mapping checkpoint")

    def delete_trace(self) -> None:
        self._request(
            "telemetry.trace.delete",
            "DELETE",
            TRACE_PATH,
            204,
            selected_ids={"user_id": self.diagnostic_user_id},
            params={
                "user_id": self.diagnostic_user_id,
                "session_id": self.session_id,
            },
        )
        self.trace_deleted = True

    def _verify_error(
        self, name: str, payload: Mapping[str, Any], expected_code: str
    ) -> None:
        detail = _mapping(payload.get("detail"), name)
        valid = detail.get("code") == expected_code and isinstance(
            detail.get("request_id"), str
        )
        self.recorder.record(name + ".contract", "passed" if valid else "failed")
        if not valid:
            raise WizardDiagnosticError(f"{name} returned an invalid error contract")

    def expect_error(
        self,
        name: str,
        method: str,
        path: str,
        expected_status: int,
        expected_code: str,
        *,
        selected_ids: Mapping[str, str] | None = None,
        **kwargs: object,
    ) -> None:
        response = self._request(
            name, method, path, expected_status, selected_ids=selected_ids, **kwargs
        )
        self._verify_error(name, _mapping(response.json(), name), expected_code)

    def create(
        self, collection: str, user_id: str, *, operation_name: str | None = None
    ) -> Mapping[str, Any]:
        prefix = COLLECTION_PREFIXES[collection]
        response = self._request(
            operation_name or f"scratch.{collection}.create",
            "POST",
            f"{prefix}/wizards",
            201,
            selected_ids={"user_id": user_id},
            json={"user_id": user_id},
        )
        payload = _mapping(response.json(), "wizard create")
        _wizard_id(payload)
        if payload.get("user_id") != user_id:
            raise WizardDiagnosticError("Created wizard has the wrong user")
        return payload

    def list_wizards(self, collection: str, user_id: str) -> list[Mapping[str, Any]]:
        prefix = COLLECTION_PREFIXES[collection]
        response = self._request(
            f"scratch.{collection}.list",
            "GET",
            f"{prefix}/wizards",
            200,
            selected_ids={"user_id": user_id},
            params={"user_id": user_id},
        )
        return _mapping_list(response.json(), "wizard list")

    def get(
        self,
        collection: str,
        user_id: str,
        wizard_id: str,
        *,
        operation_name: str | None = None,
    ) -> Mapping[str, Any]:
        prefix = COLLECTION_PREFIXES[collection]
        response = self._request(
            operation_name or f"scratch.{collection}.get",
            "GET",
            f"{prefix}/wizards/{wizard_id}",
            200,
            selected_ids={"user_id": user_id, "wizard_id": wizard_id},
            params={"user_id": user_id},
        )
        payload = _mapping(response.json(), "wizard get")
        if payload.get("wizard_id") != wizard_id or payload.get("user_id") != user_id:
            raise WizardDiagnosticError("Fetched wizard identity is incorrect")
        return payload

    def upload(
        self,
        collection: str,
        user_id: str,
        wizard_id: str,
        *,
        operation_name: str,
        paths: Sequence[Path] = (),
        parts: Sequence[UploadPart] = (),
        current_text: str | None = None,
        modified_paragraph_ids: Sequence[object] | object | None = None,
        expected_status: int = 200,
        expected_code: str | None = None,
    ) -> Mapping[str, Any]:
        prefix = COLLECTION_PREFIXES[collection]
        with ExitStack() as stack:
            multipart: list[tuple[str, tuple[str, Any, str]]] = [
                (
                    "files",
                    (path.name, stack.enter_context(path.open("rb")), "text/plain"),
                )
                for path in paths
            ]
            multipart.extend(
                ("files", (part.filename, part.content, part.media_type))
                for part in parts
            )
            response = self._request(
                operation_name,
                "POST",
                f"{prefix}/wizards/{wizard_id}/upload",
                expected_status,
                selected_ids={"user_id": user_id, "wizard_id": wizard_id},
                data=upload_form_fields(
                    user_id,
                    current_text=current_text,
                    modified_paragraph_ids=modified_paragraph_ids,
                ),
                files=multipart,
            )
        payload = _mapping(response.json(), operation_name)
        if expected_code is not None:
            self._verify_error(operation_name, payload, expected_code)
        return payload

    def submit_save(
        self,
        collection: str,
        user_id: str,
        wizard_id: str,
        current_text: str,
        modified_paragraph_ids: object,
        *,
        operation_name: str,
        expected_status: int = 202,
        expected_code: str | None = None,
    ) -> Mapping[str, Any]:
        prefix = COLLECTION_PREFIXES[collection]
        response = self._request(
            operation_name,
            "PUT",
            f"{prefix}/wizards/{wizard_id}",
            expected_status,
            selected_ids={"user_id": user_id, "wizard_id": wizard_id},
            json=save_request_payload(user_id, current_text, modified_paragraph_ids),
        )
        payload = _mapping(response.json(), operation_name)
        if expected_code is not None:
            self._verify_error(operation_name, payload, expected_code)
            return payload
        task = _task_id(payload)
        if payload.get("user_id") != user_id or payload.get("status") not in {
            "queued",
            "running",
            "succeeded",
        }:
            raise WizardDiagnosticError("Save returned an invalid task")
        operation_id = self._operation_ids.get(operation_name)
        if operation_id is not None:
            self._task_operation_ids[task] = operation_id
        return payload

    def submit_delete(
        self,
        collection: str,
        user_id: str,
        wizard_id: str,
        *,
        operation_name: str | None = None,
    ) -> Mapping[str, Any]:
        prefix = COLLECTION_PREFIXES[collection]
        name = operation_name or f"scratch.{collection}.delete"
        response = self._request(
            name,
            "DELETE",
            f"{prefix}/wizards/{wizard_id}",
            202,
            selected_ids={"user_id": user_id, "wizard_id": wizard_id},
            params={"user_id": user_id},
        )
        payload = _mapping(response.json(), "wizard delete")
        task = _task_id(payload)
        operation_id = self._operation_ids.get(name)
        if operation_id is not None:
            self._task_operation_ids[task] = operation_id
        return payload

    def poll_task(
        self,
        user_id: str,
        task_id: str,
        *,
        operation_name: str,
        expected_terminal: str = "succeeded",
        expected_error_code: str | None = None,
    ) -> Mapping[str, Any]:
        if expected_terminal not in {"succeeded", "failed"}:
            raise WizardDiagnosticError("Expected task terminal state is invalid")
        started = time.monotonic()
        attempts = 0
        statuses: list[str] = []
        while True:
            attempts += 1
            response = self.client.get(
                f"/api/tasks/{task_id}", params={"user_id": user_id}
            )
            if response.status_code != 200:
                self.recorder.record(
                    operation_name,
                    "failed",
                    method="GET",
                    path=f"/api/tasks/{task_id}",
                    expected_status=200,
                    actual_status=response.status_code,
                    poll_count=attempts,
                    selected_ids={"user_id": user_id, "task_id": task_id},
                )
                raise WizardDiagnosticError(
                    f"Task polling returned HTTP {response.status_code}"
                )
            payload = _mapping(response.json(), "task poll")
            status = payload.get("status")
            if not isinstance(status, str):
                raise WizardDiagnosticError("Task polling returned no status")
            if not statuses or statuses[-1] != status:
                statuses.append(status)
            if status in {"succeeded", "failed"}:
                valid = status == expected_terminal and (
                    expected_error_code is None
                    or payload.get("error_code") == expected_error_code
                )
                self.recorder.record(
                    operation_name,
                    "passed" if valid else "failed",
                    method="GET",
                    path=f"/api/tasks/{task_id}",
                    expected_status=200,
                    actual_status=200,
                    poll_count=attempts,
                    observed_statuses=statuses,
                    duration_ms=round((time.monotonic() - started) * 1000, 3),
                    selected_ids={"user_id": user_id, "task_id": task_id},
                )
                if not valid:
                    raise WizardDiagnosticError(
                        f"Task {task_id} ended as "
                        f"{status}/{payload.get('error_code')}"
                    )
                created_at = _parse_timestamp(payload.get("created_at"), "created_at")
                started_at = _parse_timestamp(payload.get("started_at"), "started_at")
                finished_at = _parse_timestamp(
                    payload.get("finished_at"), "finished_at"
                )
                queue_wait_ms = max(
                    0.0, (started_at - created_at).total_seconds() * 1000.0
                )
                task_execution_ms = max(
                    0.0, (finished_at - started_at).total_seconds() * 1000.0
                )
                task_total_ms = max(
                    0.0, (finished_at - created_at).total_seconds() * 1000.0
                )
                self.recorder.record(
                    operation_name + ".timings",
                    "passed",
                    queue_wait_ms=round(queue_wait_ms, 3),
                    task_execution_ms=round(task_execution_ms, 3),
                    task_total_ms=round(task_total_ms, 3),
                    selected_ids={"user_id": user_id, "task_id": task_id},
                )
                return payload
            if time.monotonic() - started >= TASK_TIMEOUT_SECONDS:
                self.recorder.record(
                    operation_name,
                    "failed",
                    method="GET",
                    path=f"/api/tasks/{task_id}",
                    expected_status=200,
                    actual_status=200,
                    poll_count=attempts,
                    observed_statuses=statuses,
                )
                raise WizardDiagnosticError(f"Task {task_id} timed out")
            time.sleep(TASK_POLL_SECONDS)


@dataclass(frozen=True)
class StorageIntegrity:
    chunk_ids: tuple[str, ...]
    fingerprint: str
    chunk_count: int
    lateon_dimension: int | None
    gte_dimension: int | None


def storage_integrity(records: Sequence[object]) -> StorageIntegrity:
    """Return a stable sanitized digest of complete persisted chunk records."""

    canonical: list[tuple[str, object]] = []
    for record in records:
        try:
            chunk_id = str(UUID(str(getattr(record, "chunk_id"))))
        except (TypeError, ValueError) as exc:
            raise WizardDiagnosticError("Persisted chunk ID is malformed") from exc
        canonical.append((chunk_id, record))
    canonical.sort(key=lambda item: item[0])
    if len({item[0] for item in canonical}) != len(canonical):
        raise WizardDiagnosticError("Persisted chunks contain duplicate IDs")

    digest = hashlib.sha256()
    _frame(digest, STORAGE_FINGERPRINT_VERSION.encode("ascii"))
    lateon_dimension: int | None = None
    gte_dimension: int | None = None
    for chunk_id, record in canonical:
        try:
            object_id = str(UUID(str(getattr(record, "object_id"))))
            document_id = str(UUID(str(getattr(record, "document_id"))))
        except (TypeError, ValueError) as exc:
            raise WizardDiagnosticError("Persisted chunk identity is malformed") from exc
        user_id = getattr(record, "user_id", None)
        paragraph_id = getattr(record, "paragraph_id", None)
        raw_text = getattr(record, "raw_text", None)
        late = getattr(record, "late_interaction", None)
        gte = getattr(record, "mmr_diversity", None)
        if object_id != chunk_id:
            raise WizardDiagnosticError("Persisted object and chunk IDs differ")
        if not isinstance(user_id, str) or not user_id:
            raise WizardDiagnosticError("Persisted chunk user is malformed")
        if (
            isinstance(paragraph_id, bool)
            or not isinstance(paragraph_id, int)
            or paragraph_id <= 0
        ):
            raise WizardDiagnosticError("Persisted paragraph ID is malformed")
        if not isinstance(raw_text, str) or not raw_text:
            raise WizardDiagnosticError("Persisted chunk text is empty")
        if not isinstance(late, Sequence) or not late:
            raise WizardDiagnosticError("Persisted LateOn vector is malformed")
        if not isinstance(gte, Sequence):
            raise WizardDiagnosticError("Persisted GTE vector is malformed")

        for value in (object_id, chunk_id, document_id, user_id):
            _frame(digest, value.encode("utf-8"))
        _frame(digest, str(paragraph_id).encode("ascii"))
        _frame(digest, raw_text.encode("utf-8"))
        _frame(digest, len(late).to_bytes(8, "big"))
        for row in late:
            if not isinstance(row, Sequence):
                raise WizardDiagnosticError("Persisted LateOn row is malformed")
            if lateon_dimension is None:
                lateon_dimension = len(row)
            elif len(row) != lateon_dimension:
                raise WizardDiagnosticError("Persisted LateOn shape is inconsistent")
            _frame(digest, len(row).to_bytes(8, "big"))
            for value in row:
                number = float(value)
                if not math.isfinite(number):
                    raise WizardDiagnosticError("Persisted LateOn value is invalid")
                _frame(digest, struct.pack(">d", number))
        if gte_dimension is None:
            gte_dimension = len(gte)
        elif len(gte) != gte_dimension:
            raise WizardDiagnosticError("Persisted GTE shape is inconsistent")
        _frame(digest, len(gte).to_bytes(8, "big"))
        for value in gte:
            number = float(value)
            if not math.isfinite(number):
                raise WizardDiagnosticError("Persisted GTE value is invalid")
            _frame(digest, struct.pack(">d", number))
    return StorageIntegrity(
        tuple(item[0] for item in canonical),
        digest.hexdigest(),
        len(canonical),
        lateon_dimension,
        gte_dimension,
    )


class CorpusStorage:
    """Direct verified access used only for inspection and document retirement."""

    def __init__(self, config: Mapping[str, str], diagnostic_user_id: str) -> None:
        self.config = config
        self.diagnostic_user_id = diagnostic_user_id
        self.manager: Any = None

    def __enter__(self) -> CorpusStorage:
        from weaviate import WeaviateClient
        from weaviate.classes.init import Auth
        from weaviate.connect import ConnectionParams

        from backend.weaviate_client.client import WeaviateManager

        secure = self.config["WEAVIATE_GRPC_SECURE"].strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        raw_client = WeaviateClient(
            connection_params=ConnectionParams.from_url(
                self.config["WEAVIATE_URL"],
                grpc_port=int(self.config["WEAVIATE_GRPC_PORT"]),
                grpc_secure=secure,
            ),
            auth_client_secret=Auth.api_key(self.config["WEAVIATE_API_KEY"]),
        )
        self.manager = WeaviateManager(raw_client)
        self.manager.connect()
        return self

    def __exit__(self, *_: object) -> None:
        if self.manager is not None:
            self.manager.disconnect()

    def _collection(self, user_id: str, collection: str) -> Any:
        if collection == "knowledge":
            from backend.weaviate_client.knowledge import KnowledgeCollection

            return KnowledgeCollection(self.manager, user_id)
        if collection == "policy":
            from backend.weaviate_client.policy import PolicyCollection

            return PolicyCollection(self.manager, user_id)
        raise WizardDiagnosticError(f"Unknown corpus collection: {collection}")

    def snapshots(
        self, user_id: str, collection: str, wizard_id: str
    ) -> tuple[Any, ...]:
        return self._collection(user_id, collection).snapshot_by_document(wizard_id)

    def inspect_document(
        self,
        user_id: str,
        collection: str,
        wizard_id: str,
        *,
        require_nonempty: bool = False,
    ) -> StorageIntegrity:
        records = self.snapshots(user_id, collection, wizard_id)
        if not records:
            if require_nonempty:
                raise WizardDiagnosticError(
                    "Saved corpus document has no persisted chunks"
                )
            return storage_integrity(())
        for record in records:
            if record.user_id != user_id or record.document_id != wizard_id:
                raise WizardDiagnosticError(
                    "Persisted corpus document escaped its scope"
                )
        result = storage_integrity(records)
        if result.lateon_dimension != LATEON_EMBEDDING_DIMENSION:
            raise WizardDiagnosticError("Persisted LateOn dimensions are invalid")
        if result.gte_dimension != GTE_EMBEDDING_DIMENSION:
            raise WizardDiagnosticError("Persisted GTE dimensions are invalid")
        return result

    def marker_count(
        self, user_id: str, collection: str, wizard_id: str, marker: str
    ) -> int:
        return marker_chunk_count(
            self.snapshots(user_id, collection, wizard_id), marker
        )

    def document_membership(
        self, user_id: str, collection: str, wizard_id: str
    ) -> dict[int, list[str]]:
        """Return a sanitized physical paragraph/chunk membership view."""

        mapping: dict[int, list[str]] = {}
        seen: set[str] = set()
        for record in self.snapshots(user_id, collection, wizard_id):
            if record.user_id != user_id or record.document_id != wizard_id:
                raise WizardDiagnosticError(
                    "Persisted corpus document escaped its scope"
                )
            try:
                chunk_id = str(UUID(str(record.chunk_id)))
            except (TypeError, ValueError) as exc:
                raise WizardDiagnosticError(
                    "Persisted chunk ID is malformed"
                ) from exc
            if chunk_id in seen:
                raise WizardDiagnosticError("Persisted chunk IDs are duplicated")
            seen.add(chunk_id)
            paragraph_id = record.paragraph_id
            if (
                isinstance(paragraph_id, bool)
                or not isinstance(paragraph_id, int)
                or paragraph_id <= 0
            ):
                raise WizardDiagnosticError(
                    "Persisted paragraph ID is malformed"
                )
            mapping.setdefault(paragraph_id, []).append(chunk_id)
        return {
            paragraph_id: sorted(chunk_ids)
            for paragraph_id, chunk_ids in sorted(mapping.items())
        }

    def inventory(self, user_id: str, collection: str) -> dict[str, str]:
        collection_type = (
            "knowledge_facts" if collection == "knowledge" else "policy"
        )
        name = get_collection_name(user_id, collection_type)
        if not self.manager.client.collections.exists(name):
            return {}
        physical = self.manager.client.collections.use(name)
        membership: dict[str, str] = {}
        for item in physical.iterator(
            include_vector=False,
            return_properties=["user_id", "document_id", "chunk_id"],
        ):
            properties = getattr(item, "properties", None)
            if not isinstance(properties, Mapping) or properties.get("user_id") != user_id:
                raise WizardDiagnosticError("Physical corpus membership is malformed")
            try:
                object_id = str(UUID(str(getattr(item, "uuid", None))))
                chunk_id = str(UUID(str(properties.get("chunk_id"))))
                document_id = str(UUID(str(properties.get("document_id"))))
            except (TypeError, ValueError) as exc:
                raise WizardDiagnosticError("Physical corpus IDs are malformed") from exc
            if object_id != chunk_id or chunk_id in membership:
                raise WizardDiagnosticError(
                    "Physical corpus chunk IDs are inconsistent"
                )
            membership[chunk_id] = document_id
        return membership

    def inventories(self) -> dict[str, dict[str, str]]:
        return {
            collection: self.inventory(self.diagnostic_user_id, collection)
            for collection in COLLECTION_PREFIXES
        }

    def validate_membership(self, state: CorpusState | None) -> None:
        validate_physical_membership(state, self.inventories())

    def ensure_empty_user_collections(self, user_id: str) -> None:
        """Create or validate exactly three empty isolation-user collections."""

        collections = self.manager.client.collections
        names = tuple(
            get_collection_name(user_id, collection_type)
            for collection_type in ("conversations", "knowledge_facts", "policy")
        )
        existing = {name for name in names if collections.exists(name)}
        if existing and len(existing) != len(names):
            raise WizardDiagnosticError(
                "Isolation user does not have zero or all three collections"
            )

        self.manager.ensure_user_collections(user_id)
        if not all(collections.exists(name) for name in names):
            raise WizardDiagnosticError(
                "Isolation user collection creation did not complete"
            )
        for name in names:
            physical = collections.use(name)
            if next(
                iter(
                    physical.iterator(
                        include_vector=False,
                        return_properties=[],
                    )
                ),
                None,
            ) is not None:
                raise WizardDiagnosticError(
                    "Isolation user collections contain unexpected data"
                )

    def delete_document(self, document: CorpusDocument) -> int:
        report = self._collection(
            self.diagnostic_user_id, document.collection
        ).delete_by_document(document.wizard_id)
        if not report.confirmed:
            raise WizardDiagnosticError(
                "Document-scoped corpus deletion was not confirmed"
            )
        remaining = self.inspect_document(
            self.diagnostic_user_id, document.collection, document.wizard_id
        )
        if remaining.chunk_count:
            raise WizardDiagnosticError(
                "Document-scoped deletion left persisted chunks"
            )
        return report.successful


def _check(
    recorder: OperationRecorder, name: str, condition: bool, **fields: object
) -> None:
    recorder.record(name, "passed" if condition else "failed", **fields)
    if not condition:
        raise WizardDiagnosticError(f"Diagnostic check failed: {name}")


def _checkpoint_paragraph_mapping(
    checkpoint: Mapping[str, Any], *, require_complete: bool
) -> dict[int, list[str]]:
    paragraph_map = _mapping(
        checkpoint.get("paragraph_map"), "mapping checkpoint paragraph map"
    )
    if require_complete and paragraph_map.get("truncated") is not False:
        raise WizardDiagnosticError(
            "Paragraph/chunk mapping exceeds the bounded proof sample"
        )
    sample = paragraph_map.get("sample")
    if not isinstance(sample, list):
        raise WizardDiagnosticError("Paragraph mapping sample is malformed")
    result: dict[int, list[str]] = {}
    for item in sample:
        entry = _mapping(item, "paragraph mapping sample")
        paragraph_id = entry.get("paragraph_id")
        chunk_ids = entry.get("chunk_ids")
        if (
            isinstance(paragraph_id, bool)
            or not isinstance(paragraph_id, int)
            or paragraph_id <= 0
            or not isinstance(chunk_ids, list)
        ):
            raise WizardDiagnosticError("Paragraph mapping sample is malformed")
        try:
            canonical = [str(UUID(str(chunk_id))) for chunk_id in chunk_ids]
        except (TypeError, ValueError) as exc:
            raise WizardDiagnosticError(
                "Paragraph mapping sample contains an invalid chunk ID"
            ) from exc
        result[paragraph_id] = canonical
    if require_complete:
        paragraph_count = paragraph_map.get("paragraph_count")
        chunk_count = paragraph_map.get("chunk_count")
        if paragraph_count != len(result) or chunk_count != sum(
            len(values) for values in result.values()
        ):
            raise WizardDiagnosticError("Paragraph mapping proof is incomplete")
    return result


def _verify_mapping_and_storage(
    checkpoint: Mapping[str, Any],
    physical_mapping: Mapping[int, Sequence[str]],
) -> None:
    paragraph_map = _mapping(
        checkpoint.get("paragraph_map"), "mapping checkpoint paragraph map"
    )
    physical_chunk_count = sum(len(values) for values in physical_mapping.values())
    if (
        paragraph_map.get("chunk_count") != physical_chunk_count
        or paragraph_map.get("membership_sha256")
        != mapping_membership_digest(physical_mapping)
    ):
        raise WizardDiagnosticError(
            "ParagraphMap and physical document membership differ"
        )


def _trace_artifact(operation: Mapping[str, Any]) -> dict[str, object]:
    """Select bounded server evidence that is safe for local artifacts."""

    return {
        key: operation.get(key)
        for key in (
            "operation_id",
            "kind",
            "user_id",
            "collection_type",
            "wizard_id",
            "task_id",
            "started_at",
            "finished_at",
            "outcome",
            "stages",
            "counts",
            "flags",
            "digests",
            "samples",
        )
    }


def _verify_save_postcondition(
    api: WizardApi,
    storage: CorpusStorage,
    collection: str,
    user_id: str,
    wizard_id: str,
    task_id: str,
    operation_name: str,
    expected_text: str,
    *,
    proof_samples: Sequence[str] = (),
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if user_id != api.diagnostic_user_id:
        return {}, {}
    started = time.monotonic()
    operation, checkpoint = api.trace_operation(
        operation_name, collection=collection, wizard_id=wizard_id
    )
    if checkpoint is None:
        raise WizardDiagnosticError("Save trace has no mapping checkpoint")
    _validate_successful_save_trace(operation, task_id)
    if proof_samples:
        validate_trace_evidence(
            operation,
            kind="save",
            outcome="succeeded",
            expected_task_id=task_id,
            proof_samples=proof_samples,
        )
    saved = api.get(
        collection,
        user_id,
        wizard_id,
        operation_name=operation_name + ".verification.get",
    )
    document = _mapping(checkpoint.get("document"), "document checkpoint")
    expected_hash = hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
    if (
        saved.get("full_text") != expected_text
        or document.get("present") is not True
        or document.get("full_text_sha256") != expected_hash
    ):
        raise WizardDiagnosticError("Saved Wizard mapping postcondition failed")
    integrity = storage.inspect_document(
        user_id, collection, wizard_id, require_nonempty=True
    )
    physical_mapping = storage.document_membership(user_id, collection, wizard_id)
    _verify_mapping_and_storage(checkpoint, physical_mapping)
    other = "policy" if collection == "knowledge" else "knowledge"
    if storage.snapshots(user_id, other, wizard_id):
        raise WizardDiagnosticError("Knowledge/Policy storage isolation failed")
    verification_latency_ms = round((time.monotonic() - started) * 1000, 3)
    api.recorder.record(
        operation_name + ".postcondition",
        "passed",
        verification_latency_ms=verification_latency_ms,
        chunk_count=integrity.chunk_count,
        lateon_dimension=integrity.lateon_dimension,
        gte_dimension=integrity.gte_dimension,
        trace=_trace_artifact(operation),
        selected_ids={
            "user_id": user_id,
            "wizard_id": wizard_id,
            "task_id": task_id,
        },
    )
    return operation, checkpoint


def _upload_without_mutation(
    api: WizardApi,
    storage: CorpusStorage,
    collection: str,
    user_id: str,
    wizard_id: str,
    *,
    operation_name: str,
    paths: Sequence[Path] = (),
    parts: Sequence[UploadPart] = (),
    current_text: str | None = None,
    modified_paragraph_ids: Sequence[object] | object | None = None,
    expected_status: int = 200,
    expected_code: str | None = None,
) -> Mapping[str, Any]:
    before_checkpoint = api.mapping_checkpoint(collection, user_id, wizard_id)
    before_map = wizard_saved_fingerprint(
        api.get(
            collection,
            user_id,
            wizard_id,
            operation_name=operation_name + ".before.get",
        )
    )
    before_storage = storage.inspect_document(user_id, collection, wizard_id)
    upload_error: BaseException | None = None
    result: Mapping[str, Any] | None = None
    try:
        result = api.upload(
            collection,
            user_id,
            wizard_id,
            operation_name=operation_name,
            paths=paths,
            parts=parts,
            current_text=current_text,
            modified_paragraph_ids=modified_paragraph_ids,
            expected_status=expected_status,
            expected_code=expected_code,
        )
    except BaseException as exc:
        upload_error = exc
    verification_started = time.monotonic()
    after_map = wizard_saved_fingerprint(
        api.get(
            collection,
            user_id,
            wizard_id,
            operation_name=operation_name + ".after.get",
        )
    )
    after_storage = storage.inspect_document(user_id, collection, wizard_id)
    operation: Mapping[str, Any] | None = None
    after_checkpoint: Mapping[str, Any] | None = None
    if user_id == api.diagnostic_user_id:
        operation, after_checkpoint = api.trace_operation(
            operation_name,
            collection=collection,
            wizard_id=wizard_id,
        )
        validate_trace_evidence(
            operation,
            kind="upload",
            outcome="succeeded" if expected_status == 200 else "failed",
            required_stages=("api.lookup_validation", "upload.server_total"),
        )
        if expected_status == 200:
            validate_upload_zero_side_effects(operation)
    selected = {"user_id": user_id, "wizard_id": wizard_id}
    _check(
        api.recorder,
        operation_name + ".saved_state_unchanged",
        before_map == after_map,
        selected_ids=selected,
    )
    _check(
        api.recorder,
        operation_name + ".paragraph_map_unchanged",
        before_checkpoint == after_checkpoint,
        verification_latency_ms=round(
            (time.monotonic() - verification_started) * 1000, 3
        ),
        trace=_trace_artifact(operation) if operation is not None else None,
        selected_ids=selected,
    )
    _check(
        api.recorder,
        operation_name + ".storage_unchanged",
        before_storage == after_storage,
        chunk_count=after_storage.chunk_count,
        selected_ids=selected,
    )
    if upload_error is not None:
        raise upload_error
    if result is None:  # pragma: no cover - guarded by the exception path above.
        raise WizardDiagnosticError("Upload returned no result")
    return result


def _failed_save_without_mutation(
    api: WizardApi,
    storage: CorpusStorage,
    collection: str,
    user_id: str,
    wizard_id: str,
    current_text: str,
) -> None:
    before_checkpoint = api.mapping_checkpoint(collection, user_id, wizard_id)
    before = api.get(
        collection,
        user_id,
        wizard_id,
        operation_name="scratch.save_invalid.before.get",
    )
    before_map = wizard_saved_fingerprint(before)
    before_storage = storage.inspect_document(user_id, collection, wizard_id)
    api.submit_save(
        collection,
        user_id,
        wizard_id,
        current_text,
        ["not-an-integer"],
        operation_name="scratch.save_invalid.structural",
        expected_status=422,
        expected_code="VALIDATION_ERROR",
    )
    nonexistent = max(_paragraph_ids(before), default=0) + 1_000_000
    task = api.submit_save(
        collection,
        user_id,
        wizard_id,
        current_text + "\ninvalid paragraph mutation",
        [nonexistent],
        operation_name="scratch.save_invalid.nonexistent",
    )
    failed_task_id = _task_id(task)
    api.poll_task(
        user_id,
        failed_task_id,
        operation_name="scratch.save_invalid.nonexistent.poll",
        expected_terminal="failed",
        expected_error_code="TASK_FAILED",
    )
    verification_started = time.monotonic()
    after = api.get(
        collection,
        user_id,
        wizard_id,
        operation_name="scratch.save_invalid.after.get",
    )
    operation, after_checkpoint = api.trace_operation(
        "scratch.save_invalid.nonexistent",
        collection=collection,
        wizard_id=wizard_id,
    )
    validate_trace_evidence(
        operation,
        kind="save",
        outcome="failed",
        expected_task_id=failed_task_id,
        required_stages=(
            "api.lookup_validation",
            "collection.preparation",
            "enqueue",
            "save.diff",
        ),
    )
    _check(
        api.recorder,
        "scratch.save_invalid.saved_state_unchanged",
        before_map == wizard_saved_fingerprint(after),
    )
    _check(
        api.recorder,
        "scratch.save_invalid.storage_unchanged",
        before_storage == storage.inspect_document(user_id, collection, wizard_id),
    )
    _check(
        api.recorder,
        "scratch.save_invalid.paragraph_map_unchanged",
        before_checkpoint == after_checkpoint,
        verification_latency_ms=round(
            (time.monotonic() - verification_started) * 1000, 3
        ),
        trace=_trace_artifact(operation),
    )


def _delete_scratch(
    api: WizardApi,
    storage: CorpusStorage,
    scratch: list[tuple[str, str, str]],
    recorder: OperationRecorder,
) -> None:
    errors: list[BaseException] = []
    for collection, user_id, wizard_id in reversed(scratch):
        try:
            delete_name = f"scratch.{collection}.delete.{wizard_id}"
            task = api.submit_delete(
                collection,
                user_id,
                wizard_id,
                operation_name=delete_name,
            )
            delete_task_id = _task_id(task)
            api.poll_task(
                user_id,
                delete_task_id,
                operation_name=f"scratch.{collection}.delete.poll",
            )
            verification_started = time.monotonic()
            api.expect_error(
                f"scratch.{collection}.delete.get",
                "GET",
                f"{COLLECTION_PREFIXES[collection]}/wizards/{wizard_id}",
                404,
                "NOT_FOUND",
                selected_ids={"user_id": user_id, "wizard_id": wizard_id},
                params={"user_id": user_id},
            )
            integrity = storage.inspect_document(user_id, collection, wizard_id)
            operation: Mapping[str, Any] | None = None
            checkpoint: Mapping[str, Any] | None = None
            if user_id == api.diagnostic_user_id:
                operation, checkpoint = api.trace_operation(
                    delete_name,
                    collection=collection,
                    wizard_id=wizard_id,
                )
                _validate_successful_delete_trace(operation, delete_task_id)
                document = _mapping(
                    checkpoint.get("document"), "delete document checkpoint"
                )
                paragraph_map = _mapping(
                    checkpoint.get("paragraph_map"),
                    "delete paragraph checkpoint",
                )
                if (
                    document.get("present") is not False
                    or paragraph_map.get("chunk_count") != 0
                    or paragraph_map.get("paragraph_count") != 0
                ):
                    raise WizardDiagnosticError(
                        "Delete left process-local Wizard mappings"
                    )
            _check(
                recorder,
                f"scratch.{collection}.delete.storage_empty",
                integrity.chunk_count == 0,
                verification_latency_ms=round(
                    (time.monotonic() - verification_started) * 1000, 3
                ),
                trace=_trace_artifact(operation) if operation is not None else None,
                selected_ids={"user_id": user_id, "wizard_id": wizard_id},
            )
        except BaseException as exc:
            errors.append(exc)
            recorder.record(
                f"scratch.{collection}.cleanup",
                "failed",
                selected_ids={"user_id": user_id, "wizard_id": wizard_id},
            )
    if errors:
        raise WizardDiagnosticError(
            f"Could not delete {len(errors)} known scratch wizard(s)"
        ) from errors[0]


def run_scratch_diagnostic(
    api: WizardApi,
    storage: CorpusStorage,
    fixtures: WizardFixtures,
    diagnostic_user_id: str,
    other_user_id: str,
    run_id: str,
    recorder: OperationRecorder,
) -> None:
    scratch: list[tuple[str, str, str]] = []
    diagnostic_error: BaseException | None = None
    try:
        knowledge = api.create("knowledge", diagnostic_user_id)
        knowledge_id = _wizard_id(knowledge)
        scratch.append(("knowledge", diagnostic_user_id, knowledge_id))
        policy = api.create("policy", diagnostic_user_id)
        policy_id = _wizard_id(policy)
        scratch.append(("policy", diagnostic_user_id, policy_id))
        other = api.create("knowledge", other_user_id)
        other_id = _wizard_id(other)
        scratch.append(("knowledge", other_user_id, other_id))

        knowledge_list = api.list_wizards("knowledge", diagnostic_user_id)
        policy_list = api.list_wizards("policy", diagnostic_user_id)
        other_list = api.list_wizards("knowledge", other_user_id)
        _check(
            recorder,
            "scratch.list.membership",
            knowledge_id in {_wizard_id(item) for item in knowledge_list}
            and policy_id in {_wizard_id(item) for item in policy_list}
            and other_id in {_wizard_id(item) for item in other_list}
            and other_id not in {_wizard_id(item) for item in knowledge_list}
            and knowledge_id not in {_wizard_id(item) for item in other_list},
        )
        api.get("knowledge", diagnostic_user_id, knowledge_id)
        api.get("policy", diagnostic_user_id, policy_id)
        api.get("knowledge", other_user_id, other_id)

        api.expect_error(
            "scratch.isolation.knowledge_to_policy",
            "GET",
            f"/api/policy/wizards/{knowledge_id}",
            404,
            "NOT_FOUND",
            params={"user_id": diagnostic_user_id},
        )
        api.expect_error(
            "scratch.isolation.policy_to_knowledge",
            "GET",
            f"/api/knowledge/wizards/{policy_id}",
            404,
            "NOT_FOUND",
            params={"user_id": diagnostic_user_id},
        )
        api.expect_error(
            "scratch.isolation.cross_user",
            "GET",
            f"/api/knowledge/wizards/{knowledge_id}",
            404,
            "NOT_FOUND",
            params={"user_id": other_user_id},
        )
        api.expect_error(
            "scratch.error.invalid_user",
            "GET",
            "/api/knowledge/wizards",
            422,
            "VALIDATION_ERROR",
            params={"user_id": "invalid/user"},
        )
        missing_id = str(uuid4())
        api.expect_error(
            "scratch.error.missing_wizard",
            "GET",
            f"/api/knowledge/wizards/{missing_id}",
            404,
            "NOT_FOUND",
            params={"user_id": diagnostic_user_id},
        )

        suffix = fixtures.rules.supported_extensions[0]
        negative_cases = (
            (
                "unsupported",
                UploadPart("unsupported", b"content"),
                415,
                "UNSUPPORTED_FILE_TYPE",
                None,
            ),
            (
                "empty",
                UploadPart("empty" + suffix, b""),
                422,
                "EMPTY_UPLOAD",
                None,
            ),
            (
                "whitespace",
                UploadPart(
                    "whitespace" + suffix,
                    "  \n".encode(fixtures.rules.text_encoding),
                ),
                200,
                None,
                None,
            ),
            (
                "invalid_modified",
                UploadPart(
                    "invalid" + suffix,
                    "text".encode(fixtures.rules.text_encoding),
                ),
                422,
                "VALIDATION_ERROR",
                [1_000_000],
            ),
        )
        for name, part, status, code, modified in negative_cases:
            result = _upload_without_mutation(
                api,
                storage,
                "knowledge",
                diagnostic_user_id,
                knowledge_id,
                operation_name=f"scratch.upload.{name}",
                parts=[part],
                modified_paragraph_ids=modified,
                expected_status=status,
                expected_code=code,
            )
            if name == "whitespace":
                _check(
                    recorder,
                    "scratch.upload.whitespace.draft_retained",
                    isinstance(result.get("full_text"), str)
                    and str(result["full_text"]).endswith("  \n"),
                )

        uploaded = _upload_without_mutation(
            api,
            storage,
            "knowledge",
            diagnostic_user_id,
            knowledge_id,
            operation_name="scratch.knowledge.upload",
            paths=[fixtures.knowledge[0].path],
        )
        uploaded_text = _required_string(uploaded.get("full_text"), "full_text")
        _, modified = upload_continuation(uploaded)
        old_marker, new_marker = diagnostic_edit_markers(run_id)
        _check(
            recorder,
            "scratch.destructive.markers_unique",
            old_marker not in uploaded_text and new_marker not in uploaded_text,
        )
        initial_text = uploaded_text + "\n\n" + old_marker
        initial = api.submit_save(
            "knowledge",
            diagnostic_user_id,
            knowledge_id,
            initial_text,
            modified,
            operation_name="scratch.knowledge.save",
        )
        initial_task_id = _task_id(initial)
        api.expect_error(
            "scratch.isolation.task_owner",
            "GET",
            f"/api/tasks/{initial_task_id}",
            404,
            "NOT_FOUND",
            params={"user_id": other_user_id},
        )
        api.poll_task(
            diagnostic_user_id,
            initial_task_id,
            operation_name="scratch.knowledge.save.poll",
        )
        _, initial_checkpoint = _verify_save_postcondition(
            api,
            storage,
            "knowledge",
            diagnostic_user_id,
            knowledge_id,
            initial_task_id,
            "scratch.knowledge.save",
            initial_text,
        )
        initial_chunk_mapping = _checkpoint_paragraph_mapping(
            initial_checkpoint, require_complete=True
        )
        _check(
            recorder,
            "scratch.partial_resave.multi_paragraph_precondition",
            len(initial_chunk_mapping) >= 2
            and sum(bool(ids) for ids in initial_chunk_mapping.values()) >= 2,
            paragraph_count=len(initial_chunk_mapping),
        )
        initially_saved = api.get(
            "knowledge",
            diagnostic_user_id,
            knowledge_id,
            operation_name="scratch.destructive.old.get",
        )
        _check(
            recorder,
            "scratch.destructive.old.persisted",
            initially_saved.get("full_text") == initial_text
            and storage.marker_count(
                diagnostic_user_id, "knowledge", knowledge_id, old_marker
            )
            > 0,
            selected_ids={"wizard_id": knowledge_id},
        )
        edited_text = initial_text.replace(old_marker, new_marker)
        _check(
            recorder,
            "scratch.destructive.replacement_exact",
            initial_text.count(old_marker) == 1
            and edited_text.count(new_marker) == 1
            and old_marker not in edited_text,
        )
        destructive = api.submit_save(
            "knowledge",
            diagnostic_user_id,
            knowledge_id,
            edited_text,
            None,
            operation_name="scratch.destructive.save",
        )
        destructive_task_id = _task_id(destructive)
        api.poll_task(
            diagnostic_user_id,
            destructive_task_id,
            operation_name="scratch.destructive.save.poll",
        )
        destructive_trace, destructive_checkpoint = _verify_save_postcondition(
            api,
            storage,
            "knowledge",
            diagnostic_user_id,
            knowledge_id,
            destructive_task_id,
            "scratch.destructive.save",
            edited_text,
            proof_samples=("modified_paragraph_ids", "targeted_old_chunk_ids"),
        )
        final_chunk_mapping = _checkpoint_paragraph_mapping(
            destructive_checkpoint, require_complete=True
        )
        trace_samples = _mapping(
            destructive_trace.get("samples"), "destructive trace samples"
        )
        modified_sample = _mapping(
            trace_samples.get("modified_paragraph_ids"),
            "modified paragraph sample",
        )
        modified_items = modified_sample.get("items")
        if (
            not isinstance(modified_items, list)
            or len(modified_items) != 1
        ):
            raise WizardDiagnosticError(
                "Partial re-Save did not identify exactly one changed paragraph"
            )
        try:
            changed_paragraph_id = int(modified_items[0])
        except (TypeError, ValueError) as exc:
            raise WizardDiagnosticError(
                "Partial re-Save returned an invalid changed paragraph ID"
            ) from exc
        old_changed_ids = initial_chunk_mapping.get(changed_paragraph_id, [])
        if len(old_changed_ids) > TRACE_SAMPLE_LIMIT:
            raise WizardDiagnosticError(
                "Changed paragraph exceeds the bounded chunk-ID proof"
            )
        _, probe_checkpoint = api.trace_operation(
            "scratch.destructive.save",
            collection="knowledge",
            wizard_id=knowledge_id,
            probe_chunk_ids=old_changed_ids,
        )
        if probe_checkpoint is None:
            raise WizardDiagnosticError("Partial re-Save probe is missing")
        probe = _mapping(probe_checkpoint.get("probe"), "chunk ownership probe")
        physical_inventory = storage.inventory(
            diagnostic_user_id, "knowledge"
        )
        proof_counts = verify_partial_resave_mappings(
            initial_chunk_mapping,
            final_chunk_mapping,
            changed_paragraph_id,
            probed_old_owner_count=int(probe.get("owned_count", -1)),
            physical_chunk_ids=tuple(physical_inventory),
        )
        _check(
            recorder,
            "scratch.partial_resave.chunk_id_proof",
            True,
            **proof_counts,
            changed_paragraph_id=changed_paragraph_id,
            selected_ids={"wizard_id": knowledge_id},
        )
        destructively_saved = api.get(
            "knowledge",
            diagnostic_user_id,
            knowledge_id,
            operation_name="scratch.destructive.new.get",
        )
        _check(
            recorder,
            "scratch.destructive.mapping_postcondition",
            destructively_saved.get("full_text") == edited_text
            and new_marker in str(destructively_saved.get("full_text"))
            and old_marker not in str(destructively_saved.get("full_text")),
            selected_ids={"wizard_id": knowledge_id},
        )
        _check(
            recorder,
            "scratch.destructive.storage_postcondition",
            storage.marker_count(
                diagnostic_user_id, "knowledge", knowledge_id, new_marker
            )
            > 0
            and storage.marker_count(
                diagnostic_user_id, "knowledge", knowledge_id, old_marker
            )
            == 0,
            selected_ids={"wizard_id": knowledge_id},
        )
        _failed_save_without_mutation(
            api,
            storage,
            "knowledge",
            diagnostic_user_id,
            knowledge_id,
            edited_text,
        )

        fifo_a_text = edited_text + f"\n\nPhase 1C FIFO A {run_id}."
        fifo_b_text = fifo_a_text + f"\n\nPhase 1C FIFO B {run_id}."
        fifo_a = api.submit_save(
            "knowledge",
            diagnostic_user_id,
            knowledge_id,
            fifo_a_text,
            None,
            operation_name="scratch.fifo.save_a",
        )
        fifo_b = api.submit_save(
            "knowledge",
            diagnostic_user_id,
            knowledge_id,
            fifo_b_text,
            None,
            operation_name="scratch.fifo.save_b",
        )
        # The two saves above are intentionally submitted before either poll.
        final_a = api.poll_task(
            diagnostic_user_id,
            _task_id(fifo_a),
            operation_name="scratch.fifo.poll_a",
        )
        final_b = api.poll_task(
            diagnostic_user_id,
            _task_id(fifo_b),
            operation_name="scratch.fifo.poll_b",
        )
        fifo_a_trace, _ = api.trace_operation("scratch.fifo.save_a")
        _validate_successful_save_trace(fifo_a_trace, _task_id(fifo_a))
        recorder.record(
            "scratch.fifo.save_a.telemetry",
            "passed",
            trace=_trace_artifact(fifo_a_trace),
            selected_ids={"task_id": _task_id(fifo_a)},
        )
        _check(
            recorder,
            "scratch.fifo.timestamps",
            _parse_timestamp(final_b.get("started_at"), "started_at")
            >= _parse_timestamp(final_a.get("finished_at"), "finished_at"),
        )
        _check(
            recorder,
            "scratch.fifo.final_text",
            api.get("knowledge", diagnostic_user_id, knowledge_id).get("full_text")
            == fifo_b_text,
        )
        _verify_save_postcondition(
            api,
            storage,
            "knowledge",
            diagnostic_user_id,
            knowledge_id,
            _task_id(fifo_b),
            "scratch.fifo.save_b",
            fifo_b_text,
        )

        for collection, user_id, wizard_id, fixture in (
            ("policy", diagnostic_user_id, policy_id, fixtures.policy[0].path),
            ("knowledge", other_user_id, other_id, fixtures.knowledge[0].path),
        ):
            isolated_upload = _upload_without_mutation(
                api,
                storage,
                collection,
                user_id,
                wizard_id,
                operation_name=f"scratch.{collection}.{user_id}.upload",
                paths=[fixture],
            )
            task = api.submit_save(
                collection,
                user_id,
                wizard_id,
                _required_string(isolated_upload.get("full_text"), "full_text"),
                isolated_upload.get("modified_paragraph_ids"),
                operation_name=f"scratch.{collection}.{user_id}.save",
            )
            api.poll_task(
                user_id,
                _task_id(task),
                operation_name=f"scratch.{collection}.{user_id}.save.poll",
            )
            if user_id == diagnostic_user_id:
                _verify_save_postcondition(
                    api,
                    storage,
                    collection,
                    user_id,
                    wizard_id,
                    _task_id(task),
                    f"scratch.{collection}.{user_id}.save",
                    _required_string(isolated_upload.get("full_text"), "full_text"),
                )

        for collection, user_id, wizard_id in (
            ("knowledge", diagnostic_user_id, knowledge_id),
            ("policy", diagnostic_user_id, policy_id),
            ("knowledge", other_user_id, other_id),
        ):
            integrity = storage.inspect_document(
                user_id, collection, wizard_id, require_nonempty=True
            )
            recorder.record(
                "scratch.weaviate.persisted",
                "passed",
                collection=collection,
                chunk_count=integrity.chunk_count,
                lateon_dimension=integrity.lateon_dimension,
                gte_dimension=integrity.gte_dimension,
                selected_ids={"user_id": user_id, "wizard_id": wizard_id},
            )
        _check(
            recorder,
            "scratch.weaviate.knowledge_policy_isolation",
            not storage.snapshots(diagnostic_user_id, "policy", knowledge_id)
            and not storage.snapshots(
                diagnostic_user_id, "knowledge", policy_id
            ),
        )
        _check(
            recorder,
            "scratch.weaviate.cross_user_isolation",
            not storage.snapshots(other_user_id, "knowledge", knowledge_id)
            and not storage.snapshots(
                diagnostic_user_id, "knowledge", other_id
            ),
        )
    except BaseException as exc:
        diagnostic_error = exc
    try:
        _delete_scratch(api, storage, scratch, recorder)
    except BaseException as cleanup_error:
        if diagnostic_error is not None:
            diagnostic_error.add_note(
                "Scratch cleanup also failed: " + repr(cleanup_error)
            )
        else:
            raise
    if diagnostic_error is not None:
        raise diagnostic_error


def _record_state(path: Path, state: CorpusState) -> CorpusState:
    write_corpus_state(path, state)
    return state


def _cleanup_pending(
    state_path: Path,
    state: CorpusState,
    storage: CorpusStorage,
    recorder: OperationRecorder,
) -> CorpusState:
    # Fail closed on unknown physical data before selecting any deletion target.
    storage.validate_membership(state)
    if state.pending_replacement is not None:
        for document in state.pending_replacement.documents:
            deleted = storage.delete_document(document)
            recorder.record(
                "corpus.recovery.pending_replacement",
                "passed",
                collection=document.collection,
                deleted_chunks=deleted,
                selected_ids={"wizard_id": document.wizard_id},
            )
        state = _record_state(state_path, discard_pending_replacement(state))
    for document in tuple(state.pending_cleanup):
        deleted = storage.delete_document(document)
        recorder.record(
            "corpus.recovery.pending_cleanup",
            "passed",
            collection=document.collection,
            deleted_chunks=deleted,
            selected_ids={"wizard_id": document.wizard_id},
        )
        state = _record_state(
            state_path, remove_pending_cleanup(state, document)
        )
    storage.validate_membership(state)
    return state


def _verify_generation(
    generation: CorpusGeneration,
    diagnostic_user_id: str,
    storage: CorpusStorage,
    recorder: OperationRecorder,
    *,
    operation_name: str,
) -> None:
    for document in generation.documents:
        integrity = storage.inspect_document(
            diagnostic_user_id,
            document.collection,
            document.wizard_id,
            require_nonempty=True,
        )
        _check(
            recorder,
            operation_name,
            document.status == "saved"
            and integrity.chunk_ids == document.chunk_ids
            and integrity.fingerprint == document.storage_fingerprint,
            collection=document.collection,
            chunk_count=integrity.chunk_count,
            lateon_dimension=integrity.lateon_dimension,
            gte_dimension=integrity.gte_dimension,
            selected_ids={"wizard_id": document.wizard_id},
        )
    knowledge = generation.document("knowledge")
    policy = generation.document("policy")
    if knowledge is None or policy is None:
        raise WizardDiagnosticError("Corpus generation is incomplete")
    _check(
        recorder,
        operation_name + ".isolation",
        not storage.snapshots(
            diagnostic_user_id, "policy", knowledge.wizard_id
        )
        and not storage.snapshots(
            diagnostic_user_id, "knowledge", policy.wizard_id
        ),
    )


def _verify_replacement_texts(
    api: WizardApi,
    generation: CorpusGeneration,
    fixtures: WizardFixtures,
    diagnostic_user_id: str,
    recorder: OperationRecorder,
) -> None:
    for collection in COLLECTION_PREFIXES:
        document = generation.document(collection)
        if document is None:
            raise WizardDiagnosticError("Corpus replacement is incomplete")
        saved = api.get(
            collection,
            diagnostic_user_id,
            document.wizard_id,
            operation_name=f"corpus.{collection}.replacement.get",
        )
        _check(
            recorder,
            f"corpus.{collection}.replacement.saved_text",
            saved.get("full_text") == fixtures.combined_text(collection),
            selected_ids={"wizard_id": document.wizard_id},
        )


def _upload_fixture_batches(
    api: WizardApi,
    storage: CorpusStorage,
    fixtures: WizardFixtures,
    collection: str,
    diagnostic_user_id: str,
    wizard_id: str,
    recorder: OperationRecorder,
) -> tuple[str, list[int]]:
    previous: Mapping[str, Any] | None = None
    batches = fixtures.batches(collection)
    if not batches:
        raise WizardDiagnosticError("Fixture upload plan is empty")
    for index, batch in enumerate(batches, start=1):
        current_text, modified = upload_continuation(previous)
        operation_name = f"corpus.{collection}.upload.batch_{index:03d}"
        recorder.record(
            operation_name + ".plan",
            "passed",
            batch_index=index,
            batch_count=len(batches),
            file_count=len(batch.files),
            size_bytes=batch.size_bytes,
            selected_ids={"wizard_id": wizard_id},
        )
        previous = _upload_without_mutation(
            api,
            storage,
            collection,
            diagnostic_user_id,
            wizard_id,
            operation_name=operation_name,
            paths=[item.path for item in batch.files],
            current_text=current_text,
            modified_paragraph_ids=modified,
        )
    full_text, modified = upload_continuation(previous)
    if full_text is None or modified is None:  # pragma: no cover - batches are nonempty.
        raise WizardDiagnosticError("Fixture upload did not produce a final draft")
    return full_text, modified


def ensure_persistent_corpus(
    api: WizardApi,
    storage: CorpusStorage,
    fixtures: WizardFixtures,
    state_path: Path,
    state: CorpusState | None,
    diagnostic_user_id: str,
    run_id: str,
    reseed: bool,
    recorder: OperationRecorder,
) -> str:
    had_active = state is not None and state.active is not None
    storage.validate_membership(state)
    if state is None:
        state = _record_state(state_path, new_corpus_state(diagnostic_user_id))
    else:
        state = _cleanup_pending(state_path, state, storage, recorder)

    if state.active is not None and not reseed:
        if state.active.fixture_digest != fixtures.digest:
            raise WizardDiagnosticError(
                "Fixture corpus changed; rerun with --reseed-corpus"
            )
        _verify_generation(
            state.active,
            diagnostic_user_id,
            storage,
            recorder,
            operation_name="corpus.verify.active",
        )
        storage.validate_membership(state)
        if dict(state.active.ingestion_policy) != fixtures.rules.ingestion_policy():
            state = _record_state(
                state_path, record_active_ingestion_policy(state, fixtures)
            )
        return "reused"

    state = _record_state(state_path, begin_replacement(state, run_id, fixtures))
    for collection in COLLECTION_PREFIXES:
        created = api.create(
            collection,
            diagnostic_user_id,
            operation_name=f"corpus.{collection}.create",
        )
        document = CorpusDocument(collection, _wizard_id(created))
        state = _record_state(
            state_path, record_replacement_document(state, document)
        )
        uploaded_text, modified = _upload_fixture_batches(
            api,
            storage,
            fixtures,
            collection,
            diagnostic_user_id,
            document.wizard_id,
            recorder,
        )
        expected_text = fixtures.combined_text(collection)
        _check(
            recorder,
            f"corpus.{collection}.upload_text",
            uploaded_text == expected_text,
            selected_ids={"wizard_id": document.wizard_id},
        )
        submitting = replace(document, status="save_submitting")
        state = _record_state(
            state_path, record_replacement_document(state, submitting)
        )
        task = api.submit_save(
            collection,
            diagnostic_user_id,
            document.wizard_id,
            expected_text,
            modified,
            operation_name=f"corpus.{collection}.save",
        )
        submitted = replace(
            submitting, status="save_submitted", task_id=_task_id(task)
        )
        state = _record_state(
            state_path, record_replacement_document(state, submitted)
        )
        api.poll_task(
            diagnostic_user_id,
            submitted.task_id,
            operation_name=f"corpus.{collection}.save.poll",
        )
        _verify_save_postcondition(
            api,
            storage,
            collection,
            diagnostic_user_id,
            document.wizard_id,
            submitted.task_id,
            f"corpus.{collection}.save",
            expected_text,
        )
        integrity = storage.inspect_document(
            diagnostic_user_id,
            collection,
            document.wizard_id,
            require_nonempty=True,
        )
        complete = replace(
            submitted,
            status="saved",
            chunk_ids=integrity.chunk_ids,
            storage_fingerprint=integrity.fingerprint,
        )
        state = _record_state(
            state_path, record_replacement_document(state, complete)
        )

    replacement = state.pending_replacement
    if replacement is None:
        raise WizardDiagnosticError("Corpus replacement disappeared")
    # Verify the two-document replacement as one generation before promotion.
    _verify_replacement_texts(
        api, replacement, fixtures, diagnostic_user_id, recorder
    )
    _verify_generation(
        replacement,
        diagnostic_user_id,
        storage,
        recorder,
        operation_name="corpus.verify.replacement",
    )
    storage.validate_membership(state)
    state = _record_state(state_path, mark_replacement_verified(state))
    state = _record_state(state_path, promote_replacement(state))
    state = _cleanup_pending(state_path, state, storage, recorder)
    if state.active is None:
        raise WizardDiagnosticError(
            "Corpus promotion did not produce an active generation"
        )
    _verify_generation(
        state.active,
        diagnostic_user_id,
        storage,
        recorder,
        operation_name="corpus.verify.final",
    )
    storage.validate_membership(state)
    return "reseeded" if had_active else "created"


def run_wizard_phase_1c(
    runtime_url: str,
    runtime_headers: Mapping[str, str],
    config: Mapping[str, str],
    fixtures: WizardFixtures,
    state_path: Path,
    state: CorpusState | None,
    diagnostic_user_id: str,
    other_user_id: str,
    run_id: str,
    reseed: bool,
    recorder: OperationRecorder,
    progress: dict[str, str],
    *,
    bootstrap_isolation_user_collections: bool = False,
) -> str:
    api = WizardApi(
        runtime_url,
        runtime_headers,
        recorder,
        diagnostic_user_id,
        run_id,
    )
    result: str | None = None
    primary_error: BaseException | None = None
    try:
        api.start_trace()
        progress["security_status"] = "passed"
        with CorpusStorage(config, diagnostic_user_id) as storage:
            # Unknown persistent objects fail before scratch or any deletion.
            storage.validate_membership(state)
            if bootstrap_isolation_user_collections:
                storage.ensure_empty_user_collections(other_user_id)
            run_scratch_diagnostic(
                api,
                storage,
                fixtures,
                diagnostic_user_id,
                other_user_id,
                run_id,
                recorder,
            )
            progress["scratch_status"] = "passed"
            recorder.record("scratch.complete", "passed")
            try:
                action = ensure_persistent_corpus(
                    api,
                    storage,
                    fixtures,
                    state_path,
                    state,
                    diagnostic_user_id,
                    run_id,
                    reseed,
                    recorder,
                )
            except BaseException:
                progress["corpus_action"] = "failed"
                raise
            progress["corpus_action"] = action
            result = action
            progress["telemetry_status"] = "passed"
    except BaseException as exc:
        progress["telemetry_status"] = "failed"
        if progress["security_status"] != "passed":
            progress["security_status"] = "failed"
        if progress["scratch_status"] != "passed":
            progress["scratch_status"] = "failed"
        if isinstance(exc, (KeyboardInterrupt, SystemExit, WizardDiagnosticError)):
            primary_error = exc
        else:
            primary_error = WizardDiagnosticError(
                "Wizard diagnostic execution failed; inspect correlated service logs"
            )
            primary_error.__cause__ = exc
    finally:
        cleanup_error: BaseException | None = None
        if api.trace_started:
            try:
                api.delete_trace()
                progress["trace_deleted"] = True
            except BaseException as exc:
                cleanup_error = exc
                progress["telemetry_status"] = "failed"
        try:
            api.close()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
            else:
                cleanup_error.add_note("HTTP client close also failed: " + repr(exc))
        if cleanup_error is not None:
            if primary_error is not None:
                primary_error.add_note(
                    "Diagnostic trace cleanup also failed: " + repr(cleanup_error)
                )
            else:
                primary_error = cleanup_error
    if primary_error is not None:
        raise primary_error
    if result is None:  # pragma: no cover - every successful path assigns it.
        raise WizardDiagnosticError("Wizard diagnostic produced no corpus result")
    recorder.record(
        "telemetry.compensation",
        "passed",
        status=COMPENSATION_STATUS,
    )
    return result


# Compatibility for callers that imported the Phase 1B executor directly.
run_wizard_phase_1b = run_wizard_phase_1c


__all__ = [
    "StorageIntegrity",
    "UploadPart",
    "diagnostic_edit_markers",
    "marker_chunk_count",
    "run_wizard_phase_1b",
    "run_wizard_phase_1c",
    "save_request_payload",
    "storage_integrity",
    "upload_continuation",
    "upload_form_fields",
    "verify_partial_resave_mappings",
    "wizard_saved_fingerprint",
    "validate_trace_evidence",
    "validate_upload_zero_side_effects",
]
