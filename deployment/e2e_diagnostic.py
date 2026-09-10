"""Local preflight and artifact contracts for the Phase 2 E2E diagnostic."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from deployment.wizard_diagnostic import (
    CORPUS_SCHEMA_VERSION,
    CorpusGeneration,
    CorpusState,
    WIZARD_FIXTURE_COLLECTIONS,
)


E2E_SCHEMA_VERSION = "1.3"
E2E_PHASE = "2D"


class E2EDiagnosticError(ValueError):
    """A safe, user-facing Phase 2 diagnostic failure."""


@dataclass(frozen=True)
class SelectedQuery:
    source_index: int
    question: str


@dataclass(frozen=True)
class QuerySelection:
    source_path: Path
    source_sha256: str
    total_queries: int
    start: int
    limit: int | None
    selected: tuple[SelectedQuery, ...]


@dataclass(frozen=True)
class E2ERun:
    run_id: str
    directory: Path
    requests_path: Path
    summary_path: Path


def utc_timestamp(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    return current.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _resolve_query_path(path: Path, project_root: Path) -> Path:
    candidate = path if path.is_absolute() else project_root / path
    if candidate.is_symlink():
        raise E2EDiagnosticError(f"Query file must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise E2EDiagnosticError(f"Query file does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise E2EDiagnosticError(f"Query path is not a file: {resolved}")
    if resolved.suffix.lower() != ".py":
        raise E2EDiagnosticError("Query file must use the .py extension")
    return resolved


def _literal_queries(source: str, path: Path) -> list[str]:
    try:
        tree = ast.parse(source, filename=os.fspath(path))
    except SyntaxError as exc:
        raise E2EDiagnosticError(f"Query file is not valid Python: {path}") from exc

    assignment: ast.expr | None = None
    for index, statement in enumerate(tree.body):
        if (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign):
            if (
                len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "QUERIES"
            ):
                value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "QUERIES"
            and statement.value is not None
        ):
            value = statement.value
        if value is None:
            raise E2EDiagnosticError(
                "Query file may contain only a module docstring and one "
                "QUERIES assignment"
            )
        if assignment is not None:
            raise E2EDiagnosticError("Query file must assign QUERIES exactly once")
        assignment = value

    if assignment is None:
        raise E2EDiagnosticError("Query file must define QUERIES")
    try:
        value = ast.literal_eval(assignment)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise E2EDiagnosticError("QUERIES must be a literal list of strings") from exc
    if not isinstance(value, list) or not value:
        raise E2EDiagnosticError("QUERIES must be a non-empty list")
    if any(not isinstance(item, str) for item in value):
        raise E2EDiagnosticError("Every QUERIES item must be a string")
    if any(not item.strip() for item in value):
        raise E2EDiagnosticError("QUERIES must not contain blank strings")
    return value


def load_query_selection(
    path: Path,
    project_root: Path,
    *,
    start: int = 0,
    limit: int | None = None,
) -> QuerySelection:
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise E2EDiagnosticError("--start must be a non-negative integer")
    if (
        limit is not None
        and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0)
    ):
        raise E2EDiagnosticError("--limit must be a positive integer")
    resolved = _resolve_query_path(path, project_root)
    try:
        source_bytes = resolved.read_bytes()
        source = source_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise E2EDiagnosticError(
            f"Query file is not readable UTF-8: {resolved}"
        ) from exc
    queries = _literal_queries(source, resolved)
    if start >= len(queries):
        raise E2EDiagnosticError("--start is outside the QUERIES list")
    stop = len(queries) if limit is None else min(len(queries), start + limit)
    selected = tuple(
        SelectedQuery(index, queries[index]) for index in range(start, stop)
    )
    if not selected:  # Defensive: validated bounds should make this unreachable.
        raise E2EDiagnosticError("Query selection must not be empty")
    return QuerySelection(
        source_path=resolved,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        total_queries=len(queries),
        start=start,
        limit=limit,
        selected=selected,
    )


def validate_reusable_corpus_state(
    state: CorpusState | None,
    diagnostic_user_id: str,
) -> CorpusGeneration:
    if state is None or state.active is None:
        raise E2EDiagnosticError(
            "Phase 1 corpus state has no active generation; "
            "run the Wizard diagnostic first"
        )
    if state.diagnostic_user_id != diagnostic_user_id:
        raise E2EDiagnosticError(
            "Phase 1 corpus belongs to a different diagnostic user"
        )
    if state.pending_replacement is not None:
        raise E2EDiagnosticError("Phase 1 corpus has an interrupted replacement")
    if state.pending_cleanup:
        raise E2EDiagnosticError("Phase 1 corpus has pending document cleanup")
    active = state.active
    if active.verified_at is None or active.completed_at is None:
        raise E2EDiagnosticError("Phase 1 active corpus is not verified and complete")
    if {item.collection for item in active.documents} != set(
        WIZARD_FIXTURE_COLLECTIONS
    ) or any(item.status != "saved" for item in active.documents):
        raise E2EDiagnosticError("Phase 1 active corpus is incomplete")
    return active


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as target:
            target.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _corpus_summary(active: CorpusGeneration) -> dict[str, object]:
    return {
        "state_schema_version": CORPUS_SCHEMA_VERSION,
        "generation_id": active.generation_id,
        "fixture_digest": active.fixture_digest,
        "documents": {
            document.collection: {
                "wizard_id": document.wizard_id,
                "chunk_count": len(document.chunk_ids),
                "storage_fingerprint": document.storage_fingerprint,
            }
            for document in active.documents
        },
    }


def create_e2e_run(
    output_root: Path,
    diagnostic_user_id: str,
    selection: QuerySelection,
    active: CorpusGeneration,
    *,
    continuous: bool,
) -> E2ERun:
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    directory = output_root / run_id
    try:
        if output_root.is_symlink():
            raise E2EDiagnosticError("E2E diagnostic output root must not be a symlink")
        directory.mkdir(parents=True, exist_ok=False)
        requests_path = directory / "requests.jsonl"
        requests_path.touch(exist_ok=False)
        summary_path = directory / "summary.json"
        _write_json_atomic(
            summary_path,
            {
                "schema_version": E2E_SCHEMA_VERSION,
                "diagnostic": "e2e",
                "phase": E2E_PHASE,
                "run_id": run_id,
                "status": "running",
                "diagnostic_user_id": diagnostic_user_id,
                "queries": {
                    "path": os.fspath(selection.source_path),
                    "sha256": selection.source_sha256,
                    "total": selection.total_queries,
                    "start": selection.start,
                    "limit": selection.limit,
                    "selected": len(selection.selected),
                },
                "session_mode": "continuous" if continuous else "fresh",
                "corpus": _corpus_summary(active),
                "corpus_validation": {"state": "succeeded", "physical": "pending"},
                "deep_trace": {"status": "pending", "session_deleted": False},
                "requests": {
                    "selected": len(selection.selected),
                    "attempted": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "query_posts": 0,
                    "individual_failed": 0,
                    "systemic_failed": 0,
                    "not_attempted": len(selection.selected),
                },
                "lifecycle": {
                    "pre_down": "pending",
                    "up": "pending",
                    "down": "pending",
                },
                "failure_stage": None,
                "started_at": utc_timestamp(started_at),
                "finished_at": None,
            },
        )
    except OSError as exc:
        raise E2EDiagnosticError(
            f"Could not create E2E diagnostic output in {directory}"
        ) from exc
    return E2ERun(run_id, directory, requests_path, summary_path)


class RequestRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.attempted = 0
        self.succeeded = 0
        self.failed = 0
        self.query_posts = 0
        self.individual_failed = 0
        self.systemic_failed = 0

    @property
    def totals(self) -> dict[str, int]:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "query_posts": self.query_posts,
            "individual_failed": self.individual_failed,
            "systemic_failed": self.systemic_failed,
        }

    def record(
        self,
        *,
        source_index: int,
        status: str,
        question: str,
        answer: str,
        answer_complete: bool,
        session_id: str | None,
        request_id: str | None,
        conversation_id: str | None,
        http_status: int | None,
        token_event_count: int,
        telemetry_schema_version: str | None,
        timings_ms: Mapping[str, object] | None,
        duration_ms: float,
        started_at: str,
        failure_stage: str | None = None,
        failure_code: str | None = None,
        deep_trace: Mapping[str, object] | None = None,
        diagnostic_operation_id: str | None = None,
        failure_scope: str | None = None,
        query_http_attempted: bool = True,
        client_timings_ms: Mapping[str, object] | None = None,
        trace_polling: Mapping[str, object] | None = None,
        post_generation: Mapping[str, object] | None = None,
        registry_verification: Mapping[str, object] | None = None,
    ) -> None:
        if status not in {"succeeded", "failed"}:
            raise E2EDiagnosticError("Request status must be succeeded or failed")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
        ):
            raise E2EDiagnosticError("Request source index is invalid")
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(answer, str)
        ):
            raise E2EDiagnosticError("Request text fields are invalid")
        if not isinstance(answer_complete, bool):
            raise E2EDiagnosticError("answer_complete must be boolean")
        if not isinstance(query_http_attempted, bool):
            raise E2EDiagnosticError("query_http_attempted must be boolean")
        if failure_scope not in {None, "individual", "systemic"}:
            raise E2EDiagnosticError("failure_scope is invalid")
        if status == "succeeded" and failure_scope is not None:
            raise E2EDiagnosticError("Successful requests cannot have a failure scope")
        if status == "failed" and failure_scope is None:
            raise E2EDiagnosticError("Failed requests require a failure scope")
        for name, value in (
            ("session_id", session_id),
            ("request_id", request_id),
            ("conversation_id", conversation_id),
            ("diagnostic_operation_id", diagnostic_operation_id),
        ):
            if value is not None:
                try:
                    UUID(value)
                except (TypeError, ValueError) as exc:
                    raise E2EDiagnosticError(f"{name} must be a UUID") from exc
        if (
            isinstance(token_event_count, bool)
            or not isinstance(token_event_count, int)
            or token_event_count < 0
            or not isinstance(duration_ms, (int, float))
            or isinstance(duration_ms, bool)
            or not math.isfinite(duration_ms)
            or duration_ms < 0
        ):
            raise E2EDiagnosticError("Request counts or duration are invalid")
        self.attempted += 1
        if query_http_attempted:
            self.query_posts += 1
        if status == "succeeded":
            self.succeeded += 1
        else:
            self.failed += 1
            if failure_scope == "individual":
                self.individual_failed += 1
            else:
                self.systemic_failed += 1
        payload = {
            "schema_version": E2E_SCHEMA_VERSION,
            "sequence": self.attempted,
            "source_index": source_index,
            "status": status,
            "question": question,
            "answer": answer,
            "answer_complete": answer_complete,
            "session_id": session_id,
            "request_id": request_id,
            "conversation_id": conversation_id,
            "http_status": http_status,
            "token_event_count": token_event_count,
            "telemetry": (
                None
                if timings_ms is None
                else {
                    "schema_version": telemetry_schema_version,
                    "timings_ms": dict(timings_ms),
                }
            ),
            "duration_ms": round(float(duration_ms), 3),
            "started_at": started_at,
            "finished_at": utc_timestamp(),
            "failure_stage": failure_stage,
            "failure_code": failure_code,
            "failure_scope": failure_scope,
            "query_http_attempted": query_http_attempted,
            "client_timings_ms": (
                None if client_timings_ms is None else dict(client_timings_ms)
            ),
            "trace_polling": (
                None if trace_polling is None else dict(trace_polling)
            ),
            "post_generation": (
                None if post_generation is None else dict(post_generation)
            ),
            "registry_verification": (
                None
                if registry_verification is None
                else dict(registry_verification)
            ),
            "diagnostic_operation_id": diagnostic_operation_id,
            "deep_trace": None if deep_trace is None else dict(deep_trace),
        }
        try:
            with self.path.open("a", encoding="utf-8") as target:
                target.write(json.dumps(payload, sort_keys=True) + "\n")
                target.flush()
                os.fsync(target.fileno())
        except OSError as exc:
            raise E2EDiagnosticError(
                f"Could not append E2E request artifact: {self.path}"
            ) from exc


def update_e2e_summary(
    run: E2ERun,
    recorder: RequestRecorder,
    *,
    status: str,
    pre_down_status: str,
    up_status: str,
    down_status: str,
    physical_corpus_status: str,
    trace_status: str = "not_started",
    trace_deleted: bool = False,
    failure_stage: str | None = None,
    finished: bool = False,
) -> None:
    try:
        payload = json.loads(run.summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("summary root must be an object")
        prior_requests = payload.get("requests")
        if not isinstance(prior_requests, Mapping):
            raise ValueError("summary requests field must be an object")
        selected = prior_requests.get("selected")
        payload.update(
            {
                "status": status,
                "corpus_validation": {
                    "state": "succeeded",
                    "physical": physical_corpus_status,
                },
                "requests": {
                    "selected": selected,
                    **recorder.totals,
                    "not_attempted": max(0, int(selected) - recorder.attempted),
                },
                "deep_trace": {
                    "status": trace_status,
                    "session_deleted": trace_deleted,
                },
                "lifecycle": {
                    "pre_down": pre_down_status,
                    "up": up_status,
                    "down": down_status,
                },
                "failure_stage": failure_stage,
                "finished_at": utc_timestamp() if finished else None,
            }
        )
        _write_json_atomic(run.summary_path, payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise E2EDiagnosticError(
            f"Could not update E2E diagnostic summary: {run.summary_path}"
        ) from exc


__all__ = [
    "E2EDiagnosticError",
    "E2E_PHASE",
    "E2E_SCHEMA_VERSION",
    "E2ERun",
    "QuerySelection",
    "RequestRecorder",
    "SelectedQuery",
    "create_e2e_run",
    "load_query_selection",
    "update_e2e_summary",
    "utc_timestamp",
    "validate_reusable_corpus_state",
]
