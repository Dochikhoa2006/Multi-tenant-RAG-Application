"""Default-off, bounded observation for the existing Wizard pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import math
from threading import RLock
from time import perf_counter
from typing import Any
from uuid import UUID


TRACE_SCHEMA_VERSION = "1.0"
TRACE_SESSION_MAX_OPERATIONS = 256
TRACE_SAMPLE_LIMIT = 32
TRACE_SESSION_HEADER = "X-Wizard-Diagnostic-Session-ID"
TRACE_OPERATION_HEADER = "X-Wizard-Diagnostic-Operation-ID"

OPERATION_KINDS = frozenset({"upload", "save", "delete"})
STAGE_NAMES = frozenset(
    {
        "api.lookup_validation",
        "collection.preparation",
        "enqueue",
        "upload.multipart_copy",
        "upload.file_decode_read",
        "upload.draft_merge",
        "upload.server_total",
        "save.diff",
        "save.recovery_snapshot",
        "save.old_chunk_delete",
        "save.semantic_split",
        "save.chunking",
        "save.lateon",
        "save.gte",
        "save.renumbering",
        "save.weaviate_insert",
        "save.weaviate_metadata_update",
        "save.weaviate_metadata_verify",
        "save.paragraph_map_commit",
        "save.document_map_commit",
        "delete.recovery_snapshot",
        "delete.storage_delete",
        "delete.paragraph_map_remove",
        "delete.document_map_remove",
        "weaviate.delete_mutation",
        "weaviate.delete_verification",
        "compensation.weaviate_metadata_restore",
        "compensation.weaviate_metadata_verify",
        "compensation.inserted_chunk_delete",
        "compensation.chunk_restore",
        "compensation.paragraph_map_restore",
        "compensation.document_map_restore",
    }
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _uuid_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a UUID string")
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid UUID") from exc


def _required_text(value: object, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty without surrounding whitespace")
    if len(value) > maximum:
        raise ValueError(f"{name} is too long")
    return value


def _framed_hash(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(b"wizard-diagnostic-mapping-v1")
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _mapping_digest(mapping: Mapping[int, Sequence[str]]) -> str:
    parts: list[bytes] = []
    for paragraph_id, chunk_ids in sorted(mapping.items()):
        parts.append(str(paragraph_id).encode("ascii"))
        for chunk_id in chunk_ids:
            parts.append(str(chunk_id).encode("ascii"))
        parts.append(b";")
    return _framed_hash(parts)


def _mapping_membership_digest(mapping: Mapping[int, Sequence[str]]) -> str:
    """Hash paragraph ownership without relying on process-local chunk order."""

    parts: list[bytes] = []
    for paragraph_id, chunk_ids in sorted(mapping.items()):
        for chunk_id in sorted(chunk_ids):
            parts.append(str(paragraph_id).encode("ascii"))
            parts.append(str(chunk_id).encode("ascii"))
    return _framed_hash(parts)


@dataclass(frozen=True)
class TraceHandle:
    session_id: str
    operation_id: str


@dataclass
class _Stage:
    call_count: int = 0
    total_ms: float = 0.0
    min_ms: float | None = None
    max_ms: float | None = None
    failure_count: int = 0

    def observe(self, elapsed_ms: float, failed: bool) -> None:
        value = max(0.0, float(elapsed_ms))
        if not math.isfinite(value):
            raise ValueError("stage duration must be finite")
        self.call_count += 1
        self.total_ms += value
        self.min_ms = value if self.min_ms is None else min(self.min_ms, value)
        self.max_ms = value if self.max_ms is None else max(self.max_ms, value)
        if failed:
            self.failure_count += 1

    def payload(self) -> dict[str, object]:
        return {
            "call_count": self.call_count,
            "total_ms": round(self.total_ms, 3),
            "min_ms": round(self.min_ms or 0.0, 3),
            "max_ms": round(self.max_ms or 0.0, 3),
            "failure_count": self.failure_count,
        }


@dataclass
class _Sample:
    exact_count: int
    items: tuple[str, ...]
    truncated: bool

    def payload(self) -> dict[str, object]:
        return {
            "exact_count": self.exact_count,
            "items": list(self.items),
            "truncated": self.truncated,
        }


@dataclass
class _Operation:
    operation_id: str
    kind: str
    user_id: str
    collection_type: str
    wizard_id: str
    started_at: str = field(default_factory=_utc_timestamp)
    finished_at: str | None = None
    outcome: str = "running"
    task_id: str | None = None
    stages: dict[str, _Stage] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)
    samples: dict[str, _Sample] = field(default_factory=dict)

    def payload(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "kind": self.kind,
            "user_id": self.user_id,
            "collection_type": self.collection_type,
            "wizard_id": self.wizard_id,
            "task_id": self.task_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outcome": self.outcome,
            "stages": {
                name: stage.payload() for name, stage in sorted(self.stages.items())
            },
            "counts": dict(sorted(self.counts.items())),
            "flags": dict(sorted(self.flags.items())),
            "digests": dict(sorted(self.digests.items())),
            "samples": {
                name: sample.payload() for name, sample in sorted(self.samples.items())
            },
        }


@dataclass
class _Session:
    session_id: str
    run_id: str
    user_id: str
    started_at: str = field(default_factory=_utc_timestamp)
    operations: dict[str, _Operation] = field(default_factory=dict)
    overflowed: bool = False
    trace_faulted: bool = False


class DiagnosticTraceRegistry:
    """Own at most one bounded diagnostic trace session."""

    def __init__(self, configured_user_id: str) -> None:
        self.configured_user_id = _required_text(
            configured_user_id, "configured_user_id"
        )
        self._session: _Session | None = None
        self._lock = RLock()

    def start(self, user_id: str, session_id: str, run_id: str) -> dict[str, object]:
        user = _required_text(user_id, "user_id")
        session = _uuid_text(session_id, "session_id")
        run = _required_text(run_id, "run_id")
        if user != self.configured_user_id:
            raise KeyError("diagnostic session")
        with self._lock:
            if self._session is not None:
                raise RuntimeError("a Wizard diagnostic session is already active")
            self._session = _Session(session, run, user)
            return self._session_payload(self._session, operation_id=None)

    def delete(self, user_id: str, session_id: str) -> None:
        with self._lock:
            session = self._required_session(user_id, session_id)
            if self._session is session:
                self._session = None

    def begin_operation(
        self,
        *,
        user_id: str,
        session_id: str | None,
        operation_id: str | None,
        kind: str,
        collection_type: str,
        wizard_id: str,
    ) -> TraceHandle | None:
        if user_id != self.configured_user_id or kind not in OPERATION_KINDS:
            return None
        try:
            session_key = _uuid_text(session_id, "session_id")
            operation_key = _uuid_text(operation_id, "operation_id")
            wizard_key = _uuid_text(wizard_id, "wizard_id")
        except (TypeError, ValueError):
            self.mark_fault()
            return None
        with self._lock:
            session = self._session
            if session is None or session.session_id != session_key:
                return None
            if operation_key in session.operations:
                session.trace_faulted = True
                return None
            if len(session.operations) >= TRACE_SESSION_MAX_OPERATIONS:
                session.overflowed = True
                return None
            session.operations[operation_key] = _Operation(
                operation_id=operation_key,
                kind=kind,
                user_id=user_id,
                collection_type=collection_type,
                wizard_id=wizard_key,
            )
            return TraceHandle(session_key, operation_key)

    def observe_stage(
        self, handle: TraceHandle, name: str, elapsed_ms: float, failed: bool
    ) -> None:
        if name not in STAGE_NAMES:
            raise ValueError(f"unknown Wizard diagnostic stage {name!r}")
        with self._lock:
            operation = self._operation(handle)
            operation.stages.setdefault(name, _Stage()).observe(elapsed_ms, failed)

    def add_count(self, handle: TraceHandle, name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("diagnostic count must be a non-negative integer")
        key = _required_text(name, "count name", maximum=96)
        with self._lock:
            operation = self._operation(handle)
            operation.counts[key] = operation.counts.get(key, 0) + value

    def set_flag(self, handle: TraceHandle, name: str, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("diagnostic flag must be boolean")
        key = _required_text(name, "flag name", maximum=96)
        with self._lock:
            self._operation(handle).flags[key] = value

    def set_digest(self, handle: TraceHandle, name: str, value: str) -> None:
        key = _required_text(name, "digest name", maximum=96)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("diagnostic digest must be a SHA-256 hex string")
        int(value, 16)
        with self._lock:
            self._operation(handle).digests[key] = value.lower()

    def set_sample(
        self,
        handle: TraceHandle,
        name: str,
        items: Iterable[object],
        *,
        exact_count: int | None = None,
    ) -> None:
        key = _required_text(name, "sample name", maximum=96)
        supplied_count = 0
        supplied_sample: list[str] = []
        for item in items:
            supplied_count += 1
            if len(supplied_sample) < TRACE_SAMPLE_LIMIT:
                rendered = str(item)
                if (
                    not rendered
                    or len(rendered) > 128
                    or any(character.isspace() for character in rendered)
                ):
                    raise ValueError(
                        "diagnostic sample items must be short sanitized tokens"
                    )
                supplied_sample.append(rendered)
        count = supplied_count if exact_count is None else exact_count
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < supplied_count
        ):
            raise ValueError("sample exact_count is invalid")
        with self._lock:
            operation = self._operation(handle)
            previous = operation.samples.get(key)
            previous_items = previous.items if previous is not None else ()
            previous_count = previous.exact_count if previous is not None else 0
            combined_items = (
                previous_items + tuple(supplied_sample)
            )[:TRACE_SAMPLE_LIMIT]
            combined_count = previous_count + count
            operation.samples[key] = _Sample(
                exact_count=combined_count,
                items=combined_items,
                truncated=combined_count > len(combined_items),
            )

    def attach_task(self, handle: TraceHandle, task_id: str) -> None:
        task = _uuid_text(task_id, "task_id")
        with self._lock:
            self._operation(handle).task_id = task

    def finish(self, handle: TraceHandle, outcome: str) -> None:
        if outcome not in {"succeeded", "failed"}:
            raise ValueError("diagnostic operation outcome is invalid")
        with self._lock:
            operation = self._operation(handle)
            operation.outcome = outcome
            operation.finished_at = _utc_timestamp()

    def mark_fault(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.trace_faulted = True

    def snapshot(
        self, user_id: str, session_id: str, operation_id: str | None = None
    ) -> dict[str, object]:
        with self._lock:
            session = self._required_session(user_id, session_id)
            return self._session_payload(session, operation_id)

    def _required_session(self, user_id: str, session_id: str) -> _Session:
        user = _required_text(user_id, "user_id")
        session_key = _uuid_text(session_id, "session_id")
        session = self._session
        if (
            session is None
            or user != self.configured_user_id
            or session.user_id != user
            or session.session_id != session_key
        ):
            raise KeyError("diagnostic session")
        return session

    def _operation(self, handle: TraceHandle) -> _Operation:
        session = self._session
        if session is None or session.session_id != handle.session_id:
            raise KeyError("diagnostic session")
        return session.operations[handle.operation_id]

    @staticmethod
    def _session_payload(
        session: _Session, operation_id: str | None
    ) -> dict[str, object]:
        operations = session.operations
        if operation_id is not None:
            operation_key = _uuid_text(operation_id, "operation_id")
            if operation_key not in operations:
                raise KeyError("diagnostic operation")
            selected = [operations[operation_key].payload()]
        else:
            selected = [item.payload() for item in operations.values()]
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "session_id": session.session_id,
            "run_id": session.run_id,
            "user_id": session.user_id,
            "started_at": session.started_at,
            "operation_count": len(session.operations),
            "overflowed": session.overflowed,
            "trace_faulted": session.trace_faulted,
            "missing_evidence": False,
            "operations": selected,
        }


_REGISTRY: DiagnosticTraceRegistry | None = None
_ACTIVE_HANDLE: ContextVar[TraceHandle | None] = ContextVar(
    "wizard_diagnostic_operation", default=None
)


def install_registry(registry: DiagnosticTraceRegistry) -> None:
    if not isinstance(registry, DiagnosticTraceRegistry):
        raise TypeError("registry must be a DiagnosticTraceRegistry")
    global _REGISTRY
    if _REGISTRY is not None and _REGISTRY is not registry:
        raise RuntimeError("a Wizard diagnostic registry is already installed")
    _REGISTRY = registry


def uninstall_registry(registry: DiagnosticTraceRegistry) -> None:
    global _REGISTRY
    if _REGISTRY is registry:
        _REGISTRY = None


def installed_registry() -> DiagnosticTraceRegistry | None:
    return _REGISTRY


def begin_request_operation(
    *,
    user_id: str,
    session_id: str | None,
    operation_id: str | None,
    kind: str,
    collection_type: str,
    wizard_id: str,
) -> TraceHandle | None:
    registry = _REGISTRY
    if registry is None:
        return None
    try:
        return registry.begin_operation(
            user_id=user_id,
            session_id=session_id,
            operation_id=operation_id,
            kind=kind,
            collection_type=collection_type,
            wizard_id=wizard_id,
        )
    except Exception:
        registry.mark_fault()
        return None


@contextmanager
def activate_operation(handle: TraceHandle | None) -> Iterator[None]:
    if handle is None:
        yield
        return
    token: Token[TraceHandle | None] = _ACTIVE_HANDLE.set(handle)
    try:
        yield
    finally:
        _ACTIVE_HANDLE.reset(token)


@contextmanager
def observe_stage(name: str, *, handle: TraceHandle | None = None) -> Iterator[None]:
    active = handle if handle is not None else _ACTIVE_HANDLE.get()
    registry = _REGISTRY
    if active is None or registry is None:
        yield
        return
    started = perf_counter()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        try:
            registry.observe_stage(
                active, name, (perf_counter() - started) * 1000.0, failed
            )
        except Exception:
            registry.mark_fault()


def _safe_observe(callback: Any, *args: object, **kwargs: object) -> None:
    registry = _REGISTRY
    handle = _ACTIVE_HANDLE.get()
    if registry is None or handle is None:
        return
    try:
        callback(handle, *args, **kwargs)
    except Exception:
        registry.mark_fault()


def add_count(name: str, value: int) -> None:
    registry = _REGISTRY
    if registry is not None:
        _safe_observe(registry.add_count, name, value)


def set_flag(name: str, value: bool) -> None:
    registry = _REGISTRY
    if registry is not None:
        _safe_observe(registry.set_flag, name, value)


def set_digest(name: str, value: str) -> None:
    registry = _REGISTRY
    if registry is not None:
        _safe_observe(registry.set_digest, name, value)


def set_mapping_digest(name: str, mapping: Mapping[int, Sequence[str]]) -> None:
    """Compute a sanitized mapping digest only for an active trace."""

    registry = _REGISTRY
    handle = _ACTIVE_HANDLE.get()
    if registry is None or handle is None:
        return
    try:
        registry.set_digest(handle, name, _mapping_digest(mapping))
    except Exception:
        registry.mark_fault()


def set_sample(
    name: str, items: Iterable[object], *, exact_count: int | None = None
) -> None:
    registry = _REGISTRY
    if registry is not None:
        _safe_observe(registry.set_sample, name, items, exact_count=exact_count)


def attach_task(handle: TraceHandle | None, task_id: str) -> None:
    registry = _REGISTRY
    if registry is None or handle is None:
        return
    try:
        registry.attach_task(handle, task_id)
    except Exception:
        registry.mark_fault()


def finish_operation(handle: TraceHandle | None, outcome: str) -> None:
    registry = _REGISTRY
    if registry is None or handle is None:
        return
    try:
        registry.finish(handle, outcome)
    except Exception:
        registry.mark_fault()


@contextmanager
def operation_lifecycle(handle: TraceHandle | None) -> Iterator[None]:
    """Finish one synchronous traced operation without affecting its errors."""

    try:
        yield
    except BaseException:
        finish_operation(handle, "failed")
        raise
    else:
        finish_operation(handle, "succeeded")


@contextmanager
def fail_operation_on_error(handle: TraceHandle | None) -> Iterator[None]:
    """Mark an asynchronous submission failed only when submission raises."""

    try:
        yield
    except BaseException:
        finish_operation(handle, "failed")
        raise


def mapping_checkpoint(
    runtime: object,
    user_id: str,
    collection_type: str,
    wizard_id: str,
    probe_chunk_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Return a sanitized, read-only view of both process-local Wizard maps."""

    if len(probe_chunk_ids) > TRACE_SAMPLE_LIMIT:
        raise ValueError("at most 32 chunk IDs may be probed")
    document_map = runtime.document_map(user_id, collection_type)
    paragraph_map = runtime.paragraph_map(user_id, collection_type)
    try:
        paragraph_data = document_map.get_paragraph_data(wizard_id)
    except KeyError:
        paragraph_data = None
    chunk_mapping = paragraph_map.get_document_chunks(wizard_id)
    flattened = [
        chunk_id
        for _, chunk_ids in sorted(chunk_mapping.items())
        for chunk_id in chunk_ids
    ]
    sampled_mapping: list[dict[str, object]] = []
    sample_budget = TRACE_SAMPLE_LIMIT
    sampled_chunk_count = 0
    for paragraph_id, chunk_ids in sorted(chunk_mapping.items()):
        if sample_budget <= 0:
            break
        sample_budget -= 1  # The paragraph identifier is one sampled item.
        selected = list(chunk_ids[:sample_budget])
        sampled_mapping.append(
            {"paragraph_id": paragraph_id, "chunk_ids": selected}
        )
        sampled_chunk_count += len(selected)
        sample_budget -= len(selected)

    probe_ids = [_uuid_text(item, "probe_chunk_id") for item in probe_chunk_ids]
    owners: dict[str, dict[str, object]] = {}
    if probe_ids:
        # This read-only diagnostic view intentionally includes any orphaned
        # ParagraphMap owner that is no longer reachable from DocumentMap.
        raw_owners = getattr(paragraph_map, "_chunk_owners", None)
        if not isinstance(raw_owners, Mapping):
            raise TypeError("ParagraphMap ownership index is unavailable")
        for chunk_id in probe_ids:
            owner = raw_owners.get(chunk_id)
            if owner is None:
                continue
            if not isinstance(owner, tuple) or len(owner) != 2:
                raise TypeError("ParagraphMap ownership index is malformed")
            paragraph_id = owner[1]
            if (
                isinstance(paragraph_id, bool)
                or not isinstance(paragraph_id, int)
                or paragraph_id <= 0
            ):
                raise TypeError("ParagraphMap ownership index is malformed")
            owners[chunk_id] = {
                "document_id": _uuid_text(owner[0], "owner document_id"),
                "paragraph_id": paragraph_id,
            }

    if paragraph_data is None:
        document_payload: dict[str, object] = {
            "present": False,
            "paragraph_count": 0,
            "paragraph_ids": [],
            "paragraph_ids_truncated": False,
            "full_text_sha256": None,
        }
    else:
        paragraph_ids = list(paragraph_data)
        full_text = "".join(paragraph_data.values())
        document_payload = {
            "present": True,
            "paragraph_count": len(paragraph_ids),
            "paragraph_ids": paragraph_ids[:TRACE_SAMPLE_LIMIT],
            "paragraph_ids_truncated": len(paragraph_ids) > TRACE_SAMPLE_LIMIT,
            "full_text_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        }
    return {
        "document": document_payload,
        "paragraph_map": {
            "paragraph_count": len(chunk_mapping),
            "chunk_count": len(flattened),
            "mapping_sha256": _mapping_digest(chunk_mapping),
            "membership_sha256": _mapping_membership_digest(chunk_mapping),
            "sample": sampled_mapping,
            "sampled_chunk_count": sampled_chunk_count,
            "truncated": (
                len(flattened) + len(chunk_mapping) > TRACE_SAMPLE_LIMIT
            ),
        },
        "probe": {
            "requested_count": len(probe_ids),
            "owned_count": len(owners),
            "owners": owners,
        },
    }


def mapping_digest(mapping: Mapping[int, Sequence[str]]) -> str:
    """Public sanitized digest helper used by the diagnostic runner tests."""

    return _mapping_digest(mapping)


def mapping_membership_digest(mapping: Mapping[int, Sequence[str]]) -> str:
    """Public order-independent paragraph/chunk ownership digest."""

    return _mapping_membership_digest(mapping)


__all__ = [
    "DiagnosticTraceRegistry",
    "TRACE_OPERATION_HEADER",
    "TRACE_SAMPLE_LIMIT",
    "TRACE_SCHEMA_VERSION",
    "TRACE_SESSION_HEADER",
    "TraceHandle",
    "activate_operation",
    "add_count",
    "attach_task",
    "begin_request_operation",
    "finish_operation",
    "fail_operation_on_error",
    "install_registry",
    "installed_registry",
    "mapping_checkpoint",
    "mapping_digest",
    "mapping_membership_digest",
    "observe_stage",
    "operation_lifecycle",
    "set_digest",
    "set_flag",
    "set_mapping_digest",
    "set_sample",
    "uninstall_registry",
]
