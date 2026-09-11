"""Local-only evidence resolution and isolated evaluator process control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from threading import Event, Lock
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from backend.config import get_collection_name
from backend.wizard.diagnostics import framed_content_digest
from deployment.wizard_diagnostic_api import CorpusStorage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = PROJECT_ROOT / "evaluation" / ".venv" / "bin" / "rag-evaluate"
EVALUATION_RECORD_SCHEMA_VERSION = "1.0"
EVALUATION_RESULT_SCHEMA_VERSION = "1.0"
EVALUATION_TIMEOUT_SECONDS = 1800.0
LOCAL_EVALUATION_CONCURRENCY = 1
LOCAL_EVALUATION_LOCK_PATH = (
    PROJECT_ROOT / ".local" / "diagnostics" / "evaluation" / "execution.lock"
)
_LOCK_POLL_SECONDS = 0.05
_PROPERTIES = ("user_id", "document_id", "paragraph_id", "chunk_id", "raw_text")


class EvaluationBridgeError(RuntimeError):
    """Local evaluation evidence or process contract failed safely."""


@dataclass(frozen=True)
class ContextIdentity:
    collection: str
    ids: tuple[str, ...]
    fingerprints: tuple[str, ...]
    digest: str
    rendered_utf8_bytes: int


@dataclass(frozen=True)
class RequestEvidence:
    user_id: str
    trace_session_id: str
    operation_id: str
    chat_session_id: str
    request_id: str
    rewritten_query: str
    knowledge: ContextIdentity
    policy: ContextIdentity


@dataclass(frozen=True)
class ResolvedContexts:
    knowledge: tuple[str, ...]
    policy: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationObservation:
    status: str
    error_code: str | None
    duration_ms: float
    record_path: str | None
    result_path: str | None
    record_sha256: str | None
    queue_wait_ms: float = 0.0

    def artifact(self) -> dict[str, object]:
        return {
            "status": self.status,
            "error_code": self.error_code,
            "evaluation_ms": round(self.duration_ms, 3),
            "queue_wait_ms": round(self.queue_wait_ms, 3),
            "record_path": self.record_path,
            "result_path": self.result_path,
            "record_sha256": self.record_sha256,
        }


@dataclass
class EvaluationLaunch:
    started: float
    process: subprocess.Popen[bytes] | None
    record_path: Path | None
    result_path: Path | None
    record_sha256: str | None
    source: str
    request_id: str
    conversation_id: str
    error_code: str | None = None


@dataclass
class _EvaluationJobState:
    lock: Lock = field(default_factory=Lock)
    launch: EvaluationLaunch | None = None


@dataclass(frozen=True)
class EvaluationJob:
    future: Future[EvaluationObservation]
    cancel_event: Event
    state: _EvaluationJobState


_LOCAL_EVALUATION_EXECUTOR = ThreadPoolExecutor(
    max_workers=LOCAL_EVALUATION_CONCURRENCY,
    thread_name_prefix="rag-local-evaluation",
)
_LOCAL_JOB_LOCK = Lock()
_LOCAL_JOB_PENDING = False


def _canonical_uuid(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise EvaluationBridgeError(f"{name} is missing")
    try:
        parsed = str(UUID(value))
    except ValueError as exc:
        raise EvaluationBridgeError(f"{name} is malformed") from exc
    if parsed != value:
        raise EvaluationBridgeError(f"{name} is not canonical")
    return parsed


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluationBridgeError(f"{name} is malformed")
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationBridgeError(f"{name} is malformed")
    return value


def _context_identity(
    operation: Mapping[str, Any], collection: str
) -> ContextIdentity:
    counts = _mapping(operation.get("counts"), "evidence counts")
    flags = _mapping(operation.get("flags"), "evidence flags")
    digests = _mapping(operation.get("digests"), "evidence digests")
    samples = _mapping(operation.get("samples"), "evidence samples")
    prefix = f"qwen_{collection}"
    count = counts.get(f"{prefix}_used_count")
    rendered_bytes = counts.get(f"{prefix}_context_utf8_bytes")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or isinstance(rendered_bytes, bool)
        or not isinstance(rendered_bytes, int)
        or rendered_bytes < 0
        or flags.get(f"{prefix}_used_proof_truncated") is not False
    ):
        raise EvaluationBridgeError(f"{collection} evidence is incomplete")

    def complete_sample(name: str) -> tuple[str, ...]:
        sample = _mapping(samples.get(name), name)
        items = sample.get("items")
        if (
            sample.get("exact_count") != count
            or sample.get("truncated") is not False
            or not isinstance(items, list)
            or len(items) != count
            or any(not isinstance(item, str) or not item for item in items)
        ):
            raise EvaluationBridgeError(f"{name} is incomplete")
        return tuple(items)

    identifiers = complete_sample(f"{prefix}_used_ids")
    fingerprints = complete_sample(f"{prefix}_used_item_fingerprints")
    canonical_ids = tuple(_canonical_uuid(item, "context ID") for item in identifiers)
    if len(set(canonical_ids)) != len(canonical_ids):
        raise EvaluationBridgeError(f"{collection} context IDs are duplicated")
    for identifier, fingerprint in zip(canonical_ids, fingerprints, strict=True):
        expected_prefix = f"{identifier}:"
        if not fingerprint.startswith(expected_prefix):
            raise EvaluationBridgeError(f"{collection} fingerprint order is invalid")
        _sha256(fingerprint[len(expected_prefix) :], "context fingerprint")
    return ContextIdentity(
        collection=collection,
        ids=canonical_ids,
        fingerprints=fingerprints,
        digest=_sha256(
            digests.get(f"{prefix}_context_sha256"),
            f"{collection} context digest",
        ),
        rendered_utf8_bytes=rendered_bytes,
    )


def parse_request_evidence(
    operation: Mapping[str, Any],
    *,
    user_id: str,
    trace_session_id: str,
    operation_id: str,
    chat_session_id: str,
    request_id: str,
) -> RequestEvidence:
    """Extract the shared minimal evidence contract from deep or ask traces."""

    canonical_trace_session_id = _canonical_uuid(trace_session_id, "trace session ID")
    canonical_operation_id = _canonical_uuid(operation_id, "operation ID")
    canonical_chat_session_id = _canonical_uuid(chat_session_id, "chat session ID")
    canonical_request_id = _canonical_uuid(request_id, "request ID")
    if (
        operation.get("operation_id") != canonical_operation_id
        or operation.get("kind") != "chat_query"
        or operation.get("user_id") != user_id
        or operation.get("collection_type") != "conversations"
        or operation.get("wizard_id") != canonical_chat_session_id
        or operation.get("outcome") != "succeeded"
        or not isinstance(operation.get("finished_at"), str)
    ):
        raise EvaluationBridgeError("evaluation evidence correlation is invalid")
    flags = _mapping(operation.get("flags"), "evidence flags")
    texts = _mapping(operation.get("texts"), "evidence texts")
    digests = _mapping(operation.get("digests"), "evidence digests")
    rewritten = texts.get("granite_rewritten_query")
    if (
        not isinstance(rewritten, str)
        or not rewritten.strip()
        or flags.get("granite_rewritten_query_truncated") is not False
    ):
        raise EvaluationBridgeError("official rewritten query evidence is incomplete")
    expected_rewrite_digest = framed_content_digest(
        "chat-granite-rewritten-query-v1", (rewritten.encode("utf-8"),)
    )
    if digests.get("granite_rewritten_query_sha256") != expected_rewrite_digest:
        raise EvaluationBridgeError("official rewritten query digest does not match")
    knowledge = _context_identity(operation, "knowledge")
    policy = _context_identity(operation, "policy")
    if set(knowledge.ids).intersection(policy.ids):
        raise EvaluationBridgeError("Knowledge and Policy context IDs overlap")
    return RequestEvidence(
        user_id=user_id,
        trace_session_id=canonical_trace_session_id,
        operation_id=canonical_operation_id,
        chat_session_id=canonical_chat_session_id,
        request_id=canonical_request_id,
        rewritten_query=rewritten,
        knowledge=knowledge,
        policy=policy,
    )


def _fetch_contexts(
    storage: CorpusStorage,
    *,
    user_id: str,
    identity: ContextIdentity,
    allowed_document_ids: frozenset[str] | None,
) -> tuple[str, ...]:
    if not identity.ids:
        expected = framed_content_digest(
            f"chat-qwen-{identity.collection}-context-v1", ()
        )
        if identity.digest != expected or identity.rendered_utf8_bytes != 0:
            raise EvaluationBridgeError(
                f"empty {identity.collection} evidence does not match"
            )
        return ()
    collection_type = (
        "knowledge_facts" if identity.collection == "knowledge" else "policy"
    )
    collection_name = get_collection_name(user_id, collection_type)
    if not storage.manager.client.collections.exists(collection_name):
        raise EvaluationBridgeError(
            f"{identity.collection} collection does not exist"
        )
    physical = storage.manager.client.collections.use(collection_name)
    response = physical.query.fetch_objects_by_ids(
        list(identity.ids),
        limit=len(identity.ids),
        include_vector=False,
        return_properties=list(_PROPERTIES),
    )
    objects = getattr(response, "objects", None)
    if not isinstance(objects, list):
        raise EvaluationBridgeError("context hydration response is malformed")
    hydrated: dict[str, str] = {}
    for item in objects:
        properties = getattr(item, "properties", None)
        if not isinstance(properties, Mapping):
            raise EvaluationBridgeError("context properties are malformed")
        object_id = _canonical_uuid(str(getattr(item, "uuid", None)), "object ID")
        chunk_id = _canonical_uuid(properties.get("chunk_id"), "chunk ID")
        document_id = _canonical_uuid(properties.get("document_id"), "document ID")
        paragraph_id = properties.get("paragraph_id")
        raw_text = properties.get("raw_text")
        if (
            object_id != chunk_id
            or chunk_id not in identity.ids
            or chunk_id in hydrated
            or properties.get("user_id") != user_id
            or isinstance(paragraph_id, bool)
            or not isinstance(paragraph_id, int)
            or paragraph_id <= 0
            or not isinstance(raw_text, str)
            or not raw_text.strip()
        ):
            raise EvaluationBridgeError("hydrated context violates storage identity")
        if allowed_document_ids is not None and document_id not in allowed_document_ids:
            raise EvaluationBridgeError("hydrated context is outside the active corpus")
        hydrated[chunk_id] = raw_text
    if set(hydrated) != set(identity.ids) or len(objects) != len(identity.ids):
        raise EvaluationBridgeError("not every exact context ID was hydrated once")
    ordered = tuple(hydrated[identifier] for identifier in identity.ids)
    digest_parts: list[bytes] = []
    rendered_bytes = max(0, len(ordered) - 1) * 2
    for identifier, raw_text, expected_fingerprint in zip(
        identity.ids, ordered, identity.fingerprints, strict=True
    ):
        identifier_bytes = identifier.encode("utf-8")
        text_bytes = raw_text.encode("utf-8")
        fingerprint = framed_content_digest(
            "chat-kp-item-v1", (identifier_bytes, text_bytes)
        )
        if expected_fingerprint != f"{identifier}:{fingerprint}":
            raise EvaluationBridgeError("hydrated context fingerprint does not match")
        digest_parts.extend((identifier_bytes, text_bytes))
        rendered_bytes += len(text_bytes)
    aggregate = framed_content_digest(
        f"chat-qwen-{identity.collection}-context-v1", digest_parts
    )
    if aggregate != identity.digest or rendered_bytes != identity.rendered_utf8_bytes:
        raise EvaluationBridgeError("hydrated ordered context digest does not match")
    return ordered


def resolve_exact_contexts(
    config: Mapping[str, str],
    *,
    user_id: str,
    evidence: RequestEvidence,
    allowed_document_ids: Mapping[str, frozenset[str]] | None = None,
) -> ResolvedContexts:
    """Hydrate exact traced chunks locally without requesting vectors."""

    with CorpusStorage(config, user_id) as storage:
        knowledge = _fetch_contexts(
            storage,
            user_id=user_id,
            identity=evidence.knowledge,
            allowed_document_ids=(
                None
                if allowed_document_ids is None
                else allowed_document_ids.get("knowledge", frozenset())
            ),
        )
        policy = _fetch_contexts(
            storage,
            user_id=user_id,
            identity=evidence.policy,
            allowed_document_ids=(
                None
                if allowed_document_ids is None
                else allowed_document_ids.get("policy", frozenset())
            ),
        )
    return ResolvedContexts(knowledge, policy)


def build_evaluation_record(
    *,
    source: str,
    request_id: str,
    conversation_id: str,
    original_query: str,
    response: str,
    telemetry: Mapping[str, object],
    evidence: RequestEvidence,
    contexts: ResolvedContexts,
    captured_at: str,
) -> dict[str, object]:
    if source not in {"e2e", "rag_ask"}:
        raise EvaluationBridgeError("evaluation source is invalid")
    if _canonical_uuid(request_id, "request ID") != evidence.request_id:
        raise EvaluationBridgeError("evaluation request correlation is invalid")
    _canonical_uuid(conversation_id, "conversation ID")
    for value, name in ((original_query, "query"), (response, "response")):
        if not isinstance(value, str) or not value.strip():
            raise EvaluationBridgeError(f"evaluation {name} is invalid")
    try:
        parsed = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise EvaluationBridgeError("captured_at is invalid") from exc
    if parsed.tzinfo is None:
        raise EvaluationBridgeError("captured_at must be timezone-aware")
    record = {
        "schema_version": EVALUATION_RECORD_SCHEMA_VERSION,
        "source": source,
        "request_id": request_id,
        "conversation_id": conversation_id,
        "original_query": original_query,
        "rewritten_query": evidence.rewritten_query,
        "response": response,
        "knowledge_contexts": list(contexts.knowledge),
        "knowledge_context_ids": list(evidence.knowledge.ids),
        "policy_contexts": list(contexts.policy),
        "policy_context_ids": list(evidence.policy.ids),
        "retrieved_contexts": list(contexts.knowledge + contexts.policy),
        "context_roles": ["knowledge"] * len(contexts.knowledge)
        + ["policy"] * len(contexts.policy),
        "telemetry": dict(telemetry),
        "reference": None,
        "reference_context_ids": [],
        "captured_at": parsed.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    try:
        json.dumps(record, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise EvaluationBridgeError("evaluation record is not finite JSON") from exc
    return record


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def write_private_json(path: Path, value: Mapping[str, object]) -> str:
    """Atomically create a private JSON artifact without overwriting a peer."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    payload = _canonical_bytes(value)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def start_local_evaluation(
    record: Mapping[str, object],
    directory: Path,
    stem: str,
    *,
    exact_names: bool = False,
    execution_lock_fd: int | None = None,
) -> EvaluationLaunch:
    started = perf_counter()
    record_path = directory / (
        "record.json" if exact_names else f"{stem}.record.json"
    )
    result_path = directory / (
        "result.json" if exact_names else f"{stem}.result.json"
    )
    source = str(record.get("source", ""))
    request_id = str(record.get("request_id", ""))
    conversation_id = str(record.get("conversation_id", ""))
    try:
        record_sha256 = write_private_json(record_path, record)
        if not EVALUATOR.is_file() or not os.access(EVALUATOR, os.X_OK):
            raise EvaluationBridgeError(
                "dedicated evaluator environment is unavailable"
            )
        environment = os.environ.copy()
        environment.update(
            {
                "RAGAS_DO_NOT_TRACK": "true",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            }
        )
        process = subprocess.Popen(
            [str(EVALUATOR), "run", str(record_path), "--output", str(result_path)],
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(
                () if execution_lock_fd is None else (execution_lock_fd,)
            ),
        )
        return EvaluationLaunch(
            started,
            process,
            record_path,
            result_path,
            record_sha256,
            source,
            request_id,
            conversation_id,
        )
    except (OSError, ValueError, EvaluationBridgeError):
        return EvaluationLaunch(
            started,
            None,
            record_path if record_path.is_file() else None,
            result_path,
            locals().get("record_sha256"),
            source,
            request_id,
            conversation_id,
            "EVALUATION_START_FAILED",
        )


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def finish_local_evaluation(
    launch: EvaluationLaunch,
    *,
    timeout_seconds: float = EVALUATION_TIMEOUT_SECONDS,
) -> EvaluationObservation:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("evaluation timeout must be finite and positive")
    if launch.process is None:
        return EvaluationObservation(
            "failed",
            launch.error_code or "EVALUATION_START_FAILED",
            (perf_counter() - launch.started) * 1000.0,
            None if launch.record_path is None else str(launch.record_path),
            None,
            launch.record_sha256,
        )
    process = launch.process
    remaining = max(0.0, timeout_seconds - (perf_counter() - launch.started))
    try:
        process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        _stop_process(process)
        return EvaluationObservation(
            "failed",
            "EVALUATION_TIMEOUT",
            (perf_counter() - launch.started) * 1000.0,
            str(launch.record_path),
            None,
            launch.record_sha256,
        )
    try:
        if launch.result_path is None or not launch.result_path.is_file():
            raise EvaluationBridgeError("evaluator result is missing")
        payload = json.loads(launch.result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise EvaluationBridgeError("evaluator result is malformed")
        status = payload.get("status")
        if (
            payload.get("schema_version") != EVALUATION_RESULT_SCHEMA_VERSION
            or status not in {"succeeded", "partial", "failed"}
            or payload.get("record_sha256") != launch.record_sha256
            or payload.get("source") != launch.source
            or payload.get("request_id") != launch.request_id
            or payload.get("conversation_id") != launch.conversation_id
        ):
            raise EvaluationBridgeError("evaluator result correlation is invalid")
        error_code = None if status == "succeeded" else "EVALUATION_INCOMPLETE"
        return EvaluationObservation(
            str(status),
            error_code,
            (perf_counter() - launch.started) * 1000.0,
            str(launch.record_path),
            str(launch.result_path),
            launch.record_sha256,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, EvaluationBridgeError):
        return EvaluationObservation(
            "failed",
            "EVALUATION_RESULT_INVALID",
            (perf_counter() - launch.started) * 1000.0,
            str(launch.record_path),
            None,
            launch.record_sha256,
        )


def cancel_local_evaluation(launch: EvaluationLaunch) -> None:
    if launch.process is not None:
        try:
            _stop_process(launch.process)
        except OSError:
            pass


def _acquire_evaluation_lock(
    path: Path,
    cancel_event: Event,
    submitted_at: float,
) -> tuple[int | None, float]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise EvaluationBridgeError("local evaluation lock path is unsafe")
    os.chmod(path.parent, 0o700)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise EvaluationBridgeError("local evaluation lock is not a regular file")
        os.fchmod(descriptor, 0o600)
        while True:
            if cancel_event.is_set():
                os.close(descriptor)
                return None, (perf_counter() - submitted_at) * 1000.0
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return descriptor, (perf_counter() - submitted_at) * 1000.0
            except BlockingIOError:
                cancel_event.wait(_LOCK_POLL_SECONDS)
    except BaseException:
        os.close(descriptor)
        raise


def _release_evaluation_lock(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _terminal_job(
    status: str,
    error_code: str,
    *,
    duration_ms: float = 0.0,
    queue_wait_ms: float = 0.0,
) -> EvaluationJob:
    future: Future[EvaluationObservation] = Future()
    future.set_result(
        EvaluationObservation(
            status,
            error_code,
            duration_ms,
            None,
            None,
            None,
            queue_wait_ms,
        )
    )
    return EvaluationJob(future, Event(), _EvaluationJobState())


def _run_evaluation_job(
    *,
    submitted_at: float,
    cancel_event: Event,
    state: _EvaluationJobState,
    execution_lock_path: Path,
    config: Mapping[str, str],
    user_id: str,
    evidence: RequestEvidence,
    allowed_document_ids: Mapping[str, frozenset[str]] | None,
    source: str,
    request_id: str,
    conversation_id: str,
    original_query: str,
    response: str,
    telemetry: Mapping[str, object],
    captured_at: str,
    directory: Path,
    stem: str,
    exact_names: bool,
) -> EvaluationObservation:
    descriptor: int | None = None
    queue_wait_ms = 0.0
    launch: EvaluationLaunch | None = None
    try:
        descriptor, queue_wait_ms = _acquire_evaluation_lock(
            execution_lock_path,
            cancel_event,
            submitted_at,
        )
        if descriptor is None or cancel_event.is_set():
            return EvaluationObservation(
                "failed",
                "EVALUATION_CANCELLED",
                (perf_counter() - submitted_at) * 1000.0,
                None,
                None,
                None,
                queue_wait_ms,
            )
        contexts = resolve_exact_contexts(
            config,
            user_id=user_id,
            evidence=evidence,
            allowed_document_ids=allowed_document_ids,
        )
        if cancel_event.is_set():
            return EvaluationObservation(
                "failed",
                "EVALUATION_CANCELLED",
                (perf_counter() - submitted_at) * 1000.0,
                None,
                None,
                None,
                queue_wait_ms,
            )
        record = build_evaluation_record(
            source=source,
            request_id=request_id,
            conversation_id=conversation_id,
            original_query=original_query,
            response=response,
            telemetry=telemetry,
            evidence=evidence,
            contexts=contexts,
            captured_at=captured_at,
        )
        if cancel_event.is_set():
            return EvaluationObservation(
                "failed",
                "EVALUATION_CANCELLED",
                (perf_counter() - submitted_at) * 1000.0,
                None,
                None,
                None,
                queue_wait_ms,
            )
        launch = start_local_evaluation(
            record,
            directory,
            stem,
            exact_names=exact_names,
            execution_lock_fd=descriptor,
        )
        with state.lock:
            state.launch = launch
        if cancel_event.is_set():
            cancel_local_evaluation(launch)
        observation = finish_local_evaluation(launch)
        if cancel_event.is_set():
            return EvaluationObservation(
                "failed",
                "EVALUATION_CANCELLED",
                (perf_counter() - submitted_at) * 1000.0,
                observation.record_path,
                None,
                observation.record_sha256,
                queue_wait_ms,
            )
        return EvaluationObservation(
            observation.status,
            observation.error_code,
            (perf_counter() - submitted_at) * 1000.0,
            observation.record_path,
            observation.result_path,
            observation.record_sha256,
            queue_wait_ms,
        )
    except Exception:
        return EvaluationObservation(
            "failed",
            "EVALUATION_EVIDENCE_INVALID",
            (perf_counter() - submitted_at) * 1000.0,
            (
                None
                if launch is None or launch.record_path is None
                else str(launch.record_path)
            ),
            None,
            None if launch is None else launch.record_sha256,
            queue_wait_ms,
        )
    finally:
        with state.lock:
            state.launch = None
        _release_evaluation_lock(descriptor)
        global _LOCAL_JOB_PENDING
        with _LOCAL_JOB_LOCK:
            _LOCAL_JOB_PENDING = False


def submit_local_evaluation(
    *,
    config: Mapping[str, str],
    user_id: str,
    evidence: RequestEvidence,
    allowed_document_ids: Mapping[str, frozenset[str]] | None,
    source: str,
    request_id: str,
    conversation_id: str,
    original_query: str,
    response: str,
    telemetry: Mapping[str, object],
    captured_at: str,
    directory: Path,
    stem: str,
    exact_names: bool = False,
    execution_lock_path: Path = LOCAL_EVALUATION_LOCK_PATH,
) -> EvaluationJob:
    """Submit one bounded local evidence-hydration and evaluator job."""

    try:
        telemetry_copy = json.loads(
            json.dumps(telemetry, ensure_ascii=False, allow_nan=False)
        )
        if not isinstance(telemetry_copy, Mapping):
            raise TypeError("telemetry must be a mapping")
        config_copy = dict(config)
        allowed_copy = (
            None
            if allowed_document_ids is None
            else {
                collection: frozenset(document_ids)
                for collection, document_ids in allowed_document_ids.items()
            }
        )
    except (TypeError, ValueError):
        return _terminal_job("failed", "EVALUATION_EVIDENCE_INVALID")

    global _LOCAL_JOB_PENDING
    with _LOCAL_JOB_LOCK:
        if _LOCAL_JOB_PENDING:
            return _terminal_job("failed", "EVALUATION_LOCAL_JOB_BUSY")
        _LOCAL_JOB_PENDING = True
    submitted_at = perf_counter()
    cancel_event = Event()
    state = _EvaluationJobState()
    try:
        future = _LOCAL_EVALUATION_EXECUTOR.submit(
            _run_evaluation_job,
            submitted_at=submitted_at,
            cancel_event=cancel_event,
            state=state,
            execution_lock_path=execution_lock_path,
            config=config_copy,
            user_id=user_id,
            evidence=evidence,
            allowed_document_ids=allowed_copy,
            source=source,
            request_id=request_id,
            conversation_id=conversation_id,
            original_query=original_query,
            response=response,
            telemetry=telemetry_copy,
            captured_at=captured_at,
            directory=directory,
            stem=stem,
            exact_names=exact_names,
        )
    except Exception:
        with _LOCAL_JOB_LOCK:
            _LOCAL_JOB_PENDING = False
        return _terminal_job("failed", "EVALUATION_START_FAILED")
    return EvaluationJob(future, cancel_event, state)


def finish_evaluation_job(job: EvaluationJob) -> EvaluationObservation:
    return job.future.result()


def cancel_evaluation_job(job: EvaluationJob) -> None:
    job.cancel_event.set()
    with job.state.lock:
        launch = job.state.launch
    if launch is not None:
        cancel_local_evaluation(launch)


__all__ = [
    "ContextIdentity",
    "EvaluationBridgeError",
    "EvaluationJob",
    "EvaluationLaunch",
    "EvaluationObservation",
    "LOCAL_EVALUATION_CONCURRENCY",
    "LOCAL_EVALUATION_LOCK_PATH",
    "RequestEvidence",
    "ResolvedContexts",
    "build_evaluation_record",
    "cancel_evaluation_job",
    "cancel_local_evaluation",
    "finish_local_evaluation",
    "finish_evaluation_job",
    "parse_request_evidence",
    "resolve_exact_contexts",
    "start_local_evaluation",
    "submit_local_evaluation",
    "write_private_json",
]
