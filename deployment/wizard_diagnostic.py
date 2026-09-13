"""Local preflight, artifacts, and durable state for wizard diagnostics."""

from __future__ import annotations

import codecs
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from backend.config import get_collection_name


DIAGNOSTIC_USER_KEY = "RAG_DIAGNOSTIC_USER_ID"
DIAGNOSTIC_SCHEMA_VERSION = "1.3"
CORPUS_SCHEMA_VERSION = "1.1"
STORAGE_FINGERPRINT_VERSION = "wizard-storage-v1"
WIZARD_DIAGNOSTIC_PHASE = "1C"
WIZARD_FIXTURE_COLLECTIONS = ("knowledge", "policy")
DOCUMENT_STATES = frozenset(
    {"created", "save_submitting", "save_submitted", "saved"}
)

DEFAULT_SUPPORTED_FILE_EXTENSIONS = ".txt,.md,.csv,.json,.xml,.log"
DEFAULT_TEXT_FILE_ENCODING = "utf-8-sig"
DEFAULT_TEXT_FILE_JOIN_SEPARATOR = "\n\n"
DEFAULT_UPLOAD_MAX_FILE_BYTES = 10_485_760
DEFAULT_UPLOAD_MAX_TOTAL_BYTES = 26_214_400
DEFAULT_UPLOAD_READ_CHUNK_BYTES = 65_536

_DOCUMENT_KEYS = frozenset(
    {
        "collection",
        "wizard_id",
        "status",
        "task_id",
        "chunk_ids",
        "storage_fingerprint",
    }
)
_GENERATION_KEYS = frozenset(
    {
        "generation_id",
        "fixture_digest",
        "fixtures",
        "text_encoding",
        "text_join_separator",
        "ingestion_policy",
        "documents",
        "created_at",
        "verified_at",
        "completed_at",
    }
)
_STATE_KEYS = frozenset(
    {
        "schema_version",
        "diagnostic",
        "diagnostic_user_id",
        "active",
        "pending_replacement",
        "pending_cleanup",
        "updated_at",
    }
)
_FIXTURE_KEYS = frozenset({"path", "size_bytes", "sha256"})
_INGESTION_POLICY_KEYS = frozenset(
    {
        "supported_extensions",
        "upload_max_file_bytes",
        "upload_max_total_bytes",
        "upload_read_chunk_bytes",
    }
)


class WizardDiagnosticError(ValueError):
    """A safe, user-facing wizard diagnostic failure."""


def _config_string(
    config: Mapping[str, str], name: str, default: str, *, strip: bool = True
) -> str:
    raw = config.get(name, default)
    if not isinstance(raw, str):
        raise WizardDiagnosticError(f"{name} must be a string")
    value = raw.strip() if strip else raw
    if strip and not value:
        raise WizardDiagnosticError(f"{name} must not be empty")
    return value


def _config_positive_int(
    config: Mapping[str, str], name: str, default: int
) -> int:
    raw = config.get(name)
    try:
        value = default if raw is None else int(raw)
    except (TypeError, ValueError) as exc:
        raise WizardDiagnosticError(f"{name} must be an integer") from exc
    if value <= 0:
        raise WizardDiagnosticError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class FixtureRules:
    """Effective production ingestion settings used by preflight and Modal."""

    supported_extensions: tuple[str, ...]
    text_encoding: str
    text_join_separator: str
    upload_max_file_bytes: int
    upload_max_total_bytes: int
    upload_read_chunk_bytes: int

    @classmethod
    def from_config(cls, config: Mapping[str, str]) -> FixtureRules:
        raw_extensions = _config_string(
            config,
            "SUPPORTED_FILE_EXTENSIONS",
            DEFAULT_SUPPORTED_FILE_EXTENSIONS,
        )
        extensions: set[str] = set()
        for item in raw_extensions.split(","):
            extension = item.strip().lower()
            if not extension:
                continue
            if not extension.startswith("."):
                extension = "." + extension
            extensions.add(extension)
        if not extensions:
            raise WizardDiagnosticError(
                "SUPPORTED_FILE_EXTENSIONS must contain at least one extension"
            )
        requested_encoding = _config_string(
            config, "TEXT_FILE_ENCODING", DEFAULT_TEXT_FILE_ENCODING
        )
        try:
            encoding = codecs.lookup(requested_encoding).name
        except LookupError as exc:
            raise WizardDiagnosticError(
                "TEXT_FILE_ENCODING must name a supported codec"
            ) from exc
        max_file = _config_positive_int(
            config, "UPLOAD_MAX_FILE_BYTES", DEFAULT_UPLOAD_MAX_FILE_BYTES
        )
        max_total = _config_positive_int(
            config, "UPLOAD_MAX_TOTAL_BYTES", DEFAULT_UPLOAD_MAX_TOTAL_BYTES
        )
        read_chunk = _config_positive_int(
            config, "UPLOAD_READ_CHUNK_BYTES", DEFAULT_UPLOAD_READ_CHUNK_BYTES
        )
        if max_file > max_total:
            raise WizardDiagnosticError(
                "UPLOAD_MAX_FILE_BYTES must not exceed UPLOAD_MAX_TOTAL_BYTES"
            )
        if read_chunk > max_file:
            raise WizardDiagnosticError(
                "UPLOAD_READ_CHUNK_BYTES must not exceed UPLOAD_MAX_FILE_BYTES"
            )
        separator = _config_string(
            config,
            "TEXT_FILE_JOIN_SEPARATOR",
            DEFAULT_TEXT_FILE_JOIN_SEPARATOR,
            strip=False,
        )
        return cls(
            tuple(sorted(extensions)),
            encoding,
            separator,
            max_file,
            max_total,
            read_chunk,
        )

    def ingestion_policy(self) -> dict[str, object]:
        return {
            "supported_extensions": list(self.supported_extensions),
            "upload_max_file_bytes": self.upload_max_file_bytes,
            "upload_max_total_bytes": self.upload_max_total_bytes,
            "upload_read_chunk_bytes": self.upload_read_chunk_bytes,
        }

    def runtime_environment(self) -> dict[str, str]:
        return {
            "SUPPORTED_FILE_EXTENSIONS": ",".join(self.supported_extensions),
            "TEXT_FILE_ENCODING": self.text_encoding,
            "TEXT_FILE_JOIN_SEPARATOR": self.text_join_separator,
            "UPLOAD_MAX_FILE_BYTES": str(self.upload_max_file_bytes),
            "UPLOAD_MAX_TOTAL_BYTES": str(self.upload_max_total_bytes),
            "UPLOAD_READ_CHUNK_BYTES": str(self.upload_read_chunk_bytes),
        }


@dataclass(frozen=True)
class FixtureFile:
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class FixtureBatch:
    files: tuple[FixtureFile, ...]
    size_bytes: int


@dataclass(frozen=True)
class WizardFixtures:
    root: Path
    knowledge: tuple[FixtureFile, ...]
    policy: tuple[FixtureFile, ...]
    rules: FixtureRules

    def files(self, collection: str) -> tuple[FixtureFile, ...]:
        if collection not in WIZARD_FIXTURE_COLLECTIONS:
            raise WizardDiagnosticError(f"Unknown fixture collection: {collection}")
        return getattr(self, collection)

    def inventory(self) -> dict[str, list[dict[str, object]]]:
        return {
            collection: [
                {
                    "path": fixture.path.relative_to(self.root).as_posix(),
                    "size_bytes": fixture.size_bytes,
                    "sha256": fixture.sha256,
                }
                for fixture in self.files(collection)
            ]
            for collection in WIZARD_FIXTURE_COLLECTIONS
        }

    def batches(self, collection: str) -> tuple[FixtureBatch, ...]:
        """Group sorted fixtures into deterministic upload-sized batches."""

        batches: list[FixtureBatch] = []
        current: list[FixtureFile] = []
        current_bytes = 0
        for fixture in self.files(collection):
            if current and (
                current_bytes + fixture.size_bytes
                > self.rules.upload_max_total_bytes
            ):
                batches.append(FixtureBatch(tuple(current), current_bytes))
                current = []
                current_bytes = 0
            current.append(fixture)
            current_bytes += fixture.size_bytes
        if current:
            batches.append(FixtureBatch(tuple(current), current_bytes))
        return tuple(batches)

    @property
    def digest(self) -> str:
        return _fixture_digest(
            self.inventory(), self.rules.text_encoding, self.rules.text_join_separator
        )

    def combined_text(self, collection: str) -> str:
        try:
            contents = [
                item.path.read_text(encoding=self.rules.text_encoding)
                for item in self.files(collection)
            ]
        except (OSError, UnicodeError, LookupError) as exc:
            raise WizardDiagnosticError(
                f"Could not decode {collection} fixtures with configured encoding"
            ) from exc
        return self.rules.text_join_separator.join(contents)


@dataclass(frozen=True)
class DiagnosticRun:
    run_id: str
    directory: Path
    operations_path: Path
    summary_path: Path

    @property
    def ingestion_report_path(self) -> Path:
        return self.directory / "ingestion-report.json"


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class CorpusDocument:
    collection: str
    wizard_id: str
    status: str = "created"
    task_id: str | None = None
    chunk_ids: tuple[str, ...] = ()
    storage_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.collection not in WIZARD_FIXTURE_COLLECTIONS:
            raise WizardDiagnosticError("Corpus document has an invalid collection")
        _validated_uuid(self.wizard_id, "wizard_id")
        if self.status not in DOCUMENT_STATES:
            raise WizardDiagnosticError("Corpus document has an invalid status")
        if self.task_id is not None:
            _validated_uuid(self.task_id, "task_id")
        chunks = tuple(_validated_uuid(item, "chunk_id") for item in self.chunk_ids)
        if chunks != tuple(sorted(chunks)) or len(chunks) != len(set(chunks)):
            raise WizardDiagnosticError(
                "Corpus document chunk_ids must be unique and lexically ordered"
            )
        if self.status in {"created", "save_submitting"} and self.task_id is not None:
            raise WizardDiagnosticError(
                "Pre-submit corpus document must not have task_id"
            )
        if self.status in {"save_submitted", "saved"} and self.task_id is None:
            raise WizardDiagnosticError("Saved corpus document is missing task_id")
        if self.status != "saved" and (chunks or self.storage_fingerprint is not None):
            raise WizardDiagnosticError(
                "Incomplete corpus document must not contain persisted integrity data"
            )
        if self.status == "saved":
            if not chunks:
                raise WizardDiagnosticError("Saved corpus document has no chunk IDs")
            if not _valid_sha256(self.storage_fingerprint):
                raise WizardDiagnosticError(
                    "Saved corpus document has an invalid storage fingerprint"
                )

    def payload(self) -> dict[str, object]:
        return {
            "collection": self.collection,
            "wizard_id": self.wizard_id,
            "status": self.status,
            "task_id": self.task_id,
            "chunk_ids": list(self.chunk_ids),
            "storage_fingerprint": self.storage_fingerprint,
        }


def _validate_ingestion_policy(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _INGESTION_POLICY_KEYS:
        raise WizardDiagnosticError("Corpus ingestion policy is malformed")
    extensions = value.get("supported_extensions")
    if (
        not isinstance(extensions, list)
        or not extensions
        or not all(isinstance(item, str) and item.startswith(".") for item in extensions)
        or extensions != sorted(set(extensions))
    ):
        raise WizardDiagnosticError("Corpus ingestion policy extensions are malformed")
    numeric = {
        key: value.get(key)
        for key in _INGESTION_POLICY_KEYS
        if key != "supported_extensions"
    }
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in numeric.values()
    ):
        raise WizardDiagnosticError("Corpus ingestion policy limits are malformed")
    if numeric["upload_max_file_bytes"] > numeric["upload_max_total_bytes"]:
        raise WizardDiagnosticError("Corpus ingestion policy limits are inconsistent")
    if numeric["upload_read_chunk_bytes"] > numeric["upload_max_file_bytes"]:
        raise WizardDiagnosticError("Corpus ingestion policy read size is inconsistent")
    return {"supported_extensions": list(extensions), **numeric}


@dataclass(frozen=True)
class CorpusGeneration:
    generation_id: str
    fixture_digest: str
    fixtures: Mapping[str, list[dict[str, object]]]
    text_encoding: str
    text_join_separator: str
    ingestion_policy: Mapping[str, object]
    documents: tuple[CorpusDocument, ...]
    created_at: str
    verified_at: str | None = None
    completed_at: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.generation_id, str)
            or not self.generation_id
            or not isinstance(self.fixture_digest, str)
            or not self.fixture_digest
        ):
            raise WizardDiagnosticError("Corpus generation identity is invalid")
        try:
            canonical_encoding = codecs.lookup(self.text_encoding).name
        except (LookupError, TypeError) as exc:
            raise WizardDiagnosticError("Corpus generation encoding is invalid") from exc
        if canonical_encoding != self.text_encoding:
            raise WizardDiagnosticError("Corpus generation encoding is not canonical")
        if not isinstance(self.text_join_separator, str):
            raise WizardDiagnosticError("Corpus generation separator is invalid")
        if (
            _fixture_digest(
                self.fixtures, self.text_encoding, self.text_join_separator
            )
            != self.fixture_digest
        ):
            raise WizardDiagnosticError("Corpus generation fixture digest is invalid")
        _validate_ingestion_policy(self.ingestion_policy)
        names = [document.collection for document in self.documents]
        if len(names) != len(set(names)):
            raise WizardDiagnosticError("Corpus generation duplicates a collection")
        if self.verified_at is not None:
            _validated_timestamp(self.verified_at, "verified_at")
            if (
                set(names) != set(WIZARD_FIXTURE_COLLECTIONS)
                or any(item.status != "saved" for item in self.documents)
            ):
                raise WizardDiagnosticError("Verified corpus generation is incomplete")
        if self.completed_at is not None:
            _validated_timestamp(self.completed_at, "completed_at")
            if self.verified_at is None:
                raise WizardDiagnosticError("Completed corpus generation was not verified")

    def document(self, collection: str) -> CorpusDocument | None:
        return next(
            (item for item in self.documents if item.collection == collection), None
        )

    def payload(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "fixture_digest": self.fixture_digest,
            "fixtures": dict(self.fixtures),
            "text_encoding": self.text_encoding,
            "text_join_separator": self.text_join_separator,
            "ingestion_policy": dict(self.ingestion_policy),
            "documents": [document.payload() for document in self.documents],
            "created_at": self.created_at,
            "verified_at": self.verified_at,
            "completed_at": self.completed_at,
        }


@dataclass(frozen=True)
class CorpusState:
    diagnostic_user_id: str
    active: CorpusGeneration | None
    pending_replacement: CorpusGeneration | None
    pending_cleanup: tuple[CorpusDocument, ...]
    updated_at: str

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "diagnostic": "wizard-corpus",
            "diagnostic_user_id": self.diagnostic_user_id,
            "active": self.active.payload() if self.active else None,
            "pending_replacement": (
                self.pending_replacement.payload()
                if self.pending_replacement
                else None
            ),
            "pending_cleanup": [item.payload() for item in self.pending_cleanup],
            "updated_at": self.updated_at,
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    current = value or _utc_now()
    return current.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _validated_timestamp(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise WizardDiagnosticError(f"{name} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WizardDiagnosticError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise WizardDiagnosticError(f"{name} must include a timezone")
    return value


def _validate_fixture_inventory(
    value: object,
) -> dict[str, list[dict[str, object]]]:
    if not isinstance(value, Mapping) or set(value) != set(WIZARD_FIXTURE_COLLECTIONS):
        raise WizardDiagnosticError("Corpus fixture manifest is malformed")
    result: dict[str, list[dict[str, object]]] = {}
    for collection in WIZARD_FIXTURE_COLLECTIONS:
        raw_files = value.get(collection)
        if not isinstance(raw_files, list) or not raw_files:
            raise WizardDiagnosticError("Corpus fixture manifest is malformed")
        files: list[dict[str, object]] = []
        names: list[str] = []
        for raw_file in raw_files:
            if not isinstance(raw_file, Mapping) or set(raw_file) != _FIXTURE_KEYS:
                raise WizardDiagnosticError("Corpus fixture entry is malformed")
            path = raw_file.get("path")
            size_bytes = raw_file.get("size_bytes")
            sha256 = raw_file.get("sha256")
            if (
                not isinstance(path, str)
                or not path.startswith(collection + "/")
                or Path(path).name.startswith(".")
                or Path(path).as_posix() != path
                or len(Path(path).parts) != 2
                or isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes < 0
                or not _valid_sha256(sha256)
            ):
                raise WizardDiagnosticError("Corpus fixture entry is malformed")
            names.append(path)
            files.append(
                {"path": path, "size_bytes": size_bytes, "sha256": sha256}
            )
        if names != sorted(names) or len(names) != len(set(names)):
            raise WizardDiagnosticError(
                "Corpus fixture manifest must be unique and lexically ordered"
            )
        result[collection] = files
    return result


def _fixture_digest(value: object, text_encoding: str, separator: str) -> str:
    identity = {
        "fixtures": _validate_fixture_inventory(value),
        "text_encoding": text_encoding,
        "text_join_separator": separator,
    }
    encoded = json.dumps(identity, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _validated_uuid(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise WizardDiagnosticError(f"{name} must be a UUID string")
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise WizardDiagnosticError(f"{name} must be a valid UUID") from exc


def validate_diagnostic_user(config: Mapping[str, str]) -> tuple[str, str]:
    raw_user_id = config.get(DIAGNOSTIC_USER_KEY)
    if raw_user_id is None:
        raise WizardDiagnosticError(
            f"Missing required .env variable: {DIAGNOSTIC_USER_KEY}"
        )
    if not raw_user_id:
        raise WizardDiagnosticError(f"{DIAGNOSTIC_USER_KEY} must not be empty")
    if raw_user_id != raw_user_id.strip():
        raise WizardDiagnosticError(
            f"{DIAGNOSTIC_USER_KEY} must not contain leading or trailing whitespace"
        )
    other_user_id = raw_user_id + "_other"
    try:
        get_collection_name(raw_user_id, "conversations")
        get_collection_name(other_user_id, "conversations")
    except (TypeError, ValueError) as exc:
        raise WizardDiagnosticError(
            f"{DIAGNOSTIC_USER_KEY} or its derived cross-user ID is invalid: {exc}"
        ) from exc
    return raw_user_id, other_user_id


def _resolve_fixture_root(fixtures: Path, project_root: Path) -> Path:
    candidate = fixtures if fixtures.is_absolute() else project_root / fixtures
    if candidate.is_symlink():
        raise WizardDiagnosticError(f"Fixture root must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise WizardDiagnosticError(f"Fixture root does not exist: {candidate}") from exc
    if not resolved.is_dir():
        raise WizardDiagnosticError(f"Fixture root is not a directory: {resolved}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixture_files(
    directory: Path, rules: FixtureRules, *, strict_private: bool = False
) -> tuple[FixtureFile, ...]:
    if directory.is_symlink():
        raise WizardDiagnosticError(
            f"Fixture collection must not be a symlink: {directory}"
        )
    if not directory.exists():
        raise WizardDiagnosticError(f"Missing fixture collection: {directory}")
    if not directory.is_dir():
        raise WizardDiagnosticError(
            f"Fixture collection is not a directory: {directory}"
        )

    paths: list[Path] = []
    for entry in sorted(directory.iterdir(), key=lambda value: value.name):
        if entry.name.startswith("."):
            if entry.name == ".gitkeep" and entry.is_file() and not entry.is_symlink():
                continue
            if strict_private:
                raise WizardDiagnosticError(f"Hidden fixture entries are not allowed: {entry}")
            continue
        if entry.is_symlink():
            raise WizardDiagnosticError(f"Fixture must not be a symlink: {entry}")
        if not entry.is_file():
            raise WizardDiagnosticError(
                f"Fixture collections may contain only top-level files: {entry}"
            )
        allowed_extensions = (".txt",) if strict_private else rules.supported_extensions
        if entry.suffix.lower() not in allowed_extensions:
            allowed = ", ".join(allowed_extensions)
            raise WizardDiagnosticError(
                f"Unsupported file extension {entry.suffix or '<none>'!r}; "
                f"expected one of: {allowed}"
            )
        paths.append(entry)
    if not paths:
        raise WizardDiagnosticError(
            f"Fixture collection contains no fixture files: {directory}"
        )

    fixtures: list[FixtureFile] = []
    contents: list[str] = []
    for path in paths:
        try:
            size_bytes = path.stat().st_size
            checksum = _file_sha256(path)
            content = path.read_text(encoding=rules.text_encoding)
        except (OSError, UnicodeError, LookupError) as exc:
            if strict_private:
                raise WizardDiagnosticError(f"Invalid private fixture: {path} ({type(exc).__name__})") from None
            raise WizardDiagnosticError(f"Invalid fixture {path}: {exc}") from exc
        if size_bytes > rules.upload_max_file_bytes:
            raise WizardDiagnosticError(
                f"Fixture exceeds UPLOAD_MAX_FILE_BYTES: {path}"
            )
        if size_bytes > rules.upload_max_total_bytes:
            raise WizardDiagnosticError(
                f"Fixture cannot fit within UPLOAD_MAX_TOTAL_BYTES: {path}"
            )
        if strict_private and not content.strip():
            raise WizardDiagnosticError(f"Private corpus fixture is empty: {path}")
        contents.append(content)
        fixtures.append(FixtureFile(path, size_bytes, checksum))
    if not any(content != "" for content in contents):
        raise WizardDiagnosticError(
            f"Fixture collection must contain non-empty text: {directory}"
        )
    return tuple(fixtures)


def preflight_wizard_fixtures(
    fixtures: Path,
    project_root: Path,
    config: Mapping[str, str] | None = None,
    *,
    strict_private: bool = False,
) -> WizardFixtures:
    rules = FixtureRules.from_config(config or {})
    root = _resolve_fixture_root(fixtures, project_root)
    if strict_private:
        for entry in root.iterdir():
            if entry.name in WIZARD_FIXTURE_COLLECTIONS:
                continue
            if entry.name == ".gitkeep" and entry.is_file() and not entry.is_symlink():
                continue
            raise WizardDiagnosticError("Private corpus root contains an unexpected entry")
    collections = {
        collection: _fixture_files(
            root / collection, rules, strict_private=strict_private
        )
        for collection in WIZARD_FIXTURE_COLLECTIONS
    }
    if strict_private:
        digests = [
            item.sha256
            for collection in WIZARD_FIXTURE_COLLECTIONS
            for item in collections[collection]
        ]
        if len(digests) != len(set(digests)):
            raise WizardDiagnosticError(
                "Private corpus contains duplicate file content"
            )
    return WizardFixtures(
        root, collections["knowledge"], collections["policy"], rules
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def corpus_state_lock(lock_path: Path) -> Iterator[None]:
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(lock_path.parent, 0o700)
        if lock_path.is_symlink():
            raise WizardDiagnosticError("Diagnostic corpus lock must not be a symlink")
        with lock_path.open("a+", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise WizardDiagnosticError(
                    "Another wizard diagnostic is already running"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise WizardDiagnosticError(
            f"Could not lock diagnostic corpus state: {exc}"
        ) from exc


def _document_from_payload(value: object) -> CorpusDocument:
    if not isinstance(value, Mapping) or set(value) != _DOCUMENT_KEYS:
        raise WizardDiagnosticError("Corpus document state must be an object")
    chunk_ids = value.get("chunk_ids")
    if not isinstance(chunk_ids, list):
        raise WizardDiagnosticError("Corpus document chunk_ids must be an array")
    return CorpusDocument(
        collection=value.get("collection"),  # type: ignore[arg-type]
        wizard_id=value.get("wizard_id"),  # type: ignore[arg-type]
        status=value.get("status"),  # type: ignore[arg-type]
        task_id=value.get("task_id"),  # type: ignore[arg-type]
        chunk_ids=tuple(chunk_ids),  # type: ignore[arg-type]
        storage_fingerprint=value.get("storage_fingerprint"),  # type: ignore[arg-type]
    )


def _generation_from_payload(value: object) -> CorpusGeneration | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _GENERATION_KEYS:
        raise WizardDiagnosticError("Corpus generation state must be an object")
    documents = value.get("documents")
    if not isinstance(documents, list):
        raise WizardDiagnosticError("Corpus generation state is malformed")
    verified_at = value.get("verified_at")
    completed_at = value.get("completed_at")
    return CorpusGeneration(
        generation_id=value.get("generation_id"),  # type: ignore[arg-type]
        fixture_digest=value.get("fixture_digest"),  # type: ignore[arg-type]
        fixtures=_validate_fixture_inventory(value.get("fixtures")),
        text_encoding=value.get("text_encoding"),  # type: ignore[arg-type]
        text_join_separator=value.get("text_join_separator"),  # type: ignore[arg-type]
        ingestion_policy=_validate_ingestion_policy(value.get("ingestion_policy")),
        documents=tuple(_document_from_payload(item) for item in documents),
        created_at=_validated_timestamp(value.get("created_at"), "created_at"),
        verified_at=(
            None
            if verified_at is None
            else _validated_timestamp(verified_at, "verified_at")
        ),
        completed_at=(
            None
            if completed_at is None
            else _validated_timestamp(completed_at, "completed_at")
        ),
    )


def load_corpus_state(path: Path, diagnostic_user_id: str) -> CorpusState | None:
    if path.is_symlink():
        raise WizardDiagnosticError("Corpus state must not be a symlink")
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WizardDiagnosticError(f"Corpus state is unreadable: {path}") from exc
    if not isinstance(payload, Mapping) or set(payload) != _STATE_KEYS:
        raise WizardDiagnosticError("Corpus state root must be an object")
    if payload.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise WizardDiagnosticError("Corpus state schema version is unsupported")
    if payload.get("diagnostic") != "wizard-corpus":
        raise WizardDiagnosticError("Corpus state diagnostic type is invalid")
    if payload.get("diagnostic_user_id") != diagnostic_user_id:
        raise WizardDiagnosticError(
            "Corpus state belongs to a different diagnostic user"
        )
    cleanup = payload.get("pending_cleanup")
    if not isinstance(cleanup, list):
        raise WizardDiagnosticError("Corpus state is malformed")
    state = CorpusState(
        diagnostic_user_id=diagnostic_user_id,
        active=_generation_from_payload(payload.get("active")),
        pending_replacement=_generation_from_payload(
            payload.get("pending_replacement")
        ),
        pending_cleanup=tuple(_document_from_payload(item) for item in cleanup),
        updated_at=_validated_timestamp(payload.get("updated_at"), "updated_at"),
    )
    if state.active is not None:
        if (
            state.active.completed_at is None
            or state.active.verified_at is None
            or {item.collection for item in state.active.documents}
            != set(WIZARD_FIXTURE_COLLECTIONS)
            or any(item.status != "saved" for item in state.active.documents)
        ):
            raise WizardDiagnosticError("Active corpus generation is incomplete")
    if (
        state.pending_replacement is not None
        and state.pending_replacement.completed_at is not None
    ):
        raise WizardDiagnosticError("Pending corpus replacement is already completed")
    if any(item.status != "saved" for item in state.pending_cleanup):
        raise WizardDiagnosticError("Pending corpus cleanup contains incomplete data")
    if state.active is None and state.pending_cleanup:
        raise WizardDiagnosticError("Corpus cleanup exists without an active generation")
    documents = (
        (() if state.active is None else state.active.documents)
        + (
            ()
            if state.pending_replacement is None
            else state.pending_replacement.documents
        )
        + state.pending_cleanup
    )
    identities = [(item.collection, item.wizard_id) for item in documents]
    if len(identities) != len(set(identities)):
        raise WizardDiagnosticError("Corpus state contains duplicate document IDs")
    return state


def validate_corpus_preflight(
    state: CorpusState | None,
    fixtures: WizardFixtures,
    *,
    reseed: bool,
) -> str:
    if (
        state is not None
        and state.active is not None
        and state.active.fixture_digest != fixtures.digest
        and not reseed
    ):
        raise WizardDiagnosticError(
            "Fixture corpus changed; rerun with --reseed-corpus"
        )
    if state is None or state.active is None:
        return "create"
    return "reseed" if reseed else "reuse"


def require_empty_first_seed(counts: Mapping[str, object]) -> None:
    if set(counts) != set(WIZARD_FIXTURE_COLLECTIONS):
        raise WizardDiagnosticError("First-seed collection counts are incomplete")
    for value in counts.values():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WizardDiagnosticError("First-seed collection count is invalid")
    if any(counts.values()):
        raise WizardDiagnosticError(
            "Corpus state is missing but diagnostic collections are not empty"
        )


def validate_physical_membership(
    state: CorpusState | None,
    observed: Mapping[str, Mapping[str, str]],
) -> dict[str, int]:
    """Validate known physical chunks without selecting any deletion targets."""

    if set(observed) != set(WIZARD_FIXTURE_COLLECTIONS):
        raise WizardDiagnosticError("Physical corpus membership is incomplete")
    normalized: dict[str, dict[str, str]] = {}
    for collection, objects in observed.items():
        if not isinstance(objects, Mapping):
            raise WizardDiagnosticError("Physical corpus membership is malformed")
        normalized[collection] = {
            _validated_uuid(chunk_id, "chunk_id"): _validated_uuid(
                document_id, "document_id"
            )
            for chunk_id, document_id in objects.items()
        }
    if state is None:
        require_empty_first_seed(
            {collection: len(objects) for collection, objects in normalized.items()}
        )
        return {collection: 0 for collection in WIZARD_FIXTURE_COLLECTIONS}

    by_collection: dict[str, list[tuple[CorpusDocument, bool, bool]]] = {
        collection: [] for collection in WIZARD_FIXTURE_COLLECTIONS
    }
    if state.active is not None:
        for document in state.active.documents:
            by_collection[document.collection].append((document, True, True))
    if state.pending_replacement is not None:
        for document in state.pending_replacement.documents:
            exact = document.status in {"created", "saved"}
            by_collection[document.collection].append(
                (document, exact, document.status == "saved")
            )
    for document in state.pending_cleanup:
        by_collection[document.collection].append((document, True, False))

    for collection, objects in normalized.items():
        documents = by_collection[collection]
        known = {item.wizard_id for item, _, _ in documents}
        if any(document_id not in known for document_id in objects.values()):
            raise WizardDiagnosticError(
                f"Physical {collection} collection contains unknown documents"
            )
        observed_by_document: dict[str, set[str]] = {}
        for chunk_id, document_id in objects.items():
            observed_by_document.setdefault(document_id, set()).add(chunk_id)
        for document, exact, required in documents:
            actual = observed_by_document.get(document.wizard_id, set())
            expected = set(document.chunk_ids)
            if document.status == "created" and actual:
                raise WizardDiagnosticError(
                    f"Created {collection} document unexpectedly has chunks"
                )
            if exact and actual.difference(expected):
                raise WizardDiagnosticError(
                    f"Physical {collection} collection contains unknown chunks"
                )
            if required and actual != expected:
                raise WizardDiagnosticError(
                    f"Physical {collection} collection is missing recorded chunks"
                )
    return {collection: len(objects) for collection, objects in normalized.items()}


def new_corpus_state(diagnostic_user_id: str) -> CorpusState:
    return CorpusState(diagnostic_user_id, None, None, (), _timestamp())


def record_active_ingestion_policy(
    state: CorpusState, fixtures: WizardFixtures
) -> CorpusState:
    """Record current operational policy without changing corpus identity."""

    if state.active is None:
        raise WizardDiagnosticError("Corpus has no active generation")
    if state.active.fixture_digest != fixtures.digest:
        raise WizardDiagnosticError(
            "Cannot record policy for a different fixture corpus"
        )
    active = replace(
        state.active, ingestion_policy=fixtures.rules.ingestion_policy()
    )
    return replace(state, active=active, updated_at=_timestamp())


def write_corpus_state(path: Path, state: CorpusState) -> None:
    try:
        if path.is_symlink():
            raise WizardDiagnosticError("Corpus state must not be a symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(path, state.payload())
    except OSError as exc:
        raise WizardDiagnosticError(f"Could not write corpus state: {path}") from exc


def begin_replacement(
    state: CorpusState,
    run_id: str,
    fixtures: WizardFixtures,
) -> CorpusState:
    if state.pending_replacement is not None:
        raise WizardDiagnosticError("A corpus replacement is already pending")
    generation = CorpusGeneration(
        generation_id=run_id,
        fixture_digest=fixtures.digest,
        fixtures=fixtures.inventory(),
        text_encoding=fixtures.rules.text_encoding,
        text_join_separator=fixtures.rules.text_join_separator,
        ingestion_policy=fixtures.rules.ingestion_policy(),
        documents=(),
        created_at=_timestamp(),
    )
    return replace(state, pending_replacement=generation, updated_at=_timestamp())


def record_replacement_document(
    state: CorpusState,
    document: CorpusDocument,
) -> CorpusState:
    generation = state.pending_replacement
    if generation is None:
        raise WizardDiagnosticError("No corpus replacement is pending")
    if generation.verified_at is not None:
        raise WizardDiagnosticError("Verified corpus replacement cannot be modified")
    existing = generation.document(document.collection)
    if existing is None and document.status != "created":
        raise WizardDiagnosticError("Corpus document must be recorded at Create")
    if existing is not None:
        if existing.wizard_id != document.wizard_id:
            raise WizardDiagnosticError("Corpus document identity cannot change")
        expected_transition = {
            "created": "save_submitting",
            "save_submitting": "save_submitted",
            "save_submitted": "saved",
        }.get(existing.status)
        if document.status != expected_transition:
            raise WizardDiagnosticError("Corpus document transition is invalid")
        if existing.task_id is not None and document.task_id != existing.task_id:
            raise WizardDiagnosticError("Corpus task identity cannot change")
    documents = tuple(
        document if item.collection == document.collection else item
        for item in generation.documents
    )
    if existing is None:
        documents += (document,)
    updated = replace(generation, documents=documents)
    return replace(state, pending_replacement=updated, updated_at=_timestamp())


def mark_replacement_verified(state: CorpusState) -> CorpusState:
    replacement = state.pending_replacement
    if replacement is None:
        raise WizardDiagnosticError("No corpus replacement is pending")
    if replacement.verified_at is not None:
        raise WizardDiagnosticError("Corpus replacement is already verified")
    if (
        {item.collection for item in replacement.documents}
        != set(WIZARD_FIXTURE_COLLECTIONS)
        or any(item.status != "saved" for item in replacement.documents)
    ):
        raise WizardDiagnosticError("Corpus replacement is not complete")
    verified = replace(replacement, verified_at=_timestamp())
    return replace(state, pending_replacement=verified, updated_at=_timestamp())


def discard_pending_replacement(state: CorpusState) -> CorpusState:
    return replace(state, pending_replacement=None, updated_at=_timestamp())


def promote_replacement(state: CorpusState) -> CorpusState:
    replacement = state.pending_replacement
    if replacement is None:
        raise WizardDiagnosticError("No corpus replacement is pending")
    if replacement.verified_at is None:
        raise WizardDiagnosticError("Corpus replacement has not been verified")
    completed = replace(replacement, completed_at=_timestamp())
    cleanup = state.pending_cleanup
    if state.active is not None:
        cleanup += state.active.documents
    return replace(
        state,
        active=completed,
        pending_replacement=None,
        pending_cleanup=cleanup,
        updated_at=_timestamp(),
    )


def remove_pending_cleanup(
    state: CorpusState, document: CorpusDocument
) -> CorpusState:
    remaining = tuple(
        item
        for item in state.pending_cleanup
        if not (
            item.collection == document.collection
            and item.wizard_id == document.wizard_id
        )
    )
    if len(remaining) == len(state.pending_cleanup):
        raise WizardDiagnosticError("Corpus cleanup document is not pending")
    return replace(state, pending_cleanup=remaining, updated_at=_timestamp())


class OperationRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.count = 0
        self.passed_count = 0
        self.failed_count = 0

    @property
    def totals(self) -> dict[str, int]:
        return {
            "total": self.count,
            "passed": self.passed_count,
            "failed": self.failed_count,
        }

    def record(self, name: str, outcome: str, **fields: object) -> None:
        if outcome not in {"passed", "failed"}:
            raise WizardDiagnosticError("Operation outcome must be passed or failed")
        self.count += 1
        if outcome == "passed":
            self.passed_count += 1
        else:
            self.failed_count += 1
        payload = {
            "sequence": self.count,
            "timestamp": _timestamp(),
            "name": name,
            "outcome": outcome,
            "method": None,
            "path": None,
            "expected_status": None,
            "actual_status": None,
            "duration_ms": None,
            "selected_ids": None,
            "poll_count": None,
            **fields,
        }
        try:
            with self.path.open("a", encoding="utf-8") as target:
                target.write(json.dumps(payload, sort_keys=True) + "\n")
                target.flush()
                os.fsync(target.fileno())
        except OSError as exc:
            raise WizardDiagnosticError(
                f"Could not append diagnostic operation: {self.path}"
            ) from exc


def create_diagnostic_run(
    output_root: Path,
    diagnostic_user_id: str,
    fixtures: WizardFixtures,
) -> DiagnosticRun:
    started_at = _utc_now()
    run_id = started_at.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    directory = output_root / run_id
    try:
        if output_root.is_symlink():
            raise WizardDiagnosticError("Diagnostic output root must not be a symlink")
        output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(output_root, 0o700)
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        os.chmod(directory, 0o700)
        operations_path = directory / "operations.jsonl"
        descriptor = os.open(
            operations_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        os.close(descriptor)
        summary_path = directory / "summary.json"
        _write_json_atomic(
            summary_path,
            {
                "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                "diagnostic": "wizard",
                "phase": WIZARD_DIAGNOSTIC_PHASE,
                "run_id": run_id,
                "status": "running",
                "diagnostic_user_id": diagnostic_user_id,
                "fixtures_root": os.fspath(fixtures.root),
                "fixture_digest": fixtures.digest,
                "fixtures": fixtures.inventory(),
                "corpus_identity": {
                    "text_encoding": fixtures.rules.text_encoding,
                    "text_join_separator": fixtures.rules.text_join_separator,
                },
                "ingestion_policy": fixtures.rules.ingestion_policy(),
                "operations_count": 0,
                "operation_totals": {"total": 0, "passed": 0, "failed": 0},
                "scratch_status": "pending",
                "corpus_action": "pending",
                "telemetry_status": "pending",
                "security_status": "pending",
                "trace_deleted": False,
                "compensation_status": (
                    "Compensation path not acceptance-tested because no safe "
                    "deterministic natural failure seam exists."
                ),
                "started_at": _timestamp(started_at),
                "finished_at": None,
                "lifecycle": {
                    "pre_down": "pending",
                    "up": "pending",
                    "down": "pending",
                },
                "failure_stage": None,
            },
        )
    except OSError as exc:
        raise WizardDiagnosticError(
            f"Could not create diagnostic output in {directory}: {exc}"
        ) from exc
    return DiagnosticRun(run_id, directory, operations_path, summary_path)


def update_run_summary(
    run: DiagnosticRun,
    *,
    status: str,
    pre_down_status: str,
    up_status: str,
    down_status: str,
    failure_stage: str | None = None,
    finished: bool = False,
    operations_count: int | None = None,
    operation_totals: Mapping[str, int] | None = None,
    scratch_status: str | None = None,
    corpus_action: str | None = None,
    telemetry_status: str | None = None,
    security_status: str | None = None,
    trace_deleted: bool | None = None,
) -> None:
    try:
        payload = json.loads(run.summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("summary root must be an object")
        payload.update(
            {
                "status": status,
                "lifecycle": {
                    "pre_down": pre_down_status,
                    "up": up_status,
                    "down": down_status,
                },
                "failure_stage": failure_stage,
                "finished_at": _timestamp() if finished else None,
            }
        )
        if operations_count is not None:
            payload["operations_count"] = operations_count
        if operation_totals is not None:
            expected = {"total", "passed", "failed"}
            if set(operation_totals) != expected or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in operation_totals.values()
            ):
                raise ValueError("operation totals are malformed")
            payload["operation_totals"] = dict(operation_totals)
        if scratch_status is not None:
            payload["scratch_status"] = scratch_status
        if corpus_action is not None:
            payload["corpus_action"] = corpus_action
        if telemetry_status is not None:
            payload["telemetry_status"] = telemetry_status
        if security_status is not None:
            payload["security_status"] = security_status
        if trace_deleted is not None:
            if not isinstance(trace_deleted, bool):
                raise ValueError("trace_deleted must be boolean")
            payload["trace_deleted"] = trace_deleted
        _write_json_atomic(run.summary_path, payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WizardDiagnosticError(
            f"Could not update diagnostic summary {run.summary_path}: {exc}"
        ) from exc


def write_ingestion_report(
    run: DiagnosticRun,
    fixtures: WizardFixtures,
    state_path: Path,
) -> None:
    """Consolidate already-recorded corpus proof without reading document text."""

    try:
        summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
        operations = [
            json.loads(line)
            for line in run.operations_path.read_text(encoding="utf-8").splitlines()
        ]
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.exists()
            else None
        )
        if not isinstance(summary, Mapping) or (
            state is not None and not isinstance(state, Mapping)
        ):
            raise ValueError
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise WizardDiagnosticError("Could not consolidate ingestion evidence") from exc
    safe_operations = []
    for item in operations:
        if not isinstance(item, Mapping):
            raise WizardDiagnosticError("Ingestion operation evidence is malformed")
        safe_operations.append({
            key: item.get(key)
            for key in (
                "sequence", "timestamp", "name", "outcome", "duration_ms",
                "collection", "batch_index", "batch_count", "file_count",
                "size_bytes", "chunk_count", "lateon_dimension", "gte_dimension",
                "selected_ids", "poll_count", "trace", "status", "verification_latency_ms",
            )
            if key in item
        })
        trace = safe_operations[-1].get("trace")
        if isinstance(trace, Mapping):
            safe_operations[-1]["trace"] = {
                key: trace.get(key)
                for key in ("schema_version", "operation_id", "outcome", "stages", "counts", "flags", "digests", "samples")
                if key in trace
            }
    inventory = fixtures.inventory()
    active = state.get("active") if isinstance(state, Mapping) else None
    if state is not None and state.get("diagnostic_user_id") != summary.get("diagnostic_user_id"):
        raise WizardDiagnosticError("Ingestion report corpus owner mismatch")
    if not isinstance(active, Mapping) or active.get("fixture_digest") != fixtures.digest:
        active = state.get("pending_replacement") if isinstance(state, Mapping) else None
    documents = active.get("documents") if isinstance(active, Mapping) else []
    generation_operations = operations
    if isinstance(active, Mapping) and active.get("generation_id") != run.run_id:
        generation_id = active.get("generation_id")
        if not isinstance(generation_id, str) or Path(generation_id).name != generation_id:
            raise WizardDiagnosticError("Ingestion evidence generation ID is invalid")
        prior_path = run.directory.parent / generation_id / "operations.jsonl"
        if prior_path.is_file() and not prior_path.is_symlink():
            generation_operations = [json.loads(line) for line in prior_path.read_text().splitlines()]
    collection_reports = {}
    for collection in WIZARD_FIXTURE_COLLECTIONS:
        document = next((item for item in documents or [] if item.get("collection") == collection), {})
        postcondition = next((item for item in generation_operations
                              if item.get("name") == f"corpus.{collection}.save.postcondition"
                              and (item.get("selected_ids") or {}).get("wizard_id") == document.get("wizard_id")
                              and item.get("outcome") == "passed"), {})
        trace = postcondition.get("trace") or {}
        counts = trace.get("counts") or {}
        verified = any(item.get("name") in {"corpus.verify.final", "corpus.verify.active"}
                       and item.get("collection") == collection and item.get("outcome") == "passed"
                       for item in operations)
        collection_reports[collection] = {
            "document_id": document.get("wizard_id"), "task_id": document.get("task_id"),
            "document_count": int(bool(document)), "chunk_ids": document.get("chunk_ids", []),
            "chunk_count": len(document.get("chunk_ids", [])),
            "paragraph_count": counts.get("final_paragraph_count"),
            "storage_fingerprint": document.get("storage_fingerprint"),
            "vector_dimensions": {"lateon": postcondition.get("lateon_dimension"), "gte": postcondition.get("gte_dimension")},
            "vector_presence_ownership_membership_verified": verified,
            "timings": trace.get("stages", {}), "counts": counts,
            "digests": trace.get("digests", {}),
            "source_files": [{**item, "status": "verified" if verified else "failed" if document else "not_attempted"}
                             for item in inventory[collection]],
        }
    verified_files = sum(len(item["source_files"]) for item in collection_reports.values()
                         if item["vector_presence_ownership_membership_verified"])
    payload = {
        "schema_version": "1.0",
        "report": "ingestion-corpus",
        "run_id": run.run_id,
        "status": summary.get("status"),
        "failure_stage": summary.get("failure_stage"),
        "user_id": summary.get("diagnostic_user_id"),
        "source": {
            "root": os.fspath(fixtures.root),
            "fixture_digest": fixtures.digest,
            "inventory": inventory,
            "intended_files": sum(len(value) for value in inventory.values()),
            "discovered_files": sum(len(value) for value in inventory.values()),
            "verified_files": verified_files,
            "failed_files": sum(f["status"] == "failed" for c in collection_reports.values() for f in c["source_files"]),
            "not_attempted_files": sum(f["status"] == "not_attempted" for c in collection_reports.values() for f in c["source_files"]),
            "silently_skipped_files": 0,
        },
        "corpus": {
            "generation_id": active.get("generation_id") if isinstance(active, Mapping) else None,
            "fixture_digest": active.get("fixture_digest") if isinstance(active, Mapping) else None,
            "documents": documents if isinstance(documents, list) else [],
            "verified_at": active.get("verified_at") if isinstance(active, Mapping) else None,
            "completed_at": active.get("completed_at") if isinstance(active, Mapping) else None,
        },
        "operation_totals": summary.get("operation_totals"),
        "collections": collection_reports,
        "operations": safe_operations,
        "lifecycle": summary.get("lifecycle"),
        "validation": {
            "scratch": summary.get("scratch_status"),
            "corpus_action": summary.get("corpus_action"),
            "telemetry": summary.get("telemetry_status"),
            "security": summary.get("security_status"),
            "trace_deleted": summary.get("trace_deleted"),
        },
        "privacy": {"raw_document_text": False},
    }
    if payload["status"] == "succeeded" and (
        verified_files != payload["source"]["intended_files"]
        or not isinstance(state.get("active"), Mapping)
        or state.get("pending_replacement") is not None or state.get("pending_cleanup")
        or not active.get("completed_at") or not active.get("verified_at")
        or active.get("fixture_digest") != fixtures.digest
        or any(not isinstance(c["paragraph_count"], int) or c["paragraph_count"] < 1
               or c["vector_dimensions"] != {"lateon": 128, "gte": 768}
               for c in collection_reports.values())
    ):
        payload.update(status="failed", failure_stage="ingestion_report_evidence_incomplete")
    if any(_file_sha256(item.path) != item.sha256 for collection in WIZARD_FIXTURE_COLLECTIONS for item in fixtures.files(collection)):
        payload.update(status="failed", failure_stage="source_fixture_changed")
    _write_json_atomic(run.ingestion_report_path, payload)
    if summary.get("status") == "succeeded" and payload["status"] != "succeeded":
        raise WizardDiagnosticError("Consolidated ingestion evidence is incomplete")


__all__ = [
    "CorpusDocument",
    "CorpusGeneration",
    "CorpusState",
    "DiagnosticRun",
    "FixtureFile",
    "FixtureBatch",
    "FixtureRules",
    "OperationRecorder",
    "STORAGE_FINGERPRINT_VERSION",
    "WizardDiagnosticError",
    "WizardFixtures",
    "begin_replacement",
    "corpus_state_lock",
    "create_diagnostic_run",
    "discard_pending_replacement",
    "load_corpus_state",
    "mark_replacement_verified",
    "new_corpus_state",
    "preflight_wizard_fixtures",
    "promote_replacement",
    "record_active_ingestion_policy",
    "record_replacement_document",
    "remove_pending_cleanup",
    "require_empty_first_seed",
    "update_run_summary",
    "write_ingestion_report",
    "validate_corpus_preflight",
    "validate_diagnostic_user",
    "validate_physical_membership",
    "write_corpus_state",
]
