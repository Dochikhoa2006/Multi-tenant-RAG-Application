"""Immutable input and output contracts for local evaluation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Literal
from uuid import UUID


RECORD_SCHEMA_VERSION = "1.0"
RESULT_SCHEMA_VERSION = "1.0"
_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "source",
        "request_id",
        "conversation_id",
        "original_query",
        "rewritten_query",
        "response",
        "knowledge_contexts",
        "knowledge_context_ids",
        "policy_contexts",
        "policy_context_ids",
        "retrieved_contexts",
        "context_roles",
        "telemetry",
        "reference",
        "reference_context_ids",
        "captured_at",
    }
)
_REQUIRED_RECORD_FIELDS = _RECORD_FIELDS - {"reference", "reference_context_ids"}


class RecordValidationError(ValueError):
    """Raised when evaluation evidence violates the immutable contract."""


class FrozenDict(Mapping[str, Any]):
    """A small immutable mapping whose nested values are also frozen."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._data = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self._data)!r})"


def _freeze_json(value: Any, *, path: str = "telemetry") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecordValidationError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecordValidationError(f"{path} contains a non-string key")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return FrozenDict(frozen)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise RecordValidationError(f"{path} contains a non-JSON value")


def thaw_json(value: Any) -> Any:
    """Return ordinary JSON-compatible containers without changing values."""

    if isinstance(value, FrozenDict):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _canonical_uuid(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise RecordValidationError(f"{field} must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as exc:
        raise RecordValidationError(f"{field} must be a canonical UUID string") from exc
    if str(parsed) != value:
        raise RecordValidationError(f"{field} must be a canonical UUID string")
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordValidationError(f"{field} must be a nonblank string")
    return value


def _text_tuple(
    value: Any,
    field: str,
    *,
    reject_blank: bool,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RecordValidationError(f"{field} must be a sequence of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or (reject_blank and not item.strip()):
            adjective = "nonblank " if reject_blank else ""
            raise RecordValidationError(
                f"{field}[{index}] must be a {adjective}string"
            )
        result.append(item)
    return tuple(result)


def _captured_at(value: Any) -> datetime:
    if isinstance(value, str):
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            value = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise RecordValidationError("captured_at must be an RFC 3339 datetime") from exc
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RecordValidationError("captured_at must be timezone-aware")
    try:
        offset = value.utcoffset()
    except ValueError as exc:
        raise RecordValidationError("captured_at has an invalid timezone") from exc
    if offset is None:
        raise RecordValidationError("captured_at must be timezone-aware")
    return value.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    schema_version: str
    source: Literal["e2e", "rag_ask"]
    request_id: str
    conversation_id: str
    original_query: str
    rewritten_query: str
    response: str
    knowledge_contexts: tuple[str, ...]
    knowledge_context_ids: tuple[str, ...]
    policy_contexts: tuple[str, ...]
    policy_context_ids: tuple[str, ...]
    retrieved_contexts: tuple[str, ...]
    context_roles: tuple[Literal["knowledge", "policy"], ...]
    telemetry: FrozenDict
    reference: str | None
    reference_context_ids: tuple[str, ...]
    captured_at: datetime

    def __post_init__(self) -> None:
        if self.schema_version != RECORD_SCHEMA_VERSION:
            raise RecordValidationError(
                f"schema_version must be {RECORD_SCHEMA_VERSION!r}"
            )
        if self.source not in {"e2e", "rag_ask"}:
            raise RecordValidationError("source must be 'e2e' or 'rag_ask'")

        object.__setattr__(self, "request_id", _canonical_uuid(self.request_id, "request_id"))
        object.__setattr__(
            self,
            "conversation_id",
            _canonical_uuid(self.conversation_id, "conversation_id"),
        )
        for field in ("original_query", "rewritten_query", "response"):
            object.__setattr__(self, field, _required_text(getattr(self, field), field))

        for field, reject_blank in (
            ("knowledge_contexts", True),
            ("knowledge_context_ids", True),
            ("policy_contexts", True),
            ("policy_context_ids", True),
            ("retrieved_contexts", True),
            ("context_roles", True),
            ("reference_context_ids", True),
        ):
            object.__setattr__(
                self,
                field,
                _text_tuple(getattr(self, field), field, reject_blank=reject_blank),
            )

        if len(self.knowledge_contexts) != len(self.knowledge_context_ids):
            raise RecordValidationError(
                "knowledge_contexts and knowledge_context_ids must align"
            )
        if len(self.policy_contexts) != len(self.policy_context_ids):
            raise RecordValidationError(
                "policy_contexts and policy_context_ids must align"
            )
        all_ids = self.knowledge_context_ids + self.policy_context_ids
        if len(set(all_ids)) != len(all_ids):
            raise RecordValidationError("generation context IDs must be unique")

        expected_contexts = self.knowledge_contexts + self.policy_contexts
        if self.retrieved_contexts != expected_contexts:
            raise RecordValidationError(
                "retrieved_contexts must equal Knowledge followed by Policy"
            )
        expected_roles = ("knowledge",) * len(self.knowledge_contexts) + (
            "policy",
        ) * len(self.policy_contexts)
        if self.context_roles != expected_roles:
            raise RecordValidationError(
                "context_roles must exactly parallel Knowledge then Policy"
            )

        if not isinstance(self.telemetry, Mapping):
            raise RecordValidationError("telemetry must be a JSON object")
        object.__setattr__(self, "telemetry", _freeze_json(self.telemetry))

        if self.reference is not None:
            object.__setattr__(self, "reference", _required_text(self.reference, "reference"))
        if len(set(self.reference_context_ids)) != len(self.reference_context_ids):
            raise RecordValidationError("reference_context_ids must be unique")
        object.__setattr__(self, "captured_at", _captured_at(self.captured_at))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationRecord":
        if not isinstance(value, Mapping):
            raise RecordValidationError("evaluation record must be a JSON object")
        keys = set(value)
        unknown = keys - _RECORD_FIELDS
        missing = _REQUIRED_RECORD_FIELDS - keys
        if unknown:
            raise RecordValidationError(
                f"unknown evaluation record fields: {', '.join(sorted(unknown))}"
            )
        if missing:
            raise RecordValidationError(
                f"missing evaluation record fields: {', '.join(sorted(missing))}"
            )
        return cls(
            schema_version=value["schema_version"],
            source=value["source"],
            request_id=value["request_id"],
            conversation_id=value["conversation_id"],
            original_query=value["original_query"],
            rewritten_query=value["rewritten_query"],
            response=value["response"],
            knowledge_contexts=value["knowledge_contexts"],
            knowledge_context_ids=value["knowledge_context_ids"],
            policy_contexts=value["policy_contexts"],
            policy_context_ids=value["policy_context_ids"],
            retrieved_contexts=value["retrieved_contexts"],
            context_roles=value["context_roles"],
            telemetry=value["telemetry"],
            reference=value.get("reference"),
            reference_context_ids=value.get("reference_context_ids", ()),
            captured_at=value["captured_at"],
        )

    @classmethod
    def from_json(cls, payload: str) -> "EvaluationRecord":
        try:
            value = json.loads(payload, parse_constant=lambda token: (_raise_nonfinite(token)))
        except (json.JSONDecodeError, RecordValidationError) as exc:
            raise RecordValidationError("invalid evaluation record JSON") from exc
        return cls.from_mapping(value)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "response": self.response,
            "knowledge_contexts": list(self.knowledge_contexts),
            "knowledge_context_ids": list(self.knowledge_context_ids),
            "policy_contexts": list(self.policy_contexts),
            "policy_context_ids": list(self.policy_context_ids),
            "retrieved_contexts": list(self.retrieved_contexts),
            "context_roles": list(self.context_roles),
            "telemetry": thaw_json(self.telemetry),
            "reference": self.reference,
            "reference_context_ids": list(self.reference_context_ids),
            "captured_at": format_utc(self.captured_at),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_mapping(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _raise_nonfinite(token: str) -> Any:
    raise RecordValidationError(f"non-finite JSON number: {token}")


MetricStatus = Literal["succeeded", "failed", "skipped"]


@dataclass(frozen=True, slots=True)
class MetricOutcome:
    name: str
    ragas_metric: str
    status: MetricStatus
    query_basis: Literal["original_query", "rewritten_query", "none"]
    score: float | None = None
    duration_ms: float | None = None
    error_code: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if self.status == "succeeded":
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise ValueError("successful metric outcome requires a numeric score")
            score = float(self.score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("metric score must be finite and within [0, 1]")
            object.__setattr__(self, "score", score)
            if self.error_code is not None or self.error_type is not None:
                raise ValueError("successful metric outcome cannot contain an error")
        elif self.score is not None:
            raise ValueError("non-successful metric outcome cannot contain a score")
        if self.duration_ms is not None:
            if (
                isinstance(self.duration_ms, bool)
                or not isinstance(self.duration_ms, (int, float))
                or not math.isfinite(float(self.duration_ms))
                or float(self.duration_ms) < 0
            ):
                raise ValueError("metric duration must be a finite nonnegative number")
            object.__setattr__(self, "duration_ms", float(self.duration_ms))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ragas_metric": self.ragas_metric,
            "status": self.status,
            "query_basis": self.query_basis,
            "score": self.score,
            "duration_ms": self.duration_ms,
            "error_code": self.error_code,
            "error_type": self.error_type,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    schema_version: str
    evaluation_id: str
    record_sha256: str
    source: Literal["e2e", "rag_ask"]
    request_id: str
    conversation_id: str
    started_at: datetime
    completed_at: datetime
    status: Literal["succeeded", "partial", "failed"]
    ragas_version: str | None
    judge: FrozenDict
    embeddings: FrozenDict
    metrics: tuple[MetricOutcome, ...]
    setup_error_code: str | None = None
    setup_error_type: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != RESULT_SCHEMA_VERSION:
            raise ValueError(f"result schema_version must be {RESULT_SCHEMA_VERSION!r}")
        _canonical_uuid(self.evaluation_id, "evaluation_id")
        _canonical_uuid(self.request_id, "request_id")
        _canonical_uuid(self.conversation_id, "conversation_id")
        if len(self.record_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.record_sha256
        ):
            raise ValueError("record_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "started_at", _captured_at(self.started_at))
        object.__setattr__(self, "completed_at", _captured_at(self.completed_at))
        object.__setattr__(self, "judge", _freeze_json(self.judge, path="judge"))
        object.__setattr__(
            self, "embeddings", _freeze_json(self.embeddings, path="embeddings")
        )
        object.__setattr__(self, "metrics", tuple(self.metrics))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "record_sha256": self.record_sha256,
            "source": self.source,
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "started_at": format_utc(self.started_at),
            "completed_at": format_utc(self.completed_at),
            "status": self.status,
            "ragas_version": self.ragas_version,
            "judge": thaw_json(self.judge),
            "embeddings": thaw_json(self.embeddings),
            "metrics": [outcome.to_mapping() for outcome in self.metrics],
            "setup_error_code": self.setup_error_code,
            "setup_error_type": self.setup_error_type,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_mapping(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

