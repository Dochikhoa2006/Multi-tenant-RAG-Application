"""Stable result types returned by the Weaviate access layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class WeaviateResponseError(RuntimeError):
    """Raised when Weaviate returns data that violates the collection contract."""


class UserIsolationError(WeaviateResponseError):
    """Raised when a result is owned by a different user than its collection."""


class IncompatibleCollectionSchemaError(RuntimeError):
    """Raised when an existing collection cannot satisfy the Stage 2 contract."""

    def __init__(self, collection_name: str, reason: str) -> None:
        self.collection_name = collection_name
        self.reason = reason
        super().__init__(f"collection {collection_name!r} is incompatible: {reason}")


class PartialParagraphUpdateError(RuntimeError):
    """Report how far a non-transactional paragraph-ID update progressed."""

    def __init__(
        self,
        completed_chunk_ids: tuple[str, ...],
        failed_chunk_id: str,
    ) -> None:
        self.completed_chunk_ids = completed_chunk_ids
        self.failed_chunk_id = failed_chunk_id
        super().__init__(
            f"paragraph-ID update failed for chunk {failed_chunk_id!r} after "
            f"{len(completed_chunk_ids)} successful update(s)"
        )


@dataclass(frozen=True)
class ChunkRecord:
    """A complete recoverable knowledge/policy chunk snapshot."""

    object_id: str
    user_id: str
    document_id: str
    paragraph_id: int
    chunk_id: str
    raw_text: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class DeletionReport:
    """Verified outcome of one scoped multi-object deletion."""

    matched: int
    successful: int
    failed: int
    deleted_ids: tuple[str, ...]
    remaining_ids: tuple[str, ...]

    @property
    def confirmed(self) -> bool:
        return self.failed == 0 and not self.remaining_ids


class IncompleteDeletionError(RuntimeError):
    """Raised when a scoped deletion cannot prove an empty postcondition."""

    def __init__(self, scope: str, report: DeletionReport) -> None:
        self.scope = scope
        self.report = report
        super().__init__(
            f"incomplete deletion for {scope}: {report.failed} failed, "
            f"{len(report.remaining_ids)} remaining"
        )


@dataclass(frozen=True)
class SearchResult:
    """One normalized first-stage hybrid-search result."""

    object_id: str
    properties: Mapping[str, Any]
    score: float
    vector: tuple[float, ...] | None = None
