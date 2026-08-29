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
class SearchResult:
    """One normalized first-stage hybrid-search result."""

    object_id: str
    properties: Mapping[str, Any]
    vector: tuple[float, ...]
    score: float
