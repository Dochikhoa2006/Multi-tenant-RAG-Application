"""Shared operations for knowledge-fact and policy chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from weaviate.classes.query import Filter

from backend.weaviate_client._base import (
    _CollectionBase,
    _positive_paragraph_id,
    _required_text,
    _result_vector,
    _uuid_text,
    _vector_values,
)
from backend.weaviate_client.models import (
    ChunkRecord,
    DeletionReport,
    IncompleteDeletionError,
    PartialParagraphUpdateError,
    SearchResult,
    UserIsolationError,
    WeaviateResponseError,
)


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WeaviateResponseError(f"delete response has invalid {name}")
    return value


def _actual_delete_response(
    value: object,
) -> tuple[int, int, int, tuple[str, ...]]:
    """Parse a verbose mutating delete response using normal count semantics."""

    matched = _count(getattr(value, "matches", None), "matches")
    successful = _count(getattr(value, "successful", None), "successful")
    failed = _count(getattr(value, "failed", None), "failed")
    objects = getattr(value, "objects", None)
    if not isinstance(objects, list) or len(objects) != matched:
        raise WeaviateResponseError(
            "verbose delete response must contain one object per match"
        )
    object_ids: list[str] = []
    successful_ids: list[str] = []
    for item in objects:
        try:
            object_id = str(UUID(str(getattr(item, "uuid", None))))
        except (TypeError, ValueError, AttributeError) as exc:
            raise WeaviateResponseError("delete response contains an invalid UUID") from exc
        item_successful = getattr(item, "successful", None)
        if not isinstance(item_successful, bool):
            raise WeaviateResponseError("delete response has invalid object status")
        object_ids.append(object_id)
        if item_successful:
            successful_ids.append(object_id)
    if successful + failed != matched:
        raise WeaviateResponseError("delete response counts are inconsistent")
    if len(set(object_ids)) != len(object_ids):
        raise WeaviateResponseError("delete response contains duplicate object UUIDs")
    if len(successful_ids) != successful:
        raise WeaviateResponseError("delete response object statuses are inconsistent")
    return matched, successful, failed, tuple(successful_ids)


def _dry_run_delete_response(value: object) -> tuple[int, tuple[str, ...]]:
    """Parse verbose dry-run discovery without assuming objects were deleted."""

    matched = _count(getattr(value, "matches", None), "matches")
    _count(getattr(value, "successful", None), "successful")
    failed = _count(getattr(value, "failed", None), "failed")
    if failed != 0:
        raise WeaviateResponseError("dry-run delete response reports failures")
    objects = getattr(value, "objects", None)
    if not isinstance(objects, list) or len(objects) != matched:
        raise WeaviateResponseError(
            "verbose dry-run response must contain one object per match"
        )
    object_ids: list[str] = []
    for item in objects:
        try:
            object_ids.append(str(UUID(str(getattr(item, "uuid", None)))))
        except (TypeError, ValueError, AttributeError) as exc:
            raise WeaviateResponseError(
                "dry-run delete response contains an invalid UUID"
            ) from exc
    if len(set(object_ids)) != len(object_ids):
        raise WeaviateResponseError(
            "dry-run delete response contains duplicate object UUIDs"
        )
    return matched, tuple(object_ids)


class _ChunkCollection(_CollectionBase):
    id_property = "chunk_id"
    return_properties = (
        "user_id",
        "document_id",
        "paragraph_id",
        "chunk_id",
        "raw_text",
    )

    def insert_chunk(
        self,
        document_id: str,
        paragraph_id: int,
        chunk_id: str,
        raw_text: str,
        vector: Sequence[float],
    ) -> str:
        parent_id = _uuid_text(document_id, "document_id")
        paragraph = _positive_paragraph_id(paragraph_id)
        object_id = _uuid_text(chunk_id, "chunk_id")
        text = _required_text(raw_text, "raw_text")
        inserted = self._collection.data.insert(
            properties={
                "user_id": self.user_id,
                "document_id": parent_id,
                "paragraph_id": paragraph,
                "chunk_id": object_id,
                "raw_text": text,
            },
            uuid=object_id,
            vector=_vector_values(vector),
        )
        return str(inserted)

    def _verified_delete(self, where: Any, scope: str) -> DeletionReport:
        collection = self._collection
        deleted = collection.data.delete_many(where=where, verbose=True)
        matched, successful, failed, deleted_ids = _actual_delete_response(deleted)
        remaining = collection.data.delete_many(
            where=where,
            verbose=True,
            dry_run=True,
        )
        remaining_count, remaining_ids = _dry_run_delete_response(remaining)
        if len(remaining_ids) != remaining_count:
            raise WeaviateResponseError("dry-run delete response is incomplete")
        report = DeletionReport(
            matched=matched,
            successful=successful,
            failed=failed,
            deleted_ids=deleted_ids,
            remaining_ids=remaining_ids,
        )
        if not report.confirmed:
            raise IncompleteDeletionError(scope, report)
        return report

    def delete_by_document(self, document_id: str) -> DeletionReport:
        """Delete matching chunks using Weaviate's non-transactional primitive.

        This method alone does not provide the atomic wizard-delete behavior
        required at the application layer.
        """

        parent_id = _uuid_text(document_id, "document_id")
        return self._verified_delete(
            Filter.by_property("document_id").equal(parent_id),
            f"document {parent_id}",
        )

    def delete_by_paragraphs(
        self,
        document_id: str,
        paragraph_ids: list[int],
    ) -> DeletionReport:
        """Delete matching chunks without application-level atomicity."""

        parent_id = _uuid_text(document_id, "document_id")
        if not isinstance(paragraph_ids, list):
            raise TypeError("paragraph_ids must be a list")
        paragraphs = [_positive_paragraph_id(item) for item in paragraph_ids]
        if not paragraphs:
            return DeletionReport(0, 0, 0, (), ())
        unique_paragraphs = list(dict.fromkeys(paragraphs))
        return self._verified_delete(
            (
                Filter.by_property("document_id").equal(parent_id)
                & Filter.by_property("paragraph_id").contains_any(unique_paragraphs)
            ),
            f"document {parent_id} paragraphs {unique_paragraphs!r}",
        )

    def delete_chunks(self, chunk_ids: list[str]) -> DeletionReport:
        """Delete specific chunk UUIDs and verify that none remain."""

        if not isinstance(chunk_ids, list):
            raise TypeError("chunk_ids must be a list")
        validated = [_uuid_text(item, "chunk_id") for item in chunk_ids]
        if not validated:
            return DeletionReport(0, 0, 0, (), ())
        unique_ids = list(dict.fromkeys(validated))
        return self._verified_delete(
            Filter.by_id().contains_any(unique_ids),
            f"chunk IDs {unique_ids!r}",
        )

    def _discover_ids(self, where: Any, scope: str) -> tuple[str, ...]:
        response = self._collection.data.delete_many(
            where=where,
            verbose=True,
            dry_run=True,
        )
        matched, object_ids = _dry_run_delete_response(response)
        if len(object_ids) != matched:
            raise WeaviateResponseError(f"could not snapshot every object for {scope}")
        return object_ids

    def _fetch_records_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, ChunkRecord]:
        if not chunk_ids:
            return {}
        response = self._collection.query.fetch_objects_by_ids(
            list(chunk_ids),
            limit=len(chunk_ids),
            include_vector=True,
            return_properties=list(self.return_properties),
        )
        objects = getattr(response, "objects", None)
        if not isinstance(objects, list):
            raise WeaviateResponseError("chunk snapshot response objects are malformed")
        records: dict[str, ChunkRecord] = {}
        for item in objects:
            properties = getattr(item, "properties", None)
            if not isinstance(properties, Mapping):
                raise WeaviateResponseError("chunk snapshot properties are malformed")
            if properties.get("user_id") != self.user_id:
                raise UserIsolationError("chunk snapshot belongs to a different user")
            try:
                object_id = _uuid_text(
                    str(getattr(item, "uuid", None)), "object UUID"
                )
                chunk_id = _uuid_text(properties.get("chunk_id"), "chunk_id")
                document_id = _uuid_text(
                    properties.get("document_id"), "document_id"
                )
                paragraph_id = _positive_paragraph_id(
                    properties.get("paragraph_id")
                )
                raw_text = _required_text(properties.get("raw_text"), "raw_text")
            except (TypeError, ValueError) as exc:
                raise WeaviateResponseError(
                    "chunk snapshot properties violate the storage contract"
                ) from exc
            if object_id != chunk_id:
                raise WeaviateResponseError("snapshot object UUID does not match chunk_id")
            record = ChunkRecord(
                object_id=object_id,
                user_id=self.user_id,
                document_id=document_id,
                paragraph_id=paragraph_id,
                chunk_id=chunk_id,
                raw_text=raw_text,
                vector=_result_vector(getattr(item, "vector", None)),
            )
            if chunk_id in records:
                raise WeaviateResponseError("chunk snapshot contains duplicate UUIDs")
            records[chunk_id] = record
        return records

    def _snapshot(self, where: Any, scope: str) -> tuple[ChunkRecord, ...]:
        object_ids = self._discover_ids(where, scope)
        records = self._fetch_records_by_ids(object_ids)
        if set(records) != set(object_ids):
            raise WeaviateResponseError(f"chunk snapshot is incomplete for {scope}")
        return tuple(records[object_id] for object_id in object_ids)

    def snapshot_by_document(self, document_id: str) -> tuple[ChunkRecord, ...]:
        parent_id = _uuid_text(document_id, "document_id")
        records = self._snapshot(
            Filter.by_property("document_id").equal(parent_id),
            f"document {parent_id}",
        )
        if any(record.document_id != parent_id for record in records):
            raise WeaviateResponseError("document snapshot escaped its requested scope")
        return records

    def snapshot_by_paragraphs(
        self,
        document_id: str,
        paragraph_ids: list[int],
    ) -> tuple[ChunkRecord, ...]:
        parent_id = _uuid_text(document_id, "document_id")
        if not isinstance(paragraph_ids, list):
            raise TypeError("paragraph_ids must be a list")
        paragraphs = [_positive_paragraph_id(item) for item in paragraph_ids]
        if not paragraphs:
            return ()
        unique_paragraphs = list(dict.fromkeys(paragraphs))
        records = self._snapshot(
            Filter.by_property("document_id").equal(parent_id)
            & Filter.by_property("paragraph_id").contains_any(unique_paragraphs),
            f"document {parent_id} paragraphs {unique_paragraphs!r}",
        )
        requested = set(unique_paragraphs)
        if any(
            record.document_id != parent_id or record.paragraph_id not in requested
            for record in records
        ):
            raise WeaviateResponseError("paragraph snapshot escaped its requested scope")
        return records

    def restore_chunks(self, records: Sequence[ChunkRecord]) -> tuple[str, ...]:
        """Best-effort insert missing snapshot records and verify exact state."""

        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("records must be a sequence of ChunkRecord values")
        supplied = list(records)
        if not all(isinstance(record, ChunkRecord) for record in supplied):
            raise TypeError("records must contain only ChunkRecord values")
        validated: list[ChunkRecord] = []
        for record in supplied:
            if record.user_id != self.user_id:
                raise UserIsolationError("cannot restore a chunk owned by another user")
            object_id = _uuid_text(record.object_id, "object_id")
            chunk_id = _uuid_text(record.chunk_id, "chunk_id")
            if object_id != chunk_id:
                raise ValueError("record object_id must match chunk_id")
            validated.append(
                ChunkRecord(
                    object_id=object_id,
                    user_id=self.user_id,
                    document_id=_uuid_text(record.document_id, "document_id"),
                    paragraph_id=_positive_paragraph_id(record.paragraph_id),
                    chunk_id=chunk_id,
                    raw_text=_required_text(record.raw_text, "raw_text"),
                    vector=tuple(_vector_values(record.vector)),
                )
            )
        expected = {record.chunk_id: record for record in validated}
        if len(expected) != len(validated):
            raise ValueError("records must not contain duplicate chunk IDs")

        existing = self._fetch_records_by_ids(list(expected))
        for chunk_id, record in existing.items():
            if record != expected[chunk_id]:
                raise WeaviateResponseError(
                    f"existing chunk {chunk_id!r} conflicts with recovery snapshot"
                )
        restored: list[str] = []
        for chunk_id, record in expected.items():
            if chunk_id in existing:
                continue
            self.insert_chunk(
                record.document_id,
                record.paragraph_id,
                record.chunk_id,
                record.raw_text,
                record.vector,
            )
            restored.append(chunk_id)

        verified = self._fetch_records_by_ids(list(expected))
        if verified != expected:
            raise WeaviateResponseError("restored chunk snapshot verification failed")
        return tuple(restored)

    def verify_paragraph_ids(self, expected: Mapping[str, int]) -> None:
        if not isinstance(expected, Mapping):
            raise TypeError("expected must be a mapping")
        validated = {
            _uuid_text(chunk_id, "chunk_id"): _positive_paragraph_id(paragraph_id)
            for chunk_id, paragraph_id in expected.items()
        }
        records = self._fetch_records_by_ids(list(validated))
        if set(records) != set(validated):
            raise WeaviateResponseError("paragraph-ID verification is incomplete")
        if any(
            records[chunk_id].paragraph_id != paragraph_id
            for chunk_id, paragraph_id in validated.items()
        ):
            raise WeaviateResponseError("paragraph-ID verification failed")

    def update_paragraph_ids(
        self,
        chunk_id_to_new_paragraph_id: Mapping[str, int],
    ) -> None:
        if not isinstance(chunk_id_to_new_paragraph_id, Mapping):
            raise TypeError("chunk_id_to_new_paragraph_id must be a mapping")
        updates = [
            (
                _uuid_text(chunk_id, "chunk_id"),
                _positive_paragraph_id(paragraph_id),
            )
            for chunk_id, paragraph_id in chunk_id_to_new_paragraph_id.items()
        ]
        completed: list[str] = []
        for chunk_id, paragraph_id in updates:
            try:
                self._collection.data.update(
                    uuid=chunk_id,
                    properties={"paragraph_id": paragraph_id},
                )
            except Exception as exc:
                raise PartialParagraphUpdateError(
                    completed_chunk_ids=tuple(completed),
                    failed_chunk_id=chunk_id,
                ) from exc
            completed.append(chunk_id)

    def hybrid_search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[SearchResult]:
        return self._hybrid_search(query_text, query_vector, top_k)
