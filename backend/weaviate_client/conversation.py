"""Conversation collection operations."""

from __future__ import annotations

from collections.abc import Sequence

from weaviate.classes.query import Filter

from backend.weaviate_client._chunk_collection import (
    _actual_delete_response,
    _dry_run_delete_response,
)
from backend.weaviate_client._base import (
    _CollectionBase,
    _required_text,
    _uuid_text,
    _vector_values,
)
from backend.weaviate_client.models import (
    DeletionReport,
    IncompleteDeletionError,
    SearchResult,
    WeaviateResponseError,
)


class ConversationCollection(_CollectionBase):
    """Access one user's standalone conversation embeddings."""

    collection_type = "conversations"
    id_property = "conversation_id"
    return_properties = ("user_id", "conversation_id", "raw_text")

    def insert(
        self,
        conversation_id: str,
        raw_text: str,
        vector: Sequence[float],
    ) -> str:
        object_id = _uuid_text(conversation_id, "conversation_id")
        text = _required_text(raw_text, "raw_text")
        inserted = self._collection.data.insert(
            properties={
                "user_id": self.user_id,
                "conversation_id": object_id,
                "raw_text": text,
            },
            uuid=object_id,
            vector=_vector_values(vector),
        )
        return str(inserted)

    def delete(self, conversation_id: str) -> None:
        self._collection.data.delete_by_id(
            _uuid_text(conversation_id, "conversation_id")
        )

    def delete_batch(self, conversation_ids: list[str]) -> None:
        """Delete object IDs as a low-level, non-transactional batch primitive.

        Application orchestration is responsible for atomic session deletion
        and any compensation required after a partial storage failure.
        """

        if not isinstance(conversation_ids, list):
            raise TypeError("conversation_ids must be a list")
        object_ids = [
            _uuid_text(item, "conversation_id") for item in conversation_ids
        ]
        if not object_ids:
            return
        unique_ids = list(dict.fromkeys(object_ids))
        self._collection.data.delete_many(
            where=Filter.by_id().contains_any(unique_ids)
        )

    def delete_batch_verified(
        self,
        conversation_ids: list[str],
    ) -> DeletionReport:
        """Delete conversation UUIDs and prove that none remain.

        Already-absent IDs are accepted so an ambiguous or partial prior
        attempt can be retried safely while the session mapping is retained.
        """

        if not isinstance(conversation_ids, list):
            raise TypeError("conversation_ids must be a list")
        object_ids = [
            _uuid_text(item, "conversation_id") for item in conversation_ids
        ]
        if not object_ids:
            return DeletionReport(0, 0, 0, (), ())
        unique_ids = list(dict.fromkeys(object_ids))
        requested = set(unique_ids)
        where = Filter.by_id().contains_any(unique_ids)
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
            raise WeaviateResponseError(
                "conversation deletion verification is incomplete"
            )
        if not set(deleted_ids).issubset(requested) or not set(
            remaining_ids
        ).issubset(requested):
            raise WeaviateResponseError(
                "conversation deletion response escaped its requested scope"
            )

        report = DeletionReport(
            matched=matched,
            successful=successful,
            failed=failed,
            deleted_ids=deleted_ids,
            remaining_ids=remaining_ids,
        )
        if not report.confirmed:
            raise IncompleteDeletionError(
                f"conversation IDs {unique_ids!r}",
                report,
            )
        return report

    def hybrid_search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[SearchResult]:
        return self._hybrid_search(query_text, query_vector, top_k)
