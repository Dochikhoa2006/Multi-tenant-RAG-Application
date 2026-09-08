"""Conversation collection operations."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid5

from weaviate.classes.query import Filter

from backend.model_config import (
    LATE_INTERACTION_VECTOR_NAME,
    MMR_DIVERSITY_VECTOR_NAME,
)
from backend.weaviate_client._chunk_collection import (
    _actual_delete_response,
    _dry_run_delete_response,
)
from backend.weaviate_client._base import (
    _CollectionBase,
    _required_text,
    _multi_vector_values,
    _uuid_text,
    _vector_values,
)
from backend.weaviate_client.models import (
    ConversationWriteRecoveryError,
    DeletionReport,
    IncompleteDeletionError,
    SearchResult,
    WeaviateResponseError,
)


class ConversationCollection(_CollectionBase):
    """Access one user's standalone conversation embeddings."""

    collection_type = "conversations"
    id_property = "segment_id"
    canonical_id_property = "conversation_id"
    search_property = "segment_text"
    segment_index_property = "segment_index"
    hydrate_raw_text = True
    return_properties = (
        "user_id",
        "conversation_id",
        "segment_id",
        "segment_index",
        "raw_text",
        "segment_text",
    )
    search_return_properties = (
        "user_id",
        "conversation_id",
        "segment_id",
        "segment_index",
        "segment_text",
    )
    hydration_return_properties = (
        "user_id",
        "conversation_id",
        "segment_id",
        "segment_index",
        "raw_text",
    )

    def insert(
        self,
        conversation_id: str,
        raw_text: str,
        segment_texts: Sequence[str],
        segment_vectors: Sequence[Sequence[Sequence[float]]],
        diversity_vector: Sequence[float],
    ) -> str:
        canonical_id = _uuid_text(conversation_id, "conversation_id")
        text = _required_text(raw_text, "raw_text")
        if isinstance(segment_texts, (str, bytes)) or not isinstance(
            segment_texts, Sequence
        ):
            raise TypeError("segment_texts must be a sequence of strings")
        segments = [_required_text(item, "segment_text") for item in segment_texts]
        if not segments or "".join(segments) != text:
            raise ValueError("Conversation segments must reconstruct raw_text exactly")
        if isinstance(segment_vectors, (str, bytes)) or not isinstance(
            segment_vectors, Sequence
        ):
            raise TypeError("segment_vectors must be a sequence of multi-vectors")
        vectors = [
            _multi_vector_values(item, "segment multi-vector")
            for item in segment_vectors
        ]
        if len(vectors) != len(segments):
            raise ValueError("Conversation segments and multi-vectors must align")
        dimensions = {len(row) for matrix in vectors for row in matrix}
        if len(dimensions) != 1:
            raise ValueError("Conversation multi-vectors must share one dimension")
        diversity = _vector_values(diversity_vector, "diversity_vector")

        namespace = UUID(canonical_id)
        segment_ids = [
            str(uuid5(namespace, f"retrieval-segment:{index}"))
            for index in range(len(segments))
        ]
        attempted: list[str] = []
        try:
            for index, (segment_id, segment_text, multi_vector) in enumerate(
                zip(segment_ids, segments, vectors, strict=True)
            ):
                attempted.append(segment_id)
                inserted = self._collection.data.insert(
                    properties={
                        "user_id": self.user_id,
                        "conversation_id": canonical_id,
                        "segment_id": segment_id,
                        "segment_index": index,
                        "raw_text": text,
                        "segment_text": segment_text,
                    },
                    uuid=segment_id,
                    vector={
                        LATE_INTERACTION_VECTOR_NAME: multi_vector,
                        MMR_DIVERSITY_VECTOR_NAME: diversity,
                    },
                )
                inserted_id = _uuid_text(str(inserted), "inserted object UUID")
                if inserted_id != segment_id:
                    raise WeaviateResponseError(
                        "inserted object UUID does not match segment_id"
                    )
        except Exception as write_error:
            try:
                self._verified_delete(
                    Filter.by_id().contains_any(attempted),
                    f"{len(attempted)} attempted conversation segment ID(s)",
                )
            except Exception as cleanup_error:
                raise ConversationWriteRecoveryError(
                    tuple(attempted),
                    write_error,
                    cleanup_error,
                ) from write_error
            raise
        return canonical_id

    def _verified_delete(self, where: object, scope: str) -> DeletionReport:
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

    def delete(self, conversation_id: str) -> None:
        self._collection.data.delete_many(
            where=Filter.by_property("conversation_id").equal(
                _uuid_text(conversation_id, "conversation_id")
            )
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
            where=Filter.by_property("conversation_id").contains_any(unique_ids)
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
        where = Filter.by_property("conversation_id").contains_any(unique_ids)
        return self._verified_delete(
            where,
            f"conversation IDs {unique_ids!r}",
        )

    def hybrid_search(
        self,
        query_text: str,
        query_vector: Sequence[Sequence[float]],
        top_k: int,
    ) -> list[SearchResult]:
        return self._hybrid_search(query_text, query_vector, top_k)
