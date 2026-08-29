"""Shared operations for knowledge-fact and policy chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from weaviate.classes.query import Filter

from backend.weaviate_client._base import (
    _CollectionBase,
    _positive_paragraph_id,
    _required_text,
    _uuid_text,
    _vector_values,
)
from backend.weaviate_client.models import PartialParagraphUpdateError, SearchResult


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

    def delete_by_document(self, document_id: str) -> None:
        """Delete matching chunks using Weaviate's non-transactional primitive.

        This method alone does not provide the atomic wizard-delete behavior
        required at the application layer.
        """

        parent_id = _uuid_text(document_id, "document_id")
        self._collection.data.delete_many(
            where=Filter.by_property("document_id").equal(parent_id)
        )

    def delete_by_paragraphs(
        self,
        document_id: str,
        paragraph_ids: list[int],
    ) -> None:
        """Delete matching chunks without application-level atomicity."""

        parent_id = _uuid_text(document_id, "document_id")
        if not isinstance(paragraph_ids, list):
            raise TypeError("paragraph_ids must be a list")
        paragraphs = [_positive_paragraph_id(item) for item in paragraph_ids]
        if not paragraphs:
            return
        unique_paragraphs = list(dict.fromkeys(paragraphs))
        self._collection.data.delete_many(
            where=(
                Filter.by_property("document_id").equal(parent_id)
                & Filter.by_property("paragraph_id").contains_any(unique_paragraphs)
            )
        )

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
