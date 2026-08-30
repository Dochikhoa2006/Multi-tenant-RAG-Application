"""In-memory paragraph-to-chunk mapping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from backend.mappings._common import (
    positive_paragraph_id,
    required_uuid,
    validated_document_collection_type,
    validated_user_id,
)


ParagraphKey: TypeAlias = tuple[str, int]


def _validated_paragraph_key(paragraph_key: object) -> ParagraphKey:
    if not isinstance(paragraph_key, tuple) or len(paragraph_key) != 2:
        raise TypeError("paragraph_key must be a (document_id, paragraph_id) tuple")
    document_id = required_uuid(paragraph_key[0], "document_id")
    paragraph_id = positive_paragraph_id(paragraph_key[1])
    return document_id, paragraph_id


class ParagraphMap:
    """Track ordered chunk IDs for one user's document collection."""

    def __init__(self, user_id: str, collection_type: str) -> None:
        self.user_id = validated_user_id(user_id)
        self.collection_type = validated_document_collection_type(collection_type)
        self._paragraphs: dict[ParagraphKey, list[str]] = {}
        self._chunk_owners: dict[str, ParagraphKey] = {}

    def set_chunks(self, paragraph_key: ParagraphKey, chunk_ids: list[str]) -> None:
        key = _validated_paragraph_key(paragraph_key)
        if not isinstance(chunk_ids, list):
            raise TypeError("chunk_ids must be a list")
        validated = [required_uuid(item, "chunk_id") for item in chunk_ids]
        if len(set(validated)) != len(validated):
            raise ValueError("chunk_ids must not contain duplicates")

        for chunk_id in validated:
            owner = self._chunk_owners.get(chunk_id)
            if owner is not None and owner != key:
                raise ValueError(
                    f"chunk {chunk_id!r} already belongs to paragraph {owner!r}"
                )

        previous = self._paragraphs.get(key, [])
        for chunk_id in previous:
            if chunk_id not in validated:
                del self._chunk_owners[chunk_id]
        for chunk_id in validated:
            self._chunk_owners[chunk_id] = key
        self._paragraphs[key] = list(validated)

    def get_chunks(self, paragraph_key: ParagraphKey) -> list[str]:
        key = _validated_paragraph_key(paragraph_key)
        return list(self._paragraphs[key])

    def get_document_chunks(self, document_id: str) -> dict[int, list[str]]:
        """Return ordered defensive chunk mappings for one document."""

        document = required_uuid(document_id, "document_id")
        return {
            paragraph_id: list(self._paragraphs[(mapped_document, paragraph_id)])
            for mapped_document, paragraph_id in sorted(self._paragraphs)
            if mapped_document == document
        }

    def _validated_document_replacement(
        self,
        document: str,
        paragraph_chunks: Mapping[int, list[str]],
    ) -> dict[int, list[str]]:
        if not isinstance(paragraph_chunks, Mapping):
            raise TypeError("paragraph_chunks must be a mapping")

        validated: dict[int, list[str]] = {}
        seen_chunks: set[str] = set()
        for raw_paragraph_id, raw_chunk_ids in paragraph_chunks.items():
            paragraph_id = positive_paragraph_id(raw_paragraph_id)
            if not isinstance(raw_chunk_ids, list):
                raise TypeError("each paragraph chunk collection must be a list")
            chunk_ids = [required_uuid(item, "chunk_id") for item in raw_chunk_ids]
            if len(set(chunk_ids)) != len(chunk_ids):
                raise ValueError("chunk_ids must not contain duplicates")
            duplicates = seen_chunks.intersection(chunk_ids)
            if duplicates:
                raise ValueError("a chunk_id cannot belong to multiple paragraphs")
            seen_chunks.update(chunk_ids)
            validated[paragraph_id] = chunk_ids

        if validated:
            expected_ids = set(range(1, len(validated) + 1))
            if set(validated) != expected_ids:
                raise ValueError("paragraph IDs must be sequential from 1 through N")
        else:
            validated = {1: []}

        for chunk_id in seen_chunks:
            owner = self._chunk_owners.get(chunk_id)
            if owner is not None and owner[0] != document:
                raise ValueError(
                    f"chunk {chunk_id!r} already belongs to paragraph {owner!r}"
                )
        return dict(sorted(validated.items()))

    def validate_document_replacement(
        self,
        document_id: str,
        paragraph_chunks: Mapping[int, list[str]],
    ) -> None:
        """Validate a prospective document replacement without mutating state."""

        document = required_uuid(document_id, "document_id")
        self._validated_document_replacement(document, paragraph_chunks)

    def replace_document(
        self,
        document_id: str,
        paragraph_chunks: Mapping[int, list[str]],
    ) -> None:
        """Atomically replace every paragraph mapping for one document."""

        document = required_uuid(document_id, "document_id")
        validated = self._validated_document_replacement(
            document,
            paragraph_chunks,
        )

        existing_keys = [key for key in self._paragraphs if key[0] == document]
        for key in existing_keys:
            for chunk_id in self._paragraphs.pop(key):
                del self._chunk_owners[chunk_id]

        for paragraph_id, chunk_ids in sorted(validated.items()):
            key = (document, paragraph_id)
            self._paragraphs[key] = list(chunk_ids)
            for chunk_id in chunk_ids:
                self._chunk_owners[chunk_id] = key

    def delete_paragraph(self, paragraph_key: ParagraphKey) -> None:
        key = _validated_paragraph_key(paragraph_key)
        chunk_ids = self._paragraphs.pop(key)
        for chunk_id in chunk_ids:
            del self._chunk_owners[chunk_id]

    def delete_document(self, document_id: str) -> None:
        """Remove every paragraph mapping owned by a document."""

        document = required_uuid(document_id, "document_id")
        keys = [key for key in self._paragraphs if key[0] == document]
        for key in keys:
            for chunk_id in self._paragraphs.pop(key):
                del self._chunk_owners[chunk_id]
