"""In-memory paragraph-to-chunk mapping."""

from __future__ import annotations

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

    def delete_paragraph(self, paragraph_key: ParagraphKey) -> None:
        key = _validated_paragraph_key(paragraph_key)
        chunk_ids = self._paragraphs.pop(key)
        for chunk_id in chunk_ids:
            del self._chunk_owners[chunk_id]
