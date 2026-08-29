"""In-memory document-to-paragraph text mapping."""

from __future__ import annotations

from collections.abc import Mapping

from backend.mappings._common import (
    positive_paragraph_id,
    required_uuid,
    validated_document_collection_type,
    validated_user_id,
)


class DocumentMap:
    """Track ordered paragraph text for one user and document collection."""

    def __init__(self, user_id: str, collection_type: str) -> None:
        self.user_id = validated_user_id(user_id)
        self.collection_type = validated_document_collection_type(collection_type)
        self._documents: dict[str, dict[int, str]] = {}

    def create_document(self, document_id: str) -> None:
        key = required_uuid(document_id, "document_id")
        if key in self._documents:
            raise ValueError(f"document {key!r} already exists")
        self._documents[key] = {1: ""}

    def get_paragraphs(self, document_id: str) -> list[int]:
        key = required_uuid(document_id, "document_id")
        return list(self._documents[key])

    def get_full_text(self, document_id: str) -> str:
        key = required_uuid(document_id, "document_id")
        return "".join(self._documents[key].values())

    def update_paragraphs(
        self,
        document_id: str,
        new_paragraph_data: Mapping[int, str],
    ) -> None:
        key = required_uuid(document_id, "document_id")
        if key not in self._documents:
            raise KeyError(key)
        if not isinstance(new_paragraph_data, Mapping):
            raise TypeError("new_paragraph_data must be a mapping")

        validated: dict[int, str] = {}
        for paragraph_id, text in new_paragraph_data.items():
            paragraph = positive_paragraph_id(paragraph_id)
            if not isinstance(text, str):
                raise TypeError("paragraph text must be a string")
            validated[paragraph] = text

        if validated:
            expected_ids = set(range(1, len(validated) + 1))
            if set(validated) != expected_ids:
                raise ValueError("paragraph IDs must be sequential from 1 through N")
            replacement = dict(sorted(validated.items()))
        else:
            replacement = {1: ""}
        self._documents[key] = replacement

    def delete_document(self, document_id: str) -> None:
        key = required_uuid(document_id, "document_id")
        del self._documents[key]
