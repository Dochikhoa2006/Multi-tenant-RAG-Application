"""Validation shared by process-local mapping classes."""

from __future__ import annotations

from uuid import UUID

from backend.config import get_collection_name


DOCUMENT_COLLECTION_TYPES = frozenset({"knowledge_facts", "policy"})


def required_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def required_uuid(value: object, name: str) -> str:
    raw_value = required_identifier(value, name)
    try:
        return str(UUID(raw_value))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid UUID") from exc


def validated_user_id(user_id: object) -> str:
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a string")
    get_collection_name(user_id, "conversations")
    return user_id


def validated_document_collection_type(collection_type: object) -> str:
    normalized = required_identifier(collection_type, "collection_type").lower()
    if normalized not in DOCUMENT_COLLECTION_TYPES:
        allowed = ", ".join(sorted(DOCUMENT_COLLECTION_TYPES))
        raise ValueError(f"collection_type must be one of: {allowed}")
    return normalized


def positive_paragraph_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("paragraph_id must be an integer")
    if value <= 0:
        raise ValueError("paragraph_id must be greater than zero")
    return value
