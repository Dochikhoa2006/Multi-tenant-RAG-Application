"""Application-level configuration for the RAG backend."""

from __future__ import annotations

import base64
import os
import re


def _required_string(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _supported_extensions() -> frozenset[str]:
    raw_value = _required_string(
        "SUPPORTED_FILE_EXTENSIONS",
        ".txt,.md,.csv,.json,.xml,.log",
    )
    extensions: set[str] = set()
    for item in raw_value.split(","):
        extension = item.strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        extensions.add(extension)

    if not extensions:
        raise ValueError("SUPPORTED_FILE_EXTENSIONS must contain at least one extension")
    return frozenset(extensions)


def _raw_string(name: str, default: str) -> str:
    return os.getenv(name, default)


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    value = default if raw_value is None else int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


WEAVIATE_URL = _required_string("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_PORT = _positive_int("WEAVIATE_GRPC_PORT", 50051)
WEAVIATE_GRPC_SECURE = _boolean(
    "WEAVIATE_GRPC_SECURE",
    WEAVIATE_URL.lower().startswith("https://"),
)
SUPPORTED_FILE_EXTENSIONS = _supported_extensions()
# These names are architecture invariants, not deployment settings. Retrieval
# and lifecycle code relies on the three stores remaining physically distinct.
COLLECTION_TYPES = frozenset({"conversations", "knowledge_facts", "policy"})
_COLLECTION_TYPE_SUFFIXES = {
    "conversations": "Conversations",
    "knowledge_facts": "KnowledgeFacts",
    "policy": "Policy",
}
USER_ID_PATTERN = _required_string("USER_ID_PATTERN", r"^[A-Za-z0-9_-]+$")
TEXT_FILE_ENCODING = _required_string("TEXT_FILE_ENCODING", "utf-8-sig")
TEXT_FILE_JOIN_SEPARATOR = _raw_string("TEXT_FILE_JOIN_SEPARATOR", "\n\n")

try:
    _USER_ID_PATTERN = re.compile(USER_ID_PATTERN)
except re.error as exc:
    raise ValueError("USER_ID_PATTERN must be a valid regular expression") from exc


def get_collection_name(user_id: str, collection_type: str) -> str:
    """Return the isolated physical Weaviate collection name for a user.

    Only the three collection types defined by the architecture are accepted.
    The exact UTF-8 bytes of the validated user identifier are encoded with
    unpadded RFC 4648 Base32. This is reversible and avoids collision-prone
    sanitization while producing only GraphQL-compatible name characters.
    """

    if not isinstance(user_id, str):
        raise TypeError("user_id must be a string")
    if not user_id:
        raise ValueError("user_id must not be empty")
    if user_id != user_id.strip():
        raise ValueError("user_id must not contain leading or trailing whitespace")
    if _USER_ID_PATTERN.fullmatch(user_id) is None:
        raise ValueError(
            f"user_id does not match configured USER_ID_PATTERN {USER_ID_PATTERN!r}"
        )

    if not isinstance(collection_type, str):
        raise TypeError("collection_type must be a string")
    normalized_collection_type = collection_type.strip().lower()
    if normalized_collection_type not in COLLECTION_TYPES:
        allowed = ", ".join(sorted(COLLECTION_TYPES))
        raise ValueError(
            f"Unsupported collection_type {collection_type!r}; expected one of: {allowed}"
        )

    user_token = base64.b32encode(user_id.encode("utf-8")).decode("ascii").rstrip("=")
    suffix = _COLLECTION_TYPE_SUFFIXES[normalized_collection_type]
    return f"RagUser_{user_token}_{suffix}"
