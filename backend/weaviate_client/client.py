"""Connection and schema management for Weaviate v4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.connect import ConnectionParams

from backend.config import (
    WEAVIATE_GRPC_PORT,
    WEAVIATE_GRPC_SECURE,
    WEAVIATE_URL,
    get_collection_name,
)
from backend.weaviate_client.models import IncompatibleCollectionSchemaError


_COLLECTION_ORDER = ("conversations", "knowledge_facts", "policy")


@dataclass(frozen=True)
class _PropertyContract:
    data_type: DataType
    index_filterable: bool
    index_searchable: bool
    index_range_filters: bool = False


_CONVERSATION_CONTRACT = {
    "user_id": _PropertyContract(DataType.TEXT, True, False),
    "conversation_id": _PropertyContract(DataType.UUID, True, False),
    "raw_text": _PropertyContract(DataType.TEXT, False, True),
}
_CHUNK_CONTRACT = {
    "user_id": _PropertyContract(DataType.TEXT, True, False),
    "document_id": _PropertyContract(DataType.UUID, True, False),
    "paragraph_id": _PropertyContract(DataType.INT, True, False),
    "chunk_id": _PropertyContract(DataType.UUID, True, False),
    "raw_text": _PropertyContract(DataType.TEXT, False, True),
}


def _field(value: object, name: str, legacy_name: str | None = None) -> object:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
        if legacy_name is not None and legacy_name in value:
            return value[legacy_name]
        return None
    if hasattr(value, name):
        return getattr(value, name)
    if legacy_name is not None and hasattr(value, legacy_name):
        return getattr(value, legacy_name)
    return None


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _schema_error(collection_name: str, reason: str) -> None:
    raise IncompatibleCollectionSchemaError(collection_name, reason)


def _validate_properties(
    collection_name: str,
    config: object,
    expected: Mapping[str, _PropertyContract],
) -> None:
    properties = _field(config, "properties")
    if isinstance(properties, (str, bytes)) or not isinstance(properties, Sequence):
        _schema_error(collection_name, "properties are missing or malformed")

    actual: dict[str, object] = {}
    for item in properties:
        name = _field(item, "name")
        if not isinstance(name, str) or not name or name in actual:
            _schema_error(collection_name, "property names are malformed or duplicated")
        actual[name] = item

    if set(actual) != set(expected):
        _schema_error(
            collection_name,
            f"expected properties {sorted(expected)!r}, got {sorted(actual)!r}",
        )

    for name, contract in expected.items():
        item = actual[name]
        actual_type = _enum_value(_field(item, "data_type", "dataType"))
        if actual_type != _enum_value(contract.data_type):
            _schema_error(collection_name, f"property {name!r} has the wrong data type")
        checks = (
            ("index_filterable", "indexFilterable", contract.index_filterable),
            ("index_searchable", "indexSearchable", contract.index_searchable),
            ("index_range_filters", "indexRangeFilters", contract.index_range_filters),
        )
        for field_name, legacy_name, required in checks:
            if _field(item, field_name, legacy_name) is not required:
                _schema_error(
                    collection_name,
                    f"property {name!r} has incompatible {field_name}",
                )


def _validate_vector_config(collection_name: str, config: object) -> None:
    vector_config = _field(config, "vector_config", "vectorConfig")
    if not isinstance(vector_config, Mapping) or len(vector_config) != 1:
        _schema_error(
            collection_name,
            "exactly one named native vector configuration is required",
        )

    named_vector = next(iter(vector_config.values()))
    vectorizer_config = _field(named_vector, "vectorizer")
    vectorizer = _enum_value(_field(vectorizer_config, "vectorizer"))
    if not isinstance(vectorizer, str) or vectorizer.lower() != "none":
        _schema_error(collection_name, "the vectorizer must be self-provided/none")

    source_properties = _field(vectorizer_config, "source_properties", "sourceProperties")
    if source_properties not in (None, []):
        _schema_error(
            collection_name,
            "self-provided vectors must not declare source properties",
        )


def _validate_existing_collection(
    collections: Any,
    collection_name: str,
    expected: Mapping[str, _PropertyContract],
) -> None:
    config = collections.use(collection_name).config.get()
    configured_name = _field(config, "name")
    if configured_name != collection_name:
        _schema_error(collection_name, "the returned schema has a mismatched name")
    references = _field(config, "references")
    if references not in (None, []):
        _schema_error(collection_name, "cross-references are not permitted")
    _validate_properties(collection_name, config, expected)
    _validate_vector_config(collection_name, config)


def _identifier_property(name: str, data_type: DataType) -> Property:
    return Property(
        name=name,
        data_type=data_type,
        index_filterable=True,
        index_searchable=False,
    )


def _conversation_properties() -> list[Property]:
    return [
        _identifier_property("user_id", DataType.TEXT),
        _identifier_property("conversation_id", DataType.UUID),
        Property(
            name="raw_text",
            data_type=DataType.TEXT,
            index_filterable=False,
            index_searchable=True,
        ),
    ]


def _chunk_properties() -> list[Property]:
    return [
        _identifier_property("user_id", DataType.TEXT),
        _identifier_property("document_id", DataType.UUID),
        _identifier_property("paragraph_id", DataType.INT),
        _identifier_property("chunk_id", DataType.UUID),
        Property(
            name="raw_text",
            data_type=DataType.TEXT,
            index_filterable=False,
            index_searchable=True,
        ),
    ]


class WeaviateManager:
    """Own one synchronous client connection and its collection schemas."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self._connected = False

    @property
    def client(self) -> Any:
        """Return the connected client or fail before network operations."""

        if not self._connected or self._client is None:
            raise RuntimeError("WeaviateManager.connect() must be called first")
        return self._client

    def connect(self) -> Any:
        """Connect once and return the synchronous Weaviate client."""

        if self._connected:
            return self._client
        if self._client is None:
            self._client = weaviate.WeaviateClient(
                connection_params=ConnectionParams.from_url(
                    WEAVIATE_URL,
                    grpc_port=WEAVIATE_GRPC_PORT,
                    grpc_secure=WEAVIATE_GRPC_SECURE,
                )
            )
        self._client.connect()
        self._connected = True
        return self._client

    def disconnect(self) -> None:
        """Close the owned connection once; repeated calls are harmless."""

        if not self._connected or self._client is None:
            return
        try:
            self._client.close()
        finally:
            self._connected = False

    def ensure_user_collections(self, user_id: str) -> None:
        """Validate existing collections, then create any missing schemas."""

        collections = self.client.collections
        collection_states: list[tuple[str, str, bool]] = []
        for collection_type in _COLLECTION_ORDER:
            name = get_collection_name(user_id, collection_type)
            collection_states.append((collection_type, name, collections.exists(name)))

        for collection_type, name, exists in collection_states:
            if exists:
                contract = (
                    _CONVERSATION_CONTRACT
                    if collection_type == "conversations"
                    else _CHUNK_CONTRACT
                )
                _validate_existing_collection(collections, name, contract)

        for collection_type, name, exists in collection_states:
            if exists:
                continue
            properties = (
                _conversation_properties()
                if collection_type == "conversations"
                else _chunk_properties()
            )
            collections.create(
                name=name,
                properties=properties,
                vector_config=Configure.Vectors.self_provided(),
            )
