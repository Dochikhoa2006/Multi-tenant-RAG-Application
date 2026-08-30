"""Synchronous Weaviate v4 access for isolated per-user collections."""

from backend.weaviate_client.client import WeaviateManager
from backend.weaviate_client.conversation import ConversationCollection
from backend.weaviate_client.knowledge import KnowledgeCollection
from backend.weaviate_client.models import (
    ChunkRecord,
    DeletionReport,
    IncompleteDeletionError,
    IncompatibleCollectionSchemaError,
    PartialParagraphUpdateError,
    SearchResult,
    UserIsolationError,
    WeaviateResponseError,
)
from backend.weaviate_client.policy import PolicyCollection

__all__ = [
    "ConversationCollection",
    "ChunkRecord",
    "DeletionReport",
    "IncompleteDeletionError",
    "IncompatibleCollectionSchemaError",
    "KnowledgeCollection",
    "PartialParagraphUpdateError",
    "PolicyCollection",
    "SearchResult",
    "UserIsolationError",
    "WeaviateManager",
    "WeaviateResponseError",
]
