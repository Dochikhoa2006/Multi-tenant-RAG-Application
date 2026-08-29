"""Knowledge-facts chunk collection operations."""

from backend.weaviate_client._chunk_collection import _ChunkCollection


class KnowledgeCollection(_ChunkCollection):
    """Access one user's knowledge-facts collection."""

    collection_type = "knowledge_facts"
