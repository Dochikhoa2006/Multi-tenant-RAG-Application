"""Policy chunk collection operations."""

from backend.weaviate_client._chunk_collection import _ChunkCollection


class PolicyCollection(_ChunkCollection):
    """Access one user's policy collection."""

    collection_type = "policy"
