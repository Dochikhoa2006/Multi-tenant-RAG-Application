"""Injectable process-local dependencies for wizard orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol
from uuid import UUID, uuid4

from backend.mappings._common import (
    validated_document_collection_type,
    validated_user_id,
)
from backend.mappings.document_map import DocumentMap
from backend.mappings.paragraph_map import ParagraphMap
from backend.processing.chunker import chunk_paragraph
from backend.processing.paragraph_splitter import split_into_paragraphs
from backend.weaviate_client.client import WeaviateManager
from backend.weaviate_client.knowledge import KnowledgeCollection
from backend.weaviate_client.models import ChunkRecord, DeletionReport
from backend.weaviate_client.policy import PolicyCollection


class ChunkEmbedder(Protocol):
    """Provider-neutral embedder with shared single and batch behavior."""

    def embed(self, text: str) -> Sequence[float]:
        """Return one dense vector for non-empty chunk text."""

    def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Prefer one provider request for a complete save batch."""


class ChunkCollection(Protocol):
    """Subset of knowledge/policy storage used by wizard operations."""

    def snapshot_by_document(self, document_id: str) -> tuple[ChunkRecord, ...]: ...

    def snapshot_by_paragraphs(
        self, document_id: str, paragraph_ids: list[int]
    ) -> tuple[ChunkRecord, ...]: ...

    def delete_by_document(self, document_id: str) -> DeletionReport: ...

    def delete_by_paragraphs(
        self,
        document_id: str,
        paragraph_ids: list[int],
    ) -> DeletionReport: ...

    def delete_chunks(self, chunk_ids: list[str]) -> DeletionReport: ...

    def restore_chunks(self, records: Sequence[ChunkRecord]) -> tuple[str, ...]: ...

    def verify_paragraph_ids(self, expected: dict[str, int]) -> None: ...

    def insert_chunk(
        self,
        document_id: str,
        paragraph_id: int,
        chunk_id: str,
        raw_text: str,
        vector: Sequence[float],
    ) -> str: ...

    def update_paragraph_ids(
        self,
        chunk_id_to_new_paragraph_id: dict[str, int],
    ) -> None: ...


CollectionFactory = Callable[[WeaviateManager, str, str], ChunkCollection]
TextProcessor = Callable[[str], list[str]]
UUIDFactory = Callable[[], UUID | str]


def _default_collection_factory(
    manager: WeaviateManager,
    user_id: str,
    collection_type: str,
) -> ChunkCollection:
    if collection_type == "knowledge_facts":
        return KnowledgeCollection(manager, user_id)
    return PolicyCollection(manager, user_id)


class WizardRuntime:
    """Own Stage 3 dependencies and authoritative process-local mappings."""

    def __init__(
        self,
        manager: WeaviateManager,
        embedder: ChunkEmbedder,
        *,
        paragraph_splitter: TextProcessor = split_into_paragraphs,
        paragraph_chunker: TextProcessor = chunk_paragraph,
        uuid_factory: UUIDFactory = uuid4,
        collection_factory: CollectionFactory = _default_collection_factory,
    ) -> None:
        if not callable(paragraph_splitter):
            raise TypeError("paragraph_splitter must be callable")
        if not callable(paragraph_chunker):
            raise TypeError("paragraph_chunker must be callable")
        if not callable(uuid_factory):
            raise TypeError("uuid_factory must be callable")
        if not callable(collection_factory):
            raise TypeError("collection_factory must be callable")
        if not callable(getattr(embedder, "embed", None)) or not callable(
            getattr(embedder, "embed_many", None)
        ):
            raise TypeError(
                "embedder must provide both embed(text) and embed_many(texts)"
            )

        self.manager = manager
        self.embedder = embedder
        self.paragraph_splitter = paragraph_splitter
        self.paragraph_chunker = paragraph_chunker
        self.uuid_factory = uuid_factory
        self.collection_factory = collection_factory
        self._document_maps: dict[tuple[str, str], DocumentMap] = {}
        self._paragraph_maps: dict[tuple[str, str], ParagraphMap] = {}

    def _scope(self, user_id: str, collection_type: str) -> tuple[str, str]:
        return (
            validated_user_id(user_id),
            validated_document_collection_type(collection_type),
        )

    def document_map(self, user_id: str, collection_type: str) -> DocumentMap:
        scope = self._scope(user_id, collection_type)
        if scope not in self._document_maps:
            self._document_maps[scope] = DocumentMap(*scope)
        return self._document_maps[scope]

    def paragraph_map(self, user_id: str, collection_type: str) -> ParagraphMap:
        scope = self._scope(user_id, collection_type)
        if scope not in self._paragraph_maps:
            self._paragraph_maps[scope] = ParagraphMap(*scope)
        return self._paragraph_maps[scope]

    def collection(self, user_id: str, collection_type: str) -> ChunkCollection:
        scope = self._scope(user_id, collection_type)
        return self.collection_factory(self.manager, *scope)


_DEFAULT_RUNTIME: WizardRuntime | None = None


def configure_default_runtime(runtime: WizardRuntime) -> None:
    """Configure the runtime used when public functions omit injection."""

    if not isinstance(runtime, WizardRuntime):
        raise TypeError("runtime must be a WizardRuntime")
    global _DEFAULT_RUNTIME
    _DEFAULT_RUNTIME = runtime


def resolve_runtime(runtime: WizardRuntime | None) -> WizardRuntime:
    """Resolve an injected runtime or the explicitly configured default."""

    if runtime is not None:
        if not isinstance(runtime, WizardRuntime):
            raise TypeError("runtime must be a WizardRuntime")
        return runtime
    if _DEFAULT_RUNTIME is None:
        raise RuntimeError("configure_default_runtime() must be called first")
    return _DEFAULT_RUNTIME
