"""Application services and process-local Stage 5 registries."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock
from uuid import uuid4

from backend.mappings._common import required_uuid, validated_user_id
from backend.mappings.session_map import SessionMap
from backend.model_config import EMBEDDING_MODEL
from backend.rag.pipeline import UserRetrievalCollections
from backend.rag.runtime import RAGRuntime
from backend.rag.session_title import validate_session_title
from backend.task_queue import InMemoryTaskQueue
from backend.weaviate_client.client import WeaviateManager
from backend.weaviate_client.conversation import ConversationCollection
from backend.weaviate_client.knowledge import KnowledgeCollection
from backend.weaviate_client.policy import PolicyCollection
from backend.wizard.runtime import WizardRuntime


@dataclass(frozen=True)
class ConversationTranscript:
    conversation_id: str
    question: str
    answer: str


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    user_id: str
    title: str
    conversations: tuple[ConversationTranscript, ...]


class ActiveChatStreamError(RuntimeError):
    """A session cannot be deleted while an answer stream owns it."""


class SessionDeletionInProgressError(RuntimeError):
    """A session cannot accept work after deletion has been reserved."""


class ChatRegistry:
    """Process-local session titles and complete ordered transcripts."""

    def __init__(self) -> None:
        self._session_maps: dict[str, SessionMap] = {}
        self._titles: dict[tuple[str, str], str] = {}
        self._transcripts: dict[tuple[str, str], dict[str, ConversationTranscript]] = {}
        self._active_streams: dict[tuple[str, str], set[str]] = {}
        self._deleting_sessions: set[tuple[str, str]] = set()
        self._lock = RLock()

    def _map(self, user_id: str) -> SessionMap:
        user = validated_user_id(user_id)
        return self._session_maps.setdefault(user, SessionMap(user))

    def create_session(self, user_id: str, session_id: str) -> SessionSnapshot:
        user = validated_user_id(user_id)
        session = required_uuid(session_id, "session_id")
        with self._lock:
            mapping = self._map(user)
            mapping.create_session(session)
            key = (user, session)
            self._titles[key] = "New Chat"
            self._transcripts[key] = {}
            return self._snapshot_locked(user, session)

    def list_sessions(self, user_id: str) -> list[SessionSnapshot]:
        user = validated_user_id(user_id)
        with self._lock:
            mapping = self._map(user)
            return [
                self._snapshot_locked(user, session_id)
                for session_id in mapping.list_sessions(user)
            ]

    def get_session(self, user_id: str, session_id: str) -> SessionSnapshot:
        user = validated_user_id(user_id)
        session = required_uuid(session_id, "session_id")
        with self._lock:
            return self._snapshot_locked(user, session)

    def record_conversation(
        self,
        user_id: str,
        session_id: str,
        conversation_id: str,
        question: str,
        answer: str,
    ) -> None:
        user = validated_user_id(user_id)
        session = required_uuid(session_id, "session_id")
        conversation = required_uuid(conversation_id, "conversation_id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be empty")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must not be empty")
        with self._lock:
            if (user, session) in self._deleting_sessions:
                raise SessionDeletionInProgressError(session)
            mapping = self._map(user)
            mapping.add_conversation(session, conversation)
            self._transcripts[(user, session)][conversation] = ConversationTranscript(
                conversation_id=conversation,
                question=question,
                answer=answer,
            )

    def update_title(self, user_id: str, session_id: str, title: str) -> str:
        user = validated_user_id(user_id)
        session = required_uuid(session_id, "session_id")
        validated = validate_session_title(title)
        with self._lock:
            self._snapshot_locked(user, session)
            self._titles[(user, session)] = validated
        return validated

    def conversation_ids(self, user_id: str, session_id: str) -> list[str]:
        user = validated_user_id(user_id)
        session = required_uuid(session_id, "session_id")
        with self._lock:
            return self._map(user).get_conversations(session)

    def begin_chat_stream(
        self,
        user_id: str,
        session_id: str,
        conversation_id: str,
    ) -> None:
        """Atomically bind one response stream to an existing session."""

        user = validated_user_id(user_id)
        session = required_uuid(session_id, "session_id")
        conversation = required_uuid(conversation_id, "conversation_id")
        key = (user, session)
        with self._lock:
            self._snapshot_locked(user, session)
            if key in self._deleting_sessions:
                raise SessionDeletionInProgressError(session)
            active = self._active_streams.setdefault(key, set())
            if conversation in active:
                raise ValueError("conversation already has an active stream")
            active.add(conversation)

    def end_chat_stream(
        self,
        user_id: str,
        session_id: str,
        conversation_id: str,
    ) -> None:
        """Release a stream marker; repeated cleanup is an idempotent no-op."""

        user = validated_user_id(user_id)
        session = required_uuid(session_id, "session_id")
        conversation = required_uuid(conversation_id, "conversation_id")
        key = (user, session)
        with self._lock:
            active = self._active_streams.get(key)
            if active is None:
                return
            active.discard(conversation)
            if not active:
                self._active_streams.pop(key, None)

    def reserve_session_deletion(
        self,
        user_id: str,
        session_id: str,
    ) -> list[str]:
        """Prevent new streams and return the full cascade-delete snapshot."""

        user = validated_user_id(user_id)
        session = required_uuid(session_id, "session_id")
        key = (user, session)
        with self._lock:
            self._snapshot_locked(user, session)
            if self._active_streams.get(key):
                raise ActiveChatStreamError(session)
            if key in self._deleting_sessions:
                raise SessionDeletionInProgressError(session)
            self._deleting_sessions.add(key)
            return self._map(user).get_conversations(session)

    def abort_session_deletion(self, user_id: str, session_id: str) -> None:
        """Release a failed or unqueued deletion reservation."""

        user = validated_user_id(user_id)
        session = required_uuid(session_id, "session_id")
        with self._lock:
            self._deleting_sessions.discard((user, session))

    def commit_session_deletion(
        self,
        user_id: str,
        session_id: str,
    ) -> list[str]:
        """Remove a reserved session after storage deletion is confirmed."""

        user = validated_user_id(user_id)
        session = required_uuid(session_id, "session_id")
        key = (user, session)
        with self._lock:
            if key not in self._deleting_sessions:
                raise RuntimeError("session deletion is not reserved")
            if self._active_streams.get(key):
                raise ActiveChatStreamError(session)
            try:
                conversations = self._map(user).delete_session(session)
                self._titles.pop(key, None)
                self._transcripts.pop(key, None)
                self._active_streams.pop(key, None)
                return conversations
            finally:
                self._deleting_sessions.discard(key)

    def delete_session(self, user_id: str, session_id: str) -> list[str]:
        self.reserve_session_deletion(user_id, session_id)
        try:
            return self.commit_session_deletion(user_id, session_id)
        except Exception:
            self.abort_session_deletion(user_id, session_id)
            raise

    def _snapshot_locked(self, user: str, session: str) -> SessionSnapshot:
        conversation_ids = self._map(user).get_conversations(session)
        transcripts = self._transcripts[(user, session)]
        return SessionSnapshot(
            session_id=session,
            user_id=user,
            title=self._titles[(user, session)],
            conversations=tuple(transcripts[item] for item in conversation_ids),
        )


class _UnavailableWizardEmbedder:
    def embed(self, text: str) -> Sequence[float]:
        raise RuntimeError("wizard embedding provider is not configured")

    def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise RuntimeError("wizard embedding provider is not configured")


class _RAGWizardEmbedder:
    def __init__(self, runtime: RAGRuntime) -> None:
        self._embeddings = runtime.embeddings

    def embed(self, text: str) -> Sequence[float]:
        return self._embeddings.embed(text, model=EMBEDDING_MODEL)

    def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._embeddings.embed_many(texts, model=EMBEDDING_MODEL)


RetrievalCollectionsFactory = Callable[[str], UserRetrievalCollections]
ConversationCollectionFactory = Callable[[str], object]


class AppServices:
    """Explicitly injected dependencies shared by the FastAPI application."""

    def __init__(
        self,
        *,
        manager: object | None = None,
        rag_runtime: RAGRuntime | None = None,
        wizard_runtime: WizardRuntime | None = None,
        task_queue: InMemoryTaskQueue | None = None,
        chat_registry: ChatRegistry | None = None,
        retrieval_collections_factory: RetrievalCollectionsFactory | None = None,
        conversation_collection_factory: ConversationCollectionFactory | None = None,
        uuid_factory: Callable[[], object] = uuid4,
    ) -> None:
        if not callable(uuid_factory):
            raise TypeError("uuid_factory must be callable")
        self.manager = manager if manager is not None else WeaviateManager()
        self.task_queue = (
            task_queue if task_queue is not None else InMemoryTaskQueue()
        )
        self.chat_registry = chat_registry if chat_registry is not None else ChatRegistry()
        self.rag_runtime = rag_runtime
        if rag_runtime is not None:
            if rag_runtime.background_queue is None:
                rag_runtime.background_queue = self.task_queue
            elif rag_runtime.background_queue is not self.task_queue:
                raise ValueError("RAG runtime and application must share one task queue")
        if wizard_runtime is not None and wizard_runtime.manager is not self.manager:
            raise ValueError("wizard runtime and application must share one manager")
        self.wizard_runtime = (
            wizard_runtime
            if wizard_runtime is not None
            else WizardRuntime(
                self.manager,  # type: ignore[arg-type]
                _RAGWizardEmbedder(rag_runtime)
                if rag_runtime is not None
                else _UnavailableWizardEmbedder(),
            )
        )
        self.retrieval_collections_factory = (
            retrieval_collections_factory or self._default_retrieval_collections
        )
        self.conversation_collection_factory = (
            conversation_collection_factory
            or (
                rag_runtime.conversation_collection_factory
                if rag_runtime is not None
                else lambda user_id: ConversationCollection(self.manager, user_id)
            )
        )
        self.uuid_factory = uuid_factory

    def new_uuid(self) -> str:
        return required_uuid(str(self.uuid_factory()), "generated UUID")

    def require_rag_runtime(self) -> RAGRuntime:
        if self.rag_runtime is None:
            raise RuntimeError("RAG provider runtime is not configured")
        return self.rag_runtime

    def require_wizard_embedding(self) -> WizardRuntime:
        if self.rag_runtime is None and isinstance(
            self.wizard_runtime.embedder, _UnavailableWizardEmbedder
        ):
            raise RuntimeError("wizard embedding provider is not configured")
        return self.wizard_runtime

    def _default_retrieval_collections(self, user_id: str) -> UserRetrievalCollections:
        return UserRetrievalCollections(
            user_id=user_id,
            conversations=ConversationCollection(self.manager, user_id),
            knowledge_facts=KnowledgeCollection(self.manager, user_id),
            policy=PolicyCollection(self.manager, user_id),
        )


__all__ = [
    "ActiveChatStreamError",
    "AppServices",
    "ChatRegistry",
    "ConversationTranscript",
    "SessionDeletionInProgressError",
    "SessionSnapshot",
]
