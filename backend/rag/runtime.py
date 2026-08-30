"""Injectable provider contracts and runtime dependencies for Stage 4."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    """Shared synchronous-completion and asynchronous-streaming LLM contract."""

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        reasoning: str | None = None,
        max_output_tokens: int | None = None,
    ) -> str: ...

    def stream(
        self,
        prompt: str,
        *,
        model: str,
        reasoning: str | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str] | Awaitable[AsyncIterator[str]]: ...


class EmbeddingClient(Protocol):
    """Provider embedding contract with explicit model selection."""

    def embed(self, text: str, *, model: str) -> Sequence[float]: ...

    def embed_many(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> Sequence[Sequence[float]]: ...


class Tokenizer(Protocol):
    """Minimal token-counting seam implemented by ``tiktoken.Encoding``."""

    def encode(self, text: str) -> Sequence[int]: ...


@dataclass(frozen=True)
class RerankResult:
    """Provider-neutral cross-encoder result referencing an input document."""

    index: int
    score: float


class CrossEncoderReranker(Protocol):
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        model: str,
        top_n: int,
    ) -> Sequence[RerankResult]: ...


class ConversationCollectionWriter(Protocol):
    def insert(
        self,
        conversation_id: str,
        raw_text: str,
        vector: Sequence[float],
    ) -> str: ...


ConversationCollectionFactory = Callable[[str], ConversationCollectionWriter]
TimingObserver = Callable[[str, float], None]


BackgroundWorkFactory = Callable[[], Awaitable[None]]


class BackgroundTaskQueue(Protocol):
    """Observable per-user FIFO dispatcher owned by the application layer."""

    async def enqueue(
        self,
        user_id: str,
        operation: str,
        work_factory: BackgroundWorkFactory,
    ) -> object:
        """Register work and return only after the queue has accepted it."""


class RAGRuntime:
    """Own the explicitly injected external capabilities used by Stage 4."""

    def __init__(
        self,
        llm: LLMClient,
        embeddings: EmbeddingClient,
        reranker: CrossEncoderReranker,
        conversation_collection_factory: ConversationCollectionFactory,
        *,
        tokenizer: Tokenizer | None = None,
        background_queue: BackgroundTaskQueue | None = None,
    ) -> None:
        if not callable(getattr(llm, "complete", None)) or not callable(
            getattr(llm, "stream", None)
        ):
            raise TypeError("llm must provide complete() and stream()")
        if not callable(getattr(embeddings, "embed", None)) or not callable(
            getattr(embeddings, "embed_many", None)
        ):
            raise TypeError("embeddings must provide embed() and embed_many()")
        if not callable(getattr(reranker, "rerank", None)):
            raise TypeError("reranker must provide rerank()")
        if not callable(conversation_collection_factory):
            raise TypeError("conversation_collection_factory must be callable")
        if tokenizer is not None and not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("tokenizer must provide encode()")
        if background_queue is not None and not callable(
            getattr(background_queue, "enqueue", None)
        ):
            raise TypeError("background_queue must provide enqueue()")
        self.llm = llm
        self.embeddings = embeddings
        self.reranker = reranker
        self.conversation_collection_factory = conversation_collection_factory
        self.tokenizer = tokenizer
        self.background_queue = background_queue


_DEFAULT_RUNTIME: RAGRuntime | None = None


def configure_default_runtime(runtime: RAGRuntime) -> None:
    if not isinstance(runtime, RAGRuntime):
        raise TypeError("runtime must be a RAGRuntime")
    global _DEFAULT_RUNTIME
    _DEFAULT_RUNTIME = runtime


def resolve_runtime(runtime: RAGRuntime | None) -> RAGRuntime:
    if runtime is not None:
        if not isinstance(runtime, RAGRuntime):
            raise TypeError("runtime must be a RAGRuntime")
        return runtime
    if _DEFAULT_RUNTIME is None:
        raise RuntimeError("configure_default_runtime() must be called first")
    return _DEFAULT_RUNTIME
