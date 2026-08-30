"""Provider-neutral Stage 4 RAG building blocks."""

from backend.rag.embedder import (
    embed_chunks,
    embed_conversation,
    embed_conversation_background,
    embed_text,
)
from backend.rag.generator import generate_answer_stream
from backend.rag.pipeline import UserRetrievalCollections, run_rag_pipeline
from backend.rag.query_rewriter import rewrite_query
from backend.rag.query_rewrite_contract import ConversationPair, QueryRewritePrompt
from backend.rag.retrieval import retrieve
from backend.rag.runtime import (
    BackgroundTaskQueue,
    BackgroundWorkFactory,
    ConversationCollectionFactory,
    ConversationCollectionWriter,
    CrossEncoderReranker,
    EmbeddingClient,
    LLMClient,
    RAGRuntime,
    RerankResult,
    TimingObserver,
    Tokenizer,
    configure_default_runtime,
)
from backend.rag.session_title import generate_session_title, validate_session_title

__all__ = [
    "BackgroundTaskQueue",
    "BackgroundWorkFactory",
    "ConversationCollectionFactory",
    "ConversationCollectionWriter",
    "ConversationPair",
    "CrossEncoderReranker",
    "EmbeddingClient",
    "LLMClient",
    "RAGRuntime",
    "RerankResult",
    "QueryRewritePrompt",
    "Tokenizer",
    "TimingObserver",
    "UserRetrievalCollections",
    "configure_default_runtime",
    "embed_chunks",
    "embed_conversation",
    "embed_conversation_background",
    "embed_text",
    "generate_answer_stream",
    "generate_session_title",
    "validate_session_title",
    "retrieve",
    "rewrite_query",
    "run_rag_pipeline",
]
