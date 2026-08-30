"""Deployment-owned provider adapters supplied to the provider-neutral runtime."""

from backend.providers.granite_query_rewriter import (
    GraniteCheckpointError,
    GraniteInferenceError,
    GraniteQueryRewriter,
    GraniteRewriteDiagnostics,
    QueryRewriteCompletionClient,
    RoleRoutingLLMClient,
)
from backend.providers.sglang_query_rewriter import (
    SGLangGraniteQueryRewriter,
    SGLangQueryRewriteError,
    SGLangTransport,
)
from backend.providers.query_rewriter_factory import create_query_rewriter
from backend.providers.onnx_embedding import (
    EMBEDDING_DIMENSION,
    ONNXEmbeddingClient,
    ONNXEmbeddingError,
)
from backend.providers.onnx_reranker import (
    ONNXCrossEncoderReranker,
    ONNXRerankerError,
)

__all__ = [
    "GraniteCheckpointError",
    "GraniteInferenceError",
    "GraniteQueryRewriter",
    "GraniteRewriteDiagnostics",
    "EMBEDDING_DIMENSION",
    "ONNXCrossEncoderReranker",
    "ONNXEmbeddingClient",
    "ONNXEmbeddingError",
    "ONNXRerankerError",
    "QueryRewriteCompletionClient",
    "RoleRoutingLLMClient",
    "SGLangGraniteQueryRewriter",
    "SGLangQueryRewriteError",
    "SGLangTransport",
    "create_query_rewriter",
]
