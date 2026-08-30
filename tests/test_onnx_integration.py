from __future__ import annotations

import os

import numpy as np
import pytest

from backend.model_config import EMBEDDING_MODEL, RERANKER_MODEL
from backend.providers.onnx_embedding import ONNXEmbeddingClient
from backend.providers.onnx_reranker import ONNXCrossEncoderReranker


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ONNX_CUDA_TESTS") != "1",
    reason="set RUN_ONNX_CUDA_TESTS=1 with provisioned CUDA models",
)


def test_provisioned_models_run_entirely_offline_on_cuda() -> None:
    embedder = ONNXEmbeddingClient()
    embeddings = embedder.embed_many(
        ["retrieval augmented generation", "vector database"],
        model=EMBEDDING_MODEL,
    )
    assert len(embeddings) == 2
    assert all(len(vector) == 768 for vector in embeddings)
    assert all(np.isfinite(vector).all() for vector in embeddings)
    assert all(np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5) for vector in embeddings)

    reranker = ONNXCrossEncoderReranker()
    results = reranker.rerank(
        "What stores vector embeddings?",
        ["A vector database stores embeddings.", "A toaster browns bread."],
        model=RERANKER_MODEL,
        top_n=2,
    )
    assert [result.index for result in results] == [0, 1]
    assert results[0].score >= results[1].score

