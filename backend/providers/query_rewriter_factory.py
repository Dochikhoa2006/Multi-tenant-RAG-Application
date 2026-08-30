"""Deployment composition helper for explicit Granite engine selection."""

from __future__ import annotations

from collections.abc import Callable

from backend.model_config import QUERY_REWRITE_ENGINE
from backend.providers.granite_query_rewriter import (
    GraniteQueryRewriter,
    QueryRewriteCompletionClient,
)
from backend.providers.sglang_query_rewriter import SGLangGraniteQueryRewriter


QueryRewriterFactory = Callable[[], QueryRewriteCompletionClient]


def create_query_rewriter(
    *,
    engine: str = QUERY_REWRITE_ENGINE,
    sglang_factory: QueryRewriterFactory = SGLangGraniteQueryRewriter,
    transformers_factory: QueryRewriterFactory = GraniteQueryRewriter,
) -> QueryRewriteCompletionClient:
    """Construct one engine; never fall back automatically to the other."""

    normalized = engine.strip().lower() if isinstance(engine, str) else ""
    if normalized == "sglang":
        return sglang_factory()
    if normalized == "transformers":
        return transformers_factory()
    raise ValueError("engine must be 'sglang' or 'transformers'")


__all__ = ["QueryRewriterFactory", "create_query_rewriter"]
