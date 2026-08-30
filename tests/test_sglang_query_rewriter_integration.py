"""Opt-in contract tests against the deployed Modal SGLang worker."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os

import httpx
import pytest

from backend.model_config import QUERY_REWRITER
from backend.providers.sglang_query_rewriter import SGLangGraniteQueryRewriter
from backend.rag.query_rewrite_contract import ConversationPair, QueryRewritePrompt


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MODAL_SGLANG_TESTS") != "1",
    reason="set RUN_MODAL_SGLANG_TESTS=1 to call the deployed Modal SGLang worker",
)


def _prompt(query: str, *pairs: ConversationPair) -> QueryRewritePrompt:
    return QueryRewritePrompt(
        "provider-compatible P1",
        original_query=query,
        conversation_pairs=pairs,
    )


def test_modal_worker_health_models_prefill_and_concurrency() -> None:
    adapter = SGLangGraniteQueryRewriter()
    root = adapter.sglang_config.base_url.removesuffix("/v1")
    headers = (
        {"Authorization": f"Bearer {adapter.sglang_config.api_key}"}
        if adapter.sglang_config.api_key
        else {}
    )
    with httpx.Client(headers=headers, timeout=10) as client:
        assert client.get(f"{root}/health").is_success
        models = client.get(f"{adapter.sglang_config.base_url}/models")
        models.raise_for_status()
        assert any(
            item.get("id") == adapter.sglang_config.served_model
            for item in models.json()["data"]
        )

    pairs = (
        ConversationPair("one", "Who is Rex?", "Rex is my dog."),
        ConversationPair("two", "What has fleas?", "Rex has fleas."),
    )
    rewritten = adapter.complete(
        _prompt("How do I get rid of them?", *pairs),
        model=QUERY_REWRITER.model,
    )
    assert "flea" in rewritten.lower()
    assert "rex" in rewritten.lower()
    assert adapter.last_diagnostics is not None
    assert adapter.last_diagnostics.strict_json is True

    standalone = adapter.complete(
        _prompt("How should I prepare for a system-design interview?"),
        model=QUERY_REWRITER.model,
    )
    assert standalone.strip()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda index: adapter.complete(
                    _prompt(f"Rewrite standalone question {index}."),
                    model=QUERY_REWRITER.model,
                ),
                range(8),
            )
        )
    assert len(results) == 8
    assert all(result.strip() for result in results)
    adapter.close()

