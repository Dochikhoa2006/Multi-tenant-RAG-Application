"""Opt-in local checkpoint smoke tests; excluded from normal pytest execution."""

from __future__ import annotations

import os
from difflib import SequenceMatcher

import pytest

from backend.model_config import QUERY_REWRITER
from backend.providers.granite_query_rewriter import GraniteQueryRewriter
from backend.rag.query_rewrite_contract import ConversationPair, QueryRewritePrompt


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LOCAL_GRANITE_TESTS") != "1",
    reason="set RUN_LOCAL_GRANITE_TESTS=1 to load the provisioned Granite checkpoint",
)


def _rewrite(adapter: GraniteQueryRewriter, query: str, *pairs: ConversationPair) -> str:
    return adapter.complete(
        QueryRewritePrompt(
            "opt-in local smoke test",
            original_query=query,
            conversation_pairs=pairs,
        ),
        model=QUERY_REWRITER.model,
    )


def test_local_granite_rex_coreference_and_standalone_query() -> None:
    adapter = GraniteQueryRewriter()
    rewritten = _rewrite(
        adapter,
        "But is he more likely to get fleas because of that?",
        ConversationPair(
            "rex",
            "I have two pets, a dog named Rex and a cat named Lucy. Rex spends a lot "
            "of time in the backyard and outdoors, and Lucy is always inside.",
            "Sounds good! Rex must love exploring outside, while Lucy probably enjoys "
            "her cozy indoor life.",
        ),
    )
    lowered = rewritten.lower()
    assert "rex" in lowered
    assert "fleas" in lowered
    assert "out" in lowered

    standalone = "What is hybrid search in a retrieval-augmented generation system?"
    unchanged = _rewrite(adapter, standalone)
    assert SequenceMatcher(None, unchanged.lower(), standalone.lower()).ratio() >= 0.9
