from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from backend.model_config import QUERY_REWRITER
from backend.prompts import QUERY_REWRITE_PROMPT, SESSION_TITLE_PROMPT
from backend.rag.query_rewrite_contract import QueryRewritePrompt
from backend.rag.query_rewriter import rewrite_query
from backend.rag.runtime import RAGRuntime, RerankResult


class FakeLLM:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete(self, prompt: str, **kwargs: object) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response  # type: ignore[return-value]

    def stream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        raise AssertionError("stream must not be used by query rewriting")


class DummyEmbeddings:
    def embed(self, text: str, *, model: str) -> Sequence[float]:
        return [1.0]

    def embed_many(self, texts: Sequence[str], *, model: str) -> Sequence[Sequence[float]]:
        return [[1.0] for _ in texts]


class DummyReranker:
    def rerank(self, query: str, documents: Sequence[str], **kwargs: object) -> Sequence[RerankResult]:
        return []


def _runtime(llm: FakeLLM) -> RAGRuntime:
    return RAGRuntime(llm, DummyEmbeddings(), DummyReranker(), lambda user_id: object())


def test_rewrite_query_composes_ordered_context_and_uses_model_a() -> None:
    llm = FakeLLM("  Explain vector databases for my system  ")
    conversations = [
        {"raw_text": "Q: What am I building? A: An interview RAG app."},
        {
            "properties": {
                "raw_text": "Q: Which database? A: Weaviate."
            }
        },
    ]

    rewritten = rewrite_query(
        "How does it work?",
        conversations,
        runtime=_runtime(llm),
    )

    assert rewritten == "Explain vector databases for my system"
    assert llm.calls[0]["model"] == QUERY_REWRITER.model
    prompt = llm.calls[0]["prompt"]
    assert isinstance(prompt, QueryRewritePrompt)
    assert "<original_query>\nHow does it work?\n</original_query>" in prompt
    assert prompt.index("[Conversation 1]") < prompt.index("[Conversation 2]")
    assert "An interview RAG app" in prompt
    assert "Weaviate" in prompt


def test_rewrite_query_calls_model_even_without_conversation_history() -> None:
    llm = FakeLLM("Standalone explicit query")

    assert rewrite_query("query", [], runtime=_runtime(llm)) == (
        "Standalone explicit query"
    )
    assert "No prior conversation context." in llm.calls[0]["prompt"]


def test_rewrite_query_attaches_only_canonical_structured_pairs_in_mmr_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = FakeLLM("Standalone explicit query")
    conversations = [
        {
            "object_id": "00000000-0000-0000-0000-000000000001",
            "raw_text": "Question:\nWho is Rex?\n\nAnswer:\nRex is my dog.",
        },
        {
            "object_id": "00000000-0000-0000-0000-000000000002",
            "raw_text": "malformed SECRET conversation content",
        },
        {
            "object_id": "00000000-0000-0000-0000-000000000003",
            "properties": {
                "raw_text": "Question:\nWhat has fleas?\n\nAnswer:\nRex has fleas."
            },
        },
    ]

    rewrite_query("How do I treat them?", conversations, runtime=_runtime(llm))

    prompt = llm.calls[0]["prompt"]
    assert isinstance(prompt, QueryRewritePrompt)
    assert prompt.original_query == "How do I treat them?"
    assert [pair.object_id for pair in prompt.conversation_pairs] == [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000003",
    ]
    warning = next(
        record for record in caplog.records if hasattr(record, "object_ids")
    )
    assert warning.object_ids == ("00000000-0000-0000-0000-000000000002",)
    assert "SECRET" not in caplog.text


def test_rewrite_query_skips_nonmapping_and_missing_text_records() -> None:
    llm = FakeLLM("Standalone explicit query")
    rewrite_query(
        "query",
        [None, {"object_id": "missing", "properties": {}}],  # type: ignore[list-item]
        runtime=_runtime(llm),
    )
    prompt = llm.calls[0]["prompt"]
    assert isinstance(prompt, QueryRewritePrompt)
    assert prompt.conversation_pairs == ()
    assert "No prior conversation context." in prompt


@pytest.mark.parametrize("response", ["", "  ", None, ["query"]])
def test_rewrite_query_rejects_malformed_model_output(response: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        rewrite_query("query", [], runtime=_runtime(FakeLLM(response)))


def test_rewrite_query_propagates_provider_failure() -> None:
    with pytest.raises(RuntimeError, match="provider failed"):
        rewrite_query(
            "query",
            [],
            runtime=_runtime(FakeLLM(RuntimeError("provider failed"))),
        )


def test_prompt_templates_expose_documented_named_fields() -> None:
    assert "{original_query}" in QUERY_REWRITE_PROMPT
    assert "{conversation_context}" in QUERY_REWRITE_PROMPT
    title = SESSION_TITLE_PROMPT.format(conversation_list="Q and A")
    assert "Q and A" in title
    assert "3-6 word" in title
