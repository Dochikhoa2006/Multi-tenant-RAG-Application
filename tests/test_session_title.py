from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
import threading

import pytest

from backend.model_config import SESSION_TITLE_GENERATOR
from backend.rag.runtime import RAGRuntime, RerankResult
from backend.rag.session_title import generate_session_title


class FakeLLM:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object], int]] = []

    def complete(self, prompt: str, **kwargs: object) -> str:
        self.calls.append((prompt, dict(kwargs), threading.get_ident()))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response  # type: ignore[return-value]

    def stream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        raise AssertionError("stream must not be used for titles")


class DummyEmbeddings:
    def embed(self, text: str, *, model: str) -> Sequence[float]:
        return [1.0]

    def embed_many(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> Sequence[Sequence[float]]:
        return [[1.0] for _ in texts]


class DummyReranker:
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        **kwargs: object,
    ) -> Sequence[RerankResult]:
        return []


def _runtime(llm: FakeLLM) -> RAGRuntime:
    return RAGRuntime(
        llm,
        DummyEmbeddings(),
        DummyReranker(),
        lambda user_id: object(),
    )


def test_session_title_uses_ordered_p3_context_configured_model_and_worker() -> None:
    llm = FakeLLM("RAG Engineer's Deep-Dive")
    main_thread = threading.get_ident()

    title = asyncio.run(
        generate_session_title(
            ["Question one and answer one", "Question two and answer two"],
            runtime=_runtime(llm),
        )
    )

    assert title == "RAG Engineer's Deep-Dive"
    prompt, options, provider_thread = llm.calls[0]
    assert prompt.index("[Conversation 1]") < prompt.index("[Conversation 2]")
    assert "Question one and answer one" in prompt
    assert options == {"model": SESSION_TITLE_GENERATOR.model}
    assert provider_thread != main_thread


@pytest.mark.parametrize(
    "response",
    [
        "",
        "two words",
        "this title contains far too many words total",
        "RAG Systems Deep Dive!",
        "RAG Systems\nDeep Dive",
        None,
    ],
)
def test_session_title_rejects_malformed_provider_output(response: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(
            generate_session_title(
                ["Completed conversation"],
                runtime=_runtime(FakeLLM(response)),
            )
        )


def test_session_title_propagates_provider_failure() -> None:
    with pytest.raises(RuntimeError, match="title provider failed"):
        asyncio.run(
            generate_session_title(
                ["Completed conversation"],
                runtime=_runtime(FakeLLM(RuntimeError("title provider failed"))),
            )
        )


@pytest.mark.parametrize("conversations", [[], [""], "not a list"])
def test_session_title_validates_conversation_input(conversations: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(
            generate_session_title(
                conversations,  # type: ignore[arg-type]
                runtime=_runtime(FakeLLM("Valid Three Word Title")),
            )
        )
