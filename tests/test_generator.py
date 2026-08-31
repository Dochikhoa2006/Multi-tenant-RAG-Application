from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest
from backend.model_config import PRIMARY_GENERATOR, TOKEN_BUDGETS
from backend.prompts import ANSWER_GENERATION_PROMPT
from backend.rag.generator import generate_answer_stream
from backend.rag.runtime import RAGRuntime, RerankResult


class FakeLLM:
    def __init__(self, chunks: Sequence[object] = (), failure: Exception | None = None) -> None:
        self.chunks = list(chunks)
        self.failure = failure
        self.calls: list[dict[str, object]] = []

    def complete(self, prompt: str, **kwargs: object) -> str:
        raise AssertionError("complete must not be used by answer generation")

    def stream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        self.calls.append({"prompt": prompt, **kwargs})

        async def iterator() -> AsyncIterator[str]:
            if self.failure is not None:
                raise self.failure
            for chunk in self.chunks:
                await asyncio.sleep(0)
                yield chunk  # type: ignore[misc]

        return iterator()


class DummyEmbeddings:
    def embed(self, text: str, *, model: str) -> Sequence[float]:
        return [1.0]

    def embed_many(
        self, texts: Sequence[str], *, model: str
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


class WordTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def _runtime(llm: FakeLLM) -> RAGRuntime:
    return RAGRuntime(
        llm,
        DummyEmbeddings(),
        DummyReranker(),
        lambda user_id: object(),
        tokenizer=WordTokenizer(),
    )


async def _collect(llm: FakeLLM, *, knowledge: list[dict[str, str]] | None = None) -> list[str]:
    return [
        chunk
        async for chunk in generate_answer_stream(
            "Explain retrieval for this architecture",
            knowledge or [{"raw_text": "Knowledge fact one."}],
            [{"raw_text": "Policy guideline one."}],
            runtime=_runtime(llm),
        )
    ]


def test_generate_answer_stream_composes_prompt_and_preserves_token_order() -> None:
    llm = FakeLLM(["First", "", " second", "."])

    chunks = asyncio.run(_collect(llm))

    assert chunks == ["First", " second", "."]
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["model"] == PRIMARY_GENERATOR.model
    assert call["reasoning"] == PRIMARY_GENERATOR.reasoning
    assert call["max_output_tokens"] == PRIMARY_GENERATOR.max_output_tokens
    prompt = call["prompt"]
    assert "<rewritten_query>\nExplain retrieval for this architecture" in prompt
    assert "<knowledge_facts>\nKnowledge fact one." in prompt
    assert "<policy_guidelines>\nPolicy guideline one." in prompt


def test_generate_answer_stream_propagates_provider_failure() -> None:
    with pytest.raises(RuntimeError, match="stream failed"):
        asyncio.run(_collect(FakeLLM(failure=RuntimeError("stream failed"))))


def test_generate_answer_stream_rejects_non_string_chunks() -> None:
    with pytest.raises(TypeError, match="must be strings"):
        asyncio.run(_collect(FakeLLM(["valid", 7])))


def test_generate_answer_stream_rejects_non_async_provider_stream() -> None:
    class InvalidStreamLLM(FakeLLM):
        def stream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
            return ["not async"]  # type: ignore[return-value]

    with pytest.raises(TypeError, match="async iterator"):
        asyncio.run(_collect(InvalidStreamLLM()))


def test_generate_answer_stream_accepts_awaitable_stream_initialization() -> None:
    class AwaitableStreamLLM(FakeLLM):
        async def stream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
            self.calls.append({"prompt": prompt, **kwargs})

            async def iterator() -> AsyncIterator[str]:
                yield "awaited"
                yield " stream"

            return iterator()  # type: ignore[return-value]

    assert asyncio.run(_collect(AwaitableStreamLLM())) == ["awaited", " stream"]


def test_generator_defensively_enforces_whole_item_context_budgets() -> None:
    item_texts = [(f"fact-{index} " * 1700).strip() for index in range(4)]
    llm = FakeLLM(["answer"])

    asyncio.run(
        _collect(
            llm,
            knowledge=[{"raw_text": text} for text in item_texts],
        )
    )

    prompt = llm.calls[0]["prompt"]
    knowledge = prompt.split("<knowledge_facts>\n", 1)[1].split(
        "\n</knowledge_facts>", 1
    )[0]
    tokenizer = WordTokenizer()
    assert len(tokenizer.encode(knowledge)) <= TOKEN_BUDGETS.knowledge_tokens
    included = [text for text in item_texts if text in knowledge]
    assert 0 < len(included) < len(item_texts)
    assert knowledge == "\n\n".join(included)


def test_answer_prompt_exposes_documented_named_fields() -> None:
    assert "{rewritten_query}" in ANSWER_GENERATION_PROMPT
    assert "{knowledge_facts}" in ANSWER_GENERATION_PROMPT
    assert "{policy_guidelines}" in ANSWER_GENERATION_PROMPT


def test_full_rendered_prompt_budget_drops_lower_scored_whole_tail() -> None:
    knowledge = [
        {"raw_text": ("primary " * 1800).strip(), "rerank_score": 0.9},
        {"raw_text": ("lower " * 600).strip(), "rerank_score": 0.1},
    ]
    policy = [
        {"raw_text": ("policy " * 740).strip(), "rerank_score": 0.5},
    ]
    llm = FakeLLM(["answer"])

    async def scenario() -> list[str]:
        return [
            chunk
            async for chunk in generate_answer_stream(
                ("query " * 600).strip(),
                knowledge,
                policy,
                runtime=_runtime(llm),
            )
        ]

    assert asyncio.run(scenario()) == ["answer"]
    prompt = llm.calls[0]["prompt"]
    tokenizer = WordTokenizer()
    assert len(tokenizer.encode(prompt)) <= TOKEN_BUDGETS.total_context_tokens
    assert knowledge[0]["raw_text"] in prompt
    assert knowledge[1]["raw_text"] not in prompt
    assert policy[0]["raw_text"] in prompt


def test_fixed_prompt_and_query_over_total_budget_is_rejected() -> None:
    llm = FakeLLM(["must not stream"])

    async def scenario() -> list[str]:
        return [
            chunk
            async for chunk in generate_answer_stream(
                ("query " * (TOKEN_BUDGETS.total_context_tokens + 1)).strip(),
                [],
                [],
                runtime=_runtime(llm),
            )
        ]

    with pytest.raises(ValueError, match="fixed answer prompt"):
        asyncio.run(scenario())
    assert llm.calls == []
