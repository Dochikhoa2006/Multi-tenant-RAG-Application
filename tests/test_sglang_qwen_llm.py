from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
import json

import httpx
import pytest

from backend.model_config import (
    PRIMARY_GENERATOR,
    QUERY_REWRITER,
    QWEN_SGLANG,
    SESSION_TITLE_GENERATOR,
)
from backend.providers.granite_query_rewriter import RoleRoutingLLMClient
from backend.providers.sglang_qwen_llm import (
    SGLangQwenError,
    SGLangQwenLLMClient,
)


class FakeResponse:
    def __init__(self, payload: object, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    def json(self) -> object:
        return self.payload


class RecordingSyncClient:
    def __init__(
        self,
        payload: object,
        *,
        error: Exception | None = None,
        attempts: list[object | BaseException] | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.attempts = list(attempts) if attempts is not None else None
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append((url, dict(kwargs)))
        if self.attempts is not None:
            index = min(len(self.calls) - 1, len(self.attempts) - 1)
            outcome = self.attempts[index]
            if isinstance(outcome, BaseException):
                raise outcome
            return FakeResponse(outcome)
        if self.error is not None:
            raise self.error
        return FakeResponse(self.payload)

    def close(self) -> None:
        self.closed = True


class FakeStreamResponse:
    def __init__(
        self,
        lines: list[object],
        *,
        error: Exception | None = None,
    ) -> None:
        self.lines = lines
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    async def aiter_lines(self) -> AsyncIterator[object]:
        for line in self.lines:
            if isinstance(line, BaseException):
                raise line
            yield line


class FakeStreamContext:
    def __init__(self, response: FakeStreamResponse) -> None:
        self.response = response

    async def __aenter__(self) -> FakeStreamResponse:
        return self.response

    async def __aexit__(self, *args: object) -> None:
        return None


class RecordingAsyncClient:
    def __init__(
        self,
        lines: list[object],
        *,
        error: Exception | None = None,
        attempts: list[tuple[list[object], Exception | None]] | None = None,
    ) -> None:
        self.lines = lines
        self.error = error
        self.attempts = list(attempts) if attempts is not None else None
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self.closed = False

    def stream(self, method: str, url: str, **kwargs: object) -> FakeStreamContext:
        self.calls.append((method, url, dict(kwargs)))
        if self.attempts is not None:
            index = min(len(self.calls) - 1, len(self.attempts) - 1)
            lines, error = self.attempts[index]
            return FakeStreamContext(FakeStreamResponse(lines, error=error))
        return FakeStreamContext(FakeStreamResponse(self.lines, error=self.error))

    async def aclose(self) -> None:
        self.closed = True


def _completion(
    content: str = "Reliable RAG Grounding",
    *,
    model: str = QWEN_SGLANG.served_model,
    reasoning_content: str | None = None,
    finish_reason: str = "stop",
) -> dict[str, object]:
    return {
        "model": model,
        "choices": [
            {
                "message": {
                    "content": content,
                    "reasoning_content": reasoning_content,
                },
                "finish_reason": finish_reason,
            }
        ],
    }


def _event(
    content: str | None,
    *,
    finish_reason: str | None = None,
    reasoning_content: str | None = None,
    model: str = QWEN_SGLANG.served_model,
) -> str:
    return "data: " + json.dumps(
        {
            "model": model,
            "choices": [
                {
                    "delta": {
                        "content": content,
                        "reasoning_content": reasoning_content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


def _client(
    *,
    completion: object | None = None,
    lines: list[object] | None = None,
) -> tuple[SGLangQwenLLMClient, RecordingSyncClient, RecordingAsyncClient]:
    sync_client = RecordingSyncClient(completion or _completion())
    async_client = RecordingAsyncClient(
        lines
        or [
            _event("first\n"),
            _event("second"),
            _event(None, finish_reason="stop"),
            "data: [DONE]",
        ]
    )
    client = SGLangQwenLLMClient(
        replace(QWEN_SGLANG, api_key="secret"),
        sync_client=sync_client,
        async_client=async_client,
    )
    return client, sync_client, async_client


def test_title_completion_uses_qwen_non_thinking_contract() -> None:
    client, sync_client, _ = _client()

    result = client.complete("Create a title", model=SESSION_TITLE_GENERATOR.model)

    assert result == "Reliable RAG Grounding"
    url, request = sync_client.calls[0]
    assert url == "http://127.0.0.1:30001/v1/chat/completions"
    assert request["headers"] == {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer secret",
    }
    body = request["json"]
    assert isinstance(body, Mapping)
    assert body == {
        "model": QWEN_SGLANG.served_model,
        "messages": [{"role": "user", "content": "Create a title"}],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "max_tokens": 32,
        "n": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_answer_stream_preserves_chunk_order_and_newlines() -> None:
    client, _, async_client = _client()

    async def collect() -> list[str]:
        return [
            item
            async for item in client.stream(
                "Answer prompt",
                model=PRIMARY_GENERATOR.model,
                reasoning="low",
                max_output_tokens=1800,
            )
        ]

    assert asyncio.run(collect()) == ["first\n", "second"]
    _, _, request = async_client.calls[0]
    body = request["json"]
    assert isinstance(body, Mapping)
    assert body["stream"] is True
    assert body["max_tokens"] == 1800
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_stream_rejects_reasoning_and_incomplete_termination() -> None:
    client, _, _ = _client(
        lines=[
            _event(None, reasoning_content="hidden reasoning"),
            _event(None, finish_reason="stop"),
            "data: [DONE]",
        ]
    )

    async def collect() -> list[str]:
        return [
            item
            async for item in client.stream("prompt", model=PRIMARY_GENERATOR.model)
        ]

    with pytest.raises(SGLangQwenError, match="reasoning"):
        asyncio.run(collect())

    incomplete, _, _ = _client(lines=[_event("partial")])

    async def collect_incomplete() -> list[str]:
        return [
            item
            async for item in incomplete.stream(
                "prompt", model=PRIMARY_GENERATOR.model
            )
        ]

    with pytest.raises(SGLangQwenError, match="confirmed completion"):
        asyncio.run(collect_incomplete())


@pytest.mark.parametrize(
    "lines",
    [
        ["not-sse"],
        ["data: {not-json}"],
        [_event("text", model="wrong-model")],
        [_event(None, finish_reason="stop"), "data: [DONE]"],
    ],
)
def test_stream_rejects_malformed_events_and_empty_output(
    lines: list[object],
) -> None:
    client, _, async_client = _client(lines=lines)

    async def collect() -> list[str]:
        return [
            item
            async for item in client.stream("prompt", model=PRIMARY_GENERATOR.model)
        ]

    with pytest.raises(SGLangQwenError):
        asyncio.run(collect())
    assert len(async_client.calls) == 1


def test_stream_preserves_cancellation() -> None:
    client, _, _ = _client(lines=[asyncio.CancelledError()])

    async def collect() -> list[str]:
        return [
            item
            async for item in client.stream("prompt", model=PRIMARY_GENERATOR.model)
        ]

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(collect())


@pytest.mark.parametrize(
    "payload",
    [
        _completion(model="wrong-model"),
        _completion(reasoning_content="hidden"),
        _completion(content=""),
        _completion(finish_reason="length"),
        {"model": QWEN_SGLANG.served_model, "choices": []},
    ],
)
def test_completion_rejects_malformed_or_non_thinking_violations(
    payload: object,
) -> None:
    client, sync_client, _ = _client(completion=payload)
    with pytest.raises(SGLangQwenError):
        client.complete("title", model=SESSION_TITLE_GENERATOR.model)
    assert len(sync_client.calls) == 1


def test_timeout_is_translated_without_leaking_provider_details() -> None:
    detail = "secret upstream endpoint detail"
    error = httpx.ReadTimeout(
        detail,
        request=httpx.Request("POST", "https://qwen.invalid"),
    )
    sync_client = RecordingSyncClient(_completion(), error=error)
    client = SGLangQwenLLMClient(
        QWEN_SGLANG,
        sync_client=sync_client,
        async_client=RecordingAsyncClient([]),
    )

    with pytest.raises(SGLangQwenError, match="timed out") as caught:
        client.complete("title", model=SESSION_TITLE_GENERATOR.model)
    assert len(sync_client.calls) == 2
    assert detail not in str(caught.value)


@pytest.mark.parametrize("status_code", [408, 429, 502, 503, 504])
def test_transient_http_failures_retry_once_for_completion_and_streaming(
    status_code: int,
) -> None:
    request = httpx.Request("POST", "https://qwen.invalid")
    response = httpx.Response(status_code, request=request)
    error = httpx.HTTPStatusError("upstream secret", request=request, response=response)
    sync_client = RecordingSyncClient(_completion(), error=error)
    async_client = RecordingAsyncClient([], error=error)
    client = SGLangQwenLLMClient(
        QWEN_SGLANG,
        sync_client=sync_client,
        async_client=async_client,
    )

    with pytest.raises(SGLangQwenError, match="request failed"):
        client.complete("title", model=SESSION_TITLE_GENERATOR.model)
    assert len(sync_client.calls) == 2

    async def collect() -> list[str]:
        return [
            item
            async for item in client.stream("prompt", model=PRIMARY_GENERATOR.model)
        ]

    with pytest.raises(SGLangQwenError, match="request failed"):
        asyncio.run(collect())
    assert len(async_client.calls) == 2


def test_completion_retries_once_and_recovers_before_output() -> None:
    timeout = httpx.ReadTimeout(
        "provider detail",
        request=httpx.Request("POST", "https://qwen.invalid"),
    )
    sync_client = RecordingSyncClient(
        _completion(),
        attempts=[timeout, _completion("Recovered Title")],
    )
    client = SGLangQwenLLMClient(
        QWEN_SGLANG,
        sync_client=sync_client,
        async_client=RecordingAsyncClient([]),
    )

    assert client.complete("title", model=SESSION_TITLE_GENERATOR.model) == (
        "Recovered Title"
    )
    assert len(sync_client.calls) == 2


def test_stream_retries_transient_failure_only_before_first_content() -> None:
    timeout = httpx.ReadTimeout(
        "provider detail",
        request=httpx.Request("POST", "https://qwen.invalid"),
    )
    async_client = RecordingAsyncClient(
        [],
        attempts=[
            ([], timeout),
            (
                [
                    ": keepalive",
                    "",
                    _event("recovered"),
                    _event(None, finish_reason="stop"),
                    "data: [DONE]",
                ],
                None,
            ),
        ],
    )
    client = SGLangQwenLLMClient(
        QWEN_SGLANG,
        sync_client=RecordingSyncClient(_completion()),
        async_client=async_client,
    )

    async def collect() -> list[str]:
        return [
            item
            async for item in client.stream("prompt", model=PRIMARY_GENERATOR.model)
        ]

    assert asyncio.run(collect()) == ["recovered"]
    assert len(async_client.calls) == 2


def test_stream_never_retries_after_first_content() -> None:
    timeout = httpx.ReadTimeout(
        "provider detail",
        request=httpx.Request("POST", "https://qwen.invalid"),
    )
    async_client = RecordingAsyncClient(
        [],
        attempts=[
            ([_event("visible"), timeout], None),
            (
                [
                    _event("duplicate"),
                    _event(None, finish_reason="stop"),
                    "data: [DONE]",
                ],
                None,
            ),
        ],
    )
    client = SGLangQwenLLMClient(
        QWEN_SGLANG,
        sync_client=RecordingSyncClient(_completion()),
        async_client=async_client,
    )

    async def collect() -> list[str]:
        return [
            item
            async for item in client.stream("prompt", model=PRIMARY_GENERATOR.model)
        ]

    with pytest.raises(SGLangQwenError, match="timed out"):
        asyncio.run(collect())
    assert len(async_client.calls) == 1


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 500])
def test_non_transient_http_failures_are_not_retried(status_code: int) -> None:
    request = httpx.Request("POST", "https://qwen.invalid")
    response = httpx.Response(status_code, request=request)
    error = httpx.HTTPStatusError("provider detail", request=request, response=response)
    sync_client = RecordingSyncClient(_completion(), error=error)
    async_client = RecordingAsyncClient([], error=error)
    client = SGLangQwenLLMClient(
        QWEN_SGLANG,
        sync_client=sync_client,
        async_client=async_client,
    )

    with pytest.raises(SGLangQwenError, match="request failed"):
        client.complete("title", model=SESSION_TITLE_GENERATOR.model)
    assert len(sync_client.calls) == 1

    async def collect() -> list[str]:
        return [
            item
            async for item in client.stream("prompt", model=PRIMARY_GENERATOR.model)
        ]

    with pytest.raises(SGLangQwenError, match="request failed"):
        asyncio.run(collect())
    assert len(async_client.calls) == 1


def test_model_reasoning_and_output_limits_are_validated_before_network() -> None:
    client, sync_client, async_client = _client()

    with pytest.raises(ValueError, match="unexpected model"):
        client.complete("title", model="some-other-model")
    with pytest.raises(ValueError, match="non-thinking"):
        client.complete("title", model=SESSION_TITLE_GENERATOR.model, reasoning="high")
    with pytest.raises(ValueError, match="greater than zero"):
        client.stream("answer", model=PRIMARY_GENERATOR.model, max_output_tokens=0)

    assert sync_client.calls == []
    assert async_client.calls == []


def test_role_router_keeps_granite_rewrite_and_delegates_qwen_roles() -> None:
    client, sync_client, async_client = _client()

    class Granite:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def complete(self, prompt: str, *, model: str, **kwargs: object) -> str:
            self.calls.append((prompt, model))
            return "rewritten"

    granite = Granite()
    router = RoleRoutingLLMClient(client, granite)

    assert router.complete("rewrite", model=QUERY_REWRITER.model) == "rewritten"
    assert router.complete("title", model=SESSION_TITLE_GENERATOR.model) == (
        "Reliable RAG Grounding"
    )

    async def collect() -> list[str]:
        return [
            item
            async for item in router.stream(
                "answer",
                model=PRIMARY_GENERATOR.model,
                reasoning="low",
                max_output_tokens=1800,
            )
        ]

    assert asyncio.run(collect()) == ["first\n", "second"]
    assert granite.calls == [("rewrite", QUERY_REWRITER.model)]
    assert len(sync_client.calls) == 1
    assert len(async_client.calls) == 1
