from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time

import httpx
import pytest

from backend.model_config import (
    GRANITE_QUERY_REWRITE,
    QUERY_REWRITER,
    SGLANG_QUERY_REWRITE,
)
from backend.providers.granite_query_rewriter import RoleRoutingLLMClient
from backend.providers.query_rewriter_factory import create_query_rewriter
from backend.providers.sglang_query_rewriter import (
    SGLangGraniteQueryRewriter,
    SGLangQueryRewriteError,
)
from backend.rag.query_rewrite_contract import ConversationPair, QueryRewritePrompt


class FakeTokenizer:
    def __init__(self, *, characters: bool = False) -> None:
        self.characters = characters
        self.template_calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []
        self.last_count = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> str:
        self.template_calls.append((messages, kwargs))
        return "".join(
            f"<{item['role']}>{item['content']}</{item['role']}>" for item in messages
        )

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.last_count = len(text) if self.characters else max(1, len(text.split()))
        return list(range(self.last_count))


class FakeResponse:
    def __init__(self, payload: object, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    def json(self) -> object:
        return self.payload


class RecordingClient:
    def __init__(
        self,
        tokenizer: FakeTokenizer,
        *,
        continuation: str = 'Standalone Rex question"}',
        response_factory: Callable[[Mapping[str, object]], object] | None = None,
        delay: float = 0.0,
        post_error: Exception | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.continuation = continuation
        self.response_factory = response_factory
        self.delay = delay
        self.post_error = post_error
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append((url, kwargs))
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.post_error is not None:
                raise self.post_error
            body = kwargs["json"]
            assert isinstance(body, Mapping)
            payload = (
                self.response_factory(body)
                if self.response_factory is not None
                else {
                    "model": SGLANG_QUERY_REWRITE.served_model,
                    "choices": [
                        {
                            "message": {"content": self.continuation},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": self.tokenizer.last_count,
                        "completion_tokens": 7,
                        "prompt_tokens_details": {"cached_tokens": 3},
                    },
                }
            )
            return FakeResponse(payload)
        finally:
            with self._guard:
                self.active -= 1

    def close(self) -> None:
        self.closed = True


def _prompt(*pairs: ConversationPair, query: str = "What about it?") -> QueryRewritePrompt:
    return QueryRewritePrompt(
        "legacy P1 text",
        original_query=query,
        conversation_pairs=pairs,
    )


def _adapter(
    *,
    tokenizer: FakeTokenizer | None = None,
    client: RecordingClient | None = None,
    max_input_tokens: int = 2048,
    constrained_output: bool = True,
) -> tuple[SGLangGraniteQueryRewriter, FakeTokenizer, RecordingClient]:
    active_tokenizer = tokenizer or FakeTokenizer()
    active_client = client or RecordingClient(active_tokenizer)
    adapter = SGLangGraniteQueryRewriter(
        granite_config=replace(
            GRANITE_QUERY_REWRITE,
            max_input_tokens=max_input_tokens,
            warmup=False,
        ),
        sglang_config=replace(
            SGLANG_QUERY_REWRITE,
            constrained_output=constrained_output,
            api_key="secret-token",
        ),
        tokenizer=active_tokenizer,
        client=active_client,
    )
    return adapter, active_tokenizer, active_client


def test_sglang_request_preserves_turns_prefill_and_sampling_contract() -> None:
    pairs = (
        ConversationPair("one", "Who is Rex?", "Rex is my dog."),
        ConversationPair("two", "What has fleas?", "Rex has fleas."),
    )
    adapter, tokenizer, client = _adapter()

    assert adapter.complete(_prompt(*pairs), model=QUERY_REWRITER.model) == (
        "Standalone Rex question"
    )

    messages = client.calls[0][1]["json"]["messages"]  # type: ignore[index]
    assert messages == [
        {"role": "user", "content": "Who is Rex?"},
        {"role": "assistant", "content": "Rex is my dog."},
        {"role": "user", "content": "What has fleas?"},
        {"role": "assistant", "content": "Rex has fleas."},
        {"role": "user", "content": "What about it?"},
        {"role": "assistant", "content": GRANITE_QUERY_REWRITE.response_prefill},
    ]
    body = client.calls[0][1]["json"]
    assert body["model"] == SGLANG_QUERY_REWRITE.served_model  # type: ignore[index]
    assert body["temperature"] == 0  # type: ignore[index]
    assert body["max_tokens"] == GRANITE_QUERY_REWRITE.max_new_tokens  # type: ignore[index]
    assert body["n"] == 1  # type: ignore[index]
    assert body["stream"] is False  # type: ignore[index]
    assert body["continue_final_message"] is True  # type: ignore[index]
    assert body["regex"] == SGLANG_QUERY_REWRITE.continuation_regex  # type: ignore[index]
    assert client.calls[0][1]["headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer secret-token",
    }
    assert tokenizer.template_calls[-1][1] == {
        "tokenize": False,
        "add_generation_prompt": False,
        "continue_final_message": True,
    }
    assert adapter.last_diagnostics is not None
    assert adapter.last_diagnostics.cached_prompt_tokens == 3
    assert adapter.last_diagnostics.strict_json is True
    assert adapter.current_diagnostics == adapter.last_diagnostics


def test_whole_tail_pairs_are_removed_before_request() -> None:
    tokenizer = FakeTokenizer(characters=True)
    client = RecordingClient(tokenizer)
    pairs = (
        ConversationPair("one", "first historical question", "first answer"),
        ConversationPair("two", "second historical question", "second answer"),
    )
    adapter, _, _ = _adapter(
        tokenizer=tokenizer,
        client=client,
        max_input_tokens=100,
    )

    adapter.complete(_prompt(*pairs, query="latest"), model=QUERY_REWRITER.model)

    sent_messages = client.calls[0][1]["json"]["messages"]  # type: ignore[index]
    assert sent_messages == [
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": GRANITE_QUERY_REWRITE.response_prefill},
    ]
    assert adapter.last_diagnostics is not None
    assert adapter.last_diagnostics.dropped_conversation_pairs == 2


def test_oversized_latest_query_is_rejected_without_network_call() -> None:
    adapter, _, client = _adapter(max_input_tokens=1)
    with pytest.raises(SGLangQueryRewriteError, match="Latest query"):
        adapter.complete(_prompt(query="too large"), model=QUERY_REWRITER.model)
    assert client.calls == []


def test_sglang_calls_are_not_serialized_by_the_legacy_lock() -> None:
    tokenizer = FakeTokenizer()
    client = RecordingClient(tokenizer, delay=0.04)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(adapter.complete, _prompt(), model=QUERY_REWRITER.model)
            for _ in range(2)
        ]
        assert [future.result() for future in futures] == [
            "Standalone Rex question",
            "Standalone Rex question",
        ]
    assert client.max_active == 2


def test_constraint_can_be_explicitly_disabled_for_compatibility_diagnosis() -> None:
    adapter, _, client = _adapter(constrained_output=False)
    adapter.complete(_prompt(), model=QUERY_REWRITER.model)
    assert "regex" not in client.calls[0][1]["json"]  # type: ignore[operator]


@pytest.mark.parametrize(
    "continuation",
    [
        GRANITE_QUERY_REWRITE.response_prefill + 'duplicate"}',
        "",
        '"}',
    ],
)
def test_ambiguous_or_empty_continuations_are_rejected(continuation: str) -> None:
    tokenizer = FakeTokenizer()
    client = RecordingClient(tokenizer, continuation=continuation)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)
    with pytest.raises(SGLangQueryRewriteError):
        adapter.complete(_prompt(), model=QUERY_REWRITER.model)


def test_malformed_json_fallback_does_not_log_generated_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    continuation = "sensitive malformed continuation"
    tokenizer = FakeTokenizer()
    client = RecordingClient(tokenizer, continuation=continuation)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)
    result = adapter.complete(_prompt(), model=QUERY_REWRITER.model)
    assert result == GRANITE_QUERY_REWRITE.response_prefill + continuation
    assert continuation not in caplog.text


def test_timeout_is_translated_to_safe_granite_error() -> None:
    tokenizer = FakeTokenizer()
    error = httpx.ReadTimeout(
        "provider detail",
        request=httpx.Request("POST", "https://worker.invalid"),
    )
    client = RecordingClient(tokenizer, post_error=error)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)
    with pytest.raises(SGLangQueryRewriteError, match="timed out") as caught:
        adapter.complete(_prompt(), model=QUERY_REWRITER.model)
    assert "provider detail" not in str(caught.value)


def test_tokenizer_server_count_mismatch_blocks_rollout() -> None:
    tokenizer = FakeTokenizer()

    def response(_: Mapping[str, object]) -> object:
        return {
            "model": SGLANG_QUERY_REWRITE.served_model,
            "choices": [
                {"message": {"content": 'query"}'}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": tokenizer.last_count + 1,
                "completion_tokens": 2,
            },
        }

    client = RecordingClient(tokenizer, response_factory=response)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)
    with pytest.raises(SGLangQueryRewriteError, match="token counts"):
        adapter.complete(_prompt(), model=QUERY_REWRITER.model)


@pytest.mark.parametrize(
    ("continuation", "completion_tokens", "message"),
    [
        ('valid"}', 0, "inconsistent"),
        ("", 2, "inconsistent"),
        ('valid"}', GRANITE_QUERY_REWRITE.max_new_tokens + 1, "output-token"),
    ],
)
def test_completion_usage_must_match_content_and_configured_limit(
    continuation: str,
    completion_tokens: int,
    message: str,
) -> None:
    tokenizer = FakeTokenizer()

    def response(_: Mapping[str, object]) -> object:
        return {
            "model": SGLANG_QUERY_REWRITE.served_model,
            "choices": [
                {
                    "message": {"content": continuation},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": tokenizer.last_count,
                "completion_tokens": completion_tokens,
            },
        }

    client = RecordingClient(tokenizer, response_factory=response)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)
    with pytest.raises(SGLangQueryRewriteError, match=message):
        adapter.complete(_prompt(), model=QUERY_REWRITER.model)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"model": SGLANG_QUERY_REWRITE.served_model, "choices": []},
        {
            "model": "wrong-model",
            "choices": [
                {"message": {"content": 'valid"}'}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        {
            "model": SGLANG_QUERY_REWRITE.served_model,
            "choices": [
                {"message": {"content": 42}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ],
)
def test_malformed_server_response_is_rejected(payload: object) -> None:
    tokenizer = FakeTokenizer()
    client = RecordingClient(tokenizer, response_factory=lambda _: payload)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)
    with pytest.raises(SGLangQueryRewriteError):
        adapter.complete(_prompt(), model=QUERY_REWRITER.model)


def test_injected_client_is_not_closed_and_api_key_is_redacted() -> None:
    adapter, _, client = _adapter()
    adapter.close()
    assert client.closed is False
    assert "secret-token" not in repr(adapter.sglang_config)


def test_role_router_delegates_gpt_and_routes_only_granite_to_sglang() -> None:
    class Delegate:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def complete(self, prompt: str, **_: object) -> str:
            self.calls.append(prompt)
            return "title"

        def stream(self, prompt: str, **_: object):
            self.calls.append(prompt)

            async def chunks():
                yield "answer"

            return chunks()

    adapter, _, client = _adapter()
    delegate = Delegate()
    router = RoleRoutingLLMClient(delegate, adapter)
    assert router.complete(_prompt(), model=QUERY_REWRITER.model) == (
        "Standalone Rex question"
    )
    assert router.complete("title", model="GPT-5.1") == "title"
    assert len(client.calls) == 1
    assert delegate.calls == ["title"]


def test_engine_factory_selects_exactly_one_backend_without_fallback() -> None:
    calls: list[str] = []
    sglang = object()
    transformers = object()

    assert create_query_rewriter(
        engine="sglang",
        sglang_factory=lambda: calls.append("sglang") or sglang,  # type: ignore[arg-type]
        transformers_factory=lambda: calls.append("transformers") or transformers,  # type: ignore[arg-type]
    ) is sglang
    assert calls == ["sglang"]

    calls.clear()
    assert create_query_rewriter(
        engine="transformers",
        sglang_factory=lambda: calls.append("sglang") or sglang,  # type: ignore[arg-type]
        transformers_factory=lambda: calls.append("transformers") or transformers,  # type: ignore[arg-type]
    ) is transformers
    assert calls == ["transformers"]

    with pytest.raises(ValueError, match="sglang.*transformers"):
        create_query_rewriter(engine="automatic")
