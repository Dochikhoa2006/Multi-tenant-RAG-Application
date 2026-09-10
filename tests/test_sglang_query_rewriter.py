from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time
from uuid import uuid4

import httpx
import pytest

from backend.model_config import (
    GRANITE_QUERY_REWRITE,
    PRIMARY_GENERATOR,
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
from backend.wizard.diagnostics import (
    DiagnosticTraceRegistry,
    activate_operation,
    install_registry,
    uninstall_registry,
)


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


def test_deep_trace_observes_the_same_granite_request_without_raw_context() -> None:
    pairs = (
        ConversationPair("one", "Who is Rex?", "Rex is my dog."),
        ConversationPair("two", "What has fleas?", "Rex has fleas."),
    )
    adapter, _, client = _adapter()
    registry = DiagnosticTraceRegistry("diagnostic_user")
    session_id = str(uuid4())
    operation_id = str(uuid4())
    registry.start("diagnostic_user", session_id, "run")
    handle = registry.begin_operation(
        user_id="diagnostic_user",
        session_id=session_id,
        operation_id=operation_id,
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    )
    assert handle is not None
    install_registry(registry)
    try:
        with activate_operation(handle):
            rewritten = adapter.complete(
                _prompt(*pairs), model=QUERY_REWRITER.model
            )
    finally:
        uninstall_registry(registry)

    operation = registry.snapshot(
        "diagnostic_user", session_id, operation_id
    )["operations"][0]
    assert rewritten == "Standalone Rex question"
    assert len(client.calls) == 1
    assert operation["stages"]["chat.granite_budgeting"]["call_count"] == 1
    assert operation["stages"]["chat.granite_http"]["call_count"] == 1
    assert operation["counts"]["granite_attempt_count"] == 1
    assert operation["counts"]["granite_final_prompt_tokens"] > 0
    assert operation["counts"]["granite_final_completion_tokens"] == 7
    assert operation["flags"]["granite_strict_json"] is True
    assert operation["flags"]["granite_repair_applied"] is False
    assert operation["texts"]["granite_rewritten_query"] == rewritten
    assert operation["samples"]["granite_retained_pair_ids"]["items"] == [
        "one",
        "two",
    ]
    serialized = str(operation)
    assert "Who is Rex?" not in serialized
    assert "Rex is my dog." not in serialized
    assert set(operation["digests"]) == {
        "granite_conversation_context_sha256",
        "granite_request_messages_sha256",
        "granite_rewritten_query_sha256",
    }


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
        '"}',
        'query","extra":"field"}',
        'query"} trailing prose',
    ],
)
def test_unrepairable_or_empty_continuations_retry_once_then_fail(
    continuation: str,
) -> None:
    tokenizer = FakeTokenizer()
    client = RecordingClient(tokenizer, continuation=continuation)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)
    with pytest.raises(SGLangQueryRewriteError):
        adapter.complete(_prompt(), model=QUERY_REWRITER.model)
    assert len(client.calls) == 2


def test_empty_content_usage_violation_is_not_retried() -> None:
    tokenizer = FakeTokenizer()
    client = RecordingClient(tokenizer, continuation="")
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)

    with pytest.raises(SGLangQueryRewriteError, match="inconsistent"):
        adapter.complete(_prompt(), model=QUERY_REWRITER.model)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("continuation", "expected"),
    [
        ("sensitive malformed continuation", "sensitive malformed continuation"),
        ('Missing brace"', "Missing brace"),
        ('{"rewritten_question":"Repeated full object"}', "Repeated full object"),
    ],
)
def test_deterministic_structural_repair_does_not_log_generated_text(
    continuation: str,
    expected: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tokenizer = FakeTokenizer()
    client = RecordingClient(tokenizer, continuation=continuation)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)
    result = adapter.complete(_prompt(), model=QUERY_REWRITER.model)
    assert result == expected
    assert adapter.last_diagnostics is not None
    assert adapter.last_diagnostics.strict_json is False
    assert len(client.calls) == 1
    assert continuation not in caplog.text


def test_unrepairable_format_can_succeed_on_the_single_retry() -> None:
    tokenizer = FakeTokenizer()

    def response(_: Mapping[str, object]) -> object:
        continuation = 'query",}' if len(client.calls) == 1 else 'Recovered query"}'
        return {
            "model": SGLANG_QUERY_REWRITE.served_model,
            "choices": [
                {"message": {"content": continuation}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": tokenizer.last_count,
                "completion_tokens": 3,
            },
        }

    client = RecordingClient(tokenizer, response_factory=response)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)

    assert adapter.complete(_prompt(), model=QUERY_REWRITER.model) == "Recovered query"
    assert len(client.calls) == 2


def test_deep_trace_accumulates_existing_granite_format_retry() -> None:
    tokenizer = FakeTokenizer()

    def response(_: Mapping[str, object]) -> object:
        continuation = 'query",}' if len(client.calls) == 1 else 'Recovered query"}'
        return {
            "model": SGLANG_QUERY_REWRITE.served_model,
            "choices": [
                {"message": {"content": continuation}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": tokenizer.last_count,
                "completion_tokens": 3,
            },
        }

    client = RecordingClient(tokenizer, response_factory=response)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)
    registry = DiagnosticTraceRegistry("diagnostic_user")
    session_id = str(uuid4())
    operation_id = str(uuid4())
    registry.start("diagnostic_user", session_id, "run")
    handle = registry.begin_operation(
        user_id="diagnostic_user",
        session_id=session_id,
        operation_id=operation_id,
        kind="chat_query",
        collection_type="conversations",
        wizard_id=str(uuid4()),
    )
    assert handle is not None
    install_registry(registry)
    try:
        with activate_operation(handle):
            assert adapter.complete(_prompt(), model=QUERY_REWRITER.model) == (
                "Recovered query"
            )
    finally:
        uninstall_registry(registry)

    operation = registry.snapshot(
        "diagnostic_user", session_id, operation_id
    )["operations"][0]
    assert len(client.calls) == 2
    assert operation["stages"]["chat.granite_http"]["call_count"] == 2
    assert operation["counts"]["granite_attempt_count"] == 2
    assert operation["counts"]["granite_format_failure_count"] == 1
    assert operation["counts"]["granite_format_retry_count"] == 1
    assert operation["counts"]["granite_usage_response_count"] == 2


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
    assert len(client.calls) == 2
    assert "provider detail" not in str(caught.value)


@pytest.mark.parametrize("status_code", [408, 429, 502, 503, 504])
def test_transient_http_statuses_retry_once(status_code: int) -> None:
    tokenizer = FakeTokenizer()
    request = httpx.Request("POST", "https://worker.invalid")
    response = httpx.Response(status_code, request=request)
    error = httpx.HTTPStatusError("provider detail", request=request, response=response)
    client = RecordingClient(tokenizer, post_error=error)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)

    with pytest.raises(SGLangQueryRewriteError, match="transiently"):
        adapter.complete(_prompt(), model=QUERY_REWRITER.model)
    assert len(client.calls) == 2


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 500])
def test_non_transient_http_statuses_are_not_retried(status_code: int) -> None:
    tokenizer = FakeTokenizer()
    request = httpx.Request("POST", "https://worker.invalid")
    response = httpx.Response(status_code, request=request)
    error = httpx.HTTPStatusError("provider detail", request=request, response=response)
    client = RecordingClient(tokenizer, post_error=error)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)

    with pytest.raises(SGLangQueryRewriteError, match="request failed"):
        adapter.complete(_prompt(), model=QUERY_REWRITER.model)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "details",
    ["malformed", {"cached_tokens": "three"}, {"cached_tokens": -1}],
)
def test_optional_cache_diagnostics_do_not_reject_valid_rewrite(
    details: object,
) -> None:
    tokenizer = FakeTokenizer()

    def response(_: Mapping[str, object]) -> object:
        return {
            "model": SGLANG_QUERY_REWRITE.served_model,
            "choices": [
                {
                    "message": {"content": 'Standalone query"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": tokenizer.last_count,
                "completion_tokens": 3,
                "prompt_tokens_details": details,
            },
        }

    client = RecordingClient(tokenizer, response_factory=response)
    adapter, _, _ = _adapter(tokenizer=tokenizer, client=client)

    assert adapter.complete(_prompt(), model=QUERY_REWRITER.model) == "Standalone query"
    assert adapter.last_diagnostics is not None
    assert adapter.last_diagnostics.cached_prompt_tokens is None
    assert len(client.calls) == 1


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
    assert router.complete("title", model=PRIMARY_GENERATOR.model) == "title"
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
