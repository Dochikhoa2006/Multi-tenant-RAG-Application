"""SGLang client for non-thinking Qwen answer and title generation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager
import hashlib
import json
from time import perf_counter
from typing import Any, Protocol

import httpx

from backend.model_config import QWEN_SGLANG, QwenSGLangConfig
from backend.wizard.diagnostics import (
    add_count,
    observe_elapsed,
    observe_stage,
    set_digest,
    set_flag,
    set_framed_digest,
    set_text,
    title_provider_trace_active,
    trace_operation_active,
    trace_utf8_bytes,
)


class SGLangQwenError(RuntimeError):
    """The Qwen SGLang worker violated the answer/title provider contract."""


_TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 502, 503, 504})


def _is_transient_http_error(error: httpx.HTTPError) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    return (
        isinstance(error, httpx.HTTPStatusError)
        and error.response.status_code in _TRANSIENT_HTTP_STATUSES
    )


def _provider_error_message(error: httpx.HTTPError, operation: str) -> str:
    if isinstance(error, httpx.TimeoutException):
        return f"Qwen SGLang {operation} timed out"
    return f"Qwen SGLang {operation} request failed"


class QwenSyncTransport(Protocol):
    def post(self, url: str, **kwargs: object) -> Any: ...

    def close(self) -> None: ...


class QwenAsyncTransport(Protocol):
    def stream(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> AbstractAsyncContextManager[Any]: ...

    async def aclose(self) -> None: ...


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _output_limit(value: int | None, default: int) -> int:
    limit = default if value is None else value
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("max_output_tokens must be an integer")
    if limit <= 0:
        raise ValueError("max_output_tokens must be greater than zero")
    return limit


def _nonnegative_usage_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_usage(value: object) -> tuple[int, int, int, int | None] | None:
    if not isinstance(value, Mapping):
        return None
    prompt_tokens = _nonnegative_usage_int(value.get("prompt_tokens"))
    completion_tokens = _nonnegative_usage_int(value.get("completion_tokens"))
    total_tokens = _nonnegative_usage_int(value.get("total_tokens"))
    if (
        prompt_tokens is None
        or completion_tokens is None
        or total_tokens is None
        or total_tokens != prompt_tokens + completion_tokens
    ):
        return None
    cached_tokens: int | None = None
    details = value.get("prompt_tokens_details")
    if details is not None:
        if not isinstance(details, Mapping):
            return None
        cached_tokens = _nonnegative_usage_int(details.get("cached_tokens"))
        if cached_tokens is None or cached_tokens > prompt_tokens:
            return None
    return prompt_tokens, completion_tokens, total_tokens, cached_tokens


class SGLangQwenLLMClient:
    """One pooled Qwen client implementing the existing ``LLMClient`` seam."""

    def __init__(
        self,
        config: QwenSGLangConfig = QWEN_SGLANG,
        *,
        sync_client: QwenSyncTransport | None = None,
        async_client: QwenAsyncTransport | None = None,
    ) -> None:
        if not isinstance(config, QwenSGLangConfig):
            raise TypeError("config must be a QwenSGLangConfig")
        timeout = httpx.Timeout(
            connect=config.connect_timeout_seconds,
            read=config.read_timeout_seconds,
            write=config.read_timeout_seconds,
            pool=config.connect_timeout_seconds,
        )
        limits = httpx.Limits(
            max_connections=config.max_connections,
            max_keepalive_connections=config.max_connections,
        )
        self.config = config
        self._owns_sync_client = sync_client is None
        self._owns_async_client = async_client is None
        self.sync_client = sync_client or httpx.Client(timeout=timeout, limits=limits)
        self.async_client = async_client or httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
        )

    @property
    def _url(self) -> str:
        return f"{self.config.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _payload(
        self,
        prompt: str,
        *,
        model: str,
        reasoning: str | None,
        max_output_tokens: int | None,
        stream: bool,
    ) -> dict[str, object]:
        text = _required_text(prompt, "prompt")
        if model != self.config.served_model:
            raise ValueError("Qwen adapter received an unexpected model identifier")
        if reasoning not in (None, "low"):
            raise ValueError("Qwen answer/title generation supports only non-thinking mode")
        return {
            "model": self.config.served_model,
            "messages": [{"role": "user", "content": text}],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "presence_penalty": self.config.presence_penalty,
            "max_tokens": _output_limit(
                max_output_tokens,
                self.config.title_max_output_tokens,
            ),
            "n": 1,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def _validated_response(self, value: object) -> str:
        if not isinstance(value, Mapping):
            raise SGLangQwenError("Qwen SGLang returned a malformed response")
        if value.get("model") != self.config.served_model:
            raise SGLangQwenError("Qwen SGLang returned an unexpected model identity")
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise SGLangQwenError("Qwen SGLang returned malformed completion choices")
        choice = choices[0]
        if not isinstance(choice, Mapping) or choice.get("finish_reason") != "stop":
            raise SGLangQwenError("Qwen title completion did not terminate normally")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise SGLangQwenError("Qwen SGLang returned a malformed completion message")
        if message.get("reasoning_content") not in (None, ""):
            raise SGLangQwenError("Qwen returned reasoning in non-thinking mode")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise SGLangQwenError("Qwen returned an empty completion")
        return content.strip()

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        reasoning: str | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        payload = self._payload(
            prompt,
            model=model,
            reasoning=reasoning,
            max_output_tokens=max_output_tokens,
            stream=False,
        )
        title_trace = title_provider_trace_active()
        if title_trace:
            for empty_count in (
                "title_qwen_attempt_count",
                "title_qwen_transient_failure_count",
                "title_qwen_retry_count",
                "title_qwen_usage_response_count",
            ):
                add_count(empty_count, 0)
            add_count("title_qwen_request_message_count", len(payload["messages"]))
            add_count("title_qwen_request_max_tokens", int(payload["max_tokens"]))
            set_text("title_qwen_requested_model", str(payload["model"]))
            prompt_bytes = trace_utf8_bytes(prompt)
            if prompt_bytes is not None:
                add_count("title_qwen_prompt_utf8_bytes", len(prompt_bytes))
                set_framed_digest(
                    "title_qwen_prompt_sha256",
                    "chat-title-prompt-v1",
                    (prompt_bytes,),
                )
        with observe_stage("chat.title_qwen_total"):
            for attempt in range(2):
                if title_trace:
                    add_count("title_qwen_attempt_count", 1)
                try:
                    with observe_stage("chat.title_qwen_http"):
                        response = self.sync_client.post(
                            self._url,
                            json=payload,
                            headers=self._headers(),
                        )
                        response.raise_for_status()
                        data = response.json()
                except httpx.HTTPError as exc:
                    transient = (
                        _is_transient_http_error(exc)
                        if attempt == 0 or title_trace
                        else False
                    )
                    if title_trace and transient:
                        add_count("title_qwen_transient_failure_count", 1)
                    if attempt == 0 and transient:
                        if title_trace:
                            add_count("title_qwen_retry_count", 1)
                        continue
                    raise SGLangQwenError(
                        _provider_error_message(exc, "completion")
                    ) from exc
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise SGLangQwenError(
                        "Qwen SGLang returned malformed JSON"
                    ) from exc
                result = self._validated_response(data)
                if title_trace:
                    usage_present = isinstance(data, Mapping) and "usage" in data
                    usage = _optional_usage(data.get("usage")) if usage_present else None
                    set_flag("title_qwen_usage_available", usage_present)
                    set_flag(
                        "title_qwen_usage_valid",
                        not usage_present or usage is not None,
                    )
                    if usage_present:
                        add_count("title_qwen_usage_response_count", 1)
                    if usage is not None:
                        prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
                            usage
                        )
                        add_count("title_qwen_prompt_tokens", prompt_tokens)
                        add_count("title_qwen_completion_tokens", completion_tokens)
                        add_count("title_qwen_total_tokens", total_tokens)
                        if cached_tokens is None:
                            set_flag("title_qwen_cached_tokens_available", False)
                        else:
                            set_flag("title_qwen_cached_tokens_available", True)
                            add_count("title_qwen_cached_prompt_tokens", cached_tokens)
                    else:
                        set_flag("title_qwen_cached_tokens_available", False)
                    set_text("title_qwen_observed_model", str(data.get("model")))
                    choices = data.get("choices")
                    if isinstance(choices, list) and choices and isinstance(
                        choices[0], Mapping
                    ):
                        set_text(
                            "title_qwen_finish_reason",
                            str(choices[0].get("finish_reason")),
                        )
                    result_bytes = trace_utf8_bytes(result)
                    if result_bytes is not None:
                        add_count("title_qwen_result_utf8_bytes", len(result_bytes))
                        set_framed_digest(
                            "title_qwen_result_sha256",
                            "chat-title-provider-result-v1",
                            (result_bytes,),
                        )
                return result
        raise AssertionError("Qwen completion retry loop exhausted unexpectedly")

    def stream(
        self,
        prompt: str,
        *,
        model: str,
        reasoning: str | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(
            prompt,
            model=model,
            reasoning=reasoning,
            max_output_tokens=max_output_tokens,
            stream=True,
        )
        if trace_operation_active():
            messages = payload["messages"]
            if isinstance(messages, list):
                add_count("qwen_request_message_count", len(messages))
            add_count("qwen_request_max_tokens", int(payload["max_tokens"]))
            set_text("qwen_requested_model", str(payload["model"]))
            prompt_bytes = trace_utf8_bytes(prompt)
            if prompt_bytes is not None:
                add_count("qwen_request_prompt_utf8_bytes", len(prompt_bytes))
                set_framed_digest(
                    "qwen_request_prompt_sha256",
                    "chat-qwen-prompt-v1",
                    (prompt_bytes,),
                )
            set_framed_digest(
                "qwen_request_messages_sha256",
                "chat-qwen-request-messages-v1",
                (
                    value
                    for message in messages
                    for value in (
                        str(message["role"]).encode("utf-8"),
                        str(message["content"]).encode("utf-8"),
                    )
                ),
            )
        return self._stream_response(payload)

    async def _stream_response(self, payload: Mapping[str, object]) -> AsyncIterator[str]:
        trace_active = trace_operation_active()
        trace_started = perf_counter() if trace_active else 0.0
        answer_digest = hashlib.sha256() if trace_active else None
        if answer_digest is not None:
            answer_digest.update(b"chat-qwen-answer-chunks-v1")
        answer_chunk_count = 0
        answer_utf8_bytes = 0
        first_content_at: float | None = None
        final_finish_reason: str | None = None
        observed_model: str | None = None
        usage_event_count = 0
        final_usage: tuple[int, int, int, int | None] | None = None
        usage_valid = True
        if trace_active:
            for empty_count in (
                "qwen_attempt_count",
                "qwen_transient_failure_count",
                "qwen_retry_count",
                "qwen_usage_event_count",
            ):
                add_count(empty_count, 0)
        for attempt in range(2):
            saw_content = False
            saw_finish = False
            saw_done = False
            if trace_active:
                add_count("qwen_attempt_count", 1)
            attempt_started = perf_counter() if trace_active else 0.0
            try:
                async with self.async_client.stream(
                    "POST",
                    self._url,
                    json=dict(payload),
                    headers=self._headers(),
                ) as response:
                    response.raise_for_status()
                    async for raw_line in response.aiter_lines():
                        if not isinstance(raw_line, str):
                            raise SGLangQwenError(
                                "Qwen SGLang returned a non-text SSE line"
                            )
                        line = raw_line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            raise SGLangQwenError("Qwen SGLang returned malformed SSE")
                        data_text = line[5:].strip()
                        if data_text == "[DONE]":
                            if saw_done:
                                raise SGLangQwenError(
                                    "Qwen SGLang repeated the SSE terminator"
                                )
                            saw_done = True
                            continue
                        if saw_done:
                            raise SGLangQwenError(
                                "Qwen SGLang sent data after SSE termination"
                            )
                        try:
                            event = json.loads(data_text)
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise SGLangQwenError(
                                "Qwen SGLang returned malformed SSE JSON"
                            ) from exc
                        content, finish_reason = self._stream_event(event)
                        if trace_active and isinstance(event, Mapping):
                            model_value = event.get("model")
                            if isinstance(model_value, str):
                                observed_model = model_value
                            if "usage" in event:
                                usage_event_count += 1
                                parsed_usage = _optional_usage(event.get("usage"))
                                if parsed_usage is None:
                                    usage_valid = False
                                else:
                                    final_usage = parsed_usage
                        if content:
                            saw_content = True
                            if trace_active:
                                now = perf_counter()
                                if first_content_at is None:
                                    first_content_at = now
                                    observe_elapsed(
                                        "chat.qwen_ttft",
                                        (now - trace_started) * 1000.0,
                                    )
                                encoded_content = trace_utf8_bytes(content)
                                if encoded_content is not None:
                                    answer_chunk_count += 1
                                    answer_utf8_bytes += len(encoded_content)
                                if (
                                    answer_digest is not None
                                    and encoded_content is not None
                                ):
                                    answer_digest.update(
                                        len(encoded_content).to_bytes(8, "big")
                                    )
                                    answer_digest.update(encoded_content)
                            yield content
                        if finish_reason is not None:
                            if saw_finish or finish_reason not in {"stop", "length"}:
                                raise SGLangQwenError(
                                    "Qwen SGLang returned an invalid stream finish reason"
                                )
                            saw_finish = True
                            final_finish_reason = finish_reason
            except httpx.HTTPError as exc:
                if trace_active:
                    observe_elapsed(
                        "chat.qwen_http",
                        (perf_counter() - attempt_started) * 1000.0,
                        failed=True,
                    )
                    add_count("qwen_transient_failure_count", 1)
                if attempt == 0 and not saw_content and _is_transient_http_error(exc):
                    if trace_active:
                        add_count("qwen_retry_count", 1)
                    continue
                raise SGLangQwenError(
                    _provider_error_message(exc, "answer stream")
                ) from exc
            if trace_active:
                observe_elapsed(
                    "chat.qwen_http",
                    (perf_counter() - attempt_started) * 1000.0,
                )
            if not saw_content:
                raise SGLangQwenError("Qwen answer stream completed without content")
            if not saw_finish or not saw_done:
                raise SGLangQwenError(
                    "Qwen answer stream ended without confirmed completion"
                )
            if trace_active:
                completed_at = perf_counter()
                if first_content_at is not None:
                    observe_elapsed(
                        "chat.qwen_generation",
                        (completed_at - first_content_at) * 1000.0,
                    )
                observe_elapsed(
                    "chat.qwen_stream_total",
                    (completed_at - trace_started) * 1000.0,
                )
                add_count("qwen_answer_chunk_count", answer_chunk_count)
                add_count("qwen_answer_utf8_bytes", answer_utf8_bytes)
                add_count("qwen_usage_event_count", usage_event_count)
                set_flag("qwen_usage_available", usage_event_count > 0)
                set_flag("qwen_usage_valid", usage_valid)
                set_flag("qwen_finish_seen", saw_finish)
                set_flag("qwen_done_seen", saw_done)
                if observed_model is not None:
                    set_text("qwen_observed_model", observed_model)
                if final_finish_reason is not None:
                    set_text("qwen_finish_reason", final_finish_reason)
                if final_usage is not None:
                    prompt_tokens, completion_tokens, total_tokens, cached_tokens = (
                        final_usage
                    )
                    add_count("qwen_final_prompt_tokens", prompt_tokens)
                    add_count("qwen_final_completion_tokens", completion_tokens)
                    add_count("qwen_final_total_tokens", total_tokens)
                    if cached_tokens is None:
                        set_flag("qwen_cached_tokens_available", False)
                    else:
                        set_flag("qwen_cached_tokens_available", True)
                        add_count("qwen_final_cached_prompt_tokens", cached_tokens)
                else:
                    set_flag("qwen_cached_tokens_available", False)
                if answer_digest is not None:
                    set_digest(
                        "qwen_answer_chunks_sha256", answer_digest.hexdigest()
                    )
            return
        raise AssertionError("Qwen stream retry loop exhausted unexpectedly")

    def _stream_event(self, value: object) -> tuple[str, object]:
        if not isinstance(value, Mapping):
            raise SGLangQwenError("Qwen SGLang returned a malformed stream event")
        if value.get("model") != self.config.served_model:
            raise SGLangQwenError("Qwen SGLang returned an unexpected model identity")
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise SGLangQwenError("Qwen SGLang returned malformed stream choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise SGLangQwenError("Qwen SGLang returned a malformed stream choice")
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            raise SGLangQwenError("Qwen SGLang returned a malformed stream delta")
        if delta.get("reasoning_content") not in (None, ""):
            raise SGLangQwenError("Qwen returned reasoning in non-thinking mode")
        content = delta.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise SGLangQwenError("Qwen SGLang returned non-text stream content")
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise SGLangQwenError("Qwen SGLang returned a malformed finish reason")
        return content, finish_reason

    def close(self) -> None:
        if self._owns_sync_client:
            self.sync_client.close()

    async def aclose(self) -> None:
        if self._owns_async_client:
            await self.async_client.aclose()


__all__ = [
    "QwenAsyncTransport",
    "QwenSyncTransport",
    "SGLangQwenError",
    "SGLangQwenLLMClient",
]
