"""SGLang client for non-thinking Qwen answer and title generation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager
import json
from typing import Any, Protocol

import httpx

from backend.model_config import QWEN_SGLANG, QwenSGLangConfig


class SGLangQwenError(RuntimeError):
    """The Qwen SGLang worker violated the answer/title provider contract."""


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
        try:
            response = self.sync_client.post(
                self._url,
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise SGLangQwenError("Qwen SGLang completion timed out") from exc
        except httpx.HTTPError as exc:
            raise SGLangQwenError("Qwen SGLang completion request failed") from exc
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SGLangQwenError("Qwen SGLang returned malformed JSON") from exc
        return self._validated_response(data)

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
        return self._stream_response(payload)

    async def _stream_response(self, payload: Mapping[str, object]) -> AsyncIterator[str]:
        saw_content = False
        saw_finish = False
        saw_done = False
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
                        raise SGLangQwenError("Qwen SGLang returned a non-text SSE line")
                    line = raw_line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        raise SGLangQwenError("Qwen SGLang returned malformed SSE")
                    data_text = line[5:].strip()
                    if data_text == "[DONE]":
                        if saw_done:
                            raise SGLangQwenError("Qwen SGLang repeated the SSE terminator")
                        saw_done = True
                        continue
                    if saw_done:
                        raise SGLangQwenError("Qwen SGLang sent data after SSE termination")
                    try:
                        event = json.loads(data_text)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise SGLangQwenError("Qwen SGLang returned malformed SSE JSON") from exc
                    content, finish_reason = self._stream_event(event)
                    if content:
                        saw_content = True
                        yield content
                    if finish_reason is not None:
                        if saw_finish or finish_reason not in {"stop", "length"}:
                            raise SGLangQwenError(
                                "Qwen SGLang returned an invalid stream finish reason"
                            )
                        saw_finish = True
        except httpx.TimeoutException as exc:
            raise SGLangQwenError("Qwen SGLang answer stream timed out") from exc
        except httpx.HTTPError as exc:
            raise SGLangQwenError("Qwen SGLang answer stream request failed") from exc
        if not saw_content:
            raise SGLangQwenError("Qwen answer stream completed without content")
        if not saw_finish or not saw_done:
            raise SGLangQwenError("Qwen answer stream ended without confirmed completion")

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
