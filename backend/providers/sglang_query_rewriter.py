"""SGLang/OpenAI-compatible adapter for the merged Granite query rewriter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from pathlib import Path
import threading
from time import perf_counter
from typing import Any, Protocol

import httpx

from backend.model_config import (
    GRANITE_QUERY_REWRITE,
    QUERY_REWRITER,
    SGLANG_QUERY_REWRITE,
    GraniteQueryRewriteConfig,
    SGLangQueryRewriteConfig,
)
from backend.providers.granite_query_rewriter import (
    GraniteCheckpointError,
    GraniteInferenceError,
    GraniteRewriteFormatError,
    GraniteRewriteDiagnostics,
    parse_granite_rewrite,
    validate_granite_checkpoint,
)
from backend.rag.query_rewrite_contract import QueryRewritePrompt
from backend.wizard.diagnostics import (
    TRACE_SAMPLE_LIMIT,
    add_count,
    capture_evaluation_rewrite,
    observe_elapsed,
    observe_stage,
    set_flag,
    set_framed_digest,
    set_sample,
    set_text,
    trace_operation_active,
)


logger = logging.getLogger(__name__)


class SGLangTransport(Protocol):
    """The subset of ``httpx.Client`` used by the adapter."""

    def post(self, url: str, **kwargs: object) -> Any: ...

    def close(self) -> None: ...


class SGLangQueryRewriteError(GraniteInferenceError):
    """The SGLang worker could not produce a trustworthy rewrite response."""


class _TransientSGLangQueryRewriteError(SGLangQueryRewriteError):
    """A request failed before any rewrite was exposed and may be retried once."""


_TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 502, 503, 504})


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_path(config: GraniteQueryRewriteConfig) -> Path:
    configured = Path(config.model_path).expanduser()
    return (
        configured.resolve()
        if configured.is_absolute()
        else (_project_root() / configured).resolve()
    )


def _load_tokenizer(config: GraniteQueryRewriteConfig) -> Any:
    model_path = _model_path(config)
    validate_granite_checkpoint(model_path)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise GraniteCheckpointError(
            "Transformers is required to load the Granite tokenizer assets"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise GraniteCheckpointError("Granite tokenizer assets could not be loaded") from exc


def _token_count(tokenizer: Any, rendered: str) -> int:
    try:
        if callable(getattr(tokenizer, "encode", None)):
            encoded = tokenizer.encode(rendered, add_special_tokens=False)
        else:
            result = tokenizer(rendered, add_special_tokens=False)
            if not isinstance(result, Mapping) or "input_ids" not in result:
                raise TypeError
            encoded = result["input_ids"]
        shape = getattr(encoded, "shape", None)
        if shape is not None:
            count = int(shape[-1])
        elif isinstance(encoded, Sequence) and not isinstance(encoded, (str, bytes)):
            if encoded and isinstance(encoded[0], Sequence):
                count = len(encoded[0])
            else:
                count = len(encoded)
        else:
            raise TypeError
    except (IndexError, TypeError, ValueError) as exc:
        raise SGLangQueryRewriteError(
            "Granite tokenizer returned malformed input IDs"
        ) from exc
    if count <= 0:
        raise SGLangQueryRewriteError("Granite tokenizer returned no input IDs")
    return count


class SGLangGraniteQueryRewriter:
    """Thread-safe HTTP client that leaves batching and scheduling to SGLang."""

    def __init__(
        self,
        *,
        granite_config: GraniteQueryRewriteConfig = GRANITE_QUERY_REWRITE,
        sglang_config: SGLangQueryRewriteConfig = SGLANG_QUERY_REWRITE,
        tokenizer: Any | None = None,
        client: SGLangTransport | None = None,
    ) -> None:
        self.granite_config = granite_config
        self.sglang_config = sglang_config
        self.tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(
            granite_config
        )
        self._owns_client = client is None
        self.client = client if client is not None else httpx.Client(
            timeout=httpx.Timeout(
                connect=sglang_config.connect_timeout_seconds,
                read=sglang_config.read_timeout_seconds,
                write=sglang_config.read_timeout_seconds,
                pool=sglang_config.connect_timeout_seconds,
            ),
            limits=httpx.Limits(
                max_connections=sglang_config.max_connections,
                max_keepalive_connections=sglang_config.max_connections,
            ),
            headers={"Accept": "application/json"},
        )
        self.last_diagnostics: GraniteRewriteDiagnostics | None = None
        self._thread_diagnostics = threading.local()

class _TransientSGLangQueryRewriteError(SGLangQueryRewriteError):
    """A request failed before any rewrite was exposed and may be retried once."""

    @property
    def current_diagnostics(self) -> GraniteRewriteDiagnostics | None:
        """Return diagnostics for the most recent call on the current thread."""

        return getattr(self._thread_diagnostics, "value", None)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    @staticmethod
    def _conversation_messages(
        prompt: QueryRewritePrompt,
        pair_count: int,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for pair in prompt.conversation_pairs[:pair_count]:
            messages.extend(
                (
                    {"role": "user", "content": pair.question},
                    {"role": "assistant", "content": pair.answer},
                )
            )
        messages.append({"role": "user", "content": prompt.original_query})
        return messages

    def _render_prefilled_messages(
        self,
        prompt: QueryRewritePrompt,
        pair_count: int,
    ) -> tuple[list[dict[str, str]], int]:
        messages = self._conversation_messages(prompt, pair_count)
        messages.append(
            {
                "role": "assistant",
                "content": self.granite_config.response_prefill,
            }
        )
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        except Exception as exc:
            raise SGLangQueryRewriteError(
                "Granite tokenizer could not render the prefilled chat"
            ) from exc
        if not isinstance(rendered, str) or not rendered:
            raise SGLangQueryRewriteError(
                "Granite tokenizer returned a malformed chat template"
            )
        return messages, _token_count(self.tokenizer, rendered)

    def _bounded_messages(
        self,
        prompt: QueryRewritePrompt,
    ) -> tuple[list[dict[str, str]], int, int]:
        pair_count = len(prompt.conversation_pairs)
        while True:
            messages, token_count = self._render_prefilled_messages(prompt, pair_count)
            if token_count <= self.granite_config.max_input_tokens:
                return messages, token_count, pair_count
            if pair_count == 0:
                raise SGLangQueryRewriteError(
                    "Latest query and Granite response prefill exceed the configured input-token limit"
                )
            pair_count -= 1

    def _request_body(self, messages: list[dict[str, str]]) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self.sglang_config.served_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.granite_config.max_new_tokens,
            "n": 1,
            "stream": False,
            "continue_final_message": True,
        }
        if self.sglang_config.constrained_output:
            body["regex"] = self.sglang_config.continuation_regex
        return body

    def _send(self, body: Mapping[str, object]) -> Mapping[str, object]:
        headers = {"Content-Type": "application/json"}
        if self.sglang_config.api_key:
            headers["Authorization"] = f"Bearer {self.sglang_config.api_key}"
        try:
            response = self.client.post(
                f"{self.sglang_config.base_url}/chat/completions",
                json=dict(body),
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise _TransientSGLangQueryRewriteError(
                "SGLang query rewriting timed out"
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in _TRANSIENT_HTTP_STATUSES:
                raise _TransientSGLangQueryRewriteError(
                    "SGLang query rewriting request failed transiently"
                ) from exc
            raise SGLangQueryRewriteError("SGLang query rewriting request failed") from exc
        except httpx.TransportError as exc:
            raise _TransientSGLangQueryRewriteError(
                "SGLang query rewriting request failed transiently"
            ) from exc
        except httpx.HTTPError as exc:
            raise SGLangQueryRewriteError("SGLang query rewriting request failed") from exc
        except (TypeError, ValueError) as exc:
            raise SGLangQueryRewriteError(
                "SGLang returned a malformed JSON response"
            ) from exc
        if not isinstance(payload, Mapping):
            raise SGLangQueryRewriteError("SGLang returned a malformed response object")
        return payload

    def _response_content(
        self,
        payload: Mapping[str, object],
        *,
        expected_prompt_tokens: int,
    ) -> tuple[str, int, int, int | None]:
        response_model = payload.get("model")
        if response_model != self.sglang_config.served_model:
            raise SGLangQueryRewriteError("SGLang returned an unexpected model identity")
        choices = payload.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], Mapping)
        ):
            raise SGLangQueryRewriteError("SGLang returned malformed completion choices")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise SGLangQueryRewriteError("SGLang returned non-text completion content")
        continuation = message["content"]
        if choice.get("finish_reason") not in {"stop", "length"}:
            raise SGLangQueryRewriteError("SGLang returned an invalid finish reason")

        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            raise SGLangQueryRewriteError("SGLang returned malformed token usage")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if (
            not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens <= 0
            or not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
            or completion_tokens < 0
        ):
            raise SGLangQueryRewriteError("SGLang returned invalid token usage")
        if prompt_tokens != expected_prompt_tokens:
            raise SGLangQueryRewriteError(
                "Local and SGLang Granite prompt token counts do not match"
            )
        if prompt_tokens > self.granite_config.max_input_tokens:
            raise SGLangQueryRewriteError(
                "SGLang reported a prompt beyond the configured input-token limit"
            )
        if completion_tokens > self.granite_config.max_new_tokens:
            raise SGLangQueryRewriteError(
                "SGLang reported a completion beyond the configured output-token limit"
            )
        if bool(continuation) != bool(completion_tokens):
            raise SGLangQueryRewriteError(
                "SGLang completion content and token usage are inconsistent"
            )
        cached_tokens: int | None = None
        details = usage.get("prompt_tokens_details")
        if details is not None:
            raw_cached = details.get("cached_tokens") if isinstance(details, Mapping) else None
            if (
                raw_cached is not None
                and isinstance(raw_cached, int)
                and not isinstance(raw_cached, bool)
                and 0 <= raw_cached <= prompt_tokens
            ):
                cached_tokens = raw_cached
            elif not isinstance(details, Mapping) or raw_cached is not None:
                logger.warning(
                    "Ignored malformed optional Granite cache diagnostics",
                    extra={"model": self.sglang_config.served_model},
                )
        return continuation, prompt_tokens, completion_tokens, cached_tokens

    def _parse_continuation(
        self,
        continuation: str,
        *,
        input_tokens: int,
        completion_tokens: int,
        cached_tokens: int | None,
        retained_pairs: int,
        total_pairs: int,
        latency_ms: float,
    ) -> str:
        result, strict_json = parse_granite_rewrite(
            self.granite_config.response_prefill,
            continuation,
        )
        if trace_operation_active():
            set_flag("granite_strict_json", strict_json)
            set_flag("granite_repair_applied", not strict_json)
            add_count("granite_final_prompt_tokens", input_tokens)
            add_count("granite_final_completion_tokens", completion_tokens)
            if cached_tokens is None:
                set_flag("granite_cached_tokens_available", False)
            else:
                set_flag("granite_cached_tokens_available", True)
                add_count("granite_final_cached_prompt_tokens", cached_tokens)
            set_framed_digest(
                "granite_rewritten_query_sha256",
                "chat-granite-rewritten-query-v1",
                (result.encode("utf-8"),),
            )
            set_text("granite_rewritten_query", result)
        diagnostics = GraniteRewriteDiagnostics(
            rendered_input_tokens=input_tokens,
            generated_tokens=completion_tokens,
            retained_conversation_pairs=retained_pairs,
            dropped_conversation_pairs=total_pairs - retained_pairs,
            strict_json=strict_json,
            cached_prompt_tokens=cached_tokens,
            service_latency_ms=latency_ms,
        )
        self._thread_diagnostics.value = diagnostics
        # Retain the legacy process-wide inspection attribute for sequential callers.
        self.last_diagnostics = diagnostics
        capture_evaluation_rewrite(result)
        return result

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        reasoning: str | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        if model != QUERY_REWRITER.model:
            raise ValueError("SGLang Granite adapter received an unexpected model identifier")
        if not isinstance(prompt, QueryRewritePrompt):
            raise TypeError("Granite query rewriting requires a structured QueryRewritePrompt")
        with observe_stage("chat.granite_budgeting"):
            messages, input_tokens, retained_pairs = self._bounded_messages(prompt)
        body = self._request_body(messages)
        if trace_operation_active():
            total_pairs = len(prompt.conversation_pairs)
            retained = prompt.conversation_pairs[:retained_pairs]
            add_count("granite_input_pair_count", total_pairs)
            add_count("granite_retained_pair_count", retained_pairs)
            add_count("granite_dropped_pair_count", total_pairs - retained_pairs)
            add_count("granite_rendered_input_tokens", input_tokens)
            for empty_count in (
                "granite_transient_failure_count",
                "granite_transient_retry_count",
                "granite_format_failure_count",
                "granite_format_retry_count",
                "granite_usage_response_count",
                "granite_prompt_tokens_total",
                "granite_completion_tokens_total",
                "granite_cached_prompt_tokens_total",
            ):
                add_count(empty_count, 0)
            set_sample(
                "granite_input_pair_ids",
                (pair.object_id for pair in prompt.conversation_pairs),
                exact_count=total_pairs,
            )
            set_sample(
                "granite_retained_pair_ids",
                (pair.object_id for pair in retained),
                exact_count=retained_pairs,
            )
            set_flag(
                "granite_input_pair_ids_proof_truncated",
                total_pairs > TRACE_SAMPLE_LIMIT,
            )
            set_flag(
                "granite_retained_pair_ids_proof_truncated",
                retained_pairs > TRACE_SAMPLE_LIMIT,
            )
            conversation_messages = messages[: retained_pairs * 2]
            set_framed_digest(
                "granite_conversation_context_sha256",
                "chat-granite-conversation-context-v1",
                (
                    value
                    for message in conversation_messages
                    for value in (
                        message["role"].encode("utf-8"),
                        message["content"].encode("utf-8"),
                    )
                ),
            )
            set_framed_digest(
                "granite_request_messages_sha256",
                "chat-granite-request-messages-v1",
                (
                    value
                    for message in messages
                    for value in (
                        message["role"].encode("utf-8"),
                        message["content"].encode("utf-8"),
                    )
                ),
            )
        total_latency_ms = 0.0
        for attempt in range(2):
            add_count("granite_attempt_count", 1)
            started = perf_counter()
            try:
                payload = self._send(body)
            except _TransientSGLangQueryRewriteError as exc:
                attempt_latency_ms = (perf_counter() - started) * 1000.0
                total_latency_ms += attempt_latency_ms
                observe_elapsed(
                    "chat.granite_http", attempt_latency_ms, failed=True
                )
                add_count("granite_transient_failure_count", 1)
                if attempt == 0:
                    add_count("granite_transient_retry_count", 1)
                    logger.warning(
                        "Retrying transient Granite query-rewrite request",
                        extra={"model": self.sglang_config.served_model},
                    )
                    continue
                raise SGLangQueryRewriteError(str(exc)) from exc
            attempt_latency_ms = (perf_counter() - started) * 1000.0
            total_latency_ms += attempt_latency_ms
            observe_elapsed("chat.granite_http", attempt_latency_ms)
            continuation, prompt_tokens, completion_tokens, cached_tokens = (
                self._response_content(payload, expected_prompt_tokens=input_tokens)
            )
            add_count("granite_usage_response_count", 1)
            add_count("granite_prompt_tokens_total", prompt_tokens)
            add_count("granite_completion_tokens_total", completion_tokens)
            if cached_tokens is not None:
                add_count("granite_cached_prompt_tokens_total", cached_tokens)
            try:
                return self._parse_continuation(
                    continuation,
                    input_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    retained_pairs=retained_pairs,
                    total_pairs=len(prompt.conversation_pairs),
                    latency_ms=total_latency_ms,
                )
            except GraniteRewriteFormatError as exc:
                add_count("granite_format_failure_count", 1)
                if attempt == 0:
                    add_count("granite_format_retry_count", 1)
                    logger.warning(
                        "Retrying unrepairable Granite formatting defect",
                        extra={"model": self.sglang_config.served_model},
                    )
                    continue
                raise SGLangQueryRewriteError(
                    "SGLang Granite rewrite did not satisfy the JSON contract"
                ) from exc
        raise AssertionError("Granite retry loop exhausted unexpectedly")


__all__ = [
    "SGLangGraniteQueryRewriter",
    "SGLangQueryRewriteError",
    "SGLangTransport",
]
