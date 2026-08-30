"""SGLang/OpenAI-compatible adapter for the merged Granite query rewriter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
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
    GraniteRewriteDiagnostics,
    validate_granite_checkpoint,
)
from backend.rag.query_rewrite_contract import QueryRewritePrompt


logger = logging.getLogger(__name__)


class SGLangTransport(Protocol):
    """The subset of ``httpx.Client`` used by the adapter."""

    def post(self, url: str, **kwargs: object) -> Any: ...

    def close(self) -> None: ...


class SGLangQueryRewriteError(GraniteInferenceError):
    """The SGLang worker could not produce a trustworthy rewrite response."""


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
            raise SGLangQueryRewriteError("SGLang query rewriting timed out") from exc
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
        if continuation.startswith(self.granite_config.response_prefill):
            raise SGLangQueryRewriteError("SGLang ambiguously repeated the response prefill")
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
            if not isinstance(details, Mapping):
                raise SGLangQueryRewriteError("SGLang returned malformed cache usage")
            raw_cached = details.get("cached_tokens")
            if raw_cached is not None:
                if (
                    not isinstance(raw_cached, int)
                    or isinstance(raw_cached, bool)
                    or not 0 <= raw_cached <= prompt_tokens
                ):
                    raise SGLangQueryRewriteError("SGLang returned invalid cache usage")
                cached_tokens = raw_cached
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
        full_output = self.granite_config.response_prefill + continuation
        strict_json = False
        try:
            parsed = json.loads(full_output)
            if not isinstance(parsed, dict) or set(parsed) != {"rewritten_question"}:
                raise ValueError("response must contain exactly rewritten_question")
            rewritten = parsed["rewritten_question"]
            if not isinstance(rewritten, str) or not rewritten.strip():
                raise ValueError("rewritten_question must be a nonempty string")
            strict_json = True
            result = rewritten.strip()
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            if not any(character.isalnum() for character in continuation):
                raise SGLangQueryRewriteError(
                    "Granite generated no meaningful continuation"
                ) from exc
            logger.warning(
                "SGLang Granite rewrite did not satisfy the strict JSON contract",
                extra={
                    "model": self.sglang_config.served_model,
                    "generated_tokens": completion_tokens,
                    "parse_error": type(exc).__name__,
                },
            )
            result = full_output
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
        messages, input_tokens, retained_pairs = self._bounded_messages(prompt)
        started = perf_counter()
        payload = self._send(self._request_body(messages))
        latency_ms = (perf_counter() - started) * 1000.0
        continuation, prompt_tokens, completion_tokens, cached_tokens = (
            self._response_content(payload, expected_prompt_tokens=input_tokens)
        )
        return self._parse_continuation(
            continuation,
            input_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            retained_pairs=retained_pairs,
            total_pairs=len(prompt.conversation_pairs),
            latency_ms=latency_ms,
        )


__all__ = [
    "SGLangGraniteQueryRewriter",
    "SGLangQueryRewriteError",
    "SGLangTransport",
]
