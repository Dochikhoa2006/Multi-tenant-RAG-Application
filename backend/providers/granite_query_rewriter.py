"""Offline Transformers/CUDA rollback adapter for merged Granite rewriting."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import threading
from typing import Any, Protocol

from backend.model_config import (
    GRANITE_QUERY_REWRITE,
    QUERY_REWRITER,
    GraniteQueryRewriteConfig,
)
from backend.rag.query_rewrite_contract import QueryRewritePrompt
from backend.rag.runtime import LLMClient


logger = logging.getLogger(__name__)
_EXPECTED_ARCHITECTURE = "GraniteForCausalLM"
_EXPECTED_MODEL_TYPE = "granite"
_INDEX_FILENAME = "model.safetensors.index.json"


class GraniteCheckpointError(RuntimeError):
    """The configured local checkpoint or execution environment is invalid."""


class GraniteInferenceError(RuntimeError):
    """Granite could not produce a usable query rewrite."""


class QueryRewriteCompletionClient(Protocol):
    """Minimal role-router contract implemented by either Granite engine."""

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        reasoning: str | None = None,
        max_output_tokens: int | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class GraniteRewriteDiagnostics:
    rendered_input_tokens: int
    generated_tokens: int
    retained_conversation_pairs: int
    dropped_conversation_pairs: int
    strict_json: bool
    cached_prompt_tokens: int | None = None
    service_latency_ms: float | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolved_model_path(config: GraniteQueryRewriteConfig) -> Path:
    configured = Path(config.model_path).expanduser()
    return (
        configured.resolve()
        if configured.is_absolute()
        else (_project_root() / configured).resolve()
    )


def validate_granite_checkpoint(path: Path) -> tuple[str, ...]:
    """Validate the complete local merged checkpoint without importing Transformers."""

    path = path.resolve()
    if not path.is_dir():
        raise GraniteCheckpointError(f"Granite model directory does not exist: {path}")
    config_path = path / "config.json"
    index_path = path / _INDEX_FILENAME
    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GraniteCheckpointError(
            f"Granite checkpoint file is missing: {exc.filename}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraniteCheckpointError("Granite checkpoint metadata is malformed") from exc
    except OSError as exc:
        raise GraniteCheckpointError("Granite checkpoint metadata cannot be read") from exc

    architectures = config_data.get("architectures")
    if (
        architectures != [_EXPECTED_ARCHITECTURE]
        or config_data.get("model_type") != _EXPECTED_MODEL_TYPE
    ):
        raise GraniteCheckpointError(
            "Checkpoint is not the expected Granite causal-LM architecture"
        )
    checkpoint_dtype = config_data.get("dtype", config_data.get("torch_dtype"))
    if checkpoint_dtype != "float16":
        raise GraniteCheckpointError("Granite checkpoint must declare float16 weights")

    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise GraniteCheckpointError("Granite Safetensors index has no weight map")
    raw_shard_names = tuple(weight_map.values())
    if any(not isinstance(name, str) for name in raw_shard_names):
        raise GraniteCheckpointError(
            "Granite Safetensors index contains an invalid shard name"
        )
    shard_names = tuple(sorted(set(raw_shard_names)))
    if len(shard_names) != 2 or any(not name.endswith(".safetensors") for name in shard_names):
        raise GraniteCheckpointError(
            "Granite checkpoint must reference exactly two Safetensors shards"
        )
    for shard_name in shard_names:
        shard_path = (path / shard_name).resolve()
        if shard_path.parent != path or not shard_path.is_file() or shard_path.stat().st_size <= 0:
            raise GraniteCheckpointError(
                f"Granite Safetensors shard is missing or invalid: {shard_name}"
            )
    return shard_names


def _load_transformers_runtime(
    config: GraniteQueryRewriteConfig,
    model_path: Path,
) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise GraniteCheckpointError(
            "Granite runtime dependencies are not installed; install backend/requirements.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise GraniteCheckpointError("NVIDIA CUDA is required and is not available")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
        device_map={"": config.device},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return tokenizer, model, torch


class GraniteQueryRewriter:
    """One eager, serialized local Granite model instance."""

    def __init__(
        self,
        *,
        config: GraniteQueryRewriteConfig = GRANITE_QUERY_REWRITE,
        tokenizer: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
        lock: threading.Lock | None = None,
    ) -> None:
        if (tokenizer is None) != (model is None):
            raise TypeError("tokenizer and model must be injected together")
        self.config = config
        self.model_path = _resolved_model_path(config)
        if tokenizer is None:
            validate_granite_checkpoint(self.model_path)
            tokenizer, model, torch_module = _load_transformers_runtime(
                config,
                self.model_path,
            )
        self.tokenizer = tokenizer
        self.model = model
        self._torch = torch_module
        self._lock = lock or threading.Lock()
        self.last_diagnostics: GraniteRewriteDiagnostics | None = None
        if config.warmup:
            self._warmup()

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        reasoning: str | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        if model != QUERY_REWRITER.model:
            raise ValueError("Granite adapter received an unexpected model identifier")
        if not isinstance(prompt, QueryRewritePrompt):
            raise TypeError("Granite query rewriting requires a structured QueryRewritePrompt")
        with self._lock:
            continuation, input_tokens, generated_tokens, retained_pairs = self._generate(
                prompt
            )
            return self._parse_response(
                continuation,
                input_tokens=input_tokens,
                generated_tokens=generated_tokens,
                retained_pairs=retained_pairs,
                total_pairs=len(prompt.conversation_pairs),
            )

    def _messages(
        self,
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

    def _render_and_encode(
        self,
        prompt: QueryRewritePrompt,
        pair_count: int,
    ) -> tuple[str, Any, int]:
        rendered_chat = self.tokenizer.apply_chat_template(
            self._messages(prompt, pair_count),
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered_chat, str):
            raise GraniteInferenceError("Granite tokenizer returned a non-text chat template")
        rendered = rendered_chat + self.config.response_prefill
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise GraniteInferenceError("Granite tokenizer returned malformed input IDs")
        try:
            token_count = int(encoded["input_ids"].shape[-1])
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise GraniteInferenceError("Granite tokenizer returned malformed input IDs") from exc
        return rendered, encoded, token_count

    def _bounded_input(self, prompt: QueryRewritePrompt) -> tuple[Any, int, int]:
        pair_count = len(prompt.conversation_pairs)
        while True:
            _, encoded, token_count = self._render_and_encode(prompt, pair_count)
            if token_count <= self.config.max_input_tokens:
                return encoded, token_count, pair_count
            if pair_count == 0:
                raise GraniteInferenceError(
                    "Latest query and Granite response prefill exceed the configured input-token limit"
                )
            pair_count -= 1

    def _to_device(self, encoded: Mapping[str, Any]) -> Mapping[str, Any]:
        if callable(getattr(encoded, "to", None)):
            return encoded.to(self.config.device)
        return {
            key: value.to(self.config.device) if callable(getattr(value, "to", None)) else value
            for key, value in encoded.items()
        }

    def _inference_context(self) -> Any:
        return self._torch.inference_mode() if self._torch is not None else nullcontext()

    def _generate(
        self,
        prompt: QueryRewritePrompt,
        *,
        max_new_tokens: int | None = None,
    ) -> tuple[str, int, int, int]:
        encoded, input_tokens, retained_pairs = self._bounded_input(prompt)
        model_inputs = self._to_device(encoded)
        generation_options: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
        }
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if eos_token_id is not None:
            generation_options["eos_token_id"] = eos_token_id
        if pad_token_id is not None:
            generation_options["pad_token_id"] = pad_token_id
        try:
            with self._inference_context():
                output = self.model.generate(**model_inputs, **generation_options)
            sequences = getattr(output, "sequences", output)
            sequence = sequences[0]
            generated_ids = sequence[input_tokens:]
            generated_tokens = len(generated_ids)
            continuation = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except GraniteInferenceError:
            raise
        except Exception as exc:
            raise GraniteInferenceError("Granite CUDA inference failed") from exc
        if not isinstance(continuation, str):
            raise GraniteInferenceError("Granite generated non-text output")
        return continuation, input_tokens, generated_tokens, retained_pairs

    def _parse_response(
        self,
        continuation: str,
        *,
        input_tokens: int,
        generated_tokens: int,
        retained_pairs: int,
        total_pairs: int,
    ) -> str:
        full_output = self.config.response_prefill + continuation
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
                raise GraniteInferenceError("Granite generated no meaningful continuation") from exc
            logger.warning(
                "Granite query rewrite did not satisfy the strict JSON contract",
                extra={
                    "model": QUERY_REWRITER.model,
                    "generated_tokens": generated_tokens,
                    "parse_error": type(exc).__name__,
                },
            )
            result = full_output
        self.last_diagnostics = GraniteRewriteDiagnostics(
            rendered_input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            retained_conversation_pairs=retained_pairs,
            dropped_conversation_pairs=total_pairs - retained_pairs,
            strict_json=strict_json,
        )
        return result

    def _warmup(self) -> None:
        prompt = QueryRewritePrompt(
            "warmup",
            original_query="Rewrite this standalone question.",
            conversation_pairs=(),
        )
        with self._lock:
            self._generate(prompt, max_new_tokens=1)


class RoleRoutingLLMClient:
    """Route only Model A to Granite; preserve all other LLM behavior."""

    def __init__(
        self,
        delegate: LLMClient,
        granite: QueryRewriteCompletionClient,
        *,
        granite_model: str = QUERY_REWRITER.model,
    ) -> None:
        if not callable(getattr(delegate, "complete", None)) or not callable(
            getattr(delegate, "stream", None)
        ):
            raise TypeError("delegate must provide complete() and stream()")
        if not callable(getattr(granite, "complete", None)):
            raise TypeError("granite must provide complete()")
        if not granite_model.strip():
            raise ValueError("granite_model must not be empty")
        self.delegate = delegate
        self.granite = granite
        self.granite_model = granite_model

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        reasoning: str | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        target = self.granite if model == self.granite_model else self.delegate
        return target.complete(
            prompt,
            model=model,
            reasoning=reasoning,
            max_output_tokens=max_output_tokens,
        )

    def stream(
        self,
        prompt: str,
        *,
        model: str,
        reasoning: str | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str] | Awaitable[AsyncIterator[str]]:
        return self.delegate.stream(
            prompt,
            model=model,
            reasoning=reasoning,
            max_output_tokens=max_output_tokens,
        )


__all__ = [
    "GraniteCheckpointError",
    "GraniteInferenceError",
    "GraniteQueryRewriter",
    "GraniteRewriteDiagnostics",
    "QueryRewriteCompletionClient",
    "RoleRoutingLLMClient",
    "validate_granite_checkpoint",
]
