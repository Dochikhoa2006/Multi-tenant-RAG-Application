"""Offline ONNX Runtime embeddings for the configured GTE checkpoint."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from backend.model_config import (
    EMBEDDING_MODEL,
    LATEON_DOCUMENT_MAX_MODEL_TOKENS,
    LATEON_EMBEDDING_DIMENSION,
    LATEON_MODEL,
    LATEON_QUERY_MAX_MODEL_TOKENS,
    ONNX_EMBEDDING,
    ONNX_LATE_INTERACTION,
    ONNXModelConfig,
)
from backend.providers.onnx_cuda import (
    enable_assignment_recording,
    validate_cuda_placement,
)


EMBEDDING_DIMENSION = 768
_MANIFEST_SCHEMA_VERSION = "1.0"
_TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
_EXPECTED_CPU_ASSIGNMENT_SHA256 = (
    "a63fd5859352eb6901eaa58f9ed6e94a0fc230987d799e38996ca8b012206ca1"
)
_REQUIRED_CUDA_OPERATORS = frozenset({"MatMul", "ReduceMean", "Softmax"})
_LATEON_REQUIRED_FILES = frozenset(
    {
        "config.json",
        "config_sentence_transformers.json",
        "onnx_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
_LATEON_SKIPLIST = (
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "{",
    "|",
    "}",
    "~",
)
# ORT 1.29/H100 profiling of the parity-gated FP16 artifact.  The 368 CPU
# assignments are exclusively integer shape/control operations plus 22 scalar
# sequence-length casts; no floating non-scalar model tensor executes on CPU.
_EXPECTED_LATEON_CPU_ASSIGNMENT_SHA256 = (
    "53dbd9af5e31b81d735e0d7bc26118edff611a0b430627b0c3f87663a9f7a2a3"
)
_REQUIRED_LATEON_CUDA_OPERATORS = frozenset({"MatMul", "Softmax"})
_LOGGER = logging.getLogger(__name__)


class ONNXEmbeddingError(RuntimeError):
    """Raised when local embedding artifacts or inference violate the contract."""


class ONNXLateOnError(RuntimeError):
    """Raised when pinned LateOn artifacts or inference violate the contract."""


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _artifact_path(root: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute():
        raise ONNXEmbeddingError("ONNX artifact paths must be relative")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ONNXEmbeddingError("ONNX artifact path escapes its model directory") from exc
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_artifacts(
    config: ONNXModelConfig,
    model_id: str,
) -> tuple[Path, Path]:
    root = Path(config.model_path).expanduser().resolve()
    if not root.is_dir():
        raise ONNXEmbeddingError("local embedding model directory does not exist")
    model_path = _artifact_path(root, config.onnx_filename)
    manifest_path = _artifact_path(root, config.manifest_filename)
    required = {config.onnx_filename, *_TOKENIZER_FILES}
    missing = sorted(name for name in required if not _artifact_path(root, name).is_file())
    if missing:
        raise ONNXEmbeddingError(
            f"local embedding model is missing required files: {', '.join(missing)}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ONNXEmbeddingError("embedding model manifest is missing or malformed") from exc
    if not isinstance(manifest, Mapping):
        raise ONNXEmbeddingError("embedding model manifest must be an object")
    if manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
        raise ONNXEmbeddingError("embedding model manifest schema is unsupported")
    if manifest.get("model_id") != model_id:
        raise ONNXEmbeddingError("embedding model manifest has the wrong model identity")
    if manifest.get("revision") != config.revision:
        raise ONNXEmbeddingError("embedding model manifest has the wrong revision")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not required.issubset(files):
        raise ONNXEmbeddingError("embedding model manifest is incomplete")
    for relative_name, expected_digest in files.items():
        if not isinstance(relative_name, str) or not isinstance(expected_digest, str):
            raise ONNXEmbeddingError("embedding model manifest entries are malformed")
        candidate = _artifact_path(root, relative_name)
        if not candidate.is_file() or _sha256(candidate) != expected_digest.lower():
            raise ONNXEmbeddingError("embedding model artifact hash verification failed")
    return root, model_path


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ONNXLateOnError(f"{label} is missing or malformed") from exc
    if not isinstance(value, Mapping):
        raise ONNXLateOnError(f"{label} must be an object")
    return value


def _validated_lateon_artifacts(
    config: ONNXModelConfig,
) -> tuple[Path, Path, Mapping[str, object]]:
    root = Path(config.model_path).expanduser().resolve()
    if not root.is_dir():
        raise ONNXLateOnError("local LateOn model directory does not exist")
    model_path = _artifact_path(root, config.onnx_filename)
    manifest_path = _artifact_path(root, config.manifest_filename)
    required = {config.onnx_filename, *_LATEON_REQUIRED_FILES}
    missing = sorted(name for name in required if not _artifact_path(root, name).is_file())
    if missing:
        raise ONNXLateOnError(
            f"local LateOn model is missing required files: {', '.join(missing)}"
        )
    manifest = _json_object(manifest_path, "LateOn model manifest")
    if manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
        raise ONNXLateOnError("LateOn model manifest schema is unsupported")
    if manifest.get("model_id") != LATEON_MODEL:
        raise ONNXLateOnError("LateOn model manifest has the wrong model identity")
    if manifest.get("revision") != config.revision:
        raise ONNXLateOnError("LateOn model manifest has the wrong revision")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not required.issubset(files):
        raise ONNXLateOnError("LateOn model manifest is incomplete")
    for relative_name, expected_digest in files.items():
        if not isinstance(relative_name, str) or not isinstance(expected_digest, str):
            raise ONNXLateOnError("LateOn model manifest entries are malformed")
        candidate = _artifact_path(root, relative_name)
        if not candidate.is_file() or _sha256(candidate) != expected_digest.lower():
            raise ONNXLateOnError("LateOn model artifact hash verification failed")

    onnx_config = _json_object(root / "onnx_config.json", "LateOn ONNX config")
    sentence_config = _json_object(
        root / "config_sentence_transformers.json",
        "LateOn sentence-transformers config",
    )
    exact_onnx_values = {
        "model_type": "ColBERT",
        "uses_token_type_ids": False,
        "query_prefix": "[Q] ",
        "document_prefix": "[D] ",
        "query_length": LATEON_QUERY_MAX_MODEL_TOKENS,
        "document_length": LATEON_DOCUMENT_MAX_MODEL_TOKENS,
        "do_query_expansion": False,
        "attend_to_expansion_tokens": False,
        "embedding_dim": LATEON_EMBEDDING_DIMENSION,
        "query_prefix_id": 50368,
        "document_prefix_id": 50369,
        "mask_token_id": 50284,
        "pad_token_id": 50284,
        "do_lower_case": False,
    }
    if any(onnx_config.get(name) != value for name, value in exact_onnx_values.items()):
        raise ONNXLateOnError("LateOn ONNX config violates the pinned model contract")
    exact_sentence_values = {
        "model_type": "ColBERT",
        "query_prefix": "[Q] ",
        "document_prefix": "[D] ",
        "query_length": LATEON_QUERY_MAX_MODEL_TOKENS,
        "document_length": LATEON_DOCUMENT_MAX_MODEL_TOKENS,
        "do_query_expansion": False,
        "attend_to_expansion_tokens": False,
        "similarity_fn_name": "MaxSim",
    }
    if any(
        sentence_config.get(name) != value
        for name, value in exact_sentence_values.items()
    ):
        raise ONNXLateOnError(
            "LateOn sentence-transformers config violates the pinned model contract"
        )
    if tuple(onnx_config.get("skiplist_words", ())) != _LATEON_SKIPLIST or tuple(
        sentence_config.get("skiplist_words", ())
    ) != _LATEON_SKIPLIST:
        raise ONNXLateOnError("LateOn punctuation skiplist is incompatible")
    return root, model_path, onnx_config


def _input_dtype(type_name: object) -> np.dtype[Any]:
    if type_name == "tensor(int32)":
        return np.dtype(np.int32)
    if type_name == "tensor(int64)":
        return np.dtype(np.int64)
    raise ONNXEmbeddingError(f"unsupported ONNX tokenizer input type {type_name!r}")


class ONNXEmbeddingClient:
    """One eagerly loaded, reusable GTE tokenizer and ONNX session."""

    def __init__(
        self,
        config: ONNXModelConfig = ONNX_EMBEDDING,
        *,
        model_id: str = EMBEDDING_MODEL,
        tokenizer: object | None = None,
        session: object | None = None,
        tokenizer_loader: Callable[..., object] | None = None,
        session_factory: Callable[..., object] | None = None,
        available_providers: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(config, ONNXModelConfig):
            raise TypeError("config must be an ONNXModelConfig")
        self._model_id = _required_text(model_id, "model_id")
        self._config = config
        root, model_path = _validated_artifacts(config, self._model_id)

        if tokenizer is None:
            if tokenizer_loader is None:
                from transformers import AutoTokenizer

                tokenizer_loader = AutoTokenizer.from_pretrained
            tokenizer = tokenizer_loader(
                str(root),
                local_files_only=True,
                trust_remote_code=False,
            )
        if not callable(tokenizer):
            raise TypeError("tokenizer must be callable")

        options: object | None = None
        if session is None:
            if session_factory is None or available_providers is None:
                import onnxruntime as ort

                session_factory = session_factory or ort.InferenceSession
                available_providers = available_providers or ort.get_available_providers()
                options = ort.SessionOptions()
                if (
                    config.disable_cpu_fallback
                    and config.execution_provider == "CUDAExecutionProvider"
                ):
                    enable_assignment_recording(options)
            if config.execution_provider not in available_providers:
                raise ONNXEmbeddingError(
                    f"required execution provider {config.execution_provider!r} is unavailable"
                )
            provider: str | tuple[str, dict[str, str]] = config.execution_provider
            if config.execution_provider == "CUDAExecutionProvider":
                provider = (
                    config.execution_provider,
                    {"device_id": str(config.device_id)},
                )
            session = session_factory(
                str(model_path),
                sess_options=options,
                providers=[provider],
                enable_fallback=False,
            )

        get_inputs = getattr(session, "get_inputs", None)
        get_outputs = getattr(session, "get_outputs", None)
        get_providers = getattr(session, "get_providers", None)
        disable_fallback = getattr(session, "disable_fallback", None)
        run = getattr(session, "run", None)
        if not all(
            callable(item)
            for item in (get_inputs, get_outputs, get_providers, disable_fallback, run)
        ):
            raise TypeError("session does not implement the ONNX inference contract")
        disable_fallback()
        providers = list(get_providers())
        if not providers or providers[0] != config.execution_provider:
            raise ONNXEmbeddingError("embedding session did not activate the required provider")
        if options is not None and config.execution_provider == "CUDAExecutionProvider":
            summary = validate_cuda_placement(
                session,
                expected_cpu_digest=_EXPECTED_CPU_ASSIGNMENT_SHA256,
                required_cuda_operators=_REQUIRED_CUDA_OPERATORS,
                error_factory=ONNXEmbeddingError,
                label="GTE embedding graph",
            )
            _LOGGER.info(
                "GTE CUDA placement validated with %d CPU bookkeeping nodes (%s)",
                summary.cpu_count,
                dict(summary.cpu_operators),
            )
        inputs = list(get_inputs())
        outputs = list(get_outputs())
        input_names = {getattr(item, "name", None) for item in inputs}
        if not {"input_ids", "attention_mask"}.issubset(input_names):
            raise ONNXEmbeddingError("embedding ONNX inputs are incompatible")
        if config.output_name not in {getattr(item, "name", None) for item in outputs}:
            raise ONNXEmbeddingError("embedding ONNX output is incompatible")

        self._tokenizer = tokenizer
        self._session = session
        self._inputs = tuple(inputs)
        self._closed = False

    def close(self) -> None:
        """Release process-owned tokenizer and ONNX session references."""

        if self._closed:
            return
        self._closed = True
        self._tokenizer = None
        self._session = None

    def _resources(self) -> tuple[object, object]:
        if self._closed or self._tokenizer is None or self._session is None:
            raise RuntimeError("embedding client is closed")
        return self._tokenizer, self._session

    def embed(self, text: str, *, model: str) -> Sequence[float]:
        return self.embed_many([_required_text(text, "text")], model=model)[0]

    def embed_many(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> Sequence[Sequence[float]]:
        if model != self._model_id:
            raise ValueError("embedding model does not match the loaded ONNX checkpoint")
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")
        validated = [_required_text(item, "text") for item in texts]
        if not validated:
            return []

        tokenizer, session = self._resources()

        embedded: list[list[float]] = []
        for start in range(0, len(validated), self._config.batch_size):
            batch = validated[start : start + self._config.batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._config.max_tokens,
                return_tensors="np",
            )
            if not isinstance(encoded, Mapping):
                raise ONNXEmbeddingError("embedding tokenizer output is malformed")
            feed: dict[str, np.ndarray[Any, Any]] = {}
            for input_meta in self._inputs:
                name = getattr(input_meta, "name", None)
                if not isinstance(name, str) or name not in encoded:
                    raise ONNXEmbeddingError("embedding tokenizer omitted a model input")
                feed[name] = np.asarray(
                    encoded[name],
                    dtype=_input_dtype(getattr(input_meta, "type", None)),
                )
            raw_outputs = session.run([self._config.output_name], feed)
            if not isinstance(raw_outputs, list) or len(raw_outputs) != 1:
                raise ONNXEmbeddingError("embedding ONNX response is malformed")
            hidden_state = np.asarray(raw_outputs[0], dtype=np.float32)
            if (
                hidden_state.ndim != 3
                or hidden_state.shape[0] != len(batch)
                or hidden_state.shape[1] == 0
                or hidden_state.shape[2] != EMBEDDING_DIMENSION
            ):
                raise ONNXEmbeddingError("embedding ONNX output has the wrong shape")
            cls_embeddings = hidden_state[:, 0, :]
            if not np.isfinite(cls_embeddings).all():
                raise ONNXEmbeddingError("embedding ONNX output contains non-finite values")
            norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or np.any(norms <= 0):
                raise ONNXEmbeddingError("embedding ONNX output has an invalid norm")
            normalized = np.asarray(cls_embeddings / norms, dtype=np.float32)
            embedded.extend(normalized.tolist())
        return embedded


class ONNXLateOnProvider:
    """One reusable, offline LateOn tokenizer and FP16 CUDA session."""

    def __init__(
        self,
        config: ONNXModelConfig = ONNX_LATE_INTERACTION,
        *,
        tokenizer: object | None = None,
        session: object | None = None,
        tokenizer_loader: Callable[..., object] | None = None,
        session_factory: Callable[..., object] | None = None,
        available_providers: Sequence[str] | None = None,
        warmup: bool = True,
    ) -> None:
        if not isinstance(config, ONNXModelConfig):
            raise TypeError("config must be an ONNXModelConfig")
        if config.execution_provider != "CUDAExecutionProvider":
            raise ONNXLateOnError("LateOn requires CUDAExecutionProvider")
        if config.max_tokens != LATEON_DOCUMENT_MAX_MODEL_TOKENS:
            raise ONNXLateOnError("LateOn document length must remain 300 tokens")
        self._config = config
        root, model_path, metadata = _validated_lateon_artifacts(config)

        if tokenizer is None:
            if tokenizer_loader is None:
                from transformers import AutoTokenizer

                tokenizer_loader = AutoTokenizer.from_pretrained
            tokenizer = tokenizer_loader(
                str(root),
                local_files_only=True,
                trust_remote_code=False,
            )
        if not callable(tokenizer):
            raise TypeError("tokenizer must be callable")
        convert_tokens = getattr(tokenizer, "convert_tokens_to_ids", None)
        if not callable(convert_tokens):
            raise TypeError("LateOn tokenizer cannot resolve prefix and skiplist IDs")
        for field, token in (
            ("query_prefix_id", "[Q] "),
            ("document_prefix_id", "[D] "),
        ):
            if convert_tokens(token) != metadata[field]:
                raise ONNXLateOnError("LateOn tokenizer prefix IDs are incompatible")
        if getattr(tokenizer, "pad_token_id", None) != metadata["pad_token_id"]:
            raise ONNXLateOnError("LateOn tokenizer padding ID is incompatible")

        options: object | None = None
        if session is None:
            if session_factory is None or available_providers is None:
                import onnxruntime as ort

                session_factory = session_factory or ort.InferenceSession
                available_providers = available_providers or ort.get_available_providers()
                options = ort.SessionOptions()
                enable_assignment_recording(options)
            if "CUDAExecutionProvider" not in available_providers:
                raise ONNXLateOnError("CUDAExecutionProvider is unavailable for LateOn")
            session = session_factory(
                str(model_path),
                sess_options=options,
                providers=[
                    (
                        "CUDAExecutionProvider",
                        {"device_id": str(config.device_id)},
                    )
                ],
                enable_fallback=False,
            )

        get_inputs = getattr(session, "get_inputs", None)
        get_outputs = getattr(session, "get_outputs", None)
        get_providers = getattr(session, "get_providers", None)
        disable_fallback = getattr(session, "disable_fallback", None)
        run = getattr(session, "run", None)
        if not all(
            callable(item)
            for item in (get_inputs, get_outputs, get_providers, disable_fallback, run)
        ):
            raise TypeError("session does not implement the ONNX inference contract")
        disable_fallback()
        providers = list(get_providers())
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise ONNXLateOnError("LateOn session did not activate CUDAExecutionProvider")
        if options is not None:
            if not _EXPECTED_LATEON_CPU_ASSIGNMENT_SHA256:
                raise ONNXLateOnError(
                    "LateOn CUDA placement fingerprint has not been provisioned"
                )
            summary = validate_cuda_placement(
                session,
                expected_cpu_digest=_EXPECTED_LATEON_CPU_ASSIGNMENT_SHA256,
                required_cuda_operators=_REQUIRED_LATEON_CUDA_OPERATORS,
                error_factory=ONNXLateOnError,
                label="LateOn graph",
            )
            _LOGGER.info(
                "LateOn CUDA placement validated with %d CPU bookkeeping nodes (%s)",
                summary.cpu_count,
                dict(summary.cpu_operators),
            )

        inputs = list(get_inputs())
        outputs = list(get_outputs())
        if {getattr(item, "name", None) for item in inputs} != {
            "input_ids",
            "attention_mask",
        }:
            raise ONNXLateOnError("LateOn ONNX inputs are incompatible")
        matching_outputs = [
            item for item in outputs if getattr(item, "name", None) == config.output_name
        ]
        if len(matching_outputs) != 1:
            raise ONNXLateOnError("LateOn ONNX output is incompatible")
        output_type = getattr(matching_outputs[0], "type", None)
        if output_type not in (None, "tensor(float16)"):
            raise ONNXLateOnError("LateOn production ONNX output must be FP16")

        self._tokenizer = tokenizer
        self._session = session
        self._inputs = tuple(inputs)
        self._query_prefix_id = int(metadata["query_prefix_id"])
        self._document_prefix_id = int(metadata["document_prefix_id"])
        self._skiplist_ids = frozenset(
            int(convert_tokens(token)) for token in _LATEON_SKIPLIST
        )
        self._closed = False
        if warmup:
            self.encode_query("LateOn query warmup")
            self.encode_documents(["LateOn document warmup."])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._tokenizer = None
        self._session = None

    def _resources(self) -> tuple[object, object]:
        if self._closed or self._tokenizer is None or self._session is None:
            raise RuntimeError("LateOn provider is closed")
        return self._tokenizer, self._session

    def _tokenized(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], list[np.ndarray[Any, Any]]]:
        tokenizer, _ = self._resources()
        normalized_texts = [text.strip() for text in texts]
        kwargs: dict[str, object] = {
            "add_special_tokens": True,
            "padding": True,
            "truncation": "longest_first",
            "max_length": (
                LATEON_QUERY_MAX_MODEL_TOKENS - 1
                if is_query
                else LATEON_DOCUMENT_MAX_MODEL_TOKENS - 1
            ),
            "return_tensors": "np",
        }
        # PyLate 1.3.4 delegates to Sentence Transformers 5.1.1, which strips
        # each text, applies longest-first truncation at configured_length - 1,
        # then inserts the configured prefix after the first special token.
        # Golden parity tests bind this sequence to the pinned dependencies.
        encoded = tokenizer(normalized_texts, **kwargs)
        if not isinstance(encoded, Mapping):
            raise ONNXLateOnError("LateOn tokenizer output is malformed")
        try:
            input_ids = np.asarray(encoded["input_ids"])
            attention_mask = np.asarray(encoded["attention_mask"])
        except KeyError as exc:
            raise ONNXLateOnError("LateOn tokenizer omitted a model input") from exc
        if (
            input_ids.ndim != 2
            or attention_mask.shape != input_ids.shape
            or input_ids.shape[0] != len(texts)
            or input_ids.shape[1] == 0
        ):
            raise ONNXLateOnError("LateOn tokenizer tensors have incompatible shapes")
        if not np.all((attention_mask == 0) | (attention_mask == 1)):
            raise ONNXLateOnError("LateOn attention mask must be binary")

        lengths = np.asarray(attention_mask.sum(axis=1), dtype=np.int64) + 1
        limit = (
            LATEON_QUERY_MAX_MODEL_TOKENS
            if is_query
            else LATEON_DOCUMENT_MAX_MODEL_TOKENS
        )
        if np.any(lengths > limit):
            if is_query:
                raise ONNXLateOnError("LateOn query tokenization exceeded 32 tokens")
            raise ONNXLateOnError(
                "LateOn document exceeds the 300-model-token retrieval-unit limit"
            )

        prefix_id = self._query_prefix_id if is_query else self._document_prefix_id
        input_ids = np.concatenate(
            (
                input_ids[:, :1],
                np.full((len(texts), 1), prefix_id, dtype=input_ids.dtype),
                input_ids[:, 1:],
            ),
            axis=1,
        )
        attention_mask = np.concatenate(
            (
                attention_mask[:, :1],
                np.ones((len(texts), 1), dtype=attention_mask.dtype),
                attention_mask[:, 1:],
            ),
            axis=1,
        )
        retained: list[np.ndarray[Any, Any]] = []
        for ids, mask in zip(input_ids, attention_mask, strict=True):
            selected = mask.astype(bool)
            if not is_query:
                selected &= ~np.isin(ids, tuple(self._skiplist_ids))
            retained.append(selected)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }, retained

    def _encode_batch(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
    ) -> list[list[list[float]]]:
        _, session = self._resources()
        encoded, retained = self._tokenized(texts, is_query=is_query)
        feed: dict[str, np.ndarray[Any, Any]] = {}
        for input_meta in self._inputs:
            name = getattr(input_meta, "name", None)
            if not isinstance(name, str) or name not in encoded:
                raise ONNXLateOnError("LateOn tokenizer omitted a model input")
            feed[name] = np.asarray(
                encoded[name],
                dtype=_input_dtype(getattr(input_meta, "type", None)),
            )
        raw_outputs = session.run([self._config.output_name], feed)
        if not isinstance(raw_outputs, list) or len(raw_outputs) != 1:
            raise ONNXLateOnError("LateOn ONNX response is malformed")
        output = np.asarray(raw_outputs[0], dtype=np.float32)
        if (
            output.ndim != 3
            or output.shape[:2] != feed["input_ids"].shape
            or output.shape[2] != LATEON_EMBEDDING_DIMENSION
            or not np.isfinite(output).all()
        ):
            raise ONNXLateOnError("LateOn ONNX output has an invalid shape or value")

        matrices: list[list[list[float]]] = []
        for token_embeddings, mask in zip(output, retained, strict=True):
            selected = token_embeddings[mask]
            if selected.shape[0] == 0:
                raise ONNXLateOnError("LateOn produced an empty token matrix")
            norms = np.linalg.norm(selected, axis=1)
            if (
                not np.isfinite(norms).all()
                or np.any(norms <= 0)
                or not np.allclose(norms, 1.0, atol=5e-3, rtol=5e-3)
            ):
                raise ONNXLateOnError("LateOn output is not L2-normalized")
            matrices.append(selected.tolist())
        return matrices

    def encode_query(self, text: str) -> Sequence[Sequence[float]]:
        value = _required_text(text, "text")
        return self._encode_batch([value], is_query=True)[0]

    def encode_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[Sequence[float]]]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")
        validated = [_required_text(item, "document") for item in texts]
        if not validated:
            return []
        for document in validated:
            if len(self.encode(document)) > LATEON_DOCUMENT_MAX_MODEL_TOKENS:
                raise ONNXLateOnError(
                    "LateOn document exceeds the 300-model-token retrieval-unit limit"
                )
        encoded: list[list[list[float]]] = []
        for start in range(0, len(validated), self._config.batch_size):
            encoded.extend(
                self._encode_batch(
                    validated[start : start + self._config.batch_size],
                    is_query=False,
                )
            )
        return encoded

    def encode(self, text: str) -> Sequence[int]:
        """Return exact pinned document-token IDs for lossless chunk sizing."""

        value = _required_text(text, "text")
        tokenizer, _ = self._resources()
        encoded = tokenizer(
            value.strip(),
            add_special_tokens=True,
            truncation=False,
        )
        if not isinstance(encoded, Mapping):
            raise ONNXLateOnError("LateOn tokenizer output is malformed")
        raw_ids = encoded.get("input_ids")
        if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
            raise ONNXLateOnError("LateOn tokenizer omitted input_ids")
        ids = [int(item) for item in raw_ids]
        if not ids:
            raise ONNXLateOnError("LateOn tokenizer returned no document tokens")
        return [ids[0], self._document_prefix_id, *ids[1:]]

    def segment_document(self, text: str) -> Sequence[str]:
        """Losslessly segment canonical text under the LateOn hard cap."""

        from backend.processing.chunker import chunk_paragraph

        canonical = _required_text(text, "text")
        segments = chunk_paragraph(canonical, tokenizer=self)
        if not segments or "".join(segments) != canonical:
            raise ONNXLateOnError("LateOn segmentation was not lossless")
        if any(
            len(self.encode(segment)) > LATEON_DOCUMENT_MAX_MODEL_TOKENS
            for segment in segments
        ):
            raise ONNXLateOnError("LateOn segmentation exceeded its document limit")
        return segments


__all__ = [
    "EMBEDDING_DIMENSION",
    "ONNXEmbeddingClient",
    "ONNXEmbeddingError",
    "ONNXLateOnError",
    "ONNXLateOnProvider",
]
