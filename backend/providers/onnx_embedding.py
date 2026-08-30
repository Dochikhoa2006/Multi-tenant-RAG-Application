"""Offline ONNX Runtime embeddings for the configured GTE checkpoint."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.model_config import EMBEDDING_MODEL, ONNX_EMBEDDING, ONNXModelConfig


EMBEDDING_DIMENSION = 768
_MANIFEST_SCHEMA_VERSION = "1.0"
_TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


class ONNXEmbeddingError(RuntimeError):
    """Raised when local embedding artifacts or inference violate the contract."""


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

        if session is None:
            if session_factory is None or available_providers is None:
                import onnxruntime as ort

                session_factory = session_factory or ort.InferenceSession
                available_providers = available_providers or ort.get_available_providers()
                options = ort.SessionOptions()
                if config.disable_cpu_fallback:
                    options.add_session_config_entry(
                        "session.disable_cpu_ep_fallback",
                        "1",
                    )
            else:
                options = None
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

        embedded: list[list[float]] = []
        for start in range(0, len(validated), self._config.batch_size):
            batch = validated[start : start + self._config.batch_size]
            encoded = self._tokenizer(
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
            raw_outputs = self._session.run([self._config.output_name], feed)
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


__all__ = ["EMBEDDING_DIMENSION", "ONNXEmbeddingClient", "ONNXEmbeddingError"]
