"""Offline ONNX Runtime cross-encoder reranking for BGE v2-m3."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from backend.model_config import ONNX_RERANKER, RERANKER_MODEL, ONNXModelConfig
from backend.rag.runtime import RerankResult


_MANIFEST_SCHEMA_VERSION = "1.0"
_TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


class ONNXRerankerError(RuntimeError):
    """Raised when local reranker artifacts or inference violate the contract."""


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _artifact_path(root: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute():
        raise ONNXRerankerError("ONNX artifact paths must be relative")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ONNXRerankerError("ONNX artifact path escapes its model directory") from exc
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
        raise ONNXRerankerError("local reranker model directory does not exist")
    model_path = _artifact_path(root, config.onnx_filename)
    manifest_path = _artifact_path(root, config.manifest_filename)
    required = {config.onnx_filename, *_TOKENIZER_FILES}
    missing = sorted(name for name in required if not _artifact_path(root, name).is_file())
    if missing:
        raise ONNXRerankerError(
            f"local reranker model is missing required files: {', '.join(missing)}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ONNXRerankerError("reranker model manifest is missing or malformed") from exc
    if not isinstance(manifest, Mapping):
        raise ONNXRerankerError("reranker model manifest must be an object")
    if manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
        raise ONNXRerankerError("reranker model manifest schema is unsupported")
    if manifest.get("model_id") != model_id:
        raise ONNXRerankerError("reranker model manifest has the wrong model identity")
    if manifest.get("revision") != config.revision:
        raise ONNXRerankerError("reranker model manifest has the wrong revision")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not required.issubset(files):
        raise ONNXRerankerError("reranker model manifest is incomplete")
    for relative_name, expected_digest in files.items():
        if not isinstance(relative_name, str) or not isinstance(expected_digest, str):
            raise ONNXRerankerError("reranker model manifest entries are malformed")
        candidate = _artifact_path(root, relative_name)
        if not candidate.is_file() or _sha256(candidate) != expected_digest.lower():
            raise ONNXRerankerError("reranker model artifact hash verification failed")
    return root, model_path


def _input_dtype(type_name: object) -> np.dtype[Any]:
    if type_name == "tensor(int32)":
        return np.dtype(np.int32)
    if type_name == "tensor(int64)":
        return np.dtype(np.int64)
    raise ONNXRerankerError(f"unsupported ONNX tokenizer input type {type_name!r}")


class ONNXCrossEncoderReranker:
    """One eagerly loaded, reusable BGE pair tokenizer and ONNX session."""

    def __init__(
        self,
        config: ONNXModelConfig = ONNX_RERANKER,
        *,
        model_id: str = RERANKER_MODEL,
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
                raise ONNXRerankerError(
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
            raise ONNXRerankerError("reranker session did not activate the required provider")
        inputs = list(get_inputs())
        outputs = list(get_outputs())
        input_names = {getattr(item, "name", None) for item in inputs}
        if not {"input_ids", "attention_mask"}.issubset(input_names):
            raise ONNXRerankerError("reranker ONNX inputs are incompatible")
        if config.output_name not in {getattr(item, "name", None) for item in outputs}:
            raise ONNXRerankerError("reranker ONNX output is incompatible")

        self._tokenizer = tokenizer
        self._session = session
        self._inputs = tuple(inputs)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        model: str,
        top_n: int,
    ) -> Sequence[RerankResult]:
        query_text = _required_text(query, "query")
        if model != self._model_id:
            raise ValueError("reranker model does not match the loaded ONNX checkpoint")
        if isinstance(top_n, bool) or not isinstance(top_n, int):
            raise TypeError("top_n must be an integer")
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero")
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence of strings")
        validated = [_required_text(item, "document") for item in documents]
        if not validated:
            return []

        scores: list[float] = []
        for start in range(0, len(validated), self._config.batch_size):
            batch = validated[start : start + self._config.batch_size]
            pairs = [(query_text, document) for document in batch]
            encoded = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self._config.max_tokens,
                return_tensors="np",
            )
            if not isinstance(encoded, Mapping):
                raise ONNXRerankerError("reranker tokenizer output is malformed")
            feed: dict[str, np.ndarray[Any, Any]] = {}
            for input_meta in self._inputs:
                name = getattr(input_meta, "name", None)
                if not isinstance(name, str) or name not in encoded:
                    raise ONNXRerankerError("reranker tokenizer omitted a model input")
                feed[name] = np.asarray(
                    encoded[name],
                    dtype=_input_dtype(getattr(input_meta, "type", None)),
                )
            raw_outputs = self._session.run([self._config.output_name], feed)
            if not isinstance(raw_outputs, list) or len(raw_outputs) != 1:
                raise ONNXRerankerError("reranker ONNX response is malformed")
            logits = np.asarray(raw_outputs[0], dtype=np.float32)
            if logits.shape == (len(batch), 1):
                logits = logits[:, 0]
            if logits.shape != (len(batch),):
                raise ONNXRerankerError("reranker ONNX output has the wrong shape")
            if not np.isfinite(logits).all():
                raise ONNXRerankerError("reranker ONNX output contains non-finite values")
            scores.extend(float(item) for item in logits)

        if len(scores) != len(validated) or any(not math.isfinite(item) for item in scores):
            raise ONNXRerankerError("reranker did not return one finite score per document")
        ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
        return [
            RerankResult(index=index, score=score)
            for index, score in ranked[: min(top_n, len(ranked))]
        ]


__all__ = ["ONNXCrossEncoderReranker", "ONNXRerankerError"]
