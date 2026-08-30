from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.model_config import EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION, ONNXModelConfig
from backend.providers.onnx_embedding import (
    EMBEDDING_DIMENSION,
    ONNXEmbeddingClient,
    ONNXEmbeddingError,
)


def _artifacts(root: Path) -> None:
    files = {
        "onnx/model_fp16.onnx": b"onnx",
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
    }
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    manifest = {
        "schema_version": "1.0",
        "model_id": EMBEDDING_MODEL,
        "revision": EMBEDDING_MODEL_REVISION,
        "files": {
            name: hashlib.sha256(contents).hexdigest()
            for name, contents in files.items()
        },
    }
    (root / "onnx-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _config(
    root: Path,
    *,
    batch_size: int = 2,
    execution_provider: str = "CUDAExecutionProvider",
) -> ONNXModelConfig:
    return ONNXModelConfig(
        model_path=str(root),
        revision=EMBEDDING_MODEL_REVISION,
        onnx_filename="onnx/model_fp16.onnx",
        manifest_filename="onnx-manifest.json",
        max_tokens=8192,
        batch_size=batch_size,
        execution_provider=execution_provider,
        device_id=0,
        output_name="last_hidden_state",
        disable_cpu_fallback=True,
    )


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, np.ndarray]:
        self.calls.append(list(texts))
        assert kwargs == {
            "padding": True,
            "truncation": True,
            "max_length": 8192,
            "return_tensors": "np",
        }
        values = np.arange(1, len(texts) + 1, dtype=np.int64)[:, None]
        return {"input_ids": values, "attention_mask": np.ones_like(values)}


class FakeEmbeddingSession:
    def __init__(
        self,
        *,
        malformed: object | None = None,
        execution_provider: str = "CUDAExecutionProvider",
    ) -> None:
        self.malformed = malformed
        self.execution_provider = execution_provider
        self.calls: list[dict[str, np.ndarray]] = []
        self.disable_fallback_calls = 0

    def get_inputs(self) -> list[object]:
        return [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
        ]

    def get_outputs(self) -> list[object]:
        return [SimpleNamespace(name="last_hidden_state")]

    def get_providers(self) -> list[str]:
        return [self.execution_provider]

    def disable_fallback(self) -> None:
        self.disable_fallback_calls += 1

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[object]:
        assert names == ["last_hidden_state"]
        self.calls.append(feed)
        if self.malformed is not None:
            return [self.malformed]
        batch = feed["input_ids"].shape[0]
        output = np.zeros((batch, 2, EMBEDDING_DIMENSION), dtype=np.float16)
        for index in range(batch):
            value = float(feed["input_ids"][index, 0])
            output[index, 0, index % EMBEDDING_DIMENSION] = value
        return [output]


def test_embedding_is_768_dimensional_finite_normalized_and_batched(tmp_path: Path) -> None:
    _artifacts(tmp_path)
    tokenizer = FakeTokenizer()
    session = FakeEmbeddingSession()
    client = ONNXEmbeddingClient(
        _config(tmp_path), tokenizer=tokenizer, session=session
    )

    vectors = client.embed_many(["one", "two", "three"], model=EMBEDDING_MODEL)

    assert len(vectors) == 3
    assert tokenizer.calls == [["one", "two"], ["three"]]
    assert len(session.calls) == 2
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in vectors)
    assert all(np.isfinite(vector).all() for vector in vectors)
    assert all(np.linalg.norm(vector) == pytest.approx(1.0) for vector in vectors)
    assert vectors[0][0] == 1.0
    assert vectors[1][1] == 1.0


def test_embed_delegates_to_shared_batch_session(tmp_path: Path) -> None:
    _artifacts(tmp_path)
    tokenizer = FakeTokenizer()
    session = FakeEmbeddingSession()
    client = ONNXEmbeddingClient(
        _config(tmp_path), tokenizer=tokenizer, session=session
    )

    first = client.embed("first", model=EMBEDDING_MODEL)
    second = client.embed("second", model=EMBEDDING_MODEL)

    assert first == second
    assert tokenizer.calls == [["first"], ["second"]]
    assert len(session.calls) == 2


def test_embedding_constructs_tokenizer_and_session_once(tmp_path: Path) -> None:
    _artifacts(tmp_path)
    loads: list[tuple[str, dict[str, object]]] = []
    sessions: list[tuple[str, dict[str, object]]] = []
    tokenizer = FakeTokenizer()
    session = FakeEmbeddingSession()

    def tokenizer_loader(path: str, **kwargs: object) -> object:
        loads.append((path, kwargs))
        return tokenizer

    def session_factory(path: str, **kwargs: object) -> object:
        sessions.append((path, kwargs))
        return session

    client = ONNXEmbeddingClient(
        _config(tmp_path),
        tokenizer_loader=tokenizer_loader,
        session_factory=session_factory,
        available_providers=["CUDAExecutionProvider"],
    )
    client.embed("one", model=EMBEDDING_MODEL)
    client.embed("two", model=EMBEDDING_MODEL)

    assert len(loads) == 1
    assert loads[0][1] == {"local_files_only": True, "trust_remote_code": False}
    assert len(sessions) == 1
    assert sessions[0][1]["providers"] == [
        ("CUDAExecutionProvider", {"device_id": "0"})
    ]
    assert session.disable_fallback_calls == 1


def test_embedding_accepts_explicit_cpu_without_cuda_options_or_fallback(
    tmp_path: Path,
) -> None:
    _artifacts(tmp_path)
    calls: list[dict[str, object]] = []
    session = FakeEmbeddingSession(execution_provider="CPUExecutionProvider")

    def session_factory(path: str, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return session

    client = ONNXEmbeddingClient(
        _config(tmp_path, execution_provider="CPUExecutionProvider"),
        tokenizer=FakeTokenizer(),
        session_factory=session_factory,
        available_providers=["CPUExecutionProvider"],
    )
    client.embed("text", model=EMBEDDING_MODEL)

    assert calls[0]["providers"] == ["CPUExecutionProvider"]
    assert session.disable_fallback_calls == 1


@pytest.mark.parametrize(
    ("malformed", "message"),
    [
        (np.zeros((1, 2, 767), dtype=np.float32), "wrong shape"),
        (np.zeros((1, 2, 768), dtype=np.float32), "invalid norm"),
        (np.full((1, 2, 768), np.nan, dtype=np.float32), "non-finite"),
    ],
)
def test_embedding_rejects_malformed_outputs(
    tmp_path: Path, malformed: object, message: str
) -> None:
    _artifacts(tmp_path)
    client = ONNXEmbeddingClient(
        _config(tmp_path),
        tokenizer=FakeTokenizer(),
        session=FakeEmbeddingSession(malformed=malformed),
    )
    with pytest.raises(ONNXEmbeddingError, match=message):
        client.embed("text", model=EMBEDDING_MODEL)


def test_embedding_fails_fast_for_missing_or_tampered_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ONNXEmbeddingError, match="directory does not exist"):
        ONNXEmbeddingClient(
            _config(tmp_path / "missing"),
            tokenizer=FakeTokenizer(),
            session=FakeEmbeddingSession(),
        )

    _artifacts(tmp_path)
    (tmp_path / "tokenizer.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ONNXEmbeddingError, match="hash verification"):
        ONNXEmbeddingClient(
            _config(tmp_path),
            tokenizer=FakeTokenizer(),
            session=FakeEmbeddingSession(),
        )


def test_embedding_requires_configured_provider_and_exact_model(tmp_path: Path) -> None:
    _artifacts(tmp_path)
    with pytest.raises(ONNXEmbeddingError, match="unavailable"):
        ONNXEmbeddingClient(
            _config(tmp_path),
            tokenizer=FakeTokenizer(),
            session_factory=lambda *args, **kwargs: FakeEmbeddingSession(),
            available_providers=["CPUExecutionProvider"],
        )

    client = ONNXEmbeddingClient(
        _config(tmp_path), tokenizer=FakeTokenizer(), session=FakeEmbeddingSession()
    )
    with pytest.raises(ValueError, match="does not match"):
        client.embed("text", model="some-other-model")
