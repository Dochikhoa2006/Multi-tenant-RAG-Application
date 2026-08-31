from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.model_config import ONNXModelConfig, RERANKER_MODEL, RERANKER_MODEL_REVISION
from backend.providers.onnx_reranker import (
    ONNXCrossEncoderReranker,
    ONNXRerankerError,
)


def _artifacts(root: Path) -> None:
    files = {
        "model_fp16.onnx": b"onnx",
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
    }
    for name, contents in files.items():
        (root / name).write_bytes(contents)
    manifest = {
        "schema_version": "1.0",
        "model_id": RERANKER_MODEL,
        "revision": RERANKER_MODEL_REVISION,
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
        revision=RERANKER_MODEL_REVISION,
        onnx_filename="model_fp16.onnx",
        manifest_filename="onnx-manifest.json",
        max_tokens=512,
        batch_size=batch_size,
        execution_provider=execution_provider,
        device_id=0,
        output_name="logits",
        disable_cpu_fallback=True,
    )


class FakePairTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def __call__(
        self, pairs: list[tuple[str, str]], **kwargs: object
    ) -> dict[str, np.ndarray]:
        self.calls.append(list(pairs))
        assert kwargs == {
            "padding": True,
            "truncation": True,
            "max_length": 512,
            "return_tensors": "np",
        }
        values = np.arange(len(pairs), dtype=np.int64)[:, None]
        return {"input_ids": values, "attention_mask": np.ones_like(values)}


class FakeRerankerSession:
    def __init__(
        self,
        batches: list[object],
        *,
        execution_provider: str = "CUDAExecutionProvider",
    ) -> None:
        self.batches = list(batches)
        self.execution_provider = execution_provider
        self.calls = 0
        self.disable_fallback_calls = 0

    def get_inputs(self) -> list[object]:
        return [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
        ]

    def get_outputs(self) -> list[object]:
        return [SimpleNamespace(name="logits")]

    def get_providers(self) -> list[str]:
        return [self.execution_provider]

    def disable_fallback(self) -> None:
        self.disable_fallback_calls += 1

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[object]:
        assert names == ["logits"]
        assert feed["input_ids"].dtype == np.int64
        self.calls += 1
        return [self.batches.pop(0)]


def test_reranker_batches_pairs_and_preserves_original_indices(tmp_path: Path) -> None:
    _artifacts(tmp_path)
    tokenizer = FakePairTokenizer()
    session = FakeRerankerSession(
        [
            np.array([[0.2], [0.9]], dtype=np.float16),
            np.array([[0.9]], dtype=np.float16),
        ]
    )
    reranker = ONNXCrossEncoderReranker(
        _config(tmp_path), tokenizer=tokenizer, session=session
    )

    results = reranker.rerank(
        "query", ["zero", "one", "two"], model=RERANKER_MODEL, top_n=2
    )

    assert [result.index for result in results] == [1, 2]
    assert [result.score for result in results] == pytest.approx([0.9, 0.9], abs=0.001)
    assert tokenizer.calls == [
        [("query", "zero"), ("query", "one")],
        [("query", "two")],
    ]
    assert session.calls == 2


def test_reranker_reuses_loaded_tokenizer_and_session(tmp_path: Path) -> None:
    _artifacts(tmp_path)
    loads: list[object] = []
    sessions: list[object] = []
    tokenizer = FakePairTokenizer()
    session = FakeRerankerSession(
        [np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)]
    )

    def tokenizer_loader(*args: object, **kwargs: object) -> object:
        loads.append((args, kwargs))
        return tokenizer

    def session_factory(*args: object, **kwargs: object) -> object:
        sessions.append((args, kwargs))
        return session

    reranker = ONNXCrossEncoderReranker(
        _config(tmp_path),
        tokenizer_loader=tokenizer_loader,
        session_factory=session_factory,
        available_providers=["CUDAExecutionProvider"],
    )
    reranker.rerank("q", ["a"], model=RERANKER_MODEL, top_n=1)
    reranker.rerank("q", ["b"], model=RERANKER_MODEL, top_n=1)

    assert len(loads) == 1
    assert len(sessions) == 1
    assert sessions[0][1]["providers"] == [
        ("CUDAExecutionProvider", {"device_id": "0"})
    ]
    assert session.disable_fallback_calls == 1


def test_reranker_close_is_idempotent_and_releases_runtime(tmp_path: Path) -> None:
    _artifacts(tmp_path)
    reranker = ONNXCrossEncoderReranker(
        _config(tmp_path),
        tokenizer=FakePairTokenizer(),
        session=FakeRerankerSession([np.array([1.0], dtype=np.float32)]),
    )

    reranker.close()
    reranker.close()

    with pytest.raises(RuntimeError, match="closed"):
        reranker.rerank("q", ["doc"], model=RERANKER_MODEL, top_n=1)


def test_reranker_accepts_another_explicit_provider_without_cuda_options(
    tmp_path: Path,
) -> None:
    _artifacts(tmp_path)
    calls: list[dict[str, object]] = []
    session = FakeRerankerSession(
        [np.array([0.4], dtype=np.float32)],
        execution_provider="OpenVINOExecutionProvider",
    )

    def session_factory(path: str, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return session

    reranker = ONNXCrossEncoderReranker(
        _config(tmp_path, execution_provider="OpenVINOExecutionProvider"),
        tokenizer=FakePairTokenizer(),
        session_factory=session_factory,
        available_providers=["OpenVINOExecutionProvider"],
    )
    reranker.rerank("q", ["document"], model=RERANKER_MODEL, top_n=1)

    assert calls[0]["providers"] == ["OpenVINOExecutionProvider"]
    assert session.disable_fallback_calls == 1


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.zeros((2, 2), dtype=np.float32), "wrong shape"),
        (np.array([0.1, np.nan], dtype=np.float32), "non-finite"),
    ],
)
def test_reranker_rejects_malformed_logits(
    tmp_path: Path, output: object, message: str
) -> None:
    _artifacts(tmp_path)
    reranker = ONNXCrossEncoderReranker(
        _config(tmp_path),
        tokenizer=FakePairTokenizer(),
        session=FakeRerankerSession([output]),
    )
    with pytest.raises(ONNXRerankerError, match=message):
        reranker.rerank("q", ["a", "b"], model=RERANKER_MODEL, top_n=2)


def test_reranker_fails_fast_without_files_or_configured_provider(tmp_path: Path) -> None:
    with pytest.raises(ONNXRerankerError, match="directory does not exist"):
        ONNXCrossEncoderReranker(
            _config(tmp_path / "missing"),
            tokenizer=FakePairTokenizer(),
            session=FakeRerankerSession([]),
        )

    _artifacts(tmp_path)
    with pytest.raises(ONNXRerankerError, match="unavailable"):
        ONNXCrossEncoderReranker(
            _config(tmp_path),
            tokenizer=FakePairTokenizer(),
            session_factory=lambda *args, **kwargs: FakeRerankerSession([]),
            available_providers=["CPUExecutionProvider"],
        )


def test_reranker_validates_model_top_n_and_empty_documents(tmp_path: Path) -> None:
    _artifacts(tmp_path)
    session = FakeRerankerSession([])
    reranker = ONNXCrossEncoderReranker(
        _config(tmp_path), tokenizer=FakePairTokenizer(), session=session
    )

    assert reranker.rerank("q", [], model=RERANKER_MODEL, top_n=4) == []
    assert session.calls == 0
    with pytest.raises(ValueError, match="does not match"):
        reranker.rerank("q", ["doc"], model="other", top_n=1)
    with pytest.raises(ValueError, match="greater than zero"):
        reranker.rerank("q", ["doc"], model=RERANKER_MODEL, top_n=0)
