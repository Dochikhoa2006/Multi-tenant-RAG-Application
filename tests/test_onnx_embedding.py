from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import backend.providers.onnx_embedding as onnx_embedding
import scripts.create_onnx_manifest as onnx_manifest
from backend.model_config import (
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
    GTE_EMBEDDING_DIMENSION,
    LATEON_EMBEDDING_DIMENSION,
    LATEON_MODEL,
    LATEON_MODEL_REVISION,
    ONNXModelConfig,
)
from backend.providers.onnx_cuda import CUDA_PROVIDER
from backend.providers.onnx_embedding import (
    EMBEDDING_DIMENSION,
    ONNXEmbeddingClient,
    ONNXEmbeddingError,
    ONNXLateOnError,
    ONNXLateOnProvider,
)


def test_embedding_provider_uses_authoritative_dimension_and_cuda_name() -> None:
    assert EMBEDDING_DIMENSION == GTE_EMBEDDING_DIMENSION
    assert onnx_embedding.CUDA_PROVIDER == CUDA_PROVIDER


def test_lateon_provisioning_and_runtime_share_frozen_identity() -> None:
    assert onnx_manifest._LATEON_MODEL_ID == LATEON_MODEL
    assert onnx_manifest._LATEON_REVISION == LATEON_MODEL_REVISION


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


def _lateon_artifacts(root: Path) -> None:
    skiplist = list(onnx_embedding._LATEON_SKIPLIST)
    files = {
        "model_fp16.onnx": b"fp16-onnx",
        "config.json": b"{}",
        "special_tokens_map.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "onnx_config.json": json.dumps(
            {
                "model_type": "ColBERT",
                "uses_token_type_ids": False,
                "query_prefix": "[Q] ",
                "document_prefix": "[D] ",
                "query_length": 32,
                "document_length": 300,
                "do_query_expansion": False,
                "attend_to_expansion_tokens": False,
                "embedding_dim": 128,
                "query_prefix_id": 50368,
                "document_prefix_id": 50369,
                "mask_token_id": 50284,
                "pad_token_id": 50284,
                "do_lower_case": False,
                "skiplist_words": skiplist,
            }
        ).encode(),
        "config_sentence_transformers.json": json.dumps(
            {
                "model_type": "ColBERT",
                "query_prefix": "[Q] ",
                "document_prefix": "[D] ",
                "query_length": 32,
                "document_length": 300,
                "do_query_expansion": False,
                "attend_to_expansion_tokens": False,
                "similarity_fn_name": "MaxSim",
                "skiplist_words": skiplist,
            }
        ).encode(),
    }
    for name, contents in files.items():
        (root / name).write_bytes(contents)
    manifest = {
        "schema_version": "1.0",
        "model_id": LATEON_MODEL,
        "revision": LATEON_MODEL_REVISION,
        "files": {
            name: hashlib.sha256(contents).hexdigest()
            for name, contents in files.items()
        },
    }
    (root / "onnx-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _lateon_config(root: Path, *, batch_size: int = 2) -> ONNXModelConfig:
    return ONNXModelConfig(
        model_path=str(root),
        revision=LATEON_MODEL_REVISION,
        onnx_filename="model_fp16.onnx",
        manifest_filename="onnx-manifest.json",
        max_tokens=300,
        batch_size=batch_size,
        execution_provider="CUDAExecutionProvider",
        device_id=0,
        output_name="output",
        disable_cpu_fallback=True,
    )


class FakeLateOnTokenizer:
    pad_token_id = 50284

    def __init__(self, *, document_tokens: int = 3) -> None:
        self.document_tokens = document_tokens
        self.calls: list[tuple[object, dict[str, object]]] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "[Q] ":
            return 50368
        if token == "[D] ":
            return 50369
        return 40000 + onnx_embedding._LATEON_SKIPLIST.index(token)

    def __call__(self, texts: object, **kwargs: object) -> dict[str, object]:
        self.calls.append((texts, dict(kwargs)))
        if isinstance(texts, str):
            count = self.document_tokens
            return {"input_ids": [50281, *range(100, 100 + count), 50282]}
        values = list(texts)  # type: ignore[arg-type]
        sequences = [[50281, 100 + index, 50282] for index, _ in enumerate(values)]
        width = max(len(item) for item in sequences)
        ids = np.full((len(values), width), self.pad_token_id, dtype=np.int64)
        mask = np.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            ids[index, : len(sequence)] = sequence
            mask[index, : len(sequence)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class FakeLateOnSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []
        self.disable_fallback_calls = 0

    def get_inputs(self) -> list[object]:
        return [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
        ]

    def get_outputs(self) -> list[object]:
        return [SimpleNamespace(name="output", type="tensor(float16)")]

    def get_providers(self) -> list[str]:
        return ["CUDAExecutionProvider"]

    def disable_fallback(self) -> None:
        self.disable_fallback_calls += 1

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[object]:
        assert names == ["output"]
        self.calls.append(feed)
        batch, tokens = feed["input_ids"].shape
        output = np.zeros((batch, tokens, LATEON_EMBEDDING_DIMENSION), dtype=np.float16)
        output[:, :, 0] = 1.0
        return [output]


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


def test_embedding_close_is_idempotent_and_releases_runtime(tmp_path: Path) -> None:
    _artifacts(tmp_path)
    client = ONNXEmbeddingClient(
        _config(tmp_path),
        tokenizer=FakeTokenizer(),
        session=FakeEmbeddingSession(),
    )

    client.close()
    client.close()

    with pytest.raises(RuntimeError, match="closed"):
        client.embed("text", model=EMBEDDING_MODEL)


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
    assert sessions[0][1]["enable_fallback"] is False
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


def test_embedding_records_placement_and_disables_runtime_recovery_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _artifacts(tmp_path)
    session = FakeEmbeddingSession()
    captured: dict[str, object] = {}

    class FakeSessionOptions:
        def __init__(self) -> None:
            self.entries: dict[str, str] = {}

        def add_session_config_entry(self, name: str, value: str) -> None:
            self.entries[name] = value

    def session_factory(path: str, **kwargs: object) -> object:
        captured.update(kwargs)
        return session

    fake_runtime = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        InferenceSession=session_factory,
        get_available_providers=lambda: ["CUDAExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_runtime)
    monkeypatch.setattr(
        onnx_embedding,
        "validate_cuda_placement",
        lambda *args, **kwargs: SimpleNamespace(cpu_count=0, cpu_operators=()),
    )

    ONNXEmbeddingClient(_config(tmp_path), tokenizer=FakeTokenizer())

    options = captured["sess_options"]
    assert isinstance(options, FakeSessionOptions)
    assert options.entries == {"session.record_ep_graph_assignment_info": "1"}
    assert captured["enable_fallback"] is False
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


def test_lateon_reuses_one_session_batches_and_preserves_document_order(
    tmp_path: Path,
) -> None:
    _lateon_artifacts(tmp_path)
    tokenizer = FakeLateOnTokenizer()
    session = FakeLateOnSession()
    provider = ONNXLateOnProvider(
        _lateon_config(tmp_path, batch_size=2),
        tokenizer=tokenizer,
        session=session,
        warmup=False,
    )

    documents = provider.encode_documents([" first ", "second", "third"])
    query = provider.encode_query(" query ")

    assert len(documents) == 3
    assert len(session.calls) == 3
    assert all(len(row) == LATEON_EMBEDDING_DIMENSION for matrix in documents for row in matrix)
    assert all(np.linalg.norm(row) == pytest.approx(1.0) for matrix in documents for row in matrix)
    assert session.calls[0]["input_ids"][:, 1].tolist() == [50369, 50369]
    assert session.calls[1]["input_ids"][:, 1].tolist() == [50369]
    assert session.calls[2]["input_ids"][:, 1].tolist() == [50368]
    assert len(query[0]) == LATEON_EMBEDDING_DIMENSION
    assert session.disable_fallback_calls == 1
    batch_calls = [call for call in tokenizer.calls if isinstance(call[0], list)]
    assert batch_calls[0] == (
        ["first", "second"],
        {
            "add_special_tokens": True,
            "padding": True,
            "truncation": "longest_first",
            "max_length": 299,
            "return_tensors": "np",
        },
    )
    assert batch_calls[-1][1]["max_length"] == 31


def test_lateon_golden_prefix_special_token_and_retained_mask_contract(
    tmp_path: Path,
) -> None:
    _lateon_artifacts(tmp_path)
    tokenizer = FakeLateOnTokenizer()
    session = FakeLateOnSession()
    provider = ONNXLateOnProvider(
        _lateon_config(tmp_path),
        tokenizer=tokenizer,
        session=session,
        warmup=False,
    )

    provider.encode_query("  query  ")
    provider.encode_documents(["  document  "])

    assert session.calls[0]["input_ids"].tolist() == [[50281, 50368, 100, 50282]]
    assert session.calls[0]["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert session.calls[1]["input_ids"].tolist() == [[50281, 50369, 100, 50282]]
    assert session.calls[1]["attention_mask"].tolist() == [[1, 1, 1, 1]]


def test_lateon_rejects_oversized_documents_before_onnx_inference(
    tmp_path: Path,
) -> None:
    _lateon_artifacts(tmp_path)
    tokenizer = FakeLateOnTokenizer(document_tokens=298)
    session = FakeLateOnSession()
    provider = ONNXLateOnProvider(
        _lateon_config(tmp_path),
        tokenizer=tokenizer,
        session=session,
        warmup=False,
    )

    with pytest.raises(ONNXLateOnError, match="300-model-token"):
        provider.encode_documents(["oversized"])
    assert session.calls == []


def test_lateon_artifacts_are_pinned_and_hash_verified(tmp_path: Path) -> None:
    _lateon_artifacts(tmp_path)
    (tmp_path / "onnx_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ONNXLateOnError, match="hash verification"):
        ONNXLateOnProvider(
            _lateon_config(tmp_path),
            tokenizer=FakeLateOnTokenizer(),
            session=FakeLateOnSession(),
            warmup=False,
        )


def test_lateon_constructs_cuda_only_with_recovery_disabled_and_validates_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _lateon_artifacts(tmp_path)
    session = FakeLateOnSession()
    captured: dict[str, object] = {}

    class FakeSessionOptions:
        def __init__(self) -> None:
            self.entries: dict[str, str] = {}

        def add_session_config_entry(self, name: str, value: str) -> None:
            self.entries[name] = value

    def session_factory(path: str, **kwargs: object) -> object:
        captured.update(kwargs)
        return session

    fake_runtime = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        InferenceSession=session_factory,
        get_available_providers=lambda: ["CUDAExecutionProvider"],
    )
    placement: dict[str, object] = {}

    def validate(*args: object, **kwargs: object) -> object:
        placement.update(kwargs)
        return SimpleNamespace(cpu_count=2, cpu_operators=(("Shape", 2),))

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_runtime)
    monkeypatch.setattr(
        onnx_embedding, "_EXPECTED_LATEON_CPU_ASSIGNMENT_SHA256", "verified-digest"
    )
    monkeypatch.setattr(onnx_embedding, "validate_cuda_placement", validate)

    ONNXLateOnProvider(
        _lateon_config(tmp_path),
        tokenizer=FakeLateOnTokenizer(),
        warmup=False,
    )

    assert captured["providers"] == [
        ("CUDAExecutionProvider", {"device_id": "0"})
    ]
    assert captured["enable_fallback"] is False
    assert captured["sess_options"].entries == {
        "session.record_ep_graph_assignment_info": "1"
    }
    assert placement["expected_cpu_digest"] == "verified-digest"
    assert placement["required_cuda_operators"] == frozenset({"MatMul", "Softmax"})
    assert session.disable_fallback_calls == 1


def test_lateon_production_cpu_assignment_is_bound_to_profiled_graph() -> None:
    assert onnx_embedding._EXPECTED_LATEON_CPU_ASSIGNMENT_SHA256 == (
        "53dbd9af5e31b81d735e0d7bc26118edff611a0b430627b0c3f87663a9f7a2a3"
    )


def test_generic_onnx_manifest_cli_behavior_remains_compatible(tmp_path: Path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"generic")

    manifest_path = onnx_manifest.create_manifest(
        tmp_path, "example/model", "immutable-revision"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "schema_version": "1.0",
        "model_id": "example/model",
        "revision": "immutable-revision",
        "files": {"model.onnx": hashlib.sha256(b"generic").hexdigest()},
    }


def test_lateon_provisioning_metadata_requires_exact_pinned_fp32_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _lateon_artifacts(tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"published-fp32")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "modules.json").write_text("[]", encoding="utf-8")
    (tmp_path / "sentence_bert_config.json").write_text("{}", encoding="utf-8")
    for directory in ("1_Dense", "2_Dense", "3_Dense"):
        path = tmp_path / directory
        path.mkdir()
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        onnx_manifest,
        "_LATEON_FP32_SHA256",
        hashlib.sha256(b"published-fp32").hexdigest(),
    )

    metadata = onnx_manifest._validate_lateon_metadata(tmp_path)
    assert metadata["embedding_dim"] == 128

    (tmp_path / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="FP32 model.onnx SHA256"):
        onnx_manifest._validate_lateon_metadata(tmp_path)
