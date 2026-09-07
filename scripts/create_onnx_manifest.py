"""Create the checksummed manifest required by the local ONNX providers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np


_MANIFEST_NAME = "onnx-manifest.json"
_LATEON_MODEL_ID = "lightonai/LateOn"
_LATEON_REVISION = "62911e105059585d244384c7d17826e35f669c17"
_PYLATE_VERSION = "1.3.4"
_PYLATE_COMMIT = "b1453fda370137f14e02fc907e388f2532b4343b"
_LATEON_FP32_SHA256 = "abe0aa8dd3e0b5cff74657273e5b767f615077e21b732e3094838241cb1c0e29"
_LATEON_PRODUCTION_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "onnx_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_LATEON_REFERENCE_FILES = (
    *_LATEON_PRODUCTION_FILES,
    "model.onnx",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "1_Dense/config.json",
    "1_Dense/model.safetensors",
    "2_Dense/config.json",
    "2_Dense/model.safetensors",
    "3_Dense/config.json",
    "3_Dense/model.safetensors",
)
_PARITY_QUERIES = (
    "Production deployment manager approval rollback checklist",
    "Ripe yellow bananas contain potassium",
)
_PARITY_DOCUMENTS = (
    "Production deployment requires explicit manager approval and a completed rollback checklist.",
    "A ripe yellow banana is a fruit that contains potassium.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_manifest(model_directory: Path, model_id: str, revision: str) -> Path:
    root = model_directory.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("model_directory must be an existing directory")
    if not model_id.strip() or not revision.strip():
        raise ValueError("model_id and revision must not be empty")
    output_path = root / _MANIFEST_NAME
    files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != output_path
    }
    if not files:
        raise ValueError("model_directory contains no artifacts")
    manifest = {
        "schema_version": "1.0",
        "model_id": model_id,
        "revision": revision,
        "files": files,
    }
    temporary = root / f".{_MANIFEST_NAME}.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return output_path


def _object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _validate_lateon_metadata(source: Path) -> Mapping[str, object]:
    missing = [name for name in _LATEON_REFERENCE_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError("pinned LateOn snapshot is incomplete: " + ", ".join(missing))
    if _sha256(source / "model.onnx") != _LATEON_FP32_SHA256:
        raise ValueError("published LateOn FP32 model.onnx SHA256 does not match")
    onnx_config = _object(source / "onnx_config.json")
    sentence_config = _object(source / "config_sentence_transformers.json")
    expected_onnx = {
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
    }
    if any(onnx_config.get(key) != value for key, value in expected_onnx.items()):
        raise ValueError("pinned LateOn ONNX metadata is incompatible")
    expected_sentence = {
        "model_type": "ColBERT",
        "query_prefix": "[Q] ",
        "document_prefix": "[D] ",
        "query_length": 32,
        "document_length": 300,
        "do_query_expansion": False,
        "attend_to_expansion_tokens": False,
        "similarity_fn_name": "MaxSim",
    }
    if any(sentence_config.get(key) != value for key, value in expected_sentence.items()):
        raise ValueError("pinned LateOn sentence-transformers metadata is incompatible")
    if onnx_config.get("skiplist_words") != sentence_config.get("skiplist_words"):
        raise ValueError("pinned LateOn skiplist metadata disagrees")
    return onnx_config


def _download_lateon(source: Path) -> None:
    from huggingface_hub import snapshot_download

    for attempt in range(4):
        try:
            snapshot_download(
                repo_id=_LATEON_MODEL_ID,
                revision=_LATEON_REVISION,
                local_dir=source,
                allow_patterns=list(_LATEON_REFERENCE_FILES),
            )
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def _tokenize_like_pylate(
    tokenizer: object,
    texts: Sequence[str],
    *,
    is_query: bool,
    metadata: Mapping[str, object],
) -> dict[str, np.ndarray]:
    limit = int(metadata["query_length"] if is_query else metadata["document_length"])
    values = tokenizer(  # type: ignore[operator]
        [text.strip() for text in texts],
        padding=True,
        truncation="longest_first",
        return_tensors="np",
        max_length=limit - 1,
    )
    input_ids = np.asarray(values["input_ids"])
    attention_mask = np.asarray(values["attention_mask"])
    prefix_id = int(
        metadata["query_prefix_id"] if is_query else metadata["document_prefix_id"]
    )
    return {
        "input_ids": np.concatenate(
            (input_ids[:, :1], np.full((len(texts), 1), prefix_id), input_ids[:, 1:]),
            axis=1,
        ).astype(input_ids.dtype),
        "attention_mask": np.concatenate(
            (
                attention_mask[:, :1],
                np.ones((len(texts), 1), dtype=attention_mask.dtype),
                attention_mask[:, 1:],
            ),
            axis=1,
        ),
    }


def _retained_masks(
    tokenized: Mapping[str, np.ndarray],
    *,
    is_query: bool,
    skiplist: frozenset[int],
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for ids, attention in zip(
        tokenized["input_ids"], tokenized["attention_mask"], strict=True
    ):
        mask = attention.astype(bool)
        if not is_query:
            mask &= ~np.isin(ids, tuple(skiplist))
        masks.append(mask)
    return masks


def _onnx_embeddings(
    session: object,
    tokenized: Mapping[str, np.ndarray],
    masks: Sequence[np.ndarray],
) -> list[np.ndarray]:
    input_types = {item.name: item.type for item in session.get_inputs()}  # type: ignore[attr-defined]
    dtypes = {"tensor(int32)": np.int32, "tensor(int64)": np.int64}
    feed = {
        name: np.asarray(value, dtype=dtypes[input_types[name]])
        for name, value in tokenized.items()
    }
    output = np.asarray(session.run(["output"], feed)[0], dtype=np.float32)  # type: ignore[attr-defined]
    if output.ndim != 3 or output.shape[:2] != feed["input_ids"].shape or output.shape[2] != 128:
        raise ValueError("LateOn ONNX output shape is incompatible")
    if not np.isfinite(output).all():
        raise ValueError("LateOn ONNX output contains non-finite values")
    return [row[mask] for row, mask in zip(output, masks, strict=True)]


def _maxsim(query: np.ndarray, document: np.ndarray) -> float:
    return float(np.max(query @ document.T, axis=1).sum())


def _assert_embedding_parity(
    reference: Sequence[np.ndarray],
    candidate: Sequence[np.ndarray],
    *,
    label: str,
) -> None:
    if len(reference) != len(candidate):
        raise ValueError(f"{label} changed batch order")
    for expected, actual in zip(reference, candidate, strict=True):
        if expected.shape != actual.shape:
            raise ValueError(f"{label} changed retained-token output shape")
        denominators = np.linalg.norm(expected, axis=1) * np.linalg.norm(actual, axis=1)
        cosines = np.sum(expected * actual, axis=1) / denominators
        if not np.isfinite(cosines).all() or float(np.min(cosines)) < 0.999:
            raise ValueError(f"{label} corresponding-token cosine parity failed")


def _assert_maxsim_and_ranking_parity(
    reference_queries: Sequence[np.ndarray],
    reference_documents: Sequence[np.ndarray],
    candidate_queries: Sequence[np.ndarray],
    candidate_documents: Sequence[np.ndarray],
    *,
    label: str,
) -> None:
    reference_scores = np.asarray(
        [[_maxsim(query, document) for document in reference_documents] for query in reference_queries]
    )
    candidate_scores = np.asarray(
        [[_maxsim(query, document) for document in candidate_documents] for query in candidate_queries]
    )
    denominator = np.maximum(np.abs(reference_scores), 1e-6)
    if float(np.max(np.abs(candidate_scores - reference_scores) / denominator)) > 0.005:
        raise ValueError(f"{label} MaxSim relative-error parity failed")
    for expected, actual in zip(reference_scores, candidate_scores, strict=True):
        ordered = np.sort(expected)
        if len(ordered) > 1 and float(np.min(np.diff(ordered))) <= 0.05:
            raise ValueError("ranking parity fixture contains a numerically indistinguishable near-tie")
        if not np.array_equal(np.argsort(-expected), np.argsort(-actual)):
            raise ValueError(f"{label} changed the exact fixture ranking")


def _cuda_session(model_path: Path) -> object:
    import onnxruntime as ort

    # The pinned GPU wheel supplies its CUDA 13/cuDNN user-space libraries as
    # Python packages.  Load those directories before constructing the EP so
    # the dynamic linker does not depend on image-specific LD_LIBRARY_PATH
    # mutation.  This is library discovery only; CUDA remains the sole
    # requested provider and ORT recovery fallback stays disabled below.
    ort.preload_dlls(directory="")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("LateOn provisioning parity requires CUDAExecutionProvider")
    session = ort.InferenceSession(
        str(model_path),
        providers=[("CUDAExecutionProvider", {"device_id": "0"})],
        enable_fallback=False,
    )
    session.disable_fallback()
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("LateOn provisioning session did not activate CUDA")
    return session


def _pylate_reference(source: Path) -> tuple[object, list[np.ndarray], list[np.ndarray]]:
    import pylate
    from pylate import models

    if getattr(pylate, "__version__", _PYLATE_VERSION) != _PYLATE_VERSION:
        raise RuntimeError(
            f"PyLate {_PYLATE_VERSION} at {_PYLATE_COMMIT} is required for parity"
        )
    model = models.ColBERT(
        str(source),
        device="cuda",
        local_files_only=True,
        trust_remote_code=False,
    )
    queries = [
        np.asarray(item, dtype=np.float32)
        for item in model.encode(
            list(_PARITY_QUERIES),
            is_query=True,
            batch_size=len(_PARITY_QUERIES),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    ]
    documents = [
        np.asarray(item, dtype=np.float32)
        for item in model.encode(
            list(_PARITY_DOCUMENTS),
            is_query=False,
            batch_size=len(_PARITY_DOCUMENTS),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    ]
    return model, queries, documents


def _verify_tokenization_parity(
    model: object,
    tokenizer: object,
    metadata: Mapping[str, object],
) -> None:
    for texts, is_query in (
        (_PARITY_QUERIES, True),
        (_PARITY_DOCUMENTS, False),
    ):
        reference = model.tokenize(list(texts), is_query=is_query)  # type: ignore[attr-defined]
        candidate = _tokenize_like_pylate(
            tokenizer, texts, is_query=is_query, metadata=metadata
        )
        for name in ("input_ids", "attention_mask"):
            expected = reference[name].detach().cpu().numpy()
            if expected.shape != candidate[name].shape or not np.array_equal(
                expected, candidate[name]
            ):
                raise ValueError(f"LateOn {name} golden tokenization parity failed")
        if is_query:
            reference_masks = reference["attention_mask"].bool()
        else:
            reference_masks = model.skiplist_mask(  # type: ignore[attr-defined]
                reference["input_ids"], model.skiplist  # type: ignore[attr-defined]
            ) & reference["attention_mask"].bool()
        candidate_masks = _retained_masks(
            candidate,
            is_query=is_query,
            skiplist=frozenset(int(item) for item in model.skiplist),  # type: ignore[attr-defined]
        )
        for expected, actual in zip(reference_masks, candidate_masks, strict=True):
            if not np.array_equal(expected.detach().cpu().numpy(), actual):
                raise ValueError("LateOn retained-token mask golden parity failed")


def _parity_outputs(
    session: object,
    tokenizer: object,
    metadata: Mapping[str, object],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    skiplist = frozenset(
        int(tokenizer.convert_tokens_to_ids(token))  # type: ignore[attr-defined]
        for token in metadata["skiplist_words"]  # type: ignore[union-attr]
    )
    outputs: list[list[np.ndarray]] = []
    for texts, is_query in (
        (_PARITY_QUERIES, True),
        (_PARITY_DOCUMENTS, False),
    ):
        tokenized = _tokenize_like_pylate(
            tokenizer, texts, is_query=is_query, metadata=metadata
        )
        masks = _retained_masks(tokenized, is_query=is_query, skiplist=skiplist)
        outputs.append(_onnx_embeddings(session, tokenized, masks))
    return outputs[0], outputs[1]


def provision_lateon(output_directory: Path) -> Path:
    """Provision the pinned, parity-gated FP16 LateOn artifact atomically."""

    output = output_directory.expanduser().resolve()
    if output.exists():
        raise ValueError("LateOn output directory must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lateon-source-") as raw_source:
        source = Path(raw_source)
        _download_lateon(source)
        metadata = _validate_lateon_metadata(source)

        from transformers import AutoTokenizer
        import onnx
        from onnxruntime.transformers import float16

        tokenizer = AutoTokenizer.from_pretrained(
            source, local_files_only=True, trust_remote_code=False
        )
        model, reference_queries, reference_documents = _pylate_reference(source)
        _verify_tokenization_parity(model, tokenizer, metadata)

        fp32_session = _cuda_session(source / "model.onnx")
        fp32_queries, fp32_documents = _parity_outputs(
            fp32_session, tokenizer, metadata
        )
        _assert_embedding_parity(reference_queries, fp32_queries, label="FP32 ONNX")
        _assert_embedding_parity(reference_documents, fp32_documents, label="FP32 ONNX")
        _assert_maxsim_and_ranking_parity(
            reference_queries,
            reference_documents,
            fp32_queries,
            fp32_documents,
            label="FP32 ONNX",
        )

        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}-stage-", dir=output.parent
        ) as raw_stage:
            stage = Path(raw_stage)
            converted_path = stage / "model_fp16.onnx"
            converted = float16.convert_float_to_float16(
                onnx.load(source / "model.onnx"),
                keep_io_types=False,
            )
            onnx.save(converted, converted_path)
            for name in _LATEON_PRODUCTION_FILES:
                shutil.copy2(source / name, stage / name)

            fp16_session = _cuda_session(converted_path)
            fp16_queries, fp16_documents = _parity_outputs(
                fp16_session, tokenizer, metadata
            )
            _assert_embedding_parity(reference_queries, fp16_queries, label="FP16 ONNX")
            _assert_embedding_parity(reference_documents, fp16_documents, label="FP16 ONNX")
            _assert_maxsim_and_ranking_parity(
                reference_queries,
                reference_documents,
                fp16_queries,
                fp16_documents,
                label="FP16 ONNX",
            )
            create_manifest(stage, _LATEON_MODEL_ID, _LATEON_REVISION)
            published = output.parent / f".{output.name}-publish"
            if published.exists():
                raise ValueError("stale LateOn publication path exists")
            os.replace(stage, published)
            os.replace(published, output)
    return output / _MANIFEST_NAME


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "provision-lateon":
        parser = argparse.ArgumentParser(description="Provision pinned LateOn FP16 artifacts")
        parser.add_argument("command")
        parser.add_argument("output_directory", type=Path)
        arguments = parser.parse_args()
        provision_lateon(arguments.output_directory)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("model_id")
    parser.add_argument("revision")
    arguments = parser.parse_args()
    create_manifest(
        arguments.model_directory,
        arguments.model_id,
        arguments.revision,
    )


if __name__ == "__main__":
    main()


__all__ = ["create_manifest", "provision_lateon"]
