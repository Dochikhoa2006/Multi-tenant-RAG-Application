"""Create the immutable hash manifest consumed by the Qwen SGLang worker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_FILES = (
    "LICENSE",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
EXPECTED_ARCHITECTURE = "Qwen3ForCausalLM"
EXPECTED_MODEL_TYPE = "qwen3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        tokenizer = json.loads(
            (root / "tokenizer_config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Qwen model metadata is malformed") from exc
    if not isinstance(config, dict) or not isinstance(tokenizer, dict):
        raise ValueError("Qwen model metadata must contain JSON objects")
    return config, tokenizer


def _validate_metadata(root: Path) -> None:
    config, tokenizer = _metadata(root)
    quantization = config.get("quantization_config")
    if (
        config.get("architectures") != [EXPECTED_ARCHITECTURE]
        or config.get("model_type") != EXPECTED_MODEL_TYPE
        or config.get("torch_dtype") != "float16"
        or config.get("auto_map") is not None
        or not isinstance(quantization, dict)
        or quantization.get("quant_method") != "awq"
        or quantization.get("bits") != 4
        or quantization.get("group_size") != 128
        or quantization.get("version") != "gemm"
        or quantization.get("zero_point") is not True
    ):
        raise ValueError("model metadata is not the expected Qwen3 AWQ checkpoint")
    chat_template = tokenizer.get("chat_template")
    if (
        not isinstance(chat_template, str)
        or "enable_thinking" not in chat_template
        or "<|im_start|>assistant" not in chat_template
        or tokenizer.get("auto_map") is not None
    ):
        raise ValueError("Qwen tokenizer does not contain the native thinking-aware template")


def create_manifest(model_directory: Path) -> dict[str, object]:
    root = model_directory.resolve()
    if not root.is_dir():
        raise ValueError("model directory does not exist")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"model directory is missing required files: {', '.join(missing)}")
    _validate_metadata(root)
    return {
        "schema_version": "1.0",
        "model_id": "qwen3-4b-awq",
        "files": {name: _sha256(root / name) for name in REQUIRED_FILES},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="defaults to MODEL_DIRECTORY/qwen-manifest.json",
    )
    arguments = parser.parse_args()
    output = arguments.output or arguments.model_directory / "qwen-manifest.json"
    manifest = create_manifest(arguments.model_directory)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


__all__ = ["REQUIRED_FILES", "create_manifest"]
