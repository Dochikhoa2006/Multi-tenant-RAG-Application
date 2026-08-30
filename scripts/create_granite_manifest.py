"""Create the immutable hash manifest consumed by the Modal SGLang service."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
EXPECTED_ARCHITECTURE = "GraniteForCausalLM"
EXPECTED_MODEL_TYPE = "granite"
EXPECTED_SHARDS = {
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_metadata(root: Path) -> None:
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        index = json.loads(
            (root / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("model metadata is malformed") from exc
    if (
        not isinstance(config, dict)
        or config.get("architectures") != [EXPECTED_ARCHITECTURE]
        or config.get("model_type") != EXPECTED_MODEL_TYPE
        or config.get("dtype", config.get("torch_dtype")) != "float16"
    ):
        raise ValueError("model metadata is not the expected FP16 Granite architecture")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index has no weight map")
    shard_names = set(weight_map.values())
    if shard_names != EXPECTED_SHARDS:
        raise ValueError("model index must reference exactly the two expected shards")


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
        "files": {name: _sha256(root / name) for name in REQUIRED_FILES},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="defaults to MODEL_DIRECTORY/granite-manifest.json",
    )
    arguments = parser.parse_args()
    output = arguments.output or arguments.model_directory / "granite-manifest.json"
    manifest = create_manifest(arguments.model_directory)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
