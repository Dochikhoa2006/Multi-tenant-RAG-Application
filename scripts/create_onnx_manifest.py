"""Create the checksummed manifest required by the local ONNX providers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


_MANIFEST_NAME = "onnx-manifest.json"


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


def main() -> None:
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


__all__ = ["create_manifest"]
