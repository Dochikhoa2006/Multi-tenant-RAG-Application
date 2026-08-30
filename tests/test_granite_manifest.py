from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.create_granite_manifest import REQUIRED_FILES, create_manifest


def _checkpoint_files(root: Path) -> None:
    for index, filename in enumerate(REQUIRED_FILES):
        (root / filename).write_bytes(f"content-{index}".encode())
    (root / "config.json").write_text(
        '{"architectures":["GraniteForCausalLM"],'
        '"model_type":"granite","dtype":"float16"}',
        encoding="utf-8",
    )
    (root / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-00001-of-00002.safetensors",'
        '"b":"model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )


def test_manifest_contains_deterministic_hashes_for_every_required_file(
    tmp_path: Path,
) -> None:
    _checkpoint_files(tmp_path)

    manifest = create_manifest(tmp_path)

    assert manifest["schema_version"] == "1.0"
    files = manifest["files"]
    assert isinstance(files, dict)
    assert set(files) == set(REQUIRED_FILES)
    expected = (tmp_path / "config.json").read_bytes()
    assert files["config.json"] == hashlib.sha256(expected).hexdigest()


def test_manifest_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required files"):
        create_manifest(tmp_path)


def test_manifest_rejects_incompatible_granite_metadata(tmp_path: Path) -> None:
    _checkpoint_files(tmp_path)
    (tmp_path / "config.json").write_text(
        '{"architectures":["OtherModel"],'
        '"model_type":"granite","dtype":"float16"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected FP16 Granite"):
        create_manifest(tmp_path)
