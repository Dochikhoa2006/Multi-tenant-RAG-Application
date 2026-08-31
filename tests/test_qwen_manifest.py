from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.create_qwen_manifest import REQUIRED_FILES, create_manifest


def _checkpoint(root: Path) -> None:
    for index, name in enumerate(REQUIRED_FILES):
        path = root / name
        path.write_bytes(f"content-{index}".encode())
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "torch_dtype": "float16",
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                    "version": "gemm",
                    "zero_point": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": (
                    "{% if enable_thinking is false %}"
                    "<|im_start|>assistant{% endif %}"
                )
            }
        ),
        encoding="utf-8",
    )


def test_manifest_hashes_complete_qwen_awq_checkpoint(tmp_path: Path) -> None:
    _checkpoint(tmp_path)

    manifest = create_manifest(tmp_path)

    assert manifest["schema_version"] == "1.0"
    assert manifest["model_id"] == "qwen3-4b-awq"
    files = manifest["files"]
    assert isinstance(files, dict)
    assert set(files) == set(REQUIRED_FILES)
    assert files["model.safetensors"] == hashlib.sha256(
        (tmp_path / "model.safetensors").read_bytes()
    ).hexdigest()


def test_manifest_rejects_missing_checkpoint_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required files"):
        create_manifest(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("architectures", ["OtherModel"]),
        ("model_type", "other"),
        ("torch_dtype", "bfloat16"),
    ],
)
def test_manifest_rejects_wrong_qwen_architecture(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    _checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="expected Qwen3 AWQ"):
        create_manifest(tmp_path)


def test_manifest_rejects_wrong_awq_contract_or_chat_template(tmp_path: Path) -> None:
    _checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["quantization_config"]["bits"] = 8
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="expected Qwen3 AWQ"):
        create_manifest(tmp_path)

    _checkpoint(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "<|im_start|>assistant"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="thinking-aware"):
        create_manifest(tmp_path)
