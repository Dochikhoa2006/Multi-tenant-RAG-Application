from __future__ import annotations

import ast
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENTS = (
    ("modal_sglang.py", "GraniteSGLangServer"),
    ("modal_qwen_sglang.py", "QwenSGLangServer"),
)


def _server_keywords(path: Path, class_name: str) -> dict[str, ast.expr]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "server"
            ):
                return {
                    keyword.arg: keyword.value
                    for keyword in decorator.keywords
                    if keyword.arg is not None
                }
    raise AssertionError(f"{class_name} @app.server decorator was not found")


@pytest.mark.parametrize(("filename", "class_name"), DEPLOYMENTS)
def test_sglang_modal_server_uses_proxy_auth_without_worker_secrets(
    filename: str,
    class_name: str,
) -> None:
    path = PROJECT_ROOT / "deployment" / filename
    source = path.read_text(encoding="utf-8")
    keywords = _server_keywords(path, class_name)

    assert isinstance(keywords["unauthenticated"], ast.Constant)
    assert keywords["unauthenticated"].value is False
    assert "secrets" not in keywords
    assert "--api-key" not in source
    assert "modal.Secret" not in source
    assert "Authorization" not in source
    assert "SGLANG_QUERY_REWRITE_API_KEY" not in source
    assert "QWEN_SGLANG_API_KEY" not in source


def test_sglang_deployments_preserve_model_and_inference_contracts() -> None:
    granite = (PROJECT_ROOT / "deployment" / "modal_sglang.py").read_text(
        encoding="utf-8"
    )
    qwen = (PROJECT_ROOT / "deployment" / "modal_qwen_sglang.py").read_text(
        encoding="utf-8"
    )

    assert '"--dtype",\n            "float16"' in granite
    assert '"--grammar-backend",\n            "xgrammar"' in granite
    assert '"continue_final_message": True' in granite
    assert '"temperature": 0' in granite
    assert '"--dtype",\n            "float16"' in qwen
    assert '"--reasoning-parser",\n            "qwen3"' in qwen
    assert '"chat_template_kwargs": {"enable_thinking": False}' in qwen
    assert '"temperature": 0.7' in qwen
