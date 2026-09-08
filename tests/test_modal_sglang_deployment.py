from __future__ import annotations

import ast
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENTS = (
    ("modal_sglang.py", "GraniteSGLangServer"),
    ("modal_qwen_sglang.py", "QwenSGLangServer"),
)
STARTUP_TIMEOUT_ENVIRONMENTS = (
    (
        "modal_sglang.py",
        "GraniteSGLangServer",
        "MODAL_SGLANG_STARTUP_TIMEOUT",
    ),
    (
        "modal_qwen_sglang.py",
        "QwenSGLangServer",
        "QWEN_MODAL_SGLANG_STARTUP_TIMEOUT",
    ),
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


@pytest.mark.parametrize(
    ("filename", "class_name", "environment_name"),
    STARTUP_TIMEOUT_ENVIRONMENTS,
)
def test_sglang_startup_timeout_has_one_import_time_owner(
    filename: str,
    class_name: str,
    environment_name: str,
) -> None:
    path = PROJECT_ROOT / "deployment" / filename
    source = path.read_text(encoding="utf-8")
    startup_timeout = _server_keywords(path, class_name)["startup_timeout"]

    assert isinstance(startup_timeout, ast.Name)
    assert startup_timeout.id == "STARTUP_TIMEOUT_SECONDS"
    assert f'os.getenv("{environment_name}", "1200")' in source
    assert "timeout_seconds: int = STARTUP_TIMEOUT_SECONDS" in source
    assert "startup_timeout=20 * 60" not in source
    assert "timeout_seconds: int = 1200" not in source


def test_example_environment_documents_modal_placement_and_startup_timeouts() -> None:
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    for assignment in (
        "MODAL_SGLANG_COMPUTE_REGION=us",
        "MODAL_SGLANG_ROUTING_REGION=us-east",
        "MODAL_SGLANG_STARTUP_TIMEOUT=1200",
        "QWEN_MODAL_SGLANG_COMPUTE_REGION=us",
        "QWEN_MODAL_SGLANG_ROUTING_REGION=us-east",
        "QWEN_MODAL_SGLANG_STARTUP_TIMEOUT=1200",
        "MODAL_RAG_COMPUTE_REGION=us",
        "MODAL_RAG_ROUTING_REGION=us-east",
    ):
        assert assignment in example
