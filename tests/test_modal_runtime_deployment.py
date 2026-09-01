from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_PATH = PROJECT_ROOT / "deployment" / "modal_runtime.py"


def _server_keywords() -> dict[str, ast.expr]:
    module = ast.parse(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != "RAGRuntimeServer":
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
    raise AssertionError("RAGRuntimeServer @app.server decorator was not found")


def test_modal_runtime_is_private_singleton_without_autoscaling_target() -> None:
    keywords = _server_keywords()

    assert isinstance(keywords["min_containers"], ast.Name)
    assert keywords["min_containers"].id == "MIN_CONTAINERS"
    assert isinstance(keywords["max_containers"], ast.Name)
    assert keywords["max_containers"].id == "MAX_CONTAINERS"
    assert "target_concurrency" not in keywords
    assert isinstance(keywords["unauthenticated"], ast.Constant)
    assert keywords["unauthenticated"].value is False


def test_modal_runtime_keeps_cuda_and_one_worker_contracts() -> None:
    source = DEPLOYMENT_PATH.read_text(encoding="utf-8")

    assert 'GPU = os.getenv("MODAL_RAG_GPU", "L40S")' in source
    assert 'COMPUTE_REGION = os.getenv("MODAL_RAG_COMPUTE_REGION", "us")' in source
    assert '"ONNX_EMBEDDING_EXECUTION_PROVIDER": "CUDAExecutionProvider"' in source
    assert '"ONNX_EMBEDDING_DISABLE_CPU_FALLBACK": "true"' in source
    assert '"ONNX_RERANKER_EXECUTION_PROVIDER": "CUDAExecutionProvider"' in source
    assert '"ONNX_RERANKER_DISABLE_CPU_FALLBACK": "true"' in source
    assert "workers=1" in source
    assert "HF_HUB_OFFLINE" in source
    assert "TRANSFORMERS_OFFLINE" in source
