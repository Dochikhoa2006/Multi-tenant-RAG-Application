from __future__ import annotations

import ast
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _import_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_production_python_never_imports_evaluation_or_ragas() -> None:
    for root_name in ("backend", "deployment"):
        for path in (REPOSITORY_ROOT / root_name).rglob("*.py"):
            imports = _import_roots(path)
            assert "evaluation" not in imports, path
            assert "rag_evaluation" not in imports, path
            assert "ragas" not in imports, path


def test_evaluator_never_imports_application_or_storage_packages() -> None:
    source = REPOSITORY_ROOT / "evaluation" / "src" / "rag_evaluation"
    for path in source.glob("*.py"):
        imports = _import_roots(path)
        assert "backend" not in imports, path
        assert "deployment" not in imports, path
        assert "weaviate" not in imports, path


def test_production_dependency_and_image_definitions_exclude_ragas() -> None:
    checked = [
        REPOSITORY_ROOT / "backend" / "requirements.txt",
        REPOSITORY_ROOT / "Dockerfile",
        REPOSITORY_ROOT / "compose.yaml",
        REPOSITORY_ROOT / "deployment" / "modal_runtime.py",
    ]
    for path in checked:
        assert "ragas" not in path.read_text(encoding="utf-8").lower(), path


def test_tracking_is_disabled_at_import_time() -> None:
    import rag_evaluation.metrics  # noqa: F401

    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"

