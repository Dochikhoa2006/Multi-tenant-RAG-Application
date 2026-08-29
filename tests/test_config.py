from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str, *, overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(overrides)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_operational_config_is_overridable_but_collection_names_are_invariant(
    tmp_path: Path,
) -> None:
    first_file = tmp_path / "first.data"
    second_file = tmp_path / "second.data"
    first_file.write_text("first", encoding="utf-16")
    second_file.write_text("second", encoding="utf-16")
    code = """
import sys
from backend.config import (
    COLLECTION_TYPES,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_GRPC_SECURE,
    WEAVIATE_URL,
    get_collection_name,
)
from backend.processing.file_reader import read_text_files

assert COLLECTION_TYPES == frozenset({'conversations', 'knowledge_facts', 'policy'})
assert WEAVIATE_URL == 'https://weaviate.example.test:8443'
assert WEAVIATE_GRPC_PORT == 50443
assert WEAVIATE_GRPC_SECURE is False
assert get_collection_name('user.name', 'conversations') == 'RagUser_OVZWK4RONZQW2ZI_Conversations'
assert get_collection_name('usr_abc123', 'conversations') == 'RagUser_OVZXEX3BMJRTCMRT_Conversations'
assert get_collection_name('客户/email@example.com', 'policy') == (
    'RagUser_4WXKFZUIW4XWK3LBNFWEAZLYMFWXA3DFFZRW63I_Policy'
)
try:
    get_collection_name('user.name', 'memory')
except ValueError:
    pass
else:
    raise AssertionError('additional collection types must remain invalid')
assert read_text_files(sys.argv[1:]) == 'first||second'
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(first_file), str(second_file)],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "COLLECTION_TYPES": "memory,rules",
            "COLLECTION_NAME_TEMPLATE": "{collection_type}__{user_id}",
            "USER_ID_PATTERN": r"^.+$",
            "WEAVIATE_URL": "https://weaviate.example.test:8443",
            "WEAVIATE_GRPC_PORT": "50443",
            "WEAVIATE_GRPC_SECURE": "false",
            "SUPPORTED_FILE_EXTENSIONS": ".data",
            "TEXT_FILE_ENCODING": "utf-16",
            "TEXT_FILE_JOIN_SEPARATOR": "||",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_physical_collection_names_are_graphql_safe_and_collision_free() -> None:
    from backend.config import get_collection_name

    expected = {
        "conversations": "RagUser_OVZXEX3BMJRTCMRT_Conversations",
        "knowledge_facts": "RagUser_OVZXEX3BMJRTCMRT_KnowledgeFacts",
        "policy": "RagUser_OVZXEX3BMJRTCMRT_Policy",
    }
    names = {
        collection_type: get_collection_name("usr_abc123", collection_type)
        for collection_type in expected
    }

    assert names == expected
    assert all(re.fullmatch(r"[A-Z][_0-9A-Za-z]*", name) for name in names.values())
    distinct_users = ["usr-1", "usr_1", "A", "a"]
    physical_names = [
        get_collection_name(user_id, "conversations") for user_id in distinct_users
    ]
    assert len(set(physical_names)) == len(distinct_users)


@pytest.mark.parametrize("user_id", [" usr_abc123", "usr_abc123 ", "\tusr_abc123"])
def test_collection_names_reject_lossy_user_id_normalization(user_id: str) -> None:
    from backend.config import get_collection_name

    with pytest.raises(ValueError, match="whitespace"):
        get_collection_name(user_id, "conversations")


def test_model_and_processing_configuration_is_environment_overridable() -> None:
    result = _run_python(
        """
from backend.model_config import (
    CHUNKING,
    HYBRID_SEARCH,
    PRIMARY_GENERATOR,
    TEXT_PROCESSING,
)

assert PRIMARY_GENERATOR.model == 'generator-override'
assert PRIMARY_GENERATOR.max_output_tokens == 900
assert HYBRID_SEARCH.components == ('dense',)
assert HYBRID_SEARCH.alpha == 0.4
assert CHUNKING.paragraph_threshold == 0.6
assert CHUNKING.min_tokens == 40
assert CHUNKING.target_tokens == 100
assert CHUNKING.max_tokens == 160
assert TEXT_PROCESSING.tokenizer_encoding == 'p50k_base'
assert TEXT_PROCESSING.sentence_boundary_pattern == r'(?<=;)\\s+'
assert TEXT_PROCESSING.markdown_heading_pattern == r'^HEADING:'
assert TEXT_PROCESSING.markdown_list_item_pattern == r'^ITEM:'
assert TEXT_PROCESSING.markdown_fence_pattern == r'^FENCE:'
assert TEXT_PROCESSING.sentence_join_separator == '|'
assert TEXT_PROCESSING.normalize_embeddings is False
assert TEXT_PROCESSING.show_progress_bar is True
assert TEXT_PROCESSING.sentence_model_cache_size == 2
assert TEXT_PROCESSING.tokenizer_cache_size == 3
""",
        overrides={
            "PRIMARY_GENERATOR_MODEL": "generator-override",
            "PRIMARY_GENERATOR_MAX_OUTPUT_TOKENS": "900",
            "HYBRID_COMPONENTS": "dense",
            "HYBRID_ALPHA": "0.4",
            "PARAGRAPH_SIMILARITY_THRESHOLD": "0.6",
            "MIN_CHUNK_TOKENS": "40",
            "TARGET_CHUNK_TOKENS": "100",
            "MAX_CHUNK_TOKENS": "160",
            "TOKENIZER_ENCODING": "p50k_base",
            "SENTENCE_BOUNDARY_PATTERN": r"(?<=;)\s+",
            "MARKDOWN_HEADING_PATTERN": r"^HEADING:",
            "MARKDOWN_LIST_ITEM_PATTERN": r"^ITEM:",
            "MARKDOWN_FENCE_PATTERN": r"^FENCE:",
            "SENTENCE_JOIN_SEPARATOR": "|",
            "NORMALIZE_SENTENCE_EMBEDDINGS": "false",
            "SENTENCE_EMBEDDING_PROGRESS_BAR": "true",
            "SENTENCE_MODEL_CACHE_SIZE": "2",
            "TOKENIZER_CACHE_SIZE": "3",
        },
    )

    assert result.returncode == 0, result.stderr


def test_invalid_environment_configuration_fails_fast() -> None:
    result = _run_python(
        "import backend.model_config",
        overrides={
            "MIN_CHUNK_TOKENS": "300",
            "TARGET_CHUNK_TOKENS": "200",
            "MAX_CHUNK_TOKENS": "320",
        },
    )

    assert result.returncode != 0
    assert "min <= target <= max" in result.stderr
