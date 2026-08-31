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


def test_segmentation_configuration_defaults_to_local_cpu() -> None:
    environment = os.environ.copy()
    for name in (
        "SEGMENTATION_EMBEDDING_MODEL",
        "SEGMENTATION_MODEL_PATH",
        "SEGMENTATION_EMBEDDING_DEVICE",
    ):
        environment.pop(name, None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from backend.model_config import (
    SEGMENTATION_EMBEDDING_DEVICE,
    SEGMENTATION_EMBEDDING_MODEL,
    SEGMENTATION_MODEL_PATH,
)

assert SEGMENTATION_EMBEDDING_MODEL == 'all-MiniLM-L6-v2'
assert SEGMENTATION_MODEL_PATH == 'models/all-MiniLM-L6-v2'
assert SEGMENTATION_EMBEDDING_DEVICE == 'cpu'
""",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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
    CORS_ALLOWED_ORIGINS,
    COLLECTION_TYPES,
    TASK_MAX_COMPLETED_RECORDS,
    UPLOAD_MAX_FILE_BYTES,
    UPLOAD_MAX_TOTAL_BYTES,
    UPLOAD_READ_CHUNK_BYTES,
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
assert CORS_ALLOWED_ORIGINS == ('https://one.example', 'https://two.example')
assert TASK_MAX_COMPLETED_RECORDS == 77
assert UPLOAD_MAX_FILE_BYTES == 1000
assert UPLOAD_MAX_TOTAL_BYTES == 2000
assert UPLOAD_READ_CHUNK_BYTES == 250
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
            "CORS_ALLOWED_ORIGINS": "https://one.example,https://two.example",
            "TASK_MAX_COMPLETED_RECORDS": "77",
            "UPLOAD_MAX_FILE_BYTES": "1000",
            "UPLOAD_MAX_TOTAL_BYTES": "2000",
            "UPLOAD_READ_CHUNK_BYTES": "250",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "overrides",
    [
        {"UPLOAD_MAX_FILE_BYTES": "11", "UPLOAD_MAX_TOTAL_BYTES": "10"},
        {
            "UPLOAD_MAX_FILE_BYTES": "10",
            "UPLOAD_MAX_TOTAL_BYTES": "20",
            "UPLOAD_READ_CHUNK_BYTES": "11",
        },
        {"CORS_ALLOWED_ORIGINS": "*,https://example.test"},
        {"TASK_MAX_COMPLETED_RECORDS": "0"},
    ],
)
def test_upload_and_cors_configuration_rejects_invalid_combinations(
    overrides: dict[str, str],
) -> None:
    result = _run_python("import backend.config", overrides=overrides)
    assert result.returncode != 0


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
    GRANITE_QUERY_REWRITE,
    HYBRID_SEARCH,
    ONNX_EMBEDDING,
    ONNX_RERANKER,
    PRIMARY_GENERATOR,
    QWEN_SGLANG,
    QUERY_REWRITE_ENGINE,
    QUERY_REWRITER,
    SEGMENTATION_EMBEDDING_DEVICE,
    SEGMENTATION_EMBEDDING_MODEL,
    SEGMENTATION_MODEL_PATH,
    SESSION_TITLE_GENERATOR,
    SGLANG_QUERY_REWRITE,
    TEXT_PROCESSING,
)

assert PRIMARY_GENERATOR.model == 'served-qwen'
assert PRIMARY_GENERATOR.max_output_tokens == 900
assert SESSION_TITLE_GENERATOR.model == 'served-qwen'
assert QWEN_SGLANG.model_path == 'local/qwen'
assert QWEN_SGLANG.base_url == 'https://qwen.example/v1'
assert QWEN_SGLANG.served_model == 'served-qwen'
assert QWEN_SGLANG.connect_timeout_seconds == 0.75
assert QWEN_SGLANG.read_timeout_seconds == 9.0
assert QWEN_SGLANG.max_connections == 15
assert QWEN_SGLANG.answer_max_output_tokens == 900
assert QWEN_SGLANG.title_max_output_tokens == 24
assert QWEN_SGLANG.temperature == 0.65
assert QWEN_SGLANG.top_p == 0.75
assert QWEN_SGLANG.top_k == 16
assert QWEN_SGLANG.min_p == 0.1
assert QWEN_SGLANG.presence_penalty == 1.25
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
assert SEGMENTATION_EMBEDDING_MODEL == 'all-MiniLM-L6-v2'
assert SEGMENTATION_MODEL_PATH == 'local/segmentation'
assert SEGMENTATION_EMBEDDING_DEVICE == 'cuda:2'
assert QUERY_REWRITER.model == 'granite-override'
assert QUERY_REWRITE_ENGINE == 'transformers'
assert GRANITE_QUERY_REWRITE.model_path == 'local/granite'
assert GRANITE_QUERY_REWRITE.device == 'cuda'
assert GRANITE_QUERY_REWRITE.max_input_tokens == 1024
assert GRANITE_QUERY_REWRITE.max_new_tokens == 64
assert GRANITE_QUERY_REWRITE.response_prefill == '{"rewritten_question":"'
assert GRANITE_QUERY_REWRITE.warmup is False
assert SGLANG_QUERY_REWRITE.base_url == 'https://sglang.example/v1'
assert SGLANG_QUERY_REWRITE.served_model == 'served-granite'
assert SGLANG_QUERY_REWRITE.connect_timeout_seconds == 0.5
assert SGLANG_QUERY_REWRITE.read_timeout_seconds == 3.0
assert SGLANG_QUERY_REWRITE.max_connections == 12
assert SGLANG_QUERY_REWRITE.constrained_output is False
assert ONNX_EMBEDDING.model_path == 'local/gte'
assert ONNX_EMBEDDING.onnx_filename == 'custom/embedding.onnx'
assert ONNX_EMBEDDING.max_tokens == 4096
assert ONNX_EMBEDDING.batch_size == 8
assert ONNX_EMBEDDING.device_id == 2
assert ONNX_EMBEDDING.execution_provider == 'CPUExecutionProvider'
assert ONNX_RERANKER.model_path == 'local/bge'
assert ONNX_RERANKER.onnx_filename == 'custom/reranker.onnx'
assert ONNX_RERANKER.max_tokens == 256
assert ONNX_RERANKER.batch_size == 4
assert ONNX_RERANKER.device_id == 3
assert ONNX_RERANKER.execution_provider == 'OpenVINOExecutionProvider'
assert 'top-secret' not in repr(SGLANG_QUERY_REWRITE)
assert 'qwen-secret' not in repr(QWEN_SGLANG)
""",
        overrides={
            "QWEN_MODEL_PATH": "local/qwen",
            "QWEN_SGLANG_BASE_URL": "https://qwen.example/v1/",
            "QWEN_SGLANG_API_KEY": "qwen-secret",
            "QWEN_SGLANG_SERVED_MODEL": "served-qwen",
            "QWEN_SGLANG_CONNECT_TIMEOUT_SECONDS": "0.75",
            "QWEN_SGLANG_READ_TIMEOUT_SECONDS": "9",
            "QWEN_SGLANG_MAX_CONNECTIONS": "15",
            "QWEN_ANSWER_MAX_OUTPUT_TOKENS": "900",
            "QWEN_TITLE_MAX_OUTPUT_TOKENS": "24",
            "QWEN_TEMPERATURE": "0.65",
            "QWEN_TOP_P": "0.75",
            "QWEN_TOP_K": "16",
            "QWEN_MIN_P": "0.1",
            "QWEN_PRESENCE_PENALTY": "1.25",
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
            "SEGMENTATION_MODEL_PATH": "local/segmentation",
            "SEGMENTATION_EMBEDDING_DEVICE": "cuda:2",
            "QUERY_REWRITER_MODEL": "granite-override",
            "QUERY_REWRITE_ENGINE": "transformers",
            "GRANITE_QUERY_REWRITE_MODEL_PATH": "local/granite",
            "GRANITE_QUERY_REWRITE_DEVICE": "cuda",
            "GRANITE_QUERY_REWRITE_MAX_INPUT_TOKENS": "1024",
            "GRANITE_QUERY_REWRITE_MAX_NEW_TOKENS": "64",
            "GRANITE_QUERY_REWRITE_WARMUP": "false",
            "SGLANG_QUERY_REWRITE_BASE_URL": "https://sglang.example/v1/",
            "SGLANG_QUERY_REWRITE_API_KEY": "top-secret",
            "SGLANG_QUERY_REWRITE_MODEL": "served-granite",
            "SGLANG_QUERY_REWRITE_CONNECT_TIMEOUT_SECONDS": "0.5",
            "SGLANG_QUERY_REWRITE_READ_TIMEOUT_SECONDS": "3",
            "SGLANG_QUERY_REWRITE_MAX_CONNECTIONS": "12",
            "SGLANG_QUERY_REWRITE_CONSTRAINED_OUTPUT": "false",
            "ONNX_EMBEDDING_MODEL_PATH": "local/gte",
            "ONNX_EMBEDDING_FILENAME": "custom/embedding.onnx",
            "ONNX_EMBEDDING_MAX_TOKENS": "4096",
            "ONNX_EMBEDDING_BATCH_SIZE": "8",
            "ONNX_EMBEDDING_CUDA_DEVICE_ID": "2",
            "ONNX_EMBEDDING_EXECUTION_PROVIDER": "CPUExecutionProvider",
            "ONNX_RERANKER_MODEL_PATH": "local/bge",
            "ONNX_RERANKER_FILENAME": "custom/reranker.onnx",
            "ONNX_RERANKER_MAX_TOKENS": "256",
            "ONNX_RERANKER_BATCH_SIZE": "4",
            "ONNX_RERANKER_CUDA_DEVICE_ID": "3",
            "ONNX_RERANKER_EXECUTION_PROVIDER": "OpenVINOExecutionProvider",
        },
    )

    assert result.returncode == 0, result.stderr


def test_answer_and_title_models_share_qwen_served_identity() -> None:
    environment = os.environ.copy()
    environment["QWEN_SGLANG_SERVED_MODEL"] = "shared-qwen"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from backend.model_config import PRIMARY_GENERATOR, "
                "SESSION_TITLE_GENERATOR; "
                "assert PRIMARY_GENERATOR.model == 'shared-qwen'; "
                "assert SESSION_TITLE_GENERATOR.model == PRIMARY_GENERATOR.model"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_onnx_execution_provider_defaults_to_cuda_without_fallback() -> None:
    environment = os.environ.copy()
    for name in (
        "ONNX_EMBEDDING_EXECUTION_PROVIDER",
        "ONNX_RERANKER_EXECUTION_PROVIDER",
        "ONNX_EMBEDDING_DISABLE_CPU_FALLBACK",
        "ONNX_RERANKER_DISABLE_CPU_FALLBACK",
    ):
        environment.pop(name, None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from backend.model_config import ONNX_EMBEDDING, ONNX_RERANKER; "
                "assert ONNX_EMBEDDING.execution_provider == 'CUDAExecutionProvider'; "
                "assert ONNX_RERANKER.execution_provider == 'CUDAExecutionProvider'; "
                "assert ONNX_EMBEDDING.disable_cpu_fallback is True; "
                "assert ONNX_RERANKER.disable_cpu_fallback is True"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
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


def test_invalid_granite_response_prefill_fails_fast() -> None:
    result = _run_python(
        "import backend.model_config",
        overrides={"GRANITE_QUERY_REWRITE_RESPONSE_PREFILL": "not-json"},
    )

    assert result.returncode != 0
    assert "rewritten_question JSON string" in result.stderr


@pytest.mark.parametrize(
    "overrides",
    [
        {"QUERY_REWRITE_ENGINE": "automatic"},
        {"SGLANG_QUERY_REWRITE_BASE_URL": "not-a-url"},
        {"SGLANG_QUERY_REWRITE_CONNECT_TIMEOUT_SECONDS": "0"},
        {"SGLANG_QUERY_REWRITE_MAX_CONNECTIONS": "0"},
        {"SGLANG_QUERY_REWRITE_CONTINUATION_REGEX": "("},
        {"QWEN_SGLANG_BASE_URL": "not-a-url"},
        {"QWEN_SGLANG_CONNECT_TIMEOUT_SECONDS": "0"},
        {"QWEN_SGLANG_MAX_CONNECTIONS": "0"},
        {"QWEN_TEMPERATURE": "0"},
        {"QWEN_TOP_P": "1.1"},
        {"QWEN_PRESENCE_PENALTY": "2.1"},
        {"GRANITE_QUERY_REWRITE_DEVICE": "mps"},
        {"ONNX_RERANKER_DISABLE_CPU_FALLBACK": "false"},
    ],
)
def test_invalid_sglang_or_cuda_configuration_fails_fast(
    overrides: dict[str, str],
) -> None:
    result = _run_python("import backend.model_config", overrides=overrides)
    assert result.returncode != 0
