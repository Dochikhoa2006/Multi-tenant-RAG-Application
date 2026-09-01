"""Private singleton Modal deployment for the integrated CUDA RAG API.

The server mounts pre-provisioned, manifest-verified model artifacts and starts
the existing FastAPI composition root with exactly one Uvicorn worker. Modal
Proxy Tokens protect the endpoint; the RAG API itself is not made public.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import modal


APP_NAME = "rag-fastapi-onnx-runtime"
PORT = int(os.getenv("MODAL_RAG_PORT", "8000"))
GPU = os.getenv("MODAL_RAG_GPU", "L40S")
COMPUTE_REGION = os.getenv("MODAL_RAG_COMPUTE_REGION", "us")
ROUTING_REGION = os.getenv("MODAL_RAG_ROUTING_REGION", "us-east")
MIN_CONTAINERS = int(os.getenv("MODAL_RAG_MIN_CONTAINERS", "1"))
MAX_CONTAINERS = int(os.getenv("MODAL_RAG_MAX_CONTAINERS", "1"))
STARTUP_TIMEOUT_SECONDS = int(os.getenv("MODAL_RAG_STARTUP_TIMEOUT", "1200"))
RUNTIME_MODEL_VOLUME_NAME = os.getenv(
    "MODAL_RAG_MODEL_VOLUME",
    "rag-runtime-models",
)
GRANITE_MODEL_VOLUME_NAME = os.getenv(
    "MODAL_SGLANG_MODEL_VOLUME",
    "rag-granite-models",
)
SECRET_NAME = os.getenv("MODAL_RAG_SECRET", "rag-runtime-secrets")

RUNTIME_MODEL_ROOT = Path("/opt/runtime-models")
GRANITE_MODEL_ROOT = Path("/opt/granite-models")
GTE_MODEL_PATH = RUNTIME_MODEL_ROOT / "gte-modernbert-base"
BGE_MODEL_PATH = RUNTIME_MODEL_ROOT / "bge-reranker-v2-m3-onnx"
SEGMENTATION_MODEL_PATH = RUNTIME_MODEL_ROOT / "all-MiniLM-L6-v2"
GRANITE_MODEL_PATH = GRANITE_MODEL_ROOT / "merged-granite-4.1-3b-query-rewrite"

for name, value in (
    ("MODAL_RAG_PORT", PORT),
    ("MODAL_RAG_MIN_CONTAINERS", MIN_CONTAINERS),
    ("MODAL_RAG_MAX_CONTAINERS", MAX_CONTAINERS),
    ("MODAL_RAG_STARTUP_TIMEOUT", STARTUP_TIMEOUT_SECONDS),
):
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
if MIN_CONTAINERS != 1 or MAX_CONTAINERS != 1:
    raise ValueError(
        "The process-local RAG runtime requires exactly one Modal container"
    )


runtime_environment = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "QUERY_REWRITE_ENGINE": "sglang",
    "ONNX_EMBEDDING_MODEL_PATH": str(GTE_MODEL_PATH),
    "ONNX_EMBEDDING_EXECUTION_PROVIDER": "CUDAExecutionProvider",
    "ONNX_EMBEDDING_CUDA_DEVICE_ID": "0",
    "ONNX_EMBEDDING_DISABLE_CPU_FALLBACK": "true",
    "ONNX_RERANKER_MODEL_PATH": str(BGE_MODEL_PATH),
    "ONNX_RERANKER_EXECUTION_PROVIDER": "CUDAExecutionProvider",
    "ONNX_RERANKER_CUDA_DEVICE_ID": "0",
    "ONNX_RERANKER_DISABLE_CPU_FALLBACK": "true",
    "SEGMENTATION_EMBEDDING_MODEL": "all-MiniLM-L6-v2",
    "SEGMENTATION_MODEL_PATH": str(SEGMENTATION_MODEL_PATH),
    "SEGMENTATION_EMBEDDING_DEVICE": "cpu",
    "GRANITE_QUERY_REWRITE_MODEL_PATH": str(GRANITE_MODEL_PATH),
    "GRANITE_QUERY_REWRITE_DEVICE": "cuda",
    "GRANITE_QUERY_REWRITE_DTYPE": "float16",
    "SGLANG_QUERY_REWRITE_MODEL": "merged-granite-4.1-3b-query-rewrite",
    "QWEN_MODEL_PATH": "qwen3-4b-awq",
    "QWEN_SGLANG_SERVED_MODEL": "qwen3-4b-awq",
    "CORS_ALLOWED_ORIGINS": "*",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ca-certificates", "libgomp1", "zlib1g")
    .pip_install_from_requirements("backend/requirements.txt")
    .add_local_python_source("backend", copy=True)
    .env(runtime_environment)
)
runtime_model_volume = modal.Volume.from_name(
    RUNTIME_MODEL_VOLUME_NAME,
    create_if_missing=False,
)
granite_model_volume = modal.Volume.from_name(
    GRANITE_MODEL_VOLUME_NAME,
    create_if_missing=False,
)
runtime_secret = modal.Secret.from_name(SECRET_NAME)
app = modal.App(APP_NAME)


def _cuda_preflight() -> None:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"Modal GPU: {gpu.stdout.strip()}", flush=True)

    import onnxruntime as ort

    ort.preload_dlls(directory="")
    providers = ort.get_available_providers()
    print(f"ONNX Runtime providers: {providers}", flush=True)
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError("CUDAExecutionProvider is unavailable in the Modal image")


_UVICORN_ENTRYPOINT = f"""
import onnxruntime as ort
ort.preload_dlls(directory="")
import uvicorn
uvicorn.run(
    "backend.runtime_app:create_runtime_app",
    factory=True,
    host="0.0.0.0",
    port={PORT},
    workers=1,
)
"""


@app.server(
    image=image,
    gpu=GPU,
    cpu=4.0,
    memory=16_384,
    volumes={
        str(RUNTIME_MODEL_ROOT): runtime_model_volume,
        str(GRANITE_MODEL_ROOT): granite_model_volume,
    },
    secrets=[runtime_secret],
    compute_region=COMPUTE_REGION,
    routing_region=ROUTING_REGION,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    exit_grace_period=60,
    port=PORT,
    unauthenticated=False,
)
class RAGRuntimeServer:
    @modal.enter()
    def start(self) -> None:
        _cuda_preflight()
        self.process = subprocess.Popen(
            [sys.executable, "-c", _UVICORN_ENTRYPOINT],
            start_new_session=True,
        )

    @modal.exit()
    def stop(self) -> None:
        process: subprocess.Popen[Any] | None = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
