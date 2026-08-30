"""Modal deployment for the always-warm FP16 SGLang Granite worker.

Deploy with ``modal deploy deployment/modal_sglang.py`` after provisioning the
checkpoint and ``granite-manifest.json`` into the configured Modal Volume.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import modal


SGLANG_IMAGE = (
    "lmsysorg/sglang:v0.5.18-cu130@"
    "sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1"
)
MODEL_NAME = os.getenv(
    "SGLANG_QUERY_REWRITE_MODEL",
    "merged-granite-4.1-3b-query-rewrite",
)
MODEL_VOLUME_ROOT = Path("/models")
MODEL_PATH = MODEL_VOLUME_ROOT / MODEL_NAME
MANIFEST_PATH = MODEL_PATH / "granite-manifest.json"
PORT = int(os.getenv("MODAL_SGLANG_PORT", "30000"))
GPU = os.getenv("MODAL_SGLANG_GPU", "L40S")
COMPUTE_REGION = os.getenv("MODAL_SGLANG_COMPUTE_REGION", "us-east")
ROUTING_REGION = os.getenv("MODAL_SGLANG_ROUTING_REGION", "us-east")
MIN_CONTAINERS = int(os.getenv("MODAL_SGLANG_MIN_CONTAINERS", "1"))
MAX_CONTAINERS = int(os.getenv("MODAL_SGLANG_MAX_CONTAINERS", "4"))
TARGET_CONCURRENCY = int(os.getenv("MODAL_SGLANG_TARGET_CONCURRENCY", "4"))
MAX_RUNNING_REQUESTS = int(os.getenv("MODAL_SGLANG_MAX_RUNNING_REQUESTS", "8"))
CUDA_GRAPH_MAX_BATCH = int(os.getenv("MODAL_SGLANG_CUDA_GRAPH_MAX_BATCH", "8"))
CONTEXT_LENGTH = int(os.getenv("MODAL_SGLANG_CONTEXT_LENGTH", "2176"))
MEM_FRACTION_STATIC = os.getenv("MODAL_SGLANG_MEM_FRACTION_STATIC", "0.70")
OTLP_TRACES_ENDPOINT = os.getenv("SGLANG_OTLP_TRACES_ENDPOINT", "").strip()
VOLUME_NAME = os.getenv("MODAL_SGLANG_MODEL_VOLUME", "rag-granite-models")
SECRET_NAME = os.getenv("MODAL_SGLANG_SECRET", "rag-sglang-secrets")

for name, value in (
    ("MODAL_SGLANG_PORT", PORT),
    ("MODAL_SGLANG_MIN_CONTAINERS", MIN_CONTAINERS),
    ("MODAL_SGLANG_MAX_CONTAINERS", MAX_CONTAINERS),
    ("MODAL_SGLANG_TARGET_CONCURRENCY", TARGET_CONCURRENCY),
    ("MODAL_SGLANG_MAX_RUNNING_REQUESTS", MAX_RUNNING_REQUESTS),
    ("MODAL_SGLANG_CUDA_GRAPH_MAX_BATCH", CUDA_GRAPH_MAX_BATCH),
    ("MODAL_SGLANG_CONTEXT_LENGTH", CONTEXT_LENGTH),
):
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
if MIN_CONTAINERS > MAX_CONTAINERS:
    raise ValueError("Modal minimum containers cannot exceed maximum containers")
if TARGET_CONCURRENCY > MAX_RUNNING_REQUESTS:
    raise ValueError("Modal target concurrency cannot exceed SGLang running requests")


image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .env(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
sglang_secret = modal.Secret.from_name(SECRET_NAME)
app = modal.App("rag-granite-query-rewrite-sglang")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_checkpoint_manifest() -> None:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Granite checkpoint manifest is missing or malformed") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0":
        raise RuntimeError("Granite checkpoint manifest has an unsupported schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("Granite checkpoint manifest contains no files")
    required = {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    }
    if not required.issubset(files):
        raise RuntimeError("Granite checkpoint manifest is incomplete")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError("Granite checkpoint manifest contains invalid entries")
        candidate = (MODEL_PATH / relative).resolve()
        if candidate.parent != MODEL_PATH.resolve() or not candidate.is_file():
            raise RuntimeError("Granite checkpoint manifest references an invalid file")
        if _sha256(candidate) != expected:
            raise RuntimeError("Granite checkpoint hash verification failed")
    try:
        config = json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8"))
        index = json.loads(
            (MODEL_PATH / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Granite checkpoint metadata is malformed") from exc
    if (
        not isinstance(config, dict)
        or config.get("architectures") != ["GraniteForCausalLM"]
        or config.get("model_type") != "granite"
        or config.get("dtype", config.get("torch_dtype")) != "float16"
    ):
        raise RuntimeError("Checkpoint is not the expected FP16 Granite architecture")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    expected_shards = {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    if not isinstance(weight_map, dict) or set(weight_map.values()) != expected_shards:
        raise RuntimeError("Granite index does not reference the expected two shards")


def _check_running(process: subprocess.Popen[Any]) -> None:
    return_code = process.poll()
    if return_code is not None:
        raise RuntimeError(f"SGLang terminated during startup with status {return_code}")


def _wait_and_warm(process: subprocess.Popen[Any], timeout_seconds: int = 1200) -> None:
    import requests

    api_key = os.getenv("SGLANG_QUERY_REWRITE_API_KEY", "")
    if not api_key:
        raise RuntimeError("SGLANG_QUERY_REWRITE_API_KEY must be provided by Modal Secret")
    deadline = time.monotonic() + timeout_seconds
    health_url = f"http://127.0.0.1:{PORT}/health"
    while time.monotonic() < deadline:
        _check_running(process)
        try:
            response = requests.get(health_url, timeout=2)
            if response.ok:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        raise TimeoutError("SGLang did not become healthy before the startup deadline")

    prefill = os.getenv(
        "GRANITE_QUERY_REWRITE_RESPONSE_PREFILL",
        '{"rewritten_question":"',
    )
    regex = os.getenv(
        "SGLANG_QUERY_REWRITE_CONTINUATION_REGEX",
        r'(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9A-Fa-f]{4}))+"}\s*',
    )
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Rewrite this standalone question."},
            {"role": "assistant", "content": prefill},
        ],
        "temperature": 0,
        "max_tokens": int(
            os.getenv("GRANITE_QUERY_REWRITE_MAX_NEW_TOKENS", "128")
        ),
        "n": 1,
        "stream": False,
        "continue_final_message": True,
        "regex": regex,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    for _ in range(3):
        response = requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        continuation = data["choices"][0]["message"]["content"]
        assembled = json.loads(prefill + continuation)
        if set(assembled) != {"rewritten_question"}:
            raise RuntimeError("SGLang warmup violated the Granite JSON contract")


@app.server(
    image=image,
    gpu=GPU,
    volumes={str(MODEL_VOLUME_ROOT): model_volume},
    secrets=[sglang_secret],
    compute_region=COMPUTE_REGION,
    routing_region=ROUTING_REGION,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    target_concurrency=TARGET_CONCURRENCY,
    startup_timeout=20 * 60,
    exit_grace_period=30,
    port=PORT,
    unauthenticated=True,
)
class GraniteSGLangServer:
    @modal.enter()
    def start(self) -> None:
        _validate_checkpoint_manifest()
        api_key = os.getenv("SGLANG_QUERY_REWRITE_API_KEY", "")
        if not api_key:
            raise RuntimeError("SGLANG_QUERY_REWRITE_API_KEY is required")
        command = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(MODEL_PATH),
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--dtype",
            "float16",
            "--tp",
            "1",
            "--context-length",
            str(CONTEXT_LENGTH),
            "--max-running-requests",
            str(MAX_RUNNING_REQUESTS),
            "--cuda-graph-max-bs",
            str(CUDA_GRAPH_MAX_BATCH),
            "--mem-fraction-static",
            MEM_FRACTION_STATIC,
            "--attention-backend",
            "flashinfer",
            "--grammar-backend",
            "xgrammar",
            "--sampling-defaults",
            "openai",
            "--enable-metrics",
            "--enable-cache-report",
            "--collect-tokens-histogram",
            "--api-key",
            api_key,
        ]
        if OTLP_TRACES_ENDPOINT:
            command.extend(
                (
                    "--enable-trace",
                    "--otlp-traces-endpoint",
                    OTLP_TRACES_ENDPOINT,
                )
            )
        self.process = subprocess.Popen(command, start_new_session=True)
        _wait_and_warm(self.process)

    @modal.exit()
    def stop(self) -> None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
