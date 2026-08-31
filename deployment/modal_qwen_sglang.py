"""Modal deployment for the always-warm AWQ Qwen SGLang worker."""

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
MODEL_NAME = os.getenv("QWEN_SGLANG_SERVED_MODEL", "qwen3-4b-awq")
MODEL_VOLUME_ROOT = Path("/models")
MODEL_PATH = MODEL_VOLUME_ROOT / os.getenv("QWEN_MODEL_PATH", "qwen3-4b-awq")
MANIFEST_PATH = MODEL_PATH / "qwen-manifest.json"
PORT = int(os.getenv("QWEN_MODAL_SGLANG_PORT", "30001"))
GPU = os.getenv("QWEN_MODAL_SGLANG_GPU", "L40S")
COMPUTE_REGION = os.getenv("QWEN_MODAL_SGLANG_COMPUTE_REGION", "us-east")
ROUTING_REGION = os.getenv("QWEN_MODAL_SGLANG_ROUTING_REGION", "us-east")
MIN_CONTAINERS = int(os.getenv("QWEN_MODAL_SGLANG_MIN_CONTAINERS", "1"))
MAX_CONTAINERS = int(os.getenv("QWEN_MODAL_SGLANG_MAX_CONTAINERS", "4"))
TARGET_CONCURRENCY = int(os.getenv("QWEN_MODAL_SGLANG_TARGET_CONCURRENCY", "4"))
MAX_RUNNING_REQUESTS = int(os.getenv("QWEN_MODAL_SGLANG_MAX_RUNNING_REQUESTS", "8"))
CUDA_GRAPH_MAX_BATCH = int(os.getenv("QWEN_MODAL_CUDA_GRAPH_MAX_BATCH", "8"))
CONTEXT_LENGTH = int(os.getenv("QWEN_MODAL_CONTEXT_LENGTH", "32768"))
MEM_FRACTION_STATIC = os.getenv("QWEN_MODAL_MEM_FRACTION_STATIC", "0.80")
OTLP_TRACES_ENDPOINT = os.getenv("QWEN_SGLANG_OTLP_TRACES_ENDPOINT", "").strip()
VOLUME_NAME = os.getenv("QWEN_MODAL_MODEL_VOLUME", "rag-qwen-models")
SECRET_NAME = os.getenv("QWEN_MODAL_SGLANG_SECRET", "rag-qwen-sglang-secrets")

for name, value in (
    ("QWEN_MODAL_SGLANG_PORT", PORT),
    ("QWEN_MODAL_SGLANG_MIN_CONTAINERS", MIN_CONTAINERS),
    ("QWEN_MODAL_SGLANG_MAX_CONTAINERS", MAX_CONTAINERS),
    ("QWEN_MODAL_SGLANG_TARGET_CONCURRENCY", TARGET_CONCURRENCY),
    ("QWEN_MODAL_SGLANG_MAX_RUNNING_REQUESTS", MAX_RUNNING_REQUESTS),
    ("QWEN_MODAL_CUDA_GRAPH_MAX_BATCH", CUDA_GRAPH_MAX_BATCH),
    ("QWEN_MODAL_CONTEXT_LENGTH", CONTEXT_LENGTH),
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
app = modal.App("rag-qwen-answer-title-sglang")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_checkpoint_manifest() -> None:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        config = json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8"))
        tokenizer = json.loads(
            (MODEL_PATH / "tokenizer_config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Qwen checkpoint metadata or manifest is malformed") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "1.0"
        or manifest.get("model_id") != "qwen3-4b-awq"
    ):
        raise RuntimeError("Qwen checkpoint manifest has an unsupported identity")
    files = manifest.get("files")
    required = {
        "LICENSE",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    if not isinstance(files, dict) or set(files) != required:
        raise RuntimeError("Qwen checkpoint manifest is incomplete")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError("Qwen checkpoint manifest contains invalid entries")
        candidate = (MODEL_PATH / relative).resolve()
        if candidate.parent != MODEL_PATH.resolve() or not candidate.is_file():
            raise RuntimeError("Qwen checkpoint manifest references an invalid file")
        if _sha256(candidate) != expected:
            raise RuntimeError("Qwen checkpoint hash verification failed")
    quantization = config.get("quantization_config") if isinstance(config, dict) else None
    chat_template = tokenizer.get("chat_template") if isinstance(tokenizer, dict) else None
    if (
        not isinstance(config, dict)
        or config.get("architectures") != ["Qwen3ForCausalLM"]
        or config.get("model_type") != "qwen3"
        or config.get("torch_dtype") != "float16"
        or config.get("auto_map") is not None
        or not isinstance(quantization, dict)
        or quantization.get("quant_method") != "awq"
        or quantization.get("bits") != 4
        or quantization.get("group_size") != 128
        or quantization.get("version") != "gemm"
        or quantization.get("zero_point") is not True
        or not isinstance(chat_template, str)
        or "enable_thinking" not in chat_template
        or tokenizer.get("auto_map") is not None
    ):
        raise RuntimeError("Checkpoint is not the expected Qwen3-4B AWQ model")


def _check_running(process: subprocess.Popen[Any]) -> None:
    return_code = process.poll()
    if return_code is not None:
        raise RuntimeError(f"Qwen SGLang terminated during startup with {return_code}")


def _request_body(prompt: str, *, stream: bool, max_tokens: int) -> dict[str, object]:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "max_tokens": max_tokens,
        "n": 1,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _wait_and_warm(process: subprocess.Popen[Any], timeout_seconds: int = 1200) -> None:
    import requests

    api_key = os.getenv("QWEN_SGLANG_API_KEY", "")
    if not api_key:
        raise RuntimeError("QWEN_SGLANG_API_KEY must be provided by Modal Secret")
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
        raise TimeoutError("Qwen SGLang did not become healthy before startup timeout")

    headers = {"Authorization": f"Bearer {api_key}"}
    endpoint = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    title = requests.post(
        endpoint,
        json=_request_body("Create a three word session title.", stream=False, max_tokens=32),
        headers=headers,
        timeout=60,
    )
    title.raise_for_status()
    title_message = title.json()["choices"][0]["message"]
    if (
        not isinstance(title_message.get("content"), str)
        or not title_message["content"].strip()
        or title_message.get("reasoning_content") not in (None, "")
    ):
        raise RuntimeError("Qwen title warmup violated the non-thinking contract")

    answer = requests.post(
        endpoint,
        json=_request_body("Answer with one short sentence.", stream=True, max_tokens=32),
        headers=headers,
        timeout=60,
        stream=True,
    )
    answer.raise_for_status()
    saw_content = False
    saw_done = False
    for raw_line in answer.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        payload = raw_line[5:].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        event = json.loads(payload)
        delta = event["choices"][0]["delta"]
        if delta.get("reasoning_content") not in (None, ""):
            raise RuntimeError("Qwen answer warmup returned reasoning content")
        saw_content = saw_content or bool(delta.get("content"))
    if not saw_content or not saw_done:
        raise RuntimeError("Qwen answer warmup violated the streaming contract")


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
class QwenSGLangServer:
    @modal.enter()
    def start(self) -> None:
        _validate_checkpoint_manifest()
        api_key = os.getenv("QWEN_SGLANG_API_KEY", "")
        if not api_key:
            raise RuntimeError("QWEN_SGLANG_API_KEY is required")
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
            "--reasoning-parser",
            "qwen3",
            "--max-running-requests",
            str(MAX_RUNNING_REQUESTS),
            "--cuda-graph-max-bs",
            str(CUDA_GRAPH_MAX_BATCH),
            "--mem-fraction-static",
            MEM_FRACTION_STATIC,
            "--attention-backend",
            "flashinfer",
            "--enable-metrics",
            "--enable-cache-report",
            "--collect-tokens-histogram",
            "--api-key",
            api_key,
        ]
        if OTLP_TRACES_ENDPOINT:
            command.extend(
                ("--enable-trace", "--otlp-traces-endpoint", OTLP_TRACES_ENDPOINT)
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
