#!/usr/bin/env python3
"""Contract-faithful lifecycle control for the private Modal RAG deployment."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Iterable, Iterator, Mapping, Sequence
import json
import math
import os
from pathlib import Path
import shlex
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.parse import urlparse
import warnings

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
STATE_PATH = PROJECT_ROOT / ".local" / "rag-state.json"
COMPOSE_FILE = PROJECT_ROOT / "deployment" / "compose.weaviate-secure.yaml"
CERT_DIR = PROJECT_ROOT / ".local" / "tailscale-certs"
CERT_PATH = CERT_DIR / "weaviate.pem"
KEY_PATH = CERT_DIR / "weaviate.pem.key"
MODAL = PROJECT_ROOT / ".venv" / "bin" / "modal"
TAILSCALE = Path("/Applications/Tailscale.app/Contents/MacOS/Tailscale")

GRANITE_APP = "rag-granite-query-rewrite-sglang"
QWEN_APP = "rag-qwen-answer-title-sglang"
RUNTIME_APP = "rag-fastapi-onnx-runtime"
MODAL_APPS = (RUNTIME_APP, GRANITE_APP, QWEN_APP)

GRANITE_CLASS = "GraniteSGLangServer"
QWEN_CLASS = "QwenSGLangServer"
RUNTIME_CLASS = "RAGRuntimeServer"
GRANITE_MODEL = "merged-granite-4.1-3b-query-rewrite"
QWEN_MODEL = "qwen3-4b-awq"

DEFAULT_USER_ID = "e2e_20260901_0cd6147d7fd3"
STARTUP_TIMEOUT_SECONDS = 1_500
POLL_SECONDS = 5.0
PROGRESS_SECONDS = 30.0

GPU_DEFAULTS = {
    "MODAL_SGLANG_GPU": "L40S",
    "QWEN_MODAL_SGLANG_GPU": "H100",
    "MODAL_RAG_GPU": "L40S",
}
ALLOWED_OPERATIONAL_GPUS = frozenset({"L40S", "H100"})
CHAT_TIMING_KEYS = (
    "original_query_embedding",
    "conversation_hybrid_search",
    "conversation_mmr_rerank",
    "query_rewrite",
    "rewritten_query_embedding",
    "knowledge_hybrid_search",
    "knowledge_cross_encoder_rerank",
    "policy_hybrid_search",
    "policy_cross_encoder_rerank",
    "prompt_construction",
    "ttft",
    "generation",
    "total_request",
)

RUNTIME_SECRET_KEYS = (
    "WEAVIATE_URL",
    "WEAVIATE_API_KEY",
    "WEAVIATE_CONNECTION_MODE",
    "WEAVIATE_GRPC_PORT",
    "WEAVIATE_GRPC_SECURE",
    "SGLANG_QUERY_REWRITE_BASE_URL",
    "SGLANG_QUERY_REWRITE_API_KEY",
    "QWEN_SGLANG_BASE_URL",
    "QWEN_SGLANG_API_KEY",
)

REQUIRED_ENV_KEYS = (
    "WEAVIATE_URL",
    "WEAVIATE_API_KEY",
    "WEAVIATE_CONNECTION_MODE",
    "WEAVIATE_GRPC_PORT",
    "WEAVIATE_GRPC_SECURE",
    "MODAL_PROXY_TOKEN_ID",
    "MODAL_PROXY_TOKEN_SECRET",
)


class RagCtlError(RuntimeError):
    """A safe, user-facing lifecycle failure."""


def load_dotenv(path: Path) -> dict[str, str]:
    """Load the small dotenv subset used by this project without evaluating it."""

    if not path.is_file():
        raise RagCtlError(f"Missing configuration file: {path}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or not key or not key.replace("_", "A").isalnum():
            raise RagCtlError(f"Malformed .env assignment on line {line_number}")
        value = raw_value.strip()
        if value.startswith(("'", '"')):
            try:
                parsed = shlex.split(value, comments=True, posix=True)
            except ValueError as exc:
                raise RagCtlError(
                    f"Malformed quoted .env value on line {line_number}"
                ) from exc
            if len(parsed) != 1:
                raise RagCtlError(f"Malformed .env value on line {line_number}")
            value = parsed[0]
        else:
            for marker in (" #", "\t#"):
                value = value.split(marker, 1)[0].rstrip()
        values[key] = value
    return values


def validate_config(config: Mapping[str, str]) -> None:
    missing = [key for key in REQUIRED_ENV_KEYS if not config.get(key, "").strip()]
    if missing:
        raise RagCtlError("Missing required .env variables: " + ", ".join(missing))
    if config["WEAVIATE_CONNECTION_MODE"].strip().lower() != "custom":
        raise RagCtlError("WEAVIATE_CONNECTION_MODE must be custom for the secure Mac tunnel")
    if config["WEAVIATE_GRPC_SECURE"].strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RagCtlError("WEAVIATE_GRPC_SECURE must remain true")
    url = urlparse(config["WEAVIATE_URL"])
    if url.scheme != "https" or not url.hostname:
        raise RagCtlError("WEAVIATE_URL must be the HTTPS Tailscale Funnel URL")
    try:
        grpc_port = int(config["WEAVIATE_GRPC_PORT"])
    except ValueError as exc:
        raise RagCtlError("WEAVIATE_GRPC_PORT must be an integer") from exc
    if grpc_port != 8443:
        raise RagCtlError("WEAVIATE_GRPC_PORT must remain 8443 for the secure Funnel")

    bearer = proxy_bearer(config)
    for name in ("SGLANG_QUERY_REWRITE_API_KEY", "QWEN_SGLANG_API_KEY"):
        configured = config.get(name, "").strip()
        if configured and configured != bearer:
            raise RagCtlError(
                f"{name} must equal MODAL_PROXY_TOKEN_ID.MODAL_PROXY_TOKEN_SECRET"
            )
    for name, default in GPU_DEFAULTS.items():
        value = config.get(name, default).strip()
        if value not in ALLOWED_OPERATIONAL_GPUS:
            allowed = ", ".join(sorted(ALLOWED_OPERATIONAL_GPUS))
            raise RagCtlError(f"{name} must be one of: {allowed}")


def proxy_bearer(config: Mapping[str, str]) -> str:
    return (
        config.get("MODAL_PROXY_TOKEN_ID", "").strip()
        + "."
        + config.get("MODAL_PROXY_TOKEN_SECRET", "").strip()
    )


def secret_values(config: Mapping[str, str]) -> tuple[str, ...]:
    names = (
        "WEAVIATE_API_KEY",
        "MODAL_PROXY_TOKEN_ID",
        "MODAL_PROXY_TOKEN_SECRET",
        "SGLANG_QUERY_REWRITE_API_KEY",
        "QWEN_SGLANG_API_KEY",
    )
    values = {config.get(name, "") for name in names}
    values.add(proxy_bearer(config))
    return tuple(sorted((value for value in values if len(value) >= 4), key=len, reverse=True))


def redact(value: object, config: Mapping[str, str]) -> str:
    rendered = str(value)
    for secret in secret_values(config):
        rendered = rendered.replace(secret, "[REDACTED]")
    return rendered


class CommandRunner:
    def __init__(self, config: Mapping[str, str]) -> None:
        self.config = config

    def run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        overrides: Mapping[str, str] | None = None,
        capture: bool = False,
        check: bool = True,
        quiet: bool = False,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [os.fspath(item) for item in args]
        if not quiet:
            shown = " ".join(shlex.quote(item) for item in command)
            print(f"→ {redact(shown, self.config)}", flush=True)
        environment = os.environ.copy()
        environment.update(self.config)
        if overrides:
            environment.update(overrides)
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=capture,
            check=False,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout or "command failed").strip()
            raise RagCtlError(redact(detail, self.config))
        return result


def _json_command(runner: CommandRunner, args: Sequence[str | os.PathLike[str]]) -> Any:
    result = runner.run(args, capture=True, quiet=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RagCtlError(f"Command returned malformed JSON: {args[0]}") from exc


def preflight(config: Mapping[str, str], runner: CommandRunner) -> str:
    if not MODAL.is_file():
        raise RagCtlError(f"Modal CLI is missing: {MODAL}")
    if not TAILSCALE.is_file():
        raise RagCtlError(
            "The Standalone Tailscale CLI is missing; do not substitute the Homebrew CLI"
        )
    for command in ("docker", "openssl"):
        if subprocess.run(
            ["/usr/bin/env", command, "--version"],
            capture_output=True,
            text=True,
            check=False,
        ).returncode != 0:
            raise RagCtlError(f"Required command is unavailable: {command}")

    if runner.run(["docker", "info"], capture=True, check=False, quiet=True).returncode:
        print("Docker is not ready; opening Docker Desktop…", flush=True)
        runner.run(["open", "-ga", "Docker"], quiet=True)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if not runner.run(
                ["docker", "info"], capture=True, check=False, quiet=True
            ).returncode:
                break
            time.sleep(3)
        else:
            raise RagCtlError("Docker Desktop did not become ready within 180 seconds")

    tailscale_status = _json_command(runner, [TAILSCALE, "status", "--json"])
    self_status = tailscale_status.get("Self", {})
    if not self_status.get("Online"):
        raise RagCtlError("The active Standalone Tailscale node is offline")
    dns_name = str(self_status.get("DNSName", "")).rstrip(".")
    hostname = urlparse(config["WEAVIATE_URL"]).hostname or ""
    if dns_name != hostname:
        raise RagCtlError(
            "WEAVIATE_URL does not match the active Standalone Tailscale node"
        )
    runner.run([MODAL, "profile", "current"], capture=True, quiet=True)
    return hostname


def ensure_certificate(hostname: str, runner: CommandRunner) -> None:
    valid = (
        CERT_PATH.is_file()
        and KEY_PATH.is_file()
        and runner.run(
            ["openssl", "x509", "-checkend", "86400", "-noout", "-in", CERT_PATH],
            capture=True,
            check=False,
            quiet=True,
        ).returncode
        == 0
    )
    if valid:
        return
    print("Renewing the Tailscale certificate used by HAProxy…", flush=True)
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rag-cert-") as temp_dir:
        temp_cert = Path(temp_dir) / "weaviate.pem"
        temp_key = Path(temp_dir) / "weaviate.pem.key"
        runner.run(
            [
                TAILSCALE,
                "cert",
                "--cert-file",
                temp_cert,
                "--key-file",
                temp_key,
                hostname,
            ]
        )
        os.replace(temp_cert, CERT_PATH)
        os.replace(temp_key, KEY_PATH)
        KEY_PATH.chmod(0o600)


def compose_up(runner: CommandRunner) -> None:
    runner.run(
        [
            "docker",
            "compose",
            "--env-file",
            ENV_PATH,
            "-f",
            COMPOSE_FILE,
            "up",
            "-d",
            "--wait",
        ]
    )


def configure_funnels(hostname: str, runner: CommandRunner) -> None:
    runner.run(
        [TAILSCALE, "funnel", "--https=443", "off"],
        check=False,
        capture=True,
        quiet=True,
    )
    runner.run(
        [
            TAILSCALE,
            "funnel",
            "--yes",
            "--bg",
            "--https=443",
            "http://127.0.0.1:8080",
        ]
    )
    runner.run(
        [TAILSCALE, "funnel", "--tcp=8443", "off"],
        check=False,
        capture=True,
        quiet=True,
    )
    runner.run(
        [
            TAILSCALE,
            "funnel",
            "--yes",
            "--bg",
            "--tcp=8443",
            "tcp://127.0.0.1:5443",
        ]
    )
    status = _json_command(runner, [TAILSCALE, "funnel", "status", "--json"])
    web = status.get("Web", {}).get(f"{hostname}:443", {}).get("Handlers", {})
    rest_proxy = web.get("/", {}).get("Proxy")
    tcp_forward = status.get("TCP", {}).get("8443", {}).get("TCPForward")
    allowed = status.get("AllowFunnel", {})
    if (
        rest_proxy != "http://127.0.0.1:8080"
        or tcp_forward != "127.0.0.1:5443"
        or allowed.get(f"{hostname}:443") is not True
        or allowed.get(f"{hostname}:8443") is not True
    ):
        raise RagCtlError("Tailscale did not register both Funnels on the active node")


def _http_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}


def verify_weaviate(config: Mapping[str, str]) -> None:
    key = config["WEAVIATE_API_KEY"]
    local = "http://127.0.0.1:8080"
    external = config["WEAVIATE_URL"].rstrip("/")
    timeout = httpx.Timeout(30.0, read=60.0)
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        local_ready = client.get(
            f"{local}/v1/.well-known/ready", headers=_http_headers(key)
        )
        if local_ready.status_code != 200:
            raise RagCtlError("Authenticated local Weaviate readiness did not return 200")
        if client.get(f"{local}/v1/schema").status_code != 401:
            raise RagCtlError("Local Weaviate anonymous schema access was not rejected")
        if client.get(f"{local}/v1/schema", headers=_http_headers(key)).status_code != 200:
            raise RagCtlError("Authenticated local Weaviate schema access did not return 200")

        deadline = time.monotonic() + 120
        while True:
            try:
                ready = client.get(
                    f"{external}/v1/.well-known/ready", headers=_http_headers(key)
                )
                schema = client.get(
                    f"{external}/v1/schema", headers=_http_headers(key)
                )
                anonymous = client.get(f"{external}/v1/schema")
                if ready.status_code == schema.status_code == 200 and anonymous.status_code == 401:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                raise RagCtlError("External Weaviate REST verification timed out")
            time.sleep(3)

    hostname = urlparse(external).hostname
    if not hostname:
        raise RagCtlError("WEAVIATE_URL has no hostname")
    context = ssl.create_default_context()
    context.set_alpn_protocols(["h2"])
    try:
        with socket.create_connection(
            (hostname, int(config["WEAVIATE_GRPC_PORT"])), timeout=30
        ) as raw_socket:
            with context.wrap_socket(raw_socket, server_hostname=hostname) as tls_socket:
                if tls_socket.selected_alpn_protocol() != "h2":
                    raise RagCtlError("External Weaviate gRPC TLS did not negotiate ALPN h2")
    except OSError as exc:
        raise RagCtlError("External Weaviate gRPC TLS connection failed") from exc

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        from weaviate.classes.init import Auth
        import weaviate
        from weaviate.connect import ConnectionParams

        client = weaviate.WeaviateClient(
            connection_params=ConnectionParams.from_url(
                external,
                grpc_port=int(config["WEAVIATE_GRPC_PORT"]),
                grpc_secure=True,
            ),
            auth_client_secret=Auth.api_key(key),
        )
        try:
            client.connect()
            if not client.is_ready():
                raise RagCtlError("Pinned Weaviate client did not report ready")
            names = set(client.collections.list_all(simple=True))
            expected = user_collection_names(config.get("RAG_USER_ID", DEFAULT_USER_ID))
            if not expected.issubset(names):
                raise RagCtlError("The persisted user does not have all three collections")
        except RagCtlError:
            raise
        except Exception as exc:
            raise RagCtlError(
                redact(f"Pinned Weaviate client failed: {exc}", config)
            ) from None
        finally:
            client.close()


def user_collection_names(user_id: str) -> set[str]:
    token = base64.b32encode(user_id.encode("utf-8")).decode("ascii").rstrip("=")
    return {
        f"RagUser_{token}_Conversations",
        f"RagUser_{token}_KnowledgeFacts",
        f"RagUser_{token}_Policy",
    }


def gpu_request(config: Mapping[str, str], name: str) -> str:
    return config.get(name, GPU_DEFAULTS[name]).strip()


def deploy_granite(config: Mapping[str, str], runner: CommandRunner) -> str:
    runner.run(
        [MODAL, "deploy", PROJECT_ROOT / "deployment" / "modal_sglang.py"],
        overrides={
            "MODAL_SGLANG_GPU": gpu_request(config, "MODAL_SGLANG_GPU"),
            "MODAL_SGLANG_COMPUTE_REGION": "us",
            "MODAL_SGLANG_ROUTING_REGION": "us-east",
        },
    )
    return resolve_server_url(GRANITE_APP, GRANITE_CLASS) + "/v1"


def deploy_qwen(config: Mapping[str, str], runner: CommandRunner) -> str:
    runner.run(
        [MODAL, "deploy", PROJECT_ROOT / "deployment" / "modal_qwen_sglang.py"],
        overrides={
            "QWEN_MODAL_SGLANG_GPU": gpu_request(
                config, "QWEN_MODAL_SGLANG_GPU"
            ),
            "QWEN_MODAL_SGLANG_COMPUTE_REGION": "us",
            "QWEN_MODAL_SGLANG_ROUTING_REGION": "us-east",
        },
    )
    return resolve_server_url(QWEN_APP, QWEN_CLASS) + "/v1"


def resolve_server_url(app_name: str, class_name: str) -> str:
    import modal

    return modal.Server.from_name(app_name, class_name).get_url().rstrip("/")


def _waiting_message(service: str, last_status: int | None, requested_gpu: str) -> str:
    status = "no response" if last_status is None else f"HTTP {last_status}"
    return f"  {service}: waiting for readiness ({status}; requested GPU {requested_gpu})"


def _wait_authenticated_health(
    base_url: str,
    bearer: str,
    service: str,
    requested_gpu: str,
) -> None:
    root = base_url.removesuffix("/v1")
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    next_progress = time.monotonic()
    headers = _http_headers(bearer)
    last_status: int | None = None
    with httpx.Client(timeout=httpx.Timeout(30.0, read=90.0)) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(f"{root}/health", headers=headers)
                last_status = response.status_code
                if response.status_code == 200:
                    anonymous = client.get(f"{root}/health")
                    if anonymous.status_code != 401:
                        raise RagCtlError(f"{service} anonymous health was not rejected")
                    print(f"  {service}: ready on requested GPU {requested_gpu}", flush=True)
                    return
            except httpx.HTTPError:
                pass
            now = time.monotonic()
            if now >= next_progress:
                print(
                    _waiting_message(service, last_status, requested_gpu),
                    flush=True,
                )
                next_progress = now + PROGRESS_SECONDS
            time.sleep(POLL_SECONDS)
    raise RagCtlError(f"{service} did not become healthy (last status {last_status})")


def _require_exact_models(base_url: str, bearer: str, expected: str, service: str) -> None:
    response = httpx.get(
        f"{base_url}/models",
        headers=_http_headers(bearer),
        timeout=60,
    )
    if response.status_code != 200:
        raise RagCtlError(f"{service} model discovery did not return 200")
    try:
        identities = {item["id"] for item in response.json()["data"]}
    except (KeyError, TypeError, ValueError) as exc:
        raise RagCtlError(f"{service} returned malformed model discovery") from exc
    if identities != {expected}:
        raise RagCtlError(f"{service} advertised an unexpected model identity")


def validate_granite(base_url: str, bearer: str, requested_gpu: str) -> None:
    _wait_authenticated_health(base_url, bearer, "Granite", requested_gpu)
    _require_exact_models(base_url, bearer, GRANITE_MODEL, "Granite")
    prefill = '{"rewritten_question":"'
    payload = {
        "model": GRANITE_MODEL,
        "messages": [
            {"role": "user", "content": "Who is Rex?"},
            {"role": "assistant", "content": "Rex is my dog."},
            {"role": "user", "content": "What has fleas?"},
            {"role": "assistant", "content": "Rex has fleas."},
            {"role": "user", "content": "How do I get rid of them?"},
            {"role": "assistant", "content": prefill},
        ],
        "temperature": 0,
        "max_tokens": 128,
        "n": 1,
        "stream": False,
        "continue_final_message": True,
        "regex": r'(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9A-Fa-f]{4}))+"}\s*',
    }
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={**_http_headers(bearer), "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if response.status_code != 200:
        raise RagCtlError("Granite query rewrite smoke request failed")
    try:
        continuation = response.json()["choices"][0]["message"]["content"]
        assembled = json.loads(prefill + continuation)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RagCtlError("Granite query rewrite was not strict JSON") from exc
    rewritten = assembled.get("rewritten_question") if isinstance(assembled, dict) else None
    if (
        not isinstance(assembled, dict)
        or set(assembled) != {"rewritten_question"}
        or not isinstance(rewritten, str)
        or not rewritten.strip()
    ):
        raise RagCtlError("Granite query rewrite violated the standalone JSON contract")


def _qwen_payload(prompt: str, *, stream: bool, max_tokens: int) -> dict[str, object]:
    return {
        "model": QWEN_MODEL,
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


def validate_qwen(base_url: str, bearer: str, requested_gpu: str) -> None:
    _wait_authenticated_health(base_url, bearer, "Qwen", requested_gpu)
    _require_exact_models(base_url, bearer, QWEN_MODEL, "Qwen")
    headers = {**_http_headers(bearer), "Content-Type": "application/json"}
    title_response = httpx.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=_qwen_payload(
            "Return only a three word title for a discussion about RAG retrieval.",
            stream=False,
            max_tokens=32,
        ),
        timeout=120,
    )
    if title_response.status_code != 200:
        raise RagCtlError("Qwen title smoke request failed")
    try:
        title_message = title_response.json()["choices"][0]["message"]
        title = title_message["content"].strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise RagCtlError("Qwen returned a malformed title response") from exc
    if title_message.get("reasoning_content") not in (None, ""):
        raise RagCtlError("Qwen returned title reasoning with thinking disabled")
    words = title.split()
    if not 3 <= len(words) <= 6:
        raise RagCtlError("Qwen title did not satisfy the 3-6 word contract")

    saw_content = False
    saw_done = False
    with httpx.Client(timeout=httpx.Timeout(30.0, read=180.0)) as client:
        with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers=headers,
            json=_qwen_payload(
                "Answer in one sentence: what is retrieval augmented generation?",
                stream=True,
                max_tokens=64,
            ),
        ) as response:
            if response.status_code != 200:
                raise RagCtlError("Qwen streaming smoke request failed")
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    continue
                event = json.loads(data)
                delta = event["choices"][0]["delta"]
                if delta.get("reasoning_content") not in (None, ""):
                    raise RagCtlError("Qwen streamed reasoning with thinking disabled")
                saw_content = saw_content or bool(delta.get("content"))
    if not saw_content or not saw_done:
        raise RagCtlError("Qwen stream did not complete with content and [DONE]")


def build_runtime_secret(
    config: Mapping[str, str], granite_url: str, qwen_url: str
) -> dict[str, str]:
    bearer = proxy_bearer(config)
    secret = {
        "WEAVIATE_URL": config["WEAVIATE_URL"],
        "WEAVIATE_API_KEY": config["WEAVIATE_API_KEY"],
        "WEAVIATE_CONNECTION_MODE": config["WEAVIATE_CONNECTION_MODE"],
        "WEAVIATE_GRPC_PORT": config["WEAVIATE_GRPC_PORT"],
        "WEAVIATE_GRPC_SECURE": config["WEAVIATE_GRPC_SECURE"],
        "SGLANG_QUERY_REWRITE_BASE_URL": granite_url,
        "SGLANG_QUERY_REWRITE_API_KEY": bearer,
        "QWEN_SGLANG_BASE_URL": qwen_url,
        "QWEN_SGLANG_API_KEY": bearer,
    }
    if tuple(secret) != RUNTIME_SECRET_KEYS:
        raise RagCtlError("Internal runtime secret definition is not the approved nine-key set")
    return secret


def deploy_runtime_secret(secret: Mapping[str, str], runner: CommandRunner) -> None:
    path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix="rag-runtime-secret-", suffix=".json")
        path = Path(raw_path)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(dict(secret), target, separators=(",", ":"))
            target.flush()
            os.fsync(target.fileno())
        runner.run(
            [
                MODAL,
                "secret",
                "create",
                "--force",
                "--from-json",
                path,
                "rag-runtime-secrets",
            ]
        )
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def deploy_runtime(config: Mapping[str, str], runner: CommandRunner) -> str:
    runner.run(
        [MODAL, "deploy", PROJECT_ROOT / "deployment" / "modal_runtime.py"],
        overrides={
            "MODAL_RAG_GPU": gpu_request(config, "MODAL_RAG_GPU"),
            "MODAL_RAG_COMPUTE_REGION": "us",
            "MODAL_RAG_ROUTING_REGION": "us-east",
        },
    )
    return resolve_server_url(RUNTIME_APP, RUNTIME_CLASS)


def _runtime_headers(config: Mapping[str, str]) -> dict[str, str]:
    return {
        "Modal-Key": config["MODAL_PROXY_TOKEN_ID"],
        "Modal-Secret": config["MODAL_PROXY_TOKEN_SECRET"],
        "Accept": "application/json",
    }


def runtime_worker_count(container_id: str, runner: CommandRunner) -> int:
    script = """from pathlib import Path
needle = b"backend.runtime_app" + b":create_runtime_app"

def matches(path):
    try:
        return needle in path.read_bytes()
    except OSError:
        return False

print(sum(matches(path) for path in Path("/proc").glob("[0-9]*/cmdline")))
"""
    result = runner.run(
        [
            MODAL,
            "container",
            "exec",
            container_id,
            "--",
            "python",
            "-c",
            script,
        ],
        capture=True,
        quiet=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    try:
        return int(lines[-1])
    except (IndexError, ValueError) as exc:
        raise RagCtlError("Could not inspect the integrated runtime worker count") from exc


def validate_runtime(
    runtime_url: str,
    config: Mapping[str, str],
    runner: CommandRunner,
    requested_gpu: str,
) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    next_progress = time.monotonic()
    last_status: int | None = None
    with httpx.Client(timeout=httpx.Timeout(30.0, read=120.0)) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(
                    f"{runtime_url}/health", headers=_runtime_headers(config)
                )
                last_status = response.status_code
                if response.status_code == 200 and response.json() == {"status": "ok"}:
                    print(
                        f"  Integrated runtime: ready on requested GPU {requested_gpu}",
                        flush=True,
                    )
                    break
            except (httpx.HTTPError, ValueError):
                pass
            now = time.monotonic()
            if now >= next_progress:
                print(
                    _waiting_message("Integrated runtime", last_status, requested_gpu),
                    flush=True,
                )
                next_progress = now + PROGRESS_SECONDS
            time.sleep(POLL_SECONDS)
        else:
            raise RagCtlError(
                f"Integrated runtime did not become healthy (last status {last_status})"
            )
        if client.get(f"{runtime_url}/health").status_code != 401:
            raise RagCtlError("Integrated runtime anonymous health was not rejected")
        if client.get(
            f"{runtime_url}/dev/e2e", headers=_runtime_headers(config)
        ).status_code != 200:
            raise RagCtlError("Integrated runtime /dev/e2e did not return 200")

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        containers = target_containers(runner)
        runtime = [item for item in containers if item.get("app_name") == RUNTIME_APP]
        if len(runtime) == 1 and runtime[0].get("start_time") != "Pending":
            container_id = str(runtime[0]["container_id"])
            if runtime_worker_count(container_id, runner) != 1:
                raise RagCtlError("Integrated runtime does not have exactly one Uvicorn worker")
            return
        time.sleep(3)
    raise RagCtlError("Integrated runtime does not have exactly one running container")


def target_containers(runner: CommandRunner) -> list[dict[str, object]]:
    payload = _json_command(runner, [MODAL, "container", "list", "--json"])
    if not isinstance(payload, list):
        raise RagCtlError("Modal container list returned an unexpected payload")
    return [item for item in payload if item.get("app_name") in MODAL_APPS]


def write_state(runtime_url: str, granite_url: str, qwen_url: str) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "runtime_url": runtime_url,
                "granite_url": granite_url,
                "qwen_url": qwen_url,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, STATE_PATH)


def read_runtime_url(config: Mapping[str, str]) -> str:
    if STATE_PATH.is_file():
        try:
            value = json.loads(STATE_PATH.read_text(encoding="utf-8"))["runtime_url"]
            if isinstance(value, str) and value.startswith("https://"):
                return value.rstrip("/")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    configured = config.get("RAG_API_BASE_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    raise RagCtlError("No runtime endpoint is known; run ./rag up first")


def up(config: Mapping[str, str], runner: CommandRunner) -> None:
    validate_config(config)
    try:
        print("[1/8] Checking local prerequisites", flush=True)
        hostname = preflight(config, runner)
        ensure_certificate(hostname, runner)

        print("[2/8] Starting authenticated Weaviate and HAProxy", flush=True)
        compose_up(runner)
        configure_funnels(hostname, runner)
        verify_weaviate(config)

        print("[3/8] Deploying and validating private Granite", flush=True)
        bearer = proxy_bearer(config)
        granite_gpu = gpu_request(config, "MODAL_SGLANG_GPU")
        granite_url = deploy_granite(config, runner)
        validate_granite(granite_url, bearer, granite_gpu)

        print("[4/8] Deploying and validating private Qwen", flush=True)
        qwen_gpu = gpu_request(config, "QWEN_MODAL_SGLANG_GPU")
        qwen_url = deploy_qwen(config, runner)
        validate_qwen(qwen_url, bearer, qwen_gpu)

        print("[5/8] Publishing the exact nine-value runtime secret", flush=True)
        deploy_runtime_secret(
            build_runtime_secret(config, granite_url, qwen_url), runner
        )

        print("[6/8] Deploying the singleton CUDA runtime", flush=True)
        runtime_gpu = gpu_request(config, "MODAL_RAG_GPU")
        runtime_url = deploy_runtime(config, runner)

        print("[7/8] Waiting for authenticated runtime readiness", flush=True)
        validate_runtime(runtime_url, config, runner, runtime_gpu)
        print("[8/8] Recording non-secret ready endpoint state", flush=True)
        write_state(runtime_url, granite_url, qwen_url)
    except BaseException:
        print("Startup failed; stopping application resources to prevent GPU charges…", file=sys.stderr)
        try:
            down(config, runner, remove_state=True)
        except Exception as cleanup_error:
            print(
                "Cleanup also reported: " + redact(cleanup_error, config),
                file=sys.stderr,
            )
        raise
    print("RAG is ready — run: ./rag ask", flush=True)


def disable_funnels(runner: CommandRunner) -> None:
    if not TAILSCALE.is_file():
        return
    runner.run(
        [TAILSCALE, "funnel", "--https=443", "off"],
        check=False,
        capture=True,
        quiet=True,
    )
    runner.run(
        [TAILSCALE, "funnel", "--tcp=8443", "off"],
        check=False,
        capture=True,
        quiet=True,
    )


def compose_down(runner: CommandRunner) -> None:
    runner.run(
        [
            "docker",
            "compose",
            "--env-file",
            ENV_PATH,
            "-f",
            COMPOSE_FILE,
            "down",
        ],
        check=False,
    )


def stop_modal_apps(runner: CommandRunner) -> None:
    for app_name in MODAL_APPS:
        runner.run([MODAL, "app", "stop", "-y", app_name], check=False)


def down(
    config: Mapping[str, str], runner: CommandRunner, *, remove_state: bool = True
) -> None:
    print("Stopping Modal GPU services…", flush=True)
    stop_modal_apps(runner)
    print("Disabling application Funnels…", flush=True)
    disable_funnels(runner)
    print("Stopping Weaviate and HAProxy (persistent volume retained)…", flush=True)
    compose_down(runner)
    if remove_state:
        STATE_PATH.unlink(missing_ok=True)

    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            if not target_containers(runner):
                break
        except Exception:
            break
        time.sleep(3)
    else:
        raise RagCtlError("One or more RAG GPU containers did not stop")

    volume = runner.run(
        ["docker", "volume", "inspect", "rag_weaviate_secure_data"],
        capture=True,
        check=False,
        quiet=True,
    )
    if volume.returncode != 0:
        raise RagCtlError("Persistent Weaviate volume is missing after shutdown")
    print("RAG is off; Modal Volumes, secrets, and Weaviate data are preserved.")


def iter_sse(lines: Iterable[str]) -> Iterator[tuple[str, object]]:
    event_name = "message"
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines:
                yield event_name, json.loads("\n".join(data_lines))
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield event_name, json.loads("\n".join(data_lines))


def validate_chat_result(
    events: Sequence[str],
    answer_parts: Sequence[str],
    telemetry: Mapping[str, object] | None,
    done_payload: Mapping[str, object] | None,
    *,
    verify_atlas_grounding: bool,
) -> tuple[str, Mapping[str, object]]:
    if "error" in events:
        raise RagCtlError("RAG stream contained an error event")
    if events.count("token") < 1:
        raise RagCtlError("RAG stream returned no tokens")
    if events.count("telemetry") != 1 or events.count("done") != 1:
        raise RagCtlError("RAG stream did not return exactly one telemetry and done event")
    if list(events[-2:]) != ["telemetry", "done"]:
        raise RagCtlError("RAG stream ordering was not token* -> telemetry -> done")
    if telemetry is None or done_payload is None:
        raise RagCtlError("RAG stream ended without telemetry or done")
    timings = telemetry.get("timings_ms")
    if not isinstance(timings, Mapping):
        raise RagCtlError("RAG telemetry is missing timings_ms")
    if set(timings) != set(CHAT_TIMING_KEYS):
        raise RagCtlError("RAG telemetry does not match the documented timing schema")
    for name, value in timings.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise RagCtlError(f"RAG telemetry timing {name!r} is invalid")

    answer = "".join(answer_parts)
    if verify_atlas_grounding:
        normalized = answer.casefold()
        groups = {
            "Atlas guide": ("atlas", "guide"),
            "manager approval": ("manager", "approval"),
            "rollback checklist": ("rollback", "checklist"),
        }
        missing = [
            label
            for label, terms in groups.items()
            if not all(term in normalized for term in terms)
        ]
        if missing:
            raise RagCtlError("Answer lacks required grounding: " + ", ".join(missing))
    return answer, timings


def ask(
    config: Mapping[str, str],
    question: str | None,
    *,
    verify_atlas_grounding: bool = False,
) -> None:
    runtime_url = read_runtime_url(config)
    question_text = (question if question is not None else input("Question: ")).strip()
    if not question_text:
        raise RagCtlError("Question must not be empty")
    user_id = config.get("RAG_USER_ID", DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
    headers = _runtime_headers(config)
    timeout = httpx.Timeout(connect=30, read=900, write=120, pool=30)
    with httpx.Client(headers=headers, timeout=timeout) as client:
        health = client.get(f"{runtime_url}/health")
        if health.status_code != 200:
            raise RagCtlError("RAG runtime is not ready; run ./rag up first")
        created = client.post(
            f"{runtime_url}/api/chat/sessions", json={"user_id": user_id}
        )
        if created.status_code != 201:
            raise RagCtlError("Could not create a fresh chat session")
        try:
            session_id = created.json()["session_id"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RagCtlError("Session creation returned malformed JSON") from exc

        print(f"Session: {session_id}")
        print("Answer: ", end="", flush=True)
        events: list[str] = []
        answer_parts: list[str] = []
        telemetry: Mapping[str, object] | None = None
        done_payload: Mapping[str, object] | None = None
        with client.stream(
            "POST",
            f"{runtime_url}/api/chat/query",
            headers={"Accept": "text/event-stream"},
            json={
                "user_id": user_id,
                "session_id": session_id,
                "question": question_text,
            },
        ) as response:
            if response.status_code != 200:
                raise RagCtlError("Chat request did not return an SSE stream")
            for event_name, payload in iter_sse(response.iter_lines()):
                events.append(event_name)
                if event_name == "error":
                    raise RagCtlError(f"RAG stream failed: {payload}")
                if event_name == "token":
                    if not isinstance(payload, Mapping) or not isinstance(
                        payload.get("text"), str
                    ) or not payload["text"]:
                        raise RagCtlError("RAG stream returned an invalid token event")
                    answer_parts.append(payload["text"])
                    print(payload["text"], end="", flush=True)
                elif event_name == "telemetry":
                    if telemetry is not None or not isinstance(payload, Mapping):
                        raise RagCtlError("RAG stream returned invalid telemetry")
                    telemetry = payload
                elif event_name == "done":
                    if done_payload is not None or not isinstance(payload, Mapping):
                        raise RagCtlError("RAG stream returned an invalid done event")
                    done_payload = payload
                else:
                    raise RagCtlError(f"RAG stream returned unexpected event {event_name!r}")
        print()

    _, timings = validate_chat_result(
        events,
        answer_parts,
        telemetry,
        done_payload,
        verify_atlas_grounding=verify_atlas_grounding,
    )

    summary = {name: timings[name] for name in CHAT_TIMING_KEYS}
    print("Telemetry (ms): " + json.dumps(summary, sort_keys=True))
    print("Done: " + json.dumps(dict(done_payload), sort_keys=True))
    print(f"SSE verified: {events.count('token')} token event(s) -> telemetry -> done")


def status(config: Mapping[str, str], runner: CommandRunner) -> None:
    print("RAG service status (credentials redacted)")
    compose = runner.run(
        [
            "docker",
            "compose",
            "--env-file",
            ENV_PATH,
            "-f",
            COMPOSE_FILE,
            "ps",
            "--format",
            "json",
        ],
        capture=True,
        check=False,
        quiet=True,
    )
    healthy_local = compose.returncode == 0 and '"Health":"healthy"' in compose.stdout
    print(f"  Local Weaviate/HAProxy: {'healthy' if healthy_local else 'off or unhealthy'}")

    funnel = runner.run(
        [TAILSCALE, "funnel", "status", "--json"],
        capture=True,
        check=False,
        quiet=True,
    )
    funnels_ready = False
    if funnel.returncode == 0:
        try:
            payload = json.loads(funnel.stdout)
            forwards = payload.get("TCP", {})
            funnels_ready = "443" in forwards and "8443" in forwards
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    print(f"  Tailscale Funnels: {'configured' if funnels_ready else 'off'}")

    try:
        containers = target_containers(runner)
    except Exception:
        containers = []
    counts = {name: 0 for name in MODAL_APPS}
    pending = {name: 0 for name in MODAL_APPS}
    for item in containers:
        name = str(item.get("app_name"))
        if item.get("start_time") == "Pending":
            pending[name] += 1
        else:
            counts[name] += 1
    for name in (GRANITE_APP, QWEN_APP, RUNTIME_APP):
        suffix = f", {pending[name]} pending" if pending[name] else ""
        print(f"  {name}: {counts[name]} running{suffix}")

    try:
        runtime_url = read_runtime_url(config)
        response = httpx.get(
            f"{runtime_url}/health",
            headers=_runtime_headers(config),
            timeout=20,
        )
        ready = response.status_code == 200
    except Exception:
        ready = False
    print(f"  Authenticated RAG health: {'200 ready' if ready else 'not ready'}")


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        prog="./rag",
        description="Start, query, inspect, or stop the private Modal RAG application.",
    )
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("up", help="start and validate every RAG service")
    ask_parser = commands.add_parser("ask", help="ask one question in a fresh session")
    ask_parser.add_argument("question", nargs="?", help="prompt when omitted")
    ask_parser.add_argument(
        "--verify-atlas-grounding",
        action="store_true",
        help="require the seeded Atlas and Policy P-17 facts in the answer",
    )
    commands.add_parser("status", help="show redacted service status")
    commands.add_parser("down", help="stop all application services and preserve data")
    return cli


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_dotenv(ENV_PATH)
    config.setdefault("RAG_USER_ID", DEFAULT_USER_ID)
    runner = CommandRunner(config)
    try:
        if args.command == "up":
            up(config, runner)
        elif args.command == "ask":
            validate_config(config)
            ask(
                config,
                args.question,
                verify_atlas_grounding=args.verify_atlas_grounding,
            )
        elif args.command == "status":
            status(config, runner)
        elif args.command == "down":
            down(config, runner)
        else:  # pragma: no cover - argparse owns this invariant
            raise RagCtlError(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (RagCtlError, httpx.HTTPError, OSError, ValueError, KeyError) as exc:
        print("ERROR: " + redact(exc, config), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
