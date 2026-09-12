#!/usr/bin/env python3
"""Contract-faithful lifecycle control for the private Modal RAG deployment."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Iterable, Iterator, Mapping, Sequence
import hashlib
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
from uuid import uuid4
import warnings

import httpx

from backend.api.telemetry import TIMING_KEYS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
STATE_PATH = PROJECT_ROOT / ".local" / "rag-state.json"
WIZARD_DIAGNOSTICS_PATH = PROJECT_ROOT / ".local" / "diagnostics" / "wizard"
WIZARD_CORPUS_STATE_PATH = WIZARD_DIAGNOSTICS_PATH / "corpus-state.json"
WIZARD_CORPUS_LOCK_PATH = WIZARD_DIAGNOSTICS_PATH / "corpus-state.lock"
E2E_DIAGNOSTICS_PATH = PROJECT_ROOT / ".local" / "diagnostics" / "e2e"
ASK_DIAGNOSTICS_PATH = PROJECT_ROOT / ".local" / "diagnostics" / "ask"
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
LOCAL_WEAVIATE_REST_URL = "http://127.0.0.1:8080"
LOCAL_WEAVIATE_GRPC_TLS_TARGET = "127.0.0.1:5443"
REST_FUNNEL_PORT = 443
GRPC_FUNNEL_PORT = 8443

DEFAULT_OPERATIONAL_GPU = "L40S"
OPERATIONAL_GPU_KEYS = (
    "MODAL_SGLANG_GPU",
    "QWEN_MODAL_SGLANG_GPU",
    "MODAL_RAG_GPU",
)
GPU_DEFAULTS = dict.fromkeys(OPERATIONAL_GPU_KEYS, DEFAULT_OPERATIONAL_GPU)
ALLOWED_OPERATIONAL_GPUS = frozenset({"L40S", "H100"})
# Compatibility name for launcher callers; the API telemetry module owns the tuple.
CHAT_TIMING_KEYS = TIMING_KEYS

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

MODAL_TARGET_KEYS = (
    "MODAL_PROFILE",
    "MODAL_WORKSPACE",
    "MODAL_ENVIRONMENT",
)

REQUIRED_ENV_KEYS = (
    "WEAVIATE_URL",
    "WEAVIATE_API_KEY",
    "WEAVIATE_CONNECTION_MODE",
    "WEAVIATE_GRPC_PORT",
    "WEAVIATE_GRPC_SECURE",
    "MODAL_PROXY_TOKEN_ID",
    "MODAL_PROXY_TOKEN_SECRET",
    *MODAL_TARGET_KEYS,
)
RUNTIME_DIAGNOSTIC_ENV_KEYS = frozenset(
    {
        "WIZARD_DIAGNOSTICS_ENABLED",
        "RAG_DIAGNOSTIC_USER_ID",
        "RAG_EVALUATION_EVIDENCE_ENABLED",
        "RAG_EVALUATION_USER_ID",
    }
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
    if grpc_port != GRPC_FUNNEL_PORT:
        raise RagCtlError(
            f"WEAVIATE_GRPC_PORT must remain {GRPC_FUNNEL_PORT} for the secure Funnel"
        )

    modal_target(config)
    if any(
        os.environ.get(name, "").strip() or config.get(name, "").strip()
        for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
    ):
        raise RagCtlError(
            "Modal token environment overrides are forbidden; use the configured profile"
        )

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


def modal_target(config: Mapping[str, str]) -> tuple[str, str, str]:
    values = tuple(config.get(name, "").strip() for name in MODAL_TARGET_KEYS)
    missing = [name for name, value in zip(MODAL_TARGET_KEYS, values) if not value]
    if missing:
        raise RagCtlError("Missing required .env variables: " + ", ".join(missing))
    return values  # type: ignore[return-value]


_MODAL_TARGET_PROBE = """
import os
import modal.config as modal_config
from modal.environments import Environment
from modal.workspace import Workspace

expected_profile = os.environ["MODAL_PROFILE"]
expected_workspace = os.environ["MODAL_WORKSPACE"]
expected_environment = os.environ["MODAL_ENVIRONMENT"]
if modal_config._config_active_profile() != expected_profile:
    raise SystemExit("the active Modal profile does not match MODAL_PROFILE")
if modal_config._profile != expected_profile:
    raise SystemExit("the effective Modal profile does not match MODAL_PROFILE")
workspace = Workspace.from_context()
workspace.hydrate()
if workspace.name != expected_workspace:
    raise SystemExit("the authenticated Modal workspace does not match MODAL_WORKSPACE")
Environment.from_name(expected_environment).hydrate()
"""


def verify_modal_target(config: Mapping[str, str]) -> None:
    """Fail closed unless the configured Modal account and environment are active."""

    profile, workspace, environment_name = modal_target(config)
    if any(
        os.environ.get(name, "").strip() or config.get(name, "").strip()
        for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
    ):
        raise RagCtlError(
            "Modal token environment overrides are forbidden; use the configured profile"
        )
    probe_environment = os.environ.copy()
    probe_environment.update(
        {
            "MODAL_PROFILE": profile,
            "MODAL_WORKSPACE": workspace,
            "MODAL_ENVIRONMENT": environment_name,
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", _MODAL_TARGET_PROBE],
        cwd=PROJECT_ROOT,
        env=probe_environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "target verification failed").strip()
        raise RagCtlError(
            "Modal target verification failed: " + redact(detail, config)
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
        if Path(command[0]).resolve() == MODAL.resolve():
            verify_modal_target(self.config)
        if not quiet:
            shown = " ".join(shlex.quote(item) for item in command)
            print(f"→ {redact(shown, self.config)}", flush=True)
        environment = os.environ.copy()
        for name in RUNTIME_DIAGNOSTIC_ENV_KEYS:
            environment.pop(name, None)
        environment.update(
            {
                name: value
                for name, value in self.config.items()
                if name not in RUNTIME_DIAGNOSTIC_ENV_KEYS
            }
        )
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
    verify_modal_target(config)
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
        [TAILSCALE, "funnel", f"--https={REST_FUNNEL_PORT}", "off"],
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
            f"--https={REST_FUNNEL_PORT}",
            LOCAL_WEAVIATE_REST_URL,
        ]
    )
    runner.run(
        [TAILSCALE, "funnel", f"--tcp={GRPC_FUNNEL_PORT}", "off"],
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
            f"--tcp={GRPC_FUNNEL_PORT}",
            f"tcp://{LOCAL_WEAVIATE_GRPC_TLS_TARGET}",
        ]
    )
    status = _json_command(runner, [TAILSCALE, "funnel", "status", "--json"])
    web = status.get("Web", {}).get(
        f"{hostname}:{REST_FUNNEL_PORT}", {}
    ).get("Handlers", {})
    rest_proxy = web.get("/", {}).get("Proxy")
    tcp_forward = status.get("TCP", {}).get(
        str(GRPC_FUNNEL_PORT), {}
    ).get("TCPForward")
    allowed = status.get("AllowFunnel", {})
    if (
        rest_proxy != LOCAL_WEAVIATE_REST_URL
        or tcp_forward != LOCAL_WEAVIATE_GRPC_TLS_TARGET
        or allowed.get(f"{hostname}:{REST_FUNNEL_PORT}") is not True
        or allowed.get(f"{hostname}:{GRPC_FUNNEL_PORT}") is not True
    ):
        raise RagCtlError("Tailscale did not register both Funnels on the active node")


def _http_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}


def verify_weaviate(config: Mapping[str, str]) -> None:
    key = config["WEAVIATE_API_KEY"]
    local = LOCAL_WEAVIATE_REST_URL
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
    )
    return resolve_server_url(GRANITE_APP, GRANITE_CLASS, config) + "/v1"


def deploy_qwen(config: Mapping[str, str], runner: CommandRunner) -> str:
    runner.run(
        [MODAL, "deploy", PROJECT_ROOT / "deployment" / "modal_qwen_sglang.py"],
    )
    return resolve_server_url(QWEN_APP, QWEN_CLASS, config) + "/v1"


def resolve_server_url(
    app_name: str,
    class_name: str,
    config: Mapping[str, str],
) -> str:
    profile, _, environment_name = modal_target(config)
    verify_modal_target(config)
    os.environ["MODAL_PROFILE"] = profile
    os.environ["MODAL_ENVIRONMENT"] = environment_name
    import modal

    url = modal.Server.from_name(
        app_name,
        class_name,
        environment_name=environment_name,
    ).get_url()
    if not isinstance(url, str) or not url.startswith("https://"):
        raise RagCtlError("Modal Server URL discovery returned an invalid endpoint")
    return url.rstrip("/")


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


def deploy_runtime(
    config: Mapping[str, str],
    runner: CommandRunner,
    *,
    wizard_diagnostic_user_id: str | None = None,
    evaluation_evidence_enabled: bool = True,
) -> str:
    evaluation_user_id = (
        config.get("RAG_USER_ID", DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
    )
    overrides = {
        "WIZARD_DIAGNOSTICS_ENABLED": "false",
        "RAG_EVALUATION_EVIDENCE_ENABLED": (
            "true" if evaluation_evidence_enabled else "false"
        ),
    }
    if evaluation_evidence_enabled:
        overrides["RAG_EVALUATION_USER_ID"] = evaluation_user_id
    if wizard_diagnostic_user_id is not None:
        overrides = {
            "WIZARD_DIAGNOSTICS_ENABLED": "true",
            "RAG_DIAGNOSTIC_USER_ID": wizard_diagnostic_user_id,
            "RAG_EVALUATION_EVIDENCE_ENABLED": "false",
        }
    runner.run(
        [MODAL, "deploy", PROJECT_ROOT / "deployment" / "modal_runtime.py"],
        overrides=overrides,
    )
    return resolve_server_url(RUNTIME_APP, RUNTIME_CLASS, config)


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


def up(
    config: Mapping[str, str],
    runner: CommandRunner,
    *,
    wizard_diagnostic_user_id: str | None = None,
    evaluation_evidence_enabled: bool = True,
) -> None:
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

        print("[5/8] Publishing the current approved runtime secret", flush=True)
        deploy_runtime_secret(
            build_runtime_secret(config, granite_url, qwen_url), runner
        )

        print("[6/8] Deploying the singleton CUDA runtime", flush=True)
        runtime_gpu = gpu_request(config, "MODAL_RAG_GPU")
        if wizard_diagnostic_user_id is None:
            runtime_url = (
                deploy_runtime(config, runner)
                if evaluation_evidence_enabled
                else deploy_runtime(
                    config, runner, evaluation_evidence_enabled=False
                )
            )
        else:
            runtime_url = deploy_runtime(
                config,
                runner,
                wizard_diagnostic_user_id=wizard_diagnostic_user_id,
            )

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
    print("Advanced RAG Application is ready — run: ./rag ask", flush=True)


def disable_funnels(runner: CommandRunner) -> None:
    if not TAILSCALE.is_file():
        return
    runner.run(
        [TAILSCALE, "funnel", f"--https={REST_FUNNEL_PORT}", "off"],
        check=False,
        capture=True,
        quiet=True,
    )
    runner.run(
        [TAILSCALE, "funnel", f"--tcp={GRPC_FUNNEL_PORT}", "off"],
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


def diagnose_wizard(
    config: Mapping[str, str],
    runner: CommandRunner,
    fixtures_path: Path,
    *,
    reseed_corpus: bool = False,
) -> None:
    """Exercise the real Wizard API and retain its verified retrieval corpus."""

    validate_config(config)
    from deployment.wizard_diagnostic import (
        OperationRecorder,
        corpus_state_lock,
        create_diagnostic_run,
        load_corpus_state,
        preflight_wizard_fixtures,
        update_run_summary,
        validate_corpus_preflight,
        validate_diagnostic_user,
    )
    from deployment.wizard_diagnostic_api import run_wizard_phase_1c

    diagnostic_user_id, other_user_id = validate_diagnostic_user(config)
    fixtures = preflight_wizard_fixtures(fixtures_path, PROJECT_ROOT, config)
    with corpus_state_lock(WIZARD_CORPUS_LOCK_PATH):
        corpus_state = load_corpus_state(
            WIZARD_CORPUS_STATE_PATH, diagnostic_user_id
        )
        validate_corpus_preflight(
            corpus_state,
            fixtures,
            reseed=reseed_corpus,
        )
        run = create_diagnostic_run(
            WIZARD_DIAGNOSTICS_PATH,
            diagnostic_user_id,
            fixtures,
        )
        recorder = OperationRecorder(run.operations_path)
        progress = {
            "scratch_status": "pending",
            "corpus_action": "pending",
            "telemetry_status": "pending",
            "security_status": "pending",
            "trace_deleted": False,
        }
        print(f"Wizard diagnostic run: {run.run_id}", flush=True)
        print(f"Artifacts: {run.directory}", flush=True)

        try:
            down(config, runner)
        except KeyboardInterrupt:
            update_run_summary(
                run,
                status="interrupted",
                pre_down_status="interrupted",
                up_status="not_started",
                down_status="not_started",
                failure_stage="pre_down",
                finished=True,
                operations_count=recorder.count,
                operation_totals=recorder.totals,
                **progress,
            )
            raise
        except BaseException:
            update_run_summary(
                run,
                status="failed",
                pre_down_status="failed",
                up_status="not_started",
                down_status="not_started",
                failure_stage="pre_down",
                finished=True,
                operations_count=recorder.count,
                operation_totals=recorder.totals,
                **progress,
            )
            raise

        update_run_summary(
            run,
            status="running",
            pre_down_status="succeeded",
            up_status="pending",
            down_status="pending",
            operations_count=recorder.count,
            operation_totals=recorder.totals,
            **progress,
        )
        try:
            up(
                config,
                runner,
                wizard_diagnostic_user_id=diagnostic_user_id,
            )
        except KeyboardInterrupt:
            update_run_summary(
                run,
                status="interrupted",
                pre_down_status="succeeded",
                up_status="interrupted",
                down_status="managed_by_up_failure_cleanup",
                failure_stage="up",
                finished=True,
                operations_count=recorder.count,
                operation_totals=recorder.totals,
                **progress,
            )
            raise
        except BaseException:
            update_run_summary(
                run,
                status="failed",
                pre_down_status="succeeded",
                up_status="failed",
                down_status="managed_by_up_failure_cleanup",
                failure_stage="up",
                finished=True,
                operations_count=recorder.count,
                operation_totals=recorder.totals,
                **progress,
            )
            raise

        phase_error: BaseException | None = None
        try:
            update_run_summary(
                run,
                status="running",
                pre_down_status="succeeded",
                up_status="succeeded",
                down_status="pending",
                operations_count=recorder.count,
                operation_totals=recorder.totals,
                **progress,
            )
            run_wizard_phase_1c(
                read_runtime_url(config),
                _runtime_headers(config),
                config,
                fixtures,
                WIZARD_CORPUS_STATE_PATH,
                corpus_state,
                diagnostic_user_id,
                other_user_id,
                run.run_id,
                reseed_corpus,
                recorder,
                progress,
            )
        except BaseException as exc:
            phase_error = exc

        try:
            down(config, runner)
        except BaseException as down_error:
            if phase_error is not None:
                down_error.add_note(
                    "Phase 1C also failed before shutdown: " + repr(phase_error)
                )
            interrupted = isinstance(down_error, KeyboardInterrupt)
            update_run_summary(
                run,
                status="interrupted" if interrupted else "failed",
                pre_down_status="succeeded",
                up_status="succeeded",
                down_status="interrupted" if interrupted else "failed",
                failure_stage="down",
                finished=True,
                operations_count=recorder.count,
                operation_totals=recorder.totals,
                **progress,
            )
            raise

        if phase_error is not None:
            interrupted = isinstance(phase_error, KeyboardInterrupt)
            failure_stage = (
                "scratch"
                if progress["scratch_status"] != "passed"
                else "corpus"
                if progress["corpus_action"] == "failed"
                else "diagnostic"
            )
            update_run_summary(
                run,
                status="interrupted" if interrupted else "failed",
                pre_down_status="succeeded",
                up_status="succeeded",
                down_status="succeeded",
                failure_stage=failure_stage,
                finished=True,
                operations_count=recorder.count,
                operation_totals=recorder.totals,
                **progress,
            )
            raise phase_error

        update_run_summary(
            run,
            status="succeeded",
            pre_down_status="succeeded",
            up_status="succeeded",
            down_status="succeeded",
            finished=True,
            operations_count=recorder.count,
            operation_totals=recorder.totals,
            **progress,
        )
        print(
            "Wizard diagnostic Phase 1C completed; "
            f"corpus {progress['corpus_action']}.",
            flush=True,
        )


def diagnose_e2e(
    config: Mapping[str, str],
    runner: CommandRunner,
    queries_path: Path,
    *,
    start: int = 0,
    limit: int | None = None,
    continuous: bool = False,
) -> None:
    """Run Phase 2D queries against the retained, verified Phase 1 corpus."""

    validate_config(config)
    from backend.wizard.diagnostics import TRACE_SESSION_MAX_OPERATIONS
    from deployment.e2e_diagnostic import (
        RequestRecorder,
        create_e2e_run,
        load_query_selection,
        update_e2e_summary,
        validate_reusable_corpus_state,
    )
    from deployment.e2e_diagnostic_api import run_e2e_phase_2d
    from deployment.wizard_diagnostic import (
        corpus_state_lock,
        load_corpus_state,
        validate_diagnostic_user,
    )

    diagnostic_user_id, _ = validate_diagnostic_user(config)
    selection = load_query_selection(
        queries_path,
        PROJECT_ROOT,
        start=start,
        limit=limit,
    )
    if len(selection.selected) > TRACE_SESSION_MAX_OPERATIONS:
        raise RagCtlError(
            "E2E selection exceeds the bounded diagnostic trace capacity"
        )
    with corpus_state_lock(WIZARD_CORPUS_LOCK_PATH):
        corpus_state = load_corpus_state(
            WIZARD_CORPUS_STATE_PATH,
            diagnostic_user_id,
        )
        active = validate_reusable_corpus_state(corpus_state, diagnostic_user_id)
        if corpus_state is None:  # pragma: no cover - validator owns this invariant.
            raise RagCtlError("Phase 1 corpus state is unavailable")
        run = create_e2e_run(
            E2E_DIAGNOSTICS_PATH,
            diagnostic_user_id,
            selection,
            active,
            continuous=continuous,
        )
        recorder = RequestRecorder(run.requests_path)
        progress: dict[str, object] = {
            "physical_corpus_status": "pending",
            "trace_status": "not_started",
            "trace_deleted": False,
        }
        print(f"E2E diagnostic run: {run.run_id}", flush=True)
        print(f"Artifacts: {run.directory}", flush=True)

        try:
            down(config, runner)
        except KeyboardInterrupt:
            update_e2e_summary(
                run,
                recorder,
                status="interrupted",
                pre_down_status="interrupted",
                up_status="not_started",
                down_status="not_started",
                physical_corpus_status=progress["physical_corpus_status"],
                trace_status=str(progress["trace_status"]),
                trace_deleted=bool(progress["trace_deleted"]),
                failure_stage="pre_down",
                finished=True,
            )
            raise
        except BaseException:
            update_e2e_summary(
                run,
                recorder,
                status="failed",
                pre_down_status="failed",
                up_status="not_started",
                down_status="not_started",
                physical_corpus_status=progress["physical_corpus_status"],
                trace_status=str(progress["trace_status"]),
                trace_deleted=bool(progress["trace_deleted"]),
                failure_stage="pre_down",
                finished=True,
            )
            raise

        update_e2e_summary(
            run,
            recorder,
            status="running",
            pre_down_status="succeeded",
            up_status="pending",
            down_status="pending",
            physical_corpus_status=progress["physical_corpus_status"],
            trace_status=str(progress["trace_status"]),
            trace_deleted=bool(progress["trace_deleted"]),
        )
        try:
            up(
                config,
                runner,
                wizard_diagnostic_user_id=diagnostic_user_id,
            )
        except KeyboardInterrupt:
            update_e2e_summary(
                run,
                recorder,
                status="interrupted",
                pre_down_status="succeeded",
                up_status="interrupted",
                down_status="managed_by_up_failure_cleanup",
                physical_corpus_status=progress["physical_corpus_status"],
                trace_status=str(progress["trace_status"]),
                trace_deleted=bool(progress["trace_deleted"]),
                failure_stage="up",
                finished=True,
            )
            raise
        except BaseException:
            update_e2e_summary(
                run,
                recorder,
                status="failed",
                pre_down_status="succeeded",
                up_status="failed",
                down_status="managed_by_up_failure_cleanup",
                physical_corpus_status=progress["physical_corpus_status"],
                trace_status=str(progress["trace_status"]),
                trace_deleted=bool(progress["trace_deleted"]),
                failure_stage="up",
                finished=True,
            )
            raise

        phase_error: BaseException | None = None
        try:
            update_e2e_summary(
                run,
                recorder,
                status="running",
                pre_down_status="succeeded",
                up_status="succeeded",
                down_status="pending",
                physical_corpus_status=progress["physical_corpus_status"],
                trace_status=str(progress["trace_status"]),
                trace_deleted=bool(progress["trace_deleted"]),
            )
            run_e2e_phase_2d(
                read_runtime_url(config),
                _runtime_headers(config),
                config,
                corpus_state,
                active,
                selection,
                recorder,
                progress,
                continuous=continuous,
                run_id=run.run_id,
            )
        except BaseException as exc:
            phase_error = exc

        try:
            down(config, runner)
        except BaseException as down_error:
            if phase_error is not None:
                down_error.add_note(
                    "Phase 2D also failed before shutdown: " + repr(phase_error)
                )
            interrupted = isinstance(down_error, KeyboardInterrupt)
            update_e2e_summary(
                run,
                recorder,
                status="interrupted" if interrupted else "failed",
                pre_down_status="succeeded",
                up_status="succeeded",
                down_status="interrupted" if interrupted else "failed",
                physical_corpus_status=progress["physical_corpus_status"],
                trace_status=str(progress["trace_status"]),
                trace_deleted=bool(progress["trace_deleted"]),
                failure_stage="down",
                finished=True,
            )
            raise

        if phase_error is not None:
            interrupted = isinstance(phase_error, KeyboardInterrupt)
            update_e2e_summary(
                run,
                recorder,
                status="interrupted" if interrupted else "failed",
                pre_down_status="succeeded",
                up_status="succeeded",
                down_status="succeeded",
                physical_corpus_status=progress["physical_corpus_status"],
                trace_status=str(progress["trace_status"]),
                trace_deleted=bool(progress["trace_deleted"]),
                failure_stage=(
                    "corpus"
                    if progress["physical_corpus_status"] != "succeeded"
                    else (
                        "trace"
                        if progress["trace_status"] == "failed"
                        or not progress["trace_deleted"]
                        else "requests"
                    )
                ),
                finished=True,
            )
            raise phase_error

        update_e2e_summary(
            run,
            recorder,
            status="succeeded",
            pre_down_status="succeeded",
            up_status="succeeded",
            down_status="succeeded",
            physical_corpus_status=progress["physical_corpus_status"],
            trace_status=str(progress["trace_status"]),
            trace_deleted=bool(progress["trace_deleted"]),
            finished=True,
        )
        print(
            f"E2E diagnostic Phase 2D completed: {recorder.succeeded} request(s).",
            flush=True,
        )


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
    acceptance_activity_path: Path | None = None,
) -> Mapping[str, object]:
    from deployment.e2e_diagnostic import utc_timestamp
    from deployment.e2e_diagnostic_api import _DeepTraceSession
    from deployment.evaluation_bridge import (
        EvaluationBridgeError,
        EvaluationObservation,
        cancel_evaluation_job,
        finish_evaluation_job,
        parse_request_evidence,
        submit_local_evaluation,
        write_private_json,
    )

    runtime_url = read_runtime_url(config)
    question_text = (question if question is not None else input("Question: ")).strip()
    if not question_text:
        raise RagCtlError("Question must not be empty")
    user_id = config.get("RAG_USER_ID", DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
    headers = _runtime_headers(config)
    timeout = httpx.Timeout(connect=30, read=900, write=120, pool=30)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid4().hex[:12]
    trace_operation_id = str(uuid4())
    trace_error_code: str | None = None
    trace_operation: Mapping[str, Any] | None = None
    trace: _DeepTraceSession | None = None
    response_request_id: str | None = None
    done_received_monotonic: float | None = None
    done_validated_monotonic: float | None = None
    request_started_monotonic: float | None = None
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

        trace_candidate = _DeepTraceSession(client, runtime_url, user_id, run_id)
        try:
            trace_candidate.start()
        except Exception:
            trace_error_code = "EVALUATION_EVIDENCE_START_FAILED"
            trace_candidate.started = True
            try:
                trace_candidate.delete()
            except Exception:
                pass
            trace = None
        else:
            trace = trace_candidate

        print(f"Session: {session_id}")
        print("Answer: ", end="", flush=True)
        events: list[str] = []
        answer_parts: list[str] = []
        telemetry: Mapping[str, object] | None = None
        done_payload: Mapping[str, object] | None = None
        try:
            query_headers = {"Accept": "text/event-stream"}
            if trace is not None:
                query_headers.update(
                    {
                        "X-Wizard-Diagnostic-Session-ID": trace.session_id,
                        "X-Wizard-Diagnostic-Operation-ID": trace_operation_id,
                    }
                )
            request_started_monotonic = time.monotonic()
            with client.stream(
                "POST",
                f"{runtime_url}/api/chat/query",
                headers=query_headers,
                json={
                    "user_id": user_id,
                    "session_id": session_id,
                    "question": question_text,
                },
            ) as response:
                response_request_id = response.headers.get("X-Request-ID")
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
                        done_received_monotonic = time.monotonic()
                    else:
                        raise RagCtlError(
                            f"RAG stream returned unexpected event {event_name!r}"
                        )
            print()
            answer, timings = validate_chat_result(
                events,
                answer_parts,
                telemetry,
                done_payload,
                verify_atlas_grounding=verify_atlas_grounding,
            )
            done_validated_monotonic = time.monotonic()
            if trace is not None:
                try:
                    _, trace_operation, _ = trace.wait_operation(trace_operation_id)
                except Exception:
                    trace_error_code = "EVALUATION_EVIDENCE_INVALID"
                finally:
                    try:
                        trace.delete()
                    except Exception:
                        trace_error_code = "EVALUATION_EVIDENCE_DELETE_FAILED"
        except BaseException:
            if trace is not None and trace.started:
                try:
                    trace.delete()
                except Exception:
                    pass
            raise

    summary = {name: timings[name] for name in CHAT_TIMING_KEYS}
    request_id = done_payload.get("request_id")
    conversation_id = done_payload.get("conversation_id")
    print("Telemetry (ms): " + json.dumps(summary, sort_keys=True))
    print("Done: " + json.dumps(dict(done_payload), sort_keys=True))
    print(f"SSE verified: {events.count('token')} token event(s) -> telemetry -> done")

    evaluation: EvaluationObservation
    try:
        if trace_error_code is not None or trace_operation is None:
            raise EvaluationBridgeError(
                trace_error_code or "evaluation evidence is unavailable"
            )
        if not isinstance(telemetry, Mapping) or not isinstance(done_payload, Mapping):
            raise EvaluationBridgeError("public response evidence is unavailable")
        if request_id != response_request_id or telemetry.get("request_id") != request_id:
            raise EvaluationBridgeError("public request correlation is invalid")
        evidence = parse_request_evidence(
            trace_operation,
            user_id=user_id,
            trace_session_id=trace.session_id,
            operation_id=trace_operation_id,
            chat_session_id=session_id,
            request_id=str(request_id),
        )
        evaluation_job = submit_local_evaluation(
            config=config,
            user_id=user_id,
            evidence=evidence,
            source="rag_ask",
            request_id=str(request_id),
            conversation_id=str(conversation_id),
            original_query=question_text,
            response=answer,
            telemetry=telemetry,
            captured_at=utc_timestamp(),
            directory=ASK_DIAGNOSTICS_PATH / run_id,
            stem="evaluation",
            exact_names=True,
            acceptance_activity_path=acceptance_activity_path,
        )
        try:
            evaluation = finish_evaluation_job(evaluation_job)
        except KeyboardInterrupt:
            cancel_evaluation_job(evaluation_job)
            raise
        except Exception:
            evaluation = EvaluationObservation(
                "failed",
                "EVALUATION_PROCESS_FAILED",
                0.0,
                None,
                None,
                None,
            )
    except Exception:
        evaluation = EvaluationObservation(
            "failed",
            trace_error_code or "EVALUATION_EVIDENCE_INVALID",
            0.0,
            None,
            None,
            None,
        )
    status_path = ASK_DIAGNOSTICS_PATH / run_id / "status.json"
    evaluation_artifact = evaluation.artifact()
    status_artifact_path: str | None = str(status_path)
    try:
        write_private_json(status_path, evaluation_artifact)
    except Exception:
        status_artifact_path = None
        evaluation_artifact = {
            **evaluation_artifact,
            "status": "failed",
            "error_code": "EVALUATION_STATUS_WRITE_FAILED",
        }
    print(
        "Evaluation: "
        f"status={evaluation_artifact['status']} "
        f"queue_wait_ms={evaluation_artifact['queue_wait_ms']} "
        f"execution_ms={evaluation_artifact['execution_ms']} "
        f"evaluation_ms={evaluation_artifact['evaluation_ms']} "
        f"record={evaluation_artifact.get('record_path')} "
        f"result={evaluation_artifact.get('result_path')} "
        f"status_artifact={status_artifact_path}",
        flush=True,
    )
    return {
        "status": "succeeded",
        "answer_complete": True,
        "question": question_text,
        "answer": answer,
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "session_id": session_id,
        "request_id": request_id,
        "conversation_id": conversation_id,
        "telemetry": dict(telemetry),
        "timings_ms": summary,
        "sse_events": list(events),
        "trace_session_id": None if trace is None else trace.session_id,
        "operation_id": trace_operation_id if trace_operation is not None else None,
        "evaluation_evidence": (
            None
            if trace_operation is None or trace is None
            else {"session_id": trace.session_id, "operation": dict(trace_operation)}
        ),
        "done_received_monotonic": done_received_monotonic,
        "done_validated_monotonic": done_validated_monotonic,
        "request_started_monotonic": request_started_monotonic,
        "evaluation": evaluation_artifact,
    }


def status(config: Mapping[str, str], runner: CommandRunner) -> None:
    print("Advanced RAG Application status (credentials redacted)")
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
            funnels_ready = (
                str(REST_FUNNEL_PORT) in forwards
                and str(GRPC_FUNNEL_PORT) in forwards
            )
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


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        prog="./rag",
        description=(
            "Start, query, inspect, or stop the private Advanced RAG Application."
        ),
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
    diagnose_parser = commands.add_parser(
        "diagnose", help="run a diagnostic against the real deployment"
    )
    diagnostics = diagnose_parser.add_subparsers(
        dest="diagnostic", required=True
    )
    wizard_parser = diagnostics.add_parser(
        "wizard", help="validate wizard diagnostic inputs and lifecycle"
    )
    wizard_parser.add_argument(
        "--fixtures",
        type=Path,
        required=True,
        help="fixture root containing knowledge/ and policy/",
    )
    wizard_parser.add_argument(
        "--reseed-corpus",
        action="store_true",
        help="replace the recorded corpus before retiring its old documents",
    )
    e2e_parser = diagnostics.add_parser(
        "e2e", help="run sequential queries against the retained Phase 1 corpus"
    )
    e2e_parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="Python file containing one literal QUERIES list",
    )
    e2e_parser.add_argument(
        "--start",
        type=_non_negative_int,
        default=0,
        help="zero-based query offset (default: 0)",
    )
    e2e_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="maximum number of queries (default: all remaining)",
    )
    e2e_parser.add_argument(
        "--continuous",
        action="store_true",
        help="reuse one chat session for every selected query",
    )
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
        elif args.command == "diagnose" and args.diagnostic == "wizard":
            diagnose_wizard(
                config,
                runner,
                args.fixtures,
                reseed_corpus=args.reseed_corpus,
            )
        elif args.command == "diagnose" and args.diagnostic == "e2e":
            diagnose_e2e(
                config,
                runner,
                args.queries,
                start=args.start,
                limit=args.limit,
                continuous=args.continuous,
            )
        else:  # pragma: no cover - argparse owns this invariant
            raise RagCtlError(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (
        RagCtlError,
        httpx.HTTPError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print("ERROR: " + redact(exc, config), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
