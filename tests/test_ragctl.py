from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess

import pytest

from deployment import ragctl


def _config() -> dict[str, str]:
    return {
        "WEAVIATE_URL": "https://mac.example.ts.net",
        "WEAVIATE_API_KEY": "weaviate-private-key",
        "WEAVIATE_CONNECTION_MODE": "custom",
        "WEAVIATE_GRPC_PORT": "8443",
        "WEAVIATE_GRPC_SECURE": "true",
        "MODAL_PROXY_TOKEN_ID": "wk-private-id",
        "MODAL_PROXY_TOKEN_SECRET": "ws-private-secret",
        "SGLANG_QUERY_REWRITE_API_KEY": "wk-private-id.ws-private-secret",
        "QWEN_SGLANG_API_KEY": "wk-private-id.ws-private-secret",
        "RAG_USER_ID": ragctl.DEFAULT_USER_ID,
    }


class FakeRunner:
    def __init__(self, config: dict[str, str] | None = None) -> None:
        self.config = config or _config()
        self.calls: list[list[str]] = []
        self.options: list[dict[str, object]] = []
        self.secret_payload: dict[str, str] | None = None
        self.secret_mode: int | None = None

    def run(self, args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = [os.fspath(item) for item in args]  # type: ignore[arg-type]
        self.calls.append(command)
        self.options.append(dict(kwargs))
        if "--from-json" in command:
            path = Path(command[command.index("--from-json") + 1])
            self.secret_payload = json.loads(path.read_text(encoding="utf-8"))
            self.secret_mode = stat.S_IMODE(path.stat().st_mode)
        stdout = ""
        if command[-4:-1] == ["container", "list", "--json"]:
            stdout = "[]"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_load_dotenv_does_not_evaluate_shell(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PLAIN=value\nQUOTED=\"value with spaces\"\n"
        f"UNTRUSTED=$(touch {marker})\n",
        encoding="utf-8",
    )

    values = ragctl.load_dotenv(env_file)

    assert values == {
        "PLAIN": "value",
        "QUOTED": "value with spaces",
        "UNTRUSTED": f"$(touch {marker})",
    }
    assert not marker.exists()


def test_runtime_secret_contains_exactly_nine_approved_values() -> None:
    secret = ragctl.build_runtime_secret(
        _config(), "https://granite.example/v1", "https://qwen.example/v1"
    )

    assert tuple(secret) == ragctl.RUNTIME_SECRET_KEYS
    assert len(secret) == 9
    assert secret["SGLANG_QUERY_REWRITE_API_KEY"] == "wk-private-id.ws-private-secret"
    assert secret["QWEN_SGLANG_API_KEY"] == "wk-private-id.ws-private-secret"
    assert "MODAL_PROXY_TOKEN_ID" not in secret
    assert "MODAL_PROXY_TOKEN_SECRET" not in secret


def test_user_collection_names_use_exact_physical_suffixes() -> None:
    names = ragctl.user_collection_names("usr_abc123")

    assert names == {
        "RagUser_OVZXEX3BMJRTCMRT_Conversations",
        "RagUser_OVZXEX3BMJRTCMRT_KnowledgeFacts",
        "RagUser_OVZXEX3BMJRTCMRT_Policy",
    }


def test_runtime_secret_uses_mode_0600_temporary_file_and_removes_it() -> None:
    runner = FakeRunner()
    secret = ragctl.build_runtime_secret(
        _config(), "https://granite.example/v1", "https://qwen.example/v1"
    )

    ragctl.deploy_runtime_secret(secret, runner)  # type: ignore[arg-type]

    assert runner.secret_payload == secret
    assert runner.secret_mode == 0o600
    secret_path = Path(runner.calls[0][runner.calls[0].index("--from-json") + 1])
    assert not secret_path.exists()
    assert not any(value in " ".join(runner.calls[0]) for value in secret.values())


def test_redaction_covers_all_credentials_and_combined_proxy_token() -> None:
    config = _config()
    text = " ".join(
        (
            config["WEAVIATE_API_KEY"],
            config["MODAL_PROXY_TOKEN_ID"],
            config["MODAL_PROXY_TOKEN_SECRET"],
            ragctl.proxy_bearer(config),
        )
    )

    rendered = ragctl.redact(text, config)

    assert rendered.count("[REDACTED]") >= 3
    assert not any(secret in rendered for secret in ragctl.secret_values(config))


def test_up_preserves_dependency_order(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    config = _config()
    runner = FakeRunner(config)
    monkeypatch.setattr(ragctl, "preflight", lambda *_: events.append("preflight") or "mac.example.ts.net")
    monkeypatch.setattr(ragctl, "ensure_certificate", lambda *_: events.append("certificate"))
    monkeypatch.setattr(ragctl, "compose_up", lambda *_: events.append("compose"))
    monkeypatch.setattr(ragctl, "configure_funnels", lambda *_: events.append("funnels"))
    monkeypatch.setattr(ragctl, "verify_weaviate", lambda *_: events.append("weaviate"))
    monkeypatch.setattr(
        ragctl,
        "deploy_granite",
        lambda *_: events.append("deploy-granite") or "https://granite/v1",
    )
    monkeypatch.setattr(ragctl, "validate_granite", lambda *_: events.append("granite"))
    monkeypatch.setattr(
        ragctl,
        "deploy_qwen",
        lambda *_: events.append("deploy-qwen") or "https://qwen/v1",
    )
    monkeypatch.setattr(ragctl, "validate_qwen", lambda *_: events.append("qwen"))
    monkeypatch.setattr(ragctl, "deploy_runtime_secret", lambda *_: events.append("secret"))
    monkeypatch.setattr(
        ragctl,
        "deploy_runtime",
        lambda *_: events.append("deploy-runtime") or "https://runtime",
    )
    monkeypatch.setattr(ragctl, "validate_runtime", lambda *_: events.append("runtime"))
    monkeypatch.setattr(ragctl, "write_state", lambda *_: events.append("state"))

    ragctl.up(config, runner)  # type: ignore[arg-type]

    assert events == [
        "preflight",
        "certificate",
        "compose",
        "funnels",
        "weaviate",
        "deploy-granite",
        "granite",
        "deploy-qwen",
        "qwen",
        "secret",
        "deploy-runtime",
        "runtime",
        "state",
    ]


def test_failed_up_invokes_fail_safe_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    runner = FakeRunner(config)
    events: list[str] = []
    monkeypatch.setattr(ragctl, "preflight", lambda *_: "mac.example.ts.net")
    monkeypatch.setattr(ragctl, "ensure_certificate", lambda *_: None)
    monkeypatch.setattr(
        ragctl, "compose_up", lambda *_: (_ for _ in ()).throw(ragctl.RagCtlError("boom"))
    )
    monkeypatch.setattr(
        ragctl,
        "down",
        lambda *_, **__: events.append("down"),
    )

    with pytest.raises(ragctl.RagCtlError, match="boom"):
        ragctl.up(config, runner)  # type: ignore[arg-type]

    assert events == ["down"]


def test_interrupted_up_invokes_fail_safe_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    runner = FakeRunner(config)
    events: list[str] = []
    monkeypatch.setattr(
        ragctl,
        "preflight",
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(ragctl, "down", lambda *_, **__: events.append("down"))

    with pytest.raises(KeyboardInterrupt):
        ragctl.up(config, runner)  # type: ignore[arg-type]

    assert events == ["down"]


@pytest.mark.parametrize("value", ["CPU", "A10G", "H200", ""])
def test_config_rejects_unapproved_operational_gpu(value: str) -> None:
    config = _config()
    config["MODAL_SGLANG_GPU"] = value

    with pytest.raises(ragctl.RagCtlError, match="MODAL_SGLANG_GPU"):
        ragctl.validate_config(config)


def test_deployments_use_configured_gpu_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config.update(
        {
            "MODAL_SGLANG_GPU": "H100",
            "QWEN_MODAL_SGLANG_GPU": "H100",
            "MODAL_RAG_GPU": "L40S",
        }
    )
    runner = FakeRunner(config)
    monkeypatch.setattr(ragctl, "resolve_server_url", lambda *_: "https://worker")

    ragctl.deploy_granite(config, runner)  # type: ignore[arg-type]
    ragctl.deploy_qwen(config, runner)  # type: ignore[arg-type]
    ragctl.deploy_runtime(config, runner)  # type: ignore[arg-type]

    assert runner.options[0]["overrides"]["MODAL_SGLANG_GPU"] == "H100"  # type: ignore[index]
    assert runner.options[1]["overrides"]["QWEN_MODAL_SGLANG_GPU"] == "H100"  # type: ignore[index]
    assert runner.options[2]["overrides"]["MODAL_RAG_GPU"] == "L40S"  # type: ignore[index]


def test_waiting_progress_is_redacted_and_actionable() -> None:
    message = ragctl._waiting_message("Granite", 503, "H100")

    assert message == "  Granite: waiting for readiness (HTTP 503; requested GPU H100)"
    assert "http" not in message.lower().replace("http 503", "")


def test_runtime_worker_count_uses_proc_and_avoids_self_match() -> None:
    class WorkerRunner(FakeRunner):
        def run(
            self, args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            command = [os.fspath(item) for item in args]  # type: ignore[arg-type]
            self.calls.append(command)
            self.options.append(dict(kwargs))
            return subprocess.CompletedProcess(command, 0, stdout="1\n", stderr="")

    runner = WorkerRunner()

    count = ragctl.runtime_worker_count("container-id", runner)  # type: ignore[arg-type]

    assert count == 1
    command = runner.calls[0]
    assert command[-2] == "-c"
    assert 'b"backend.runtime_app" + b":create_runtime_app"' in command[-1]
    assert 'b"backend.runtime_app:create_runtime_app"' not in command[-1]


def test_down_order_and_compose_never_removes_volume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _config()
    runner = FakeRunner(config)
    events: list[str] = []
    original_compose_down = ragctl.compose_down
    state = tmp_path / "rag-state.json"
    state.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ragctl, "STATE_PATH", state)
    monkeypatch.setattr(ragctl, "stop_modal_apps", lambda *_: events.append("modal"))
    monkeypatch.setattr(ragctl, "disable_funnels", lambda *_: events.append("funnels"))
    monkeypatch.setattr(ragctl, "compose_down", lambda *_: events.append("compose"))
    monkeypatch.setattr(ragctl, "target_containers", lambda *_: [])

    ragctl.down(config, runner)  # type: ignore[arg-type]

    assert events == ["modal", "funnels", "compose"]
    assert not state.exists()

    compose_runner = FakeRunner(config)
    original_compose_down(compose_runner)  # type: ignore[arg-type]
    command = compose_runner.calls[0]
    assert command[-1] == "down"
    assert "-v" not in command and "--volumes" not in command


def test_sse_parser_and_result_validation_accept_contract_order() -> None:
    timings = {name: 0.0 for name in ragctl.CHAT_TIMING_KEYS}
    timings["ttft"] = 1.5
    timings["total_request"] = 8.0
    lines = [
        "event: token",
        'data: {"text":"Atlas guide, manager approval, rollback checklist"}',
        "",
        "event: telemetry",
        "data: " + json.dumps({"timings_ms": timings}),
        "",
        "event: done",
        'data: {"conversation_id":"conversation"}',
        "",
    ]
    parsed = list(ragctl.iter_sse(lines))

    answer, timings = ragctl.validate_chat_result(
        [name for name, _ in parsed],
        [parsed[0][1]["text"]],  # type: ignore[index]
        parsed[1][1],  # type: ignore[arg-type]
        parsed[2][1],  # type: ignore[arg-type]
        verify_atlas_grounding=True,
    )

    assert answer.startswith("Atlas guide")
    assert timings["ttft"] == 1.5


@pytest.mark.parametrize(
    "events",
    [
        ["token", "done", "telemetry"],
        ["token", "telemetry", "done", "done"],
        ["token", "error", "telemetry", "done"],
    ],
)
def test_result_validation_rejects_bad_terminal_contract(events: list[str]) -> None:
    with pytest.raises(ragctl.RagCtlError):
        ragctl.validate_chat_result(
            events,
            ["answer"],
            {"timings_ms": {"ttft": 1.0}},
            {"conversation_id": "conversation"},
            verify_atlas_grounding=False,
        )


def test_launcher_keeps_gpu_overrides_and_source_defaults() -> None:
    granite = (ragctl.PROJECT_ROOT / "deployment" / "modal_sglang.py").read_text(
        encoding="utf-8"
    )
    qwen = (ragctl.PROJECT_ROOT / "deployment" / "modal_qwen_sglang.py").read_text(
        encoding="utf-8"
    )
    runtime = (ragctl.PROJECT_ROOT / "deployment" / "modal_runtime.py").read_text(
        encoding="utf-8"
    )

    assert ragctl.GPU_DEFAULTS == {
        "MODAL_SGLANG_GPU": "L40S",
        "QWEN_MODAL_SGLANG_GPU": "H100",
        "MODAL_RAG_GPU": "L40S",
    }
    assert 'GPU = os.getenv("MODAL_SGLANG_GPU", "L40S")' in granite
    assert 'GPU = os.getenv("QWEN_MODAL_SGLANG_GPU", "L40S")' in qwen
    assert 'GPU = os.getenv("MODAL_RAG_GPU", "L40S")' in runtime
