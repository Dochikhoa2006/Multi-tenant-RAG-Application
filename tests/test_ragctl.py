from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import pytest

from backend.api.telemetry import TIMING_KEYS
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
        "MODAL_PROFILE": "test-workspace",
        "MODAL_WORKSPACE": "test-workspace",
        "MODAL_ENVIRONMENT": "main",
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


def test_runtime_secret_contains_only_the_current_required_values() -> None:
    secret = ragctl.build_runtime_secret(
        _config(), "https://granite.example/v1", "https://qwen.example/v1"
    )

    assert tuple(secret) == ragctl.RUNTIME_SECRET_KEYS
    assert secret["SGLANG_QUERY_REWRITE_API_KEY"] == "wk-private-id.ws-private-secret"
    assert secret["QWEN_SGLANG_API_KEY"] == "wk-private-id.ws-private-secret"
    assert "MODAL_PROXY_TOKEN_ID" not in secret
    assert "MODAL_PROXY_TOKEN_SECRET" not in secret


def test_config_requires_complete_modal_target() -> None:
    config = _config()
    config["MODAL_WORKSPACE"] = ""

    with pytest.raises(ragctl.RagCtlError, match="MODAL_WORKSPACE"):
        ragctl.validate_config(config)


def test_config_rejects_modal_token_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_TOKEN_ID", "unexpected-token")

    with pytest.raises(ragctl.RagCtlError, match="token environment overrides"):
        ragctl.validate_config(_config())


def test_command_runner_verifies_modal_target_and_propagates_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config.update(
        {
            "MODAL_SGLANG_GPU": "H100",
            "MODAL_SGLANG_COMPUTE_REGION": "eu",
            "MODAL_SGLANG_ROUTING_REGION": "eu-west",
            "SUPPORTED_FILE_EXTENSIONS": ".note",
            "TEXT_FILE_ENCODING": "utf-16",
            "TEXT_FILE_JOIN_SEPARATOR": "--",
            "UPLOAD_MAX_FILE_BYTES": "1000",
            "UPLOAD_MAX_TOTAL_BYTES": "2000",
            "UPLOAD_READ_CHUNK_BYTES": "500",
            "WIZARD_DIAGNOSTICS_ENABLED": "true",
            "RAG_DIAGNOSTIC_USER_ID": "wizard_diagnostic",
        }
    )
    events: list[str] = []
    captured_environment: dict[str, str] = {}
    monkeypatch.setenv("WIZARD_DIAGNOSTICS_ENABLED", "true")
    monkeypatch.setenv("RAG_DIAGNOSTIC_USER_ID", "ambient_user")

    monkeypatch.setattr(
        ragctl,
        "verify_modal_target",
        lambda value: events.append(value["MODAL_WORKSPACE"]),
    )

    def fake_run(command: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_environment.update(kwargs["env"])  # type: ignore[arg-type]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(ragctl.subprocess, "run", fake_run)
    ragctl.CommandRunner(config).run([ragctl.MODAL, "app", "list"], quiet=True)

    assert events == ["test-workspace"]
    assert captured_environment["MODAL_PROFILE"] == "test-workspace"
    assert captured_environment["MODAL_ENVIRONMENT"] == "main"
    assert captured_environment["MODAL_SGLANG_GPU"] == "H100"
    assert captured_environment["MODAL_SGLANG_COMPUTE_REGION"] == "eu"
    assert captured_environment["MODAL_SGLANG_ROUTING_REGION"] == "eu-west"
    assert captured_environment["SUPPORTED_FILE_EXTENSIONS"] == ".note"
    assert captured_environment["TEXT_FILE_ENCODING"] == "utf-16"
    assert captured_environment["TEXT_FILE_JOIN_SEPARATOR"] == "--"
    assert captured_environment["UPLOAD_MAX_FILE_BYTES"] == "1000"
    assert captured_environment["UPLOAD_MAX_TOTAL_BYTES"] == "2000"
    assert captured_environment["UPLOAD_READ_CHUNK_BYTES"] == "500"
    assert "WIZARD_DIAGNOSTICS_ENABLED" not in captured_environment
    assert "RAG_DIAGNOSTIC_USER_ID" not in captured_environment


def test_non_modal_command_does_not_run_modal_target_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ragctl,
        "verify_modal_target",
        lambda *_: (_ for _ in ()).throw(AssertionError("unexpected probe")),
    )
    monkeypatch.setattr(
        ragctl.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        ),
    )

    ragctl.CommandRunner(_config()).run(["docker", "info"], quiet=True)


def test_modal_target_probe_failure_is_fail_closed_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    monkeypatch.setattr(
        ragctl.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="workspace mismatch " + config["WEAVIATE_API_KEY"],
        ),
    )

    with pytest.raises(ragctl.RagCtlError) as caught:
        ragctl.verify_modal_target(config)

    assert "workspace mismatch" in str(caught.value)
    assert config["WEAVIATE_API_KEY"] not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)


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


def test_only_runtime_deployment_receives_default_off_diagnostic_override(
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

    assert all("overrides" not in options for options in runner.options[:2])
    assert runner.options[2]["overrides"] == {
        "WIZARD_DIAGNOSTICS_ENABLED": "false"
    }
    assert ragctl.gpu_request(config, "MODAL_SGLANG_GPU") == "H100"
    assert ragctl.gpu_request(config, "QWEN_MODAL_SGLANG_GPU") == "H100"
    assert ragctl.gpu_request(config, "MODAL_RAG_GPU") == "L40S"


def test_diagnostic_runtime_override_is_scoped_to_modal_runtime_deploy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    runner = FakeRunner(config)
    monkeypatch.setattr(ragctl, "resolve_server_url", lambda *_: "https://worker")

    ragctl.deploy_runtime(
        config,
        runner,  # type: ignore[arg-type]
        wizard_diagnostic_user_id="wizard_diagnostic",
    )

    assert runner.options == [
        {
            "overrides": {
                "WIZARD_DIAGNOSTICS_ENABLED": "true",
                "RAG_DIAGNOSTIC_USER_ID": "wizard_diagnostic",
            }
        }
    ]


def _diagnostic_fixture_tree(root: Path) -> Path:
    fixtures = root / "fixtures"
    (fixtures / "knowledge").mkdir(parents=True)
    (fixtures / "policy").mkdir()
    (fixtures / "knowledge" / "fact.txt").write_text("fact", encoding="utf-8")
    (fixtures / "policy" / "rule.txt").write_text("rule", encoding="utf-8")
    return fixtures


def _prepare_diagnostic_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, str], FakeRunner, list[str], Path]:
    from deployment import wizard_diagnostic_api

    config = _config()
    config["RAG_DIAGNOSTIC_USER_ID"] = "wizard_diagnostic"
    runner = FakeRunner(config)
    events: list[str] = []
    output = tmp_path / "output"
    monkeypatch.setattr(ragctl, "WIZARD_DIAGNOSTICS_PATH", output)
    monkeypatch.setattr(
        ragctl, "WIZARD_CORPUS_STATE_PATH", output / "corpus-state.json"
    )
    monkeypatch.setattr(
        ragctl, "WIZARD_CORPUS_LOCK_PATH", output / "corpus-state.lock"
    )
    monkeypatch.setattr(ragctl, "read_runtime_url", lambda *_: "https://runtime")
    monkeypatch.setattr(ragctl, "_runtime_headers", lambda *_: {"auth": "hidden"})
    monkeypatch.setattr(
        wizard_diagnostic_api,
        "run_wizard_phase_1c",
        lambda *_, **__: events.append("diagnostic"),
    )
    return config, runner, events, _diagnostic_fixture_tree(tmp_path)


def test_diagnostic_lifecycle_is_pre_down_up_diagnostic_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, runner, events, fixtures = _prepare_diagnostic_test(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(ragctl, "down", lambda *_: events.append("down"))
    monkeypatch.setattr(ragctl, "up", lambda *_, **__: events.append("up"))

    ragctl.diagnose_wizard(config, runner, fixtures)

    assert events == ["down", "up", "diagnostic", "down"]


def test_diagnostic_pre_down_failure_stops_before_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, runner, events, fixtures = _prepare_diagnostic_test(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(
        ragctl,
        "down",
        lambda *_: (_ for _ in ()).throw(ragctl.RagCtlError("pre-down")),
    )
    monkeypatch.setattr(ragctl, "up", lambda *_, **__: events.append("up"))

    with pytest.raises(ragctl.RagCtlError, match="pre-down"):
        ragctl.diagnose_wizard(config, runner, fixtures)

    assert events == []


def test_diagnostic_ingestion_preflight_runs_before_any_lifecycle_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, runner, events, fixtures = _prepare_diagnostic_test(
        monkeypatch, tmp_path
    )
    config["UPLOAD_MAX_FILE_BYTES"] = "0"
    monkeypatch.setattr(ragctl, "down", lambda *_: events.append("down"))
    monkeypatch.setattr(ragctl, "up", lambda *_: events.append("up"))

    with pytest.raises(ValueError, match="greater than zero"):
        ragctl.diagnose_wizard(config, runner, fixtures)

    assert events == []


def test_diagnostic_failed_up_does_not_add_duplicate_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, runner, events, fixtures = _prepare_diagnostic_test(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(ragctl, "down", lambda *_: events.append("down"))

    def failed_up(*_: object, **__: object) -> None:
        events.extend(("up", "up-owned-cleanup"))
        raise ragctl.RagCtlError("startup")

    monkeypatch.setattr(ragctl, "up", failed_up)

    with pytest.raises(ragctl.RagCtlError, match="startup"):
        ragctl.diagnose_wizard(config, runner, fixtures)

    assert events == ["down", "up", "up-owned-cleanup"]


def test_diagnostic_failure_after_startup_always_runs_final_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from deployment import wizard_diagnostic_api

    config, runner, events, fixtures = _prepare_diagnostic_test(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(ragctl, "down", lambda *_: events.append("down"))
    monkeypatch.setattr(ragctl, "up", lambda *_, **__: events.append("up"))
    monkeypatch.setattr(
        wizard_diagnostic_api,
        "run_wizard_phase_1c",
        lambda *_, **__: (_ for _ in ()).throw(ValueError("diagnostic")),
    )

    with pytest.raises(ValueError, match="diagnostic"):
        ragctl.diagnose_wizard(config, runner, fixtures)

    assert events == ["down", "up", "down"]


def test_diagnostic_interruption_after_startup_always_runs_final_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from deployment import wizard_diagnostic_api

    config, runner, events, fixtures = _prepare_diagnostic_test(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(ragctl, "down", lambda *_: events.append("down"))
    monkeypatch.setattr(ragctl, "up", lambda *_, **__: events.append("up"))
    monkeypatch.setattr(
        wizard_diagnostic_api,
        "run_wizard_phase_1c",
        lambda *_, **__: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        ragctl.diagnose_wizard(config, runner, fixtures)

    assert events == ["down", "up", "down"]


def _prepare_e2e_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, str], FakeRunner, list[str], Path]:
    from deployment import e2e_diagnostic, e2e_diagnostic_api, wizard_diagnostic

    config = _config()
    config["RAG_DIAGNOSTIC_USER_ID"] = "wizard_diagnostic"
    runner = FakeRunner(config)
    events: list[str] = []
    queries = tmp_path / "queries.py"
    queries.write_text('QUERIES = ["one", "two"]\n', encoding="utf-8")
    output = tmp_path / "e2e-output"
    request_path = output / "run" / "requests.jsonl"
    request_path.parent.mkdir(parents=True)
    request_path.touch()
    summary_path = request_path.with_name("summary.json")

    monkeypatch.setattr(ragctl, "E2E_DIAGNOSTICS_PATH", output)
    monkeypatch.setattr(
        ragctl, "WIZARD_CORPUS_STATE_PATH", tmp_path / "corpus-state.json"
    )
    monkeypatch.setattr(
        ragctl, "WIZARD_CORPUS_LOCK_PATH", tmp_path / "corpus-state.lock"
    )
    monkeypatch.setattr(ragctl, "read_runtime_url", lambda *_: "https://runtime")
    monkeypatch.setattr(ragctl, "_runtime_headers", lambda *_: {"auth": "hidden"})
    corpus_state = object()
    active = object()
    monkeypatch.setattr(
        wizard_diagnostic, "load_corpus_state", lambda *_: corpus_state
    )
    monkeypatch.setattr(
        e2e_diagnostic,
        "validate_reusable_corpus_state",
        lambda *_: active,
    )
    monkeypatch.setattr(
        e2e_diagnostic,
        "create_e2e_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            run_id="run",
            directory=request_path.parent,
            requests_path=request_path,
            summary_path=summary_path,
        ),
    )
    monkeypatch.setattr(e2e_diagnostic, "update_e2e_summary", lambda *_, **__: None)

    def run_phase(*args: object, **kwargs: object) -> None:
        progress = args[-1]
        progress["physical_corpus_status"] = "succeeded"  # type: ignore[index]
        progress["trace_status"] = "succeeded"  # type: ignore[index]
        progress["trace_deleted"] = True  # type: ignore[index]
        events.append("diagnostic")

    monkeypatch.setattr(e2e_diagnostic_api, "run_e2e_phase_2d", run_phase)
    return config, runner, events, queries


def test_e2e_parser_defaults_and_selection_flags() -> None:
    cli = ragctl.parser()
    defaults = cli.parse_args(
        ["diagnose", "e2e", "--queries", "diagnostics/queries.py"]
    )
    selected = cli.parse_args(
        [
            "diagnose",
            "e2e",
            "--queries",
            "diagnostics/queries.py",
            "--start",
            "3",
            "--limit",
            "5",
            "--continuous",
        ]
    )

    assert defaults.diagnostic == "e2e"
    assert defaults.queries == Path("diagnostics/queries.py")
    assert defaults.start == 0
    assert defaults.limit is None
    assert defaults.continuous is False
    assert selected.start == 3
    assert selected.limit == 5
    assert selected.continuous is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["--start", "-1"],
        ["--start", "no"],
        ["--limit", "0"],
        ["--limit", "-1"],
        ["--limit", "no"],
    ],
)
def test_e2e_parser_rejects_invalid_slice_values(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        ragctl.parser().parse_args(
            [
                "diagnose",
                "e2e",
                "--queries",
                "diagnostics/queries.py",
                *arguments,
            ]
        )


def test_e2e_lifecycle_is_down_up_diagnostic_down_and_traced_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, runner, events, queries = _prepare_e2e_test(monkeypatch, tmp_path)
    monkeypatch.setattr(ragctl, "down", lambda *_: events.append("down"))

    def diagnostic_up(*_: object, **kwargs: object) -> None:
        assert kwargs == {"wizard_diagnostic_user_id": "wizard_diagnostic"}
        events.append("up")

    monkeypatch.setattr(ragctl, "up", diagnostic_up)

    ragctl.diagnose_e2e(config, runner, queries, start=1, limit=1)

    assert events == ["down", "up", "diagnostic", "down"]


def test_e2e_preflight_failure_happens_before_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, runner, events, _ = _prepare_e2e_test(monkeypatch, tmp_path)
    monkeypatch.setattr(ragctl, "down", lambda *_: events.append("down"))
    monkeypatch.setattr(ragctl, "up", lambda *_: events.append("up"))

    with pytest.raises(ValueError, match="does not exist"):
        ragctl.diagnose_e2e(config, runner, tmp_path / "missing.py")

    assert events == []


def test_e2e_trace_capacity_is_checked_before_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, runner, events, queries = _prepare_e2e_test(monkeypatch, tmp_path)
    queries.write_text(
        "QUERIES = " + repr([f"query-{index}" for index in range(257)]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ragctl, "down", lambda *_: events.append("down"))
    monkeypatch.setattr(ragctl, "up", lambda *_, **__: events.append("up"))

    with pytest.raises(ragctl.RagCtlError, match="trace capacity"):
        ragctl.diagnose_e2e(config, runner, queries)

    assert events == []


def test_e2e_failed_up_does_not_add_duplicate_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, runner, events, queries = _prepare_e2e_test(monkeypatch, tmp_path)
    monkeypatch.setattr(ragctl, "down", lambda *_: events.append("down"))

    def failed_up(*_: object, **__: object) -> None:
        events.extend(("up", "up-owned-cleanup"))
        raise ragctl.RagCtlError("startup")

    monkeypatch.setattr(ragctl, "up", failed_up)

    with pytest.raises(ragctl.RagCtlError, match="startup"):
        ragctl.diagnose_e2e(config, runner, queries)

    assert events == ["down", "up", "up-owned-cleanup"]


def test_e2e_phase_failure_and_interruption_always_run_final_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from deployment import e2e_diagnostic_api

    config, runner, events, queries = _prepare_e2e_test(monkeypatch, tmp_path)
    monkeypatch.setattr(ragctl, "down", lambda *_: events.append("down"))
    monkeypatch.setattr(ragctl, "up", lambda *_, **__: events.append("up"))
    monkeypatch.setattr(
        e2e_diagnostic_api,
        "run_e2e_phase_2d",
        lambda *_, **__: (_ for _ in ()).throw(ValueError("requests")),
    )

    with pytest.raises(ValueError, match="requests"):
        ragctl.diagnose_e2e(config, runner, queries)

    assert events == ["down", "up", "down"]

    events.clear()
    monkeypatch.setattr(
        e2e_diagnostic_api,
        "run_e2e_phase_2d",
        lambda *_, **__: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        ragctl.diagnose_e2e(config, runner, queries)

    assert events == ["down", "up", "down"]


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


def test_launcher_uses_the_public_telemetry_key_owner() -> None:
    assert ragctl.CHAT_TIMING_KEYS is TIMING_KEYS


def test_launcher_help_uses_the_short_product_name() -> None:
    assert "Advanced RAG Application" in ragctl.parser().description


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


def test_launcher_keeps_one_gpu_default_and_independent_source_defaults() -> None:
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
        "QWEN_MODAL_SGLANG_GPU": "L40S",
        "MODAL_RAG_GPU": "L40S",
    }
    assert set(ragctl.GPU_DEFAULTS.values()) == {ragctl.DEFAULT_OPERATIONAL_GPU}
    assert 'GPU = os.getenv("MODAL_SGLANG_GPU", "L40S")' in granite
    assert 'GPU = os.getenv("QWEN_MODAL_SGLANG_GPU", "L40S")' in qwen
    assert 'GPU = os.getenv("MODAL_RAG_GPU", "L40S")' in runtime


def test_launcher_secure_tunnel_topology_has_single_in_process_owners() -> None:
    assert ragctl.LOCAL_WEAVIATE_REST_URL == "http://127.0.0.1:8080"
    assert ragctl.LOCAL_WEAVIATE_GRPC_TLS_TARGET == "127.0.0.1:5443"
    assert ragctl.REST_FUNNEL_PORT == 443
    assert ragctl.GRPC_FUNNEL_PORT == 8443
