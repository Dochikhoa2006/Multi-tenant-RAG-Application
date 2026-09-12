"""Harmless OS-process checks for the local evaluation admission boundary."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from deployment import evaluation_bridge as bridge


_BLOCKED_WORKER = """
import os, pathlib, signal, sys, time
from deployment import evaluation_bridge_worker as worker
worker._STOP_GRACE_SECONDS = 0.1
def blocked(payload, lock_fd, deadline):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pathlib.Path(payload['ready']).write_text(str(os.getpid()))
    while True:
        time.sleep(0.01)
worker._execute_payload = blocked
raise SystemExit(worker.main())
"""


def _wait_ready(path: Path, process: subprocess.Popen) -> int:
    deadline = time.monotonic() + 5
    while not path.exists():
        assert process.poll() is None, "worker exited before admitted work"
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return int(path.read_text())


@pytest.mark.parametrize("cancel", [False, True])
def test_blocked_hydration_is_killed_reaped_and_releases_inherited_lock(
    tmp_path: Path, cancel: bool
) -> None:
    lock_path = tmp_path / "execution.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    ready = tmp_path / "ready"
    process = subprocess.Popen(
        [
            sys.executable, "-c", _BLOCKED_WORKER,
            "--lock-fd", str(descriptor),
            "--deadline-monotonic", str(time.monotonic() + 0.7),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        pass_fds=(descriptor,), start_new_session=True,
        env=bridge._sanitized_child_environment(evaluator=False),
    )
    # Closing without LOCK_UN simulates loss of the launcher's ownership.
    # Only the supervised process tree now retains the admission descriptor.
    os.close(descriptor)
    child_pid = None
    try:
        assert process.stdin is not None
        process.stdin.write(json.dumps({"ready": str(ready)}).encode())
        process.stdin.close()
        process.stdin = None
        child_pid = _wait_ready(ready, process)
        contender = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if cancel:
                process.terminate()
            process.communicate(timeout=5)
            assert process.returncode == (130 if cancel else 124)
            with pytest.raises(ProcessLookupError):
                os.kill(child_pid, 0)
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_admitted_deadline_also_covers_incomplete_private_input(tmp_path: Path) -> None:
    descriptor = os.open(tmp_path / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    process = subprocess.Popen(
        [sys.executable, "-m", "deployment.evaluation_bridge_worker",
         "--lock-fd", str(descriptor),
         "--deadline-monotonic", str(time.monotonic() + 0.2)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        pass_fds=(descriptor,), start_new_session=True,
    )
    try:
        # Keep the pipe open with no payload. The deadline must still fire.
        assert process.wait(timeout=5) == 124
    finally:
        os.close(descriptor)
        if process.stdin is not None:
            process.stdin.close()
            process.stdin = None
        process.communicate(timeout=5)


def test_supervisor_crash_keeps_slot_until_orphan_deadline(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock"
    ready = tmp_path / "ready"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    process = subprocess.Popen(
        [sys.executable, "-c", _BLOCKED_WORKER,
         "--lock-fd", str(descriptor),
         "--deadline-monotonic", str(time.monotonic() + 0.7)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        pass_fds=(descriptor,), start_new_session=True,
    )
    try:
        assert process.stdin is not None
        process.stdin.write(json.dumps({"ready": str(ready)}).encode())
        process.stdin.close()
        _wait_ready(ready, process)
        process.kill()
        process.wait(timeout=5)
        # The bridge must close its copy, not unlock the inherited flock.
        bridge._release_evaluation_lock(descriptor)
        descriptor = -1
        contender = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            deadline = time.monotonic() + 5
            while True:
                try:
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    assert time.monotonic() < deadline, "orphan retained slot past deadline"
                    time.sleep(0.02)
        finally:
            os.close(contender)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_success_file_cannot_hide_nonzero_evaluator_exit(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    payload = {
        "schema_version": "1.0", "status": "succeeded", "source": "rag_ask",
        "request_id": "request", "conversation_id": "conversation",
        "record_sha256": "a" * 64,
    }
    result.write_text(json.dumps(payload))
    launch = bridge.EvaluationLaunch(
        bridge.perf_counter(), subprocess.Popen(["/usr/bin/false"]),
        tmp_path / "record.json", result, "a" * 64, "rag_ask", "request", "conversation",
    )
    observation = bridge.finish_local_evaluation(launch)
    assert observation.status == "failed"
    assert observation.error_code == "EVALUATION_RESULT_INVALID"


def test_ordinary_ask_submission_accepts_omitted_corpus_restriction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from uuid import UUID, uuid4

    received = []

    def worker(payload, **kwargs):
        received.append(json.loads(payload))
        return {"status": "failed", "error_code": "EVALUATION_START_FAILED"}

    monkeypatch.setattr(bridge, "_run_supervised_worker", worker)
    evidence = bridge.RequestEvidence(
        "user", str(uuid4()), str(uuid4()), str(uuid4()), str(uuid4()), "rewrite",
        bridge.ContextIdentity("knowledge", (), (), "a" * 64, 0),
        bridge.ContextIdentity("policy", (), (), "b" * 64, 0),
    )
    # These are the actual ordinary-ask keyword arguments: no Phase-1 corpus
    # restriction. A required keyword formerly failed before job submission.
    job = bridge.submit_local_evaluation(
        config={}, user_id="user", evidence=evidence, source="rag_ask",
        request_id=evidence.request_id, conversation_id=str(uuid4()),
        original_query="question", response="answer", telemetry={},
        captured_at="2026-09-11T00:00:00Z", directory=tmp_path, stem="evaluation",
        exact_names=True, execution_lock_path=tmp_path / "lock",
    )
    assert bridge.finish_evaluation_job(job).error_code == "EVALUATION_START_FAILED"
    assert len(received) == 1
    assert received[0]["allowed_document_ids"] is None
    assert received[0]["evaluation_job_id"] == job.evaluation_job_id
    assert str(UUID(job.evaluation_job_id)) == job.evaluation_job_id


def test_terminal_timings_include_cleanup_and_freeze_before_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(bridge, "perf_counter", lambda: clock[0])

    def completed(**kwargs):
        clock[0] = 8.0  # admitted work plus cleanup completed at this boundary
        return bridge.EvaluationObservation(
            "succeeded", None, 7000.0, None, None, "a" * 64, 2000.0, 5000.0
        )

    monkeypatch.setattr(bridge, "_execute_evaluation_job", completed)
    observation = bridge._run_evaluation_job(submitted_at=0.0)
    clock[0] = 80.0  # later remote polling / join cannot inflate evaluation
    assert observation.duration_ms == 8000.0
    assert observation.queue_wait_ms == 2000.0
    assert observation.execution_ms == 6000.0
