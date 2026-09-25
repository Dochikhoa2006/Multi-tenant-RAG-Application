"""Supervise one admitted local job, including unresponsive hydration children."""

from __future__ import annotations

import argparse
from contextlib import suppress
import json
import math
import os
import select
import signal
import sys
from threading import Event, Thread
from time import monotonic


_MAX_INPUT_BYTES = 32 * 1024 * 1024
_MAX_OUTPUT_BYTES = 1_048_576
_STOP_GRACE_SECONDS = 5.1


def _execute_payload(payload: dict, lock_fd: int, deadline: float) -> dict:
    # Potentially blocking imports and hydration belong to the child, never
    # the process enforcing the admitted deadline.
    from deployment.evaluation_bridge import run_admitted_evaluation_payload

    return run_admitted_evaluation_payload(
        payload,
        execution_lock_fd=lock_fd,
        timeout_seconds=max(0.001, deadline - monotonic()),
    )

def _orphan_deadline(deadline: float) -> None:
    """Last resort if the supervisor itself disappears unexpectedly."""
    # Give the supervisor's TERM/KILL/reap sequence first ownership; this
    # fallback fires only if that supervisor is gone or unresponsive.
    Event().wait(max(0.0, deadline + _STOP_GRACE_SECONDS + 1.0 - monotonic()))
    os.killpg(os.getpgrp(), signal.SIGKILL)

def _execute_payload(payload: dict, lock_fd: int, deadline: float) -> dict:
    # Potentially blocking imports and hydration belong to the child, never
    # the process enforcing the admitted deadline.
    from deployment.evaluation_bridge import run_admitted_evaluation_payload

    return run_admitted_evaluation_payload(
        payload,
        execution_lock_fd=lock_fd,
        timeout_seconds=max(0.001, deadline - monotonic()),
    )

def _stop_child(child_pid: int) -> None:
    # Evaluator descendants inherit the child's group. Stop the group even
    # when its leader has already exited, then reap the direct child.
    with suppress(ProcessLookupError):
        os.killpg(child_pid, signal.SIGTERM)
    grace = monotonic() + _STOP_GRACE_SECONDS
    reaped = False
    while monotonic() < grace:
        if not reaped:
            try:
                reaped = os.waitpid(child_pid, os.WNOHANG)[0] == child_pid
            except ChildProcessError:
                reaped = True
        try:
            os.killpg(child_pid, 0)
        except ProcessLookupError:
            break
        select.select([], [], [], 0.02)
    with suppress(ProcessLookupError):
        os.killpg(child_pid, signal.SIGKILL)
    if not reaped:
        with suppress(ChildProcessError):
            os.waitpid(child_pid, 0)



def _supervise(payload: dict, lock_fd: int, deadline: float, cancelled: Event) -> bytes:
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        try:
            os.setpgid(0, 0)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            Thread(target=_orphan_deadline, args=(deadline,), daemon=True).start()
            result = _execute_payload(payload, lock_fd, deadline)
            encoded = json.dumps(result, allow_nan=False, separators=(",", ":")).encode()
            if len(encoded) > _MAX_OUTPUT_BYTES:
                os._exit(2)
            with os.fdopen(write_fd, "wb") as output:
                output.write(encoded)
            os._exit(0)
        except BaseException:
            os._exit(2)
    os.close(write_fd)
    with suppress(ProcessLookupError, PermissionError):
        os.setpgid(child_pid, child_pid)
    output = bytearray()
    try:
        while True:
            if cancelled.is_set():
                raise InterruptedError
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError
            ready, _, _ = select.select([read_fd], [], [], min(0.05, remaining))
            if not ready:
                continue
            part = os.read(read_fd, 65_536)
            if not part:
                _, status = os.waitpid(child_pid, 0)
                if os.waitstatus_to_exitcode(status) != 0:
                    raise RuntimeError("admitted child failed")
                return bytes(output)
            output.extend(part)
            if len(output) > _MAX_OUTPUT_BYTES:
                raise ValueError("admitted child output exceeded bound")
    finally:
        os.close(read_fd)
        _stop_child(child_pid)


def _read_payload(deadline: float, cancelled: Event) -> dict:
    contents = bytearray()
    descriptor = sys.stdin.fileno()
    while True:
        if cancelled.is_set():
            raise InterruptedError
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError
        ready, _, _ = select.select([descriptor], [], [], min(0.05, remaining))
        if not ready:
            continue
        part = os.read(descriptor, 65_536)
        if not part:
            break
        contents.extend(part)
        if len(contents) > _MAX_INPUT_BYTES:
            raise ValueError("job payload exceeded bound")
    payload = json.loads(contents.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("job payload must be an object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lock-fd", required=True, type=int)
    parser.add_argument("--deadline-monotonic", required=True, type=float)
    args = parser.parse_args(argv)
    if args.lock_fd < 0 or not math.isfinite(args.deadline_monotonic):
        return 2
    cancelled = Event()
    signal.signal(signal.SIGTERM, lambda *_: cancelled.set())
    signal.signal(signal.SIGINT, lambda *_: cancelled.set())
    try:
        os.fstat(args.lock_fd)
        payload = _read_payload(args.deadline_monotonic, cancelled)
        encoded = _supervise(payload, args.lock_fd, args.deadline_monotonic, cancelled)
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()
        return 0
    except TimeoutError:
        return 124
    except InterruptedError:
        return 130
    except Exception:
        return 2
    finally:
        with suppress(OSError):
            os.close(args.lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
