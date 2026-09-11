"""Supervise one admitted local evidence hydration and Ragas execution."""

from __future__ import annotations

import argparse
from contextlib import suppress
import json
import math
import os
import signal
import sys
from threading import Event, Thread
from deployment.evaluation_bridge import run_admitted_evaluation_payload


_MAX_INPUT_BYTES = 32 * 1024 * 1024


def _expire(cancelled: Event, timeout_seconds: float) -> None:
    if cancelled.wait(timeout_seconds):
        return
    if os.getpgrp() == os.getpid():
        with suppress(ProcessLookupError):
            os.killpg(os.getpgrp(), signal.SIGTERM)
    os._exit(124)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lock-fd", required=True, type=int)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.lock_fd < 0
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
    ):
        return 2
    try:
        os.fstat(args.lock_fd)
    except OSError:
        return 2
    payload_bytes = sys.stdin.buffer.read(_MAX_INPUT_BYTES + 1)
    if len(payload_bytes) > _MAX_INPUT_BYTES:
        return 2
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return 2
    if not isinstance(payload, dict):
        return 2

    cancelled = Event()
    watchdog = Thread(
        target=_expire,
        args=(cancelled, args.timeout_seconds),
        daemon=True,
        name="rag-evaluation-deadline",
    )
    watchdog.start()
    try:
        result = run_admitted_evaluation_payload(
            payload,
            execution_lock_fd=args.lock_fd,
            timeout_seconds=args.timeout_seconds,
        )
        encoded = json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()
        return 0
    finally:
        cancelled.set()
        with suppress(OSError):
            os.close(args.lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
