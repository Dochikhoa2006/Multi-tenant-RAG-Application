"""Command-line entry point for the isolated local evaluator."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Sequence
from uuid import uuid4

from .evaluator import evaluate_record, write_result_atomic
from .metrics import EvaluationSettings, compatibility_smoke, safe_exception_type
from .models import EvaluationRecord, RecordValidationError


EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIRECTORY = EVALUATION_ROOT / "results"


def _settings_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ollama-url", help="Loopback Ollama origin or /v1 URL")
    parser.add_argument("--judge-model", help="Installed local Ollama judge model")
    parser.add_argument(
        "--embedding-model",
        type=Path,
        help="Local SentenceTransformer directory",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-evaluate",
        description="Evaluate one immutable RAG record using local models only.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser(
        "smoke",
        help="Validate Ragas APIs and local resources without a judge completion.",
    )
    _settings_arguments(smoke)

    run = commands.add_parser("run", help="Evaluate one JSON record.")
    run.add_argument("record", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--noise-sensitivity", action="store_true")
    _settings_arguments(run)
    return parser


def _settings(args: argparse.Namespace) -> EvaluationSettings:
    return EvaluationSettings.from_environment(
        ollama_url=args.ollama_url,
        judge_model=args.judge_model,
        embedding_model_path=args.embedding_model,
    )


def _load_record(path: Path) -> EvaluationRecord:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise RecordValidationError("record path must be a regular non-symlink file")
    try:
        payload = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RecordValidationError("record file is not readable UTF-8") from exc
    return EvaluationRecord.from_json(payload)


def _default_output(record: EvaluationRecord) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    suffix = uuid4().hex[:12]
    return DEFAULT_RESULTS_DIRECTORY / f"{record.request_id}-{timestamp}-{suffix}.json"


async def _run_smoke(args: argparse.Namespace) -> int:
    try:
        result = await compatibility_smoke(_settings(args))
    except Exception as exc:
        print(f"smoke_status=failed error_type={safe_exception_type(exc)}")
        return 2
    print(
        " ".join(
            (
                "smoke_status=succeeded",
                f"ragas_version={result['ragas_version']}",
                f"judge_model={result['judge_model']}",
                f"embedding_dimension={result['embedding_dimension']}",
                "judge_completion_issued=false",
            )
        )
    )
    return 0


async def _run_evaluation(args: argparse.Namespace) -> int:
    try:
        record = _load_record(args.record)
        settings = _settings(args)
    except Exception as exc:
        print(f"evaluation_status=invalid error_type={safe_exception_type(exc)}")
        return 2

    result = await evaluate_record(
        record,
        settings=settings,
        include_noise_sensitivity=args.noise_sensitivity,
    )
    output = args.output or _default_output(record)
    try:
        written = write_result_atomic(result, output)
    except Exception as exc:
        print(f"evaluation_status=output_failed error_type={safe_exception_type(exc)}")
        return 2

    print(f"evaluation_status={result.status} result={written}")
    if result.setup_error_code is not None:
        return 2
    return 0 if result.status == "succeeded" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "smoke":
            return asyncio.run(_run_smoke(args))
        if args.command == "run":
            return asyncio.run(_run_evaluation(args))
    except KeyboardInterrupt:
        print("evaluation_status=interrupted")
        return 130
    raise AssertionError("unreachable command")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

