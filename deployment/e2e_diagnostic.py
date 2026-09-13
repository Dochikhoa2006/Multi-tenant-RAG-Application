"""Local preflight and artifact contracts for the Phase 2 E2E diagnostic."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from deployment.wizard_diagnostic import (
    CORPUS_SCHEMA_VERSION,
    CorpusGeneration,
    CorpusState,
    WIZARD_FIXTURE_COLLECTIONS,
)


E2E_SCHEMA_VERSION = "1.4"
E2E_PHASE = "2D"


class E2EDiagnosticError(ValueError):
    """A safe, user-facing Phase 2 diagnostic failure."""


@dataclass(frozen=True)
class SelectedQuery:
    source_index: int
    question: str
    query_id: str | None = None
    reference: str | None = None
    reference_context_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuerySelection:
    source_path: Path
    source_sha256: str
    total_queries: int
    start: int
    limit: int | None
    selected: tuple[SelectedQuery, ...]


@dataclass(frozen=True)
class E2ERun:
    run_id: str
    directory: Path
    requests_path: Path
    summary_path: Path

    @property
    def dataset_report_path(self) -> Path:
        return self.directory / "dataset-report.json"


def utc_timestamp(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    return current.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _resolve_query_path(path: Path, project_root: Path) -> Path:
    candidate = path if path.is_absolute() else project_root / path
    if candidate.is_symlink():
        raise E2EDiagnosticError(f"Query file must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise E2EDiagnosticError(f"Query file does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise E2EDiagnosticError(f"Query path is not a file: {resolved}")
    if resolved.suffix.lower() not in {".py", ".jsonl"}:
        raise E2EDiagnosticError("Query file must use the .py extension or .jsonl extension")
    return resolved


def _literal_queries(source: str, path: Path) -> list[str]:
    try:
        tree = ast.parse(source, filename=os.fspath(path))
    except SyntaxError as exc:
        raise E2EDiagnosticError(f"Query file is not valid Python: {path}") from exc

    assignment: ast.expr | None = None
    for index, statement in enumerate(tree.body):
        if (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign):
            if (
                len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "QUERIES"
            ):
                value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "QUERIES"
            and statement.value is not None
        ):
            value = statement.value
        if value is None:
            raise E2EDiagnosticError(
                "Query file may contain only a module docstring and one "
                "QUERIES assignment"
            )
        if assignment is not None:
            raise E2EDiagnosticError("Query file must assign QUERIES exactly once")
        assignment = value

    if assignment is None:
        raise E2EDiagnosticError("Query file must define QUERIES")
    try:
        value = ast.literal_eval(assignment)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise E2EDiagnosticError("QUERIES must be a literal list of strings") from exc
    if not isinstance(value, list) or not value:
        raise E2EDiagnosticError("QUERIES must be a non-empty list")
    if any(not isinstance(item, str) for item in value):
        raise E2EDiagnosticError("Every QUERIES item must be a string")
    if any(not item.strip() for item in value):
        raise E2EDiagnosticError("QUERIES must not contain blank strings")
    return value


def _jsonl_queries(source: str, path: Path) -> list[tuple[str, str, str | None, tuple[str, ...]]]:
    def unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
        if len({key for key, _ in pairs}) != len(pairs):
            raise E2EDiagnosticError("Query dataset has duplicate JSON fields")
        return dict(pairs)

    records: list[tuple[str, str, str | None, tuple[str, ...]]] = []
    identifiers: set[str] = set()
    allowed = {"query_id", "question", "reference", "reference_context_ids"}
    for line_number, line in enumerate(source.splitlines(), start=1):
        if not line.strip():
            raise E2EDiagnosticError(f"Query dataset contains a blank line at {line_number}")
        try:
            value = json.loads(line, object_pairs_hook=unique_fields)
        except json.JSONDecodeError as exc:
            raise E2EDiagnosticError(f"Query dataset line {line_number} is invalid JSON") from exc
        if not isinstance(value, dict) or not set(value) <= allowed:
            raise E2EDiagnosticError(f"Query dataset line {line_number} has unknown fields")
        query_id = value.get("query_id")
        question = value.get("question")
        reference = value.get("reference")
        reference_ids = value.get("reference_context_ids", [])
        if (
            not isinstance(query_id, str)
            or not query_id.strip()
            or query_id != query_id.strip()
            or not isinstance(question, str)
            or not question.strip()
        ):
            raise E2EDiagnosticError(f"Query dataset line {line_number} has invalid identity or question")
        if query_id in identifiers:
            raise E2EDiagnosticError(f"Query dataset duplicates a query ID at line {line_number}")
        if reference is not None and (
            not isinstance(reference, str)
            or not reference.strip()
        ):
            raise E2EDiagnosticError(f"Query dataset line {line_number} has an invalid reference")
        if not isinstance(reference_ids, list):
            raise E2EDiagnosticError(f"Query dataset line {line_number} has invalid reference_context_ids")
        try:
            canonical_ids = tuple(str(UUID(item)) for item in reference_ids if isinstance(item, str))
        except ValueError as exc:
            raise E2EDiagnosticError(f"Query dataset line {line_number} has invalid reference_context_ids") from exc
        if len(canonical_ids) != len(reference_ids) or list(canonical_ids) != reference_ids or len(set(canonical_ids)) != len(canonical_ids):
            raise E2EDiagnosticError(f"Query dataset line {line_number} has invalid reference_context_ids")
        identifiers.add(query_id)
        records.append((query_id, question, reference, canonical_ids))
    if not records:
        raise E2EDiagnosticError("Query dataset must contain at least one record")
    return records


def load_query_selection(
    path: Path,
    project_root: Path,
    *,
    start: int = 0,
    limit: int | None = None,
) -> QuerySelection:
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise E2EDiagnosticError("--start must be a non-negative integer")
    if (
        limit is not None
        and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0)
    ):
        raise E2EDiagnosticError("--limit must be a positive integer")
    resolved = _resolve_query_path(path, project_root)
    try:
        source_bytes = resolved.read_bytes()
        source = source_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise E2EDiagnosticError(
            f"Query file is not readable UTF-8: {resolved}"
        ) from exc
    dataset_records = (
        _jsonl_queries(source, resolved) if resolved.suffix.lower() == ".jsonl" else None
    )
    queries = (
        [item[1] for item in dataset_records]
        if dataset_records is not None
        else _literal_queries(source, resolved)
    )
    if start >= len(queries):
        raise E2EDiagnosticError("--start is outside the QUERIES list")
    stop = len(queries) if limit is None else min(len(queries), start + limit)
    selected = tuple(
        SelectedQuery(
            index,
            queries[index],
            (
                dataset_records[index][0]
                if dataset_records is not None
                else f"legacy-{index:06d}-{hashlib.sha256((str(index) + ':' + queries[index]).encode('utf-8')).hexdigest()[:16]}"
            ),
            None if dataset_records is None else dataset_records[index][2],
            () if dataset_records is None else dataset_records[index][3],
        )
        for index in range(start, stop)
    )
    if not selected:  # Defensive: validated bounds should make this unreachable.
        raise E2EDiagnosticError("Query selection must not be empty")
    return QuerySelection(
        source_path=resolved,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        total_queries=len(queries),
        start=start,
        limit=limit,
        selected=selected,
    )


def validate_reusable_corpus_state(
    state: CorpusState | None,
    diagnostic_user_id: str,
) -> CorpusGeneration:
    if state is None or state.active is None:
        raise E2EDiagnosticError(
            "Phase 1 corpus state has no active generation; "
            "run the Wizard diagnostic first"
        )
    if state.diagnostic_user_id != diagnostic_user_id:
        raise E2EDiagnosticError(
            "Phase 1 corpus belongs to a different diagnostic user"
        )
    if state.pending_replacement is not None:
        raise E2EDiagnosticError("Phase 1 corpus has an interrupted replacement")
    if state.pending_cleanup:
        raise E2EDiagnosticError("Phase 1 corpus has pending document cleanup")
    active = state.active
    if active.verified_at is None or active.completed_at is None:
        raise E2EDiagnosticError("Phase 1 active corpus is not verified and complete")
    if {item.collection for item in active.documents} != set(
        WIZARD_FIXTURE_COLLECTIONS
    ) or any(item.status != "saved" for item in active.documents):
        raise E2EDiagnosticError("Phase 1 active corpus is incomplete")
    return active


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _corpus_summary(active: CorpusGeneration) -> dict[str, object]:
    return {
        "state_schema_version": CORPUS_SCHEMA_VERSION,
        "generation_id": active.generation_id,
        "fixture_digest": active.fixture_digest,
        "documents": {
            document.collection: {
                "wizard_id": document.wizard_id,
                "chunk_count": len(document.chunk_ids),
                "storage_fingerprint": document.storage_fingerprint,
            }
            for document in active.documents
        },
    }


def _finite_samples(values: list[object], population_count: int) -> dict[str, object]:
    if population_count < len(values):
        raise E2EDiagnosticError("Statistics population is smaller than its observations")
    samples = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise E2EDiagnosticError("Statistics contain an invalid numeric observation")
        samples.append(float(value))
    samples.sort()
    count = len(samples)
    result = {"population_count": population_count, "sample_count": count,
              "null_count": population_count - count}
    names = ("min", "max", "sum", "mean", "p50", "p95", "p99",
             "population_variance", "population_stddev")
    if not count:
        return {**result, **dict.fromkeys(names)}
    total = math.fsum(samples)
    mean = total / count
    variance = math.fsum((value - mean) ** 2 for value in samples) / count
    values = (samples[0], samples[-1], total, mean,
              *(samples[math.ceil(p * count) - 1] for p in (0.5, 0.95, 0.99)),
              variance, math.sqrt(variance))
    return {**result, **{name: round(value, 6 if name == "population_variance" else 3) or 0.0
                        for name, value in zip(names, values, strict=True)}}


def _safe_result(path_value: object, directory: Path) -> Mapping[str, object] | None:
    if not isinstance(path_value, str):
        return None
    try:
        candidate = Path(path_value)
        if candidate.is_symlink() or not candidate.is_file():
            return None
        candidate.resolve(strict=True).relative_to(directory.resolve())
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def dataset_identities(config: Mapping[str, str], state: CorpusState) -> dict[str, object]:
    """Inspect the existing deployment's pure environment construction offline."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    preflight = subprocess.run(
        [sys.executable, str(root / "evaluation/non_regression_gate.py"), "preflight"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if preflight.returncode:
        raise E2EDiagnosticError("Protected-contract preflight failed")
    # Execute only the environment-declaration prefix, before any Modal image,
    # volume, secret or app construction. Model config itself is pure stdlib.
    script = """
import ast, dataclasses, json, os, runpy
from pathlib import Path
tree = ast.parse(Path('deployment/modal_runtime.py').read_text())
prefix = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'image' for t in node.targets):
        break
    if isinstance(node, ast.Import) and any(a.name == 'modal' for a in node.names):
        continue
    prefix.append(node)
ns = {}
exec(compile(ast.Module(body=prefix, type_ignores=[]), '<runtime-environment>', 'exec'), ns)
os.environ.clear()
os.environ.update(ns['runtime_environment'])
cfg = runpy.run_path('backend/model_config.py')
names = ('CONVERSATION_SEARCH', 'KNOWLEDGE_SEARCH', 'POLICY_SEARCH', 'HYBRID_SEARCH',
         'TOKEN_BUDGETS', 'PRIMARY_GENERATOR', 'QUERY_REWRITER', 'SESSION_TITLE_GENERATOR',
         'EMBEDDING_MODEL', 'EMBEDDING_VECTOR_PROFILE', 'RERANKER_MODEL', 'RERANKER_MODEL_REVISION',
         'LATEON_MODEL', 'LATEON_MODEL_REVISION', 'LATEON_EMBEDDING_DIMENSION',
         'GTE_EMBEDDING_DIMENSION', 'LATE_INTERACTION_VECTOR_NAME', 'MMR_DIVERSITY_VECTOR_NAME')
out = {n: dataclasses.asdict(cfg[n]) if dataclasses.is_dataclass(cfg[n]) else cfg[n] for n in names}
out['tokenizer_encoding'] = cfg['TEXT_PROCESSING'].tokenizer_encoding
out['granite_input_limit'] = cfg['GRANITE_QUERY_REWRITE'].max_input_tokens
out['qwen'] = {k: v for k,v in dataclasses.asdict(cfg['QWEN_SGLANG']).items() if k not in ('base_url','api_key')}
print(json.dumps(out, sort_keys=True, allow_nan=False))
"""
    environment = {"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"}
    # Only ingestion policy affects this prefix; other deployment settings are
    # operational or explicit fixed image values. No credentials enter it.
    from deployment.wizard_diagnostic import FixtureRules
    rules = FixtureRules.from_config(config)
    environment.update({
        "SUPPORTED_FILE_EXTENSIONS": ",".join(rules.supported_extensions),
        "TEXT_FILE_ENCODING": rules.text_encoding,
        "TEXT_FILE_JOIN_SEPARATOR": rules.text_join_separator,
    })
    inspected = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=environment,
        capture_output=True, text=True, check=False,
    )
    if inspected.returncode:
        raise E2EDiagnosticError("Effective runtime configuration inspection failed")
    effective = json.loads(inspected.stdout)
    manifest_path = root / "evaluation/protected_contracts.json"
    manifest = json.loads(manifest_path.read_text())
    files = set(manifest["files"]) | {
        "deployment/wizard_diagnostic.py", "deployment/wizard_diagnostic_api.py",
        "evaluation/uv.lock", "evaluation/src/rag_evaluation/metrics.py",
        "evaluation/src/rag_evaluation/models.py", "evaluation/src/rag_evaluation/evaluator.py",
    }
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    return {
        "source_commit": commit,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(files)},
        "protected_contract_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "configuration_sha256": digest(dict(config)),
        "effective_configuration": effective,
        "effective_configuration_sha256": digest(effective),
        "configuration_basis": "current Modal image environment and model_config defaults; .env values not forwarded by deployment are not effective overrides",
        "corpus_sha256": digest(state.payload()),
        "corpus_owner": state.diagnostic_user_id,
        "evaluator_configuration": {
            "provider": "ollama", "transport": "loopback HTTP", "judge_model": "qwen3.5:4b",
            "reasoning_effort": "none", "max_tokens": 4096, "temperature": 0.0,
            "embedding_dimension": 384, "embedding_device": "cpu", "offline_embeddings": True,
        },
    }


def write_dataset_report(
    run: E2ERun, recorder: RequestRecorder, summary: Mapping[str, object],
) -> None:
    selection, active, identities = recorder.dataset
    rows = [json.loads(line) for line in run.requests_path.read_text().splitlines()]
    by_index = {}
    for row in rows:
        index = row.get("source_index")
        if index in by_index or index not in {q.source_index for q in selection.selected}:
            raise E2EDiagnosticError("Dataset query accounting is duplicated or unexpected")
        by_index[index] = row
    for query in selection.selected:
        if query.source_index not in by_index:
            recorder.record_not_attempted(query, "INTERRUPTED" if summary["status"] == "interrupted" else "SYSTEMIC_BATCH_ABORT")
    rows = [json.loads(line) for line in run.requests_path.read_text().splitlines()]
    rows.sort(key=lambda row: row["source_index"])
    if [r["source_index"] for r in rows] != [q.source_index for q in selection.selected]:
        raise E2EDiagnosticError("Dataset query accounting is incomplete")
    output = []
    failures: dict[str, int] = {}
    def failure(scope: str, code: object) -> None:
        if isinstance(code, str):
            key = scope + ":" + code
            failures[key] = failures.get(key, 0) + 1
    for row, query in zip(rows, selection.selected, strict=True):
        if row.get("query_id") != query.query_id:
            raise E2EDiagnosticError("Dataset query identity mismatch")
        evaluation = dict(row.get("evaluation") or {})
        result = _safe_result(evaluation.get("result_path"), run.directory) or {}
        required = {"faithfulness", "response_relevancy", "context_utilization"}
        if query.reference is not None:
            required |= {"context_recall", "context_precision_with_reference", "factual_correctness"}
        metrics = []
        metric_items = result.get("metrics", [])
        if not isinstance(metric_items, list) or any(not isinstance(item, Mapping) or not isinstance(item.get("name"), str) for item in metric_items):
            metric_items = []
            evaluation.update(status="failed", error_code="DATASET_RESULT_INVALID")
        for item in metric_items:
            outcome = {key: item.get(key) for key in ("name", "status", "score", "duration_ms", "error_code")}
            score = outcome["score"]
            if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))
                                      or not math.isfinite(score) or not 0 <= score <= 1):
                outcome.update(status="failed", score=None, error_code="INVALID_METRIC_SCORE")
            metrics.append(outcome)
            if outcome["status"] == "failed":
                failure("metric." + outcome["name"], outcome["error_code"])
        names = [item["name"] for item in metrics]
        valid_scores = all(
            item["status"] == "succeeded" and isinstance(item["score"], (int, float))
            and not isinstance(item["score"], bool) and math.isfinite(item["score"])
            and 0 <= item["score"] <= 1
            for item in metrics if item["name"] in required
        )
        if evaluation.get("status") == "succeeded" and (
            not required.issubset(names) or len(set(names)) != len(names) or not valid_scores
            or result.get("status") != "succeeded" or result.get("schema_version") != "1.0"
            or result.get("request_id") != row.get("request_id")
            or result.get("conversation_id") != row.get("conversation_id")
            or result.get("record_sha256") != evaluation.get("record_sha256")
        ):
            evaluation.update(status="failed", error_code="DATASET_RESULT_INVALID")
        cleanup = recorder.cleanups.get(row.get("session_id"), {"status": "not_applicable"})
        status = row["status"]
        if status == "succeeded" and (evaluation.get("status") != "succeeded" or cleanup.get("status") != "succeeded"):
            status = "failed"
        failure(str(row.get("failure_stage") or "rag"), row.get("failure_code"))
        failure("evaluation", evaluation.get("error_code") if evaluation.get("status") == "failed" else None)
        failure("cleanup", cleanup.get("error_code"))
        trace = row.get("deep_trace") or {}
        operation = trace.get("operation") or {}
        output.append({
            **{key: row.get(key) for key in (
                "source_index", "session_id", "request_id", "conversation_id",
                "diagnostic_operation_id", "diagnostic_trace_session_id",
                "diagnostic_trace_session_sequence", "query_http_attempted", "answer_complete",
                "answer_sha256", "answer_utf8_bytes", "telemetry", "duration_ms",
                "client_timings_ms", "trace_polling", "post_generation",
                "failure_stage", "failure_code", "failure_scope",
            )},
            "query_id": query.query_id,
            "question_sha256": hashlib.sha256(query.question.encode("utf-8")).hexdigest(),
            "status": status, "rag_status": row["status"],
            "session_cleanup": cleanup,
            "evaluation": evaluation, "metrics": metrics,
            "evaluator_identity": {key: result.get(key) for key in ("ragas_version", "judge", "embeddings")},
            "final_context_evidence": {
                group: {key: value for key, value in (operation.get(group) or {}).items()
                        if key.startswith(("qwen_knowledge_", "qwen_policy_", "granite_rewritten_query_"))}
                for group in ("counts", "digests", "samples", "flags")
            },
        })
    def timing_values(row: Mapping[str, object]) -> dict[str, object]:
        values = {"diagnostic.duration_ms": row.get("duration_ms")}
        for group, payload in (
            ("rag", (row.get("telemetry") or {}).get("timings_ms") or {}),
            ("client", row.get("client_timings_ms") or {}),
            ("trace", row.get("trace_polling") or {}),
            ("registry", row.get("registry_verification") or {}),
        ):
            for name, value in payload.items():
                if group == "rag" or name.endswith("_ms"):
                    values[group + "." + name] = value
        for name, stage in ((row.get("deep_trace") or {}).get("operation") or {}).get("stages", {}).items():
            values["stage." + name] = stage.get("total_ms")
        for name, task in (row.get("post_generation") or {}).items():
            for timing in ("queue_wait_ms", "execution_ms", "total_ms"):
                values["task." + name + "." + timing] = task.get(timing)
        return values
    cohorts = {
        "all_attempts": [r for r in rows if r.get("query_http_attempted")],
        "successful_attempts": [r for r in rows if r["status"] == "succeeded"],
        "failed_attempts": [r for r in rows if r["status"] == "failed" and r.get("query_http_attempted")],
        "completed_streams": [r for r in rows if r.get("answer_complete")],
    }
    latency = {}
    for cohort, members in cohorts.items():
        maps = [timing_values(row) if cohort == "completed_streams" else
                {"diagnostic.duration_ms": row.get("duration_ms")} for row in members]
        names = set().union(*(mapping.keys() for mapping in maps)) or {"diagnostic.duration_ms"}
        latency[cohort] = {name: _finite_samples([m.get(name) for m in maps], len(members)) for name in sorted(names)}
    metric_names = ("faithfulness", "response_relevancy", "context_utilization",
                    "context_recall", "context_precision_with_reference", "factual_correctness", "noise_sensitivity")
    metric_statistics = {}
    for name in metric_names:
        outcomes = [next((m for m in row["metrics"] if m["name"] == name), None) for row in output]
        counts = {status: sum(m is not None and m["status"] == status for m in outcomes) for status in ("succeeded", "failed", "skipped")}
        applicable = [bool(row.get("answer_complete")) and (name in metric_names[:3] or
                      (name in metric_names[3:6] and query.reference is not None))
                      for row, query in zip(output, selection.selected, strict=True)]
        counts["failed"] += sum(needed and outcome is None for needed, outcome in zip(applicable, outcomes, strict=True))
        metric_statistics[name] = {
            **counts, "missing": sum(m is None for m in outcomes),
            "applicable": sum(applicable),
            "scores": _finite_samples([m.get("score") if m is not None and m["status"] == "succeeded" else None for m in outcomes], len(output)),
        }
    counts = {status: sum(row["status"] == status for row in output) for status in ("succeeded", "failed", "not_attempted")}
    payload = {
        "schema_version": "1.0", "report": "rag-ragas-dataset", "run_id": run.run_id,
        "status": "succeeded" if summary["status"] == "succeeded" and counts["succeeded"] == len(output) else "failed",
        "dataset": {"sha256": selection.source_sha256, "format": selection.source_path.suffix[1:]},
        "accounting": {"intended": len(selection.selected),
                       "attempted": sum(bool(r.get("query_http_attempted")) for r in rows),
                       **counts, "row_count": len(output), "silently_dropped": 0, "identity_complete": True},
        "queries": output, "failure_counts": failures, "identities": {**identities, "corpus": _corpus_summary(active)},
        "trace_rotation": {"registry_capacity": 256, "rotation_size": 64,
                           "planned_sessions": math.ceil(len(selection.selected) / 64),
                           "sessions": recorder.trace_sessions,
                           "all_started_sessions_deleted": all(s["status"] == "deleted" for s in recorder.trace_sessions)},
        "latency_statistics": latency,
        "evaluation_statistics": {name: _finite_samples([(r.get("evaluation") or {}).get(name) for r in rows], len(rows))
                                  for name in ("queue_wait_ms", "execution_ms", "evaluation_ms")},
        "ragas_statistics": metric_statistics,
        "statistics_definition": {"percentiles": "nearest rank ceil(p*n)", "variance": "population denominator n",
                                 "missing": "null; never zero", "time_units": "ms; variance ms^2",
                                 "score_units": "dimensionless [0,1]", "rounding": "once: 3 decimals; variance 6 decimals",
                                 "failed_attempts": "observed time to diagnostic failure; separate from success",
                                 "evaluation_timing": "excluded from RAG and diagnostic duration"},
        "lifecycle": summary["lifecycle"], "failure_stage": summary.get("failure_stage"),
        "started_at": summary["started_at"], "finished_at": summary["finished_at"],
    }
    _write_json_atomic(run.dataset_report_path, payload)


def create_e2e_run(
    output_root: Path,
    diagnostic_user_id: str,
    selection: QuerySelection,
    active: CorpusGeneration,
    *,
    continuous: bool,
) -> E2ERun:
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    directory = output_root / run_id
    try:
        if output_root.is_symlink():
            raise E2EDiagnosticError("E2E diagnostic output root must not be a symlink")
        output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(output_root, 0o700)
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        os.chmod(directory, 0o700)
        requests_path = directory / "requests.jsonl"
        descriptor = os.open(requests_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        summary_path = directory / "summary.json"
        _write_json_atomic(
            summary_path,
            {
                "schema_version": E2E_SCHEMA_VERSION,
                "diagnostic": "e2e",
                "phase": E2E_PHASE,
                "run_id": run_id,
                "status": "running",
                "diagnostic_user_id": diagnostic_user_id,
                "queries": {
                    "path": os.fspath(selection.source_path),
                    "sha256": selection.source_sha256,
                    "total": selection.total_queries,
                    "start": selection.start,
                    "limit": selection.limit,
                    "selected": len(selection.selected),
                },
                "session_mode": "continuous" if continuous else "fresh",
                "corpus": _corpus_summary(active),
                "corpus_validation": {"state": "succeeded", "physical": "pending"},
                "deep_trace": {"status": "pending", "session_deleted": False},
                "requests": {
                    "selected": len(selection.selected),
                    "attempted": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "query_posts": 0,
                    "individual_failed": 0,
                    "systemic_failed": 0,
                    "not_attempted": len(selection.selected),
                },
                "lifecycle": {
                    "pre_down": "pending",
                    "up": "pending",
                    "down": "pending",
                },
                "failure_stage": None,
                "started_at": utc_timestamp(started_at),
                "finished_at": None,
            },
        )
    except OSError as exc:
        raise E2EDiagnosticError(
            f"Could not create E2E diagnostic output in {directory}"
        ) from exc
    return E2ERun(run_id, directory, requests_path, summary_path)


class RequestRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.attempted = 0
        self.succeeded = 0
        self.failed = 0
        self.query_posts = 0
        self.individual_failed = 0
        self.systemic_failed = 0
        self.evaluation_succeeded = 0
        self.evaluation_partial = 0
        self.evaluation_failed = 0
        self.evaluation_skipped = 0
        self.not_attempted = 0
        self.dataset: tuple[QuerySelection, CorpusGeneration, Mapping[str, object]] | None = None
        self.cleanups: dict[str, dict[str, object]] = {}
        self.trace_sessions: list[dict[str, object]] = []
        self.recorded_indexes: set[int] = set()

    @property
    def totals(self) -> dict[str, int]:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "query_posts": self.query_posts,
            "individual_failed": self.individual_failed,
            "systemic_failed": self.systemic_failed,
        }

    @property
    def evaluation_totals(self) -> dict[str, int]:
        return {
            "succeeded": self.evaluation_succeeded,
            "partial": self.evaluation_partial,
            "failed": self.evaluation_failed,
            "skipped": self.evaluation_skipped,
        }

    def record(
        self,
        *,
        source_index: int,
        query_id: str | None = None,
        status: str,
        question: str,
        answer: str,
        answer_complete: bool,
        session_id: str | None,
        request_id: str | None,
        conversation_id: str | None,
        http_status: int | None,
        token_event_count: int,
        telemetry_schema_version: str | None,
        timings_ms: Mapping[str, object] | None,
        duration_ms: float,
        started_at: str,
        failure_stage: str | None = None,
        failure_code: str | None = None,
        deep_trace: Mapping[str, object] | None = None,
        diagnostic_operation_id: str | None = None,
        failure_scope: str | None = None,
        query_http_attempted: bool = True,
        client_timings_ms: Mapping[str, object] | None = None,
        trace_polling: Mapping[str, object] | None = None,
        post_generation: Mapping[str, object] | None = None,
        registry_verification: Mapping[str, object] | None = None,
        evaluation: Mapping[str, object] | None = None,
    ) -> None:
        if source_index in self.recorded_indexes:
            raise E2EDiagnosticError("Request source index was already recorded")
        if status not in {"succeeded", "failed"}:
            raise E2EDiagnosticError("Request status must be succeeded or failed")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
        ):
            raise E2EDiagnosticError("Request source index is invalid")
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(answer, str)
        ):
            raise E2EDiagnosticError("Request text fields are invalid")
        if not isinstance(answer_complete, bool):
            raise E2EDiagnosticError("answer_complete must be boolean")
        if not isinstance(query_http_attempted, bool):
            raise E2EDiagnosticError("query_http_attempted must be boolean")
        if failure_scope not in {None, "individual", "systemic"}:
            raise E2EDiagnosticError("failure_scope is invalid")
        if status == "succeeded" and failure_scope is not None:
            raise E2EDiagnosticError("Successful requests cannot have a failure scope")
        if status == "failed" and failure_scope is None:
            raise E2EDiagnosticError("Failed requests require a failure scope")
        for name, value in (
            ("session_id", session_id),
            ("request_id", request_id),
            ("conversation_id", conversation_id),
            ("diagnostic_operation_id", diagnostic_operation_id),
        ):
            if value is not None:
                try:
                    UUID(value)
                except (TypeError, ValueError) as exc:
                    raise E2EDiagnosticError(f"{name} must be a UUID") from exc
        if (
            isinstance(token_event_count, bool)
            or not isinstance(token_event_count, int)
            or token_event_count < 0
            or not isinstance(duration_ms, (int, float))
            or isinstance(duration_ms, bool)
            or not math.isfinite(duration_ms)
            or duration_ms < 0
        ):
            raise E2EDiagnosticError("Request counts or duration are invalid")
        evaluation_status = "skipped"
        if evaluation is not None:
            candidate = evaluation.get("status")
            if candidate not in {"succeeded", "partial", "failed", "skipped"}:
                raise E2EDiagnosticError("Evaluation status is invalid")
            evaluation_status = str(candidate)
        self.attempted += 1
        if query_http_attempted:
            self.query_posts += 1
        if status == "succeeded":
            self.succeeded += 1
        else:
            self.failed += 1
            if failure_scope == "individual":
                self.individual_failed += 1
            else:
                self.systemic_failed += 1
        if evaluation_status == "succeeded":
            self.evaluation_succeeded += 1
        elif evaluation_status == "partial":
            self.evaluation_partial += 1
        elif evaluation_status == "failed":
            self.evaluation_failed += 1
        else:
            self.evaluation_skipped += 1
        payload = {
            "schema_version": E2E_SCHEMA_VERSION,
            "sequence": self.attempted,
            "source_index": source_index,
            "query_id": query_id,
            "status": status,
            "question": question,
            "answer": answer,
            "answer_complete": answer_complete,
            "session_id": session_id,
            "request_id": request_id,
            "conversation_id": conversation_id,
            "http_status": http_status,
            "token_event_count": token_event_count,
            "telemetry": (
                None
                if timings_ms is None
                else {
                    "schema_version": telemetry_schema_version,
                    "timings_ms": dict(timings_ms),
                }
            ),
            "duration_ms": round(float(duration_ms), 3),
            "started_at": started_at,
            "finished_at": utc_timestamp(),
            "failure_stage": failure_stage,
            "failure_code": failure_code,
            "failure_scope": failure_scope,
            "query_http_attempted": query_http_attempted,
            "client_timings_ms": (
                None if client_timings_ms is None else dict(client_timings_ms)
            ),
            "trace_polling": (
                None if trace_polling is None else dict(trace_polling)
            ),
            "post_generation": (
                None if post_generation is None else dict(post_generation)
            ),
            "registry_verification": (
                None
                if registry_verification is None
                else dict(registry_verification)
            ),
            "diagnostic_operation_id": diagnostic_operation_id,
            "deep_trace": None if deep_trace is None else dict(deep_trace),
            "evaluation": None if evaluation is None else dict(evaluation),
        }
        if self.dataset is not None:
            payload["answer_sha256"] = hashlib.sha256(answer.encode("utf-8")).hexdigest()
            payload["answer_utf8_bytes"] = len(answer.encode("utf-8"))
            payload["question_sha256"] = hashlib.sha256(question.encode("utf-8")).hexdigest()
            payload.pop("question")
            payload.pop("answer")
            if isinstance(payload["deep_trace"], Mapping):
                payload["deep_trace"] = {**payload["deep_trace"], "operation": {
                    key: value for key, value in payload["deep_trace"]["operation"].items()
                    if key != "texts"
                }}
            if isinstance(payload["registry_verification"], Mapping):
                payload["registry_verification"] = {
                    key: value for key, value in payload["registry_verification"].items() if key != "title"
                }
            payload["diagnostic_trace_session_id"] = self.trace_sessions[-1]["session_id"] if self.trace_sessions else None
            payload["diagnostic_trace_session_sequence"] = len(self.trace_sessions) or None
        try:
            with self.path.open("a", encoding="utf-8") as target:
                target.write(json.dumps(payload, sort_keys=True) + "\n")
                target.flush()
                os.fsync(target.fileno())
            self.recorded_indexes.add(source_index)
        except OSError as exc:
            raise E2EDiagnosticError(
                f"Could not append E2E request artifact: {self.path}"
            ) from exc

    def record_not_attempted(self, query: SelectedQuery, reason: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise E2EDiagnosticError("Not-attempted reason is invalid")
        if query.source_index in self.recorded_indexes:
            raise E2EDiagnosticError("Request source index was already recorded")
        self.not_attempted += 1
        payload = {
            "schema_version": E2E_SCHEMA_VERSION,
            "sequence": self.attempted + self.not_attempted,
            "source_index": query.source_index,
            "query_id": query.query_id,
            "status": "not_attempted",
            "question_sha256": hashlib.sha256(query.question.encode("utf-8")).hexdigest(),
            "answer_complete": False,
            "session_id": None,
            "request_id": None,
            "conversation_id": None,
            "diagnostic_operation_id": None,
            "query_http_attempted": False,
            "failure_scope": "systemic",
            "failure_stage": "batch",
            "failure_code": reason,
            "telemetry": None,
            "evaluation": {"status": "skipped", "error_code": "RAG_NOT_ATTEMPTED"},
            "started_at": None,
            "finished_at": utc_timestamp(),
        }
        try:
            with self.path.open("a", encoding="utf-8") as target:
                target.write(json.dumps(payload, sort_keys=True) + "\n")
                target.flush()
                os.fsync(target.fileno())
            self.recorded_indexes.add(query.source_index)
        except OSError as exc:
            raise E2EDiagnosticError(f"Could not append E2E request artifact: {self.path}") from exc


def update_e2e_summary(
    run: E2ERun,
    recorder: RequestRecorder,
    *,
    status: str,
    pre_down_status: str,
    up_status: str,
    down_status: str,
    physical_corpus_status: str,
    trace_status: str = "not_started",
    trace_deleted: bool = False,
    failure_stage: str | None = None,
    finished: bool = False,
) -> None:
    try:
        payload = json.loads(run.summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("summary root must be an object")
        prior_requests = payload.get("requests")
        if not isinstance(prior_requests, Mapping):
            raise ValueError("summary requests field must be an object")
        selected = prior_requests.get("selected")
        payload.update(
            {
                "status": status,
                "corpus_validation": {
                    "state": "succeeded",
                    "physical": physical_corpus_status,
                },
                "requests": {
                    "selected": selected,
                    **recorder.totals,
                    "not_attempted": recorder.not_attempted + max(
                        0, int(selected) - recorder.attempted - recorder.not_attempted
                    ),
                },
                "deep_trace": {
                    "status": trace_status,
                    "session_deleted": trace_deleted,
                },
                "evaluation": recorder.evaluation_totals,
                "lifecycle": {
                    "pre_down": pre_down_status,
                    "up": up_status,
                    "down": down_status,
                },
                "failure_stage": failure_stage,
                "finished_at": utc_timestamp() if finished else None,
            }
        )
        _write_json_atomic(run.summary_path, payload)
        if finished and recorder.dataset is not None:
            write_dataset_report(run, recorder, payload)
            report = json.loads(run.dataset_report_path.read_text())
            if payload["status"] == "succeeded" and report["status"] != "succeeded":
                raise E2EDiagnosticError("Dataset report validation failed")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise E2EDiagnosticError(
            f"Could not update E2E diagnostic summary: {run.summary_path}"
        ) from exc


__all__ = [
    "E2EDiagnosticError",
    "E2E_PHASE",
    "E2E_SCHEMA_VERSION",
    "E2ERun",
    "QuerySelection",
    "RequestRecorder",
    "SelectedQuery",
    "create_e2e_run",
    "load_query_selection",
    "update_e2e_summary",
    "utc_timestamp",
    "validate_reusable_corpus_state",
]
