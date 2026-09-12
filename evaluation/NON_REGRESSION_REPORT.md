# Evaluation Non-Regression Report

## Status

| Gate | Status |
|---|---|
| Offline safety | PASS |
| Core RAG preservation | PASS |
| Evaluator isolation | PASS |
| Evidence integrity | PASS (offline proof) |
| Concurrency | PASS (offline proof) |
| Live ordinary ask | BLOCKED — NOT RUN |
| Live E2E | BLOCKED — NOT RUN |
| OFF/ON latency | BLOCKED — NOT RUN |
| Evaluator-overlap latency | BLOCKED — NOT RUN |
| Real Ragas execution | BLOCKED — NOT RUN |
| Overall live/deployment acceptance | **BLOCKED — NOT PASS** |

Behavioral baseline: `fba3c3481d5dc438b6081ab85afcbaa99dbc2987`

Approved Prompt 1–3 checkpoint: `d37c82710a8bd166af4c68f5f57d1068a7cdc880`

Resume checkpoint: `31cb4a6a` (clean worktree on resumption). The interrupted
rewrite-correlation test passed on resumption; this report supersedes the earlier
803-test report.

## Production-facing hardening

Prompt 4 changes these implementation areas only:

- `backend/wizard/diagnostics.py` restores fail-closed malformed-correlation handling for the deep singleton while retaining per-session evaluation fault isolation.
- `deployment/evaluation_bridge.py` and `deployment/evaluation_bridge_worker.py` bound the admitted hydration/evaluation lifecycle and remove production credentials from evaluator child environments.
- `deployment/e2e_diagnostic_api.py` adds the three local timing fields to every terminal evaluation artifact without changing its historical request `duration_ms` boundary.
- `deployment/ragctl.py` prints the same local queue, execution, and total timing fields after an ordinary successful answer.

The resumed review corrected three demonstrated local defects: ordinary ask's
omitted corpus restriction previously raised before evaluation submission;
a nonzero evaluator exit could be hidden by a success-shaped result file; and
the original watchdog could exit before stopping a surviving child. Ask now
defaults its corpus restriction to `None`, success requires a zero child exit,
and a separate supervisor enforces the admitted deadline and reaps its child.
Inherited lock descriptors are closed rather than explicitly unlocked, so an
orphan retains the slot until its independent deadline terminates it. Timing
values freeze after cleanup.

Review anchors: `backend/wizard/diagnostics.py:376`,
`deployment/evaluation_bridge.py:598`, `:710`, `:952`, `:1170`, `:1184`,
`deployment/evaluation_bridge_worker.py:67`, `:138`.
The complete production-facing diff is reproducible with
`git diff d37c82710a8bd166af4c68f5f57d1068a7cdc880 -- backend deployment`.

The main RAG pipeline, retrieval and reranking, prompts, Granite/Qwen clients,
public API models, SSE, telemetry, Weaviate schemas/mutations, task queue,
persistence, and title flow were not changed by Prompt 4.

## Offline evidence

Before Prompt 4 changes:

- Main suite: 789 passed, 4 skipped.
- Isolated evaluator suite: 47 passed.
- `git diff --check`: passed.

After Prompt 4 changes:

- Dedicated non-regression, supervision, and live-artifact gate modules: 35 passed.
- Focused bridge/pipeline/supervision set: 44 passed.
- Main suite: 825 passed, 4 expected integration skips; two dependency deprecation warnings.
- Isolated evaluator suite: 48 passed.
- Python compilation: passed.
- Protected-contract preflight: passed, 30 files checked.
- `git diff --check`: passed.

The protected manifest records the behavioral-baseline hashes, identifies each
approved evidence-hook file, and pins 30 current production/deployment sources
plus the public telemetry, request/task, retrieval-ceiling, context-budget,
Granite, Qwen, and SSE contracts. The real-artifact verifier refuses live
evidence if this preflight fails.

Controlled offline tests require identical final prompts, tokenizer call
transcripts, and answer bytes for the same provider stream with evidence capture
disabled and enabled. They also verify exact ordered final Knowledge/Policy IDs,
Unicode content fingerprints and aggregate digests, and reject missing,
duplicated, substituted, pre-budget, or digest-mismatched evidence locally.

The evaluator project remains isolated under `evaluation/`. Backend and Modal
requirements contain no Ragas dependency, and production modules import neither
Ragas nor `rag_evaluation`. The local worker performs read-only evidence
hydration and launches a loopback-only evaluator after validated SSE completion.

The baseline comparison tests assert byte identity for every protected core
source outside the explicitly approved evidence files. Granite and generator
diffs are checked against the literal two additions per file (one import and one
observation call), without stripping or normalizing production code. Controlled
pipeline runs compare ordered calls within each K/P branch; concurrent worker
scheduling is not incorrectly treated as a fixed global order.

| Failure/integrity case | Offline evidence |
|---|---|
| Missing Ragas/Ollama, setup timeout, metric exceptions | `evaluation/tests/test_evaluator.py`: independent safe failed outcomes |
| Missing evaluator, crash, corrupt result, nonzero exit with success file | `tests/test_evaluation_bridge.py`, `tests/test_evaluation_supervision.py` |
| Missing/substituted context, Unicode digest mismatch, wrong rewrite | Bridge and dedicated non-regression tests |
| Evidence capacity and owning-session fault/delete isolation | Existing bridge and registry tests |
| Blocked hydration, cancellation, incomplete IPC, supervisor crash | Separate harmless OS processes; lock retention, termination, and recovery verified |
| Inference concurrency versus local admission | Concurrent ask orchestration tests plus OS-backed lock/supervision tests |
| Evaluation timing versus remote post-generation polling | Historical E2E duration test and post-cleanup timing freeze test |
| Real latency or real judge quality | NOT RUN; synthetic fixtures are not live evidence |

Commands used (from repository root):

```sh
env PYTHONDONTWRITEBYTECODE=1 RUN_MODAL_SGLANG_TESTS=0 \
  RUN_QWEN_SGLANG_TESTS=0 RUN_LOCAL_GRANITE_TESTS=0 RUN_ONNX_CUDA_TESTS=0 \
  .venv/bin/python -m pytest -q
env PYTHONDONTWRITEBYTECODE=1 RAGAS_DO_NOT_TRACK=true HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 RUN_MODAL_SGLANG_TESTS=0 RUN_QWEN_SGLANG_TESTS=0 \
  RUN_LOCAL_GRANITE_TESTS=0 RUN_ONNX_CUDA_TESTS=0 \
  evaluation/.venv/bin/python -m pytest -o addopts='' -q evaluation/tests
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python evaluation/non_regression_gate.py preflight
git diff --check
```

Python sources were compiled in memory with `compile(..., 'exec')`, excluding
virtual environments; no model imports or bytecode artifacts were needed.

## Live acceptance

Live PASS requires successful ordinary ask and E2E runs, successful persistence
and title tasks, exact evidence hydration, and successful `faithfulness`,
`response_relevancy`, and `context_utilization` metrics from real local Ragas.

The OFF/ON and evaluator-contention gates each require at least 30 complete
matched pairs. They use 10,000 deterministic bootstrap resamples and require the
one-sided 95% upper confidence bound of the paired median change to be strictly
below +5% for `ttft`, `generation`, and `total_request`.

The verifier checks OFF → ON → ON → OFF runtime identity, three warmups per
block, identical configuration/corpus/model digests, complete paired queries,
authoritative telemetry, and successful persistence/title FIFO. Contended
requests must begin during an actual Ollama request for another successfully
evaluated sample. A live process alone is insufficient; incomplete pairs are
rejected rather than discarded.

This checkout provides read-only prerequisite and collected-artifact gates,
not an automatically executed benchmark. The authorized live collector must
retain enriched ask evidence before trace deletion, task observations, raw
latency samples, and actual judge request intervals. Ordinary ask `status.json`
alone lacks that proof and cannot pass this gate. No raw live sample files,
bootstrap results, answer-quality claims, or real judge results exist yet.

Live answer digests are recorded but independent stochastic Qwen responses are
not required to be byte-identical. An evidence-OFF runtime does not expose
rewrite or context traces, so the live report makes no claim that those
unobserved values are equal.

No live run was performed. The repository currently has only `.gitkeep` Wizard
fixture markers, an empty query list, and no local Phase 1 corpus state. Modal,
real Weaviate, Ollama inference, `./rag ask`, Wizard diagnostics, and E2E remain
unexecuted pending explicit authorization and genuine prerequisites.
