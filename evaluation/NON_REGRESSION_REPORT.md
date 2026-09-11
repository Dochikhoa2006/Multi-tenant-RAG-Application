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

## Production-facing hardening

Prompt 4 changes these implementation areas only:

- `backend/wizard/diagnostics.py` restores fail-closed malformed-correlation handling for the deep singleton while retaining per-session evaluation fault isolation.
- `deployment/evaluation_bridge.py` and `deployment/evaluation_bridge_worker.py` bound the admitted hydration/evaluation lifecycle and remove production credentials from evaluator child environments.
- `deployment/e2e_diagnostic_api.py` adds the three local timing fields to every terminal evaluation artifact without changing its historical request `duration_ms` boundary.
- `deployment/ragctl.py` prints the same local queue, execution, and total timing fields after an ordinary successful answer.

The main RAG pipeline, retrieval and reranking, prompts, Granite/Qwen clients,
public API models, SSE, telemetry, Weaviate schemas/mutations, task queue,
persistence, and title flow were not changed by Prompt 4.

## Offline evidence

Before Prompt 4 changes:

- Main suite: 789 passed, 4 skipped.
- Isolated evaluator suite: 47 passed.
- `git diff --check`: passed.

After Prompt 4 changes:

- Dedicated Prompt 4 non-regression module: 14 passed.
- Focused semantic/integration set: 465 passed.
- Main suite: 803 passed, 4 expected integration skips.
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

## Live acceptance

Live PASS requires successful ordinary ask and E2E runs, successful persistence
and title tasks, exact evidence hydration, and successful `faithfulness`,
`response_relevancy`, and `context_utilization` metrics from real local Ragas.

The OFF/ON and evaluator-contention gates each require at least 30 complete
matched pairs. They use 10,000 deterministic bootstrap resamples and require the
one-sided 95% upper confidence bound of the paired median change to be strictly
below +5% for `ttft`, `generation`, and `total_request`.

Live answer digests are recorded but independent stochastic Qwen responses are
not required to be byte-identical. An evidence-OFF runtime does not expose
rewrite or context traces, so the live report makes no claim that those
unobserved values are equal.

No live run was performed. The repository currently has only `.gitkeep` Wizard
fixture markers, an empty query list, and no local Phase 1 corpus state. Modal,
real Weaviate, Ollama inference, `./rag ask`, Wizard diagnostics, and E2E remain
unexecuted pending explicit authorization and genuine prerequisites.
