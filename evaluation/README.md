# Local Ragas evaluation

This directory is an isolated, post-generation evaluator. It consumes one
immutable evidence record that was captured from a completed generated-query
path and sends only evaluation prompts to a local Ollama judge. It never calls
the RAG API, production SGLang servers, Weaviate, Modal, or session storage.

Wizard CRUD diagnostics are deliberately unsupported: Phase 1 validates the
corpus and lifecycle but does not produce an answer that Ragas can evaluate.

## Environment

The evaluator has its own Python 3.12 environment and lockfile:

```sh
uv sync --project evaluation --locked
```

The dependency pins intentionally keep Ragas 0.4.3 on the compatible
LangChain 0.3 family. Do not resolve an evaluator compatibility problem by
changing `backend/requirements.txt` or the application environment.

Defaults:

- judge: the local Ollama model `qwen3.5:4b`;
- Ollama origin: `http://127.0.0.1:11434`;
- embeddings: `models/all-MiniLM-L6-v2` on CPU;
- Ragas tracking and Hugging Face network access: forcibly disabled.

Only a loopback Ollama URL is accepted. The judge can be changed to another
installed, non-production local model with `RAG_EVAL_JUDGE_MODEL`; the
production `qwen3-4b-awq` identifier is rejected. Optional configuration:

```sh
export RAG_EVAL_OLLAMA_URL=http://127.0.0.1:11434
export RAG_EVAL_JUDGE_MODEL=qwen3.5:4b
export RAG_EVAL_EMBEDDING_MODEL_PATH="$PWD/models/all-MiniLM-L6-v2"
export RAG_EVAL_TIMEOUT_SECONDS=300
```

## Compatibility smoke test

This verifies the pinned collections API, the installed judge metadata, metric
construction, and a 384-dimensional local embedding. It does not issue a
judge completion:

```sh
uv run --project evaluation rag-evaluate smoke
```

## Record contract

An input is one JSON object with schema `1.0`. The context arrays contain the
exact raw chunks supplied to Qwen, without role prefixes or other mutation.
Knowledge precedes Policy in `retrieved_contexts`, matching prompt order.

```json
{
  "schema_version": "1.0",
  "source": "e2e",
  "request_id": "11111111-1111-4111-8111-111111111111",
  "conversation_id": "22222222-2222-4222-8222-222222222222",
  "original_query": "What is covered?",
  "rewritten_query": "coverage facts and policy",
  "response": "The generated answer.",
  "knowledge_contexts": ["Exact knowledge chunk."],
  "knowledge_context_ids": ["33333333-3333-4333-8333-333333333333"],
  "policy_contexts": ["Exact policy chunk."],
  "policy_context_ids": ["44444444-4444-4444-8444-444444444444"],
  "retrieved_contexts": ["Exact knowledge chunk.", "Exact policy chunk."],
  "context_roles": ["knowledge", "policy"],
  "telemetry": {"schema_version": "1.0", "timings_ms": {}},
  "reference": null,
  "reference_context_ids": [],
  "captured_at": "2026-09-10T12:00:00Z"
}
```

References must be supplied by a human or dataset. The evaluator never derives
one from the generated response.

## Run one evaluation

```sh
uv run --project evaluation rag-evaluate run /path/to/record.json
uv run --project evaluation rag-evaluate run /path/to/record.json \
  --noise-sensitivity
uv run --project evaluation rag-evaluate run /path/to/record.json \
  --output /path/to/result.json
```

By default, a private, uniquely named result is atomically created under
`evaluation/results/`. Existing files are never overwritten. Exit status is
`0` when every applicable metric succeeds, `1` for a partial or failed metric
run, `2` for input/setup/output failure, and `130` for interruption.

Each metric has an independent status. Reference-based metrics are explicitly
skipped when no reference exists. Errors contain stable codes and exception
class names, never raw provider errors, prompts, contexts, or judge reasoning.

## Application integration

`./rag ask` and the Phase 2 E2E diagnostic invoke this executable only after a
successful `token* -> telemetry -> done` response. The launcher resolves the
exact traced Knowledge/Policy chunk IDs from Weaviate with a local, read-only,
vector-free query and writes a private schema-`1.0` record before starting this
process. Ragas is never imported by the deployed backend or the deployment
launcher.

Ask evaluation is automatic; there is no evaluation flag or evaluation-owned
runtime restart. Ordinary deployment keeps full Wizard diagnostics disabled
while enabling only bounded, metadata-only evidence for the configured RAG
user. Concurrent asks each own a request-scoped evidence session. The backend
retains up to 256 such sessions independently of the deep diagnostic session's
256-operation bound.

After `done`, the launcher retrieves and deletes its server evidence before it
waits for local execution. Hydration, record construction, and Ragas execution
share a cross-process file lock at
`.local/diagnostics/evaluation/execution.lock`; the initial local concurrency is
one. This serializes only post-response evaluation, never inference.

E2E submits local evaluation before polling existing remote persistence/title
tasks, so those activities may overlap. Its historical `duration_ms` stops at
the original post-generation diagnostic boundary before evaluator join.
`evaluation_ms` is independent and includes local admission through terminal
completion; `queue_wait_ms` reports admission delay separately.

A missing evaluator environment, Ollama failure, invalid score, evidence
mismatch, timeout, or local artifact failure cannot change the generated answer
or remote persistence/title tasks. E2E records likewise keep inference and
evaluation statuses independent. See `ARCHITECTURE_AUDIT.md` for the exact
evidence and concurrency contracts.

## Tests

The unit tests use synthetic immutable records and injected local scorers; no
Ollama inference, Modal service, Weaviate service, or production endpoint is
required:

```sh
uv run --project evaluation pytest -c evaluation/pyproject.toml evaluation/tests
```
