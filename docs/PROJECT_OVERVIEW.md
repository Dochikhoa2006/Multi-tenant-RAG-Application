# Multi-Tenant Multi-Agent Advanced RAG Application
## General-Purpose Retrieval-Augmented Generation

---

## Product Vision

A general-purpose retrieval-augmented generation application that answers
questions from user-owned knowledge and policy material while using completed
conversation history to resolve follow-up context.

Each user owns three physically isolated Weaviate collections: Conversation,
Knowledge Facts, and Policy. Wizards provide lifecycle management for knowledge
and policy content. The query path combines BM25/LateOn hybrid retrieval, BGE
reranking, Adaptive-K, application MMR, Granite query rewriting, and grounded
Qwen answer generation.

The product's multi-agent name describes this fixed coordination of specialized
model-backed roles. The current implementation is not an autonomous-agent,
tool-calling, or dynamic agent-delegation framework.

---

## Core Concepts and Terminology

| Term | Definition |
|---|---|
| **User** | An authenticated individual identified by a unique `user_id`. Each user owns an isolated data environment. |
| **Collection** | One of three physically separate Weaviate collections per user: **Conversation**, **Knowledge Facts**, or **Policy**. Its logical identity is the pair of the exact `user_id` and canonical collection type. Its physical Weaviate name uses reversible Base32 user-ID encoding as specified in `BACKEND.md`. |
| **Wizard** | A user-facing content board mapped one-to-one to a **Document ID**. Users view, edit, upload into, and delete wizards through the UI. |
| **Document ID** | The internal identifier for a wizard. Each wizard is exactly one document. |
| **Paragraph ID** | A segment boundary produced by **Semantic Paragraph Splitting** within a document. Paragraphs are numbered sequentially from top to bottom of the wizard text. |
| **Chunk ID** | The stable object identifier for one lossless Knowledge Facts or Policy semantic chunk. Each chunk stores a LateOn token matrix for first-stage retrieval and a GTE dense vector for MMR diversity. |
| **Conversation ID** | The stable canonical identity of one completed question-answer pair. A long canonical pair may be stored as several lossless retrieval-segment objects that all retain the same `conversation_id`; segment hits collapse back to that identity after BGE reranking. |
| **Session** | A UI-level grouping of multiple conversations within chat mode. Sessions have auto-generated titles but do not affect backend retrieval. |

---

## Current Repository Structure

```
repository/
├── backend/
│   ├── api/                 # Chat, task, and shared Knowledge/Policy routers
│   ├── mappings/            # Process-local session/document/paragraph ownership
│   ├── processing/          # File reading, paragraph splitting, and chunking
│   ├── providers/           # Granite/Qwen and ONNX CUDA provider adapters
│   ├── rag/                 # Retrieval, rewrite, generation, persistence orchestration
│   ├── weaviate_client/     # Schema, collection lifecycle, retrieval, and hydration
│   ├── wizard/              # Wizard CRUD, save, rollback, and recovery
│   ├── main.py              # Provider-neutral FastAPI factory
│   ├── runtime_app.py       # Integrated single-process production composition
│   ├── services.py          # Shared application services and chat registry
│   └── model_config.py      # Model, retrieval, and token-budget configuration
├── deployment/              # Modal services, secure Weaviate Compose, and ragctl
├── scripts/                 # Artifact manifests, migration, and opt-in benchmarks
├── tests/                   # Offline contract and integration tests
├── docs/                    # Architecture, API, configuration, and UI specifications
├── compose.yaml             # Linux NVIDIA development topology
├── Dockerfile               # Single-worker integrated API image
└── rag                      # One-command lifecycle shim
```

The repository does not currently contain the production SPA described in
`FRONTEND.md`. The integrated backend exposes a small development-only
`/dev/e2e` harness; `FRONTEND.md` remains the contract for a future UI.

---

## System Architecture Overview

```text
Client or development harness
            │ REST / SSE
            ▼
Single-worker FastAPI runtime
  ├─ original query → Conversation hybrid C50 → BGE → collapse
  │                    → Adaptive-K → MMR → Granite rewrite
  ├─ rewritten query ─┬→ Knowledge hybrid K50 → BGE → Adaptive-K → MMR
  │                   └→ Policy hybrid P40 → BGE → Adaptive-K → MMR
  └─ bounded context → Qwen non-thinking answer stream
            │
            ├─ completed Q+A → background Conversation vectors
            └─ completed answer → background session-title generation

Per-user Weaviate collections
  ├─ Conversation segment objects → one canonical conversation_id
  ├─ Knowledge Facts chunk objects
  └─ Policy chunk objects

Every object supplies two named vectors:
  late_interaction → 128-D LateOn token vectors for MaxSim hybrid search
  mmr_diversity   → 768-D GTE dense vector for application MMR only
```

---

## Related Specification Documents

| Document | Description |
|---|---|
| [BACKEND.md](./BACKEND.md) | Weaviate data model, retrieval pipeline, wizard CRUD operations, text processing pipeline |
| [FRONTEND.md](./FRONTEND.md) | UI layout, chat mode, knowledge facts mode, policy mode |
| [API.md](./API.md) | REST endpoint reference for chat, knowledge facts, and policy wizards |
| [CONFIG_SPECS.md](./CONFIG_SPECS.md) | Models, hyperparameters, search top-k, chunking, and token budgets |
| [DESIGN_DECISIONS.md](./DESIGN_DECISIONS.md) | Architectural rationale, non-functional requirements, future considerations |
