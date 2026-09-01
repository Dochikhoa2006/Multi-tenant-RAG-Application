# Smart RAG Interview Preparation System

Full-stack RAG application for Generative AI Engineer FAANG interview coaching. Backed by Weaviate vector database with three isolated collections (Conversation, Knowledge Facts, Policy) per user.

## One-command private deployment

After the model Volumes, Modal account, `.env`, Standalone Tailscale node, and
persistent Weaviate data have been provisioned as described in
[deployment/README.md](./deployment/README.md), use the repo-local lifecycle
command:

```bash
./rag up
./rag ask
./rag status
./rag down
```

`./rag up` starts authenticated Weaviate/HAProxy and both Funnels, deploys and
validates Granite, Qwen, and the singleton CUDA runtime, and exits only after
authenticated health succeeds. `./rag ask` creates a fresh process-local chat
session and streams one real RAG answer. `./rag down` stops all three Modal GPU
apps, disables both application Funnels, and stops the local containers without
deleting the Weaviate volume, Modal Volumes, or secrets. Credentials are loaded
from `.env` and are never printed.

The operational GPU requests are also read from `.env` as
`MODAL_SGLANG_GPU`, `QWEN_MODAL_SGLANG_GPU`, and `MODAL_RAG_GPU`. Only L40S and
the infrastructure-only H100 capacity substitution are accepted; deployment
source defaults remain L40S and no automatic GPU switching occurs.

Production query rewriting uses the merged Granite 4.1-3B checkpoint through an
always-warm SGLang/Modal CUDA service. Answer streaming and title completion use
the external `qwen3-4b-awq` checkpoint through a second always-warm SGLang
worker, with thinking disabled. Dense embeddings and Knowledge/Policy reranking
run locally from FP16 graphs through reusable ONNX Runtime sessions (CUDA by
default). See
[deployment/README.md](./deployment/README.md).

## Integrated Development Runtime

Stage 7A provides the complete single-process composition at
`backend.runtime_app:create_runtime_app`. It loads one shared ONNX embedding
client, reranker, Granite client, Qwen client, Weaviate manager, and task queue.
The Granite and Qwen SGLang servers remain external services.

For a native start, provision the configured local model paths and external
SGLang endpoints in `.env`, then run:

```bash
set -a
source .env
set +a
uvicorn backend.runtime_app:create_runtime_app \
  --factory --host 0.0.0.0 --port 8000 --workers 1
```

For the supported CUDA Docker runtime:

```bash
cp .env.example .env
# Edit .env with absolute model paths, SGLang URLs, and credentials.
docker compose up --build --wait
```

Open <http://localhost:8000/dev/e2e> for the minimal development-only browser
harness, or <http://localhost:8000/health> for readiness. The fake console
allows only one active manual chat request and is not the Stage 6 frontend. The Docker profile
requires Linux/amd64, NVIDIA Container Toolkit, one compatible GPU, provisioned
FP16 ONNX artifacts, the local Granite checkpoint/tokenizer, and a local copy of
the Stage 1 segmentation model. Segmentation loads only from
`SEGMENTATION_MODEL_PATH`, uses `SEGMENTATION_EMBEDDING_DEVICE=cpu` by default,
and never downloads at runtime. Docker Compose starts Weaviate but deliberately
does not bundle either SGLang worker.

## Documentation

| Document | What It Covers |
|---|---|
| [PROJECT_OVERVIEW.md](./docs/PROJECT_OVERVIEW.md) | Product vision, terminology, directory structure, architecture diagram |
| [BACKEND.md](./docs/BACKEND.md) | Weaviate data model, retrieval pipeline, wizard CRUD, text processing |
| [FRONTEND.md](./docs/FRONTEND.md) | UI layout, chat mode, knowledge facts mode, policy mode |
| [API.md](./docs/API.md) | REST endpoint reference (chat, knowledge facts, policy wizards) |
| [CONFIG_SPECS.md](./docs/CONFIG_SPECS.md) | Models, hyperparameters, search top-k, chunking, and token budgets |
| [DESIGN_DECISIONS.md](./docs/DESIGN_DECISIONS.md) | Architectural rationale, non-functional requirements, future roadmap |
