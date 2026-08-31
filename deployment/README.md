# Modal SGLang Deployment

Production uses two dedicated SGLang 0.5.18 CUDA services. Granite handles only
query rewriting. The separate `qwen3-4b-awq` worker handles non-thinking answer
streaming and title completion through the existing `RoleRoutingLLMClient`.

## Provision local ONNX retrieval models

Provision these artifacts in a separate, network-enabled build environment.
The API runtime itself remains offline. Model directories are ignored by Git.

```bash
python -m venv .onnx-export-venv
source .onnx-export-venv/bin/activate
python -m pip install -r deployment/requirements-onnx-export.txt

hf download Alibaba-NLP/gte-modernbert-base \
  --revision e7f32e3c00f91d699e8c43b53106206bcc72bb22 \
  --include config.json tokenizer.json tokenizer_config.json onnx/model_fp16.onnx \
  --local-dir models/gte-modernbert-base

optimum-cli export onnx \
  --model BAAI/bge-reranker-v2-m3 \
  --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
  --task text-classification \
  --device cuda \
  --dtype fp16 \
  models/bge-reranker-v2-m3-onnx
mv models/bge-reranker-v2-m3-onnx/model.onnx \
  models/bge-reranker-v2-m3-onnx/model_fp16.onnx

python scripts/create_onnx_manifest.py \
  models/gte-modernbert-base \
  Alibaba-NLP/gte-modernbert-base \
  e7f32e3c00f91d699e8c43b53106206bcc72bb22
python scripts/create_onnx_manifest.py \
  models/bge-reranker-v2-m3-onnx \
  BAAI/bge-reranker-v2-m3 \
  953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
```

Before deployment, run the opt-in offline CUDA smoke test. It checks the GTE
768-dimensional normalized output and BGE pair-ranking contract without any
network access:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RUN_ONNX_CUDA_TESTS=1 \
  pytest tests/test_onnx_integration.py
```

The configured ONNX execution provider must be installed and able to execute
the provisioned FP16 graph. `CUDAExecutionProvider` is the production default;
another provider may be selected explicitly with the corresponding
`ONNX_*_EXECUTION_PROVIDER` setting. Each client registers only that provider,
passes `device_id` only to CUDA, and disables ONNX Runtime fallback. There is no
automatic CPU or remote-inference fallback.

Production must construct exactly one `ONNXEmbeddingClient` and one
`ONNXCrossEncoderReranker` in its ASGI composition root, inject both into the
existing `RAGRuntime`, and reuse that embedding client through the wizard
adapter. This repository does not currently contain that production ASGI
composition module, so the shared-client wiring cannot yet be verified and is
a deployment blocker. `backend.main:app` remains providerless.

## Disposable empty-state vector-profile migration

The new 768-dimensional vectors are incompatible with the former index, but
this prototype cannot safely migrate a populated application. Session,
transcript/title, document, paragraph, and chunk-ownership mappings are
process-local and are lost when FastAPI stops. Weaviate object export alone
cannot restore that state.

The maintenance command is therefore limited to explicitly disposable state
where every canonical Conversation, Knowledge Facts, and Policy collection is
empty. Stop the API, drain its task queue, confirm that all process-local state
may be discarded, take and verify a Weaviate backup, then run:

```bash
python scripts/migrate_onnx_vectors.py export /secure/vector-migration \
  --confirm-disposable-process-state
python scripts/migrate_onnx_vectors.py rebuild \
  /secure/vector-migration --confirm-maintenance \
  --confirm-disposable-process-state
python scripts/migrate_onnx_vectors.py verify /secure/vector-migration
```

Export validates complete canonical triplets and zero objects before writing a
permission-restricted manifest; it exports no records or vectors and does not
load an ONNX model. Rebuild repeats the zero-count check immediately before
mutation, recreates the empty schemas under their canonical names, and marks
them ready only after schema and zero-count verification. Any populated
collection, old populated manifest, or newly inserted object fails before
mutation. Keep the API stopped if a collection remains `rebuilding`; rerunning
`rebuild` safely completes an interrupted empty rebuild.

## Provision the Granite checkpoint

Install the Modal CLI separately from backend runtime dependencies:

```bash
python -m pip install -r deployment/requirements.txt
modal setup
python scripts/create_granite_manifest.py merged-granite-4.1-3b-query-rewrite
modal volume create rag-granite-models
modal volume put rag-granite-models \
  merged-granite-4.1-3b-query-rewrite \
  /merged-granite-4.1-3b-query-rewrite
modal secret create rag-sglang-secrets \
  SGLANG_QUERY_REWRITE_API_KEY="REPLACE_WITH_A_RANDOM_SECRET"
```

The service verifies the manifest and refuses to start when any required model
or tokenizer file is missing or has changed. Runtime Hugging Face downloads and
remote model code are disabled.

## Provision the Qwen checkpoint

The local `qwen3-4b-awq` directory is an external deployment artifact and is
ignored by Git. Create its manifest before uploading it to its dedicated Modal
Volume:

```bash
python scripts/create_qwen_manifest.py qwen3-4b-awq
modal volume create rag-qwen-models
modal volume put rag-qwen-models qwen3-4b-awq /qwen3-4b-awq
modal secret create rag-qwen-sglang-secrets \
  QWEN_SGLANG_API_KEY="REPLACE_WITH_A_DIFFERENT_RANDOM_SECRET"
```

Startup verifies the manifest, `Qwen3ForCausalLM` architecture, FP16 metadata,
AWQ GEMM 4-bit weights with group size 128, and the native thinking-aware chat
template. The worker loads only the provisioned local files and never enables
remote model code or downloads.

## Deploy both workers

Start with L40S in the same compute and routing region as the FastAPI deployment:

```bash
MODAL_SGLANG_GPU=L40S \
MODAL_SGLANG_COMPUTE_REGION=us-east \
MODAL_SGLANG_ROUTING_REGION=us-east \
modal deploy deployment/modal_sglang.py

QWEN_MODAL_SGLANG_GPU=L40S \
QWEN_MODAL_SGLANG_COMPUTE_REGION=us-east \
QWEN_MODAL_SGLANG_ROUTING_REGION=us-east \
modal deploy deployment/modal_qwen_sglang.py
```

Configure the API composition with:

```text
QUERY_REWRITE_ENGINE=sglang
SGLANG_QUERY_REWRITE_BASE_URL=https://YOUR-MODAL-SERVER/v1
SGLANG_QUERY_REWRITE_API_KEY=the-same-secret
SGLANG_QUERY_REWRITE_MODEL=merged-granite-4.1-3b-query-rewrite
QWEN_MODEL_PATH=qwen3-4b-awq
QWEN_SGLANG_BASE_URL=https://YOUR-QWEN-MODAL-SERVER/v1
QWEN_SGLANG_API_KEY=the-qwen-secret
QWEN_SGLANG_SERVED_MODEL=qwen3-4b-awq
```

At the deployment composition root, construct exactly one pooled
`SGLangQwenLLMClient`, call `create_query_rewriter()`, wrap the Qwen client with
`RoleRoutingLLMClient`, and inject that router into the unchanged `RAGRuntime`.
The router sends only query-rewrite completion to Granite; title completion and
all answer streams go to Qwen. The composition owns closing the Qwen sync and
async HTTP clients. Keep the FastAPI deployment at one process and one replica
because its mappings and queue remain process-local. This repository still has
no production ASGI composition module; deployment is blocked until that
deployment-owned module supplies this wiring.

The server keeps one GPU replica warm, targets four concurrent inputs, accepts
at most eight running rewrites per replica, and may scale to four replicas.
Override these settings only after running the workload benchmark.

The Qwen worker separately keeps one L40S replica warm, accepts at most eight
running generations per replica, and supports a native context length of
32,768 tokens without YaRN. It starts with `--reasoning-parser qwen3`, relies on
SGLang AWQ auto-detection, and warms one non-thinking title completion and one
non-thinking answer stream before readiness. All application requests also send
`chat_template_kwargs.enable_thinking=false`; reasoning output is rejected.

## Validate and select a GPU

Run normal offline tests first, then the opt-in Modal contract test and benchmark:

```bash
RUN_MODAL_SGLANG_TESTS=1 pytest tests/test_sglang_query_rewriter_integration.py
RUN_QWEN_SGLANG_TESTS=1 pytest tests/test_sglang_qwen_integration.py
python scripts/benchmark_sglang_query_rewriter.py \
  --concurrency 1,4,8 \
  --input-tokens 128,512,1024,2048 \
  --baseline-p95-ms 2033.349
```

The report includes application-visible server TTFT, total rewrite latency,
throughput, cached-token ratio, JSON validity, estimated cost per 1,000
rewrites, and a paired prefilled/unprefilled comparison of latency and generated
tokens. Read GPU utilization and peak memory from the enabled SGLang Prometheus
metrics during the same run; retain both the benchmark JSON and metrics snapshot
with the rollout record.

Use L40S if warm rewrite p95 is at most 500 ms and the full RAG TTFT p95 remains
at most two seconds. Otherwise repeat on H100 and select H100 only if it passes.
GPU memory snapshots remain disabled until a separate cold-start benchmark proves
they are correct and beneficial.

## Rollback

`QUERY_REWRITE_ENGINE=transformers` is an explicit redeploy-only rollback. Its
existing adapter now requires CUDA/FP16. There is no automatic per-request retry,
original-query fallback, CPU fallback, MPS path, or hidden model substitution.
