# Modal-first RAG Deployment

The no-GCP E2E topology uses three Modal services. Two dedicated SGLang 0.5.18
CUDA services run Granite and Qwen, while a private singleton Modal Server runs
the integrated FastAPI, GTE ONNX, and BGE ONNX runtime. Granite handles only
query rewriting. The separate `qwen3-4b-awq` worker handles non-thinking answer
streaming and title completion through the existing `RoleRoutingLLMClient`.
Weaviate runs outside Modal so mutable database files are never placed on a
Modal model Volume.

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

Stage 1 segmentation separately loads `all-MiniLM-L6-v2` from the required
`SEGMENTATION_MODEL_PATH` with `local_files_only=True`. Its explicit
`SEGMENTATION_EMBEDDING_DEVICE` defaults to `cpu`; no Hugging Face download is
allowed during API startup. The Docker profile mounts the provisioned directory
read-only at `/opt/models/all-MiniLM-L6-v2`.

Production must construct exactly one `ONNXEmbeddingClient` and one
`ONNXCrossEncoderReranker` in its ASGI composition root, inject both into the
existing `RAGRuntime`, and reuse that embedding client through the wizard
adapter. `backend.runtime_app:create_runtime_app` now owns that wiring and its
lifespan. `backend.main:app` remains providerless and import-safe for tests that
inject their own services.

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
```

Startup verifies the manifest, `Qwen3ForCausalLM` architecture, FP16 metadata,
AWQ GEMM 4-bit weights with group size 128, and the native thinking-aware chat
template. The worker loads only the provisioned local files and never enables
remote model code or downloads.

## Protect and deploy both workers

Both Modal Servers are private. Create one Modal Proxy Token in the Modal
dashboard and store its ID and secret directly in the local `.env`; do not paste
them into terminals whose output is retained. Modal accepts the combined
`TOKEN_ID.TOKEN_SECRET` value as an OpenAI-compatible bearer key, so configure
both existing provider settings with that combined value:

```text
MODAL_PROXY_TOKEN_ID=wk-...
MODAL_PROXY_TOKEN_SECRET=ws-...
SGLANG_QUERY_REWRITE_API_KEY=wk-....ws-...
QWEN_SGLANG_API_KEY=wk-....ws-...
```

The token is enforced by Modal's proxy. It is not mounted in either worker and
is never passed to SGLang as `--api-key`; localhost startup warmups therefore
carry no authorization header. This prevents replacement credentials from
appearing in SGLang's serialized server arguments.

Start with L40S in the same compute and routing region as the FastAPI deployment:

```bash
MODAL_SGLANG_GPU=L40S \
MODAL_SGLANG_COMPUTE_REGION=us \
MODAL_SGLANG_ROUTING_REGION=us-east \
modal deploy deployment/modal_sglang.py

QWEN_MODAL_SGLANG_GPU=L40S \
QWEN_MODAL_SGLANG_COMPUTE_REGION=us \
QWEN_MODAL_SGLANG_ROUTING_REGION=us-east \
modal deploy deployment/modal_qwen_sglang.py
```

Configure the API composition with:

```text
QUERY_REWRITE_ENGINE=sglang
SGLANG_QUERY_REWRITE_BASE_URL=https://YOUR-MODAL-SERVER/v1
SGLANG_QUERY_REWRITE_API_KEY=TOKEN_ID.TOKEN_SECRET
SGLANG_QUERY_REWRITE_MODEL=merged-granite-4.1-3b-query-rewrite
QWEN_MODEL_PATH=qwen3-4b-awq
QWEN_SGLANG_BASE_URL=https://YOUR-QWEN-MODAL-SERVER/v1
QWEN_SGLANG_API_KEY=TOKEN_ID.TOKEN_SECRET
QWEN_SGLANG_SERVED_MODEL=qwen3-4b-awq
```

After authenticated health, model-list, Granite, and Qwen checks succeed,
delete the retired `rag-sglang-secrets` and `rag-qwen-sglang-secrets`. Never
delete them before the private replacements are verified.

### Granite integration diagnostic

The deployed Granite worker may rewrite `How do I get rid of them?` as
`How do I get rid of fleas on my dog?`. That output is equivalent and
standalone under the documented query-rewrite contract, but the opt-in
integration test also requires the literal string `Rex`. Keep that assertion
and the production prompt unchanged; record it as a stricter semantic-evaluation
failure, not a deployment or RAG-contract blocker.

## Run persistent authenticated Weaviate on the Mac

The former 14-day Weaviate Cloud Sandbox is no longer the applicable free
offering. The current managed free cluster permits only one collection, while
this application intentionally creates three physically distinct collections
per user. Do not collapse the schemas or replace them with tenants.

Use the dedicated secure Compose definition with a strong key stored only in
`.env`:

```text
WEAVIATE_CONNECTION_MODE=custom
WEAVIATE_URL=https://YOUR-MAC.YOUR-TAILNET.ts.net
WEAVIATE_API_KEY=YOUR-RANDOM-WEAVIATE-KEY
WEAVIATE_GRPC_PORT=8443
WEAVIATE_GRPC_SECURE=true
```

```bash
docker compose --env-file .env \
  -f deployment/compose.weaviate-secure.yaml up -d
```

The container binds REST and gRPC only to Mac loopback, disables anonymous
access and autoschema, grants the `rag-runtime` API-key identity administrator
access, and persists database files in `rag_weaviate_secure_data`.

After installing and signing in to Tailscale, expose the two protocols from the
same stable tailnet hostname:

```bash
sudo tailscale funnel --bg --https=443 http://127.0.0.1:8080
sudo tailscale funnel --bg --tls-terminated-tcp=8443 \
  tcp://127.0.0.1:50051
tailscale funnel status
```

The Mac must remain awake, Docker must remain running, and both Funnels must
remain active for Modal to reach Weaviate. `WEAVIATE_CONNECTION_MODE=custom`
uses the explicit REST/gRPC endpoints with optional API-key authentication;
`cloud` requires an API key and uses the official Cloud helper; `auto` preserves
the prior key-selects-cloud behavior. No connection mode enables a vectorizer,
generative module, embedded database, or alternate retrieval path.

## Provision and deploy the integrated Modal runtime

Create a dedicated Volume and upload only the already-provisioned artifacts:

```bash
modal volume create rag-runtime-models
modal volume put rag-runtime-models \
  models/gte-modernbert-base /gte-modernbert-base
modal volume put rag-runtime-models \
  models/bge-reranker-v2-m3-onnx /bge-reranker-v2-m3-onnx
modal volume put rag-runtime-models \
  models/all-MiniLM-L6-v2 /all-MiniLM-L6-v2
modal volume ls rag-runtime-models
```

Create `rag-runtime-secrets` with exactly these secret values; do not upload the
whole local `.env`:

```text
WEAVIATE_URL
WEAVIATE_API_KEY
WEAVIATE_CONNECTION_MODE
WEAVIATE_GRPC_PORT
WEAVIATE_GRPC_SECURE
SGLANG_QUERY_REWRITE_BASE_URL
SGLANG_QUERY_REWRITE_API_KEY
QWEN_SGLANG_BASE_URL
QWEN_SGLANG_API_KEY
```

Create a Modal Proxy Token for clients of the private API, then deploy:

```text
RAG_API_BASE_URL=https://YOUR-PRIVATE-MODAL-SERVER
MODAL_PROXY_TOKEN_ID=wk-...
MODAL_PROXY_TOKEN_SECRET=ws-...
```

```bash
MODAL_RAG_GPU=L40S \
MODAL_RAG_COMPUTE_REGION=us \
MODAL_RAG_ROUTING_REGION=us-east \
modal deploy deployment/modal_runtime.py
```

`deployment/modal_runtime.py` keeps exactly one container and starts exactly
one Uvicorn worker. It mounts the retrieval/segmentation Volume plus the
existing Granite Volume, preloads the pip-provisioned CUDA and cuDNN libraries,
runs `nvidia-smi`, and refuses startup unless ONNX Runtime exposes
`CUDAExecutionProvider`. Requests to its URL must carry `Modal-Key` and
`Modal-Secret` headers. Do not set `unauthenticated=True` merely to make the
development browser page convenient.

`backend.runtime_app:create_runtime_app` constructs exactly one pooled
`SGLangQwenLLMClient` and `SGLangGraniteQueryRewriter`, wraps them with
`RoleRoutingLLMClient`, and injects that router into the unchanged `RAGRuntime`.
Only query rewriting reaches Granite; title completion and answer streaming go
to Qwen. Lifespan shutdown drains the task queue before closing both HTTP client
pools and disconnecting Weaviate. Keep the FastAPI deployment at one process
and one replica because its mappings and queue remain process-local.

The API validates `/v1/models` on both external workers during startup and
fails closed when either worker is unavailable or advertises the wrong model.
Start it with:

```bash
uvicorn backend.runtime_app:create_runtime_app \
  --factory --host 0.0.0.0 --port 8000 --workers 1
```

For Docker development, copy `.env.example` to `.env`, supply the four absolute
model directories and both SGLang endpoints, then run `docker compose up
--build --wait`. Compose starts only FastAPI and Weaviate; the SGLang workers
retain their existing independent deployments.

`GET /dev/e2e` is a fake development E2E console, not the Stage 6 frontend. It
permits one active manual query, consumes the real SSE stream through `done` or
`error`, and then restores its Submit control.

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
at most two seconds. If L40S cannot obtain capacity after the image, secrets,
Volumes, region configuration, and deployment health are otherwise confirmed,
H100 may be used as an infrastructure-only substitution. Do not alter model
identity, quantization, context, prompt, tokenizer, sampling, or inference
behavior. Otherwise repeat performance validation on H100 and select it only
if it passes.
GPU memory snapshots remain disabled until a separate cold-start benchmark proves
they are correct and beneficial.

If the FastAPI container restarts after wizards or sessions are created, that
E2E attempt is invalid because those mappings and the task queue are
process-local. Use a new user ID and recreate both wizards and the session. A
restart is not an infrastructure failure and does not corrupt Weaviate.

## Rollback

The Transformers query-rewriter adapter remains a separately implemented
redeploy-only rollback, but it is not permitted for this contract-faithful E2E.
There is no automatic per-request retry, original-query fallback, CPU fallback,
MPS path, or hidden model substitution.
