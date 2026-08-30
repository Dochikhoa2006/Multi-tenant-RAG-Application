# Modal SGLang Deployment

The production query-rewrite path is a dedicated SGLang 0.5.18 CUDA service.
GPT-5.1 answer streaming and title completion remain direct provider calls from
the existing `RoleRoutingLLMClient`; they are not proxied through SGLang.

## Provision the checkpoint

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

## Deploy

Start with L40S in the same compute and routing region as the FastAPI deployment:

```bash
MODAL_SGLANG_GPU=L40S \
MODAL_SGLANG_COMPUTE_REGION=us-east \
MODAL_SGLANG_ROUTING_REGION=us-east \
modal deploy deployment/modal_sglang.py
```

Configure the API composition with:

```text
QUERY_REWRITE_ENGINE=sglang
SGLANG_QUERY_REWRITE_BASE_URL=https://YOUR-MODAL-SERVER/v1
SGLANG_QUERY_REWRITE_API_KEY=the-same-secret
SGLANG_QUERY_REWRITE_MODEL=merged-granite-4.1-3b-query-rewrite
```

At the deployment composition root, call `create_query_rewriter()`, wrap the
existing direct-provider LLM with `RoleRoutingLLMClient`, and inject that router
into the unchanged `RAGRuntime`. Keep the FastAPI deployment at one process and
one replica because its mappings and queue remain process-local.

The server keeps one GPU replica warm, targets four concurrent inputs, accepts
at most eight running rewrites per replica, and may scale to four replicas.
Override these settings only after running the workload benchmark.

## Validate and select a GPU

Run normal offline tests first, then the opt-in Modal contract test and benchmark:

```bash
RUN_MODAL_SGLANG_TESTS=1 pytest tests/test_sglang_query_rewriter_integration.py
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
