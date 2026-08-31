# Configuration & Model Specifications

This document defines all model selections, hyperparameter configurations, token budgets, and prompt definitions for the Smart RAG Interview Preparation System.

Stage 5 retains at most `TASK_MAX_COMPLETED_RECORDS` completed in-memory task
records (default `10000`). This operational setting does not affect queued or
running work and is not a durable task store.

---

## 1. System Configuration Hierarchy

```text
LLM
├── Answer: qwen3-4b-awq (SGLang / Modal CUDA, non-thinking)
├── Rewrite: merged-granite-4.1-3b-query-rewrite (SGLang / Modal CUDA)
└── Title: qwen3-4b-awq (SGLang / Modal CUDA, non-thinking)

Embedding
└── Alibaba-NLP/gte-modernbert-base (local ONNX Runtime CUDA/FP16)

Reranker
└── BAAI/bge-reranker-v2-m3 (local ONNX Runtime CUDA/FP16)

Hybrid
├── Dense + BM25
├── relativeScoreFusion
└── alpha = 0.70

Conversation
├── candidate = 20
├── MMR
├── lambda = 0.70
└── final = 5

Knowledge
├── candidate = 30
├── cross-encoder
└── final = 8

Policy
├── candidate = 20
├── cross-encoder
└── final = 5

Chunking
├── paragraph threshold = 0.72
├── chunk threshold = 0.76
├── target = 220 tokens
├── min = 80
└── max = 320

Prompts
├── P1 Query Rewriter
├── P2 Final Answer
└── P3 Session Title

Generation
├── qwen3-4b-awq
├── enable_thinking = false
├── max output = 1800
└── SSE

Context
├── Knowledge ≤ 4000 tokens
├── Policy ≤ 1500 tokens
└── Total ≤ 6000 tokens
```

---

## 2. Detailed Specifications

### 2.1 Model Registry

| Role | Target Model | Provider / Engine | Purpose |
|---|---|---|---|
| **Primary Generator (Answer)** | `qwen3-4b-awq` | SGLang 0.5.18 / Modal NVIDIA CUDA | Non-thinking answer synthesis using retrieved Knowledge Facts and Policy guidelines. |
| **Query Rewriter (Model A)** | `merged-granite-4.1-3b-query-rewrite` | SGLang 0.5.18 / Modal NVIDIA CUDA | IBM Granite 4.1-3B with the standard query-rewrite LoRA permanently merged. Produces a standalone query from structured dialogue history. |
| **Session Title Generator** | `qwen3-4b-awq` | SGLang 0.5.18 / Modal NVIDIA CUDA | Non-thinking asynchronous summarization of chat sessions for UI sidebar display. Uses the same served model identity as the primary generator. |
| **Embedding Model** | `Alibaba-NLP/gte-modernbert-base` | Local ONNX Runtime, CUDA/FP16 | 768-dimensional CLS-pooled, float32 L2-normalized vectors for chunks and full Q&A pairs. |
| **Reranker (Cross-Encoder)** | `BAAI/bge-reranker-v2-m3` | Local ONNX Runtime, CUDA/FP16 | Batched query/document pair scoring for Knowledge Facts and Policy results. |

---

### 2.2 Vector Search & Retrieval Parameters

#### Hybrid Search Configuration
- **Components**: Dense Vector (`Alibaba-NLP/gte-modernbert-base`) + Sparse BM25
- **Fusion Method**: `relativeScoreFusion`
- **Dense Weight (`alpha`)**: `0.70` (70% Vector Similarity, 30% BM25 Sparse Keyword Score)

#### Per-Collection Retrieval Settings

| Collection | Stage 1 Candidate (Top-K) | Reranking Algorithm | Reranker / Strategy Config | Stage 2 Final (Top-K) |
|---|---|---|---|---|
| **Conversation** | `20` | **MMR** (Maximal Marginal Relevance) | $\lambda = 0.70$ (Balance between relevance & diversity) | `5` |
| **Knowledge Facts** | `30` | **Cross-Encoder** | local `BAAI/bge-reranker-v2-m3` | `8` |
| **Policy** | `20` | **Cross-Encoder** | local `BAAI/bge-reranker-v2-m3` | `5` |

#### Local ONNX configuration

The embedding checkpoint is pinned to revision
`e7f32e3c00f91d699e8c43b53106206bcc72bb22`; the reranker source is pinned to
revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Both tokenizer and ONNX
artifacts are provisioned before deployment and verified through a SHA-256
manifest. Runtime downloads and remote model code are disabled.
`CUDAExecutionProvider` is the production default. A deployment may explicitly
select another ONNX Runtime provider that is installed and can execute the FP16
artifact. Only the selected provider is registered, runtime fallback is
disabled, and there is no automatic CPU or paid-API fallback. CUDA `device_id`
settings apply only when CUDA is selected.

| Environment variable | Default |
|---|---|
| `ONNX_EMBEDDING_MODEL_PATH` | `models/gte-modernbert-base` |
| `ONNX_EMBEDDING_FILENAME` | `onnx/model_fp16.onnx` |
| `ONNX_EMBEDDING_MANIFEST_FILENAME` | `onnx-manifest.json` |
| `ONNX_EMBEDDING_MAX_TOKENS` | `8192` |
| `ONNX_EMBEDDING_BATCH_SIZE` | `32` |
| `ONNX_EMBEDDING_EXECUTION_PROVIDER` | `CUDAExecutionProvider` |
| `ONNX_EMBEDDING_CUDA_DEVICE_ID` | `0` |
| `ONNX_EMBEDDING_OUTPUT_NAME` | `last_hidden_state` |
| `ONNX_EMBEDDING_DISABLE_CPU_FALLBACK` | `true` |
| `ONNX_RERANKER_MODEL_PATH` | `models/bge-reranker-v2-m3-onnx` |
| `ONNX_RERANKER_FILENAME` | `model_fp16.onnx` |
| `ONNX_RERANKER_MANIFEST_FILENAME` | `onnx-manifest.json` |
| `ONNX_RERANKER_MAX_TOKENS` | `512` |
| `ONNX_RERANKER_BATCH_SIZE` | `16` |
| `ONNX_RERANKER_EXECUTION_PROVIDER` | `CUDAExecutionProvider` |
| `ONNX_RERANKER_CUDA_DEVICE_ID` | `0` |
| `ONNX_RERANKER_OUTPUT_NAME` | `logits` |
| `ONNX_RERANKER_DISABLE_CPU_FALLBACK` | `true` |

Every ready Weaviate collection carries vector profile
`gte-modernbert-base-e7f32e3-fp16-cls-l2-768-v1`. A missing, stale, or
`rebuilding` profile fails schema validation. This prevents 1,536-dimensional
legacy vectors from being queried or mixed with the new 768-dimensional index.

---

### 2.3 Text Processing & Chunking Parameters

| Parameter | Value | Description |
|---|---|---|
| **Paragraph Splitting Threshold** | `0.72` | Embedding cosine similarity drop boundary for Semantic Paragraph Splitting. |
| **Intra-Paragraph Chunk Threshold** | `0.76` | Cosine similarity threshold for grouping consecutive sentences within a paragraph. |
| **Target Chunk Size** | `220 tokens` | Desired token length for each generated semantic chunk. |
| **Min Chunk Size** | `80 tokens` | Minimum allowed chunk size before merging with adjacent sentence group. |
| **Max Chunk Size** | `320 tokens` | Hard ceiling on chunk length to prevent embedding information dilution. |

---

### 2.4 Generation & Token Budgeting

#### Generation Settings
- **Model**: `qwen3-4b-awq`
- **Thinking Mode**: disabled (`chat_template_kwargs.enable_thinking=false`)
- **Max Output Tokens**: `1800`
- **Streaming Protocol**: Server-Sent Events (`SSE`)

#### Qwen/SGLang configuration

| Environment variable | Default |
|---|---|
| `QWEN_MODEL_PATH` | `qwen3-4b-awq` |
| `QWEN_SGLANG_BASE_URL` | `http://127.0.0.1:30001/v1` |
| `QWEN_SGLANG_API_KEY` | empty locally; required in Modal |
| `QWEN_SGLANG_SERVED_MODEL` | `qwen3-4b-awq` |
| `QWEN_SGLANG_CONNECT_TIMEOUT_SECONDS` | `1.0` |
| `QWEN_SGLANG_READ_TIMEOUT_SECONDS` | `120.0` |
| `QWEN_SGLANG_MAX_CONNECTIONS` | `32` |
| `QWEN_ANSWER_MAX_OUTPUT_TOKENS` | `1800` |
| `QWEN_TITLE_MAX_OUTPUT_TOKENS` | `32` |
| `QWEN_TEMPERATURE` | `0.7` |
| `QWEN_TOP_P` | `0.8` |
| `QWEN_TOP_K` | `20` |
| `QWEN_MIN_P` | `0.0` |
| `QWEN_PRESENCE_PENALTY` | `1.5` |

The Qwen checkpoint is an external AWQ 4-bit deployment artifact. The worker
loads it once from local storage, permits no runtime downloads, and has no
remote-provider or alternate-model fallback. P2 and P3 are each sent as one
Qwen `user` message. All calls explicitly disable thinking.

#### Token Budget Allocation
- **Knowledge Facts Context Budget**: $\le 4,000\text{ tokens}$
- **Policy Guidelines Context Budget**: $\le 1,500\text{ tokens}$
- **Total Combined Prompt Context Ceiling**: $\le 6,000\text{ tokens}$

---

### 2.5 Prompt Specifications

#### P1: Query Rewriter (Model A)
- **Role**: The local merged Granite model analyzes the user's latest question together with MMR-selected canonical Q&A pairs.
- **Input contract**: Alternating `user`/`assistant` chat turns ending in the latest `user` query. The generic P1 instruction text is retained for provider compatibility but is not sent to this fine-tuned adapter.
- **Output contract**: `{"rewritten_question":"..."}`. Generation begins after the configured `{"rewritten_question":"` assistant-response prefill, and only the parsed field value becomes the official query when strict parsing succeeds.
- **Rule**: The rewritten query becomes the official query executed across the Knowledge Facts and Policy retrieval stages, as well as grounding for the final answer.

#### Granite/SGLang configuration

| Environment variable | Default |
|---|---|
| `QUERY_REWRITE_ENGINE` | `sglang` |
| `GRANITE_QUERY_REWRITE_MODEL_PATH` | `merged-granite-4.1-3b-query-rewrite` |
| `GRANITE_QUERY_REWRITE_DEVICE` | `cuda` (explicit Transformers rollback only) |
| `GRANITE_QUERY_REWRITE_DTYPE` | `float16` |
| `GRANITE_QUERY_REWRITE_MAX_INPUT_TOKENS` | `2048` |
| `GRANITE_QUERY_REWRITE_MAX_NEW_TOKENS` | `128` |
| `GRANITE_QUERY_REWRITE_RESPONSE_PREFILL` | `{"rewritten_question":"` |
| `GRANITE_QUERY_REWRITE_WARMUP` | `true` |
| `SGLANG_QUERY_REWRITE_BASE_URL` | `http://127.0.0.1:30000/v1` |
| `SGLANG_QUERY_REWRITE_API_KEY` | empty locally; required in Modal |
| `SGLANG_QUERY_REWRITE_MODEL` | query-rewriter model identifier |
| `SGLANG_QUERY_REWRITE_CONNECT_TIMEOUT_SECONDS` | `1.0` |
| `SGLANG_QUERY_REWRITE_READ_TIMEOUT_SECONDS` | `2.0` |
| `SGLANG_QUERY_REWRITE_MAX_CONNECTIONS` | `32` |
| `SGLANG_QUERY_REWRITE_CONSTRAINED_OUTPUT` | `true` |

The checkpoint is English-only, is not downloaded at runtime, and has no CPU,
MPS, or automatic provider fallback. SGLang receives an assistant response
prefill with `continue_final_message=true`; XGrammar constrains only the
continuation that completes the JSON value. Fully rendered chat-template tokens
plus the prefill must fit the input limit. Lowest-ranked whole Q&A pairs are
dropped from the tail when necessary; the current query is never truncated. The task contract follows
[IBM's Granite Query Rewrite model card](https://huggingface.co/ibm-granite/granitelib-rag-r1.0/blob/main/query_rewrite/README.md).

#### P2: Final Answer Generation
- **Role**: Non-thinking `qwen3-4b-awq` synthesizing the final response through SGLang.
- **Inputs**:
  - Official user query (rewritten query)
  - Retrieved Knowledge Facts chunks (Top-8, within 4,000 tokens)
  - Retrieved Policy guidelines chunks (Top-5, within 1,500 tokens)
- **Delivery**: Streamed token-by-token over SSE.

#### P3: Session Title Generator
- **Role**: Background task using non-thinking `qwen3-4b-awq` through SGLang, reading all completed conversations within the active session.
- **Configuration**: The title and answer roles share `QWEN_SGLANG_SERVED_MODEL`. Title completion uses the configured 32-token ceiling; P3 and output validation continue to enforce the 3–6 word contract.
- **Output**: A concise 3–6 word title describing the session theme for the sidebar.

---

## 3. Authoritative Stage 5 Chat Timing Telemetry

Successful `/api/chat/query` streams emit one `telemetry` SSE event after all
answer tokens and before the final `done` event. The event data uses this exact
versioned JSON shape. This is the authoritative Stage 5 telemetry contract:

```json
{
  "schema_version": "1.0",
  "request_id": "uuid",
  "timings_ms": {
    "original_query_embedding": 0.0,
    "conversation_hybrid_search": 0.0,
    "conversation_mmr_rerank": 0.0,
    "query_rewrite": 0.0,
    "rewritten_query_embedding": 0.0,
    "knowledge_hybrid_search": 0.0,
    "knowledge_cross_encoder_rerank": 0.0,
    "policy_hybrid_search": 0.0,
    "policy_cross_encoder_rerank": 0.0,
    "prompt_construction": 0.0,
    "ttft": 0.0,
    "generation": 0.0,
    "total_request": 0.0
  }
}
```

All values are non-negative milliseconds rounded to three decimal places.
`ttft` measures request start through the first answer token. `generation`
measures the first token through successful stream completion. Failed or
cancelled streams emit neither telemetry nor a `done` event.
