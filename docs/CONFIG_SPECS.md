# Configuration & Model Specifications

This document defines all model selections, hyperparameter configurations, token budgets, and prompt definitions for the Smart RAG Interview Preparation System.

Stage 5 retains at most `TASK_MAX_COMPLETED_RECORDS` completed in-memory task
records (default `10000`). This operational setting does not affect queued or
running work and is not a durable task store.

---

## 1. System Configuration Hierarchy

```text
LLM
├── Answer: GPT-5.1
├── Rewrite: merged-granite-4.1-3b-query-rewrite (SGLang / Modal CUDA)
└── Title: GPT-5.1

Embedding
└── text-embedding-3-small

Reranker
└── Cohere rerank-v4.0-fast

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
├── GPT-5.1
├── reasoning = low
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
| **Primary Generator (Answer)** | `GPT-5.1` | OpenAI API | Answer synthesis using retrieved Knowledge Facts and Policy guidelines. |
| **Query Rewriter (Model A)** | `merged-granite-4.1-3b-query-rewrite` | SGLang 0.5.18 / Modal NVIDIA CUDA | IBM Granite 4.1-3B with the standard query-rewrite LoRA permanently merged. Produces a standalone query from structured dialogue history. |
| **Session Title Generator** | `GPT-5.1` | OpenAI API | Asynchronous summarization of chat sessions for UI sidebar display. Defaults to the primary generator model while retaining an independent environment override. |
| **Embedding Model** | `text-embedding-3-small` | OpenAI API | Vector embeddings for chunks (Knowledge, Policy) and full Q&A pairs (Conversation). |
| **Reranker (Cross-Encoder)** | `rerank-v4.0-fast` | Cohere API | Second-stage cross-encoder reranking for Knowledge Facts and Policy results. |

---

### 2.2 Vector Search & Retrieval Parameters

#### Hybrid Search Configuration
- **Components**: Dense Vector (`text-embedding-3-small`) + Sparse BM25
- **Fusion Method**: `relativeScoreFusion`
- **Dense Weight (`alpha`)**: `0.70` (70% Vector Similarity, 30% BM25 Sparse Keyword Score)

#### Per-Collection Retrieval Settings

| Collection | Stage 1 Candidate (Top-K) | Reranking Algorithm | Reranker / Strategy Config | Stage 2 Final (Top-K) |
|---|---|---|---|---|
| **Conversation** | `20` | **MMR** (Maximal Marginal Relevance) | $\lambda = 0.70$ (Balance between relevance & diversity) | `5` |
| **Knowledge Facts** | `30` | **Cross-Encoder** | `Cohere rerank-v4.0-fast` | `8` |
| **Policy** | `20` | **Cross-Encoder** | `Cohere rerank-v4.0-fast` | `5` |

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
- **Model**: `GPT-5.1`
- **Reasoning Effort**: `low`
- **Max Output Tokens**: `1800`
- **Streaming Protocol**: Server-Sent Events (`SSE`)

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
- **Role**: Primary LLM (`GPT-5.1`, reasoning = `low`) synthesizing the final response.
- **Inputs**:
  - Official user query (rewritten query)
  - Retrieved Knowledge Facts chunks (Top-8, within 4,000 tokens)
  - Retrieved Policy guidelines chunks (Top-5, within 1,500 tokens)
- **Delivery**: Streamed token-by-token over SSE.

#### P3: Session Title Generator
- **Role**: Background task using `GPT-5.1` reading all completed conversations within the active session.
- **Configuration**: `SESSION_TITLE_MODEL` remains independently overridable and otherwise defaults to `PRIMARY_GENERATOR_MODEL`. Title completion passes only the model selection; P3 and output validation enforce the 3–6 word contract.
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
