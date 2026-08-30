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
├── Rewrite: GPT-5 mini
└── Title: GPT-5 mini

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
| **Query Rewriter (Model A)** | `GPT-5 mini` | OpenAI API | Fast contextual analysis of conversation history to produce an enriched query. |
| **Session Title Generator** | `GPT-5 mini` | OpenAI API | Asynchronous summarization of chat sessions for UI sidebar display. |
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
- **Role**: Model A analyzes the user's latest question together with MMR-selected past conversation context.
- **Output**: An explicit, disambiguated, context-enriched query.
- **Rule**: The rewritten query becomes the official query executed across the Knowledge Facts and Policy retrieval stages, as well as grounding for the final answer.

#### P2: Final Answer Generation
- **Role**: Primary LLM (`GPT-5.1`, reasoning = `low`) synthesizing the final response.
- **Inputs**:
  - Official user query (rewritten query)
  - Retrieved Knowledge Facts chunks (Top-8, within 4,000 tokens)
  - Retrieved Policy guidelines chunks (Top-5, within 1,500 tokens)
- **Delivery**: Streamed token-by-token over SSE.

#### P3: Session Title Generator
- **Role**: Background task using `GPT-5 mini` reading all completed conversations within the active session.
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
