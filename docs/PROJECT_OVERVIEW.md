# Smart RAG Interview Preparation System
## Full-Stack Application for Generative AI Engineer FAANG Interview Coaching

---

## Product Vision

A retrieval-augmented generation (RAG) application that helps users prepare for Generative AI Engineer interviews at FAANG and big-tech companies. The system provides an intelligent chatbot backed by user-curated knowledge bases and policy guidelines, with every interaction growing smarter through embedded conversation history.

Users maintain two structured knowledge stores — **Knowledge Facts** (technical content, papers, concepts) and **Policies** (behavioral guidelines, interview strategies, rubrics) — organized into editable units called **Wizards**. A conversational interface leverages all three data sources to deliver context-rich, interview-grade answers.

---

## Core Concepts and Terminology

| Term | Definition |
|---|---|
| **User** | An authenticated individual identified by a unique `user_id`. Each user owns an isolated data environment. |
| **Collection** | One of three physically separate Weaviate collections per user: **Conversation**, **Knowledge Facts**, or **Policy**. Named using `{user_id}_conversations`, `{user_id}_knowledge_facts`, `{user_id}_policy`. |
| **Wizard** | A user-facing content board mapped one-to-one to a **Document ID**. Users view, edit, upload into, and delete wizards through the UI. |
| **Document ID** | The internal identifier for a wizard. Each wizard is exactly one document. |
| **Paragraph ID** | A segment boundary produced by **Semantic Paragraph Splitting** within a document. Paragraphs are numbered sequentially from top to bottom of the wizard text. |
| **Chunk ID** | The atomic embedding unit for knowledge facts and policy collections. A chunk is a small, semantically coherent sentence group within a paragraph. Vector embeddings are generated at this level — never at the paragraph or document level. Conversations do not use chunk IDs (see Conversation ID). |
| **Conversation ID** | A single question-answer pair treated as one embedding unit. Serves as both the record identifier and the embedding unit ID — no separate chunk ID is needed. Unlike knowledge/policy data, conversations skip paragraph, document, and chunk hierarchy. |
| **Session** | A UI-level grouping of multiple conversations within chat mode. Sessions have auto-generated titles but do not affect backend retrieval. |

---

## Project Directory Structure

```
RAG Application/
├── backend/
│   ├── main.py                      # App entrypoint, server startup
│   ├── config.py                    # Environment variables, constants
│   ├── model_config.py              # LLM and embedding model settings
│   ├── prompts.py                   # All prompt templates (Model A rewrite, answer generation, session title)
│   ├── api/
│   │   ├── chat.py                  # Chat endpoints (query, sessions)
│   │   ├── knowledge.py             # Knowledge Facts wizard endpoints
│   │   └── policy.py                # Policy wizard endpoints
│   ├── rag/
│   │   ├── retrieval.py             # Hybrid search + reranking (MMR, cross-encoder)
│   │   ├── query_rewriter.py        # Model A: conversation-aware query rewriting
│   │   ├── generator.py             # Final LLM answer generation (SSE streaming)
│   │   └── embedder.py              # Background embedding (conversation, chunk)
│   ├── processing/
│   │   ├── paragraph_splitter.py    # Semantic Paragraph Splitting
│   │   ├── chunker.py               # Intra-paragraph semantic chunking
│   │   └── file_reader.py           # Text-only file extraction
│   ├── weaviate_client/
│   │   ├── client.py                # Weaviate connection and collection init
│   │   ├── conversation.py          # Conversation collection CRUD
│   │   ├── knowledge.py             # Knowledge Facts collection CRUD
│   │   └── policy.py                # Policy collection CRUD
│   ├── mappings/
│   │   ├── session_map.py           # Session → conversation IDs
│   │   ├── document_map.py          # Document (wizard) → paragraph IDs + raw text
│   │   └── paragraph_map.py         # Paragraph → chunk IDs
│   └── wizard/
│       ├── crud.py                  # Create, delete wizard logic
│       └── save.py                  # Re-embed pipeline (diff, merge, re-split, re-chunk)
├── frontend/
│   ├── index.html
│   ├── index.css
│   ├── app.js                       # App shell, mode switching, routing
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatView.js          # Message area, input bar
│   │   │   ├── SessionList.js       # Sidebar session list
│   │   │   └── StreamRenderer.js    # SSE token-by-token rendering
│   │   ├── wizard/
│   │   │   ├── WizardGallery.js     # Grid of wizard cards
│   │   │   ├── WizardEditor.js      # Zoom-in edit view (text area, upload, save, cancel)
│   │   │   └── WizardCard.js        # Single wizard card with delete button
│   │   └── shared/
│   │       ├── ModeNav.js           # Top-level mode tabs (Chat, Knowledge Facts, Policy)
│   │       └── ConfirmDialog.js     # Delete confirmation modal
│   ├── services/
│   │   ├── api.js                   # HTTP client for backend endpoints
│   │   └── sse.js                   # SSE connection handler
│   └── utils/
│       └── changeDetector.js        # Smart text diff (enables/disables Save button)
└── README.md
```

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (SPA)                           │
│                                                                 │
│  ┌─────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │  Chat Mode  │   │ Knowledge Facts  │   │   Policy Mode   │  │
│  │             │   │     Mode         │   │                 │  │
│  │ - Sessions  │   │ - Wizard Cards   │   │ - Wizard Cards  │  │
│  │ - Q&A Flow  │   │ - Inline Editor  │   │ - Inline Editor │  │
│  │ - New Chat  │   │ - File Upload    │   │ - File Upload   │  │
│  └──────┬──────┘   └────────┬─────────┘   └────────┬────────┘  │
│         │                   │                       │           │
└─────────┼───────────────────┼───────────────────────┼───────────┘
          │                   │                       │
          ▼                   ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (API Server)                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  RAG Orchestration Layer                  │   │
│  │                                                          │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐  │   │
│  │  │ Query      │  │ Retrieval    │  │ Generation       │  │   │
│  │  │ Rewriting  │  │ Pipeline     │  │ (SSE Streaming)  │  │   │
│  │  └────────────┘  └──────────────┘  └──────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                CRUD + Embedding Pipeline                  │   │
│  │                                                          │   │
│  │  Semantic Paragraph Splitting → Chunking → Embedding     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     WEAVIATE (Vector DB)                        │
│                                                                 │
│    Per User ID:                                                 │
│    ┌─────────────────┐ ┌───────────────┐ ┌───────────────────┐  │
│    │  Conversation   │ │ Knowledge     │ │     Policy        │  │
│    │  Collection     │ │ Facts         │ │     Collection    │  │
│    │                 │ │ Collection    │ │                   │  │
│    │ conversation_id │ │ document_id   │ │ document_id       │  │
│    │ vector          │ │ paragraph_id  │ │ paragraph_id      │  │
│    │ raw_text        │ │ chunk_id      │ │ chunk_id          │  │
│    │                 │ │ vector        │ │ vector            │  │
│    │                 │ │ raw_text      │ │ raw_text          │  │
│    └─────────────────┘ └───────────────┘ └───────────────────┘  │
│                                                                 │
│    Collections are FULLY ISOLATED — no cross-collection         │
│    interference during retrieval, chunking, or mutation.        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
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
