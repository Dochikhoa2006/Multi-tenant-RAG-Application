# Smart RAG Interview Preparation System

Full-stack RAG application for Generative AI Engineer FAANG interview coaching. Backed by Weaviate vector database with three isolated collections (Conversation, Knowledge Facts, Policy) per user.

Production query rewriting uses the merged Granite 4.1-3B checkpoint through an
always-warm SGLang/Modal CUDA service. GPT-5.1 answer and title calls remain
direct provider requests. See [deployment/README.md](./deployment/README.md).

## Documentation

| Document | What It Covers |
|---|---|
| [PROJECT_OVERVIEW.md](./docs/PROJECT_OVERVIEW.md) | Product vision, terminology, directory structure, architecture diagram |
| [BACKEND.md](./docs/BACKEND.md) | Weaviate data model, retrieval pipeline, wizard CRUD, text processing |
| [FRONTEND.md](./docs/FRONTEND.md) | UI layout, chat mode, knowledge facts mode, policy mode |
| [API.md](./docs/API.md) | REST endpoint reference (chat, knowledge facts, policy wizards) |
| [CONFIG_SPECS.md](./docs/CONFIG_SPECS.md) | Models, hyperparameters, search top-k, chunking, and token budgets |
| [DESIGN_DECISIONS.md](./docs/DESIGN_DECISIONS.md) | Architectural rationale, non-functional requirements, future roadmap |
