# Design Decisions

Architectural rationale, non-functional requirements, and future considerations for the Smart RAG Interview Preparation System.

---

## 1. Key Design Decisions and Rationale

### Why Only Weaviate?

This project uses a single Weaviate instance as both the source of truth and the embedding store. There is no separate relational database, no Redis cache, no secondary storage. Weaviate handles all data persistence, vector indexing, and retrieval. This keeps the architecture simple — one system to deploy, query, and maintain.

### Why Three Separate Collections?

Interview preparation involves fundamentally different types of information:
- **Conversations** are experiential — they capture how the user practices and what the AI has previously answered.
- **Knowledge Facts** are declarative — technical content, definitions, algorithms.
- **Policies** are prescriptive — rules, strategies, evaluation frameworks.

Separating them allows independent retrieval strategies (MMR for diverse conversation recall, cross-encoder for precise fact/policy lookup), independent top-k tuning, and zero risk of cross-contamination during CRUD operations.

### Why Per-User Named Collections Instead of Shared Collections with Filtering?

Each user gets three physically separate Weaviate collections rather than sharing global collections with a `user_id` filter. Physical names use `RagUser_{unpadded_base32_utf8_user_id}_{collection_suffix}` so every validated user ID maps reversibly to a GraphQL-compatible namespace without collision-prone sanitization. This ensures retrieval queries only ever touch the current user's data at the storage level. A shared collection with post-retrieval filtering would risk a scenario where one user's overwhelming volume of data dominates the top-k results before filtering, leaving the actual user with poor or empty retrieval. Per-user named collections eliminate this class of failure entirely.

### Stage 2 Mapping Durability

Session, document, and paragraph mappings are intentionally in-memory for the
single-process Stage 2 prototype. They are not restart-safe and must not be
shared across independent backend workers. Before production deployment, a
durable design consistent with Weaviate as the source of truth must be approved;
that change may require revisiting the current three-collection data model.

### Why Cascade-Delete Sessions?

When a user deletes a chat session, all conversation embeddings belonging to that session are permanently removed from Weaviate. This ensures the conversation collection stays clean and relevant — stale or unwanted practice sessions don't pollute future retrieval results.

### Why Embed at the Chunk Level, Not the Paragraph or Document Level?

Large text blocks produce low-resolution embeddings that dilute important details. Chunk-level embeddings ensure that each vector represents a focused semantic unit, improving retrieval precision. The paragraph and document hierarchy exists only for lifecycle management (grouping, editing, deleting).

### Why Flat Retrieval Within Each Collection?

Even though chunks are organized under paragraphs and documents, retrieval treats all chunks as equals. This prevents the system from biasing results toward certain documents and ensures that the best matching content surfaces regardless of its structural location.

### Why Rewrite the Query Using Conversation History?

Past conversations contain valuable context about the user's preparation focus, knowledge gaps, and conversation patterns. By feeding relevant past conversations to a rewriting model, the system produces a query that captures implicit intent — leading to more targeted knowledge and policy retrieval.

### Why Server-Sent Events for Streaming?

SSE provides a lightweight, HTTP-native streaming mechanism that is simpler to implement than WebSockets for unidirectional server-to-client data flow. Since the LLM generates tokens sequentially and the client only needs to receive them, SSE is the ideal transport.

### Why Optimistic UI with a Background Queue?

Every button action (create, delete, save, etc.) immediately updates the UI to reflect the expected outcome, while the actual backend work is dispatched to a per-user background queue. This ensures the interface feels instant and responsive regardless of how long Weaviate operations take. Per-user FIFO queues serialize operations within the same user's data to prevent race conditions (e.g., a save and a delete on the same wizard), while different users' queues run in parallel.

---

## 2. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Latency** | Chat responses must begin streaming within 2 seconds of query submission. Wizard save operations should complete within 5 seconds for typical document sizes. |
| **Scalability** | The system should support hundreds of concurrent users, each with independent data. Weaviate collection isolation ensures no user contention. |
| **Data Privacy** | All user data is scoped by `user_id`. No cross-user data access is permitted at any layer. |
| **Consistency** | All mutating operations (wizard create, delete, save, session delete, conversation embedding) use eventual consistency via the background task queue. The UI reflects the expected state immediately; Weaviate is updated asynchronously. Retrieval results may briefly lag behind the latest mutation until the queue processes it. |
| **Reliability** | Wizard delete operations must be atomic — either all associated data is removed or none is. Partial deletes are not acceptable. |
| **File Support** | Upload supports text-only formats: `.txt`, `.md`, `.csv`, `.json`, `.xml`, `.log`, and similar plain text files. Binary formats (PDF, DOCX) are out of scope for the initial version. |

---

## 3. Future Considerations

- **PDF and DOCX Support:** Add text extraction from binary document formats via libraries like PyMuPDF or python-docx.
- **Multi-Language Embedding:** Support non-English interview preparation with multilingual embedding models.
- **Collaborative Wizards:** Allow sharing wizard content between users for study group preparation.
- **Evaluation Mode:** Timed mock interviews with scoring rubrics pulled from the Policy collection.
- **Export and Backup:** Allow users to export their knowledge bases and conversation history.
- **Fine-Tuned Models:** Train domain-specific embedding and reranking models on GenAI interview content for improved retrieval quality.
