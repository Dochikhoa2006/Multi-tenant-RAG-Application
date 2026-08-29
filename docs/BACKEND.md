# Backend Specification

All backend logic for the Smart RAG Interview Preparation System. Covers Weaviate data model, retrieval pipeline, wizard CRUD operations, and text processing pipeline.

---

## 1. Weaviate Data Model

### 1.1 Design Principles

1. **Full Collection Isolation.** The three collections per user are completely independent. No retrieval, chunking, update, or delete operation on one collection may read from, write to, or affect another.

2. **Flat Chunk Equality.** Although metadata carries a parent–child hierarchy (document → paragraph → chunk), all chunk embeddings within a collection are treated as equals during retrieval. A chunk from Document A, Paragraph 2 competes on identical footing with a chunk from Document B, Paragraph 7. The hierarchy exists solely for lifecycle management (e.g., deleting a wizard removes all its descendant chunks).

3. **Per-User Named Collections.** Each user gets three physically separate Weaviate collections, named using the convention `{user_id}_conversations`, `{user_id}_knowledge_facts`, and `{user_id}_policy`. Retrieval queries target only the current user's collections — there is no retrieve-then-filter pattern where overwhelming data from other users could dominate top-k results before filtering. One user's collections are invisible to and unreachable by another user.

### 1.2 Collection Schemas (Weaviate — Chunk-Level Only)

Only chunk-level records have their own Weaviate schema. Parent entities (session, document/wizard, paragraph) are lightweight mapping structures that track their child IDs — they do not have independent Weaviate schemas. This keeps the system clean and ensures cascading deletes work by following parent→child ID mappings.

#### Conversation Collection

Each record is one complete Q&A exchange embedded as a single vector.

| Field | Type | Description |
|---|---|---|
| `user_id` | string | Owner of this record. |
| `conversation_id` | string (UUID) | Unique identifier for the Q&A pair. Also serves as the embedding unit ID — since each conversation is exactly one Q&A pair embedded as a single vector, no separate `chunk_id` is needed. |
| `raw_text` | text | The concatenated question + answer text. |
| `vector` | float[] | Dense embedding of the raw text. |

> **Note:** Conversation records have no `document_id`, `paragraph_id`, `chunk_id`, or `session_id`. Each conversation is a standalone embedding. `conversation_id` alone identifies both the record and its embedding. Session grouping is handled outside Weaviate via parent-child mappings (see Section 1.3).

#### Knowledge Facts Collection

Each record is one chunk within a paragraph within a wizard document.

| Field | Type | Description |
|---|---|---|
| `user_id` | string | Owner of this record. |
| `document_id` | string (UUID) | The wizard this chunk belongs to. |
| `paragraph_id` | integer | Sequential paragraph index within the document (top-to-bottom, starting at 1). |
| `chunk_id` | string (UUID) | Unique identifier for this chunk. |
| `raw_text` | text | The original text of this chunk. |
| `vector` | float[] | Dense embedding of the chunk text. |

#### Policy Collection

Identical schema to Knowledge Facts Collection. Stored and queried in a completely separate collection namespace.

#### Collection Naming Convention

Collections are created dynamically per user using the pattern:

| Collection Type | Naming Pattern | Example |
|---|---|---|
| Conversation | `{user_id}_conversations` | `usr_abc123_conversations` |
| Knowledge Facts | `{user_id}_knowledge_facts` | `usr_abc123_knowledge_facts` |
| Policy | `{user_id}_policy` | `usr_abc123_policy` |

Each user's collections are physically separate Weaviate collections. Retrieval queries target only the specific user's collection — no post-retrieval filtering is needed.

### 1.3 Parent-Child Mapping Structure (Non-Weaviate)

Parent entities do not have independent Weaviate schemas. They are lightweight mappings that track their child IDs, enabling lifecycle operations like cascading deletes. These mappings live in the application layer (e.g., in-memory, a relational database, or a key-value store — not in Weaviate).

```
Session
  └── tracks: [conversation_id, conversation_id, ...]

Document (Wizard)
  └── tracks: [paragraph_id, paragraph_id, ...]

Paragraph
  └── tracks: [chunk_id, chunk_id, ...]
```

**Cascading Delete Examples:**
- **Delete a session** → look up all `conversation_id`s it tracks → delete those conversation records from Weaviate → remove the session from the sidebar.
- **Delete a wizard** → look up all `paragraph_id`s it tracks → for each paragraph, look up all `chunk_id`s → delete those chunk records from Weaviate.
- **Re-embed (save)** → identify modified `paragraph_id`s → look up their old `chunk_id`s → delete old chunks from Weaviate → insert new chunks → update the paragraph→chunk mappings.

---

## 2. Retrieval Pipeline (Shared Across All Collections)

All three collections follow the same two-stage retrieval procedure. Only the reranking strategy and top-k values differ.

```
                    ┌──────────────┐
                    │  User Query  │
                    └──────┬───────┘
                           │
                           ▼
              ┌────────────────────────┐
              │     Hybrid Search      │
              │  (Dense + Sparse BM25) │
              └────────────┬───────────┘
                           │
                     Top-K results
                           │
                           ▼
              ┌────────────────────────┐
              │   Second-Stage Fusion  │
              │                        │
              │  Conversation → MMR    │
              │  Knowledge   → Cross-  │
              │  Policy      → Encoder │
              └────────────┬───────────┘
                           │
                   Reranked Top-K
                           │
                           ▼
              ┌────────────────────────┐
              │   Return to Caller     │
              └────────────────────────┘
```

**Stage 1 — Hybrid Search:** Weaviate executes a combined dense vector similarity search (`text-embedding-3-small`) and sparse BM25 keyword search, fused via `relativeScoreFusion` with `alpha = 0.70` (70% dense, 30% BM25).

**Stage 2 — Reranking:**
- **Conversation Collection:** Apply Maximal Marginal Relevance (MMR) with $\lambda = 0.70$ to promote diversity among retrieved past conversations and reduce redundancy.
- **Knowledge Facts & Policy Collections:** Apply a cross-encoder reranker (`Cohere rerank-v4.0-fast`) for higher precision on factual/policy content where accuracy matters more than diversity.

**Top-K Configuration:**

| Collection | Hybrid Candidate (Stage 1) | Reranking Strategy | Stage 2 Final (Top-K) |
|---|---|---|---|
| Conversation | `20` | MMR ($\lambda = 0.70$) | `5` |
| Knowledge Facts | `30` | Cross-Encoder (`Cohere rerank-v4.0-fast`) | `8` |
| Policy | `20` | Cross-Encoder (`Cohere rerank-v4.0-fast`) | `5` |

*(See [CONFIG_SPECS.md](./CONFIG_SPECS.md) for full configuration details.)*

---

## 3. User Query Walkthrough (RAG Answer Generation)

The full pipeline for answering a user question in chat mode:

```
Step 1:  User submits a question.

Step 2:  Retrieve from the Conversation Collection.
         → Hybrid search + MMR reranking.
         → Returns the most relevant past Q&A pairs.

Step 3:  Pass retrieved conversations + original question to Model A
         (a lightweight LLM or prompt-engineered call).
         → Model A extracts the most relevant conversation context
           and produces a REWRITTEN QUERY that is more specific (rewritten query becomes official user query in the remaining pipeline of RAG)
           and contextually enriched.

Step 4:  Use the rewritten query to retrieve from Knowledge Facts Collection.
         → Hybrid search + cross-encoder reranking.

Step 5:  Use the rewritten query to retrieve from Policy Collection.
         → Hybrid search + cross-encoder reranking.

Step 6:  Compose final prompt:
           - Original user question
           - Retrieved knowledge facts
           - Retrieved policy guidelines
         → Send to the primary LLM.
         → Stream the response back to the frontend via Server-Sent Events (SSE).

Step 7:  After the full answer is delivered to the UI,
         the backend asynchronously embeds the Q&A pair
         into the Conversation Collection as a background task.
         (The user does not wait for this operation.)
```

---

## 4. Wizard CRUD Operations

### 4.1 Create Wizard (`new wizard` function)

**Trigger:** User clicks "Create New Wizard" in Knowledge Facts or Policy mode.

**Behavior:**
1. Generate a new `document_id` (UUID).
2. Initialize the wizard with one paragraph (`paragraph_id = 1`) containing empty text.
3. No embeddings are created yet (empty text produces no meaningful chunks).
4. Return the new wizard to the frontend for display.

### 4.2 Upload Text-Only File (`scan and assign` function)

**Trigger:** User clicks the "Upload File" button inside a specific wizard's edit view.

**Behavior:**
1. Accept one or more text-only files (`.txt`, `.md`, `.csv`, `.json`, etc.).
2. Extract all plain text content from every uploaded file.
3. Concatenate and append the extracted text to the end of the wizard's existing plain text.
4. Mark the final paragraph of the current wizard as **modified** (since new text was appended to it or after it).
5. The wizard UI now displays the combined text. The user sees and can freely edit all content, regardless of its origin (typed or uploaded). The system does not retain original file boundaries.
6. The **Save** button becomes active because content has changed.

> **Important:** Files are not stored. The system copies their text content into the wizard's editable region and discards the original files.

### 4.3 Cancel (`discard local state` function)

**Trigger:** User clicks "Cancel" while editing a wizard.

**Behavior:**
1. Discard all unsaved local changes in the frontend.
2. Revert the wizard's displayed text to the last saved state.
3. No backend calls are made. Nothing changes in Weaviate.

### 4.4 Save (`re-embed` function)

**Trigger:** User clicks "Save" after modifying text or uploading files into a wizard. The Save button is only enabled when the wizard content has actually changed (e.g., typing a character then deleting it returns the state to unchanged — Save remains disabled).

**Behavior (step by step):**

```
Step 1:  IDENTIFY MODIFIED PARAGRAPHS
         Compare the current wizard text against the last saved state.
         Mark every paragraph ID whose content has changed.
         If text was uploaded, the final paragraph (where text was appended)
         is automatically marked.

Step 2:  MERGE CONSECUTIVE MARKED PARAGRAPHS INTO UNIONS
         Group adjacent marked paragraph IDs into contiguous unions.
         Example:
           Marked IDs = {1, 2, 3, 6, 7, 9}
           → Union A = [1, 2, 3]
           → Union B = [6, 7]
           → Union C = [9]

Step 3:  DELETE OLD EMBEDDINGS
         For each union from Step 2, permanently delete all Weaviate chunk
         records associated with the old paragraph IDs in those unions.
         All old chunks must be removed before any re-splitting, re-chunking,
         or re-embedding occurs.

Step 4:  RE-SPLIT EACH UNION
         For each union, take the combined raw text of those paragraphs
         and run Semantic Paragraph Splitting to determine new paragraph
         boundaries.

Step 5:  RE-CHUNK AND GENERATE EMBEDDINGS
         For each newly split paragraph:
           a. Chunk the paragraph into small, semantically coherent
              sentence groups.
           b. Generate a vector embedding for each chunk.
           c. Map each chunk to its raw text, paragraph ID, and
              the parent document ID.
         New chunks are staged in memory — not yet inserted into Weaviate.

Step 6:  RENUMBER ALL PARAGRAPH IDs
         After splitting, some unions may produce paragraph IDs that
         collide with existing (unmodified) paragraph IDs in the
         same document.
         Renumber ALL paragraph IDs (both modified and unmodified) to be:
           - Distinct within the document.
           - Sorted in ascending order.
           - Ordered from top of wizard text to bottom.

Step 7:  INSERT NEW EMBEDDINGS AND UPDATE EXISTING RECORDS
         a. Write the newly created chunk records (with final paragraph IDs)
            into Weaviate.
         b. Update the paragraph_id metadata on any unmodified chunks
            whose paragraph IDs changed during renumbering in Step 6.

Step 8:  CONFIRM SAVE
         Update the stored "last saved state" so future change detection
         compares against this new version.
```

### 4.5 Delete Wizard (`wizard delete` function)

**Trigger:** User clicks the "Delete" button on a wizard card.

**Behavior:**
1. Permanently delete all Weaviate records where `document_id` matches the wizard being deleted.
2. This removes every paragraph and every chunk embedding associated with the wizard.
3. The deletion is permanent and irreversible. There is no soft-delete or recycle bin.
4. Remove the wizard from the frontend display.

---

## 5. Conversation Embedding (Background)

After a Q&A exchange is fully rendered in the chat UI:

1. The backend receives the complete question + answer text.
2. Concatenate them into a single string.
3. Generate a vector embedding for the concatenated text.
4. Create a new record in the Conversation Collection with a new `conversation_id`.
5. This operation runs asynchronously. The user continues interacting with the chat without delay.

---

## 6. Text Processing Pipeline

### 6.1 Semantic Paragraph Splitting

**Input:** A block of raw text (from a wizard or uploaded file).

**Output:** An ordered list of paragraphs, each representing a semantically coherent section of the input.

**Method:** Use an embedding-based approach:
1. Split the text into individual sentences.
2. Generate embeddings for each sentence.
3. Compute similarity between consecutive sentences.
4. Identify significant drops in similarity as paragraph boundaries.
5. Group sentences between boundaries into paragraphs.

This produces semantically meaningful paragraph divisions that respect topic shifts rather than relying on superficial formatting cues (like double newlines).

### 6.2 Chunking

**Input:** A single paragraph.

**Output:** A list of chunks, each a small group of semantically related sentences within the paragraph.

**Method:**
1. Split the paragraph into sentences.
2. Group consecutive sentences that share high semantic similarity.
3. Each group becomes one chunk.
4. Chunks are the smallest unit and the only level at which vector embeddings are generated.

### 6.3 Embedding

**Input:** A chunk's raw text (or for conversations, the full Q&A text).

**Output:** A dense vector embedding stored in Weaviate.

**Model:** `text-embedding-3-small` (OpenAI API). The same model must be used for both indexing and query-time embedding to ensure compatible vector spaces. (See [CONFIG_SPECS.md](./CONFIG_SPECS.md)).

---

## 7. Background Task Queue

The frontend uses an optimistic UI pattern — every button action immediately updates the UI, and the backend processes the actual work asynchronously via a background task queue. This section specifies how the backend manages that queue.

### 7.1 Design

- Each mutating API request returns an immediate success response (e.g., the newly generated `document_id`) without waiting for Weaviate operations to complete.
- The actual work (Weaviate writes, deletes, re-embedding) is dispatched to a **per-user FIFO task queue** processed in the background.
- Per-user queues ensure that operations on the same user's data are serialized (no race conditions between a save and a delete on the same wizard), while operations across different users run in parallel.

### 7.2 Queued Operations

| Operation | What the API Returns Immediately | What the Queue Processes |
|---|---|---|
| **Create Wizard** | New `document_id` | Create Weaviate collection entries (if needed), initialize document and paragraph mappings. |
| **Delete Wizard** | Confirmation | Delete all Weaviate chunk records for the `document_id`, remove document and paragraph mappings. |
| **Save Wizard** | Confirmation | Execute the full 8-step re-embed pipeline (Section 4.4). |
| **Delete Session** | Confirmation | Look up all `conversation_id`s in the session, delete their Weaviate records, remove session mapping. |
| **Embed Conversation** | N/A (triggered internally after SSE completes) | Concatenate Q+A, generate embedding, insert into Conversation Collection. |
| **Generate Session Title** | N/A (triggered internally after SSE completes) | Call LLM with session conversations, update session title in mapping. |

### 7.3 Error Handling

- If a queued task fails, log the error with full context (user_id, operation type, affected IDs).
- Expose a task status endpoint so the frontend can check whether a background operation succeeded or failed.
- On failure, the frontend displays a non-blocking toast notification with a retry option.
- Failed tasks do **not** automatically retry — the user must explicitly trigger a retry.

