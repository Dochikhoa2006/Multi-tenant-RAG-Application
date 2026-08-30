# API Endpoints

REST API reference for the Smart RAG Interview Preparation System.

---

## 1. Chat

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat/query` | Submit a user question. Returns streamed SSE response. |
| POST | `/api/chat/sessions` | Create a new chat session. |
| GET | `/api/chat/sessions` | List all sessions for the current user. |
| GET | `/api/chat/sessions/{session_id}` | Get all conversations in a session. |
| PATCH | `/api/chat/sessions/{session_id}/title` | Update a session's auto-generated title. |
| DELETE | `/api/chat/sessions/{session_id}` | Delete a session and cascade-delete all its conversation embeddings from Weaviate. |

Successful chat streams emit SSE events in the fixed order
`token* → telemetry → done`. Failed streams emit any tokens already delivered
followed by one safe `error` event and never emit telemetry or `done`.
Conversation embedding and title generation are scheduled only after complete
generation. Session deletion is queued and removes the process-local session
only after a verbose deletion plus dry-run check confirms that no mapped
conversation objects remain; a partial failure retains the session for retry.
An active response stream causes deletion to return `409 SESSION_ACTIVE`.
Once deletion is reserved, new queries return
`409 SESSION_DELETION_IN_PROGRESS` until deletion succeeds or the reservation
is released after failure.

---

## 2. Knowledge Facts Wizards

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/knowledge/wizards` | Create a new wizard (empty document). |
| GET | `/api/knowledge/wizards` | List all wizards for the current user. |
| GET | `/api/knowledge/wizards/{wizard_id}` | Get the full text content of a wizard. |
| PUT | `/api/knowledge/wizards/{wizard_id}` | Save modified wizard text (triggers re-embed pipeline). |
| DELETE | `/api/knowledge/wizards/{wizard_id}` | Permanently delete a wizard and all its embeddings. |
| POST | `/api/knowledge/wizards/{wizard_id}/upload` | Upload text-only files to append to wizard content. |

Wizard creation is synchronous because Stage 3 creates only the process-local
empty document and paragraph mappings; it performs no embedding or Weaviate
mutation. The save request's `modified_paragraph_ids` array is optional because
Stage 3 derives changes from `current_text`; supplied IDs remain conservative
hints. Upload accepts existing modified IDs as repeated multipart fields or one
JSON-array field. A nonempty upload preserves those IDs and adds the final saved
paragraph ID because content was appended. If every decoded file is empty,
upload returns `422 EMPTY_UPLOAD` and changes nothing; whitespace-only content
is preserved and considered nonempty.

---

## 3. Policy Wizards

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/policy/wizards` | Create a new policy wizard. |
| GET | `/api/policy/wizards` | List all policy wizards. |
| GET | `/api/policy/wizards/{wizard_id}` | Get policy wizard content. |
| PUT | `/api/policy/wizards/{wizard_id}` | Save modified policy wizard (triggers re-embed). |
| DELETE | `/api/policy/wizards/{wizard_id}` | Permanently delete a policy wizard. |
| POST | `/api/policy/wizards/{wizard_id}/upload` | Upload text files to append to policy wizard. |

---

## 4. Background Tasks

Wizard saves, wizard deletes, and session cascade deletes return HTTP `202`
with a tracked task resource. Task state is process-local in the prototype.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/tasks/{task_id}?user_id={user_id}` | Read `queued`, `running`, `succeeded`, or `failed` state for a task owned by the user. A wrong-user lookup returns `404`. |

Task resources include `task_id`, `user_id`, `operation`, `status`, optional
safe `error_code`/`error` fields, and created/started/finished timestamps. Raw
provider and storage exceptions are logged internally but never returned.
Public failures direct the caller to retry the original API operation; task IDs
cannot be generically retried. Tasks run FIFO for each user; different users may
run concurrently. Completed task history is capped by
`TASK_MAX_COMPLETED_RECORDS` (default `10000`), with oldest completed records
evicted first; queued/running work is never evicted. Task state is not durable
across a backend restart.

## 5. Errors, Upload Limits, and CORS

HTTP failures use `detail: {code, message, request_id}`. SSE failures use the
same three public fields directly. `X-Request-ID` carries the corresponding ID
on every HTTP response. Public messages contain no provider or storage details.

Uploads default to 10 MiB per file and 25 MiB total per request. Deployments can
override `UPLOAD_MAX_FILE_BYTES`, `UPLOAD_MAX_TOTAL_BYTES`, and
`UPLOAD_READ_CHUNK_BYTES`. Files remain temporary and uploading never updates a
wizard mapping, creates embeddings, writes Weaviate, or enqueues work.

Development CORS defaults to `*`. Production deployments set the comma-separated
`CORS_ALLOWED_ORIGINS` environment variable to the exact permitted origins.

## 6. Production Runtime Composition

`backend.main:app` is the provider-neutral bootstrap and intentionally has no
concrete GPT-5.1 adapter. It starts without loading local model artifacts;
provider-dependent endpoints return safe `503` responses until a runtime is
injected.

No production ASGI composition module currently exists in this repository.
Deployment must supply one that constructs the concrete `LLMClient`, exactly
one shared `ONNXEmbeddingClient`, exactly one shared
`ONNXCrossEncoderReranker`, one shared `WeaviateManager` and
`InMemoryTaskQueue`, wraps the provider LLM in `RoleRoutingLLMClient` with one
shared `SGLangGraniteQueryRewriter`, then creates `RAGRuntime`, `AppServices`,
and `create_app(services)`. Until that module is supplied, production provider
wiring—including shared ONNX client reuse—cannot be verified. The provider
adapters and credentials remain deployment dependencies outside Stage 5;
`backend.main:app` is intentionally providerless.

Provision the merged checkpoint and its SHA-256 manifest in the Modal model
Volume before deploying SGLang. The directory must contain `config.json`,
`model.safetensors.index.json`, both Safetensors shards, tokenizer, and chat
template. Runtime downloads and remote code are disabled. The FastAPI deployment
uses a pooled authenticated SGLang connection in the same Modal region while
GPT-5.1 remains direct. Importing `backend.main:app` does not contact SGLang or
load model weights. See `deployment/README.md` for provisioning and rollout.
