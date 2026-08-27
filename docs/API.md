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
