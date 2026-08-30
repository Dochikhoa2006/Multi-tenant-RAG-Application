# Frontend Specification

All frontend UI design for the Smart RAG Interview Preparation System. Covers application layout, three modes (Chat, Knowledge Facts, Policy), and component behavior.

---

## 1. Application Layout

The frontend is a single-page application with three primary modes accessible via a top-level navigation or tab bar.

```
┌──────────────────────────────────────────────────────────┐
│  [ Chat ]    [ Knowledge Facts ]    [ Policy ]           │
├───────────┬──────────────────────────────────────────────┤
│           │                                              │
│  Sidebar  │              Main Content Area               │
│           │                                              │
│  (Chat    │  (Changes based on active mode)              │
│   history │                                              │
│   list)   │                                              │
│           │                                              │
│           │                                              │
│           │                                              │
│           │                                              │
│           │                                              │
└───────────┴──────────────────────────────────────────────┘
```

---

## 2. Chat Mode

**Purpose:** Conversational interface for interview Q&A practice.

**UI Elements:**
- **Sidebar:** Scrollable list of past chat sessions, each showing an auto-generated title. Each session has a **Delete** button (with confirmation dialog) that permanently removes the session and cascade-deletes all its conversation embeddings from Weaviate.
- **New Chat Button:** Creates a fresh session with an empty conversation history.
- **Message Area:** Displays the conversation as alternating user questions and AI answers, rendered with rich text/markdown support.
- **Input Bar:** Text input field with a send button at the bottom.

**Session Title Generation:**
- After each Q&A exchange completes, a background LLM call reads all conversations in the current session and generates a brief, descriptive title.
- This title is displayed on the sidebar session tag.
- This operation is purely cosmetic. It does not affect backend retrieval because each conversation is embedded independently of its session.

**Answer Streaming:**
- Responses arrive via Server-Sent Events (SSE), rendering token-by-token in real time.

---

## 3. Knowledge Facts Mode

**Purpose:** Manage the user's technical knowledge base — concepts, algorithms, papers, frameworks, and any factual content relevant to interview preparation.

**UI Elements:**

**Default View (Wizard Gallery):**
- A grid or list of wizard cards, each displaying:
  - A preview or title of the wizard's content.
  - A **Delete** button (with confirmation dialog).
- A prominent **"Create New Wizard"** button.
- Wizards are displayed in an orderly, visually attractive layout with clean spacing and subtle animations on hover/creation.

**Edit View (Wizard Zoom-In):**
- Clicking a wizard card transitions smoothly into an expanded edit view (zoom animation).
- The edit view contains:
  - A large, editable text area showing the wizard's full plain text content.
  - An **"Upload Text-Only File"** button — accepts `.txt`, `.md`, and similar text formats. Uploaded text is appended to the end of the editor.
  - A **"Cancel"** button — always available. Discards local changes and returns to the gallery view.
  - A **"Save"** button — **disabled by default**. Becomes enabled only when the text content has actually changed from the last saved state. Smart change detection ensures that typing and then fully undoing the edit does not enable Save.
- The text area supports free-form editing. Users can type, paste, rearrange, or delete any portion of the text.

---

## 4. Policy Mode

**Purpose:** Manage policy documents — behavioral guidelines, interview rubrics, evaluation criteria, do's and don'ts, and strategic frameworks.

**UI and Behavior:** Identical to Knowledge Facts Mode in every way. The only difference is the underlying Weaviate collection (Policy Collection instead of Knowledge Facts Collection).

---

## 5. Optimistic UI & Background Queue

**Principle:** Every button action immediately updates the UI to reflect the expected outcome. The actual backend work is dispatched to a background queue. The user never waits for backend processing to complete before seeing the result.

**Per-Operation Behavior:**

| Operation | Immediate UI Response | Background Queue Work |
|---|---|---|
| **Create Wizard** | New wizard card appears after the synchronous `201` response. | Backend generates `document_id` and initializes empty process-local document/paragraph mappings; no Weaviate or embedding work occurs. |
| **Delete Wizard** | Wizard card is removed from the gallery immediately. | Backend deletes all Weaviate chunk records and mapping entries for that `document_id`. |
| **Save Wizard** | Editor shows "Saved" state, Save button disables, baseline text updates to current. | Backend runs the full 8-step re-embed pipeline (delete old chunks → re-split → re-chunk → re-embed → renumber → insert). |
| **Upload File** | Nonempty uploaded text appears appended in the editor textarea and Save enables. Empty uploads show an error and leave the editor unchanged. | No backend work at this point — backend processes only when Save is clicked. |
| **Delete Session** | Show deletion as pending after `202`; retain the session when the API returns `409` for an active stream. | Backend reserves the session, cascade-deletes its conversation embeddings, and removes it only after verified success. |
| **New Chat** | Fresh empty chat panel appears, new session added to sidebar. | Backend creates session mapping record. |
| **Send Message** | User message bubble appears instantly. AI response streams in via SSE. | After streaming completes, the Q&A pair is embedded into the Conversation Collection in the background. |

**Error Handling:**
- If a background queue task fails, the UI displays a **non-blocking toast notification** describing the failure.
- The toast offers a **Retry** action that resubmits the original operation;
  task IDs do not have a generic retry endpoint.
- The UI does **not** automatically roll back — the user is informed and decides how to proceed.
- Failed tasks are logged for debugging.
- Error responses contain a stable public code, safe message, and request ID.
  Provider credentials, storage details, and raw exception messages are never
  shown. Support/debug tooling correlates the request or task ID with internal
  logs.
