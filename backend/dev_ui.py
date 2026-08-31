"""Minimal same-origin development UI for exercising the real chat API."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Development E2E</title>
  <style>
    body { font: 16px/1.45 system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
    label { display: block; margin-top: 1rem; }
    input, select, textarea, button { box-sizing: border-box; font: inherit; padding: .55rem; }
    input, select, textarea { width: 100%; }
    textarea { min-height: 8rem; }
    button { margin: .75rem .5rem 0 0; }
    pre { background: #f4f4f4; padding: 1rem; white-space: pre-wrap; overflow-wrap: anywhere; }
    #answer { min-height: 7rem; }
  </style>
</head>
<body>
  <h1>RAG Development E2E</h1>
  <p>This page calls the real same-origin API and consumes its SSE response.</p>
  <label>User ID <input id="user" value="usr_dev"></label>
  <label>Session <select id="session"></select></label>
  <button id="refresh" type="button">Refresh Sessions</button>
  <button id="create" type="button">Create Session</button>
  <label>Question <textarea id="question">What should I study next?</textarea></label>
  <button id="submit" type="button">Submit</button>
  <h2>Answer</h2><pre id="answer"></pre>
  <h2>Status</h2><pre id="status"></pre>
  <h2>Telemetry</h2><pre id="telemetry"></pre>
<script>
const user = document.querySelector('#user');
const session = document.querySelector('#session');
const question = document.querySelector('#question');
const answer = document.querySelector('#answer');
const statusBox = document.querySelector('#status');
const telemetry = document.querySelector('#telemetry');
const submitButton = document.querySelector('#submit');
let requestInFlight = false;

async function jsonResponse(response) {
  const payload = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(payload));
  return payload;
}

async function refreshSessions(selectedId = '') {
  const response = await fetch(`/api/chat/sessions?user_id=${encodeURIComponent(user.value)}`);
  const items = await jsonResponse(response);
  session.replaceChildren(...items.map(item => {
    const option = document.createElement('option');
    option.value = item.session_id;
    option.textContent = `${item.title} (${item.conversation_count})`;
    option.selected = item.session_id === selectedId;
    return option;
  }));
}

async function createSession() {
  const response = await fetch('/api/chat/sessions', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user_id: user.value})
  });
  const created = await jsonResponse(response);
  await refreshSessions(created.session_id);
}

function dispatchEvent(name, payload, state) {
  if (name === 'token') answer.textContent += payload.text;
  else if (name === 'telemetry') telemetry.textContent = JSON.stringify(payload, null, 2);
  else statusBox.textContent = `${name}: ${JSON.stringify(payload, null, 2)}`;
  if (name === 'done' || name === 'error') state.terminalEvent = name;
}

function consumeFrames(state, text, flush = false) {
  state.buffer += text.replace(/\r\n/g, '\n');
  const frames = state.buffer.split('\n\n');
  state.buffer = flush ? '' : frames.pop();
  for (const frame of frames) {
    if (!frame.trim()) continue;
    let name = 'message';
    const data = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) name = line.slice(6).trim();
      if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
    }
    dispatchEvent(name, JSON.parse(data.join('\n')), state);
  }
}

async function submitQuestion() {
  if (requestInFlight) return;
  if (!session.value) throw new Error('Create or select a session first.');
  requestInFlight = true;
  submitButton.disabled = true;
  try {
    answer.textContent = ''; telemetry.textContent = ''; statusBox.textContent = 'streaming';
    const response = await fetch('/api/chat/query', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({user_id: user.value, session_id: session.value, question: question.value})
    });
    if (!response.ok || !response.body) throw new Error(await response.text());
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const state = {buffer: '', terminalEvent: null};
    while (!state.terminalEvent) {
      const {value, done} = await reader.read();
      if (done) {
        consumeFrames(state, decoder.decode(), true);
        break;
      }
      consumeFrames(state, decoder.decode(value, {stream: true}));
    }
    if (!state.terminalEvent) throw new Error('SSE stream ended without done or error.');
    if (state.terminalEvent === 'done') await refreshSessions(session.value);
    await reader.cancel();
  } finally {
    requestInFlight = false;
    submitButton.disabled = false;
  }
}

async function report(action) {
  try { await action(); }
  catch (error) { statusBox.textContent = `error: ${error.message}`; }
}
document.querySelector('#refresh').onclick = () => report(() => refreshSessions());
document.querySelector('#create').onclick = () => report(createSession);
submitButton.onclick = () => report(submitQuestion);
report(() => refreshSessions());
</script>
</body>
</html>
"""


def install_dev_ui(application: FastAPI) -> None:
    """Install the development-only browser harness on an integrated app."""

    @application.get("/dev/e2e", response_class=HTMLResponse, include_in_schema=False)
    async def development_e2e() -> HTMLResponse:
        return HTMLResponse(_PAGE)


__all__ = ["install_dev_ui"]
