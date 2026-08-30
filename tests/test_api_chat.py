from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import AsyncIterator, Sequence
from uuid import uuid4

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from backend.api.models import QueryRequest
from backend.api.telemetry import TELEMETRY_SCHEMA_VERSION, TIMING_KEYS
from backend.api.chat import delete_session as delete_session_endpoint
from backend.api.chat import query as query_endpoint
from backend.main import create_app
from backend.rag.pipeline import UserRetrievalCollections
from backend.rag.runtime import RAGRuntime, RerankResult
from backend.services import AppServices
from backend.weaviate_client.models import DeletionReport, SearchResult


USER_ID = "usr_api"


class FakeManager:
    def __init__(self) -> None:
        self.connected = 0
        self.disconnected = 0
        self.ensured: list[str] = []
        self.ensure_failure: Exception | None = None

    def connect(self) -> None:
        self.connected += 1

    def disconnect(self) -> None:
        self.disconnected += 1

    def ensure_user_collections(self, user_id: str) -> None:
        if self.ensure_failure is not None:
            raise self.ensure_failure
        self.ensured.append(user_id)


class WordTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class FakeCollection:
    def __init__(self, user_id: str, collection_type: str, text: str) -> None:
        self.user_id = user_id
        self.collection_type = collection_type
        self.text = text
        self.calls: list[tuple[str, list[float], int]] = []

    def hybrid_search(
        self, query_text: str, query_vector: Sequence[float], top_k: int
    ) -> list[SearchResult]:
        self.calls.append((query_text, list(query_vector), top_k))
        return [
            SearchResult(
                object_id=str(uuid4()),
                properties={"raw_text": self.text},
                vector=(1.0, 0.0),
                score=0.9,
            )
        ]


class FakeEmbeddings:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[str] = []

    def embed(self, text: str, *, model: str) -> Sequence[float]:
        self.calls.append(text)
        self.events.append(f"embed:{text[:12]}")
        return [1.0, 0.0]

    def embed_many(
        self, texts: Sequence[str], *, model: str
    ) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeReranker:
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        model: str,
        top_n: int,
    ) -> Sequence[RerankResult]:
        return [RerankResult(index=0, score=0.95)] if documents else []


class FakeLLM:
    def __init__(
        self,
        events: list[str],
        *,
        fail_stream: bool = False,
        chunks: Sequence[str] = ("first ", "second"),
        stream_error: str = "generation failed",
    ) -> None:
        self.events = events
        self.fail_stream = fail_stream
        self.chunks = list(chunks)
        self.stream_error = stream_error

    def complete(self, prompt: str, **kwargs: object) -> str:
        if "<conversation_list>" in prompt:
            self.events.append("title")
            return "Useful Retrieval Session"
        return "official rewritten query"

    def stream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        async def iterator() -> AsyncIterator[str]:
            for index, chunk in enumerate(self.chunks):
                yield chunk
                if self.fail_stream and index == 0:
                    raise RuntimeError(self.stream_error)

        return iterator()


class FakeConversationWriter:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.inserted: list[tuple[str, str, list[float]]] = []
        self.deleted: list[list[str]] = []
        self.stored_ids: set[str] = set()
        self.delete_failure: Exception | None = None

    def insert(
        self, conversation_id: str, raw_text: str, vector: Sequence[float]
    ) -> str:
        self.events.append("conversation_insert")
        self.inserted.append((conversation_id, raw_text, list(vector)))
        self.stored_ids.add(conversation_id)
        return conversation_id

    def delete_batch(self, conversation_ids: list[str]) -> None:
        if self.delete_failure is not None:
            raise self.delete_failure
        self.events.append("conversation_delete")
        self.deleted.append(list(conversation_ids))
        self.stored_ids.difference_update(conversation_ids)

    def delete_batch_verified(
        self, conversation_ids: list[str]
    ) -> DeletionReport:
        if self.delete_failure is not None:
            raise self.delete_failure
        self.events.append("conversation_delete")
        self.deleted.append(list(conversation_ids))
        self.stored_ids.difference_update(conversation_ids)
        ids = tuple(conversation_ids)
        return DeletionReport(len(ids), len(ids), 0, ids, ())


def _events(response_text: str) -> list[tuple[str, dict[str, object]]]:
    parsed: list[tuple[str, dict[str, object]]] = []
    for block in response_text.strip().split("\n\n"):
        lines = block.splitlines()
        parsed.append((lines[0].removeprefix("event: "), json.loads(lines[1][6:])))
    return parsed


def _services(
    *,
    fail_stream: bool = False,
    chunks: Sequence[str] = ("first ", "second"),
    stream_error: str = "generation failed",
):
    events: list[str] = []
    manager = FakeManager()
    embeddings = FakeEmbeddings(events)
    llm = FakeLLM(
        events,
        fail_stream=fail_stream,
        chunks=chunks,
        stream_error=stream_error,
    )
    writer = FakeConversationWriter(events)
    runtime = RAGRuntime(
        llm,
        embeddings,
        FakeReranker(),
        lambda user_id: writer,
        tokenizer=WordTokenizer(),
    )
    conversations = FakeCollection(USER_ID, "conversations", "previous answer")
    knowledge = FakeCollection(USER_ID, "knowledge_facts", "knowledge fact")
    policy = FakeCollection(USER_ID, "policy", "policy guidance")
    bundle = UserRetrievalCollections(
        USER_ID, conversations, knowledge, policy
    )
    services = AppServices(
        manager=manager,
        rag_runtime=runtime,
        retrieval_collections_factory=lambda user_id: bundle,
        conversation_collection_factory=lambda user_id: writer,
    )
    return services, manager, embeddings, writer, events, bundle


def _wait_for_task(client: TestClient, task_id: str) -> dict[str, object]:
    for _ in range(100):
        payload = client.get(
            f"/api/tasks/{task_id}", params={"user_id": USER_ID}
        ).json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.005)
    raise AssertionError("background task did not finish")


def test_chat_query_streams_json_sse_and_persists_only_complete_answer() -> None:
    services, manager, embeddings, writer, events, bundle = _services()
    app = create_app(services)
    with TestClient(app) as client:
        created = client.post("/api/chat/sessions", json={"user_id": USER_ID})
        assert created.status_code == 201
        session_id = created.json()["session_id"]

        response = client.post(
            "/api/chat/query",
            json={
                "user_id": USER_ID,
                "session_id": session_id,
                "question": "original question",
            },
        )
        assert response.status_code == 200
        parsed = _events(response.text)
        assert [event for event, _ in parsed] == [
            "token",
            "token",
            "telemetry",
            "done",
        ]
        assert [payload["text"] for event, payload in parsed if event == "token"] == [
            "first ",
            "second",
        ]
        telemetry = parsed[-2][1]
        assert telemetry["schema_version"] == TELEMETRY_SCHEMA_VERSION
        assert set(telemetry["timings_ms"]) == set(TIMING_KEYS)
        for key in TIMING_KEYS:
            value = telemetry["timings_ms"][key]
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            assert math.isfinite(value) and value >= 0
            assert value == round(value, 3)
        request_ids = {payload["request_id"] for _, payload in parsed}
        assert request_ids == {response.headers["x-request-id"]}

        detail = client.get(
            f"/api/chat/sessions/{session_id}", params={"user_id": USER_ID}
        ).json()
        assert detail["conversation_count"] == 1
        assert detail["conversations"][0]["answer"] == "first second"
        assert client.options(
            "/api/chat/sessions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        ).headers["access-control-allow-origin"] == "*"

    assert manager.connected == 1
    assert manager.disconnected == 1
    assert [call[0] for call in bundle.conversations.calls] == ["original question"]
    assert [call[0] for call in bundle.knowledge_facts.calls] == [
        "official rewritten query"
    ]
    assert [call[0] for call in bundle.policy.calls] == [
        "official rewritten query"
    ]
    assert len(writer.inserted) == 1
    assert "Question:\noriginal question\n\nAnswer:\nfirst second" in writer.inserted[0][1]
    assert events.index("conversation_insert") < events.index("title")
    assert len(embeddings.calls) == 3
    assert services.chat_registry.get_session(USER_ID, session_id).title == (
        "Useful Retrieval Session"
    )


def test_failed_generation_emits_error_without_partial_transcript_or_embedding() -> None:
    services, _, embeddings, writer, _, _ = _services(fail_stream=True)
    with TestClient(create_app(services)) as client:
        session_id = client.post(
            "/api/chat/sessions", json={"user_id": USER_ID}
        ).json()["session_id"]
        response = client.post(
            "/api/chat/query",
            json={
                "user_id": USER_ID,
                "session_id": session_id,
                "question": "original question",
            },
        )
        parsed = _events(response.text)
        assert [event for event, _ in parsed] == ["token", "error"]
        assert parsed[-1][1] == {
            "code": "CHAT_PROCESSING_FAILED",
            "message": "The response could not be completed.",
            "request_id": response.headers["x-request-id"],
        }
        detail = client.get(
            f"/api/chat/sessions/{session_id}", params={"user_id": USER_ID}
        ).json()
        assert detail["conversation_count"] == 0
        deletion = client.delete(
            f"/api/chat/sessions/{session_id}", params={"user_id": USER_ID}
        )
        assert deletion.status_code == 202
        assert _wait_for_task(client, deletion.json()["task_id"])["status"] == (
            "succeeded"
        )
    assert writer.inserted == []
    assert len(embeddings.calls) == 2


def test_session_crud_cascade_delete_and_task_ownership() -> None:
    services, _, _, writer, _, _ = _services()
    with TestClient(create_app(services)) as client:
        session = client.post(
            "/api/chat/sessions", json={"user_id": USER_ID}
        ).json()
        session_id = session["session_id"]
        assert client.get("/api/chat/sessions", params={"user_id": USER_ID}).json()[0][
            "title"
        ] == "New Chat"
        updated = client.patch(
            f"/api/chat/sessions/{session_id}/title",
            json={"user_id": USER_ID, "title": "Manual Interview Topic"},
        )
        assert updated.status_code == 200
        assert updated.json()["title"] == "Manual Interview Topic"

        services.chat_registry.record_conversation(
            USER_ID,
            session_id,
            str(uuid4()),
            "question",
            "answer",
        )
        queued = client.delete(
            f"/api/chat/sessions/{session_id}", params={"user_id": USER_ID}
        )
        assert queued.status_code == 202
        task_id = queued.json()["task_id"]
        assert client.get(
            f"/api/tasks/{task_id}", params={"user_id": "usr_other"}
        ).status_code == 404
        assert _wait_for_task(client, task_id)["status"] == "succeeded"
        assert client.get(
            f"/api/chat/sessions/{session_id}", params={"user_id": USER_ID}
        ).status_code == 404
    assert len(writer.deleted) == 1


def test_provider_endpoint_returns_503_without_runtime() -> None:
    services = AppServices(manager=FakeManager())
    with TestClient(create_app(services)) as client:
        session_id = client.post(
            "/api/chat/sessions", json={"user_id": USER_ID}
        ).json()["session_id"]
        response = client.post(
            "/api/chat/query",
            json={
                "user_id": USER_ID,
                "session_id": session_id,
                "question": "question",
            },
        )
        assert response.status_code == 503
        assert response.json()["detail"] == {
            "code": "SERVICE_UNAVAILABLE",
            "message": "A required service is temporarily unavailable.",
            "request_id": response.headers["x-request-id"],
        }


def test_storage_503_logs_but_does_not_expose_sensitive_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    services, manager, _, _, _, _ = _services()
    secret = "weaviate-secret-credential"
    manager.ensure_failure = RuntimeError(secret)
    with TestClient(create_app(services)) as client:
        session_id = client.post(
            "/api/chat/sessions", json={"user_id": USER_ID}
        ).json()["session_id"]
        response = client.post(
            "/api/chat/query",
            json={
                "user_id": USER_ID,
                "session_id": session_id,
                "question": "question",
            },
        )
        assert response.status_code == 503
        assert secret not in response.text
        detail = response.json()["detail"]
        assert detail["code"] == "SERVICE_UNAVAILABLE"
        assert detail["request_id"] == response.headers["x-request-id"]
    assert secret in caplog.text
    assert any(
        getattr(record, "request_id", None) == detail["request_id"]
        for record in caplog.records
    )


def test_failed_session_cascade_keeps_registry_state_and_reports_task_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    services, _, _, writer, _, _ = _services()
    writer.delete_failure = RuntimeError("delete rejected")
    with TestClient(create_app(services)) as client:
        session_id = client.post(
            "/api/chat/sessions", json={"user_id": USER_ID}
        ).json()["session_id"]
        services.chat_registry.record_conversation(
            USER_ID,
            session_id,
            str(uuid4()),
            "question",
            "answer",
        )
        queued = client.delete(
            f"/api/chat/sessions/{session_id}", params={"user_id": USER_ID}
        ).json()
        task = _wait_for_task(client, queued["task_id"])
        assert task["status"] == "failed"
        assert task["error_code"] == "TASK_FAILED"
        assert task["error"] == (
            "Background operation failed. Retry the original operation."
        )
        assert "delete rejected" not in task["error"]
        assert client.get(
            f"/api/chat/sessions/{session_id}", params={"user_id": USER_ID}
        ).json()["conversation_count"] == 1
        writer.delete_failure = None
        retry = client.delete(
            f"/api/chat/sessions/{session_id}", params={"user_id": USER_ID}
        ).json()
        assert _wait_for_task(client, retry["task_id"])["status"] == "succeeded"
        assert client.get(
            f"/api/chat/sessions/{session_id}", params={"user_id": USER_ID}
        ).status_code == 404
    assert "delete rejected" in caplog.text


def test_sse_preserves_newlines_inside_json_token() -> None:
    services, _, _, writer, _, _, = _services(chunks=("line one\nline two",))
    with TestClient(create_app(services)) as client:
        session_id = client.post(
            "/api/chat/sessions", json={"user_id": USER_ID}
        ).json()["session_id"]
        response = client.post(
            "/api/chat/query",
            json={
                "user_id": USER_ID,
                "session_id": session_id,
                "question": "question",
            },
        )
        parsed = _events(response.text)
        assert [event for event, _ in parsed] == ["token", "telemetry", "done"]
        assert parsed[0][1]["text"] == "line one\nline two"
    assert len(writer.inserted) == 1


def test_sensitive_provider_error_is_logged_but_not_exposed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "provider-secret-token-123"
    services, _, _, writer, _, _ = _services(
        fail_stream=True,
        stream_error=secret,
    )
    with TestClient(create_app(services)) as client:
        session_id = client.post(
            "/api/chat/sessions", json={"user_id": USER_ID}
        ).json()["session_id"]
        response = client.post(
            "/api/chat/query",
            json={
                "user_id": USER_ID,
                "session_id": session_id,
                "question": "question",
            },
        )
        assert secret not in response.text
        error = _events(response.text)[-1][1]
        assert error["code"] == "CHAT_PROCESSING_FAILED"
        assert error["request_id"] == response.headers["x-request-id"]
    assert secret in caplog.text
    assert any(
        getattr(record, "request_id", None) == error["request_id"]
        for record in caplog.records
    )
    assert writer.inserted == []


def test_cross_user_cannot_read_session() -> None:
    services, _, _, _, _, _ = _services()
    with TestClient(create_app(services)) as client:
        session_id = client.post(
            "/api/chat/sessions", json={"user_id": USER_ID}
        ).json()["session_id"]
        response = client.get(
            f"/api/chat/sessions/{session_id}",
            params={"user_id": "usr_other"},
        )
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "NOT_FOUND"


def test_cors_uses_configured_production_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services, _, _, _, _, _ = _services()
    monkeypatch.setattr(
        "backend.main.CORS_ALLOWED_ORIGINS",
        ("https://app.example",),
    )
    with TestClient(create_app(services)) as client:
        allowed = client.options(
            "/api/chat/sessions",
            headers={
                "Origin": "https://app.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        denied = client.options(
            "/api/chat/sessions",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert allowed.headers["access-control-allow-origin"] == (
            "https://app.example"
        )
        assert "access-control-allow-origin" not in denied.headers


def test_cancelled_response_body_creates_no_partial_state() -> None:
    async def scenario() -> None:
        services, _, _, writer, events, _ = _services()
        session_id = services.chat_registry.create_session(
            USER_ID, str(uuid4())
        ).session_id
        app = create_app(services)
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/chat/query",
                "headers": [],
                "app": app,
            }
        )
        response = await query_endpoint(
            QueryRequest(
                user_id=USER_ID,
                session_id=session_id,
                question="question",
            ),
            request,
            services,
        )
        first = await response.body_iterator.__anext__()
        assert "event: token" in first
        await response.body_iterator.aclose()
        reserved = services.chat_registry.reserve_session_deletion(
            USER_ID,
            session_id,
        )
        assert reserved == []
        services.chat_registry.abort_session_deletion(USER_ID, session_id)
        await services.task_queue.close()
        assert services.chat_registry.get_session(
            USER_ID, session_id
        ).conversations == ()
        assert writer.inserted == []
        assert "title" not in events

    asyncio.run(scenario())


def test_active_stream_blocks_delete_and_final_delete_leaves_no_orphan() -> None:
    async def scenario() -> None:
        services, _, _, writer, events, _ = _services()
        session_id = services.chat_registry.create_session(
            USER_ID, str(uuid4())
        ).session_id
        app = create_app(services)
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/chat/query",
                "headers": [],
                "app": app,
            }
        )
        response = await query_endpoint(
            QueryRequest(
                user_id=USER_ID,
                session_id=session_id,
                question="question",
            ),
            request,
            services,
        )

        with pytest.raises(HTTPException) as active_error:
            await delete_session_endpoint(
                session_id,
                request,
                USER_ID,
                services,
            )
        assert active_error.value.status_code == 409
        assert active_error.value.detail["code"] == "SESSION_ACTIVE"
        assert writer.deleted == []

        streamed = [item async for item in response.body_iterator]
        assert "event: done" in streamed[-1]
        deletion = await delete_session_endpoint(
            session_id,
            request,
            USER_ID,
            services,
        )
        result = await services.task_queue.wait(deletion.task_id, USER_ID)
        assert result.status == "succeeded"
        with pytest.raises(KeyError):
            services.chat_registry.get_session(USER_ID, session_id)
        assert writer.stored_ids == set()
        assert events.index("conversation_insert") < events.index(
            "conversation_delete"
        )
        await services.task_queue.close()

    asyncio.run(scenario())


def test_delete_reservation_blocks_new_query_but_not_another_user() -> None:
    async def scenario() -> None:
        services, _, _, _, _, _ = _services()
        session_id = services.chat_registry.create_session(
            USER_ID, str(uuid4())
        ).session_id
        other_user = "usr_other"
        other_session = services.chat_registry.create_session(
            other_user, str(uuid4())
        ).session_id
        services.chat_registry.reserve_session_deletion(USER_ID, session_id)

        app = create_app(services)
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/chat/query",
                "headers": [],
                "app": app,
            }
        )
        with pytest.raises(HTTPException) as deleting_error:
            await query_endpoint(
                QueryRequest(
                    user_id=USER_ID,
                    session_id=session_id,
                    question="question",
                ),
                request,
                services,
            )
        assert deleting_error.value.status_code == 409
        assert deleting_error.value.detail["code"] == (
            "SESSION_DELETION_IN_PROGRESS"
        )

        assert services.chat_registry.reserve_session_deletion(
            other_user,
            other_session,
        ) == []
        services.chat_registry.abort_session_deletion(other_user, other_session)
        services.chat_registry.abort_session_deletion(USER_ID, session_id)
        await services.task_queue.close()

    asyncio.run(scenario())
