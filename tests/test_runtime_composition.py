from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from backend.main import create_app
from backend.providers.granite_query_rewriter import RoleRoutingLLMClient
from backend.runtime_app import RuntimeDependencyError, create_runtime_app
from backend.services import AppServices


class RecordingManager:
    def __init__(self, events: list[str], *, connect_error: Exception | None = None) -> None:
        self.events = events
        self.connect_error = connect_error
        self.client = SimpleNamespace()

    def connect(self) -> object:
        self.events.append("manager.connect")
        if self.connect_error is not None:
            raise self.connect_error
        return self.client

    def disconnect(self) -> None:
        self.events.append("manager.disconnect")


class RecordingQueue:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def enqueue(self, *args: object, **kwargs: object) -> str:
        return "task-id"

    async def close(self) -> None:
        self.events.append("queue.close")


class RecordingEmbedding:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def embed(self, text: str, *, model: str) -> Sequence[float]:
        return [1.0]

    def embed_many(self, texts: Sequence[str], *, model: str) -> Sequence[Sequence[float]]:
        return [[1.0] for _ in texts]

    def close(self) -> None:
        self.events.append("embedding.close")


class RecordingReranker:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def rerank(self, *args: object, **kwargs: object) -> list[object]:
        return []

    def close(self) -> None:
        self.events.append("reranker.close")


class ModelResponse:
    def __init__(self, model: str) -> None:
        self.model = model

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"object": "list", "data": [{"id": self.model}]}


class ModelClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs: object) -> ModelResponse:
        self.calls.append((url, dict(kwargs)))
        return ModelResponse(self.model)


class RecordingGranite:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.sglang_config = SimpleNamespace(
            base_url="https://granite.example/v1",
            api_key="granite-secret",
            served_model="granite-model",
        )
        self.client = ModelClient(self.sglang_config.served_model)

    def complete(self, *args: object, **kwargs: object) -> str:
        return "rewritten"

    def close(self) -> None:
        self.events.append("granite.close")


class RecordingQwen:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.config = SimpleNamespace(
            base_url="https://qwen.example/v1",
            api_key="qwen-secret",
            served_model="qwen-model",
        )
        self.sync_client = ModelClient(self.config.served_model)

    def complete(self, *args: object, **kwargs: object) -> str:
        return "Three Word Title"

    def stream(self, *args: object, **kwargs: object) -> AsyncIterator[str]:
        async def output() -> AsyncIterator[str]:
            yield "answer"

        return output()

    def close(self) -> None:
        self.events.append("qwen.close")

    async def aclose(self) -> None:
        self.events.append("qwen.aclose")


class RecordingTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def _application(
    events: list[str],
    *,
    validator: object | None = None,
    connect_error: Exception | None = None,
    processing_warmup: object | None = None,
):
    created: dict[str, object] = {}
    calls: dict[str, int] = {}

    def factory(name: str, constructor: object):
        def create() -> object:
            calls[name] = calls.get(name, 0) + 1
            value = constructor()  # type: ignore[operator]
            created[name] = value
            return value

        return create

    kwargs: dict[str, object] = {}
    if validator is not None:
        kwargs["model_endpoint_validator"] = validator
    app = create_runtime_app(
        manager_factory=factory(
            "manager", lambda: RecordingManager(events, connect_error=connect_error)
        ),
        task_queue_factory=factory("queue", lambda: RecordingQueue(events)),
        embedding_factory=factory("embedding", lambda: RecordingEmbedding(events)),
        reranker_factory=factory("reranker", lambda: RecordingReranker(events)),
        granite_factory=factory("granite", lambda: RecordingGranite(events)),
        qwen_factory=factory("qwen", lambda: RecordingQwen(events)),
        tokenizer_factory=factory("tokenizer", RecordingTokenizer),
        processing_warmup=(
            processing_warmup
            if processing_warmup is not None
            else lambda: events.append("processing.warmup")
        ),
        processing_cleanup=lambda: events.append("processing.cleanup"),
        **kwargs,
    )
    return app, created, calls


def test_runtime_composes_singletons_and_closes_owned_resources_in_order() -> None:
    events: list[str] = []
    validations: list[tuple[object, str, str, str, str]] = []

    def validate(*args: object) -> None:
        validations.append(args)  # type: ignore[arg-type]

    app, created, calls = _application(events, validator=validate)
    services = app.state.services

    assert all(count == 1 for count in calls.values())
    assert services.manager is created["manager"]
    assert services.task_queue is created["queue"]
    assert services.rag_runtime.embeddings is created["embedding"]
    assert services.rag_runtime.reranker is created["reranker"]
    assert services.rag_runtime.background_queue is created["queue"]
    assert services.rag_runtime.tokenizer is created["tokenizer"]
    assert isinstance(services.rag_runtime.llm, RoleRoutingLLMClient)
    assert services.rag_runtime.llm.granite is created["granite"]
    assert services.rag_runtime.llm.delegate is created["qwen"]
    assert services.wizard_runtime.embedder._embeddings is created["embedding"]

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert app.state.runtime_ready is True
        assert len(validations) == 2

    assert app.state.runtime_ready is False
    assert events == [
        "manager.connect",
        "processing.warmup",
        "queue.close",
        "qwen.aclose",
        "qwen.close",
        "granite.close",
        "embedding.close",
        "reranker.close",
        "processing.cleanup",
        "manager.disconnect",
    ]


def test_default_model_discovery_uses_expected_urls_and_bearer_tokens() -> None:
    events: list[str] = []
    app, created, _ = _application(events)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    granite = created["granite"]
    qwen = created["qwen"]
    assert granite.client.calls == [  # type: ignore[union-attr]
        (
            "https://granite.example/v1/models",
            {"headers": {"Accept": "application/json", "Authorization": "Bearer granite-secret"}},
        )
    ]
    assert qwen.sync_client.calls == [  # type: ignore[union-attr]
        (
            "https://qwen.example/v1/models",
            {"headers": {"Accept": "application/json", "Authorization": "Bearer qwen-secret"}},
        )
    ]


def test_startup_dependency_failure_is_fail_closed_and_cleans_up() -> None:
    events: list[str] = []

    def fail(*args: object) -> None:
        raise RuntimeDependencyError("wrong served model")

    app, _, _ = _application(events, validator=fail)
    with pytest.raises(RuntimeDependencyError, match="wrong served model"):
        with TestClient(app):
            pass

    assert app.state.runtime_ready is False
    assert events == [
        "manager.connect",
        "processing.warmup",
        "queue.close",
        "qwen.aclose",
        "qwen.close",
        "granite.close",
        "embedding.close",
        "reranker.close",
        "processing.cleanup",
        "manager.disconnect",
    ]


def test_wrong_advertised_sglang_model_blocks_startup() -> None:
    events: list[str] = []
    app, created, _ = _application(events)
    created["granite"].client.model = "wrong-model"  # type: ignore[union-attr]

    with pytest.raises(RuntimeDependencyError, match="does not serve"):
        with TestClient(app):
            pass

    assert app.state.runtime_ready is False
    assert "manager.disconnect" in events


def test_invalid_segmentation_artifact_blocks_startup_and_never_becomes_ready() -> None:
    events: list[str] = []

    def fail_segmentation() -> None:
        raise RuntimeError("locally provisioned segmentation model is invalid")

    app, _, _ = _application(events, processing_warmup=fail_segmentation)
    with pytest.raises(RuntimeError, match="segmentation model is invalid"):
        with TestClient(app):
            pass

    assert app.state.runtime_ready is False
    assert events == [
        "manager.connect",
        "queue.close",
        "qwen.aclose",
        "qwen.close",
        "granite.close",
        "embedding.close",
        "reranker.close",
        "processing.cleanup",
        "manager.disconnect",
    ]


def test_weaviate_connection_failure_still_closes_non_storage_resources() -> None:
    events: list[str] = []
    app, _, _ = _application(events, connect_error=RuntimeError("offline"))

    with pytest.raises(RuntimeError, match="offline"):
        with TestClient(app):
            pass

    assert events == [
        "manager.connect",
        "queue.close",
        "qwen.aclose",
        "qwen.close",
        "granite.close",
        "embedding.close",
        "reranker.close",
        "processing.cleanup",
    ]


def test_dev_ui_is_integrated_only_and_targets_real_api() -> None:
    events: list[str] = []
    app, _, _ = _application(events, validator=lambda *args: None)
    providerless = create_app(
        AppServices(
            manager=RecordingManager([]),
            task_queue=RecordingQueue([]),
        )
    )

    with TestClient(app) as client:
        page = client.get("/dev/e2e")
        assert page.status_code == 200
        assert "fetch('/api/chat/sessions'" in page.text
        assert "fetch('/api/chat/query'" in page.text
        assert "response.body.getReader()" in page.text
        assert "answer.textContent += payload.text" in page.text
        assert "name === 'telemetry'" in page.text
        assert "consumeFrames(state" in page.text
        assert "if (requestInFlight) return" in page.text
        assert "submitButton.disabled = true" in page.text
        assert "name === 'done' || name === 'error'" in page.text
        assert "SSE stream ended without done or error" in page.text
        assert "finally {" in page.text
        assert "requestInFlight = false" in page.text
        assert "submitButton.disabled = false" in page.text
        created = client.post("/api/chat/sessions", json={"user_id": "usr_dev"})
        assert created.status_code == 201
        sessions = client.get("/api/chat/sessions?user_id=usr_dev")
        assert sessions.status_code == 200
        assert sessions.json()[0]["session_id"] == created.json()["session_id"]

    with TestClient(providerless) as client:
        assert client.get("/dev/e2e").status_code == 404


def test_partial_construction_failure_closes_only_created_resources_in_reverse() -> None:
    events: list[str] = []
    original = RuntimeError("reranker construction failed")

    def fail_reranker() -> object:
        raise original

    with pytest.raises(RuntimeError) as error:
        create_runtime_app(
            manager_factory=lambda: RecordingManager(events),
            task_queue_factory=lambda: RecordingQueue(events),
            embedding_factory=lambda: RecordingEmbedding(events),
            reranker_factory=fail_reranker,
        )

    assert error.value is original
    assert events == [
        "embedding.close",
        "queue.close",
        "manager.disconnect",
    ]


def test_full_construction_cleanup_preserves_original_when_cleanup_fails() -> None:
    events: list[str] = []
    original = RuntimeError("tokenizer construction failed")

    class FailingCleanupQwen(RecordingQwen):
        async def aclose(self) -> None:
            self.events.append("qwen.aclose")
            raise RuntimeError("cleanup failed")

    def fail_tokenizer() -> object:
        raise original

    with pytest.raises(RuntimeError) as error:
        create_runtime_app(
            manager_factory=lambda: RecordingManager(events),
            task_queue_factory=lambda: RecordingQueue(events),
            embedding_factory=lambda: RecordingEmbedding(events),
            reranker_factory=lambda: RecordingReranker(events),
            granite_factory=lambda: RecordingGranite(events),
            qwen_factory=lambda: FailingCleanupQwen(events),
            tokenizer_factory=fail_tokenizer,
        )

    assert error.value is original
    assert events == [
        "qwen.aclose",
        "qwen.close",
        "granite.close",
        "reranker.close",
        "embedding.close",
        "queue.close",
        "manager.disconnect",
    ]
