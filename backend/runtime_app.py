"""Integrated ASGI composition for the complete single-process RAG runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
import logging
import threading
from typing import Any

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
import httpx
import tiktoken

from backend.dev_ui import install_dev_ui
from backend.main import create_app
from backend.model_config import TEXT_PROCESSING
from backend.processing.chunker import _get_tokenizer as _get_chunk_tokenizer
from backend.processing.paragraph_splitter import (
    _get_sentence_transformer,
)
from backend.providers.granite_query_rewriter import RoleRoutingLLMClient
from backend.providers.onnx_embedding import ONNXEmbeddingClient
from backend.providers.onnx_reranker import ONNXCrossEncoderReranker
from backend.providers.sglang_query_rewriter import SGLangGraniteQueryRewriter
from backend.providers.sglang_qwen_llm import SGLangQwenLLMClient
from backend.rag.runtime import RAGRuntime
from backend.services import AppServices
from backend.task_queue import InMemoryTaskQueue
from backend.weaviate_client.client import WeaviateManager
from backend.weaviate_client.conversation import ConversationCollection


class RuntimeDependencyError(RuntimeError):
    """A required integrated-runtime dependency is unavailable or incompatible."""


Factory = Callable[[], Any]
ModelEndpointValidator = Callable[[object, str, str, str, str], None]
_LOGGER = logging.getLogger(__name__)


def _response_json(response: object) -> object:
    raise_for_status = getattr(response, "raise_for_status", None)
    json_method = getattr(response, "json", None)
    if not callable(raise_for_status) or not callable(json_method):
        raise RuntimeDependencyError("SGLang model discovery returned an invalid response")
    raise_for_status()
    return json_method()


def _validate_sglang_model(
    client: object,
    base_url: str,
    api_key: str,
    expected_model: str,
    service_name: str,
) -> None:
    """Verify one authenticated OpenAI-compatible SGLang model endpoint."""

    get = getattr(client, "get", None)
    if not callable(get):
        raise RuntimeDependencyError(f"{service_name} client cannot discover models")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        payload = _response_json(get(f"{base_url}/models", headers=headers))
    except RuntimeDependencyError:
        raise
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        raise RuntimeDependencyError(
            f"{service_name} SGLang endpoint is unavailable or malformed"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeDependencyError(f"{service_name} model list is malformed")
    data = payload.get("data")
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
        raise RuntimeDependencyError(f"{service_name} model list is malformed")
    model_ids = {
        item.get("id")
        for item in data
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if expected_model not in model_ids:
        raise RuntimeDependencyError(
            f"{service_name} SGLang does not serve the configured model"
        )


def _warm_processing_runtime() -> None:
    _get_sentence_transformer()
    _get_chunk_tokenizer()


def _clear_processing_runtime() -> None:
    _get_sentence_transformer.cache_clear()
    _get_chunk_tokenizer.cache_clear()


def _close(resource: object) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def _run_async_construction_cleanup(callback: Callable[[], Any]) -> None:
    """Run one async close outside any event loop invoking the app factory."""

    errors: list[BaseException] = []

    def run() -> None:
        try:
            asyncio.run(callback())
        except BaseException as exc:  # Preserve the construction exception.
            errors.append(exc)

    worker = threading.Thread(target=run, name="rag-construction-cleanup")
    worker.start()
    worker.join()
    if errors:
        raise errors[0]


def create_runtime_app(
    *,
    manager_factory: Factory = WeaviateManager,
    task_queue_factory: Factory = InMemoryTaskQueue,
    embedding_factory: Factory = ONNXEmbeddingClient,
    reranker_factory: Factory = ONNXCrossEncoderReranker,
    granite_factory: Factory = SGLangGraniteQueryRewriter,
    qwen_factory: Factory = SGLangQwenLLMClient,
    tokenizer_factory: Factory | None = None,
    processing_warmup: Callable[[], None] = _warm_processing_runtime,
    processing_cleanup: Callable[[], None] = _clear_processing_runtime,
    model_endpoint_validator: ModelEndpointValidator = _validate_sglang_model,
) -> FastAPI:
    """Build one fully wired runtime; factories are an offline testing seam."""

    factories = (
        manager_factory,
        task_queue_factory,
        embedding_factory,
        reranker_factory,
        granite_factory,
        qwen_factory,
        processing_warmup,
        processing_cleanup,
        model_endpoint_validator,
    )
    if any(not callable(item) for item in factories):
        raise TypeError("runtime factories and lifecycle callbacks must be callable")

    construction_cleanup: list[tuple[str, Callable[[], None]]] = []
    try:
        manager = manager_factory()
        construction_cleanup.append(("Weaviate manager", manager.disconnect))

        task_queue = task_queue_factory()
        construction_cleanup.append(
            (
                "task queue",
                lambda: _run_async_construction_cleanup(task_queue.close),
            )
        )

        embedding = embedding_factory()
        construction_cleanup.append(("embedding client", embedding.close))

        reranker = reranker_factory()
        construction_cleanup.append(("reranker", reranker.close))

        granite = granite_factory()
        construction_cleanup.append(("Granite client", granite.close))

        qwen = qwen_factory()

        def close_qwen() -> None:
            try:
                _run_async_construction_cleanup(qwen.aclose)
            finally:
                _close(qwen)

        construction_cleanup.append(("Qwen client", close_qwen))

        router = RoleRoutingLLMClient(delegate=qwen, granite=granite)
        active_tokenizer_factory = tokenizer_factory or (
            lambda: tiktoken.get_encoding(TEXT_PROCESSING.tokenizer_encoding)
        )
        tokenizer = active_tokenizer_factory()
        runtime = RAGRuntime(
            router,
            embedding,
            reranker,
            lambda user_id: ConversationCollection(manager, user_id),
            tokenizer=tokenizer,
            background_queue=task_queue,
        )
        services = AppServices(
            manager=manager,
            rag_runtime=runtime,
            task_queue=task_queue,
        )
    except BaseException:
        for resource_name, cleanup in reversed(construction_cleanup):
            try:
                cleanup()
            except BaseException:
                _LOGGER.exception(
                    "Owned resource cleanup failed during runtime construction",
                    extra={"resource": resource_name},
                )
        raise

    application: FastAPI

    async def startup() -> None:
        await asyncio.to_thread(processing_warmup)
        await asyncio.gather(
            asyncio.to_thread(
                model_endpoint_validator,
                granite.client,
                granite.sglang_config.base_url,
                granite.sglang_config.api_key,
                granite.sglang_config.served_model,
                "Granite",
            ),
            asyncio.to_thread(
                model_endpoint_validator,
                qwen.sync_client,
                qwen.config.base_url,
                qwen.config.api_key,
                qwen.config.served_model,
                "Qwen",
            ),
        )
        application.state.runtime_ready = True

    async def shutdown() -> None:
        application.state.runtime_ready = False
        try:
            await qwen.aclose()
        finally:
            try:
                await asyncio.to_thread(_close, qwen)
            finally:
                try:
                    await asyncio.to_thread(_close, granite)
                finally:
                    try:
                        await asyncio.to_thread(_close, embedding)
                    finally:
                        try:
                            await asyncio.to_thread(_close, reranker)
                        finally:
                            await asyncio.to_thread(processing_cleanup)

    application = create_app(
        services,
        startup_hook=startup,
        shutdown_hook=shutdown,
    )
    application.state.runtime_ready = False

    @application.get("/health", include_in_schema=False)
    async def health() -> JSONResponse:
        if not application.state.runtime_ready:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "starting"},
            )
        return JSONResponse(content={"status": "ok"})

    install_dev_ui(application)
    return application


__all__ = ["RuntimeDependencyError", "create_runtime_app"]
