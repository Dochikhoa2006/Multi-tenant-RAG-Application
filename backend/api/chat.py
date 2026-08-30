"""Chat query streaming and process-local session endpoints."""

from __future__ import annotations

import asyncio
import json
from time import perf_counter

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import StreamingResponse

from backend.api.dependencies import get_services
from backend.api.errors import (
    log_internal_error,
    not_found,
    public_detail,
    public_http_error,
    request_id,
    service_unavailable,
    validation_error,
)
from backend.api.models import (
    ConversationResource,
    QueryRequest,
    SessionCreateRequest,
    SessionDetailResource,
    SessionResource,
    SessionTitleRequest,
    TaskResource,
)
from backend.api.telemetry import TelemetryCollector
from backend.mappings._common import validated_user_id
from backend.rag.pipeline import run_rag_pipeline
from backend.rag.session_title import generate_session_title
from backend.services import (
    ActiveChatStreamError,
    AppServices,
    SessionDeletionInProgressError,
    SessionSnapshot,
)


router = APIRouter(prefix="/api/chat", tags=["chat"])


def _session_resource(snapshot: SessionSnapshot) -> SessionResource:
    return SessionResource(
        session_id=snapshot.session_id,
        user_id=snapshot.user_id,
        title=snapshot.title,
        conversation_count=len(snapshot.conversations),
    )


def _session_detail(snapshot: SessionSnapshot) -> SessionDetailResource:
    return SessionDetailResource(
        session_id=snapshot.session_id,
        user_id=snapshot.user_id,
        title=snapshot.title,
        conversation_count=len(snapshot.conversations),
        conversations=[
            ConversationResource(
                conversation_id=item.conversation_id,
                question=item.question,
                answer=item.answer,
            )
            for item in snapshot.conversations
        ],
    )


def _sse(event: str, payload: object) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _task_resource(snapshot: object) -> TaskResource:
    return TaskResource(**snapshot.__dict__)


async def _ensure_collections(
    services: AppServices,
    user_id: str,
    correlation_id: str,
) -> None:
    try:
        validated_user_id(user_id)
    except (TypeError, ValueError) as exc:
        raise validation_error(str(exc), correlation_id) from exc
    try:
        await asyncio.to_thread(services.manager.ensure_user_collections, user_id)
    except Exception as exc:
        log_internal_error(
            "Could not ensure user collections",
            correlation_id,
            user_id=user_id,
        )
        raise service_unavailable(correlation_id) from exc


@router.post("/sessions", status_code=status.HTTP_201_CREATED, response_model=SessionResource)
async def create_session(
    body: SessionCreateRequest,
    request: Request,
    services: AppServices = Depends(get_services),
) -> SessionResource:
    correlation_id = request_id(request)
    try:
        snapshot = services.chat_registry.create_session(
            body.user_id,
            services.new_uuid(),
        )
    except (TypeError, ValueError) as exc:
        raise validation_error(str(exc), correlation_id) from exc
    return _session_resource(snapshot)


@router.get("/sessions", response_model=list[SessionResource])
async def list_sessions(
    request: Request,
    user_id: str = Query(...),
    services: AppServices = Depends(get_services),
) -> list[SessionResource]:
    correlation_id = request_id(request)
    try:
        return [
            _session_resource(item)
            for item in services.chat_registry.list_sessions(user_id)
        ]
    except (TypeError, ValueError) as exc:
        raise validation_error(str(exc), correlation_id) from exc


@router.get("/sessions/{session_id}", response_model=SessionDetailResource)
async def get_session(
    session_id: str,
    request: Request,
    user_id: str = Query(...),
    services: AppServices = Depends(get_services),
) -> SessionDetailResource:
    correlation_id = request_id(request)
    try:
        snapshot = services.chat_registry.get_session(user_id, session_id)
    except KeyError as exc:
        raise not_found("session", correlation_id) from exc
    except (TypeError, ValueError) as exc:
        raise validation_error(str(exc), correlation_id) from exc
    return _session_detail(snapshot)


@router.patch("/sessions/{session_id}/title", response_model=SessionResource)
async def update_session_title(
    session_id: str,
    body: SessionTitleRequest,
    request: Request,
    services: AppServices = Depends(get_services),
) -> SessionResource:
    correlation_id = request_id(request)
    try:
        services.chat_registry.update_title(body.user_id, session_id, body.title)
        snapshot = services.chat_registry.get_session(body.user_id, session_id)
    except KeyError as exc:
        raise not_found("session", correlation_id) from exc
    except (TypeError, ValueError) as exc:
        raise validation_error(str(exc), correlation_id) from exc
    return _session_resource(snapshot)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TaskResource,
)
async def delete_session(
    session_id: str,
    request: Request,
    user_id: str = Query(...),
    services: AppServices = Depends(get_services),
) -> TaskResource:
    correlation_id = request_id(request)
    try:
        conversation_ids = services.chat_registry.reserve_session_deletion(
            user_id,
            session_id,
        )
    except ActiveChatStreamError as exc:
        raise public_http_error(
            status.HTTP_409_CONFLICT,
            "SESSION_ACTIVE",
            "The session has an active response stream. Try again after it completes.",
            correlation_id,
        ) from exc
    except SessionDeletionInProgressError as exc:
        raise public_http_error(
            status.HTTP_409_CONFLICT,
            "SESSION_DELETION_IN_PROGRESS",
            "The session is already being deleted.",
            correlation_id,
        ) from exc
    except KeyError as exc:
        raise not_found("session", correlation_id) from exc
    except (TypeError, ValueError) as exc:
        raise validation_error(str(exc), correlation_id) from exc

    try:
        await _ensure_collections(services, user_id, correlation_id)
    except asyncio.CancelledError:
        services.chat_registry.abort_session_deletion(user_id, session_id)
        raise
    except Exception:
        services.chat_registry.abort_session_deletion(user_id, session_id)
        raise

    async def work() -> None:
        committed = False
        try:
            collection = services.conversation_collection_factory(user_id)
            report = await asyncio.to_thread(
                collection.delete_batch_verified,
                conversation_ids,
            )
            if not report.confirmed:
                raise RuntimeError("conversation deletion was not confirmed")
            services.chat_registry.commit_session_deletion(user_id, session_id)
            committed = True
        finally:
            if not committed:
                services.chat_registry.abort_session_deletion(user_id, session_id)

    try:
        task_id = await services.task_queue.enqueue(user_id, "delete_session", work)
    except asyncio.CancelledError:
        services.chat_registry.abort_session_deletion(user_id, session_id)
        raise
    except Exception:
        services.chat_registry.abort_session_deletion(user_id, session_id)
        raise
    return _task_resource(services.task_queue.get(task_id, user_id))


@router.post("/query")
async def query(
    body: QueryRequest,
    request: Request,
    services: AppServices = Depends(get_services),
) -> StreamingResponse:
    request_started = perf_counter()
    correlation_id = request_id(request)
    try:
        if not body.question.strip():
            raise ValueError("question must not be empty")
        services.chat_registry.get_session(body.user_id, body.session_id)
        conversation_id = services.new_uuid()
    except KeyError as exc:
        raise not_found("session", correlation_id) from exc
    except (TypeError, ValueError) as exc:
        raise validation_error(str(exc), correlation_id) from exc
    try:
        runtime = services.require_rag_runtime()
    except RuntimeError as exc:
        log_internal_error(
            "RAG runtime is unavailable",
            correlation_id,
            user_id=body.user_id,
        )
        raise service_unavailable(correlation_id) from exc
    try:
        collections = services.retrieval_collections_factory(body.user_id)
    except Exception as exc:
        log_internal_error(
            "Could not construct retrieval collections",
            correlation_id,
            user_id=body.user_id,
        )
        raise service_unavailable(correlation_id) from exc

    try:
        services.chat_registry.begin_chat_stream(
            body.user_id,
            body.session_id,
            conversation_id,
        )
    except SessionDeletionInProgressError as exc:
        raise public_http_error(
            status.HTTP_409_CONFLICT,
            "SESSION_DELETION_IN_PROGRESS",
            "The session is being deleted and cannot accept new messages.",
            correlation_id,
        ) from exc
    except KeyError as exc:
        raise not_found("session", correlation_id) from exc
    except (TypeError, ValueError) as exc:
        raise validation_error(str(exc), correlation_id) from exc

    try:
        await _ensure_collections(services, body.user_id, correlation_id)
    except asyncio.CancelledError:
        services.chat_registry.end_chat_stream(
            body.user_id,
            body.session_id,
            conversation_id,
        )
        raise
    except Exception:
        services.chat_registry.end_chat_stream(
            body.user_id,
            body.session_id,
            conversation_id,
        )
        raise

    telemetry = TelemetryCollector()

    async def stream_events():
        answer_parts: list[str] = []
        first_token_at: float | None = None
        try:
            async for chunk in run_rag_pipeline(
                body.user_id,
                conversation_id,
                body.question,
                collections,
                runtime=runtime,
                timing_observer=telemetry.observe,
            ):
                now = perf_counter()
                if first_token_at is None:
                    first_token_at = now
                    telemetry.set("ttft", (now - request_started) * 1000.0)
                answer_parts.append(chunk)
                yield _sse("token", {"request_id": correlation_id, "text": chunk})

            answer = "".join(answer_parts)
            if first_token_at is None:
                raise ValueError("answer stream completed without tokens")
            services.chat_registry.record_conversation(
                body.user_id,
                body.session_id,
                conversation_id,
                body.question,
                answer,
            )

            async def title_work() -> None:
                snapshot = services.chat_registry.get_session(
                    body.user_id,
                    body.session_id,
                )
                conversation_list = [
                    f"Question:\n{item.question}\n\nAnswer:\n{item.answer}"
                    for item in snapshot.conversations
                ]
                title = await generate_session_title(
                    conversation_list,
                    runtime=runtime,
                )
                services.chat_registry.update_title(
                    body.user_id,
                    body.session_id,
                    title,
                )

            await services.task_queue.enqueue(
                body.user_id,
                "generate_session_title",
                title_work,
            )
            telemetry.set("total_request", (perf_counter() - request_started) * 1000.0)
            yield _sse("telemetry", telemetry.payload(correlation_id))
            yield _sse(
                "done",
                {
                    "request_id": correlation_id,
                    "conversation_id": conversation_id,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_internal_error(
                "Chat request failed",
                correlation_id,
                user_id=body.user_id,
                session_id=body.session_id,
                conversation_id=conversation_id,
            )
            yield _sse(
                "error",
                public_detail(
                    "CHAT_PROCESSING_FAILED",
                    "The response could not be completed.",
                    correlation_id,
                ),
            )
        finally:
            services.chat_registry.end_chat_stream(
                body.user_id,
                body.session_id,
                conversation_id,
            )

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router"]
