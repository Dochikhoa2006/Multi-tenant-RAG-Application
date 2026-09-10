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
from backend.wizard.diagnostics import (
    TRACE_OPERATION_HEADER,
    TRACE_SESSION_HEADER,
    activate_operation,
    attach_related_task,
    begin_request_operation,
    fail_operation_on_error,
    finish_operation,
    add_count,
    observe_elapsed,
    observe_stage,
    set_flag,
    set_framed_digest,
    set_sample,
    set_text,
    trace_operation_active,
    trace_utf8_bytes,
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
    trace_handle = begin_request_operation(
        user_id=body.user_id,
        session_id=request.headers.get(TRACE_SESSION_HEADER),
        operation_id=request.headers.get(TRACE_OPERATION_HEADER),
        kind="chat_query",
        collection_type="conversations",
        wizard_id=body.session_id,
    )
    try:
        with fail_operation_on_error(trace_handle):
            with observe_stage(
                "chat.api_validation_session", handle=trace_handle
            ):
                if not body.question.strip():
                    raise ValueError("question must not be empty")
                services.chat_registry.get_session(body.user_id, body.session_id)
                conversation_id = services.new_uuid()
    except KeyError as exc:
        raise not_found("session", correlation_id) from exc
    except (TypeError, ValueError) as exc:
        raise validation_error(str(exc), correlation_id) from exc
    try:
        with fail_operation_on_error(trace_handle):
            with observe_stage("chat.runtime_setup", handle=trace_handle):
                runtime = services.require_rag_runtime()
    except RuntimeError as exc:
        log_internal_error(
            "RAG runtime is unavailable",
            correlation_id,
            user_id=body.user_id,
        )
        raise service_unavailable(correlation_id) from exc
    try:
        with fail_operation_on_error(trace_handle):
            with observe_stage("chat.collection_factory", handle=trace_handle):
                collections = services.retrieval_collections_factory(body.user_id)
    except Exception as exc:
        log_internal_error(
            "Could not construct retrieval collections",
            correlation_id,
            user_id=body.user_id,
        )
        raise service_unavailable(correlation_id) from exc

    try:
        with fail_operation_on_error(trace_handle):
            with observe_stage(
                "chat.session_stream_reservation", handle=trace_handle
            ):
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
        with fail_operation_on_error(trace_handle):
            with observe_stage("chat.collection_ensure", handle=trace_handle):
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
        trace_outcome = "failed"
        sse_event_count = 0
        with activate_operation(trace_handle):
            for empty_count in (
                "sse_token_event_count",
                "sse_telemetry_event_count",
                "sse_done_event_count",
                "sse_error_event_count",
            ):
                add_count(empty_count, 0)
            observe_elapsed(
                "chat.endpoint_pre_pipeline_total",
                (perf_counter() - request_started) * 1000.0,
            )
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
                    sse_event_count += 1
                    add_count("sse_token_event_count", 1)
                    if sse_event_count == 1:
                        add_count("sse_first_token_ordinal", 1)
                    yield _sse(
                        "token", {"request_id": correlation_id, "text": chunk}
                    )

                answer = "".join(answer_parts)
                if first_token_at is None:
                    raise ValueError("answer stream completed without tokens")
                with observe_stage("chat.conversation_registry_update"):
                    services.chat_registry.record_conversation(
                        body.user_id,
                        body.session_id,
                        conversation_id,
                        body.question,
                        answer,
                    )
                add_count("conversation_registry_update_count", 1)
                set_sample(
                    "conversation_registry_ids",
                    (conversation_id,),
                    exact_count=1,
                )

                async def title_work() -> None:
                    with activate_operation(trace_handle):
                        add_count("title_task_started_count", 1)
                        try:
                            with observe_stage("chat.title_task_execution"):
                                with observe_stage("chat.title_snapshot"):
                                    snapshot = services.chat_registry.get_session(
                                        body.user_id,
                                        body.session_id,
                                    )
                                add_count(
                                    "title_snapshot_conversation_count",
                                    len(snapshot.conversations),
                                )
                                if snapshot.conversations:
                                    set_text(
                                        "title_snapshot_last_conversation_id",
                                        snapshot.conversations[-1].conversation_id,
                                    )
                                    set_flag(
                                        "title_snapshot_trigger_matches",
                                        snapshot.conversations[-1].conversation_id
                                        == conversation_id,
                                    )
                                set_sample(
                                    "title_snapshot_conversation_ids",
                                    (
                                        item.conversation_id
                                        for item in snapshot.conversations
                                    ),
                                    exact_count=len(snapshot.conversations),
                                )
                                set_framed_digest(
                                    "title_snapshot_sha256",
                                    "chat-title-snapshot-v1",
                                    (
                                        part
                                        for item in snapshot.conversations
                                        for part in (
                                            item.conversation_id.encode("ascii"),
                                            item.question.encode("utf-8"),
                                            item.answer.encode("utf-8"),
                                        )
                                    ),
                                )
                                with observe_stage("chat.title_transcript_render"):
                                    conversation_list = [
                                        "Question:\n"
                                        f"{item.question}\n\nAnswer:\n{item.answer}"
                                        for item in snapshot.conversations
                                    ]
                                title = await generate_session_title(
                                    conversation_list,
                                    runtime=runtime,
                                )
                                with observe_stage("chat.title_registry_update"):
                                    services.chat_registry.update_title(
                                        body.user_id,
                                        body.session_id,
                                        title,
                                    )
                                add_count("title_registry_update_count", 1)
                        except BaseException:
                            set_flag("title_task_succeeded", False)
                            raise
                        else:
                            set_flag("title_task_succeeded", True)
                            add_count("title_task_completed_count", 1)

                try:
                    with observe_stage("chat.title_enqueue"):
                        title_task_id = await services.task_queue.enqueue(
                            body.user_id,
                            "generate_session_title",
                            title_work,
                        )
                    add_count("title_enqueue_count", 1)
                    set_flag("title_enqueue_accepted", True)
                    attach_related_task(
                        trace_handle,
                        "session_title",
                        title_task_id,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    set_flag("title_enqueue_accepted", False)
                    log_internal_error(
                        "Could not schedule optional session title generation",
                        correlation_id,
                        user_id=body.user_id,
                        session_id=body.session_id,
                        conversation_id=conversation_id,
                    )
                telemetry.set(
                    "total_request", (perf_counter() - request_started) * 1000.0
                )
                add_count("sse_last_token_ordinal", len(answer_parts))
                if trace_operation_active():
                    answer_bytes = trace_utf8_bytes(answer)
                    if answer_bytes is not None:
                        add_count("sse_answer_utf8_bytes", len(answer_bytes))
                    set_framed_digest(
                        "sse_answer_chunks_sha256",
                        "chat-qwen-answer-chunks-v1",
                        (part.encode("utf-8") for part in answer_parts),
                    )
                sse_event_count += 1
                add_count("sse_telemetry_event_count", 1)
                add_count("sse_telemetry_ordinal", sse_event_count)
                yield _sse("telemetry", telemetry.payload(correlation_id))
                sse_event_count += 1
                add_count("sse_done_event_count", 1)
                add_count("sse_done_ordinal", sse_event_count)
                add_count("sse_total_event_count", sse_event_count)
                set_flag("sse_success_order_valid", True)
                set_text("sse_terminal_event", "done")
                yield _sse(
                    "done",
                    {
                        "request_id": correlation_id,
                        "conversation_id": conversation_id,
                    },
                )
                trace_outcome = "succeeded"
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
                add_count("sse_last_token_ordinal", len(answer_parts))
                if trace_operation_active():
                    answer_bytes = trace_utf8_bytes("".join(answer_parts))
                    if answer_bytes is not None:
                        add_count("sse_answer_utf8_bytes", len(answer_bytes))
                    set_framed_digest(
                        "sse_answer_chunks_sha256",
                        "chat-qwen-answer-chunks-v1",
                        (part.encode("utf-8") for part in answer_parts),
                    )
                sse_event_count += 1
                add_count("sse_error_event_count", 1)
                add_count("sse_error_ordinal", sse_event_count)
                add_count("sse_total_event_count", sse_event_count)
                set_text("sse_terminal_event", "error")
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
                finish_operation(trace_handle, trace_outcome)

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
