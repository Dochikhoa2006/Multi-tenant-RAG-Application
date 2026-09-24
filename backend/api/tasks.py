"""Observable process-local background task status endpoints."""

import inspect
import os

from fastapi import APIRouter, Depends, Request

from backend.api.dependencies import get_services
from backend.api.errors import not_found, request_id, validation_error
from backend.api.models import TaskResource
from backend.config import (
    RAG_ACCEPTANCE_EXPERIMENT_SHA256, RAG_ACCEPTANCE_OBSERVER_ENABLED,
    RAG_EVALUATION_EVIDENCE_ENABLED,
)
from backend.services import AppServices


router = APIRouter(prefix="/api/tasks", tags=["tasks"])
_ACCEPTANCE_SCAN_LIMIT = 1024


@router.get("/_acceptance/post-generation", include_in_schema=False)
async def get_post_generation_acceptance(
    request: Request,
    user_id: str,
    session_id: str,
    conversation_id: str,
    services: AppServices = Depends(get_services),
) -> dict[str, object]:
    """Return bounded, read-only task/runtime evidence for an acceptance run."""

    correlation_id = request_id(request)
    if not RAG_ACCEPTANCE_OBSERVER_ENABLED:
        raise not_found("post-generation tasks", correlation_id)
    try:
        session = services.chat_registry.get_session(user_id, session_id)
    except (KeyError, TypeError, ValueError) as exc:
        raise not_found("session", correlation_id) from exc
    if conversation_id not in {
        item.conversation_id for item in session.conversations
    }:
        raise not_found("conversation", correlation_id)

    records = tuple(services.task_queue._records.values())
    if len(records) > _ACCEPTANCE_SCAN_LIMIT:
        raise not_found("post-generation tasks", correlation_id)
    matches: dict[str, list[object]] = {
        name: [] for name in ("embed_conversation", "generate_session_title")
    }
    for record in records:
        operation = getattr(record, "operation", None)
        if operation not in matches or getattr(record, "user_id", None) != user_id:
            continue
        try:
            closed = inspect.getclosurevars(record.work_factory).nonlocals
        except (AttributeError, TypeError, ValueError):
            continue
        key = "conversation" if operation == "embed_conversation" else "conversation_id"
        if closed.get(key) == conversation_id:
            matches[str(operation)].append(record)
    if any(len(items) != 1 for items in matches.values()):
        raise not_found("post-generation tasks", correlation_id)
    return {
        "schema_version": "1.0",
        "runtime_worker_id": f"pid:{os.getpid()}",
        "evaluation_evidence_enabled": RAG_EVALUATION_EVIDENCE_ENABLED,
        "acceptance_experiment_sha256": RAG_ACCEPTANCE_EXPERIMENT_SHA256,
        "tasks": [
            {
                key: getattr(snapshot, key)
                for key in (
                    "task_id", "operation", "status", "error_code",
                    "created_at", "started_at", "finished_at",
                )
            }
            for operation in ("embed_conversation", "generate_session_title")
            for snapshot in (matches[operation][0].snapshot(),)
        ],
    }

class SessionDetailResource(SessionResource):
    conversations: list[ConversationResource]


class WizardCreateRequest(BaseModel):
    user_id: str = Field(min_length=1)
    
@router.get("/{task_id}", response_model=TaskResource)
async def get_task(
    task_id: str,
    request: Request,
    user_id: str,
    services: AppServices = Depends(get_services),
) -> TaskResource:
    correlation_id = request_id(request)
    try:
        snapshot = services.task_queue.get(task_id, user_id)
    except KeyError as exc:
        raise not_found("task", correlation_id) from exc
    except (TypeError, ValueError) as exc:
        raise validation_error(str(exc), correlation_id) from exc
    return TaskResource(**snapshot.__dict__)


__all__ = ["router"]
