"""Observable process-local background task status endpoint."""

from fastapi import APIRouter, Depends, Request

from backend.api.dependencies import get_services
from backend.api.errors import not_found, request_id, validation_error
from backend.api.models import TaskResource
from backend.services import AppServices


router = APIRouter(prefix="/api/tasks", tags=["tasks"])


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
