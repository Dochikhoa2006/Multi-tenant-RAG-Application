"""Tracked process-local per-user FIFO background work queue."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import logging
from uuid import uuid4

from backend.config import TASK_MAX_COMPLETED_RECORDS
from backend.mappings._common import required_identifier, validated_user_id


WorkFactory = Callable[[], Awaitable[None]]
_STOP = object()
_LOGGER = logging.getLogger("backend.task_queue")
_PUBLIC_TASK_ERROR = "Background operation failed. Retry the original operation."
_PUBLIC_TASK_CANCELLED = (
    "Background operation was cancelled. Retry the original operation."
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TaskSnapshot:
    task_id: str
    user_id: str
    operation: str
    status: str
    error_code: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


@dataclass
class _TaskRecord:
    task_id: str
    user_id: str
    operation: str
    work_factory: WorkFactory
    status: str
    error_code: str | None
    error: str | None
    created_at: datetime
    completed: asyncio.Event
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def snapshot(self) -> TaskSnapshot:
        return TaskSnapshot(
            task_id=self.task_id,
            user_id=self.user_id,
            operation=self.operation,
            status=self.status,
            error_code=self.error_code,
            error=self.error,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )


class InMemoryTaskQueue:
    """Run work serially per user while retaining observable task outcomes."""

    def __init__(
        self,
        *,
        max_completed_records: int = TASK_MAX_COMPLETED_RECORDS,
    ) -> None:
        if isinstance(max_completed_records, bool) or not isinstance(
            max_completed_records, int
        ):
            raise TypeError("max_completed_records must be an integer")
        if max_completed_records <= 0:
            raise ValueError("max_completed_records must be greater than zero")
        self._records: dict[str, _TaskRecord] = {}
        self._completed_order: deque[str] = deque()
        self._queues: dict[str, asyncio.Queue[object]] = {}
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._max_completed_records = max_completed_records
        self._accepting = True

    async def enqueue(
        self,
        user_id: str,
        operation: str,
        work_factory: WorkFactory,
    ) -> str:
        user = validated_user_id(user_id)
        operation_name = required_identifier(operation, "operation")
        if not callable(work_factory):
            raise TypeError("work_factory must be callable")
        if not self._accepting:
            raise RuntimeError("background task queue is shutting down")

        task_id = str(uuid4())
        record = _TaskRecord(
            task_id=task_id,
            user_id=user,
            operation=operation_name,
            work_factory=work_factory,
            status="queued",
            error_code=None,
            error=None,
            created_at=_utc_now(),
            completed=asyncio.Event(),
        )
        self._records[task_id] = record
        queue = self._queues.setdefault(user, asyncio.Queue())
        worker = self._workers.get(user)
        if worker is None or worker.done():
            self._workers[user] = asyncio.create_task(
                self._worker(user, queue),
                name=f"rag-user-worker-{user}",
            )
        await queue.put(task_id)
        return task_id

    async def _worker(self, user_id: str, queue: asyncio.Queue[object]) -> None:
        while True:
            item = await queue.get()
            try:
                if item is _STOP:
                    return
                record = self._records[str(item)]
                record.status = "running"
                record.started_at = _utc_now()
                try:
                    work = record.work_factory()
                    if not inspect.isawaitable(work):
                        raise TypeError("work_factory must return an awaitable")
                    await work
                except asyncio.CancelledError:
                    record.status = "failed"
                    record.error_code = "TASK_CANCELLED"
                    record.error = _PUBLIC_TASK_CANCELLED
                    _LOGGER.warning(
                        "Background task cancelled",
                        extra={
                            "task_id": record.task_id,
                            "user_id": record.user_id,
                            "operation": record.operation,
                        },
                    )
                    raise
                except Exception:
                    record.status = "failed"
                    record.error_code = "TASK_FAILED"
                    record.error = _PUBLIC_TASK_ERROR
                    _LOGGER.exception(
                        "Background task failed",
                        extra={
                            "task_id": record.task_id,
                            "user_id": record.user_id,
                            "operation": record.operation,
                        },
                    )
                else:
                    record.status = "succeeded"
                finally:
                    record.finished_at = _utc_now()
                    record.completed.set()
                    if record.status in {"succeeded", "failed"}:
                        self._retain_completed(record.task_id)
            finally:
                queue.task_done()

    def _retain_completed(self, task_id: str) -> None:
        self._completed_order.append(task_id)
        while len(self._completed_order) > self._max_completed_records:
            expired = self._completed_order.popleft()
            self._records.pop(expired, None)

    def get(self, task_id: str, user_id: str) -> TaskSnapshot:
        key = required_identifier(task_id, "task_id")
        user = validated_user_id(user_id)
        record = self._records.get(key)
        if record is None or record.user_id != user:
            raise KeyError(key)
        return record.snapshot()

    async def wait(self, task_id: str, user_id: str) -> TaskSnapshot:
        key = required_identifier(task_id, "task_id")
        user = validated_user_id(user_id)
        record = self._records.get(key)
        if record is None or record.user_id != user:
            raise KeyError(key)
        await record.completed.wait()
        return record.snapshot()

    async def close(self) -> None:
        """Stop accepting work, drain all accepted work, then stop workers."""

        if not self._accepting and not self._workers:
            return
        self._accepting = False
        queues = list(self._queues.values())
        await asyncio.gather(*(queue.join() for queue in queues))
        for queue in queues:
            await queue.put(_STOP)
        workers = list(self._workers.values())
        if workers:
            await asyncio.gather(*workers)
        self._workers.clear()
        self._queues.clear()


__all__ = ["InMemoryTaskQueue", "TaskSnapshot", "WorkFactory"]
