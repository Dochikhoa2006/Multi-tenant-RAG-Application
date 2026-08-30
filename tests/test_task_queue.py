from __future__ import annotations

import asyncio

import pytest

from backend.task_queue import InMemoryTaskQueue


def test_queue_is_fifo_per_user_and_parallel_across_users() -> None:
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        events: list[str] = []
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        other_completed = asyncio.Event()

        async def first() -> None:
            events.append("a1-start")
            first_started.set()
            await release_first.wait()
            events.append("a1-end")

        async def second() -> None:
            events.append("a2")

        async def other_user() -> None:
            events.append("b1")
            other_completed.set()

        first_id = await queue.enqueue("usr_a", "first", first)
        second_id = await queue.enqueue("usr_a", "second", second)
        other_id = await queue.enqueue("usr_b", "other", other_user)
        await first_started.wait()
        await other_completed.wait()
        assert "a2" not in events
        release_first.set()
        assert (await queue.wait(first_id, "usr_a")).status == "succeeded"
        assert (await queue.wait(second_id, "usr_a")).status == "succeeded"
        assert (await queue.wait(other_id, "usr_b")).status == "succeeded"
        assert events.index("a1-end") < events.index("a2")
        assert events.index("b1") < events.index("a1-end")
        await queue.close()

    asyncio.run(scenario())


def test_queue_tracks_safe_failures_and_logs_internal_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        queue = InMemoryTaskQueue()

        async def failing() -> None:
            raise RuntimeError("provider unavailable")

        task_id = await queue.enqueue("usr_owner", "failure", failing)
        result = await queue.wait(task_id, "usr_owner")
        assert result.status == "failed"
        assert result.error_code == "TASK_FAILED"
        assert result.error == (
            "Background operation failed. Retry the original operation."
        )
        assert "provider unavailable" not in result.error
        with pytest.raises(KeyError):
            queue.get(task_id, "usr_other")
        await queue.close()
        with pytest.raises(RuntimeError, match="shutting down"):
            await queue.enqueue("usr_owner", "late", failing)

    asyncio.run(scenario())
    record = next(
        item for item in caplog.records if getattr(item, "task_id", None)
    )
    assert record.task_id
    assert record.user_id == "usr_owner"
    assert record.operation == "failure"
    assert "provider unavailable" in caplog.text


def test_queue_cancellation_is_recorded_and_propagated() -> None:
    async def scenario() -> None:
        queue = InMemoryTaskQueue()

        async def cancelled() -> None:
            raise asyncio.CancelledError

        task_id = await queue.enqueue("usr_owner", "cancelled", cancelled)
        result = await queue.wait(task_id, "usr_owner")
        assert result.status == "failed"
        assert result.error_code == "TASK_CANCELLED"
        assert result.error == (
            "Background operation was cancelled. Retry the original operation."
        )
        with pytest.raises(asyncio.CancelledError):
            await queue._workers["usr_owner"]

    asyncio.run(scenario())


def test_queue_does_not_swallow_fatal_base_exceptions() -> None:
    class FatalSignal(BaseException):
        pass

    async def scenario() -> None:
        queue = InMemoryTaskQueue()

        async def fatal() -> None:
            raise FatalSignal("stop process")

        task_id = await queue.enqueue("usr_owner", "fatal", fatal)
        await queue._records[task_id].completed.wait()
        with pytest.raises(FatalSignal, match="stop process"):
            await queue._workers["usr_owner"]

    asyncio.run(scenario())


def test_queue_bounds_completed_history_without_evicting_pending_work() -> None:
    async def scenario() -> None:
        queue = InMemoryTaskQueue(max_completed_records=2)

        async def complete() -> None:
            return None

        completed_ids: list[str] = []
        for index in range(3):
            task_id = await queue.enqueue("usr_owner", f"complete-{index}", complete)
            await queue.wait(task_id, "usr_owner")
            completed_ids.append(task_id)

        with pytest.raises(KeyError):
            queue.get(completed_ids[0], "usr_owner")
        assert queue.get(completed_ids[1], "usr_owner").status == "succeeded"
        assert queue.get(completed_ids[2], "usr_owner").status == "succeeded"

        started = asyncio.Event()
        release = asyncio.Event()

        async def blocking() -> None:
            started.set()
            await release.wait()

        running_id = await queue.enqueue("usr_blocked", "running", blocking)
        queued_id = await queue.enqueue("usr_blocked", "queued", complete)
        await started.wait()
        for index in range(3):
            task_id = await queue.enqueue("usr_other", f"other-{index}", complete)
            await queue.wait(task_id, "usr_other")

        assert queue.get(running_id, "usr_blocked").status == "running"
        assert queue.get(queued_id, "usr_blocked").status == "queued"
        release.set()
        assert (await queue.wait(running_id, "usr_blocked")).status == "succeeded"
        assert (await queue.wait(queued_id, "usr_blocked")).status == "succeeded"
        await queue.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_queue_rejects_invalid_completed_record_limit(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        InMemoryTaskQueue(max_completed_records=value)  # type: ignore[arg-type]
