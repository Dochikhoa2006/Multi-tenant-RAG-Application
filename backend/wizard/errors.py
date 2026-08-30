"""Typed failures for recoverable wizard storage mutations."""

from __future__ import annotations

from collections.abc import Sequence


class WizardSaveError(RuntimeError):
    """A save aborted and its previous state was successfully restored."""

    def __init__(
        self,
        failed_stage: str,
        *,
        inserted_chunk_ids: Sequence[str] = (),
        updated_chunk_ids: Sequence[str] = (),
        affected_chunk_ids: Sequence[str] = (),
    ) -> None:
        self.failed_stage = failed_stage
        self.inserted_chunk_ids = tuple(inserted_chunk_ids)
        self.updated_chunk_ids = tuple(updated_chunk_ids)
        self.affected_chunk_ids = tuple(affected_chunk_ids)
        super().__init__(f"wizard save failed during {failed_stage}; prior state restored")


class WizardSaveRecoveryError(WizardSaveError):
    """A save failed and storage or mapping compensation remained incomplete."""

    def __init__(
        self,
        failed_stage: str,
        *,
        inserted_chunk_ids: Sequence[str] = (),
        updated_chunk_ids: Sequence[str] = (),
        affected_chunk_ids: Sequence[str] = (),
        unresolved_chunk_ids: Sequence[str] = (),
        recovery_errors: Sequence[BaseException] = (),
    ) -> None:
        self.unresolved_chunk_ids = tuple(unresolved_chunk_ids)
        self.recovery_errors = tuple(recovery_errors)
        super().__init__(
            failed_stage,
            inserted_chunk_ids=inserted_chunk_ids,
            updated_chunk_ids=updated_chunk_ids,
            affected_chunk_ids=affected_chunk_ids,
        )
        self.args = (
            f"wizard save failed during {failed_stage}; recovery is incomplete; "
            f"{len(self.unresolved_chunk_ids)} storage chunk(s) unresolved",
        )


class WizardDeleteError(RuntimeError):
    """A wizard deletion aborted and its previous state was restored."""

    def __init__(self, failed_stage: str, affected_chunk_ids: Sequence[str]) -> None:
        self.failed_stage = failed_stage
        self.affected_chunk_ids = tuple(affected_chunk_ids)
        super().__init__(
            f"wizard deletion failed during {failed_stage}; prior state restored"
        )


class WizardDeleteRecoveryError(WizardDeleteError):
    """A wizard deletion failed and compensation remained incomplete."""

    def __init__(
        self,
        failed_stage: str,
        affected_chunk_ids: Sequence[str],
        recovery_errors: Sequence[BaseException],
    ) -> None:
        self.recovery_errors = tuple(recovery_errors)
        super().__init__(failed_stage, affected_chunk_ids)
        self.args = (
            f"wizard deletion failed during {failed_stage}; recovery is incomplete",
        )
