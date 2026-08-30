"""Wizard creation and permanent deletion orchestration."""

from __future__ import annotations

from uuid import UUID

from backend.wizard.errors import WizardDeleteError, WizardDeleteRecoveryError
from backend.wizard.runtime import WizardRuntime, resolve_runtime


def _generated_uuid(runtime: WizardRuntime, name: str) -> str:
    value = runtime.uuid_factory()
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"uuid_factory returned an invalid {name}") from exc


def create_wizard(
    user_id: str,
    collection_type: str,
    *,
    runtime: WizardRuntime | None = None,
) -> str:
    """Create one empty process-local wizard without generating embeddings."""

    active_runtime = resolve_runtime(runtime)
    document_map = active_runtime.document_map(user_id, collection_type)
    paragraph_map = active_runtime.paragraph_map(user_id, collection_type)
    document_id = _generated_uuid(active_runtime, "document_id")

    try:
        document_map.get_paragraph_data(document_id)
    except KeyError:
        pass
    else:
        raise ValueError(f"generated document_id {document_id!r} already exists")
    if paragraph_map.get_document_chunks(document_id):
        raise ValueError(f"generated document_id {document_id!r} already exists")

    document_map.create_document(document_id)
    try:
        paragraph_map.replace_document(document_id, {1: []})
    except Exception as exc:
        recovery_errors: list[BaseException] = []
        try:
            paragraph_map.delete_document(document_id)
        except Exception as recovery_exc:
            recovery_errors.append(recovery_exc)
        try:
            document_map.delete_document(document_id)
        except Exception as recovery_exc:
            recovery_errors.append(recovery_exc)
        for recovery_error in recovery_errors:
            exc.add_note(f"create-wizard cleanup failed: {recovery_error!r}")
        raise
    return document_id


def delete_wizard(
    user_id: str,
    document_id: str,
    collection_type: str,
    *,
    runtime: WizardRuntime | None = None,
) -> None:
    """Delete a wizard with verified live-process compensation.

    This is all-or-none while operations are serialized and this process stays
    alive. Weaviate does not provide a transaction spanning storage and the
    process-local maps, so process-crash atomicity is not claimed.
    """

    active_runtime = resolve_runtime(runtime)
    document_map = active_runtime.document_map(user_id, collection_type)
    paragraph_map = active_runtime.paragraph_map(user_id, collection_type)

    # Validate and snapshot both mappings before the destructive storage call.
    paragraph_data = document_map.get_paragraph_data(document_id)
    chunk_mappings = paragraph_map.get_document_chunks(document_id)
    collection = active_runtime.collection(user_id, collection_type)
    snapshots = collection.snapshot_by_document(document_id)
    failed_stage = "storage_delete"
    try:
        collection.delete_by_document(document_id)
        failed_stage = "mapping_commit"
        paragraph_map.delete_document(document_id)
        document_map.delete_document(document_id)
    except Exception as exc:
        recovery_errors: list[BaseException] = []
        try:
            collection.restore_chunks(snapshots)
        except Exception as recovery_exc:
            recovery_errors.append(recovery_exc)
        try:
            paragraph_map.replace_document(document_id, chunk_mappings)
        except Exception as recovery_exc:
            recovery_errors.append(recovery_exc)
        try:
            document_map.restore_document(document_id, paragraph_data)
        except Exception as recovery_exc:
            recovery_errors.append(recovery_exc)
        affected = [record.chunk_id for record in snapshots]
        if recovery_errors:
            raise WizardDeleteRecoveryError(
                failed_stage, affected, recovery_errors
            ) from exc
        raise WizardDeleteError(failed_stage, affected) from exc
