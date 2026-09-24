"""Shared Knowledge and Policy wizard endpoint implementation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
from typing import Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Query,
    Request,
    UploadFile,
    status,
)

def validated_user_id(user_id: object) -> str:
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a string")
    get_collection_name(user_id, "conversations")
    return user_id


def validated_document_collection_type(collection_type: object) -> str:
    normalized = required_identifier(collection_type, "collection_type").lower()
    if normalized not in DOCUMENT_COLLECTION_TYPES:
        allowed = ", ".join(sorted(DOCUMENT_COLLECTION_TYPES))
        raise ValueError(f"collection_type must be one of: {allowed}")
    return normalized

from backend.api.dependencies import get_services
from backend.api.errors import (
    log_internal_error,
    not_found,
    public_http_error,
    request_id,
    service_unavailable,
    validation_error,
)
from backend.api.models import (
    TaskResource,
    WizardCreateRequest,
    WizardResource,
    WizardSaveRequest,
    WizardUploadResource,
)
from backend.config import (
    SUPPORTED_FILE_EXTENSIONS,
    TEXT_FILE_JOIN_SEPARATOR,
    UPLOAD_MAX_FILE_BYTES,
    UPLOAD_MAX_TOTAL_BYTES,
    UPLOAD_READ_CHUNK_BYTES,
)
from backend.mappings._common import positive_paragraph_id, validated_user_id
from backend.processing.file_reader import read_text_files
from backend.services import AppServices
from backend.wizard.crud import create_wizard, delete_wizard
from backend.wizard.diagnostics import (
    TRACE_OPERATION_HEADER,
    TRACE_SESSION_HEADER,
    activate_operation,
    add_count,
    attach_task,
    begin_request_operation,
    fail_operation_on_error,
    finish_operation,
    observe_stage,
    operation_lifecycle,
    set_flag,
)
from backend.wizard.save import save_wizard


WizardCollectionType = Literal["knowledge_facts", "policy"]


class _UploadTooLargeError(ValueError):
    pass


class _EmptyUploadError(ValueError):
    pass


def _modified_paragraph_ids(
    raw_values: list[str] | None,
    saved_paragraph_ids: list[int],
) -> list[int]:
    values: list[object] = []
    for raw_value in raw_values or []:
        value = raw_value.strip()
        if value.startswith("["):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "modified_paragraph_ids contains malformed JSON"
                ) from exc
            if not isinstance(decoded, list):
                raise ValueError("modified_paragraph_ids JSON must be an array")
            values.extend(decoded)
        else:
            values.append(value)

    saved = set(saved_paragraph_ids)
    modified: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TypeError("modified paragraph IDs must be integers")
        try:
            paragraph_id = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("modified paragraph IDs must be integers") from exc
        paragraph_id = positive_paragraph_id(paragraph_id)
        if paragraph_id not in saved:
            raise ValueError(
                f"modified paragraph_id {paragraph_id} does not exist"
            )
        modified.add(paragraph_id)
    return sorted(modified)


def _read_uploaded_text(file_paths: list[str]) -> str:
    contents = [read_text_files([file_path]) for file_path in file_paths]
    add_count("upload.decoded_file_count", len(contents))
    add_count("upload.decoded_character_count", sum(len(item) for item in contents))
    if not any(content != "" for content in contents):
        raise _EmptyUploadError("all uploaded files are empty")
    return TEXT_FILE_JOIN_SEPARATOR.join(contents)


def _copy_uploads_with_limits(
    files: list[UploadFile],
    directory: str,
) -> list[str]:
    paths: list[str] = []
    total_bytes = 0
    for index, upload_file in enumerate(files):
        add_count("upload.file_count", 1)
        suffix = Path(upload_file.filename or "").suffix
        path = Path(directory) / f"upload-{index}{suffix}"
        file_bytes = 0
        with path.open("wb") as destination:
            while True:
                chunk = upload_file.file.read(UPLOAD_READ_CHUNK_BYTES)
                if not chunk:
                    break
                file_bytes += len(chunk)
                total_bytes += len(chunk)
                add_count("upload.byte_count", len(chunk))
                if file_bytes > UPLOAD_MAX_FILE_BYTES:
                    raise _UploadTooLargeError("file upload limit exceeded")
                if total_bytes > UPLOAD_MAX_TOTAL_BYTES:
                    raise _UploadTooLargeError("total upload limit exceeded")
                destination.write(chunk)
        paths.append(str(path))
    return paths


def _resource(
    services: AppServices,
    user_id: str,
    collection_type: WizardCollectionType,
    wizard_id: str,
) -> WizardResource:
    document_map = services.wizard_runtime.document_map(user_id, collection_type)
    return WizardResource(
        wizard_id=wizard_id,
        user_id=user_id,
        collection_type=collection_type,
        full_text=document_map.get_full_text(wizard_id),
        paragraph_ids=document_map.get_paragraphs(wizard_id),
    )


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


def build_wizard_router(
    *,
    prefix: str,
    collection_type: WizardCollectionType,
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=[collection_type])

    @router.post(
        "/wizards",
        status_code=status.HTTP_201_CREATED,
        response_model=WizardResource,
    )
    async def create(
        body: WizardCreateRequest,
        request: Request,
        services: AppServices = Depends(get_services),
    ) -> WizardResource:
        correlation_id = request_id(request)
        try:
            validated_user_id(body.user_id)
        except (TypeError, ValueError) as exc:
            raise validation_error(str(exc), correlation_id) from exc
        try:
            wizard_id = await asyncio.to_thread(
                create_wizard,
                body.user_id,
                collection_type,
                runtime=services.wizard_runtime,
            )
            return _resource(services, body.user_id, collection_type, wizard_id)
        except ValueError as exc:
            raise public_http_error(
                status.HTTP_409_CONFLICT,
                "RESOURCE_CONFLICT",
                "The wizard could not be created because its identifier conflicts.",
                correlation_id,
            ) from exc
        except TypeError as exc:
            raise validation_error(str(exc), correlation_id) from exc

    @router.get("/wizards", response_model=list[WizardResource])
    async def list_wizards(
        request: Request,
        user_id: str = Query(...),
        services: AppServices = Depends(get_services),
    ) -> list[WizardResource]:
        correlation_id = request_id(request)
        try:
            document_map = services.wizard_runtime.document_map(
                user_id, collection_type
            )
            return [
                _resource(services, user_id, collection_type, wizard_id)
                for wizard_id in document_map.list_documents()
            ]
        except (TypeError, ValueError) as exc:
            raise validation_error(str(exc), correlation_id) from exc

    @router.get("/wizards/{wizard_id}", response_model=WizardResource)
    async def get_wizard(
        wizard_id: str,
        request: Request,
        user_id: str = Query(...),
        services: AppServices = Depends(get_services),
    ) -> WizardResource:
        correlation_id = request_id(request)
        try:
            return _resource(services, user_id, collection_type, wizard_id)
        except KeyError as exc:
            raise not_found("wizard", correlation_id) from exc
        except (TypeError, ValueError) as exc:
            raise validation_error(str(exc), correlation_id) from exc

    @router.put(
        "/wizards/{wizard_id}",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=TaskResource,
    )
    async def save(
        wizard_id: str,
        body: WizardSaveRequest,
        request: Request,
        services: AppServices = Depends(get_services),
    ) -> TaskResource:
        correlation_id = request_id(request)
        trace_handle = begin_request_operation(
            user_id=body.user_id,
            session_id=request.headers.get(TRACE_SESSION_HEADER),
            operation_id=request.headers.get(TRACE_OPERATION_HEADER),
            kind="save",
            collection_type=collection_type,
            wizard_id=wizard_id,
        )
        with fail_operation_on_error(trace_handle), activate_operation(trace_handle):
            with observe_stage("api.lookup_validation"):
                try:
                    runtime = services.require_wizard_embedding()
                except RuntimeError as exc:
                    log_internal_error(
                        "Wizard embedding runtime is unavailable",
                        correlation_id,
                        user_id=body.user_id,
                        wizard_id=wizard_id,
                        collection_type=collection_type,
                    )
                    raise service_unavailable(correlation_id) from exc
                try:
                    runtime.document_map(
                        body.user_id, collection_type
                    ).get_paragraph_data(wizard_id)
                except KeyError as exc:
                    raise not_found("wizard", correlation_id) from exc
                except (TypeError, ValueError) as exc:
                    raise validation_error(str(exc), correlation_id) from exc
            with observe_stage("collection.preparation"):
                await _ensure_collections(services, body.user_id, correlation_id)

            async def work() -> None:
                with activate_operation(trace_handle):
                    try:
                        await asyncio.to_thread(
                            save_wizard,
                            body.user_id,
                            wizard_id,
                            collection_type,
                            body.current_text,
                            body.modified_paragraph_ids,
                            runtime=runtime,
                        )
                    except BaseException:
                        finish_operation(trace_handle, "failed")
                        raise
                    else:
                        finish_operation(trace_handle, "succeeded")

            with observe_stage("enqueue"):
                task_id = await services.task_queue.enqueue(
                    body.user_id,
                    f"save_{collection_type}_wizard",
                    work,
                )
            add_count("enqueue_count", 1)
            attach_task(trace_handle, task_id)
            set_flag("task_enqueued", True)
            set_flag("task_id_present", True)
            return _task_resource(services.task_queue.get(task_id, body.user_id))

    @router.delete(
        "/wizards/{wizard_id}",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=TaskResource,
    )
    async def delete(
        wizard_id: str,
        request: Request,
        user_id: str = Query(...),
        services: AppServices = Depends(get_services),
    ) -> TaskResource:
        correlation_id = request_id(request)
        trace_handle = begin_request_operation(
            user_id=user_id,
            session_id=request.headers.get(TRACE_SESSION_HEADER),
            operation_id=request.headers.get(TRACE_OPERATION_HEADER),
            kind="delete",
            collection_type=collection_type,
            wizard_id=wizard_id,
        )
        with fail_operation_on_error(trace_handle), activate_operation(trace_handle):
            with observe_stage("api.lookup_validation"):
                try:
                    services.wizard_runtime.document_map(
                        user_id, collection_type
                    ).get_paragraph_data(wizard_id)
                except KeyError as exc:
                    raise not_found("wizard", correlation_id) from exc
                except (TypeError, ValueError) as exc:
                    raise validation_error(str(exc), correlation_id) from exc
            with observe_stage("collection.preparation"):
                await _ensure_collections(services, user_id, correlation_id)

            async def work() -> None:
                with activate_operation(trace_handle):
                    try:
                        await asyncio.to_thread(
                            delete_wizard,
                            user_id,
                            wizard_id,
                            collection_type,
                            runtime=services.wizard_runtime,
                        )
                    except BaseException:
                        finish_operation(trace_handle, "failed")
                        raise
                    else:
                        finish_operation(trace_handle, "succeeded")

            with observe_stage("enqueue"):
                task_id = await services.task_queue.enqueue(
                    user_id,
                    f"delete_{collection_type}_wizard",
                    work,
                )
            add_count("enqueue_count", 1)
            attach_task(trace_handle, task_id)
            set_flag("task_enqueued", True)
            set_flag("task_id_present", True)
            return _task_resource(services.task_queue.get(task_id, user_id))

    @router.post(
        "/wizards/{wizard_id}/upload",
        response_model=WizardUploadResource,
    )
    async def upload(
        wizard_id: str,
        request: Request,
        user_id: str = Form(...),
        files: list[UploadFile] = File(...),
        current_text: str | None = Form(default=None),
        modified_paragraph_ids: list[str] | None = Form(default=None),
        services: AppServices = Depends(get_services),
    ) -> WizardUploadResource:
        correlation_id = request_id(request)
        trace_handle = begin_request_operation(
            user_id=user_id,
            session_id=request.headers.get(TRACE_SESSION_HEADER),
            operation_id=request.headers.get(TRACE_OPERATION_HEADER),
            kind="upload",
            collection_type=collection_type,
            wizard_id=wizard_id,
        )
        with (
            activate_operation(trace_handle),
            operation_lifecycle(trace_handle),
            observe_stage("upload.server_total"),
        ):
            with observe_stage("api.lookup_validation"):
                if not files:
                    raise validation_error("at least one file is required", correlation_id)
                try:
                    document_map = services.wizard_runtime.document_map(
                        user_id, collection_type
                    )
                    saved_text = document_map.get_full_text(wizard_id)
                    paragraph_ids = document_map.get_paragraphs(wizard_id)
                except KeyError as exc:
                    raise not_found("wizard", correlation_id) from exc
                except (TypeError, ValueError) as exc:
                    raise validation_error(str(exc), correlation_id) from exc

                for upload_file in files:
                    filename = upload_file.filename or ""
                    if Path(filename).suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
                        raise public_http_error(
                            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            "UNSUPPORTED_FILE_TYPE",
                            "One or more uploaded files use an unsupported type.",
                            correlation_id,
                        )

                try:
                    modified = _modified_paragraph_ids(
                        modified_paragraph_ids,
                        paragraph_ids,
                    )
                except (TypeError, ValueError) as exc:
                    raise validation_error(str(exc), correlation_id) from exc

            with tempfile.TemporaryDirectory(prefix="rag-upload-") as directory:
                try:
                    with observe_stage("upload.multipart_copy"):
                        paths = await asyncio.to_thread(
                            _copy_uploads_with_limits,
                            files,
                            directory,
                        )
                    with observe_stage("upload.file_decode_read"):
                        uploaded_text = await asyncio.to_thread(
                            _read_uploaded_text, paths
                        )
                except _UploadTooLargeError as exc:
                    raise public_http_error(
                        status.HTTP_413_CONTENT_TOO_LARGE,
                        "UPLOAD_TOO_LARGE",
                        "The uploaded files exceed the configured size limit.",
                        correlation_id,
                    ) from exc
                except _EmptyUploadError as exc:
                    raise public_http_error(
                        status.HTTP_422_UNPROCESSABLE_CONTENT,
                        "EMPTY_UPLOAD",
                        "At least one uploaded file must contain text.",
                        correlation_id,
                    ) from exc
                except UnicodeDecodeError as exc:
                    raise public_http_error(
                        status.HTTP_422_UNPROCESSABLE_CONTENT,
                        "INVALID_TEXT_ENCODING",
                        "An uploaded file is not valid configured text.",
                        correlation_id,
                    ) from exc

            with observe_stage("upload.draft_merge"):
                base_text = saved_text if current_text is None else current_text
                if base_text and uploaded_text:
                    updated_text = base_text + TEXT_FILE_JOIN_SEPARATOR + uploaded_text
                else:
                    updated_text = base_text + uploaded_text
                modified = sorted({*modified, paragraph_ids[-1]})
            set_flag("task_enqueued", False)
            set_flag("task_id_present", False)
            set_flag("save_delete_executed", False)
            set_flag("lateon_executed", False)
            set_flag("gte_executed", False)
            set_flag("weaviate_mutation_executed", False)
            return WizardUploadResource(
                wizard_id=wizard_id,
                user_id=user_id,
                collection_type=collection_type,
                full_text=updated_text,
                paragraph_ids=paragraph_ids,
                modified_paragraph_ids=modified,
            )

    return router


__all__ = ["build_wizard_router"]
