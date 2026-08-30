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
            runtime.document_map(body.user_id, collection_type).get_paragraph_data(
                wizard_id
            )
        except KeyError as exc:
            raise not_found("wizard", correlation_id) from exc
        except (TypeError, ValueError) as exc:
            raise validation_error(str(exc), correlation_id) from exc
        await _ensure_collections(services, body.user_id, correlation_id)

        async def work() -> None:
            await asyncio.to_thread(
                save_wizard,
                body.user_id,
                wizard_id,
                collection_type,
                body.current_text,
                body.modified_paragraph_ids,
                runtime=runtime,
            )

        task_id = await services.task_queue.enqueue(
            body.user_id,
            f"save_{collection_type}_wizard",
            work,
        )
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
        try:
            services.wizard_runtime.document_map(
                user_id, collection_type
            ).get_paragraph_data(wizard_id)
        except KeyError as exc:
            raise not_found("wizard", correlation_id) from exc
        except (TypeError, ValueError) as exc:
            raise validation_error(str(exc), correlation_id) from exc
        await _ensure_collections(services, user_id, correlation_id)

        async def work() -> None:
            await asyncio.to_thread(
                delete_wizard,
                user_id,
                wizard_id,
                collection_type,
                runtime=services.wizard_runtime,
            )

        task_id = await services.task_queue.enqueue(
            user_id,
            f"delete_{collection_type}_wizard",
            work,
        )
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
                paths = await asyncio.to_thread(
                    _copy_uploads_with_limits,
                    files,
                    directory,
                )
                uploaded_text = await asyncio.to_thread(_read_uploaded_text, paths)
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

        base_text = saved_text if current_text is None else current_text
        if base_text and uploaded_text:
            updated_text = base_text + TEXT_FILE_JOIN_SEPARATOR + uploaded_text
        else:
            updated_text = base_text + uploaded_text
        modified = sorted({*modified, paragraph_ids[-1]})
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
