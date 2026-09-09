"""Hidden authenticated control plane for bounded Wizard diagnostics."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from backend.api.dependencies import get_services
from backend.services import AppServices
from backend.wizard.diagnostics import (
    DiagnosticTraceRegistry,
    mapping_checkpoint,
)


class _TraceStartRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    session_id: str = Field(min_length=1, max_length=64)
    run_id: str = Field(min_length=1, max_length=256)


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")


def build_wizard_diagnostic_router(
    registry: DiagnosticTraceRegistry,
) -> APIRouter:
    """Build the three-route diagnostic API for the integrated runtime."""

    if not isinstance(registry, DiagnosticTraceRegistry):
        raise TypeError("registry must be a DiagnosticTraceRegistry")
    router = APIRouter(
        prefix="/api/_diagnostics/wizard",
        include_in_schema=False,
    )

    @router.post("/trace", status_code=status.HTTP_201_CREATED)
    async def start_trace(body: _TraceStartRequest) -> dict[str, object]:
        try:
            return registry.start(body.user_id, body.session_id, body.run_id)
        except KeyError as exc:
            raise _not_found() from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="The diagnostic session request is invalid",
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A diagnostic session is already active",
            ) from exc

    @router.get("/trace")
    async def get_trace(
        user_id: str,
        session_id: str,
        operation_id: str | None = None,
        collection_type: Literal["knowledge_facts", "policy"] | None = None,
        wizard_id: str | None = None,
        probe_chunk_ids: list[str] | None = Query(default=None),
        services: AppServices = Depends(get_services),
    ) -> dict[str, object]:
        try:
            payload = registry.snapshot(user_id, session_id, operation_id)
        except KeyError as exc:
            raise _not_found() from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="The diagnostic trace request is invalid",
            ) from exc
        if (collection_type is None) != (wizard_id is None):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="collection_type and wizard_id must be supplied together",
            )
        if wizard_id is not None and collection_type is not None:
            try:
                payload["mapping_checkpoint"] = mapping_checkpoint(
                    services.wizard_runtime,
                    user_id,
                    collection_type,
                    wizard_id,
                    probe_chunk_ids or (),
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="The mapping checkpoint request is invalid",
                ) from exc
        elif probe_chunk_ids:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="chunk probes require a Wizard target",
            )
        return payload

    @router.delete("/trace", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_trace(user_id: str, session_id: str) -> Response:
        try:
            registry.delete(user_id, session_id)
        except KeyError as exc:
            raise _not_found() from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="The diagnostic trace request is invalid",
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router


__all__ = ["build_wizard_diagnostic_router"]
