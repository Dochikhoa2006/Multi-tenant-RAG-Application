"""FastAPI application factory composing Stages 1 through 5."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.chat import router as chat_router
from backend.api.knowledge import router as knowledge_router
from backend.api.policy import router as policy_router
from backend.api.tasks import router as tasks_router
from backend.api.errors import LOGGER, public_detail, request_id
from backend.config import CORS_ALLOWED_ORIGINS
from backend.services import AppServices


def create_app(services: AppServices | None = None) -> FastAPI:
    application_services = services if services is not None else AppServices()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await asyncio.to_thread(application_services.manager.connect)
        try:
            yield
        finally:
            await application_services.task_queue.close()
            await asyncio.to_thread(application_services.manager.disconnect)

    application = FastAPI(title="RAG Application API", lifespan=lifespan)
    application.state.services = application_services

    @application.middleware("http")
    async def correlated_requests(request: Request, call_next):
        correlation_id = request_id(request)
        response = await call_next(request)
        response.headers["X-Request-ID"] = correlation_id
        return response

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "detail": public_detail(
                    "VALIDATION_ERROR",
                    "The request payload is invalid.",
                    request_id(request),
                )
            },
        )

    @application.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        correlation_id = request_id(request)
        LOGGER.exception(
            "Unhandled API error",
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={"request_id": correlation_id},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": public_detail(
                    "INTERNAL_ERROR",
                    "The request could not be completed.",
                    correlation_id,
                )
            },
        )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(CORS_ALLOWED_ORIGINS),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(chat_router)
    application.include_router(knowledge_router)
    application.include_router(policy_router)
    application.include_router(tasks_router)
    return application


app = create_app()


__all__ = ["app", "create_app"]
