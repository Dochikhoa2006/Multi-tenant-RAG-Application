"""Safe public API errors and correlated internal logging."""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import HTTPException, Request, status


LOGGER = logging.getLogger("backend.api")


def request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    if isinstance(value, str) and value:
        return value
    generated = str(uuid4())
    request.state.request_id = generated
    return generated


def public_detail(code: str, message: str, correlation_id: str) -> dict[str, str]:
    return {
        "code": code,
        "message": message,
        "request_id": correlation_id,
    }


def public_http_error(
    status_code: int,
    code: str,
    message: str,
    correlation_id: str,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=public_detail(code, message, correlation_id),
    )


def not_found(name: str, correlation_id: str) -> HTTPException:
    return public_http_error(
        status.HTTP_404_NOT_FOUND,
        "NOT_FOUND",
        f"{name} not found",
        correlation_id,
    )


def validation_error(message: str, correlation_id: str) -> HTTPException:
    return public_http_error(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "VALIDATION_ERROR",
        message,
        correlation_id,
    )


def service_unavailable(correlation_id: str) -> HTTPException:
    return public_http_error(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "SERVICE_UNAVAILABLE",
        "A required service is temporarily unavailable.",
        correlation_id,
    )


def log_internal_error(
    message: str,
    correlation_id: str,
    **context: object,
) -> None:
    LOGGER.exception(
        message,
        extra={"request_id": correlation_id, **context},
    )


__all__ = [
    "LOGGER",
    "log_internal_error",
    "not_found",
    "public_detail",
    "public_http_error",
    "request_id",
    "service_unavailable",
    "validation_error",
]
