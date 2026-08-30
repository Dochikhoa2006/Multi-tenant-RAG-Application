"""FastAPI dependency and common error helpers."""

from __future__ import annotations

from fastapi import Request

from backend.services import AppServices


def get_services(request: Request) -> AppServices:
    services = getattr(request.app.state, "services", None)
    if not isinstance(services, AppServices):
        raise RuntimeError("application services are not configured")
    return services


__all__ = ["get_services"]
