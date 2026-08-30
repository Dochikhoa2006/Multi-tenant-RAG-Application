"""Policy wizard endpoints."""

from backend.api._wizard import build_wizard_router


router = build_wizard_router(prefix="/api/policy", collection_type="policy")

__all__ = ["router"]
