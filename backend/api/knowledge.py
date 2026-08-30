"""Knowledge-facts wizard endpoints."""

from backend.api._wizard import build_wizard_router


router = build_wizard_router(
    prefix="/api/knowledge",
    collection_type="knowledge_facts",
)

__all__ = ["router"]
