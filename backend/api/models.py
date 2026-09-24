"""Typed HTTP request and response contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class SessionCreateRequest(BaseModel):
    user_id: str = Field(min_length=1)


def validated_document_collection_type(collection_type: object) -> str:
    normalized = required_identifier(collection_type, "collection_type").lower()
    if normalized not in DOCUMENT_COLLECTION_TYPES:
        allowed = ", ".join(sorted(DOCUMENT_COLLECTION_TYPES))
        raise ValueError(f"collection_type must be one of: {allowed}")
    return normalized


def positive_paragraph_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("paragraph_id must be an integer")
    if value <= 0:
        raise ValueError("paragraph_id must be greater than zero")
    return value

class SessionTitleRequest(BaseModel):
    user_id: str = Field(min_length=1)
    title: str = Field(min_length=1)


class ConversationResource(BaseModel):
    conversation_id: str
    question: str
    answer: str


class SessionResource(BaseModel):
    session_id: str
    user_id: str
    title: str
    conversation_count: int


class SessionDetailResource(SessionResource):
    conversations: list[ConversationResource]


class WizardCreateRequest(BaseModel):
    user_id: str = Field(min_length=1)


class WizardSaveRequest(BaseModel):
    user_id: str = Field(min_length=1)
    current_text: str
    modified_paragraph_ids: list[int] | None = None


class WizardResource(BaseModel):
    wizard_id: str
    user_id: str
    collection_type: Literal["knowledge_facts", "policy"]
    full_text: str
    paragraph_ids: list[int]


class WizardUploadResource(WizardResource):
    modified_paragraph_ids: list[int]


class TaskResource(BaseModel):
    task_id: str
    user_id: str
    operation: str
    status: Literal["queued", "running", "succeeded", "failed"]
    error_code: str | None = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


__all__ = [
    "ConversationResource",
    "QueryRequest",
    "SessionCreateRequest",
    "SessionDetailResource",
    "SessionResource",
    "SessionTitleRequest",
    "TaskResource",
    "WizardCreateRequest",
    "WizardResource",
    "WizardSaveRequest",
    "WizardUploadResource",
]
