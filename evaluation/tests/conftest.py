from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from rag_evaluation.models import EvaluationRecord


@pytest.fixture
def record_mapping() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "source": "e2e",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "conversation_id": "22222222-2222-4222-8222-222222222222",
        "original_query": "What is covered?",
        "rewritten_query": "coverage facts and policy",
        "response": "The answer is grounded.",
        "knowledge_contexts": ["Café knowledge — exact bytes."],
        "knowledge_context_ids": ["knowledge-1"],
        "policy_contexts": ["Policy context 🧭"],
        "policy_context_ids": ["policy-1"],
        "retrieved_contexts": ["Café knowledge — exact bytes.", "Policy context 🧭"],
        "context_roles": ["knowledge", "policy"],
        "telemetry": {
            "schema_version": "1.0",
            "timings_ms": {"total": 12.5},
            "nested": [1, {"ok": True}],
        },
        "reference": None,
        "reference_context_ids": [],
        "captured_at": "2026-09-10T12:00:00-04:00",
    }


@pytest.fixture
def record(record_mapping: dict[str, Any]) -> EvaluationRecord:
    return EvaluationRecord.from_mapping(deepcopy(record_mapping))

