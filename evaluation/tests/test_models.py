from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math

import pytest

from rag_evaluation.models import EvaluationRecord, RecordValidationError


def test_record_round_trip_preserves_unicode_and_canonical_digest(record) -> None:
    payload = record.canonical_json()
    restored = EvaluationRecord.from_json(payload)

    assert restored == record
    assert restored.knowledge_contexts[0] == "Café knowledge — exact bytes."
    assert restored.policy_contexts[0] == "Policy context 🧭"
    assert restored.sha256() == record.sha256()
    assert restored.to_mapping()["captured_at"] == "2026-09-10T16:00:00Z"


def test_record_is_deeply_immutable(record) -> None:
    with pytest.raises(FrozenInstanceError):
        record.response = "changed"
    with pytest.raises(TypeError):
        record.telemetry["new"] = "value"
    with pytest.raises(TypeError):
        record.telemetry["timings_ms"]["total"] = 0
    assert isinstance(record.telemetry["nested"], tuple)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "wizard"),
        ("request_id", "not-a-uuid"),
        ("request_id", "11111111-1111-4111-8111-11111111111A"),
        ("original_query", "  "),
        ("captured_at", "2026-09-10T12:00:00"),
    ],
)
def test_record_rejects_invalid_scalar_fields(record_mapping, field, value) -> None:
    record_mapping[field] = value
    with pytest.raises(RecordValidationError):
        EvaluationRecord.from_mapping(record_mapping)


def test_record_rejects_unknown_and_missing_fields(record_mapping) -> None:
    record_mapping["unexpected"] = True
    with pytest.raises(RecordValidationError, match="unknown"):
        EvaluationRecord.from_mapping(record_mapping)

    record_mapping.pop("unexpected")
    record_mapping.pop("response")
    with pytest.raises(RecordValidationError, match="missing"):
        EvaluationRecord.from_mapping(record_mapping)


def test_record_rejects_misaligned_or_mutated_context_evidence(record_mapping) -> None:
    record_mapping["retrieved_contexts"] = list(reversed(record_mapping["retrieved_contexts"]))
    with pytest.raises(RecordValidationError, match="Knowledge followed by Policy"):
        EvaluationRecord.from_mapping(record_mapping)

    record_mapping["retrieved_contexts"] = (
        record_mapping["knowledge_contexts"] + record_mapping["policy_contexts"]
    )
    record_mapping["context_roles"] = ["policy", "knowledge"]
    with pytest.raises(RecordValidationError, match="context_roles"):
        EvaluationRecord.from_mapping(record_mapping)


def test_record_rejects_blank_contexts_and_duplicate_ids(record_mapping) -> None:
    record_mapping["knowledge_contexts"] = ["\t"]
    record_mapping["retrieved_contexts"][0] = "\t"
    with pytest.raises(RecordValidationError, match="nonblank"):
        EvaluationRecord.from_mapping(record_mapping)

    record_mapping["knowledge_contexts"] = ["knowledge"]
    record_mapping["retrieved_contexts"] = ["knowledge", "Policy context 🧭"]
    record_mapping["policy_context_ids"] = ["knowledge-1"]
    with pytest.raises(RecordValidationError, match="unique"):
        EvaluationRecord.from_mapping(record_mapping)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_record_rejects_nonfinite_telemetry(record_mapping, value) -> None:
    record_mapping["telemetry"]["timings_ms"]["total"] = value
    with pytest.raises(RecordValidationError, match="non-finite"):
        EvaluationRecord.from_mapping(record_mapping)


def test_json_parser_rejects_nonfinite_constants(record_mapping) -> None:
    payload = json.dumps(record_mapping).replace("12.5", "NaN")
    with pytest.raises(RecordValidationError, match="invalid evaluation record JSON"):
        EvaluationRecord.from_json(payload)

