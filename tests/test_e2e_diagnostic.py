from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from backend.api.telemetry import TELEMETRY_SCHEMA_VERSION, TIMING_KEYS
from backend.model_config import PRIMARY_GENERATOR, SESSION_TITLE_GENERATOR
from deployment.e2e_diagnostic import (
    E2EDiagnosticError,
    E2E_PHASE,
    E2E_SCHEMA_VERSION,
    RequestRecorder,
    SelectedQuery,
    create_e2e_run,
    load_query_selection,
    update_e2e_summary,
    validate_reusable_corpus_state,
)
from deployment.e2e_diagnostic_api import (
    DeepTraceContractError,
    PostGenerationContractError,
    QueryAttemptResult,
    StreamContractError,
    TaskObservation,
    ValidatedStream,
    format_query_terminal,
    validate_post_generation_evidence,
    _task_observation,
    _DeepTraceSession,
    validate_chat_trace_evidence,
    validate_e2e_stream,
)
from deployment.wizard_diagnostic import (
    CorpusDocument,
    begin_replacement,
    mark_replacement_verified,
    new_corpus_state,
    preflight_wizard_fixtures,
    promote_replacement,
    record_replacement_document,
)


USER_ID = "wizard_diagnostic"
KNOWLEDGE_ID = "10000000-0000-0000-0000-000000000001"
POLICY_ID = "10000000-0000-0000-0000-000000000002"
KNOWLEDGE_TASK = "20000000-0000-0000-0000-000000000001"
POLICY_TASK = "20000000-0000-0000-0000-000000000002"
KNOWLEDGE_CHUNK = "30000000-0000-0000-0000-000000000001"
POLICY_CHUNK = "30000000-0000-0000-0000-000000000002"
TRACE_SESSION = "40000000-0000-0000-0000-000000000010"
TRACE_OPERATION = "40000000-0000-0000-0000-000000000011"
CHAT_SESSION = "40000000-0000-0000-0000-000000000012"
PERSISTENCE_TASK = "40000000-0000-0000-0000-000000000013"
TITLE_TASK = "40000000-0000-0000-0000-000000000014"
KNOWLEDGE_RESULT = "50000000-0000-0000-0000-000000000001"
POLICY_RESULT = "50000000-0000-0000-0000-000000000002"
ANSWER = "answer"


def _framed_digest(domain: str, value: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest()


def _valid_stream() -> ValidatedStream:
    return ValidatedStream(
        answer=ANSWER,
        request_id="60000000-0000-0000-0000-000000000001",
        conversation_id="60000000-0000-0000-0000-000000000002",
        token_event_count=1,
        answer_utf8_bytes=len(ANSWER.encode("utf-8")),
        answer_chunks_sha256=_framed_digest(
            "chat-qwen-answer-chunks-v1", ANSWER
        ),
        telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
        timings_ms={},
    )


def _valid_deep_trace() -> dict[str, object]:
    stage = {
        "call_count": 1,
        "total_ms": 1.0,
        "min_ms": 1.0,
        "max_ms": 1.0,
        "failure_count": 0,
    }
    stage_names = (
        "chat.endpoint_pre_pipeline_total",
        "chat.api_validation_session",
        "chat.runtime_setup",
        "chat.collection_factory",
        "chat.session_stream_reservation",
        "chat.collection_ensure",
        "chat.original_query_lateon",
        "chat.conversation_hybrid",
        "chat.conversation_bge",
        "chat.conversation_collapse",
        "chat.conversation_relevance_floor",
        "chat.conversation_adaptive_k",
        "chat.conversation_hydration",
        "chat.conversation_mmr",
        "chat.conversation_finalize",
        "chat.granite_budgeting",
        "chat.granite_http",
        "chat.rewritten_query_lateon",
        "chat.knowledge_policy_fork_join",
        "chat.knowledge_hybrid",
        "chat.knowledge_bge",
        "chat.knowledge_relevance_floor",
        "chat.knowledge_adaptive_k",
        "chat.knowledge_hydration",
        "chat.knowledge_mmr",
        "chat.knowledge_finalize",
        "chat.knowledge_budgeting",
        "chat.policy_hybrid",
        "chat.policy_bge",
        "chat.policy_relevance_floor",
        "chat.policy_adaptive_k",
        "chat.policy_hydration",
        "chat.policy_mmr",
        "chat.policy_finalize",
        "chat.policy_budgeting",
        "chat.qwen_prompt_construction",
        "chat.qwen_http",
        "chat.qwen_ttft",
        "chat.qwen_generation",
        "chat.qwen_stream_total",
        "chat.conversation_registry_update",
        "chat.persistence_enqueue",
        "chat.persistence_task_execution",
        "chat.persistence_segmentation",
        "chat.persistence_embeddings_fork_join",
        "chat.persistence_lateon",
        "chat.persistence_gte",
        "chat.persistence_collection_factory",
        "chat.persistence_storage_total",
        "chat.persistence_weaviate_insert",
        "chat.title_enqueue",
        "chat.title_task_execution",
        "chat.title_snapshot",
        "chat.title_transcript_render",
        "chat.title_context_build",
        "chat.title_prompt_build",
        "chat.title_runtime_setup",
        "chat.title_generation",
        "chat.title_qwen_http",
        "chat.title_qwen_total",
        "chat.title_validation",
        "chat.title_registry_update",
    )
    conversation_id = "70000000-0000-0000-0000-000000000001"
    rewritten = "standalone query"
    return {
        "schema_version": "1.0",
        "session_id": TRACE_SESSION,
        "run_id": "run",
        "user_id": USER_ID,
        "overflowed": False,
        "trace_faulted": False,
        "missing_evidence": False,
        "operations": [
            {
                "operation_id": TRACE_OPERATION,
                "kind": "chat_query",
                "user_id": USER_ID,
                "collection_type": "conversations",
                "wizard_id": CHAT_SESSION,
                "task_id": None,
                "related_tasks": {
                    "conversation_persistence": PERSISTENCE_TASK,
                    "session_title": TITLE_TASK,
                },
                "outcome": "succeeded",
                "finished_at": "2026-09-09T00:00:01Z",
                "stages": {name: dict(stage) for name in stage_names},
                "counts": {
                    "original_query_lateon_rows": 2,
                    "original_query_lateon_dimension": 128,
                    "rewritten_query_lateon_rows": 2,
                    "rewritten_query_lateon_dimension": 128,
                    "conversation_hybrid_candidate_count": 1,
                    "conversation_hybrid_storage_call_count": 1,
                    "conversation_bge_input_count": 1,
                    "conversation_bge_result_count": 1,
                    "conversation_bge_model_call_count": 1,
                    "conversation_collapse_input_count": 1,
                    "conversation_collapse_result_count": 1,
                    "conversation_relevance_input_count": 1,
                    "conversation_relevance_eligible_count": 1,
                    "conversation_adaptive_eligible_count": 1,
                    "conversation_adaptive_selected_count": 1,
                    "conversation_hydration_requested_count": 1,
                    "conversation_hydration_result_count": 1,
                    "conversation_hydration_quarantined_count": 0,
                    "conversation_hydration_storage_call_count": 1,
                    "conversation_mmr_pool_count": 1,
                    "conversation_mmr_selected_count": 1,
                    "conversation_final_count": 1,
                    "granite_input_pair_count": 1,
                    "granite_retained_pair_count": 1,
                    "granite_dropped_pair_count": 0,
                    "granite_attempt_count": 1,
                    "granite_transient_failure_count": 0,
                    "granite_transient_retry_count": 0,
                    "granite_format_failure_count": 0,
                    "granite_format_retry_count": 0,
                    "granite_usage_response_count": 1,
                    "granite_rendered_input_tokens": 20,
                    "granite_prompt_tokens_total": 20,
                    "granite_completion_tokens_total": 3,
                    "granite_final_prompt_tokens": 20,
                    "granite_final_completion_tokens": 3,
                    "granite_rewritten_query_utf8_bytes": len(rewritten),
                    "knowledge_hybrid_candidate_count": 1,
                    "knowledge_hybrid_storage_call_count": 1,
                    "knowledge_bge_input_count": 1,
                    "knowledge_bge_result_count": 1,
                    "knowledge_bge_model_call_count": 1,
                    "knowledge_relevance_input_count": 1,
                    "knowledge_relevance_eligible_count": 1,
                    "knowledge_adaptive_eligible_count": 1,
                    "knowledge_adaptive_selected_count": 1,
                    "knowledge_hydration_requested_count": 1,
                    "knowledge_hydration_result_count": 1,
                    "knowledge_hydration_quarantined_count": 0,
                    "knowledge_hydration_storage_call_count": 1,
                    "knowledge_mmr_pool_count": 1,
                    "knowledge_mmr_selected_count": 1,
                    "knowledge_prebudget_count": 1,
                    "knowledge_final_count": 1,
                    "knowledge_budget_dropped_count": 0,
                    "policy_hybrid_candidate_count": 1,
                    "policy_hybrid_storage_call_count": 1,
                    "policy_bge_input_count": 1,
                    "policy_bge_result_count": 1,
                    "policy_bge_model_call_count": 1,
                    "policy_relevance_input_count": 1,
                    "policy_relevance_eligible_count": 1,
                    "policy_adaptive_eligible_count": 1,
                    "policy_adaptive_selected_count": 1,
                    "policy_hydration_requested_count": 1,
                    "policy_hydration_result_count": 1,
                    "policy_hydration_quarantined_count": 0,
                    "policy_hydration_storage_call_count": 1,
                    "policy_mmr_pool_count": 1,
                    "policy_mmr_selected_count": 1,
                    "policy_prebudget_count": 1,
                    "policy_final_count": 1,
                    "policy_budget_dropped_count": 0,
                    "qwen_knowledge_used_count": 1,
                    "qwen_knowledge_context_utf8_bytes": 10,
                    "qwen_policy_used_count": 1,
                    "qwen_policy_context_utf8_bytes": 10,
                    "qwen_constructed_prompt_utf8_bytes": 100,
                    "qwen_stream_prompt_utf8_bytes": 100,
                    "qwen_request_prompt_utf8_bytes": 100,
                    "qwen_request_message_count": 1,
                    "qwen_request_max_tokens": PRIMARY_GENERATOR.max_output_tokens,
                    "qwen_attempt_count": 1,
                    "qwen_transient_failure_count": 0,
                    "qwen_retry_count": 0,
                    "qwen_usage_event_count": 0,
                    "qwen_answer_chunk_count": 1,
                    "qwen_answer_utf8_bytes": len(ANSWER.encode("utf-8")),
                    "sse_token_event_count": 1,
                    "sse_telemetry_event_count": 1,
                    "sse_done_event_count": 1,
                    "sse_error_event_count": 0,
                    "sse_first_token_ordinal": 1,
                    "sse_last_token_ordinal": 1,
                    "sse_telemetry_ordinal": 2,
                    "sse_done_ordinal": 3,
                    "sse_total_event_count": 3,
                    "sse_answer_utf8_bytes": len(ANSWER.encode("utf-8")),
                    "conversation_registry_update_count": 1,
                    "persistence_enqueue_count": 1,
                    "persistence_task_started_count": 1,
                    "persistence_task_completed_count": 1,
                    "persistence_segment_count": 1,
                    "persistence_lateon_document_count": 1,
                    "persistence_lateon_total_rows": 2,
                    "persistence_lateon_min_rows": 2,
                    "persistence_lateon_max_rows": 2,
                    "persistence_lateon_dimension": 128,
                    "persistence_gte_vector_count": 1,
                    "persistence_gte_dimension": 768,
                    "persistence_weaviate_expected_insert_count": 1,
                    "persistence_weaviate_insert_attempt_count": 1,
                    "persistence_weaviate_insert_success_count": 1,
                    "title_enqueue_count": 1,
                    "title_task_started_count": 1,
                    "title_task_completed_count": 1,
                    "title_registry_update_count": 1,
                    "title_snapshot_conversation_count": 1,
                    "title_conversation_count": 1,
                    "title_qwen_attempt_count": 1,
                    "title_qwen_transient_failure_count": 0,
                    "title_qwen_retry_count": 0,
                    "title_qwen_usage_response_count": 0,
                },
                "flags": {
                    "original_query_lateon_values_finite": True,
                    "rewritten_query_lateon_values_finite": True,
                    "conversation_bge_scores_finite": True,
                    "granite_input_pair_ids_proof_truncated": False,
                    "granite_retained_pair_ids_proof_truncated": False,
                    "granite_strict_json": True,
                    "granite_repair_applied": False,
                    "granite_cached_tokens_available": False,
                    "granite_rewritten_query_truncated": False,
                    "knowledge_bge_scores_finite": True,
                    "knowledge_mmr_usable": True,
                    "knowledge_hydration_values_finite": True,
                    "knowledge_mmr_fallback": False,
                    "knowledge_final_proof_truncated": False,
                    "policy_bge_scores_finite": True,
                    "policy_mmr_usable": True,
                    "policy_hydration_values_finite": True,
                    "policy_mmr_fallback": False,
                    "policy_final_proof_truncated": False,
                    "qwen_knowledge_used_proof_truncated": False,
                    "qwen_policy_used_proof_truncated": False,
                    "qwen_usage_available": False,
                    "qwen_usage_valid": True,
                    "qwen_cached_tokens_available": False,
                    "qwen_finish_seen": True,
                    "qwen_done_seen": True,
                    "sse_success_order_valid": True,
                    "persistence_enqueue_accepted": True,
                    "persistence_task_succeeded": True,
                    "persistence_lateon_values_finite": True,
                    "persistence_gte_values_finite": True,
                    "persistence_returned_conversation_id_matches": True,
                    "title_enqueue_accepted": True,
                    "title_task_succeeded": True,
                    "title_snapshot_trigger_matches": True,
                    "title_qwen_usage_available": False,
                    "title_qwen_usage_valid": True,
                    "title_qwen_cached_tokens_available": False,
                },
                "digests": {
                    "conversation_final_context_sha256": "a" * 64,
                    "granite_conversation_context_sha256": "b" * 64,
                    "granite_request_messages_sha256": "c" * 64,
                    "granite_rewritten_query_sha256": _framed_digest(
                        "chat-granite-rewritten-query-v1", rewritten
                    ),
                    "knowledge_final_context_sha256": "d" * 64,
                    "policy_final_context_sha256": "e" * 64,
                    "qwen_knowledge_context_sha256": "f" * 64,
                    "qwen_policy_context_sha256": "1" * 64,
                    "qwen_constructed_prompt_sha256": "2" * 64,
                    "qwen_stream_prompt_sha256": "2" * 64,
                    "qwen_request_prompt_sha256": "2" * 64,
                    "qwen_request_messages_sha256": "3" * 64,
                    "qwen_answer_chunks_sha256": _framed_digest(
                        "chat-qwen-answer-chunks-v1", ANSWER
                    ),
                    "sse_answer_chunks_sha256": _framed_digest(
                        "chat-qwen-answer-chunks-v1", ANSWER
                    ),
                    "persistence_payload_sha256": "4" * 64,
                    "persistence_segments_sha256": "5" * 64,
                    "persistence_segment_ids_sha256": "6" * 64,
                    "title_prompt_sha256": "7" * 64,
                    "title_qwen_prompt_sha256": "7" * 64,
                },
                "samples": {
                    "conversation_final_ids": {
                        "exact_count": 1,
                        "items": [conversation_id],
                        "truncated": False,
                    },
                    "granite_input_pair_ids": {
                        "exact_count": 1,
                        "items": [conversation_id],
                        "truncated": False,
                    },
                    "granite_retained_pair_ids": {
                        "exact_count": 1,
                        "items": [conversation_id],
                        "truncated": False,
                    },
                    "knowledge_final_ids": {
                        "exact_count": 1,
                        "items": [KNOWLEDGE_RESULT],
                        "truncated": False,
                    },
                    "knowledge_final_item_fingerprints": {
                        "exact_count": 1,
                        "items": [f"{KNOWLEDGE_RESULT}:{'4' * 64}"],
                        "truncated": False,
                    },
                    "policy_final_ids": {
                        "exact_count": 1,
                        "items": [POLICY_RESULT],
                        "truncated": False,
                    },
                    "policy_final_item_fingerprints": {
                        "exact_count": 1,
                        "items": [f"{POLICY_RESULT}:{'5' * 64}"],
                        "truncated": False,
                    },
                    "qwen_knowledge_used_ids": {
                        "exact_count": 1,
                        "items": [KNOWLEDGE_RESULT],
                        "truncated": False,
                    },
                    "qwen_knowledge_used_item_fingerprints": {
                        "exact_count": 1,
                        "items": [f"{KNOWLEDGE_RESULT}:{'4' * 64}"],
                        "truncated": False,
                    },
                    "qwen_policy_used_ids": {
                        "exact_count": 1,
                        "items": [POLICY_RESULT],
                        "truncated": False,
                    },
                    "qwen_policy_used_item_fingerprints": {
                        "exact_count": 1,
                        "items": [f"{POLICY_RESULT}:{'5' * 64}"],
                        "truncated": False,
                    },
                },
                "texts": {
                    "granite_rewritten_query": rewritten,
                    "qwen_requested_model": PRIMARY_GENERATOR.model,
                    "qwen_observed_model": PRIMARY_GENERATOR.model,
                    "qwen_finish_reason": "stop",
                    "sse_terminal_event": "done",
                    "persistence_conversation_id": _valid_stream().conversation_id,
                    "title_snapshot_last_conversation_id": _valid_stream().conversation_id,
                    "title_qwen_requested_model": SESSION_TITLE_GENERATOR.model,
                    "title_qwen_observed_model": SESSION_TITLE_GENERATOR.model,
                    "title_qwen_finish_reason": "stop",
                    "title": "Useful Retrieval Session",
                },
            }
        ],
    }


def _query_file(
    root: Path,
    body: str = 'QUERIES = ["first", "second", "first"]\n',
) -> Path:
    path = root / "queries.py"
    path.write_text(body, encoding="utf-8")
    return path


def _complete_state(root: Path):
    fixtures_root = root / "fixtures"
    (fixtures_root / "knowledge").mkdir(parents=True)
    (fixtures_root / "policy").mkdir()
    (fixtures_root / "knowledge" / "fact.txt").write_text("fact", encoding="utf-8")
    (fixtures_root / "policy" / "rule.txt").write_text("rule", encoding="utf-8")
    fixtures = preflight_wizard_fixtures(fixtures_root, root)
    state = begin_replacement(new_corpus_state(USER_ID), "generation-1", fixtures)
    for collection, wizard_id, task_id, chunk_id, fingerprint in (
        ("knowledge", KNOWLEDGE_ID, KNOWLEDGE_TASK, KNOWLEDGE_CHUNK, "a" * 64),
        ("policy", POLICY_ID, POLICY_TASK, POLICY_CHUNK, "b" * 64),
    ):
        created = CorpusDocument(collection, wizard_id)
        state = record_replacement_document(state, created)
        submitting = replace(created, status="save_submitting")
        state = record_replacement_document(state, submitting)
        submitted = replace(submitting, status="save_submitted", task_id=task_id)
        state = record_replacement_document(state, submitted)
        saved = replace(
            submitted,
            status="saved",
            chunk_ids=(chunk_id,),
            storage_fingerprint=fingerprint,
        )
        state = record_replacement_document(state, saved)
    state = mark_replacement_verified(state)
    return promote_replacement(state), fixtures


@pytest.mark.parametrize("failure", [None, "hydration", "fingerprint", "membership"])
def test_phase2_full_corpus_verification_precedes_any_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None,
) -> None:
    from deployment import e2e_diagnostic_api as api
    from deployment import wizard_diagnostic_api as wizard
    from backend.config import get_collection_name
    from backend.weaviate_client.models import ChunkRecord
    from deployment.wizard_diagnostic import WizardDiagnosticError

    state, _ = _complete_state(tmp_path)
    monkeypatch.setattr(wizard, "LATEON_EMBEDDING_DIMENSION", 1)
    monkeypatch.setattr(wizard, "GTE_EMBEDDING_DIMENSION", 1)
    documents = []
    records = {}
    for document in state.active.documents:
        count = 10_001 if document.collection == "knowledge" else 2
        base = 100_000 if document.collection == "knowledge" else 200_000
        items = tuple(
            ChunkRecord(
                str(UUID(int=base + index)), USER_ID, document.wizard_id,
                index + 1, str(UUID(int=base + index)), "persisted text",
                ((0.25,),), (0.5,),
            )
            for index in range(count)
        )
        integrity = wizard.storage_integrity(items)
        documents.append(replace(
            document, chunk_ids=integrity.chunk_ids,
            storage_fingerprint=integrity.fingerprint,
        ))
        records[document.collection] = {item.chunk_id: item for item in items}
    state = replace(state, active=replace(state.active, documents=tuple(documents)))
    before = state.payload()
    last_id = documents[0].chunk_ids[-1]
    if failure == "fingerprint":
        records["knowledge"][last_id] = replace(
            records["knowledge"][last_id], raw_text="unexpected change"
        )
    if failure == "membership":
        records["knowledge"][last_id] = replace(
            records["knowledge"][last_id], user_id="another_user"
        )
    batches = []

    def physical(collection):
        def iterator(**kwargs):
            assert kwargs == {
                "include_vector": False,
                "return_properties": ["user_id", "document_id", "chunk_id"],
            }
            for record in records[collection].values():
                yield SimpleNamespace(uuid=UUID(record.object_id), properties={
                    "user_id": record.user_id,
                    "document_id": UUID(record.document_id),
                    "chunk_id": UUID(record.chunk_id),
                })
        return SimpleNamespace(iterator=iterator)

    class Storage(wizard.CorpusStorage):
        def __enter__(self):
            physicals = {
                get_collection_name(USER_ID, kind): physical(collection)
                for collection, kind in (("knowledge", "knowledge_facts"), ("policy", "policy"))
            }
            self.manager = SimpleNamespace(client=SimpleNamespace(collections=SimpleNamespace(
                exists=lambda name: name in physicals, use=physicals.__getitem__,
            )))
            return self

        def __exit__(self, *_):
            pass

        def _collection(self, user_id, collection):
            assert user_id == USER_ID

            def fetch(ids):
                assert 0 < len(ids) <= 64
                batches.append(tuple(ids))
                return {
                    key: records[collection][key] for key in ids
                    if not (failure == "hydration" and key == last_id)
                }

            # No bounded snapshot method: the real vector-free full scan and
            # batched hydration must handle the complete physical document.
            return SimpleNamespace(_collection=physical(collection), _fetch_records_by_ids=fetch)

    monkeypatch.setattr(api, "CorpusStorage", Storage)
    if failure is None:
        assert api.verify_physical_corpus({}, state, state.active) == {
            "knowledge": 10_001, "policy": 2,
        }
        assert sum(map(len, batches)) == 10_003
    else:
        def forbidden(*args, **kwargs):
            pytest.fail("Corpus verification failure must precede HTTP/query submission")

        monkeypatch.setattr(api.httpx, "Client", forbidden)
        monkeypatch.setattr(api, "_execute_query", forbidden)
        progress = {}
        with pytest.raises((E2EDiagnosticError, WizardDiagnosticError)):
            api.run_e2e_phase_2d(
                "https://unused.invalid", {}, {}, state, state.active,
                None, None, progress, continuous=False, run_id="test",
            )
        assert progress["physical_corpus_status"] == "failed"
    assert state.payload() == before


def test_query_loader_is_static_ordered_and_preserves_exact_strings(
    tmp_path: Path,
) -> None:
    source = (
        '"""queries"""\n'
        'QUERIES: list[str] = [" first ", "duplicate", "duplicate", "雪"]\n'
    )
    path = _query_file(tmp_path, source)

    selection = load_query_selection(path, tmp_path, start=1, limit=20)

    assert selection.source_path == path.resolve()
    assert selection.source_sha256 == hashlib.sha256(source.encode()).hexdigest()
    assert selection.total_queries == 4
    assert [(item.source_index, item.question) for item in selection.selected] == [
        (1, "duplicate"),
        (2, "duplicate"),
        (3, "雪"),
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("VALUE = []\n", "only a module docstring"),
        ("QUERIES = []\n", "non-empty"),
        ("QUERIES = ('one',)\n", "non-empty list"),
        ("QUERIES = ['one', 2]\n", "must be a string"),
        ("QUERIES = ['  ']\n", "blank"),
        ("QUERIES = build_queries()\n", "literal list"),
        ("QUERIES = ['one']\nOTHER = 1\n", "only a module docstring"),
        ("QUERIES = ['one']\nQUERIES = ['two']\n", "exactly once"),
    ],
)
def test_query_loader_rejects_invalid_contract(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(E2EDiagnosticError, match=message):
        load_query_selection(_query_file(tmp_path, body), tmp_path)


def test_query_loader_rejects_missing_wrong_type_encoding_and_symlink(
    tmp_path: Path,
) -> None:
    with pytest.raises(E2EDiagnosticError, match="does not exist"):
        load_query_selection(Path("missing.py"), tmp_path)

    wrong_type = tmp_path / "queries.txt"
    wrong_type.write_text("QUERIES = ['one']\n", encoding="utf-8")
    with pytest.raises(E2EDiagnosticError, match=".py extension"):
        load_query_selection(wrong_type, tmp_path)

    invalid = tmp_path / "invalid.py"
    invalid.write_bytes(b"QUERIES = ['\xff']\n")
    with pytest.raises(E2EDiagnosticError, match="UTF-8"):
        load_query_selection(invalid, tmp_path)

    target = _query_file(tmp_path)
    link = tmp_path / "linked.py"
    link.symlink_to(target)
    with pytest.raises(E2EDiagnosticError, match="symlink"):
        load_query_selection(link, tmp_path)


@pytest.mark.parametrize(
    ("start", "limit", "message"),
    [(-1, None, "non-negative"), (0, 0, "positive"), (3, None, "outside")],
)
def test_query_selection_rejects_invalid_bounds(
    tmp_path: Path, start: int, limit: int | None, message: str
) -> None:
    with pytest.raises(E2EDiagnosticError, match=message):
        load_query_selection(
            _query_file(tmp_path), tmp_path, start=start, limit=limit
        )


def test_query_selection_defaults_to_all_remaining(tmp_path: Path) -> None:
    selection = load_query_selection(_query_file(tmp_path), tmp_path, start=1)

    assert selection.limit is None
    assert [item.source_index for item in selection.selected] == [1, 2]


def test_reusable_corpus_requires_one_settled_active_generation(tmp_path: Path) -> None:
    state, fixtures = _complete_state(tmp_path)

    active = validate_reusable_corpus_state(state, USER_ID)

    assert active is state.active
    assert {item.collection for item in active.documents} == {"knowledge", "policy"}

    with pytest.raises(E2EDiagnosticError, match="no active generation"):
        validate_reusable_corpus_state(None, USER_ID)
    with pytest.raises(E2EDiagnosticError, match="no active generation"):
        validate_reusable_corpus_state(new_corpus_state(USER_ID), USER_ID)
    with pytest.raises(E2EDiagnosticError, match="different diagnostic user"):
        validate_reusable_corpus_state(state, "another_user")

    pending = begin_replacement(state, "generation-2", fixtures)
    with pytest.raises(E2EDiagnosticError, match="interrupted replacement"):
        validate_reusable_corpus_state(pending, USER_ID)

    cleanup_document = CorpusDocument(
        "knowledge",
        "10000000-0000-0000-0000-000000000003",
        status="saved",
        task_id="20000000-0000-0000-0000-000000000003",
        chunk_ids=("30000000-0000-0000-0000-000000000003",),
        storage_fingerprint="c" * 64,
    )
    cleanup = replace(state, pending_cleanup=(cleanup_document,))
    with pytest.raises(E2EDiagnosticError, match="pending document cleanup"):
        validate_reusable_corpus_state(cleanup, USER_ID)


def test_e2e_run_and_request_artifacts_have_stable_contract(tmp_path: Path) -> None:
    state, _ = _complete_state(tmp_path)
    active = validate_reusable_corpus_state(state, USER_ID)
    selection = load_query_selection(_query_file(tmp_path), tmp_path, start=1, limit=1)
    output = tmp_path / "output"

    first = create_e2e_run(output, USER_ID, selection, active, continuous=False)
    second = create_e2e_run(output, USER_ID, selection, active, continuous=True)

    assert first.run_id != second.run_id
    assert first.requests_path.read_bytes() == b""
    summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
    assert E2E_SCHEMA_VERSION == "1.4"
    assert E2E_PHASE == "2D"
    assert summary["schema_version"] == E2E_SCHEMA_VERSION
    assert summary["phase"] == E2E_PHASE
    assert summary["session_mode"] == "fresh"
    assert summary["queries"]["selected"] == 1
    assert summary["corpus"]["generation_id"] == "generation-1"

    recorder = RequestRecorder(first.requests_path)
    recorder.record(
        source_index=1,
        status="succeeded",
        question="second",
        answer="answer",
        answer_complete=True,
        session_id="40000000-0000-0000-0000-000000000001",
        request_id="50000000-0000-0000-0000-000000000001",
        conversation_id="60000000-0000-0000-0000-000000000001",
        http_status=200,
        token_event_count=2,
        telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
        timings_ms={name: 0.0 for name in TIMING_KEYS},
        duration_ms=2.5,
        started_at="2026-09-09T00:00:00.000Z",
        diagnostic_operation_id=TRACE_OPERATION,
        deep_trace={"schema_version": "1.0", "operation": {"kind": "chat_query"}},
        client_timings_ms={"ttft_ms": 1.0, "stream_total_ms": 2.0},
        trace_polling={"poll_count": 2, "wait_ms": 0.5},
        post_generation={"conversation_persistence": {"status": "succeeded"}},
        registry_verification={"title": "Useful Retrieval Session"},
        evaluation={
            "status": "succeeded",
            "error_code": None,
            "evaluation_ms": 12.5,
            "record_sha256": "a" * 64,
        },
    )
    update_e2e_summary(
        first,
        recorder,
        status="succeeded",
        pre_down_status="succeeded",
        up_status="succeeded",
        down_status="succeeded",
        physical_corpus_status="succeeded",
        trace_status="succeeded",
        trace_deleted=True,
        finished=True,
    )

    rows = [json.loads(line) for line in first.requests_path.read_text().splitlines()]
    assert rows[0]["question"] == "second"
    assert rows[0]["answer"] == "answer"
    assert rows[0]["answer_complete"] is True
    assert rows[0]["telemetry"]["schema_version"] == TELEMETRY_SCHEMA_VERSION
    assert rows[0]["diagnostic_operation_id"] == TRACE_OPERATION
    assert rows[0]["deep_trace"]["operation"]["kind"] == "chat_query"
    assert rows[0]["query_http_attempted"] is True
    assert rows[0]["client_timings_ms"]["ttft_ms"] == 1.0
    assert rows[0]["registry_verification"]["title"] == (
        "Useful Retrieval Session"
    )
    assert rows[0]["evaluation"]["status"] == "succeeded"
    finished = json.loads(first.summary_path.read_text())
    assert finished["requests"] == {
        "selected": 1,
        "attempted": 1,
        "succeeded": 1,
        "failed": 0,
        "query_posts": 1,
        "individual_failed": 0,
        "systemic_failed": 0,
        "not_attempted": 0,
    }
    assert finished["corpus_validation"]["physical"] == "succeeded"
    assert finished["deep_trace"] == {
        "status": "succeeded",
        "session_deleted": True,
    }
    assert finished["evaluation"] == {
        "succeeded": 1,
        "partial": 0,
        "failed": 0,
        "skipped": 0,
    }
    assert finished["finished_at"].endswith("Z")
    assert not first.summary_path.with_suffix(".json.tmp").exists()


def test_request_failure_artifact_contains_only_sanitized_failure_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "requests.jsonl"
    path.touch()
    recorder = RequestRecorder(path)

    recorder.record(
        source_index=0,
        status="failed",
        question="question",
        answer="partial",
        answer_complete=False,
        session_id=None,
        request_id=None,
        conversation_id=None,
        http_status=503,
        token_event_count=1,
        telemetry_schema_version=None,
        timings_ms=None,
        duration_ms=1.0,
        started_at="2026-09-09T00:00:00.000Z",
        failure_stage="query",
        failure_code="QUERY_HTTP_ERROR",
        failure_scope="individual",
    )

    payload = json.loads(path.read_text())
    assert payload["answer"] == "partial"
    assert payload["failure_code"] == "QUERY_HTTP_ERROR"
    assert "exception" not in payload
    assert "headers" not in payload
    assert "runtime_url" not in payload


def test_request_recorder_keeps_failure_then_success_accounting(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    path.touch()
    recorder = RequestRecorder(path)
    common = {
        "answer_complete": False,
        "session_id": None,
        "request_id": None,
        "conversation_id": None,
        "http_status": None,
        "token_event_count": 0,
        "telemetry_schema_version": None,
        "timings_ms": None,
        "duration_ms": 1.0,
        "started_at": "2026-09-09T00:00:00.000Z",
    }
    recorder.record(
        source_index=0,
        status="failed",
        question="first",
        answer="",
        failure_stage="query",
        failure_code="TRANSPORT_ERROR",
        failure_scope="individual",
        **common,
    )
    recorder.record(
        source_index=1,
        status="succeeded",
        question="second",
        answer="answer",
        **{
            **common,
            "answer_complete": True,
            "session_id": "40000000-0000-0000-0000-000000000001",
            "request_id": "50000000-0000-0000-0000-000000000001",
            "conversation_id": "60000000-0000-0000-0000-000000000001",
            "http_status": 200,
            "token_event_count": 1,
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "timings_ms": {name: 0.0 for name in TIMING_KEYS},
        },
    )

    assert recorder.totals == {
        "attempted": 2,
        "succeeded": 1,
        "failed": 1,
        "query_posts": 2,
        "individual_failed": 1,
        "systemic_failed": 0,
    }
    assert [json.loads(line)["status"] for line in path.read_text().splitlines()] == [
        "failed",
        "succeeded",
    ]


def test_e2e_output_root_rejects_symlink(tmp_path: Path) -> None:
    state, _ = _complete_state(tmp_path)
    active = validate_reusable_corpus_state(state, USER_ID)
    selection = load_query_selection(_query_file(tmp_path), tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "output"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(E2EDiagnosticError, match="symlink"):
        create_e2e_run(link, USER_ID, selection, active, continuous=False)


def _valid_events(request_id: str):
    timings = {name: 0.0 for name in TIMING_KEYS}
    return [
        ("token", {"request_id": request_id, "text": "first "}),
        ("token", {"request_id": request_id, "text": "second"}),
        (
            "telemetry",
            {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "request_id": request_id,
                "timings_ms": timings,
            },
        ),
        (
            "done",
            {
                "request_id": request_id,
                "conversation_id": "60000000-0000-0000-0000-000000000001",
            },
        ),
    ]


def test_e2e_stream_validation_requires_full_correlated_public_contract() -> None:
    request_id = "50000000-0000-0000-0000-000000000001"

    result = validate_e2e_stream(_valid_events(request_id), request_id)

    assert result.answer == "first second"
    assert result.request_id == request_id
    assert result.token_event_count == 2
    assert result.answer_utf8_bytes == len(b"first second")
    expected_digest = hashlib.sha256()
    expected_digest.update(b"chat-qwen-answer-chunks-v1")
    for chunk in (b"first ", b"second"):
        expected_digest.update(len(chunk).to_bytes(8, "big"))
        expected_digest.update(chunk)
    assert result.answer_chunks_sha256 == expected_digest.hexdigest()
    assert set(result.timings_ms) == set(TIMING_KEYS)


def test_e2e_stream_validation_rejects_request_and_schema_mismatch() -> None:
    request_id = "50000000-0000-0000-0000-000000000001"
    mismatched = _valid_events(request_id)
    mismatched[0][1]["request_id"] = "50000000-0000-0000-0000-000000000002"
    with pytest.raises(StreamContractError, match="request IDs"):
        validate_e2e_stream(mismatched, request_id)

    wrong_schema = _valid_events(request_id)
    wrong_schema[-2][1]["schema_version"] = "other"
    with pytest.raises(StreamContractError, match="schema"):
        validate_e2e_stream(wrong_schema, request_id)


def test_e2e_stream_error_exposes_only_public_code() -> None:
    request_id = "50000000-0000-0000-0000-000000000001"
    events = [
        (
            "error",
            {
                "request_id": request_id,
                "code": "CHAT_PROCESSING_FAILED",
                "message": "safe public message",
            },
        )
    ]

    with pytest.raises(StreamContractError) as caught:
        validate_e2e_stream(events, request_id)

    assert caught.value.code == "CHAT_PROCESSING_FAILED"
    assert "safe public message" not in str(caught.value)


def test_phase_2c_deep_trace_contract_accepts_complete_correlated_evidence() -> None:
    payload = _valid_deep_trace()

    operation = validate_chat_trace_evidence(
        payload,
        user_id=USER_ID,
        trace_session_id=TRACE_SESSION,
        operation_id=TRACE_OPERATION,
        chat_session_id=CHAT_SESSION,
        stream=_valid_stream(),
        http_status=200,
    )

    assert operation["texts"]["granite_rewritten_query"] == "standalone query"
    assert operation["counts"]["granite_attempt_count"] == 1


def test_phase_2c_deep_trace_accepts_existing_mmr_fallback() -> None:
    payload = _valid_deep_trace()
    operation = payload["operations"][0]
    for prefix in ("knowledge", "policy"):
        operation["flags"][f"{prefix}_mmr_usable"] = False
        operation["flags"][f"{prefix}_hydration_values_finite"] = False
        operation["flags"][f"{prefix}_mmr_fallback"] = True

    validated = validate_chat_trace_evidence(
        payload,
        user_id=USER_ID,
        trace_session_id=TRACE_SESSION,
        operation_id=TRACE_OPERATION,
        chat_session_id=CHAT_SESSION,
        stream=_valid_stream(),
        http_status=200,
    )

    assert validated["flags"]["knowledge_mmr_fallback"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"overflowed": True}),
        lambda payload: payload.update({"trace_faulted": True}),
        lambda payload: payload["operations"][0]["stages"].pop(
            "chat.conversation_hydration"
        ),
        lambda payload: payload["operations"][0]["samples"][
            "granite_retained_pair_ids"
        ].update({"truncated": True}),
        lambda payload: payload["operations"][0]["counts"].update(
            {"conversation_hydration_storage_call_count": 0}
        ),
        lambda payload: payload["operations"][0]["texts"].update(
            {"granite_rewritten_query": "changed"}
        ),
        lambda payload: payload["operations"][0]["stages"].pop(
            "chat.rewritten_query_lateon"
        ),
        lambda payload: payload["operations"][0]["counts"].update(
            {"knowledge_final_count": 2}
        ),
        lambda payload: payload["operations"][0]["samples"][
            "policy_final_ids"
        ].update({"truncated": True}),
        lambda payload: payload["operations"][0]["samples"][
            "qwen_knowledge_used_ids"
        ].update({"items": [POLICY_RESULT]}),
        lambda payload: payload["operations"][0]["digests"].update(
            {"qwen_request_prompt_sha256": "9" * 64}
        ),
        lambda payload: payload["operations"][0]["counts"].update(
            {"qwen_retry_count": 1}
        ),
        lambda payload: payload["operations"][0]["flags"].update(
            {"qwen_usage_valid": False}
        ),
        lambda payload: payload["operations"][0]["texts"].update(
            {"qwen_finish_reason": "invalid"}
        ),
        lambda payload: payload["operations"][0]["counts"].update(
            {"sse_done_ordinal": 2}
        ),
        lambda payload: payload["operations"][0]["digests"].update(
            {"sse_answer_chunks_sha256": "8" * 64}
        ),
    ],
)
def test_phase_2c_deep_trace_fails_closed_on_incomplete_evidence(mutate) -> None:
    payload = _valid_deep_trace()
    mutate(payload)

    with pytest.raises(DeepTraceContractError):
        validate_chat_trace_evidence(
            payload,
            user_id=USER_ID,
            trace_session_id=TRACE_SESSION,
            operation_id=TRACE_OPERATION,
            chat_session_id=CHAT_SESSION,
            stream=_valid_stream(),
            http_status=200,
        )


@pytest.mark.parametrize("http_status", [0, 202, 500])
def test_phase_2c_deep_trace_requires_a_successful_public_stream(
    http_status: int,
) -> None:
    with pytest.raises(DeepTraceContractError):
        validate_chat_trace_evidence(
            _valid_deep_trace(),
            user_id=USER_ID,
            trace_session_id=TRACE_SESSION,
            operation_id=TRACE_OPERATION,
            chat_session_id=CHAT_SESSION,
            stream=_valid_stream(),
            http_status=http_status,
        )


def _task(
    task_id: str,
    operation: str,
    *,
    started_at: str,
    finished_at: str,
) -> TaskObservation:
    return TaskObservation(
        task_id=task_id,
        operation=operation,
        status="succeeded",
        error_code=None,
        created_at="2026-09-09T00:00:00+00:00",
        started_at=started_at,
        finished_at=finished_at,
        poll_count=2,
        queue_wait_ms=100.0,
        execution_ms=900.0,
        total_ms=1000.0,
    )


def test_phase_2d_post_generation_accepts_correlated_fifo_tasks() -> None:
    payload = _valid_deep_trace()
    operation = payload["operations"][0]
    persistence = _task(
        PERSISTENCE_TASK,
        "embed_conversation",
        started_at="2026-09-09T00:00:00.100000+00:00",
        finished_at="2026-09-09T00:00:01+00:00",
    )
    title = _task(
        TITLE_TASK,
        "generate_session_title",
        started_at="2026-09-09T00:00:01+00:00",
        finished_at="2026-09-09T00:00:02+00:00",
    )

    assert validate_post_generation_evidence(
        operation,
        conversation_id=_valid_stream().conversation_id,
        persistence_task=persistence,
        title_task=title,
    ) == "Useful Retrieval Session"


def test_phase_2d_task_status_derives_existing_queue_timings() -> None:
    observation = _task_observation(
        {
            "task_id": PERSISTENCE_TASK,
            "user_id": USER_ID,
            "operation": "embed_conversation",
            "status": "succeeded",
            "error_code": None,
            "created_at": "2026-09-09T00:00:00Z",
            "started_at": "2026-09-09T00:00:00.100000Z",
            "finished_at": "2026-09-09T00:00:01Z",
        },
        task_id=PERSISTENCE_TASK,
        user_id=USER_ID,
        expected_operation="embed_conversation",
        poll_count=3,
    )

    assert observation.queue_wait_ms == 100.0
    assert observation.execution_ms == 900.0
    assert observation.total_ms == 1000.0
    assert observation.poll_count == 3

    with pytest.raises(ValueError, match="out of order"):
        _task_observation(
            {
                "task_id": PERSISTENCE_TASK,
                "user_id": USER_ID,
                "operation": "embed_conversation",
                "status": "succeeded",
                "error_code": None,
                "created_at": "2026-09-09T00:00:01Z",
                "started_at": "2026-09-09T00:00:00Z",
                "finished_at": "2026-09-09T00:00:02Z",
            },
            task_id=PERSISTENCE_TASK,
            user_id=USER_ID,
            expected_operation="embed_conversation",
            poll_count=1,
        )


def test_phase_2d_trace_polling_tolerates_404_and_running_race() -> None:
    class Response:
        def __init__(self, status_code: int, payload: object = None) -> None:
            self.status_code = status_code
            self.payload = payload

        def json(self) -> object:
            return self.payload

    class Client:
        def __init__(self, responses: list[Response]) -> None:
            self.responses = responses
            self.calls = 0

        def get(self, *_args: object, **_kwargs: object) -> Response:
            response = self.responses[self.calls]
            self.calls += 1
            return response

    running = _valid_deep_trace()
    running["operations"][0]["outcome"] = "running"
    running["operations"][0]["finished_at"] = None
    terminal = _valid_deep_trace()
    client = Client([Response(404), Response(200, running), Response(200, terminal)])
    trace = _DeepTraceSession(
        client,  # type: ignore[arg-type]
        "https://runtime",
        USER_ID,
        "run",
    )
    trace.session_id = TRACE_SESSION

    payload, operation, polling = trace.wait_operation(
        TRACE_OPERATION,
        timeout_seconds=1.0,
        poll_seconds=0.0,
    )

    assert payload is terminal
    assert operation["outcome"] == "succeeded"
    assert polling["poll_count"] == 3
    assert client.calls == 3


@pytest.mark.parametrize(
    "mutate",
    [
        lambda operation: operation["stages"].pop("chat.persistence_gte"),
        lambda operation: operation["counts"].update(
            {"persistence_weaviate_insert_success_count": 0}
        ),
        lambda operation: operation["flags"].update(
            {"title_qwen_usage_valid": False}
        ),
        lambda operation: operation["digests"].update(
            {"title_qwen_prompt_sha256": "9" * 64}
        ),
    ],
)
def test_phase_2d_post_generation_fails_closed_on_missing_or_invalid_evidence(
    mutate,
) -> None:
    operation = _valid_deep_trace()["operations"][0]
    mutate(operation)
    persistence = _task(
        PERSISTENCE_TASK,
        "embed_conversation",
        started_at="2026-09-09T00:00:00.100000+00:00",
        finished_at="2026-09-09T00:00:01+00:00",
    )
    title = _task(
        TITLE_TASK,
        "generate_session_title",
        started_at="2026-09-09T00:00:01+00:00",
        finished_at="2026-09-09T00:00:02+00:00",
    )

    with pytest.raises((DeepTraceContractError, PostGenerationContractError)):
        validate_post_generation_evidence(
            operation,
            conversation_id=_valid_stream().conversation_id,
            persistence_task=persistence,
            title_task=title,
        )


def test_phase_2d_terminal_report_contains_required_fields_without_trace_secrets() -> None:
    operation = _valid_deep_trace()["operations"][0]
    attempt = QueryAttemptResult(
        query=SelectedQuery(3, "What is the policy?"),
        status="succeeded",
        failure_scope=None,
        failure_stage=None,
        failure_code=None,
        question="What is the policy?",
        answer="The answer.",
        answer_complete=True,
        session_id=CHAT_SESSION,
        request_id=_valid_stream().request_id,
        conversation_id=_valid_stream().conversation_id,
        diagnostic_operation_id=TRACE_OPERATION,
        http_status=200,
        query_http_attempted=True,
        token_event_count=1,
        telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
        timings_ms={},
        client_timings_ms={"ttft_ms": 12.0, "stream_total_ms": 34.0},
        trace_polling={"poll_count": 2, "wait_ms": 1.0},
        post_generation={
            "conversation_persistence": {"status": "succeeded"},
            "session_title": {"status": "succeeded"},
        },
        registry_verification={"title": "Useful Retrieval Session"},
        deep_trace={"operation": operation},
        duration_ms=50.0,
        started_at="2026-09-09T00:00:00Z",
    )

    rendered = format_query_terminal(attempt)

    for expected in (
        "What is the policy?",
        "standalone query",
        "ttft_ms=12.0",
        "Useful Retrieval Session",
        "The answer.",
    ):
        assert expected in rendered
    assert "credential" not in rendered.lower()
