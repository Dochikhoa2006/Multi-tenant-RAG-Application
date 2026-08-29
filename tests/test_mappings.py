from __future__ import annotations

import pytest

from backend.mappings.document_map import DocumentMap
from backend.mappings.paragraph_map import ParagraphMap
from backend.mappings.session_map import SessionMap


USER_ID = "usr_abc123"
DOCUMENT_ID = "10000000-0000-0000-0000-000000000001"
CONVERSATION_ID = "20000000-0000-0000-0000-000000000001"
SECOND_CONVERSATION_ID = "20000000-0000-0000-0000-000000000002"
CHUNK_ID = "30000000-0000-0000-0000-000000000001"
SECOND_CHUNK_ID = "30000000-0000-0000-0000-000000000002"
THIRD_CHUNK_ID = "30000000-0000-0000-0000-000000000003"


def test_session_operations_and_cascade_ids() -> None:
    sessions = SessionMap(USER_ID)
    sessions.create_session("session-1")
    sessions.add_conversation("session-1", CONVERSATION_ID)
    sessions.add_conversation("session-1", SECOND_CONVERSATION_ID)
    sessions.add_conversation("session-1", CONVERSATION_ID)

    returned = sessions.get_conversations("session-1")
    returned.append("external-mutation")

    assert sessions.list_sessions(USER_ID) == ["session-1"]
    listed = sessions.list_sessions(USER_ID)
    listed.append("external-session")
    assert sessions.list_sessions(USER_ID) == ["session-1"]
    assert sessions.get_conversations("session-1") == [
        CONVERSATION_ID,
        SECOND_CONVERSATION_ID,
    ]
    assert sessions.delete_session("session-1") == [
        CONVERSATION_ID,
        SECOND_CONVERSATION_ID,
    ]
    assert sessions.list_sessions(USER_ID) == []


def test_session_duplicate_missing_and_user_isolation() -> None:
    sessions = SessionMap(USER_ID)
    sessions.create_session("session-1")

    with pytest.raises(ValueError, match="already exists"):
        sessions.create_session("session-1")
    with pytest.raises(KeyError):
        sessions.get_conversations("missing")
    with pytest.raises(KeyError):
        sessions.delete_session("missing")
    with pytest.raises(ValueError, match="does not match"):
        sessions.list_sessions("usr_other")
    with pytest.raises(ValueError, match="UUID"):
        sessions.add_conversation("session-1", "not-a-uuid")
    with pytest.raises(KeyError):
        sessions.add_conversation("missing", CONVERSATION_ID)


def test_conversation_has_only_one_session_owner_until_parent_deletion() -> None:
    sessions = SessionMap(USER_ID)
    sessions.create_session("first")
    sessions.create_session("second")
    sessions.add_conversation("first", CONVERSATION_ID)

    with pytest.raises(ValueError, match="already belongs"):
        sessions.add_conversation("second", CONVERSATION_ID)

    assert sessions.get_conversations("first") == [CONVERSATION_ID]
    assert sessions.get_conversations("second") == []
    sessions.delete_session("first")
    sessions.add_conversation("second", CONVERSATION_ID)
    assert sessions.get_conversations("second") == [CONVERSATION_ID]


def test_document_create_update_order_and_exact_text() -> None:
    documents = DocumentMap(USER_ID, "knowledge_facts")
    documents.create_document(DOCUMENT_ID)

    assert documents.get_paragraphs(DOCUMENT_ID) == [1]
    assert documents.get_full_text(DOCUMENT_ID) == ""

    documents.update_paragraphs(
        DOCUMENT_ID,
        {3: "third\n", 1: " first ", 2: "second\n\n"},
    )

    assert documents.get_paragraphs(DOCUMENT_ID) == [1, 2, 3]
    returned_paragraphs = documents.get_paragraphs(DOCUMENT_ID)
    returned_paragraphs.append(99)
    assert documents.get_paragraphs(DOCUMENT_ID) == [1, 2, 3]
    assert documents.get_full_text(DOCUMENT_ID) == " first second\n\nthird\n"


def test_document_update_validates_atomically_and_empty_resets() -> None:
    documents = DocumentMap(USER_ID, "policy")
    documents.create_document(DOCUMENT_ID)
    documents.update_paragraphs(DOCUMENT_ID, {1: "original"})

    with pytest.raises(TypeError, match="paragraph text"):
        documents.update_paragraphs(DOCUMENT_ID, {1: "replacement", 2: 3})
    assert documents.get_full_text(DOCUMENT_ID) == "original"

    documents.update_paragraphs(DOCUMENT_ID, {})
    assert documents.get_paragraphs(DOCUMENT_ID) == [1]
    assert documents.get_full_text(DOCUMENT_ID) == ""


@pytest.mark.parametrize("paragraphs", [{2: "second"}, {1: "first", 3: "third"}])
def test_document_rejects_nonsequential_paragraph_ids_atomically(
    paragraphs: dict[int, str],
) -> None:
    documents = DocumentMap(USER_ID, "knowledge_facts")
    documents.create_document(DOCUMENT_ID)
    documents.update_paragraphs(DOCUMENT_ID, {1: "original"})

    with pytest.raises(ValueError, match="sequential"):
        documents.update_paragraphs(DOCUMENT_ID, paragraphs)

    assert documents.get_paragraphs(DOCUMENT_ID) == [1]
    assert documents.get_full_text(DOCUMENT_ID) == "original"


def test_document_duplicate_missing_delete_and_type_validation() -> None:
    documents = DocumentMap(USER_ID, "knowledge_facts")
    documents.create_document(DOCUMENT_ID)

    with pytest.raises(ValueError, match="already exists"):
        documents.create_document(DOCUMENT_ID)
    documents.delete_document(DOCUMENT_ID)
    with pytest.raises(KeyError):
        documents.get_full_text(DOCUMENT_ID)
    with pytest.raises(KeyError):
        documents.delete_document(DOCUMENT_ID)
    with pytest.raises(ValueError, match="collection_type"):
        DocumentMap(USER_ID, "conversations")
    with pytest.raises(ValueError, match="UUID"):
        documents.create_document("not-a-uuid")


def test_paragraph_chunk_operations_are_ordered_and_defensive() -> None:
    paragraphs = ParagraphMap(USER_ID, "knowledge_facts")
    key = (DOCUMENT_ID, 1)
    paragraphs.set_chunks(key, [CHUNK_ID, SECOND_CHUNK_ID])

    returned = paragraphs.get_chunks(key)
    returned.append("external-mutation")

    assert paragraphs.get_chunks(key) == [CHUNK_ID, SECOND_CHUNK_ID]
    paragraphs.delete_paragraph(key)
    with pytest.raises(KeyError):
        paragraphs.delete_paragraph(key)
    with pytest.raises(KeyError):
        paragraphs.get_chunks(key)


def test_paragraph_map_rejects_invalid_keys_and_duplicate_chunks() -> None:
    paragraphs = ParagraphMap(USER_ID, "policy")

    with pytest.raises(ValueError, match="duplicates"):
        paragraphs.set_chunks((DOCUMENT_ID, 1), [CHUNK_ID, CHUNK_ID])
    with pytest.raises(ValueError, match="greater than zero"):
        paragraphs.set_chunks((DOCUMENT_ID, 0), [])
    with pytest.raises(TypeError, match="tuple"):
        paragraphs.get_chunks("document-1:1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="UUID"):
        paragraphs.set_chunks((DOCUMENT_ID, 1), ["not-a-uuid"])
    with pytest.raises(ValueError, match="UUID"):
        paragraphs.set_chunks(("not-a-uuid", 1), [])


def test_chunk_has_only_one_paragraph_owner_and_replacements_are_atomic() -> None:
    paragraphs = ParagraphMap(USER_ID, "knowledge_facts")
    first = (DOCUMENT_ID, 1)
    second = (DOCUMENT_ID, 2)
    paragraphs.set_chunks(first, [CHUNK_ID, SECOND_CHUNK_ID])
    paragraphs.set_chunks(second, [THIRD_CHUNK_ID])

    with pytest.raises(ValueError, match="already belongs"):
        paragraphs.set_chunks(second, [THIRD_CHUNK_ID, CHUNK_ID])

    assert paragraphs.get_chunks(first) == [CHUNK_ID, SECOND_CHUNK_ID]
    assert paragraphs.get_chunks(second) == [THIRD_CHUNK_ID]

    paragraphs.set_chunks(first, [SECOND_CHUNK_ID])
    paragraphs.set_chunks(second, [THIRD_CHUNK_ID, CHUNK_ID])
    assert paragraphs.get_chunks(second) == [THIRD_CHUNK_ID, CHUNK_ID]

    paragraphs.delete_paragraph(second)
    paragraphs.set_chunks(first, [SECOND_CHUNK_ID, CHUNK_ID, THIRD_CHUNK_ID])
    assert paragraphs.get_chunks(first) == [
        SECOND_CHUNK_ID,
        CHUNK_ID,
        THIRD_CHUNK_ID,
    ]


def test_document_and_paragraph_namespaces_are_isolated() -> None:
    knowledge_documents = DocumentMap(USER_ID, "knowledge_facts")
    policy_documents = DocumentMap(USER_ID, "policy")
    knowledge_paragraphs = ParagraphMap(USER_ID, "knowledge_facts")
    policy_paragraphs = ParagraphMap(USER_ID, "policy")

    knowledge_documents.create_document(DOCUMENT_ID)
    policy_documents.create_document(DOCUMENT_ID)
    knowledge_documents.update_paragraphs(DOCUMENT_ID, {1: "knowledge"})
    policy_documents.update_paragraphs(DOCUMENT_ID, {1: "policy"})
    knowledge_paragraphs.set_chunks((DOCUMENT_ID, 1), [CHUNK_ID])
    policy_paragraphs.set_chunks((DOCUMENT_ID, 1), [SECOND_CHUNK_ID])

    assert knowledge_documents.get_full_text(DOCUMENT_ID) == "knowledge"
    assert policy_documents.get_full_text(DOCUMENT_ID) == "policy"
    assert knowledge_paragraphs.get_chunks((DOCUMENT_ID, 1)) == [CHUNK_ID]
    assert policy_paragraphs.get_chunks((DOCUMENT_ID, 1)) == [SECOND_CHUNK_ID]


def test_wizard_cascade_mapping_flow_returns_all_chunk_ids() -> None:
    documents = DocumentMap(USER_ID, "knowledge_facts")
    paragraphs = ParagraphMap(USER_ID, "knowledge_facts")
    documents.create_document(DOCUMENT_ID)
    documents.update_paragraphs(DOCUMENT_ID, {1: "first", 2: "second"})
    paragraphs.set_chunks((DOCUMENT_ID, 1), [CHUNK_ID, SECOND_CHUNK_ID])
    paragraphs.set_chunks((DOCUMENT_ID, 2), [THIRD_CHUNK_ID])

    cascade_ids = [
        chunk_id
        for paragraph_id in documents.get_paragraphs(DOCUMENT_ID)
        for chunk_id in paragraphs.get_chunks((DOCUMENT_ID, paragraph_id))
    ]
    for paragraph_id in documents.get_paragraphs(DOCUMENT_ID):
        paragraphs.delete_paragraph((DOCUMENT_ID, paragraph_id))
    documents.delete_document(DOCUMENT_ID)

    assert cascade_ids == [CHUNK_ID, SECOND_CHUNK_ID, THIRD_CHUNK_ID]
    with pytest.raises(KeyError):
        paragraphs.get_chunks((DOCUMENT_ID, 1))
    with pytest.raises(KeyError):
        documents.get_paragraphs(DOCUMENT_ID)
