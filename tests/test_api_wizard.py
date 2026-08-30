from __future__ import annotations

import time
from collections.abc import Sequence
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from backend.config import TEXT_FILE_JOIN_SEPARATOR
from backend.main import create_app
from backend.services import AppServices
from backend.weaviate_client.models import ChunkRecord, DeletionReport
from backend.wizard.runtime import WizardRuntime


USER_ID = "usr_wizard_api"


class FakeManager:
    def __init__(self) -> None:
        self.ensured: list[str] = []

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def ensure_user_collections(self, user_id: str) -> None:
        self.ensured.append(user_id)


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def embed(self, text: str) -> Sequence[float]:
        self.calls.append(("embed", text))
        return [1.0, 0.0]

    def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(("embed_many", list(texts)))
        return [[1.0, 0.0] for _ in texts]


class MemoryChunkCollection:
    def __init__(self) -> None:
        self.records: dict[str, ChunkRecord] = {}
        self.fail_insert = False

    def snapshot_by_document(self, document_id: str) -> tuple[ChunkRecord, ...]:
        return tuple(
            item for item in self.records.values() if item.document_id == document_id
        )

    def snapshot_by_paragraphs(
        self, document_id: str, paragraph_ids: list[int]
    ) -> tuple[ChunkRecord, ...]:
        return tuple(
            item
            for item in self.records.values()
            if item.document_id == document_id and item.paragraph_id in paragraph_ids
        )

    def _delete(self, records: Sequence[ChunkRecord]) -> DeletionReport:
        ids = tuple(record.chunk_id for record in records)
        for chunk_id in ids:
            self.records.pop(chunk_id, None)
        return DeletionReport(len(ids), len(ids), 0, ids, ())

    def delete_by_document(self, document_id: str) -> DeletionReport:
        return self._delete(self.snapshot_by_document(document_id))

    def delete_by_paragraphs(
        self, document_id: str, paragraph_ids: list[int]
    ) -> DeletionReport:
        return self._delete(self.snapshot_by_paragraphs(document_id, paragraph_ids))

    def delete_chunks(self, chunk_ids: list[str]) -> DeletionReport:
        records = [self.records[item] for item in chunk_ids if item in self.records]
        return self._delete(records)

    def restore_chunks(self, records: Sequence[ChunkRecord]) -> tuple[str, ...]:
        for record in records:
            self.records[record.chunk_id] = record
        return tuple(record.chunk_id for record in records)

    def verify_paragraph_ids(self, expected: dict[str, int]) -> None:
        for chunk_id, paragraph_id in expected.items():
            if self.records[chunk_id].paragraph_id != paragraph_id:
                raise AssertionError("paragraph ID mismatch")

    def insert_chunk(
        self,
        document_id: str,
        paragraph_id: int,
        chunk_id: str,
        raw_text: str,
        vector: Sequence[float],
    ) -> str:
        if self.fail_insert:
            raise RuntimeError("insert failed")
        self.records[chunk_id] = ChunkRecord(
            object_id=chunk_id,
            user_id=USER_ID,
            document_id=document_id,
            paragraph_id=paragraph_id,
            chunk_id=chunk_id,
            raw_text=raw_text,
            vector=tuple(float(item) for item in vector),
        )
        return chunk_id

    def update_paragraph_ids(self, updates: dict[str, int]) -> None:
        for chunk_id, paragraph_id in updates.items():
            item = self.records[chunk_id]
            self.records[chunk_id] = ChunkRecord(
                object_id=item.object_id,
                user_id=item.user_id,
                document_id=item.document_id,
                paragraph_id=paragraph_id,
                chunk_id=item.chunk_id,
                raw_text=item.raw_text,
                vector=item.vector,
            )


def _wait(client: TestClient, task_id: str) -> dict[str, object]:
    for _ in range(100):
        result = client.get(
            f"/api/tasks/{task_id}", params={"user_id": USER_ID}
        ).json()
        if result["status"] in {"succeeded", "failed"}:
            return result
        time.sleep(0.005)
    raise AssertionError("queued wizard task did not finish")


def _harness():
    manager = FakeManager()
    collections: dict[tuple[str, str], MemoryChunkCollection] = {}

    def collection_factory(manager: object, user_id: str, collection_type: str):
        return collections.setdefault((user_id, collection_type), MemoryChunkCollection())

    embedder = FakeEmbedder()
    runtime = WizardRuntime(
        manager,  # type: ignore[arg-type]
        embedder,
        paragraph_splitter=lambda text: [text] if text else [],
        paragraph_chunker=lambda text: [text] if text else [],
        uuid_factory=uuid4,
        collection_factory=collection_factory,
    )
    services = AppServices(manager=manager, wizard_runtime=runtime)
    return services, manager, collections, embedder


def test_knowledge_wizard_create_upload_save_delete_lifecycle() -> None:
    services, manager, collections, _ = _harness()
    with TestClient(create_app(services)) as client:
        created = client.post(
            "/api/knowledge/wizards", json={"user_id": USER_ID}
        )
        assert created.status_code == 201
        wizard_id = created.json()["wizard_id"]
        assert created.json()["paragraph_ids"] == [1]
        assert created.json()["full_text"] == ""
        assert len(
            client.get(
                "/api/knowledge/wizards", params={"user_id": USER_ID}
            ).json()
        ) == 1
        assert client.get(
            "/api/policy/wizards", params={"user_id": USER_ID}
        ).json() == []

        uploaded = client.post(
            f"/api/knowledge/wizards/{wizard_id}/upload",
            data={"user_id": USER_ID, "current_text": "live editor"},
            files=[
                ("files", ("first.txt", b" first ", "text/plain")),
                ("files", ("second.MD", b"# Heading\n", "text/markdown")),
            ],
        )
        assert uploaded.status_code == 200
        expected = (
            "live editor"
            + TEXT_FILE_JOIN_SEPARATOR
            + " first "
            + TEXT_FILE_JOIN_SEPARATOR
            + "# Heading\n"
        )
        assert uploaded.json()["full_text"] == expected
        assert uploaded.json()["modified_paragraph_ids"] == [1]

        queued = client.put(
            f"/api/knowledge/wizards/{wizard_id}",
            json={
                "user_id": USER_ID,
                "current_text": expected,
            },
        )
        assert queued.status_code == 202
        assert _wait(client, queued.json()["task_id"])["status"] == "succeeded"
        fetched = client.get(
            f"/api/knowledge/wizards/{wizard_id}", params={"user_id": USER_ID}
        )
        assert fetched.json()["full_text"] == expected
        assert len(collections[(USER_ID, "knowledge_facts")].records) == 1

        appended_to_saved = client.post(
            f"/api/knowledge/wizards/{wizard_id}/upload",
            data={"user_id": USER_ID},
            files={"files": ("third.txt", b"tail", "text/plain")},
        )
        assert appended_to_saved.json()["full_text"] == (
            expected + TEXT_FILE_JOIN_SEPARATOR + "tail"
        )

        deletion = client.delete(
            f"/api/knowledge/wizards/{wizard_id}", params={"user_id": USER_ID}
        )
        assert deletion.status_code == 202
        assert _wait(client, deletion.json()["task_id"])["status"] == "succeeded"
        assert client.get(
            f"/api/knowledge/wizards/{wizard_id}", params={"user_id": USER_ID}
        ).status_code == 404
    assert manager.ensured == [USER_ID, USER_ID]


def test_policy_isolation_unsupported_upload_and_failed_task_status() -> None:
    services, _, collections, _ = _harness()
    with TestClient(create_app(services)) as client:
        created = client.post(
            "/api/policy/wizards", json={"user_id": USER_ID}
        ).json()
        wizard_id = created["wizard_id"]
        unsupported = client.post(
            f"/api/policy/wizards/{wizard_id}/upload",
            data={"user_id": USER_ID},
            files={"files": ("notes.pdf", b"pdf", "application/pdf")},
        )
        assert unsupported.status_code == 415

        collection = collections.setdefault((USER_ID, "policy"), MemoryChunkCollection())
        collection.fail_insert = True
        queued = client.put(
            f"/api/policy/wizards/{wizard_id}",
            json={
                "user_id": USER_ID,
                "current_text": "policy text",
                "modified_paragraph_ids": [1],
            },
        )
        result = _wait(client, queued.json()["task_id"])
        assert result["status"] == "failed"
        assert result["error_code"] == "TASK_FAILED"
        assert result["error"] == (
            "Background operation failed. Retry the original operation."
        )
        assert "step_7a" not in result["error"]
        assert client.get(
            f"/api/policy/wizards/{wizard_id}", params={"user_id": USER_ID}
        ).json()["full_text"] == ""
        assert client.get(
            "/api/knowledge/wizards", params={"user_id": USER_ID}
        ).json() == []


def test_wizard_save_requires_configured_embedding_provider() -> None:
    services = AppServices(manager=FakeManager())
    with TestClient(create_app(services)) as client:
        wizard_id = client.post(
            "/api/knowledge/wizards", json={"user_id": USER_ID}
        ).json()["wizard_id"]
        response = client.put(
            f"/api/knowledge/wizards/{wizard_id}",
            json={
                "user_id": USER_ID,
                "current_text": "new text",
                "modified_paragraph_ids": [1],
            },
        )
        assert response.status_code == 503
        assert response.json()["detail"] == {
            "code": "SERVICE_UNAVAILABLE",
            "message": "A required service is temporarily unavailable.",
            "request_id": response.headers["x-request-id"],
        }


def test_upload_preserves_unsaved_modified_ids_and_performs_no_persistence() -> None:
    services, _, collections, embedder = _harness()
    with TestClient(create_app(services)) as client:
        wizard_id = client.post(
            "/api/knowledge/wizards", json={"user_id": USER_ID}
        ).json()["wizard_id"]
        document_map = services.wizard_runtime.document_map(
            USER_ID, "knowledge_facts"
        )
        paragraph_map = services.wizard_runtime.paragraph_map(
            USER_ID, "knowledge_facts"
        )
        document_map.update_paragraphs(
            wizard_id,
            {1: "one", 2: "two", 3: "three"},
        )
        paragraph_map.replace_document(wizard_id, {1: [], 2: [], 3: []})
        before_documents = document_map.get_paragraph_data(wizard_id)
        before_chunks = paragraph_map.get_document_chunks(wizard_id)

        repeated = client.post(
            f"/api/knowledge/wizards/{wizard_id}/upload",
            data={
                "user_id": USER_ID,
                "current_text": "one changedtwothree",
                "modified_paragraph_ids": ["1", "2"],
            },
            files={"files": ("notes.txt", b"tail", "text/plain")},
        )
        assert repeated.status_code == 200
        assert repeated.json()["modified_paragraph_ids"] == [1, 2, 3]
        assert repeated.json()["full_text"] == (
            "one changedtwothree" + TEXT_FILE_JOIN_SEPARATOR + "tail"
        )

        json_encoded = client.post(
            f"/api/knowledge/wizards/{wizard_id}/upload",
            data={
                "user_id": USER_ID,
                "current_text": "unsaved editor",
                "modified_paragraph_ids": "[1, 2]",
            },
            files={"files": ("more.md", b"more", "text/markdown")},
        )
        assert json_encoded.status_code == 200
        assert json_encoded.json()["modified_paragraph_ids"] == [1, 2, 3]

        assert document_map.get_paragraph_data(wizard_id) == before_documents
        assert paragraph_map.get_document_chunks(wizard_id) == before_chunks
        assert collections == {}
        assert embedder.calls == []
        assert services.task_queue._records == {}


def test_upload_enforces_per_file_and_total_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services, _, _, _ = _harness()
    monkeypatch.setattr("backend.api._wizard.UPLOAD_READ_CHUNK_BYTES", 2)
    with TestClient(create_app(services)) as client:
        wizard_id = client.post(
            "/api/knowledge/wizards", json={"user_id": USER_ID}
        ).json()["wizard_id"]

        monkeypatch.setattr("backend.api._wizard.UPLOAD_MAX_FILE_BYTES", 5)
        monkeypatch.setattr("backend.api._wizard.UPLOAD_MAX_TOTAL_BYTES", 10)
        oversized_file = client.post(
            f"/api/knowledge/wizards/{wizard_id}/upload",
            data={"user_id": USER_ID},
            files={"files": ("large.txt", b"123456", "text/plain")},
        )
        assert oversized_file.status_code == 413
        assert oversized_file.json()["detail"]["code"] == "UPLOAD_TOO_LARGE"
        assert b"123456" not in oversized_file.content

        monkeypatch.setattr("backend.api._wizard.UPLOAD_MAX_FILE_BYTES", 6)
        monkeypatch.setattr("backend.api._wizard.UPLOAD_MAX_TOTAL_BYTES", 7)
        oversized_total = client.post(
            f"/api/knowledge/wizards/{wizard_id}/upload",
            data={"user_id": USER_ID},
            files=[
                ("files", ("one.txt", b"1234", "text/plain")),
                ("files", ("two.md", b"5678", "text/markdown")),
            ],
        )
        assert oversized_total.status_code == 413
        assert oversized_total.json()["detail"]["code"] == "UPLOAD_TOO_LARGE"


def test_cross_user_cannot_read_or_upload_wizard() -> None:
    services, _, _, _ = _harness()
    with TestClient(create_app(services)) as client:
        wizard_id = client.post(
            "/api/knowledge/wizards", json={"user_id": USER_ID}
        ).json()["wizard_id"]
        assert client.get(
            f"/api/knowledge/wizards/{wizard_id}",
            params={"user_id": "usr_other"},
        ).status_code == 404
        assert client.post(
            f"/api/knowledge/wizards/{wizard_id}/upload",
            data={"user_id": "usr_other"},
            files={"files": ("notes.txt", b"text", "text/plain")},
        ).status_code == 404


@pytest.mark.parametrize(
    "files",
    [
        {"files": ("empty.txt", b"", "text/plain")},
        [
            ("files", ("empty.txt", b"", "text/plain")),
            ("files", ("bom.md", b"\xef\xbb\xbf", "text/markdown")),
        ],
    ],
)
def test_empty_upload_is_rejected_without_persistence(files: object) -> None:
    services, _, collections, embedder = _harness()
    with TestClient(create_app(services)) as client:
        wizard_id = client.post(
            "/api/knowledge/wizards", json={"user_id": USER_ID}
        ).json()["wizard_id"]
        document_map = services.wizard_runtime.document_map(
            USER_ID, "knowledge_facts"
        )
        before = document_map.get_paragraph_data(wizard_id)
        response = client.post(
            f"/api/knowledge/wizards/{wizard_id}/upload",
            data={
                "user_id": USER_ID,
                "current_text": "unsaved editor",
                "modified_paragraph_ids": "1",
            },
            files=files,
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "EMPTY_UPLOAD"
        assert document_map.get_paragraph_data(wizard_id) == before
        assert collections == {}
        assert embedder.calls == []
        assert services.task_queue._records == {}


def test_whitespace_upload_is_content_and_preserves_modified_ids() -> None:
    services, _, _, _ = _harness()
    with TestClient(create_app(services)) as client:
        wizard_id = client.post(
            "/api/knowledge/wizards", json={"user_id": USER_ID}
        ).json()["wizard_id"]
        response = client.post(
            f"/api/knowledge/wizards/{wizard_id}/upload",
            data={
                "user_id": USER_ID,
                "current_text": "unsaved editor",
                "modified_paragraph_ids": "1",
            },
            files={"files": ("space.txt", b"  \n", "text/plain")},
        )
        assert response.status_code == 200
        assert response.json()["full_text"] == (
            "unsaved editor" + TEXT_FILE_JOIN_SEPARATOR + "  \n"
        )
        assert response.json()["modified_paragraph_ids"] == [1]
