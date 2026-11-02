from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services import document_management


@pytest.mark.asyncio
async def test_delete_document_preserves_file_outside_upload_directory(
    tmp_path,
    monkeypatch,
):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    outside_file = tmp_path / "important.txt"
    outside_file.write_text("Do not delete this file.")

    document = SimpleNamespace(
        id="document-123",
        file_path=str(outside_file),
    )

    db = SimpleNamespace(
        delete=AsyncMock(),
        commit=AsyncMock(),
    )

    mock_qdrant_delete = AsyncMock()

    monkeypatch.setattr(
        document_management.settings,
        "upload_dir",
        str(upload_dir),
    )
    monkeypatch.setattr(
        document_management,
        "get_document",
        AsyncMock(return_value=document),
    )
    monkeypatch.setattr(
        document_management.qdrant_client,
        "delete",
        mock_qdrant_delete,
    )

    result = await document_management.delete_document(
        "document-123",
        db,
    )

    assert result is document
    assert outside_file.exists()
    assert outside_file.read_text() == "Do not delete this file."

    mock_qdrant_delete.assert_awaited_once()
    db.delete.assert_awaited_once_with(document)
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_document_removes_file_inside_upload_directory(
    tmp_path,
    monkeypatch,
):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    uploaded_file = upload_dir / "policy.txt"
    uploaded_file.write_text("Policy document.")

    document = SimpleNamespace(
        id="document-456",
        file_path=str(uploaded_file),
    )

    db = SimpleNamespace(
        delete=AsyncMock(),
        commit=AsyncMock(),
    )

    monkeypatch.setattr(
        document_management.settings,
        "upload_dir",
        str(upload_dir),
    )
    monkeypatch.setattr(
        document_management,
        "get_document",
        AsyncMock(return_value=document),
    )
    monkeypatch.setattr(
        document_management.qdrant_client,
        "delete",
        AsyncMock(),
    )

    result = await document_management.delete_document(
        "document-456",
        db,
    )

    assert result is document
    assert not uploaded_file.exists()
    db.delete.assert_awaited_once_with(document)
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_all_documents_continues_after_failure(monkeypatch):
    documents = [
        SimpleNamespace(id="doc-1", filename="first.pdf"),
        SimpleNamespace(id="doc-2", filename="second.pdf"),
        SimpleNamespace(id="doc-3", filename="third.pdf"),
    ]

    db = SimpleNamespace(rollback=AsyncMock())

    async def mock_delete(document_id, db):
        if document_id == "doc-2":
            raise RuntimeError("Qdrant unavailable")

        return next(doc for doc in documents if doc.id == document_id)

    monkeypatch.setattr(
        document_management,
        "list_documents",
        AsyncMock(return_value=documents),
    )
    monkeypatch.setattr(
        document_management,
        "delete_document",
        mock_delete,
    )

    deleted, failed = await document_management.delete_all_documents(db)

    assert [doc.id for doc in deleted] == ["doc-1", "doc-3"]
    assert len(failed) == 1
    assert failed[0][0].id == "doc-2"
    assert failed[0][1] == "Qdrant unavailable"
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_all_documents_when_empty(monkeypatch):
    db = SimpleNamespace(rollback=AsyncMock())

    monkeypatch.setattr(
        document_management,
        "list_documents",
        AsyncMock(return_value=[]),
    )

    deleted, failed = await document_management.delete_all_documents(db)

    assert deleted == []
    assert failed == []
    db.rollback.assert_not_awaited()
