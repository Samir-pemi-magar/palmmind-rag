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
