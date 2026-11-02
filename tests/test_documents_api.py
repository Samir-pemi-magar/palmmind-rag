from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes import documents as documents_route
from app.main import app


def make_document(filename: str):
    return SimpleNamespace(
        id=f"id-{filename}",
        filename=filename,
        status="completed",
        chunking_strategy="recursive",
        total_chunks=3,
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_batch_upload_processes_multiple_files(monkeypatch):
    mock_ingest = AsyncMock(
        side_effect=[
            make_document("first.txt"),
            make_document("second.txt"),
        ]
    )
    monkeypatch.setattr(documents_route, "ingest_document", mock_ingest)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/documents/ingest/batch",
            data={"strategy": "recursive"},
            files=[
                ("uploads", ("first.txt", b"First document", "text/plain")),
                ("uploads", ("second.txt", b"Second document", "text/plain")),
            ],
        )

    assert response.status_code == 200

    result = response.json()
    assert result["total_files"] == 2
    assert len(result["successful"]) == 2
    assert result["failed"] == []
    assert result["successful"][0]["filename"] == "first.txt"
    assert result["successful"][1]["filename"] == "second.txt"
    assert mock_ingest.await_count == 2


@pytest.mark.asyncio
async def test_batch_upload_continues_after_file_failure(monkeypatch):
    mock_ingest = AsyncMock(
        side_effect=[
            ValueError("Unsupported file type."),
            make_document("second.txt"),
        ]
    )
    monkeypatch.setattr(documents_route, "ingest_document", mock_ingest)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/documents/ingest/batch",
            data={"strategy": "fixed"},
            files=[
                ("uploads", ("bad.exe", b"Invalid file", "application/octet-stream")),
                ("uploads", ("second.txt", b"Valid document", "text/plain")),
            ],
        )

    assert response.status_code == 200

    result = response.json()
    assert result["total_files"] == 2
    assert len(result["successful"]) == 1
    assert len(result["failed"]) == 1
    assert result["successful"][0]["filename"] == "second.txt"
    assert result["failed"][0] == {
        "filename": "bad.exe",
        "error": "Unsupported file type.",
    }
    assert mock_ingest.await_count == 2


@pytest.mark.asyncio
async def test_batch_upload_records_unexpected_failure(monkeypatch):
    mock_ingest = AsyncMock(side_effect=RuntimeError("Database connection details"))
    monkeypatch.setattr(documents_route, "ingest_document", mock_ingest)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/documents/ingest/batch",
            data={"strategy": "fixed"},
            files=[
                ("uploads", ("document.txt", b"Document", "text/plain")),
            ],
        )

    assert response.status_code == 200

    result = response.json()
    assert result["total_files"] == 1
    assert result["successful"] == []
    assert result["failed"][0]["filename"] == "document.txt"
    assert "Unexpected ingestion error" in result["failed"][0]["error"]


@pytest.mark.asyncio
async def test_single_document_endpoint_remains_compatible(monkeypatch):
    mock_ingest = AsyncMock(return_value=make_document("single.txt"))
    monkeypatch.setattr(documents_route, "ingest_document", mock_ingest)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/documents/ingest",
            data={"strategy": "fixed"},
            files={
                "upload": ("single.txt", b"Single document", "text/plain"),
            },
        )

    assert response.status_code == 201
    assert response.json()["filename"] == "single.txt"
    assert response.json()["chunking_strategy"] == "recursive"
    mock_ingest.assert_awaited_once()


@pytest.mark.asyncio
async def test_batch_upload_requires_files():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/documents/ingest/batch",
            data={"strategy": "fixed"},
        )

    assert response.status_code == 422
