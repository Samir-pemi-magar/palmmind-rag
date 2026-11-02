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
async def test_upload_endpoint_processes_multiple_files(monkeypatch):
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
            "/documents",
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
async def test_upload_endpoint_continues_after_file_failure(monkeypatch):
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
            "/documents",
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
async def test_upload_endpoint_records_unexpected_failure(monkeypatch):
    mock_ingest = AsyncMock(side_effect=RuntimeError("Database connection details"))
    monkeypatch.setattr(documents_route, "ingest_document", mock_ingest)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/documents",
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
async def test_upload_endpoint_processes_single_file(monkeypatch):
    mock_ingest = AsyncMock(return_value=make_document("single.txt"))
    monkeypatch.setattr(documents_route, "ingest_document", mock_ingest)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/documents",
            data={"strategy": "fixed"},
            files=[
                ("uploads", ("single.txt", b"Single document", "text/plain")),
            ],
        )

    assert response.status_code == 200
    result = response.json()
    assert result["total_files"] == 1
    assert len(result["successful"]) == 1
    assert result["failed"] == []
    assert result["successful"][0]["filename"] == "single.txt"
    assert result["successful"][0]["chunking_strategy"] == "recursive"
    mock_ingest.assert_awaited_once()


@pytest.mark.asyncio
async def test_upload_endpoint_requires_files():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/documents",
            data={"strategy": "fixed"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_documents(monkeypatch):
    documents = [
        make_document("first.pdf"),
        make_document("second.txt"),
    ]

    mock_list = AsyncMock(return_value=documents)
    monkeypatch.setattr(documents_route, "list_documents", mock_list)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/documents")

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0]["filename"] == "first.pdf"
    assert response.json()[1]["filename"] == "second.txt"
    mock_list.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_document_by_id(monkeypatch):
    document = make_document("policy.pdf")

    mock_get = AsyncMock(return_value=document)
    monkeypatch.setattr(documents_route, "get_document", mock_get)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/documents/id-policy.pdf")

    assert response.status_code == 200
    assert response.json()["filename"] == "policy.pdf"


@pytest.mark.asyncio
async def test_get_document_returns_404_when_missing(monkeypatch):
    mock_get = AsyncMock(return_value=None)
    monkeypatch.setattr(documents_route, "get_document", mock_get)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/documents/nonexistent")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


@pytest.mark.asyncio
async def test_delete_document(monkeypatch):
    document = make_document("policy.pdf")

    mock_delete = AsyncMock(return_value=document)
    monkeypatch.setattr(documents_route, "delete_document", mock_delete)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.delete("/documents/id-policy.pdf")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Document deleted successfully.",
        "document_id": "id-policy.pdf",
        "filename": "policy.pdf",
    }


@pytest.mark.asyncio
async def test_delete_document_returns_404_when_missing(monkeypatch):
    mock_delete = AsyncMock(return_value=None)
    monkeypatch.setattr(documents_route, "delete_document", mock_delete)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.delete("/documents/nonexistent")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."
