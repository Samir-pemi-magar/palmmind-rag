import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import models

from app.services import retrieval


def mock_embeddings(monkeypatch, result):
    """Replace the embedding module without loading PyTorch or Transformers."""
    fake_module = ModuleType("app.core.embeddings")
    fake_module.generate_embeddings = lambda texts: result
    monkeypatch.setitem(sys.modules, "app.core.embeddings", fake_module)


@pytest.mark.asyncio
async def test_rejects_empty_query():
    with pytest.raises(ValueError, match="Search query cannot be empty"):
        await retrieval.retrieve_chunks("   ")


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [0, -1, 101])
async def test_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k must be between 1 and 100"):
        await retrieval.retrieve_chunks("What is RAG?", top_k=top_k)


@pytest.mark.asyncio
async def test_retrieves_and_formats_chunks(monkeypatch):
    fake_embedding = [[0.1] * 1024]

    fake_point = SimpleNamespace(
        payload={
            "document_id": "document-123",
            "filename": "example.txt",
            "chunk_index": 2,
            "chunk_text": "Qdrant stores document embeddings.",
            "chunking_strategy": "recursive",
        },
        score=0.85,
    )

    fake_results = SimpleNamespace(points=[fake_point])
    fake_query = AsyncMock(return_value=fake_results)

    mock_embeddings(monkeypatch, fake_embedding)
    monkeypatch.setattr(
        retrieval.qdrant_client,
        "query_points",
        fake_query,
    )

    chunks = await retrieval.retrieve_chunks(
        query="Where are embeddings stored?",
        top_k=3,
    )

    assert len(chunks) == 1
    assert chunks[0]["document_id"] == "document-123"
    assert chunks[0]["filename"] == "example.txt"
    assert chunks[0]["chunk_index"] == 2
    assert chunks[0]["chunk_text"] == "Qdrant stores document embeddings."
    assert chunks[0]["score"] == 0.85

    fake_query.assert_awaited_once()

    call_kwargs = fake_query.await_args.kwargs
    assert call_kwargs["limit"] == 3
    assert call_kwargs["query"] == fake_embedding[0]


@pytest.mark.asyncio
async def test_applies_document_filter(monkeypatch):
    fake_query = AsyncMock(return_value=SimpleNamespace(points=[]))

    mock_embeddings(monkeypatch, [[0.1] * 1024])
    monkeypatch.setattr(
        retrieval.qdrant_client,
        "query_points",
        fake_query,
    )

    await retrieval.retrieve_chunks(
        query="Explain this document",
        document_id="document-123",
    )

    query_filter = fake_query.await_args.kwargs["query_filter"]

    assert isinstance(query_filter, models.Filter)
    assert query_filter.must[0].key == "document_id"
    assert query_filter.must[0].match.value == "document-123"


@pytest.mark.asyncio
async def test_searches_all_documents_when_document_id_is_none(monkeypatch):
    fake_points = [
        SimpleNamespace(
            payload={
                "document_id": "document-123",
                "filename": "policy.pdf",
                "chunk_index": 0,
                "chunk_text": "Employees receive annual leave.",
                "chunking_strategy": "recursive",
            },
            score=0.91,
        ),
        SimpleNamespace(
            payload={
                "document_id": "document-456",
                "filename": "handbook.txt",
                "chunk_index": 3,
                "chunk_text": "Employees receive sick leave.",
                "chunking_strategy": "fixed",
            },
            score=0.87,
        ),
    ]

    fake_query = AsyncMock(return_value=SimpleNamespace(points=fake_points))

    mock_embeddings(monkeypatch, [[0.1] * 1024])
    monkeypatch.setattr(
        retrieval.qdrant_client,
        "query_points",
        fake_query,
    )

    chunks = await retrieval.retrieve_chunks(
        query="What types of leave do employees receive?",
        document_id=None,
    )

    assert len(chunks) == 2
    assert {chunk["document_id"] for chunk in chunks} == {
        "document-123",
        "document-456",
    }

    query_filter = fake_query.await_args.kwargs["query_filter"]
    assert query_filter is None
