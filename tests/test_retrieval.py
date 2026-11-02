from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import models

from app.services import retrieval


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

    monkeypatch.setattr(
        retrieval,
        "generate_embeddings",
        lambda texts: fake_embedding,
    )
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
    fake_query = AsyncMock(
        return_value=SimpleNamespace(points=[])
    )

    monkeypatch.setattr(
        retrieval,
        "generate_embeddings",
        lambda texts: [[0.1] * 1024],
    )
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
