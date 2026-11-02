from unittest.mock import AsyncMock

import pytest

from app.services import rag


@pytest.mark.asyncio
async def test_rejects_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        await rag.generate_rag_answer("   ")


@pytest.mark.asyncio
async def test_returns_message_when_no_chunks_found(monkeypatch):
    mock_retrieval = AsyncMock(return_value=[])

    monkeypatch.setattr(
        rag,
        "retrieve_chunks",
        mock_retrieval,
    )

    result = await rag.generate_rag_answer(question="What is the answer?")

    assert "couldn't find relevant information" in result["answer"]
    assert result["sources"] == []


@pytest.mark.asyncio
async def test_generates_answer_with_retrieved_context(monkeypatch):
    fake_chunks = [
        {
            "document_id": "document-123",
            "filename": "example.txt",
            "chunk_index": 2,
            "chunk_text": "Qdrant stores document embeddings.",
            "chunking_strategy": "recursive",
            "score": 0.85,
        }
    ]

    mock_retrieval = AsyncMock(return_value=fake_chunks)
    mock_llm = AsyncMock(return_value="Qdrant stores the document embeddings.")

    monkeypatch.setattr(
        rag,
        "retrieve_chunks",
        mock_retrieval,
    )
    monkeypatch.setattr(
        rag,
        "generate_response",
        mock_llm,
    )

    result = await rag.generate_rag_answer(
        question="Where are embeddings stored?",
        top_k=3,
    )

    assert result["answer"] == "Qdrant stores the document embeddings."

    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_id"] == "document-123"
    assert result["sources"][0]["filename"] == "example.txt"
    assert result["sources"][0]["chunk_index"] == 2
    assert result["sources"][0]["score"] == 0.85

    mock_retrieval.assert_awaited_once()
    mock_llm.assert_awaited_once()

    llm_messages = mock_llm.await_args.kwargs["messages"]

    assert "Qdrant stores document embeddings." in llm_messages[1]["content"]
    assert "Where are embeddings stored?" in llm_messages[1]["content"]


@pytest.mark.asyncio
async def test_rewrites_follow_up_question_using_history(monkeypatch):
    history = [
        {"role": "user", "content": "What vector database does this project use?"},
        {"role": "assistant", "content": "It uses Qdrant."},
    ]

    mock_llm = AsyncMock(return_value="Why was Qdrant chosen as the vector database?")

    monkeypatch.setattr(rag, "generate_response", mock_llm)

    result = await rag.rewrite_search_query(
        question="Why did they choose it?",
        conversation_history=history,
    )

    assert result == "Why was Qdrant chosen as the vector database?"
    mock_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_rag_uses_rewritten_query_and_conversation_history(monkeypatch):
    history = [
        {"role": "user", "content": "What vector database does this project use?"},
        {"role": "assistant", "content": "It uses Qdrant."},
    ]

    fake_chunks = [
        {
            "document_id": "document-123",
            "filename": "example.txt",
            "chunk_index": 2,
            "chunk_text": "Qdrant stores document embeddings.",
            "chunking_strategy": "recursive",
            "score": 0.85,
        }
    ]

    mock_retrieval = AsyncMock(return_value=fake_chunks)

    mock_llm = AsyncMock(
        side_effect=[
            "Why was Qdrant chosen as the vector database?",
            "The project uses Qdrant to store document embeddings.",
        ]
    )

    monkeypatch.setattr(rag, "retrieve_chunks", mock_retrieval)
    monkeypatch.setattr(rag, "generate_response", mock_llm)

    result = await rag.generate_rag_answer(
        question="Why did they choose it?",
        conversation_history=history,
    )

    assert result["answer"] == ("The project uses Qdrant to store document embeddings.")

    mock_retrieval.assert_awaited_once_with(
        query="Why was Qdrant chosen as the vector database?",
        top_k=None,
        document_id=None,
    )

    assert mock_llm.await_count == 2

    answer_messages = mock_llm.await_args_list[1].kwargs["messages"]

    assert history[0] in answer_messages
    assert history[1] in answer_messages
    assert "Why did they choose it?" in answer_messages[-1]["content"]
    assert "Qdrant stores document embeddings." in answer_messages[-1]["content"]
