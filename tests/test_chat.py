from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from app.services import chat as chat_service


@pytest.mark.asyncio
async def test_creates_new_conversation(monkeypatch):
    mock_history = AsyncMock(return_value=[])
    mock_rag = AsyncMock(
        return_value={
            "answer": "Qdrant stores the embeddings.",
            "sources": [{"filename": "example.txt", "chunk_index": 0}],
        }
    )
    mock_save = AsyncMock()

    monkeypatch.setattr(
        chat_service,
        "get_conversation_history",
        mock_history,
    )
    monkeypatch.setattr(
        chat_service,
        "generate_rag_answer",
        mock_rag,
    )
    monkeypatch.setattr(
        chat_service,
        "save_conversation_turn",
        mock_save,
    )

    result = await chat_service.chat(
        message="Where are the embeddings stored?"
    )

    # A new valid UUID should be generated.
    UUID(result["conversation_id"])

    assert result["answer"] == "Qdrant stores the embeddings."
    assert result["sources"] == [
        {"filename": "example.txt", "chunk_index": 0}
    ]

    mock_history.assert_awaited_once_with(result["conversation_id"])

    mock_rag.assert_awaited_once_with(
        question="Where are the embeddings stored?",
        document_id=None,
        conversation_history=[],
    )

    mock_save.assert_awaited_once_with(
        conversation_id=result["conversation_id"],
        user_message="Where are the embeddings stored?",
        assistant_message="Qdrant stores the embeddings.",
    )


@pytest.mark.asyncio
async def test_continues_existing_conversation(monkeypatch):
    conversation_id = "conversation-123"

    history = [
        {"role": "user", "content": "Which vector database is used?"},
        {"role": "assistant", "content": "Qdrant is used."},
    ]

    mock_history = AsyncMock(return_value=history)
    mock_rag = AsyncMock(
        return_value={
            "answer": "It was selected for vector similarity search.",
            "sources": [],
        }
    )
    mock_save = AsyncMock()

    monkeypatch.setattr(
        chat_service,
        "get_conversation_history",
        mock_history,
    )
    monkeypatch.setattr(
        chat_service,
        "generate_rag_answer",
        mock_rag,
    )
    monkeypatch.setattr(
        chat_service,
        "save_conversation_turn",
        mock_save,
    )

    result = await chat_service.chat(
        message="Why did they choose it?",
        conversation_id=conversation_id,
        document_id="document-123",
    )

    assert result["conversation_id"] == conversation_id

    mock_history.assert_awaited_once_with(conversation_id)

    mock_rag.assert_awaited_once_with(
        question="Why did they choose it?",
        document_id="document-123",
        conversation_history=history,
    )

    mock_save.assert_awaited_once_with(
        conversation_id=conversation_id,
        user_message="Why did they choose it?",
        assistant_message="It was selected for vector similarity search.",
    )


@pytest.mark.asyncio
async def test_rejects_empty_message():
    with pytest.raises(ValueError, match="Message cannot be empty"):
        await chat_service.chat(message="   ")


@pytest.mark.asyncio
async def test_rejects_empty_conversation_id():
    with pytest.raises(ValueError, match="Conversation ID cannot be empty"):
        await chat_service.chat(
            message="Hello",
            conversation_id="   ",
        )
