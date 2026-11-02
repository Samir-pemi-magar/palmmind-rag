import json
from unittest.mock import AsyncMock

import pytest

from app.services import conversation_memory as memory


class FakePipeline:
    def __init__(self):
        self.commands = []
        self.execute = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def rpush(self, *args):
        self.commands.append(("rpush", args))
        return self

    def ltrim(self, *args):
        self.commands.append(("ltrim", args))
        return self

    def expire(self, *args):
        self.commands.append(("expire", args))
        return self


@pytest.mark.asyncio
async def test_gets_conversation_history(monkeypatch):
    stored_messages = [
        json.dumps({"role": "user", "content": "What is RAG?"}),
        json.dumps(
            {
                "role": "assistant",
                "content": "RAG combines retrieval and generation.",
            }
        ),
    ]

    mock_lrange = AsyncMock(return_value=stored_messages)

    monkeypatch.setattr(memory.redis_client, "lrange", mock_lrange)

    result = await memory.get_conversation_history("conversation-123")

    assert result == [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG combines retrieval and generation."},
    ]

    mock_lrange.assert_awaited_once_with(
        "conversation:conversation-123",
        0,
        -1,
    )


@pytest.mark.asyncio
async def test_saves_conversation_turn(monkeypatch):
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(
        memory.redis_client,
        "pipeline",
        lambda transaction=True: fake_pipeline,
    )

    await memory.save_conversation_turn(
        conversation_id="conversation-123",
        user_message="What is RAG?",
        assistant_message="RAG combines retrieval and generation.",
    )

    commands = fake_pipeline.commands

    assert len(commands) == 4

    assert commands[0][0] == "rpush"
    assert commands[0][1][0] == "conversation:conversation-123"
    assert json.loads(commands[0][1][1]) == {
        "role": "user",
        "content": "What is RAG?",
    }

    assert commands[1][0] == "rpush"
    assert json.loads(commands[1][1][1]) == {
        "role": "assistant",
        "content": "RAG combines retrieval and generation.",
    }

    assert commands[2] == (
        "ltrim",
        ("conversation:conversation-123", -10, -1),
    )

    assert commands[3] == (
        "expire",
        ("conversation:conversation-123", 86400),
    )

    fake_pipeline.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_deletes_conversation_history(monkeypatch):
    mock_delete = AsyncMock(return_value=1)

    monkeypatch.setattr(
        memory.redis_client,
        "delete",
        mock_delete,
    )

    await memory.delete_conversation_history("conversation-123")

    mock_delete.assert_awaited_once_with("conversation:conversation-123")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conversation_id",
    ["", "   "],
)
async def test_rejects_empty_conversation_id(conversation_id):
    with pytest.raises(ValueError, match="Conversation ID cannot be empty"):
        await memory.get_conversation_history(conversation_id)


@pytest.mark.asyncio
async def test_rejects_empty_user_message():
    with pytest.raises(ValueError, match="User message cannot be empty"):
        await memory.save_conversation_turn(
            "conversation-123",
            "   ",
            "Hello",
        )


@pytest.mark.asyncio
async def test_rejects_empty_assistant_message():
    with pytest.raises(ValueError, match="Assistant message cannot be empty"):
        await memory.save_conversation_turn(
            "conversation-123",
            "Hello",
            "   ",
        )
