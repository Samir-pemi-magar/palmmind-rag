from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes import chat as chat_route
from app.main import app


@pytest.mark.asyncio
async def test_chat_endpoint_returns_booking_response(monkeypatch):
    conversation_id = "booking-api-123"

    mock_chat = AsyncMock(
        return_value={
            "conversation_id": conversation_id,
            "answer": "What is your full name?",
            "sources": [],
        }
    )

    monkeypatch.setattr(chat_route, "chat", mock_chat)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/chat",
            json={
                "message": "I want to book an interview.",
                "conversation_id": conversation_id,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": conversation_id,
        "answer": "What is your full name?",
        "sources": [],
    }

    mock_chat.assert_awaited_once_with(
        message="I want to book an interview.",
        conversation_id=conversation_id,
        document_id=None,
    )


@pytest.mark.asyncio
async def test_chat_endpoint_returns_400_for_invalid_message(monkeypatch):
    mock_chat = AsyncMock(side_effect=ValueError("Message cannot be empty"))

    monkeypatch.setattr(chat_route, "chat", mock_chat)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/chat",
            json={"message": "   "},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Message cannot be empty"


@pytest.mark.asyncio
async def test_chat_endpoint_validates_request_schema():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/chat",
            json={"message": ""},
        )

    assert response.status_code == 422
