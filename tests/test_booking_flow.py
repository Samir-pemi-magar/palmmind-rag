from datetime import UTC
from unittest.mock import AsyncMock, Mock

import pytest

from app.services import booking as booking_service


@pytest.fixture
def booking_state_store(monkeypatch):
    """Simulate Redis booking state for isolated orchestration tests."""

    states = {}

    async def mock_get(conversation_id):
        return states.get(conversation_id, {}).copy()

    async def mock_save(conversation_id, details):
        current = states.get(conversation_id, {}).copy()

        for field, value in details.items():
            if value is not None:
                current[field] = value

        states[conversation_id] = current
        return current.copy()

    async def mock_clear(conversation_id):
        states.pop(conversation_id, None)

    monkeypatch.setattr(
        booking_service,
        "get_booking_state",
        mock_get,
    )
    monkeypatch.setattr(
        booking_service,
        "save_booking_state",
        mock_save,
    )
    monkeypatch.setattr(
        booking_service,
        "clear_booking_state",
        mock_clear,
    )

    return states


def extraction(
    *,
    booking_intent=True,
    candidate_name=None,
    candidate_email=None,
    date=None,
    time=None,
    timezone=None,
):
    """Build a response matching the booking extractor's output."""

    return {
        "booking_intent": booking_intent,
        "candidate_name": candidate_name,
        "candidate_email": candidate_email,
        "date": date,
        "time": time,
        "timezone": timezone,
    }


@pytest.mark.asyncio
async def test_starting_booking_activates_state(
    monkeypatch,
    booking_state_store,
):
    monkeypatch.setattr(
        booking_service,
        "extract_booking_details",
        AsyncMock(return_value=extraction()),
    )

    result = await booking_service.handle_booking_message(
        message="I want to book an interview.",
        conversation_id="booking-001",
    )

    assert result["answer"] == "What is your full name?"
    assert booking_state_store["booking-001"]["booking_active"] is True
    assert booking_state_store["booking-001"].get("awaiting_confirmation") is not True


@pytest.mark.asyncio
async def test_short_answer_continues_active_booking(
    monkeypatch,
    booking_state_store,
):
    conversation_id = "booking-002"

    booking_state_store[conversation_id] = {
        "booking_active": True,
    }

    mock_extractor = AsyncMock(return_value=extraction(candidate_name="Alex Smith"))

    monkeypatch.setattr(
        booking_service,
        "extract_booking_details",
        mock_extractor,
    )

    result = await booking_service.handle_booking_message(
        message="Alex Smith",
        conversation_id=conversation_id,
    )

    assert result["answer"] == ("What email address should we use for your interview?")

    assert booking_state_store[conversation_id]["booking_active"] is True
    assert booking_state_store[conversation_id]["candidate_name"] == "Alex Smith"

    mock_extractor.assert_awaited_once_with(
        "Alex Smith",
        current_state={"booking_active": True},
    )


@pytest.mark.asyncio
async def test_unrelated_question_preserves_active_booking(
    monkeypatch,
    booking_state_store,
):
    conversation_id = "booking-003"

    original_state = {
        "booking_active": True,
        "candidate_name": "Alex Smith",
    }
    booking_state_store[conversation_id] = original_state.copy()

    mock_extractor = AsyncMock(return_value=extraction(booking_intent=False))

    monkeypatch.setattr(
        booking_service,
        "extract_booking_details",
        mock_extractor,
    )

    result = await booking_service.handle_booking_message(
        message="What is a vector database?",
        conversation_id=conversation_id,
    )

    assert result is None
    assert booking_state_store[conversation_id] == original_state


@pytest.mark.asyncio
async def test_cancellation_clears_booking_before_confirmation(
    monkeypatch,
    booking_state_store,
):
    conversation_id = "booking-004"

    booking_state_store[conversation_id] = {
        "booking_active": True,
        "candidate_name": "Alex Smith",
        "candidate_email": "alex@example.com",
    }

    mock_extractor = AsyncMock()

    monkeypatch.setattr(
        booking_service,
        "extract_booking_details",
        mock_extractor,
    )

    result = await booking_service.handle_booking_message(
        message="Cancel",
        conversation_id=conversation_id,
    )

    assert "cancelled" in result["answer"].lower()
    assert conversation_id not in booking_state_store
    mock_extractor.assert_not_awaited()


@pytest.mark.asyncio
async def test_complete_details_request_confirmation(
    monkeypatch,
    booking_state_store,
):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    conversation_id = "booking-005"

    mock_extractor = AsyncMock(
        return_value=extraction(
            candidate_name="Alex Smith",
            candidate_email="alex@example.com",
            date="November 12, 2026",
            time="10 AM",
            timezone="Asia/Kathmandu",
        )
    )

    mock_parse = Mock(
        return_value=datetime(
            2026,
            11,
            12,
            4,
            15,
            tzinfo=UTC,
        )
    )

    mock_email = Mock(return_value="alex@example.com")

    monkeypatch.setattr(
        booking_service,
        "extract_booking_details",
        mock_extractor,
    )
    monkeypatch.setattr(
        booking_service,
        "parse_scheduled_datetime",
        mock_parse,
    )
    monkeypatch.setattr(
        booking_service,
        "validate_candidate_email",
        mock_email,
    )
    monkeypatch.setattr(
        booking_service,
        "get_valid_timezone",
        lambda name: ZoneInfo(name),
    )

    result = await booking_service.handle_booking_message(
        message="I want to book an interview for Alex Smith.",
        conversation_id=conversation_id,
    )

    state = booking_state_store[conversation_id]

    assert "Alex Smith" in result["answer"]
    assert "alex@example.com" in result["answer"]
    assert "November 12, 2026" in result["answer"]
    assert "10:00 AM" in result["answer"]
    assert "Asia/Kathmandu" in result["answer"]

    assert state["booking_active"] is True
    assert state["awaiting_confirmation"] is True

    assert state["candidate_name"] == "Alex Smith"
    assert state["candidate_email"] == "alex@example.com"

    mock_parse.assert_called_once()
    mock_email.assert_called_once_with("alex@example.com")


@pytest.mark.asyncio
async def test_explicit_confirmation_saves_and_clears_booking(
    monkeypatch,
    booking_state_store,
):
    conversation_id = "booking-006"

    booking_state_store[conversation_id] = {
        "booking_active": True,
        "awaiting_confirmation": True,
        "candidate_name": "Alex Smith",
        "candidate_email": "alex@example.com",
        "date": "November 12, 2026",
        "time": "10 AM",
        "timezone": "Asia/Kathmandu",
    }

    async def mock_save_confirmed(conversation_id, state):
        await booking_service.clear_booking_state(conversation_id)
        return "Interview request saved. Status: Pending confirmation."

    mock_save = AsyncMock(side_effect=mock_save_confirmed)

    monkeypatch.setattr(
        booking_service,
        "_save_confirmed_booking",
        mock_save,
    )

    result = await booking_service.handle_booking_message(
        message="Yes",
        conversation_id=conversation_id,
    )

    assert "saved" in result["answer"].lower()
    assert conversation_id not in booking_state_store

    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_correcting_time_requires_new_confirmation(
    monkeypatch,
    booking_state_store,
):
    from datetime import datetime

    conversation_id = "booking-007"

    # The candidate has already reviewed the original 10 AM summary.
    booking_state_store[conversation_id] = {
        "booking_active": True,
        "awaiting_confirmation": True,
        "candidate_name": "Alex Smith",
        "candidate_email": "alex@example.com",
        "date": "November 12, 2026",
        "time": "10 AM",
        "timezone": "Asia/Kathmandu",
    }

    # The candidate corrects the interview time to 11 AM.
    monkeypatch.setattr(
        booking_service,
        "extract_booking_details",
        AsyncMock(return_value=extraction(time="11 AM")),
    )

    # Return 11 AM Kathmandu converted to UTC.
    monkeypatch.setattr(
        booking_service,
        "parse_scheduled_datetime",
        Mock(
            return_value=datetime(
                2026,
                11,
                12,
                5,
                15,
                tzinfo=UTC,
            )
        ),
    )

    result = await booking_service.handle_booking_message(
        message="Actually, make it 11 AM.",
        conversation_id=conversation_id,
    )

    state = booking_state_store[conversation_id]

    assert state["time"] == "11 AM"
    assert state["booking_active"] is True
    assert state["awaiting_confirmation"] is True

    assert "11:00 AM" in result["answer"]
    assert "10:00 AM" not in result["answer"]
    assert "reply **yes** to confirm" in result["answer"]


@pytest.mark.asyncio
async def test_invalid_email_preserves_other_booking_details(
    monkeypatch,
    booking_state_store,
):
    conversation_id = "booking-008"

    booking_state_store[conversation_id] = {
        "booking_active": True,
        "candidate_name": "Alex Smith",
    }

    monkeypatch.setattr(
        booking_service,
        "extract_booking_details",
        AsyncMock(
            return_value=extraction(
                candidate_email="not-an-email",
            )
        ),
    )

    result = await booking_service.handle_booking_message(
        message="not-an-email",
        conversation_id=conversation_id,
    )

    state = booking_state_store[conversation_id]

    assert "invalid email" in result["answer"].lower()
    assert state["booking_active"] is True
    assert state["candidate_name"] == "Alex Smith"
    assert state["candidate_email"] == "not-an-email"
    assert state.get("awaiting_confirmation") is False


@pytest.mark.parametrize(
    "message",
    [
        "Cancel the interview booking.",
        "I changed my mind. Cancel the interview booking.",
        "Please cancel my interview appointment.",
    ],
)
def test_natural_language_cancellation_is_recognized(message):
    assert booking_service._is_cancellation(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "How do I cancel an interview?",
        "Can I cancel my booking?",
        "What is the cancellation policy?",
    ],
)
def test_cancellation_questions_are_not_commands(message):
    assert booking_service._is_cancellation(message) is False
