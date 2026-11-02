from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.database import Base
from app.models.booking import InterviewBooking
from app.repositories.booking import create_booking, get_booking_by_id


@pytest_asyncio.fixture
async def test_session():
    """Create an isolated in-memory SQLite database for each test."""

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_create_and_retrieve_booking(test_session):
    scheduled_at = datetime(
        2026,
        11,
        12,
        4,
        15,
        tzinfo=UTC,
    )

    booking = await create_booking(
        test_session,
        candidate_name="Alex Smith",
        candidate_email="alex@example.com",
        scheduled_at=scheduled_at,
        timezone="Asia/Kathmandu",
        conversation_id="repository-test-001",
    )

    assert booking.id is not None
    assert booking.candidate_name == "Alex Smith"
    assert booking.candidate_email == "alex@example.com"
    assert booking.timezone == "Asia/Kathmandu"
    assert booking.status == "pending"
    assert booking.conversation_id == "repository-test-001"

    retrieved = await get_booking_by_id(
        test_session,
        booking.id,
    )

    assert retrieved is not None
    assert retrieved.id == booking.id
    assert retrieved.candidate_name == "Alex Smith"
    assert retrieved.candidate_email == "alex@example.com"


@pytest.mark.asyncio
async def test_create_booking_rejects_naive_datetime(test_session):
    naive_datetime = datetime(2026, 11, 12, 10, 0)

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        await create_booking(
            test_session,
            candidate_name="Alex Smith",
            candidate_email="alex@example.com",
            scheduled_at=naive_datetime,
            timezone="Asia/Kathmandu",
        )


@pytest.mark.asyncio
async def test_retrieve_nonexistent_booking_returns_none(test_session):
    result = await get_booking_by_id(
        test_session,
        "nonexistent-booking-id",
    )

    assert result is None


@pytest.mark.asyncio
async def test_confirmed_booking_persists_and_clears_temporary_state(
    test_session,
    monkeypatch,
):
    from unittest.mock import AsyncMock, Mock

    from app.services import booking as booking_service

    scheduled_at = datetime(
        2026,
        11,
        12,
        4,
        15,
        tzinfo=UTC,
    )

    session_factory = async_sessionmaker(
        test_session.bind,
        expire_on_commit=False,
    )

    mock_clear = AsyncMock()

    monkeypatch.setattr(
        booking_service,
        "AsyncSessionLocal",
        session_factory,
    )
    monkeypatch.setattr(
        booking_service,
        "parse_scheduled_datetime",
        Mock(return_value=scheduled_at),
    )
    monkeypatch.setattr(
        booking_service,
        "clear_booking_state",
        mock_clear,
    )

    state = {
        "candidate_name": "Alex Smith",
        "candidate_email": "alex@example.com",
        "date": "November 12, 2026",
        "time": "10 AM",
        "timezone": "Asia/Kathmandu",
    }

    answer = await booking_service._save_confirmed_booking(
        "integration-test-001",
        state,
    )

    assert "saved successfully" in answer.lower()
    assert "Pending confirmation" in answer

    mock_clear.assert_awaited_once_with("integration-test-001")

    # Verify that the booking was committed and can be retrieved.
    result = await test_session.execute(
        __import__("sqlalchemy")
        .select(InterviewBooking)
        .where(InterviewBooking.conversation_id == "integration-test-001")
    )
    saved_booking = result.scalar_one()

    assert saved_booking.candidate_name == "Alex Smith"
    assert saved_booking.candidate_email == "alex@example.com"
    assert saved_booking.timezone == "Asia/Kathmandu"
    assert saved_booking.status == "pending"
    assert saved_booking.scheduled_at == scheduled_at.replace(tzinfo=None)


@pytest.mark.asyncio
async def test_database_failure_preserves_temporary_booking_state(
    test_session,
    monkeypatch,
):
    from unittest.mock import AsyncMock, Mock

    from app.services import booking as booking_service

    session_factory = async_sessionmaker(
        test_session.bind,
        expire_on_commit=False,
    )

    mock_clear = AsyncMock()

    async def fail_create_booking(*args, **kwargs):
        raise RuntimeError("Simulated database failure")

    monkeypatch.setattr(
        booking_service,
        "AsyncSessionLocal",
        session_factory,
    )
    monkeypatch.setattr(
        booking_service,
        "parse_scheduled_datetime",
        Mock(
            return_value=datetime(
                2026,
                11,
                12,
                4,
                15,
                tzinfo=UTC,
            )
        ),
    )
    monkeypatch.setattr(
        booking_service,
        "create_booking",
        fail_create_booking,
    )
    monkeypatch.setattr(
        booking_service,
        "clear_booking_state",
        mock_clear,
    )

    state = {
        "candidate_name": "Alex Smith",
        "candidate_email": "alex@example.com",
        "date": "November 12, 2026",
        "time": "10 AM",
        "timezone": "Asia/Kathmandu",
    }

    with pytest.raises(
        RuntimeError,
        match="Simulated database failure",
    ):
        await booking_service._save_confirmed_booking(
            "integration-test-failure",
            state,
        )

    # Temporary booking data must not be cleared after a failed save.
    mock_clear.assert_not_awaited()
