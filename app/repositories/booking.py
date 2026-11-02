from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.booking import InterviewBooking


async def create_booking(
    session: AsyncSession,
    *,
    candidate_name: str,
    candidate_email: str,
    scheduled_at: datetime,
    timezone: str,
    conversation_id: str | None = None,
) -> InterviewBooking:
    """
    Create and persist an interview booking request.

    scheduled_at must be timezone-aware and normalized to UTC
    by the booking service before reaching this repository.
    """

    if scheduled_at.tzinfo is None or scheduled_at.utcoffset() is None:
        raise ValueError("scheduled_at must be timezone-aware.")

    booking = InterviewBooking(
        candidate_name=candidate_name,
        candidate_email=candidate_email,
        scheduled_at=scheduled_at,
        timezone=timezone,
        conversation_id=conversation_id,
        status="pending",
    )

    session.add(booking)

    try:
        await session.commit()
        await session.refresh(booking)
    except Exception:
        await session.rollback()
        raise

    return booking


async def get_booking_by_id(
    session: AsyncSession,
    booking_id: str,
) -> InterviewBooking | None:
    """Retrieve a booking using its unique ID."""

    result = await session.execute(
        select(InterviewBooking).where(InterviewBooking.id == booking_id)
    )

    return result.scalar_one_or_none()
