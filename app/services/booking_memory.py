import json

from app.core.redis import redis_client

BOOKING_TTL = 86400  # 24 hours

BOOKING_FIELDS = {
    "candidate_name",
    "candidate_email",
    "date",
    "time",
    "timezone",
    "awaiting_confirmation",
    "booking_active",
}


def _booking_key(conversation_id: str) -> str:
    """Build and validate the Redis key for a booking conversation."""

    if not conversation_id.strip():
        raise ValueError("Conversation ID cannot be empty.")

    return f"booking:{conversation_id}"


async def get_booking_state(
    conversation_id: str,
) -> dict:
    """Retrieve temporarily collected booking details from Redis."""

    key = _booking_key(conversation_id)

    stored_state = await redis_client.get(key)

    if stored_state is None:
        return {}

    return json.loads(stored_state)


async def save_booking_state(
    conversation_id: str,
    details: dict,
) -> dict:
    """
    Merge newly collected booking details with existing details.

    Missing or None values do not overwrite previously collected values.
    """

    key = _booking_key(conversation_id)

    if not isinstance(details, dict):
        raise ValueError("Booking details must be a dictionary.")

    invalid_fields = set(details) - BOOKING_FIELDS

    if invalid_fields:
        raise ValueError(f"Invalid booking fields: {', '.join(sorted(invalid_fields))}")

    current_state = await get_booking_state(conversation_id)

    for field, value in details.items():
        if value is not None:
            current_state[field] = value

    await redis_client.set(
        key,
        json.dumps(current_state),
        ex=BOOKING_TTL,
    )

    return current_state


async def clear_booking_state(
    conversation_id: str,
) -> None:
    """Delete temporary booking details from Redis."""

    key = _booking_key(conversation_id)

    await redis_client.delete(key)
