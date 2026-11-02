from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import dateparser
from email_validator import EmailNotValidError, validate_email


def validate_candidate_email(email: str) -> str:
    """Validate an email address and return its normalized form."""

    try:
        result = validate_email(
            email,
            check_deliverability=False,
        )
    except EmailNotValidError as exc:
        raise ValueError(f"Invalid email address: {exc}") from exc

    return result.normalized


def get_valid_timezone(timezone_name: str) -> ZoneInfo:
    """Return a valid IANA timezone."""

    try:
        return ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Invalid timezone: {timezone_name}") from exc


def parse_scheduled_datetime(
    date_text: str,
    time_text: str,
    timezone_name: str,
    *,
    now: datetime | None = None,
) -> datetime:
    """
    Parse a candidate's requested date and time.

    Returns a timezone-aware datetime normalized to UTC.
    Relative dates use the supplied reference time.
    """

    if not date_text.strip():
        raise ValueError("Interview date is required.")

    if not time_text.strip():
        raise ValueError("Interview time is required.")

    candidate_timezone = get_valid_timezone(timezone_name)

    current_time = now or datetime.now(UTC)

    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ValueError("The current time must be timezone-aware.")

    current_local_time = current_time.astimezone(candidate_timezone)

    parsed = dateparser.parse(
        f"{date_text.strip()} {time_text.strip()}",
        settings={
            "TIMEZONE": timezone_name,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": current_local_time,
        },
    )

    if parsed is None:
        raise ValueError(
            "Could not understand the interview date and time. "
            "Please provide a clearer date and time."
        )

    local_datetime = parsed.astimezone(candidate_timezone)

    if local_datetime <= current_local_time:
        raise ValueError("Interview date and time must be in the future.")

    return local_datetime.astimezone(UTC)
