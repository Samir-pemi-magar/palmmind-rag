import logging
import re
from zoneinfo import ZoneInfo

from app.core.database import AsyncSessionLocal
from app.repositories.booking import create_booking
from app.services.booking_extractor import extract_booking_details
from app.services.booking_memory import (
    clear_booking_state,
    get_booking_state,
    save_booking_state,
)
from app.services.booking_validation import (
    get_valid_timezone,
    parse_scheduled_datetime,
    validate_candidate_email,
)

logger = logging.getLogger(__name__)


REQUIRED_FIELDS = (
    "candidate_name",
    "candidate_email",
    "date",
    "time",
    "timezone",
)


def _normalize_message(message: str) -> str:
    """Normalize standalone confirmation and cancellation replies."""

    return re.sub(r"[.!?,]+$", "", message.strip().lower())


def _is_confirmation(message: str) -> bool:
    """Recognize natural, affirmative booking confirmations."""

    normalized = _normalize_message(message)

    confirmations = {
        "yes",
        "yeah",
        "yep",
        "yes please",
        "yes confirm",
        "confirm",
        "confirmed",
        "i confirm",
        "that's correct",
        "that is correct",
        "correct",
        "book it",
        "save it",
        "please save it",
        "proceed",
        "go ahead",
        "go ahead and save it",
        "sure",
        "sure thing",
        "okay",
        "ok",
        "sounds good",
        "looks good",
        "that's fine",
        "that is fine",
        "please do",
    }

    if normalized in confirmations:
        return True

    return bool(
        re.fullmatch(
            r"(yes|yeah|yep|sure|okay|ok)"
            r"(,?\s+(please|confirm|save it|go ahead|proceed))*"
            r"(,?\s+and\s+(confirm|save it|proceed))?",
            normalized,
        )
    )


def _is_cancellation(
    message: str,
    awaiting_confirmation: bool = False,
) -> bool:
    """Recognize explicit cancellation without misclassifying ordinary answers."""

    normalized = _normalize_message(message)

    # A bare negative response only cancels after the booking summary.
    if normalized in {"no", "no thanks", "no thank you", "stop"}:
        return awaiting_confirmation

    cancellations = {
        "cancel",
        "cancel it",
        "cancel booking",
        "cancel the booking",
        "cancel interview",
        "cancel the interview",
        "cancel interview booking",
        "cancel the interview booking",
        "never mind",
        "nevermind",
    }

    if normalized in cancellations:
        return True

    # Do not interpret questions about cancellation as cancellation commands.
    if "?" in message:
        return False

    # Do not treat negative statements as cancellation commands.
    if re.search(
        r"\b(?:don't|do not|never|not)\b.{0,40}\b(cancel|cancelled|canceled|stop)\b",
        normalized,
    ):
        return False

    return bool(
        re.search(r"\b(cancel|cancelled|canceled|stop)\b", normalized)
        and re.search(
            r"\b(booking|interview|appointment|it|request)\b",
            normalized,
        )
    )


def _looks_like_booking_request(message: str) -> bool:
    """Identify likely new interview-booking requests without calling the LLM."""

    normalized = " ".join(message.lower().split())

    booking_phrases = (
        "book an interview",
        "book interview",
        "schedule an interview",
        "schedule interview",
        "arrange an interview",
        "arrange interview",
        "interview appointment",
        "appointment for an interview",
        "reschedule my interview",
        "reschedule an interview",
        "change my interview",
        "change my booking",
        "modify my interview",
        "modify my booking",
        "update my interview",
        "i want an interview",
        "i need an interview",
        "i would like an interview",
        "i'd like an interview",
    )

    return any(phrase in normalized for phrase in booking_phrases)


def _missing_fields(state: dict) -> list[str]:
    """Return required booking fields that have not been collected."""

    return [field for field in REQUIRED_FIELDS if not state.get(field)]


def _question_for_missing_fields(fields: list[str]) -> str:
    """Ask for the next missing booking detail."""

    questions = {
        "candidate_name": "What is your full name?",
        "candidate_email": ("What email address should we use for your interview?"),
        "date": "What date would you prefer for the interview?",
        "time": "What time would you prefer?",
        "timezone": (
            "Which timezone should I use? For example, Asia/Kathmandu or Europe/London."
        ),
    }

    return questions[fields[0]]


async def _save_confirmed_booking(
    conversation_id: str,
    state: dict,
) -> str:
    """Persist a confirmed booking and return its confirmation message."""

    scheduled_at = parse_scheduled_datetime(
        state["date"],
        state["time"],
        state["timezone"],
    )

    async with AsyncSessionLocal() as session:
        booking = await create_booking(
            session,
            candidate_name=state["candidate_name"],
            candidate_email=state["candidate_email"],
            scheduled_at=scheduled_at,
            timezone=state["timezone"],
            conversation_id=conversation_id,
        )

    # Clear temporary Redis state only after the database commit succeeds.
    await clear_booking_state(conversation_id)

    interview_time = scheduled_at.astimezone(ZoneInfo(state["timezone"])).strftime(
        "%B %d, %Y at %I:%M %p"
    )

    return (
        "Your interview request has been saved successfully.\n\n"
        f"**Reference:** {booking.id}\n"
        f"**Name:** {booking.candidate_name}\n"
        f"**Email:** {booking.candidate_email}\n"
        f"**Interview:** {interview_time}\n"
        f"**Timezone:** {state['timezone']}\n"
        "**Status:** Pending confirmation\n\n"
        "Your request is saved, but the appointment is not yet "
        "confirmed by an interviewer."
    )


async def handle_booking_message(
    message: str,
    conversation_id: str,
) -> dict | None:
    """
    Handle one candidate message in the interview booking workflow.

    Returns an answer dictionary when the message belongs to booking.
    Returns None when the normal RAG chat should handle the message.
    """

    message = message.strip()

    if not message:
        raise ValueError("Message cannot be empty.")

    if not conversation_id.strip():
        raise ValueError("Conversation ID cannot be empty.")

    state = await get_booking_state(conversation_id)

    booking_active = state.get("booking_active", False)

    # Explicit cancellation is allowed at every stage.
    # A standalone "no" only cancels while awaiting confirmation.
    if booking_active and _is_cancellation(
        message,
        awaiting_confirmation=state.get("awaiting_confirmation", False),
    ):
        await clear_booking_state(conversation_id)

        return {
            "answer": (
                "Your interview booking request has been cancelled. "
                "No booking was saved."
            ),
            "sources": [],
        }

    # Confirmation is only accepted after the candidate has reviewed
    # the complete booking summary.
    if state.get("awaiting_confirmation"):
        if _is_confirmation(message):
            try:
                answer = await _save_confirmed_booking(
                    conversation_id,
                    state,
                )

                return {"answer": answer, "sources": []}

            except ValueError as exc:
                await save_booking_state(
                    conversation_id,
                    {"awaiting_confirmation": False},
                )

                return {
                    "answer": (
                        f"{exc}\nPlease provide a new future interview date and time."
                    ),
                    "sources": [],
                }

    # Skip the booking LLM call for ordinary messages when no booking is active.
    # Active bookings still use extraction to recognize short answers and corrections.
    if not booking_active and not _looks_like_booking_request(message):
        return None

    # Ask Qwen to extract information from the latest message.
    # Existing Redis state gives context to short replies.
    try:
        extracted = await extract_booking_details(
            message,
            current_state=state,
        )
    except Exception:
        logger.exception(
            "Booking extraction failed for conversation %s; falling back to RAG.",
            conversation_id,
        )
        return None

    # Ordinary questions should continue to the RAG service.
    if not extracted["booking_intent"]:
        return None

    updates = {
        field: extracted[field]
        for field in REQUIRED_FIELDS
        if extracted[field] is not None
    }

    # Starting or continuing a booking always preserves its active state.
    updates["booking_active"] = True

    # Any correction invalidates the previous confirmation.
    if any(field in updates for field in REQUIRED_FIELDS):
        updates["awaiting_confirmation"] = False

    state = await save_booking_state(
        conversation_id,
        updates,
    )

    # Validate the candidate's name.
    name = state.get("candidate_name")

    if name is not None:
        name = name.strip()

        if not name or len(name) > 150:
            return {
                "answer": (
                    "Please provide a valid full name (maximum 150 characters)."
                ),
                "sources": [],
            }

        state = await save_booking_state(
            conversation_id,
            {"candidate_name": name},
        )

    # Validate email as soon as it is provided.
    email = state.get("candidate_email")

    if email:
        try:
            normalized_email = validate_candidate_email(email)

            if len(normalized_email) > 255:
                raise ValueError("Email address is too long.")

            state = await save_booking_state(
                conversation_id,
                {"candidate_email": normalized_email},
            )

        except ValueError as exc:
            return {
                "answer": (f"{exc}\nPlease provide a valid email address."),
                "sources": [],
            }

    # Validate timezone as soon as it is provided.
    timezone_name = state.get("timezone")

    if timezone_name:
        try:
            get_valid_timezone(timezone_name)

        except ValueError:
            return {
                "answer": (
                    f"I couldn't recognize the timezone '{timezone_name}'. "
                    "Please provide an IANA timezone such as Asia/Kathmandu."
                ),
                "sources": [],
            }

    # Ask for any details that are still missing.
    missing = _missing_fields(state)

    if missing:
        return {
            "answer": _question_for_missing_fields(missing),
            "sources": [],
        }

    # All fields are present. Validate the appointment date and time.
    try:
        scheduled_at = parse_scheduled_datetime(
            state["date"],
            state["time"],
            state["timezone"],
        )

    except ValueError as exc:
        return {
            "answer": (
                f"{exc}\nPlease provide a valid future interview date and time."
            ),
            "sources": [],
        }

    local_time = scheduled_at.astimezone(ZoneInfo(state["timezone"]))

    # The candidate must explicitly approve the complete summary.
    state = await save_booking_state(
        conversation_id,
        {
            "booking_active": True,
            "awaiting_confirmation": True,
        },
    )

    summary = (
        "Here are your interview details:\n\n"
        f"**Name:** {state['candidate_name']}\n"
        f"**Email:** {state['candidate_email']}\n"
        f"**Date:** {local_time:%B %d, %Y}\n"
        f"**Time:** {local_time:%I:%M %p}\n"
        f"**Timezone:** {state['timezone']}\n\n"
        "Would you like me to save this interview request? "
        "Please reply **yes** to confirm or **no** to cancel."
    )

    return {"answer": summary, "sources": []}
