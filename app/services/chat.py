from uuid import uuid4

from app.services.booking import handle_booking_message
from app.services.conversation_memory import (
    get_conversation_history,
    save_conversation_turn,
)
from app.services.rag import generate_rag_answer


async def chat(
    message: str,
    conversation_id: str | None = None,
    document_id: str | None = None,
) -> dict:
    """
    Handle conversational RAG and interview booking requests.

    Booking messages are processed first. Messages that are not
    booking-related continue through the normal RAG pipeline.
    """

    message = message.strip()

    if not message:
        raise ValueError("Message cannot be empty.")

    # Step 1: Create a new conversation or continue an existing one.
    if conversation_id is None:
        conversation_id = str(uuid4())
    elif not conversation_id.strip():
        raise ValueError("Conversation ID cannot be empty.")

    # Step 2: Load previous conversation messages from Redis.
    history = await get_conversation_history(conversation_id)

    # Step 3: Check whether this message belongs to interview booking.
    booking_result = await handle_booking_message(
        message=message,
        conversation_id=conversation_id,
    )

    if booking_result is not None:
        result = booking_result
    else:
        # Step 4: If not a booking request, use conversational RAG.
        result = await generate_rag_answer(
            question=message,
            document_id=document_id,
            conversation_history=history,
        )

    answer = result["answer"]

    # Step 5: Save both booking and RAG exchanges in conversation history.
    await save_conversation_turn(
        conversation_id=conversation_id,
        user_message=message,
        assistant_message=answer,
    )

    # Step 6: Return the response and conversation identifier.
    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "sources": result["sources"],
    }
