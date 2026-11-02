from uuid import uuid4

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
    Handle one conversational RAG request.

    Creates a conversation ID when one is not supplied.
    Loads previous messages, generates an answer, and saves the exchange.
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

    # Step 3: Generate a response using conversational RAG.
    result = await generate_rag_answer(
        question=message,
        document_id=document_id,
        conversation_history=history,
    )

    answer = result["answer"]

    # Step 4: Save this exchange for future follow-up questions.
    await save_conversation_turn(
        conversation_id=conversation_id,
        user_message=message,
        assistant_message=answer,
    )

    # Step 5: Return the response and conversation identifier.
    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "sources": result["sources"],
    }
