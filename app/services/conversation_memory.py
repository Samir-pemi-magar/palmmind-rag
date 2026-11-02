import json

from app.core.redis import redis_client

CONVERSATION_TTL = 86400  # 24 hours
MAX_HISTORY_MESSAGES = 10


def _conversation_key(conversation_id: str) -> str:
    if not conversation_id.strip():
        raise ValueError("Conversation ID cannot be empty.")

    return f"conversation:{conversation_id}"


async def get_conversation_history(
    conversation_id: str,
) -> list[dict[str, str]]:
    """
    Retrieve the recent user and assistant messages for a conversation.
    """

    key = _conversation_key(conversation_id)

    stored_messages = await redis_client.lrange(
        key,
        0,
        -1,
    )

    return [json.loads(message) for message in stored_messages]


async def save_conversation_turn(
    conversation_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """
    Save one user/assistant exchange and maintain the history limit.
    """

    key = _conversation_key(conversation_id)

    if not user_message.strip():
        raise ValueError("User message cannot be empty.")

    if not assistant_message.strip():
        raise ValueError("Assistant message cannot be empty.")

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]

    async with redis_client.pipeline(transaction=True) as pipeline:
        for message in messages:
            pipeline.rpush(key, json.dumps(message))

        pipeline.ltrim(
            key,
            -MAX_HISTORY_MESSAGES,
            -1,
        )

        pipeline.expire(key, CONVERSATION_TTL)

        await pipeline.execute()


async def delete_conversation_history(
    conversation_id: str,
) -> None:
    """Delete the history associated with a conversation."""

    key = _conversation_key(conversation_id)
    await redis_client.delete(key)
