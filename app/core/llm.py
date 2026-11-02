import httpx

from app.core.config import settings


async def generate_response(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """
    Send a conversation to the local Qwen3 model and return its response.
    """

    url = f"{settings.llm_base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            url,
            headers=headers,
            json=payload,
        )

        response.raise_for_status()
        result = response.json()

    return result["choices"][0]["message"]["content"].strip()
