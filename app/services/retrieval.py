import asyncio

from qdrant_client import models

from app.core.config import settings
from app.core.embeddings import generate_embeddings
from app.core.qdrant import qdrant_client


async def retrieve_chunks(
    query: str,
    top_k: int | None = None,
    document_id: str | None = None,
) -> list[dict]:
    """
    Find the most relevant document chunks for a user's query.

    Args:
        query: The user's question or search text.
        top_k: Maximum number of chunks to retrieve.
        document_id: Optional filter to search within one document.

    Returns:
        A list of relevant chunks with metadata and similarity scores.
    """

    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    limit = top_k if top_k is not None else settings.retrieval_top_k

    if not 1 <= limit <= 100:
        raise ValueError("top_k must be between 1 and 100.")

    # Generate the query embedding without blocking the async event loop.
    embeddings = await asyncio.to_thread(
        generate_embeddings,
        [query],
    )

    query_vector = embeddings[0]

    # Optionally restrict the search to a particular document.
    query_filter = None

    if document_id:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )

    # Search Qdrant using cosine similarity.
    results = await qdrant_client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    retrieved_chunks = []

    for point in results.points:
        payload = point.payload or {}

        retrieved_chunks.append(
            {
                "document_id": payload.get("document_id"),
                "filename": payload.get("filename"),
                "chunk_index": payload.get("chunk_index"),
                "chunk_text": payload.get("chunk_text"),
                "chunking_strategy": payload.get("chunking_strategy"),
                "score": point.score,
            }
        )

    return retrieved_chunks
