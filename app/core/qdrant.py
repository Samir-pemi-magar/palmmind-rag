from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings


qdrant_client = AsyncQdrantClient(
    url=settings.qdrant_url,
)


async def create_collection_if_not_exists() -> None:
    exists = await qdrant_client.collection_exists(
        collection_name=settings.qdrant_collection,
    )

    if not exists:
        await qdrant_client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )


async def close_qdrant() -> None:
    await qdrant_client.close()
