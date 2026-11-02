from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.documents import router as documents_router
from app.core.database import close_db, create_db_and_tables
from app.core.qdrant import close_qdrant, create_collection_if_not_exists
from app.core.redis import close_redis
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize infrastructure when the application starts.
    await create_db_and_tables()
    await create_collection_if_not_exists()

    yield

    # Release connections when the application shuts down.
    await close_db()
    await close_qdrant()
    await close_redis()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(documents_router)


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "application": settings.app_name,
    }
