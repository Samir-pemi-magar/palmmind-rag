from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.core.config import settings
from app.core.database import close_db
from app.core.qdrant import close_qdrant, create_collection_if_not_exists
from app.core.redis import close_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize external services used by the application.
    await create_collection_if_not_exists()

    yield

    # Release application resources during shutdown.
    await close_db()
    await close_qdrant()
    await close_redis()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(chat_router)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
        openapi_version="3.0.3",
    )

    batch_schema = openapi_schema["components"]["schemas"][
        "Body_upload_documents_documents_ingest_batch_post"
    ]

    upload_items = batch_schema["properties"]["uploads"]["items"]

    upload_items.pop("contentMediaType", None)
    upload_items["format"] = "binary"

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "application": settings.app_name,
    }
