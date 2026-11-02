from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


# Base class for all SQLAlchemy database models.
class Base(DeclarativeBase):
    pass


# Create the asynchronous database engine.
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
)


# Create a factory for database sessions.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# Provide a database session to FastAPI endpoints.
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


# Dispose of database connections when the application shuts down.
async def close_db() -> None:
    await engine.dispose()
