from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Palm Mind RAG API"
    app_version: str = "0.1.0"
    debug: bool = True

    database_url: str = "sqlite+aiosqlite:///./palmmind.db"
    upload_dir: str = "./data/uploads"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "document_chunks"

    redis_url: str = "redis://localhost:6379/0"

    llm_base_url: str = "http://localhost:3000/v1"
    llm_model: str = "Qwen3-8B-Q4_K_M.gguf"
    llm_api_key: str = "not-needed"

    embedding_model_path: str = "./models/bge-m3"
    embedding_dimension: int = 1024

    chunk_size: int = 500
    chunk_overlap: int = 75
    retrieval_top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
