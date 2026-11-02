from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    conversation_id: str | None = None
    document_id: str | None = None


class ChatSource(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[ChatSource]
