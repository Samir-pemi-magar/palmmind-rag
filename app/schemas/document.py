from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: str
    chunking_strategy: Literal["fixed", "recursive"]
    total_chunks: int
    created_at: datetime


class DocumentIngestionFailure(BaseModel):
    filename: str
    error: str


class BatchDocumentResponse(BaseModel):
    total_files: int
    successful: list[DocumentResponse]
    failed: list[DocumentIngestionFailure]


class DocumentDeleteResponse(BaseModel):
    message: str
    document_id: str
    filename: str


class DocumentDeletionFailure(BaseModel):
    document_id: str
    filename: str
    error: str


class BulkDocumentDeleteResponse(BaseModel):
    total_documents: int
    deleted_count: int
    deleted: list[DocumentDeleteResponse]
    failed: list[DocumentDeletionFailure]
