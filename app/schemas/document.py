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
