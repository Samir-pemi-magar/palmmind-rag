from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.document import DocumentResponse
from app.services.ingestion import ingest_document

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/ingest",
    response_model=DocumentResponse,
    status_code=201,
)
async def upload_document(
    upload: UploadFile = File(...),
    strategy: Literal["fixed", "recursive"] = Form(...),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    try:
        document = await ingest_document(
            upload=upload,
            strategy=strategy,
            db=db,
        )
        return DocumentResponse.model_validate(document)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
