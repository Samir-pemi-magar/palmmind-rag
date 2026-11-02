import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.document import (
    BatchDocumentResponse,
    DocumentIngestionFailure,
    DocumentResponse,
)
from app.services.ingestion import ingest_document

logger = logging.getLogger(__name__)

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


@router.post(
    "/ingest/batch",
    response_model=BatchDocumentResponse,
    status_code=200,
)
async def upload_documents(
    uploads: list[UploadFile] = File(...),
    strategy: Literal["fixed", "recursive"] = Form(...),
    db: AsyncSession = Depends(get_db),
) -> BatchDocumentResponse:
    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="At least one file must be uploaded.",
        )

    successful = []
    failed = []

    for upload in uploads:
        filename = upload.filename or "unknown"

        try:
            document = await ingest_document(
                upload=upload,
                strategy=strategy,
                db=db,
            )
            successful.append(DocumentResponse.model_validate(document))

        except ValueError as exc:
            failed.append(
                DocumentIngestionFailure(
                    filename=filename,
                    error=str(exc),
                )
            )

        except Exception:
            logger.exception("Failed to ingest uploaded file: %s", filename)
            failed.append(
                DocumentIngestionFailure(
                    filename=filename,
                    error="Unexpected ingestion error. Check the server logs.",
                )
            )

        finally:
            await upload.close()

    return BatchDocumentResponse(
        total_files=len(uploads),
        successful=successful,
        failed=failed,
    )
