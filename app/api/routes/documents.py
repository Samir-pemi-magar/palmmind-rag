import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.document import (
    BatchDocumentResponse,
    BulkDocumentDeleteResponse,
    DocumentDeleteResponse,
    DocumentDeletionFailure,
    DocumentIngestionFailure,
    DocumentResponse,
)
from app.services.document_management import (
    delete_all_documents,
    delete_document,
    get_document,
    list_documents,
)
from app.services.ingestion import ingest_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "",
    response_model=BatchDocumentResponse,
    status_code=200,
)
async def upload_documents(
    uploads: list[UploadFile] = File(...),
    strategy: Literal["fixed", "recursive"] = Form(...),
    db: AsyncSession = Depends(get_db),
) -> BatchDocumentResponse:
    """Upload and ingest one or more PDF/TXT documents."""

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


@router.get(
    "",
    response_model=list[DocumentResponse],
)
async def get_documents(
    db: AsyncSession = Depends(get_db),
) -> list[DocumentResponse]:
    """List all uploaded documents, newest first."""

    documents = await list_documents(db)

    return [DocumentResponse.model_validate(document) for document in documents]


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
async def get_document_details(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Retrieve a document by its ID."""

    document = await get_document(document_id, db)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return DocumentResponse.model_validate(document)


@router.delete(
    "",
    response_model=BulkDocumentDeleteResponse,
)
async def remove_all_documents(
    db: AsyncSession = Depends(get_db),
) -> BulkDocumentDeleteResponse:
    """Delete all documents, reporting individual failures."""

    deleted, failed = await delete_all_documents(db)

    return BulkDocumentDeleteResponse(
        total_documents=len(deleted) + len(failed),
        deleted_count=len(deleted),
        deleted=[
            DocumentDeleteResponse(
                message="Document deleted successfully.",
                document_id=document.id,
                filename=document.filename,
            )
            for document in deleted
        ],
        failed=[
            DocumentDeletionFailure(
                document_id=document.id,
                filename=document.filename,
                error="Deletion failed. Check server logs and retry.",
            )
            for document, _error in failed
        ],
    )


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
)
async def remove_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> DocumentDeleteResponse:
    """Delete a document and its associated data."""

    document = await delete_document(document_id, db)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return DocumentDeleteResponse(
        message="Document deleted successfully.",
        document_id=document.id,
        filename=document.filename,
    )
