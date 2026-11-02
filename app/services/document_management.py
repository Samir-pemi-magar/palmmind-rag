import logging
from pathlib import Path

from qdrant_client import models
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.qdrant import qdrant_client
from app.models.document import Document

logger = logging.getLogger(__name__)


async def list_documents(db: AsyncSession) -> list[Document]:
    """Return all uploaded documents, newest first."""

    result = await db.execute(
        select(Document).order_by(Document.created_at.desc())
    )

    return list(result.scalars().all())


async def get_document(
    document_id: str,
    db: AsyncSession,
) -> Document | None:
    """Find a document by its ID."""

    return await db.get(Document, document_id)


async def delete_document(
    document_id: str,
    db: AsyncSession,
) -> Document | None:
    """
    Delete a document and its associated Qdrant vectors and original file.

    Qdrant is cleaned first so a Qdrant failure does not remove the
    database record or original file.
    """

    document = await get_document(document_id, db)

    if document is None:
        return None

    file_path = Path(document.file_path)

    # Delete every Qdrant point belonging to this document.
    await qdrant_client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        ),
        wait=True,
    )

    # Remove the document metadata from SQLite.
    await db.delete(document)
    await db.commit()

    # Remove the original file only if it is inside the upload directory.
    upload_dir = Path(settings.upload_dir).resolve()
    resolved_file_path = file_path.resolve()

    if resolved_file_path.is_relative_to(upload_dir):
        try:
            resolved_file_path.unlink(missing_ok=True)
        except OSError:
            logger.exception(
                "Document %s was deleted from the database, "
                "but its original file could not be removed: %s",
                document_id,
                resolved_file_path,
            )
    else:
        logger.error(
            "Refusing to delete document file outside upload directory. "
            "Document ID: %s, file path: %s",
            document_id,
            resolved_file_path,
        )

    return document
