import asyncio
import logging
from uuid import uuid4

from fastapi import UploadFile
from qdrant_client import models
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.qdrant import qdrant_client
from app.models.document import Document
from app.services.chunking import chunk_text
from app.services.document_extractor import extract_text
from app.services.document_storage import save_uploaded_document
from app.core.embeddings import generate_embeddings


logger = logging.getLogger(__name__)


async def ingest_document(
    upload: UploadFile,
    strategy: str,
    db: AsyncSession,
) -> Document:
    """
    Process an uploaded document and store its chunks and metadata.

    Pipeline:
        Upload -> Storage -> Extraction -> Chunking
        -> Embeddings -> Qdrant -> SQLite
    """

    if strategy not in {"fixed", "recursive"}:
        raise ValueError(
            "Unsupported chunking strategy. Choose 'fixed' or 'recursive'."
        )

    original_filename = upload.filename

    if not original_filename:
        raise ValueError("Uploaded file must have a filename.")

    file_path = None
    point_ids = []

    try:
        # 1. Save the original document.
        file_path, stored_filename = await save_uploaded_document(upload)

        # 2. Extract readable text.
        extracted_text = extract_text(file_path)

        # 3. Divide extracted text into chunks.
        chunks = chunk_text(
            text=extracted_text,
            strategy=strategy,
        )

        if not chunks:
            raise ValueError("Document did not produce any text chunks.")

        # 4. Generate BGE-M3 embeddings without blocking the event loop.
        embeddings = await asyncio.to_thread(generate_embeddings, chunks)

        if len(embeddings) != len(chunks):
            raise ValueError("Embedding count does not match chunk count.")

        # 5. Create Qdrant points.
        document_id = str(uuid4())

        points = []

        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid4())
            point_ids.append(point_id)

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "filename": original_filename,
                        "chunk_index": index,
                        "chunk_text": chunk,
                        "chunking_strategy": strategy,
                        "stored_filename": stored_filename,
                    },
                )
            )

        # 6. Store vectors and their payloads in Qdrant.
        await qdrant_client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
            wait=True,
        )

        # 7. Store document metadata in SQLite.
        document = Document(
            id=document_id,
            filename=original_filename,
            file_path=str(file_path),
            status="completed",
            chunking_strategy=strategy,
            total_chunks=len(chunks),
        )

        db.add(document)
        await db.commit()
        await db.refresh(document)

        return document

    except Exception:
        # Roll back any pending SQLAlchemy changes.
        await db.rollback()

        # Best-effort cleanup of Qdrant points.
        if point_ids:
            try:
                await qdrant_client.delete(
                    collection_name=settings.qdrant_collection,
                    points_selector=models.PointIdsList(points=point_ids),
                    wait=True,
                )
            except Exception:
                logger.exception("Failed to clean up Qdrant points.")

        # Remove the saved original file.
        if file_path is not None:
            file_path.unlink(missing_ok=True)

        raise
