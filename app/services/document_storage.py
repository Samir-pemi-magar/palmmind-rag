from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import settings


ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB


async def save_uploaded_document(
    upload: UploadFile,
) -> tuple[Path, str]:
    """
    Validate and save an uploaded PDF or TXT document.

    Returns:
        A tuple containing the saved file path and stored filename.
    """

    if not upload.filename:
        raise ValueError("Uploaded file must have a filename.")

    extension = Path(upload.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only PDF and TXT documents are supported.")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{uuid4()}{extension}"
    file_path = upload_dir / stored_filename

    total_size = 0

    try:
        with file_path.open("wb") as destination:
            while True:
                content = await upload.read(1024 * 1024)

                if not content:
                    break

                total_size += len(content)

                if total_size > MAX_UPLOAD_SIZE:
                    raise ValueError("Uploaded file exceeds the 20 MB limit.")

                destination.write(content)

        if total_size == 0:
            raise ValueError("Uploaded file is empty.")

        return file_path, stored_filename

    except Exception:
        file_path.unlink(missing_ok=True)
        raise

    finally:
        await upload.close()
