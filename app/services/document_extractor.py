from pathlib import Path

import pymupdf


def extract_text(file_path: str | Path) -> str:
    """
    Extract readable text from a PDF or TXT document.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        text_parts = []

        with pymupdf.open(path) as document:
            for page in document:
                page_text = page.get_text("text").strip()

                if page_text:
                    text_parts.append(page_text)

        extracted_text = "\n\n".join(text_parts)

    elif extension == ".txt":
        extracted_text = path.read_text(encoding="utf-8-sig")

    else:
        raise ValueError(f"Unsupported document format: {extension}")

    extracted_text = extracted_text.strip()

    if not extracted_text:
        raise ValueError("The document contains no extractable text.")

    return extracted_text
