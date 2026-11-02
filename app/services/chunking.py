from app.core.config import settings


def fixed_chunking(
    text: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[str]:
    """
    Split text into fixed-size character chunks with overlap.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than chunk size."
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks


def recursive_chunking(
    text: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[str]:
    """
    Split text using natural boundaries where possible,
    falling back to smaller separators when necessary.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than chunk size."
        )

    text = text.strip()

    if not text:
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            window = text[start:end]

            for separator in separators:
                if not separator:
                    break

                boundary = window.rfind(separator)

                # Avoid producing very small chunks.
                if boundary >= chunk_size * 0.5:
                    end = start + boundary + len(separator)
                    break

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(start + 1, end - chunk_overlap)

        # Avoid starting the next chunk in the middle of a word.
        while (
            next_start < end
            and text[next_start] != " "
            and text[next_start - 1] != " "
            and text[next_start] != "\n"
        ):
            next_start += 1

        start = next_start

    return chunks


def chunk_text(
    text: str,
    strategy: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[str]:
    """
    Select the requested chunking strategy.
    """

    if strategy == "fixed":
        return fixed_chunking(text, chunk_size, chunk_overlap)

    if strategy == "recursive":
        return recursive_chunking(text, chunk_size, chunk_overlap)

    raise ValueError("Unsupported chunking strategy. Choose 'fixed' or 'recursive'.")
