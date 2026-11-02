from functools import lru_cache

import numpy as np
from FlagEmbedding import FlagModel

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> FlagModel:
    """
    Load BGE-M3 once and reuse it throughout the application.
    """

    return FlagModel(
        settings.embedding_model_path,
        use_fp16=False,
        devices=["cpu"],
    )


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of texts into normalized dense embedding vectors.
    """

    if not texts:
        return []

    cleaned_texts = [text.strip() for text in texts]

    if any(not text for text in cleaned_texts):
        raise ValueError("Embedding input cannot contain empty text.")

    model = get_embedding_model()

    vectors = model.encode(
        cleaned_texts,
        batch_size=8,
        max_length=512,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("Embedding model returned an invalid shape.")

    if vectors.shape[1] != settings.embedding_dimension:
        raise ValueError(
            f"Expected {settings.embedding_dimension} dimensions, "
            f"received {vectors.shape[1]}."
        )

    if not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned invalid numerical values.")

    return vectors.tolist()
