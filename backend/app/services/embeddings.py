from functools import lru_cache

import numpy as np

from app.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    MAX_SEQ_LENGTH,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
)


@lru_cache(maxsize=1)
def get_model():
    """Load the sentence-transformer once (first call downloads it)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def _encode(texts: list[str], prefix: str, show_progress: bool) -> np.ndarray:
    vectors = get_model().encode(
        [prefix + t for t in texts],
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,  # unit vectors -> dot product == cosine
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return np.asarray(vectors, dtype=np.float32)


def encode_queries(texts: list[str], show_progress: bool = False) -> np.ndarray:
    return _encode(texts, QUERY_PREFIX, show_progress)


def encode_passages(texts: list[str], show_progress: bool = True) -> np.ndarray:
    return _encode(texts, PASSAGE_PREFIX, show_progress)