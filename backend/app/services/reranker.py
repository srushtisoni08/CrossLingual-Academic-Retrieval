import logging
from functools import lru_cache

from app.config import MAX_SEQ_LENGTH, RERANKER_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL, max_length=max(MAX_SEQ_LENGTH, 384))


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Re-sort candidates (dicts with 'text'); 'score' becomes a 0-1 relevance probability."""
    if not candidates:
        return candidates
    try:
        scores = _load().predict([(query, c["text"]) for c in candidates], batch_size=16)
        out = [{**c, "score": float(s)} for c, s in zip(candidates, scores)]
        return sorted(out, key=lambda c: -c["score"])
    except Exception:
        logger.exception("Reranking failed; keeping first-stage order")
        return candidates