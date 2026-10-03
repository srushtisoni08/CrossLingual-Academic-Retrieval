import logging
from functools import lru_cache

from app.config import LANGUAGES
from app.services import bm25, dense_retrieval
from app.services.data_loader import build_corpus

logger = logging.getLogger(__name__)

RRF_K = 60          # standard Reciprocal Rank Fusion constant
BM25_WEIGHT = 1.0   # fusion weights: dense is more reliable here (esp. cross-lingual),
DENSE_WEIGHT = 2.0  # so it gets double the vote. Try 1.0 / 3.0 and re-run evaluation.
CANDIDATES = 50     # how many results each retriever contributes to fusion

# Unicode block -> language (the 3 non-Latin languages in this project)
_SCRIPT_RANGES = [
    (0x0900, 0x097F, "hi"),  # Devanagari
    (0x0980, 0x09FF, "bn"),  # Bengali
    (0x0C00, 0x0C7F, "te"),  # Telugu
]


def detect_language(text: str) -> str:
    """Cheap script-based detection; anything else is treated as English."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, lang in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
    return max(counts, key=counts.get) if counts else "en"


@lru_cache(maxsize=None)
def _passages(lang: str) -> dict[str, dict]:
    """doc_id -> {doc_id, source_id, text} for one language."""
    corpus = build_corpus(lang)
    return {
        r.doc_id: {"doc_id": r.doc_id, "source_id": int(r.source_id), "text": r.text}
        for r in corpus.itertuples()
    }


def get_passage(doc_id: str, lang: str) -> dict | None:
    return _passages(lang).get(doc_id)


def split_title(text: str) -> tuple[str, str]:
    """MIRACL passages are 'Title\\nbody'. Returns (title, body)."""
    if "\n" in text:
        title, body = text.split("\n", 1)
        return title.strip(), body.strip()
    return "", text


def _rrf(
    rankings: list[list[str]],
    weights: tuple[float, ...] = (BM25_WEIGHT, DENSE_WEIGHT),
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion. rankings are ordered [bm25, dense]."""
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def search(
    query: str,
    corpus_lang: str,
    mode: str = "hybrid",
    top_k: int = 10,
    query_lang: str | None = None,
) -> tuple[str, list[dict]]:
    """Returns (detected/used query_lang, results). Each result has doc_id, source_id, score, text."""
    if corpus_lang not in LANGUAGES:
        raise ValueError(f"Unsupported corpus language '{corpus_lang}'")
    query_lang = query_lang or detect_language(query)
    n = max(top_k, CANDIDATES) if mode == "hybrid" else top_k

    bm25_hits = (
        bm25.get_index(corpus_lang).search(query, n, query_lang=query_lang)
        if mode in ("bm25", "hybrid") else []
    )
    dense_hits = (
        dense_retrieval.get_index(corpus_lang).search(query, n)
        if mode in ("dense", "hybrid") else []
    )

    if mode == "bm25":
        ranked = bm25_hits
    elif mode == "dense":
        ranked = dense_hits
    else:
        # Cross-script queries give BM25 no matches -> fusion naturally falls back to dense
        ranked = _rrf([[d for d, _ in bm25_hits], [d for d, _ in dense_hits]])

    passages = _passages(corpus_lang)
    results = []
    for doc_id, score in ranked[:top_k]:
        p = passages[doc_id]
        results.append({**p, "score": float(score)})
    return query_lang, results


def warm_up() -> None:
    """Load every index + the embedding model so the first request is fast."""
    from app.services.embeddings import encode_queries

    for lang in LANGUAGES:
        bm25.get_index(lang)
        dense_retrieval.get_index(lang)
        _passages(lang)
    encode_queries(["warm up"])
    logger.info("Retrieval indexes ready for: %s", ", ".join(LANGUAGES))