import logging
from functools import lru_cache

from app.config import LANGUAGES, RERANK, RERANK_TOP_N, TRANSLATE_QUERY
from app.services import bm25, dense_retrieval
from app.services.data_loader import build_corpus

logger = logging.getLogger(__name__)

RRF_K = 60          # standard Reciprocal Rank Fusion constant
BM25_WEIGHT = 1.0   # fusion weights: dense is more reliable here (esp. cross-lingual),
DENSE_WEIGHT = 2.0  # so it gets double the vote. Try 1.0 / 3.0 and re-run evaluation.
CANDIDATES = 50     # how many results each retriever contributes to fusion
# Cross-lingual weights, chosen from docs/methodology.md section 4.5:
# the translated query is the strong signal; the original query and BM25 are low-weight helpers.
CROSS_TRANS_WEIGHT = 2.0
CROSS_ORIG_WEIGHT = 0.5
CROSS_BM25_WEIGHT = 0.5
MIN_BODY_CHARS = 50 # skip stub passages (e.g. a title + 2 words) in search results

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
    n = max(top_k * 3, CANDIDATES)
    cross = query_lang != corpus_lang

    translated = None
    if cross and TRANSLATE_QUERY:
        from app.services.translate import translate
        translated = translate(query.strip(), query_lang, corpus_lang)

    dense_idx = dense_retrieval.get_index(corpus_lang) if mode in ("dense", "hybrid") else None
    bm25_idx = bm25.get_index(corpus_lang) if mode in ("bm25", "hybrid") else None

    rankings, weights = [], []
    if dense_idx:
        if translated:
            # dense mode: translated query only (best measured MRR);
            # hybrid mode: original query joins at low weight as a safety net.
            rankings.append(dense_idx.search(translated, n))
            weights.append(CROSS_TRANS_WEIGHT)
            if mode == "hybrid":
                rankings.append(dense_idx.search(query, n))
                weights.append(CROSS_ORIG_WEIGHT)
        else:  # same-language query, or translation unavailable
            rankings.append(dense_idx.search(query, n))
            weights.append(DENSE_WEIGHT)
    if bm25_idx:
        if not cross:
            rankings.append(bm25_idx.search(query, n, query_lang=query_lang))
            weights.append(BM25_WEIGHT)
        elif translated:  # BM25 only works across scripts via the translated query
            rankings.append(bm25_idx.search(translated, n, query_lang=corpus_lang))
            weights.append(CROSS_BM25_WEIGHT)

    if len(rankings) == 1:
        ranked = rankings[0]
    else:
        ranked = _rrf([[d for d, _ in r] for r in rankings], tuple(weights))

    passages = _passages(corpus_lang)
    pool_size = max(top_k, RERANK_TOP_N) if RERANK else top_k
    pool = []
    for doc_id, score in ranked:
        p = passages[doc_id]
        if len(split_title(p["text"])[1]) < MIN_BODY_CHARS:
            continue
        pool.append({**p, "score": float(score)})
        if len(pool) == pool_size:
            break

    if RERANK and pool:
        from app.services.reranker import rerank
        pool = rerank(query, pool)
    return query_lang, pool[:top_k] 


def warm_up() -> None:
    """Load every index + the embedding model so the first request is fast."""
    from app.services.embeddings import encode_queries

    for lang in LANGUAGES:
        bm25.get_index(lang)
        dense_retrieval.get_index(lang)
        _passages(lang)
    encode_queries(["warm up"])
    logger.info("Retrieval indexes ready for: %s", ", ".join(LANGUAGES))


def warm_up_optional() -> None:
    """Preload the translation / reranker models in the background so the server
    starts accepting requests immediately (first cross-lingual query waits for it)."""
    if TRANSLATE_QUERY:
        from app.services.translate import translate
        translate("warm up", "en", "hi")
    if RERANK:
        from app.services.reranker import rerank
        rerank("warm up", [{"text": "warm up"}])
    logger.info("Translation/rerank models ready")