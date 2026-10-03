import logging
from pathlib import Path

import numpy as np

from app.config import EMBEDDING_MODEL, LANGUAGES, MODELS_DIR
from app.services.data_loader import build_corpus, build_queries
from app.services.embeddings import encode_passages, encode_queries

logger = logging.getLogger(__name__)

_MODEL_SLUG = EMBEDDING_MODEL.replace("/", "__")


class DenseIndex:
    """Passage embeddings for one language. Search = one matrix multiply."""

    def __init__(self, lang: str, doc_ids: list[str], embeddings: np.ndarray):
        self.lang = lang
        self.doc_ids = list(doc_ids)
        self.embeddings = embeddings  # (n_docs, dim), L2-normalized
        self.doc_pos = {d: i for i, d in enumerate(self.doc_ids)}

    # ---- build / persistence ----
    @staticmethod
    def path_for(lang: str) -> Path:
        return MODELS_DIR / f"dense_{_MODEL_SLUG}_{lang}.npz"

    @classmethod
    def build(cls, lang: str) -> "DenseIndex":
        corpus = build_corpus(lang)
        logger.info("Encoding %d passages (%s)...", len(corpus), lang)
        emb = encode_passages(corpus["text"].tolist())
        return cls(lang, corpus["doc_id"].tolist(), emb)

    def save(self) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = self.path_for(self.lang)
        np.savez(path, embeddings=self.embeddings, doc_ids=np.array(self.doc_ids))
        return path

    @classmethod
    def load(cls, lang: str) -> "DenseIndex":
        data = np.load(cls.path_for(lang), allow_pickle=False)
        return cls(lang, data["doc_ids"].tolist(), data["embeddings"])

    # ---- search ----
    def search_vector(self, qvec: np.ndarray, top_k: int = 10):
        scores = self.embeddings @ qvec
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[i], float(scores[i])) for i in top]

    def search(self, query: str, top_k: int = 10):
        """Query can be in ANY language - the embedding space is shared."""
        return self.search_vector(encode_queries([query])[0], top_k)


_cache: dict[str, DenseIndex] = {}


def get_index(lang: str) -> DenseIndex:
    """Load from disk if built, otherwise build (and save) on first use."""
    if lang not in _cache:
        if DenseIndex.path_for(lang).exists():
            _cache[lang] = DenseIndex.load(lang)
        else:
            index = DenseIndex.build(lang)
            index.save()
            _cache[lang] = index
    return _cache[lang]


def evaluate_matrix(k: int = 10) -> dict:
    """Recall@k / MRR@k for every (query language -> corpus language) pair."""
    results = {}
    for ql in LANGUAGES:
        queries = build_queries(ql)
        qemb = encode_queries(queries["query"].tolist(), show_progress=True)
        for cl in LANGUAGES:
            index = get_index(cl)
            scores = qemb @ index.embeddings.T  # (n_queries, n_docs)
            rel = np.array([index.doc_pos[d] for d in queries["relevant_doc_id"]])
            rel_score = scores[np.arange(len(rel)), rel][:, None]
            rank = (scores > rel_score).sum(axis=1)  # 0-based rank of the relevant doc
            hit = rank < k
            results[(ql, cl)] = {
                "recall": float(hit.mean()),
                "mrr": float(np.where(hit, 1.0 / (rank + 1), 0.0).mean()),
            }
    return results


if __name__ == "__main__":
    # python -m app.services.dense_retrieval   (run from backend/)
    logging.basicConfig(level=logging.INFO)
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"{'query -> corpus':<16}{'Recall@10':>10}{'MRR@10':>10}")
    for (ql, cl), r in evaluate_matrix().items():
        print(f"{ql} -> {cl:<11}{r['recall']:>10.3f}{r['mrr']:>10.3f}")