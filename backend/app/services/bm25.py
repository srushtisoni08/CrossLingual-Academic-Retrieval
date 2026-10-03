import logging
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from app.config import LANGUAGES, MODELS_DIR
from app.services.data_loader import build_corpus, build_queries
from app.services.preprocessing import tokenize

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 (Okapi) over one language's corpus, backed by a scipy sparse matrix."""

    def __init__(self, lang: str, k1: float = 1.5, b: float = 0.75):
        self.lang = lang
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.vocab: dict[str, int] = {}
        self.matrix: csc_matrix | None = None  # (n_docs x n_terms) BM25 weights

    def fit(self, doc_ids: list[str], texts: list[str]) -> "BM25Index":
        self.doc_ids = list(doc_ids)
        n_docs = len(texts)
        vocab: dict[str, int] = {}
        rows, cols, data = [], [], []

        for i, text in enumerate(texts):
            for tok, count in Counter(tokenize(text, self.lang)).items():
                j = vocab.setdefault(tok, len(vocab))
                rows.append(i)
                cols.append(j)
                data.append(count)

        tf = csr_matrix((data, (rows, cols)), shape=(n_docs, len(vocab)), dtype=np.float32)
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = doc_len.mean() or 1.0
        df = np.diff(tf.tocsc().indptr)
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        coo = tf.tocoo()
        denom = coo.data + self.k1 * (1 - self.b + self.b * doc_len[coo.row] / avgdl)
        weights = idf[coo.col] * coo.data * (self.k1 + 1) / denom

        self.vocab = vocab
        self.matrix = csc_matrix((weights, (coo.row, coo.col)), shape=tf.shape, dtype=np.float32)
        return self

    def search(self, query: str, top_k: int = 10, query_lang: str | None = None):
        """Return [(doc_id, score), ...] sorted by score, best first."""
        tokens = set(tokenize(query, query_lang or self.lang))
        term_ids = [self.vocab[t] for t in tokens if t in self.vocab]
        if not term_ids:
            return []  # no overlapping terms (e.g. cross-script query)

        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        m = self.matrix
        for j in term_ids:
            start, end = m.indptr[j], m.indptr[j + 1]
            scores[m.indices[start:end]] += m.data[start:end]

        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    # ---- persistence ----
    # Save plain data (not the class object) so the file loads no matter
    # whether it was built via `python -m app.services.bm25` or from the API.
    @staticmethod
    def path_for(lang: str) -> Path:
        return MODELS_DIR / f"bm25_{lang}_v2.joblib"

    def save(self) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = self.path_for(self.lang)
        joblib.dump(
            {
                "lang": self.lang,
                "k1": self.k1,
                "b": self.b,
                "doc_ids": self.doc_ids,
                "vocab": self.vocab,
                "matrix": self.matrix,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, lang: str) -> "BM25Index":
        state = joblib.load(cls.path_for(lang))
        index = cls(state["lang"], state["k1"], state["b"])
        index.doc_ids = state["doc_ids"]
        index.vocab = state["vocab"]
        index.matrix = state["matrix"]
        return index


def build_index(lang: str, save: bool = True) -> BM25Index:
    corpus = build_corpus(lang)
    index = BM25Index(lang).fit(corpus["doc_id"].tolist(), corpus["text"].tolist())
    if save:
        index.save()
    return index


_cache: dict[str, BM25Index] = {}


def get_index(lang: str) -> BM25Index:
    """Load from disk if built, otherwise build (and save) on first use."""
    if lang not in _cache:
        path = BM25Index.path_for(lang)
        _cache[lang] = BM25Index.load(lang) if path.exists() else build_index(lang)
    return _cache[lang]


def evaluate(query_lang: str, corpus_lang: str, k: int = 10) -> dict:
    """Recall@k and MRR@k: queries in query_lang searched against corpus_lang."""
    index = get_index(corpus_lang)
    queries = build_queries(query_lang)
    hits, rr = 0, 0.0
    for q in queries.itertuples():
        ranked = [d for d, _ in index.search(q.query, k, query_lang=query_lang)]
        if q.relevant_doc_id in ranked:
            hits += 1
            rr += 1.0 / (ranked.index(q.relevant_doc_id) + 1)
    n = len(queries)
    return {"recall": hits / n, "mrr": rr / n}


if __name__ == "__main__":
    # python -m app.services.bm25   (run from backend/)
    logging.basicConfig(level=logging.INFO)
    langs = list(LANGUAGES)
    print(f"{'query -> corpus':<16}{'Recall@10':>10}{'MRR@10':>10}")
    for ql in langs:
        for cl in langs:
            r = evaluate(ql, cl)
            print(f"{ql} -> {cl:<11}{r['recall']:>10.3f}{r['mrr']:>10.3f}")