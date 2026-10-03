import logging

import numpy as np
import pandas as pd

from app.config import BACKEND_DIR, EMBEDDING_MODEL, LANGUAGES
from app.services import bm25, dense_retrieval
from app.services.data_loader import build_queries
from app.services.embeddings import encode_queries
from app.services.retrieval import CANDIDATES, _rrf

logger = logging.getLogger(__name__)

RESULTS_DIR = BACKEND_DIR.parent / "experiments" / "results"
MODES = ("bm25", "dense", "hybrid")


def _rank_of(ranked: list[str], target: str) -> int | None:
    """1-based rank of target in ranked, or None."""
    try:
        return ranked.index(target) + 1
    except ValueError:
        return None


def _metrics(ranks: list[int | None], k: int = 10) -> dict:
    n = len(ranks)
    return {
        "recall@1": sum(r == 1 for r in ranks) / n,
        "recall@10": sum(r is not None and r <= k for r in ranks) / n,
        "mrr@10": sum(1 / r for r in ranks if r is not None and r <= k) / n,
    }


def run_evaluation() -> pd.DataFrame:
    rows = []
    for ql in LANGUAGES:
        queries = build_queries(ql)
        texts = queries["query"].tolist()
        targets = queries["relevant_doc_id"].tolist()
        qemb = encode_queries(texts, show_progress=True)

        for cl in LANGUAGES:
            logger.info("Evaluating %s -> %s", ql, cl)
            b_index, d_index = bm25.get_index(cl), dense_retrieval.get_index(cl)

            # dense: top-N for all queries at once
            scores = qemb @ d_index.embeddings.T
            n = min(CANDIDATES, scores.shape[1])
            top = np.argpartition(-scores, n - 1, axis=1)[:, :n]
            order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
            top = np.take_along_axis(top, order, axis=1)

            ranks = {m: [] for m in MODES}
            for i, (q, target) in enumerate(zip(texts, targets)):
                dense_ids = [d_index.doc_ids[j] for j in top[i]]
                bm25_ids = [d for d, _ in b_index.search(q, CANDIDATES, query_lang=ql)]
                hybrid_ids = [d for d, _ in _rrf([bm25_ids, dense_ids])]
                ranks["bm25"].append(_rank_of(bm25_ids, target))
                ranks["dense"].append(_rank_of(dense_ids, target))
                ranks["hybrid"].append(_rank_of(hybrid_ids, target))

            for mode in MODES:
                rows.append({"query_lang": ql, "corpus_lang": cl, "mode": mode,
                             **_metrics(ranks[mode])})
    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame) -> None:
    wide = df.pivot_table(index=["query_lang", "corpus_lang"], columns="mode",
                          values=["recall@10", "mrr@10"])
    print(f"\nModel: {EMBEDDING_MODEL}")
    print(f"{'pair':<9}" + "".join(f"{'R@10 ' + m:>14}" for m in MODES)
          + "".join(f"{'MRR ' + m:>13}" for m in MODES))
    for (ql, cl), r in wide.iterrows():
        print(f"{ql}->{cl:<6}"
              + "".join(f"{r[('recall@10', m)]:>14.3f}" for m in MODES)
              + "".join(f"{r[('mrr@10', m)]:>13.3f}" for m in MODES))

    df = df.assign(kind=np.where(df.query_lang == df.corpus_lang, "monolingual", "cross-lingual"))
    summary = df.groupby(["kind", "mode"])[["recall@1", "recall@10", "mrr@10"]].mean().round(3)
    print("\nAverages:\n", summary.to_string())


if __name__ == "__main__":
    # python -m app.services.evaluation   (run from backend/)
    logging.basicConfig(level=logging.INFO)
    result = run_evaluation()
    print_report(result)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = EMBEDDING_MODEL.replace("/", "__")
    out = RESULTS_DIR / f"retrieval_eval_{slug}.csv"
    result.round(4).to_csv(out, index=False)
    print(f"\nSaved: {out}")