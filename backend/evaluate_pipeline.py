import json
import random
import sys
import time

import numpy as np

from app.config import MODELS_DIR
from app.services import bm25, dense_retrieval
from app.services.data_loader import build_queries
from app.services.embeddings import encode_queries
from app.services.retrieval import _rrf
from app.services.translate import translate

K = 50  # candidates each retriever contributes
CACHE = MODELS_DIR / "eval_translations.json"

# name -> weight per ranking. d_orig = dense with the original query,
# d_trans = dense with the translated query, bm = BM25 with the translated query.
FUSIONS = {
    "fuse: d_orig2 + d_trans2   (app, mode=dense)": {"d_orig": 2, "d_trans": 2},
    "fuse: bm1 + d_orig2 + d_trans2 (app, hybrid)": {"bm": 1, "d_orig": 2, "d_trans": 2},
    "fuse: d_trans2 + bm1": {"d_trans": 2, "bm": 1},
    "fuse: d_orig1 + d_trans1 + bm1": {"d_orig": 1, "d_trans": 1, "bm": 1},
    "fuse: d_trans2 + d_orig0.5": {"d_trans": 2, "d_orig": 0.5},
    "fuse: d_trans2 + d_orig1": {"d_trans": 2, "d_orig": 1},
    "fuse: d_trans2 + d_orig0.5 + bm0.5": {"d_trans": 2, "d_orig": 0.5, "bm": 0.5},
    "fuse: d_trans2 + d_orig1 + bm0.5": {"d_trans": 2, "d_orig": 1, "bm": 0.5},
    "fuse: d_trans3 + d_orig1 + bm1": {"d_trans": 3, "d_orig": 1, "bm": 1},
}


def parse_args():
    pairs, n = [], 200
    for a in sys.argv[1:]:
        if a.startswith("n="):
            n = int(a[2:])
        else:
            ql, cl = a.split(":")
            pairs.append((ql, cl))
    return pairs or [("en", "hi"), ("en", "te"), ("hi", "en"), ("te", "en")], n


def top_ids(scores: np.ndarray, ids: list[str], k: int = K) -> list[str]:
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [ids[i] for i in top]


def fuse(lists: dict[str, list[str]], weights: dict[str, float]) -> list[str]:
    names = [n for n in weights if lists.get(n)]
    if not names:
        return []
    return [d for d, _ in _rrf([lists[n] for n in names], tuple(weights[n] for n in names))]


def main():
    pairs, n = parse_args()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    overall: dict[str, list[tuple[float, float]]] = {}

    for ql, cl in pairs:
        queries = build_queries(ql)
        rows = random.Random(0).sample(range(len(queries)), min(n, len(queries)))
        texts = [queries.iloc[i]["query"] for i in rows]
        rel = [queries.iloc[i]["relevant_doc_id"] for i in rows]

        t0 = time.time()
        trans = []
        for j, text in enumerate(texts):
            key = f"{ql}|{cl}|{text}"
            if key not in cache:
                cache[key] = translate(text, ql, cl) or text
            trans.append(cache[key])
            if (j + 1) % 25 == 0:
                print(f"  [{ql}->{cl}] translated {j + 1}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

        dense = dense_retrieval.get_index(cl)
        bm = bm25.get_index(cl)
        s_orig = encode_queries(texts) @ dense.embeddings.T
        s_trans = encode_queries(trans) @ dense.embeddings.T

        results: dict[str, list[tuple[float, float]]] = {}

        def add(name: str, ranking: list[str], target: str):
            top = ranking[:10]
            hit = target in top
            results.setdefault(name, []).append(
                (1.0 if hit else 0.0, 1.0 / (top.index(target) + 1) if hit else 0.0)
            )

        for i, target in enumerate(rel):
            lists = {
                "d_orig": top_ids(s_orig[i], dense.doc_ids),
                "d_trans": top_ids(s_trans[i], dense.doc_ids),
                "bm": [d for d, _ in bm.search(trans[i], K, query_lang=cl)],
            }
            add("dense, original query", lists["d_orig"], target)
            add("dense, translated query", lists["d_trans"], target)
            add("bm25, translated query", lists["bm"], target)
            for name, w in FUSIONS.items():
                add(name, fuse(lists, w), target)

        print(f"\n=== {ql} -> {cl}   ({len(rel)} queries)")
        print(f"{'strategy':<48}{'Recall@10':>10}{'MRR@10':>9}")
        for name, vals in results.items():
            arr = np.array(vals)
            print(f"{name:<48}{arr[:, 0].mean():>10.3f}{arr[:, 1].mean():>9.3f}")
            overall.setdefault(name, []).extend(vals)

    if len(pairs) > 1:
        print(f"\n=== ALL PAIRS COMBINED")
        print(f"{'strategy':<48}{'Recall@10':>10}{'MRR@10':>9}")
        for name, vals in overall.items():
            arr = np.array(vals)
            print(f"{name:<48}{arr[:, 0].mean():>10.3f}{arr[:, 1].mean():>9.3f}")


if __name__ == "__main__":
    main()