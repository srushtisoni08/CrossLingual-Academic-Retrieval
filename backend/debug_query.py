"""Show what each retrieval stage returns for one query.

Run from the backend folder:
    python debug_query.py "disney tales" hi
"""
import sys

from app.services import bm25, dense_retrieval, retrieval
from app.services.translate import translate

if len(sys.argv) < 3:
    sys.exit('usage: python debug_query.py "<query>" <corpus_lang: en|hi|bn|te>')

query, corpus = sys.argv[1], sys.argv[2]
query_lang = retrieval.detect_language(query)
translated = translate(query, query_lang, corpus)
print(f"detected query language: {query_lang}")
print(f"translated to {corpus}:   {translated}")

passages = retrieval._passages(corpus)


def show(name, hits, n=10):
    print(f"\n== {name}")
    if not hits:
        print("   (no results)")
    for i, (doc_id, score) in enumerate(hits[:n], 1):
        title, _ = retrieval.split_title(passages[doc_id]["text"])
        print(f"{i:>3}. {score:7.3f}  {doc_id:<10} {title[:60]}")


dense = dense_retrieval.get_index(corpus)
show("dense - original query", dense.search(query, 30))
if translated:
    show("dense - translated query", dense.search(translated, 30))
    show("bm25  - translated query", bm25.get_index(corpus).search(translated, 30, query_lang=corpus))

_, final = retrieval.search(query, corpus, mode="dense", top_k=10)
show("FINAL (what the app shows, mode=dense)", [(r["doc_id"], r["score"]) for r in final])