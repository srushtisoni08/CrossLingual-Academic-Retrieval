import time

from fastapi import APIRouter, HTTPException, Query

from app.schemas.search import Lang, Mode, SearchRequest, SearchResponse, SearchResult
from app.services import retrieval

router = APIRouter(prefix="/search", tags=["search"])

SNIPPET_CHARS = 300


def _run(req: SearchRequest) -> SearchResponse:
    start = time.perf_counter()
    try:
        query_lang, hits = retrieval.search(
            req.query, req.corpus_lang, req.mode, req.top_k, req.query_lang
        )
    except FileNotFoundError as e:  # index/data not built yet
        raise HTTPException(status_code=503, detail=str(e))

    results = []
    for rank, h in enumerate(hits, start=1):
        title, body = retrieval.split_title(h["text"])
        results.append(SearchResult(
            rank=rank,
            doc_id=h["doc_id"],
            source_id=h["source_id"],
            title=title,
            snippet=body[:SNIPPET_CHARS] + ("..." if len(body) > SNIPPET_CHARS else ""),
            score=h["score"],
            lang=req.corpus_lang,
        ))
    return SearchResponse(
        query=req.query,
        query_lang=query_lang,
        corpus_lang=req.corpus_lang,
        mode=req.mode,
        took_ms=round((time.perf_counter() - start) * 1000, 1),
        results=results,
    )


@router.post("", response_model=SearchResponse)
def search_post(req: SearchRequest):
    return _run(req)


@router.get("", response_model=SearchResponse)
def search_get(
    q: str = Query(..., min_length=1, max_length=500),
    corpus_lang: Lang = "en",
    query_lang: Lang | None = None,
    mode: Mode = "dense",
    top_k: int = Query(10, ge=1, le=50),
):
    """Same as POST, handy for the browser address bar and quick tests."""
    return _run(SearchRequest(
        query=q, corpus_lang=corpus_lang, query_lang=query_lang, mode=mode, top_k=top_k
    ))