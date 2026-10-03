from fastapi import APIRouter, HTTPException

from app.config import LANGUAGES
from app.schemas.search import Lang, Paper
from app.services import retrieval

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("/languages")
def list_languages():
    return [{"code": c, "name": n} for c, n in LANGUAGES.items()]


@router.get("/{doc_id}", response_model=Paper)
def get_paper(doc_id: str, lang: Lang = "en"):
    """Fetch a passage. doc_ids are shared across languages, so the same
    doc_id with ?lang=hi returns the Hindi version of the same passage."""
    try:
        p = retrieval.get_passage(doc_id, lang)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if p is None:
        raise HTTPException(status_code=404, detail=f"Passage '{doc_id}' not found in '{lang}'")
    title, body = retrieval.split_title(p["text"])
    return Paper(doc_id=doc_id, source_id=p["source_id"], lang=lang, title=title, text=body)