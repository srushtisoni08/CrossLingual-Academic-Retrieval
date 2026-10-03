from typing import Literal

from pydantic import BaseModel, Field

Lang = Literal["en", "hi", "bn", "te"]
Mode = Literal["bm25", "dense", "hybrid"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    corpus_lang: Lang = "en"              # language of the documents to search
    query_lang: Lang | None = None        # None = auto-detect from script
    mode: Mode = "dense"
    top_k: int = Field(10, ge=1, le=50)


class SearchResult(BaseModel):
    rank: int
    doc_id: str
    source_id: int
    title: str
    snippet: str
    score: float
    lang: Lang


class SearchResponse(BaseModel):
    query: str
    query_lang: Lang
    corpus_lang: Lang
    mode: Mode
    took_ms: float
    results: list[SearchResult]


class Paper(BaseModel):
    doc_id: str
    source_id: int
    lang: Lang
    title: str
    text: str