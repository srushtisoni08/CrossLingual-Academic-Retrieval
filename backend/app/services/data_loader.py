import logging
from functools import lru_cache

import pandas as pd

from app.config import LANGUAGES, RAW_DATA_DIR, SPLIT

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"id", "query", "positive", "negative"}


def _check_lang(lang: str) -> None:
    if lang not in LANGUAGES:
        raise ValueError(f"Unsupported language '{lang}'. Choose from {list(LANGUAGES)}")


@lru_cache(maxsize=None)
def load_language(lang: str) -> pd.DataFrame:
    """Load the raw triplets (id, query, positive, negative) for one language."""
    _check_lang(lang)
    lang_dir = RAW_DATA_DIR / lang
    parquet_path = lang_dir / f"{SPLIT}.parquet"
    csv_path = lang_dir / f"{SPLIT}.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No {SPLIT}.parquet/.csv found in {lang_dir}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{lang}: missing columns {missing}. Found {list(df.columns)}")

    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    for col in ("query", "positive", "negative"):
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col] != ""]
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)

    if len(df) < before:
        logger.warning("%s: dropped %d invalid rows", lang, before - len(df))
    return df


def build_corpus(lang: str) -> pd.DataFrame:
    """
    One row per passage. doc_id is language-independent ("<id>-p" / "<id>-n"),
    so the same passage has the same doc_id in every language. This is what
    makes cross-lingual evaluation possible later.
    """
    df = load_language(lang)
    pos = pd.DataFrame({
        "doc_id": df["id"].astype(str) + "-p",
        "text": df["positive"],
        "label": "positive",
        "source_id": df["id"],
    })
    neg = pd.DataFrame({
        "doc_id": df["id"].astype(str) + "-n",
        "text": df["negative"],
        "label": "negative",
        "source_id": df["id"],
    })
    corpus = pd.concat([pos, neg], ignore_index=True)
    corpus["lang"] = lang
    return corpus


def build_queries(lang: str) -> pd.DataFrame:
    """One row per query, with the doc_id of its relevant (positive) passage."""
    df = load_language(lang)
    return pd.DataFrame({
        "query_id": df["id"],
        "query": df["query"],
        "relevant_doc_id": df["id"].astype(str) + "-p",
        "lang": lang,
    })


if __name__ == "__main__":
    # Sanity check: python -m app.services.data_loader  (run from backend/)
    logging.basicConfig(level=logging.INFO)
    for code in LANGUAGES:
        try:
            q, c = build_queries(code), build_corpus(code)
            print(f"[{code}] queries={len(q)} passages={len(c)}")
            print("   sample query  :", q.iloc[0]["query"])
            print("   sample passage:", c.iloc[0]["text"][:100])
        except (FileNotFoundError, ValueError) as e:
            print(f"[{code}] ERROR: {e}")