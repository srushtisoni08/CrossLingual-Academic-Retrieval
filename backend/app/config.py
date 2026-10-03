from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BACKEND_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BACKEND_DIR / "data" / "processed"
MODELS_DIR = BACKEND_DIR / "models"

# language code -> display name
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
}

SPLIT = "train"  

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256

# E5 models need these prefixes; other models don't.
_IS_E5 = "e5" in EMBEDDING_MODEL.lower()
QUERY_PREFIX = "query: " if _IS_E5 else ""
PASSAGE_PREFIX = "passage: " if _IS_E5 else ""