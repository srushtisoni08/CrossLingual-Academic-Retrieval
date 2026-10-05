import logging
from functools import lru_cache

from app.config import TRANSLATION_MODEL

logger = logging.getLogger(__name__)

NLLB_CODES = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "te": "tel_Telu"}


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL).eval()
    return tok, model


@lru_cache(maxsize=2048)
def translate(text: str, src: str, tgt: str) -> str | None:
    """Translation, or None if unsupported/failed (callers fall back to the original query)."""
    if src == tgt or src not in NLLB_CODES or tgt not in NLLB_CODES:
        return None
    try:
        import torch

        tok, model = _load()
        tok.src_lang = NLLB_CODES[src]
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                forced_bos_token_id=tok.convert_tokens_to_ids(NLLB_CODES[tgt]),
                max_new_tokens=64,
                num_beams=4,
            )
        return tok.batch_decode(out, skip_special_tokens=True)[0].strip() or None
    except Exception:
        logger.exception("Query translation failed; falling back to original query")
        return None