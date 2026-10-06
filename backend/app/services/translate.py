import logging
import threading
from functools import lru_cache

from app.config import TRANSLATION_MODEL

logger = logging.getLogger(__name__)

NLLB_CODES = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "te": "tel_Telu"}


_load_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load_unlocked():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # hides the repeated max_new_tokens/max_length warning
    tok = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL).eval()
    try:  # int8 weights: ~2x faster on CPU and ~4x less RAM, tiny quality cost
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception:
        logger.warning("Quantization unavailable; using full-precision translation model")
    return tok, model


def _load():
    with _load_lock:  # background preload and a first request must not load it twice
        return _load_unlocked()


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
                num_beams=2,
            )
        return tok.batch_decode(out, skip_special_tokens=True)[0].strip() or None
    except Exception:
        logger.exception("Query translation failed; falling back to original query")
        return None