import logging

import numpy as np

from app.config import EMBEDDING_MODEL, LANGUAGES
from app.services import dense_retrieval
from app.services.data_loader import build_queries
from app.services.embeddings import encode_queries
from app.services.retrieval import MIN_BODY_CHARS, _passages, split_title

logger = logging.getLogger(__name__)

N_REAL = 500  # real questions per query language (first N rows; keeps this fast)

# Off-topic probes: chit-chat and gibberish that no passage should answer
PROBES = {
    "en": ["what are u doing?", "I love you", "hello how are you", "good morning",
           "what is your name", "asdf qwerty zxcv"],
    "hi": ["तुम क्या कर रहे हो?", "मैं तुमसे प्यार करता हूँ", "आप कैसे हैं?", "सुप्रभात",
           "तुम्हारा नाम क्या है?", "asdf qwerty zxcv"],
    "bn": ["তুমি কী করছো?", "আমি তোমাকে ভালোবাসি", "আপনি কেমন আছেন?", "সুপ্রভাত",
           "তোমার নাম কী?", "asdf qwerty zxcv"],
    "te": ["నువ్వు ఏమి చేస్తున్నావు?", "నేను నిన్ను ప్రేమిస్తున్నాను", "మీరు ఎలా ఉన్నారు?",
           "శుభోదయం", "నీ పేరు ఏమిటి?", "asdf qwerty zxcv"],
}


def _top1(qemb: np.ndarray, index, valid: np.ndarray) -> np.ndarray:
    """Best similarity per query, ignoring stub passages (same filter as the API)."""
    return (qemb @ index.embeddings[valid].T).max(axis=1)


def _pct(x, *qs):
    return " / ".join(f"{np.percentile(x, q):.3f}" for q in qs)


if __name__ == "__main__":
    # python -m app.services.calibration   (run from backend/)
    logging.basicConfig(level=logging.WARNING)
    probe_emb = {l: encode_queries(q) for l, q in PROBES.items()}
    real_emb = {l: encode_queries(build_queries(l)["query"].tolist()[:N_REAL], show_progress=True)
                for l in LANGUAGES}

    print(f"\nModel: {EMBEDDING_MODEL}   (top-1 cosine similarity; stub passages excluded)")
    for cl in LANGUAGES:
        index = dense_retrieval.get_index(cl)
        passages = _passages(cl)
        valid = np.array([len(split_title(passages[d]["text"])[1]) >= MIN_BODY_CHARS
                          for d in index.doc_ids])

        probes = np.concatenate([_top1(probe_emb[ql], index, valid) for ql in LANGUAGES])
        same = _top1(real_emb[cl], index, valid)
        cross = np.concatenate([_top1(real_emb[ql], index, valid) for ql in LANGUAGES if ql != cl])
        threshold = float(probes.max()) + 0.005

        print(f"\n== Corpus: {cl} ==")
        print(f"  off-topic probes   min / median / max : "
              f"{probes.min():.3f} / {np.median(probes):.3f} / {probes.max():.3f}")
        print(f"  real, same-lang    p5 / p25 / median  : {_pct(same, 5, 25, 50)}")
        print(f"  real, cross-lang   p5 / p25 / median  : {_pct(cross, 5, 25, 50)}")
        print(f"  threshold = max probe + 0.005 = {threshold:.3f}  ->  would wrongly flag "
              f"{(same < threshold).mean():.1%} of same-lang and {(cross < threshold).mean():.1%} of cross-lang real queries")