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


def _signals(qemb: np.ndarray, index, valid: np.ndarray):
    """Per query: (top-1 similarity, gap between top-1 and top-10), stub passages excluded."""
    scores = qemb @ index.embeddings[valid].T
    top1 = scores.max(axis=1)
    top10 = np.partition(scores, -10, axis=1)[:, -10]
    return top1, top1 - top10


def _auc(real: np.ndarray, probe: np.ndarray) -> float:
    """P(a random real query scores higher than a random off-topic probe)."""
    return float((real[:, None] > probe[None, :]).mean())


def _line(name: str, real: np.ndarray, probe: np.ndarray) -> str:
    return (f"{name}: probes max {probe.max():.3f} | real p5 {np.percentile(real, 5):.3f}"
            f" median {np.median(real):.3f} | AUC {_auc(real, probe):.2f}")


if __name__ == "__main__":
    # python -m app.services.calibration   (run from backend/)
    logging.basicConfig(level=logging.WARNING)
    probe_emb = {l: encode_queries(q) for l, q in PROBES.items()}
    real_emb = {l: encode_queries(build_queries(l)["query"].tolist()[:N_REAL], show_progress=True)
                for l in LANGUAGES}

    print(f"\nModel: {EMBEDDING_MODEL}   (stub passages excluded; AUC 1.0 = perfect, 0.5 = useless)")
    for cl in LANGUAGES:
        index = dense_retrieval.get_index(cl)
        passages = _passages(cl)
        valid = np.array([len(split_title(passages[d]["text"])[1]) >= MIN_BODY_CHARS
                          for d in index.doc_ids])
        others = [l for l in LANGUAGES if l != cl]

        def collect(emb_by_lang, langs):
            parts = [_signals(emb_by_lang[l], index, valid) for l in langs]
            return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))

        real = {"same": collect(real_emb, [cl]), "cross": collect(real_emb, others)}
        probe = {"same": collect(probe_emb, [cl]), "cross": collect(probe_emb, others)}

        print(f"\n== Corpus: {cl} ==")
        for rel in ("same", "cross"):
            print(f"  [{rel}-lang]")
            print("    similarity  " + _line("", real[rel][0], probe[rel][0])[2:])
            print("    top1-top10  " + _line("", real[rel][1], probe[rel][1])[2:])