import unicodedata

import regex  # third-party 'regex' (already in requirements) - needed for Indic scripts

# Python's built-in re.\w+ splits Hindi/Bengali/Telugu words at vowel signs
# (matras), because those are combining marks (\p{M}), not letters.
# So we match letters + marks + numbers explicitly.
_TOKEN_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+")
_INVISIBLE_RE = regex.compile(r"[\u200b\u200c\u200d\ufeff]")  # zero-width chars
_SPACE_RE = regex.compile(r"\s+")

# Small, conservative stopword lists (extend later if evaluation shows a need)
STOPWORDS = {
    "en": {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in",
        "on", "at", "to", "for", "from", "by", "with", "and", "or", "as", "that",
        "this", "it", "its", "what", "which", "who", "when", "where", "how",
        "did", "does", "do",
    },
    "hi": {
        "का", "के", "की", "है", "हैं", "में", "और", "से", "को", "पर", "यह",
        "था", "थे", "थी", "ने", "कि", "एक", "भी", "तो", "कब", "क्या", "कैसे",
    },
    "bn": {
        "এবং", "এর", "এই", "যে", "থেকে", "করে", "ছিল", "না", "জন্য", "একটি",
        "তার", "হয়", "কখন", "কী", "কি",
    },
    "te": {
        "మరియు", "ఈ", "ఆ", "ఒక", "లో", "కు", "యొక్క", "అది", "ఇది", "ఉంది",
        "ఎప్పుడు", "ఏమిటి",
    },
}


def normalize_text(text: str) -> str:
    """Unicode-normalize, lowercase, drop zero-width chars, collapse whitespace."""
    text = unicodedata.normalize("NFKC", str(text))
    text = _INVISIBLE_RE.sub("", text)
    text = text.lower()
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text: str, lang: str | None = None, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(normalize_text(text))
    if remove_stopwords and lang in STOPWORDS:
        stop = STOPWORDS[lang]
        tokens = [t for t in tokens if t not in stop]
    return tokens


if __name__ == "__main__":
    for lang, s in [
        ("en", "When was quantum field theory developed?"),
        ("hi", "क्वांटम क्षेत्र सिद्धांत का विकास कब हुआ?"),
        ("bn", "কোয়ান্টাম ক্ষেত্র তত্ত্ব কখন উদ্ভাবিত হয়েছিল?"),
        ("te", "క్వాంటం క్షేత్ర సిద్ధాంతం ఎప్పుడు అభివృద్ధి చేయబడింది?"),
    ]:
        print(lang, tokenize(s, lang))