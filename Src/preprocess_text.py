"""
Src/preprocess_text.py — Multilingual-aware text cleaner.

Upgraded to support Amharic, Afan Oromo, Tigrinya, and English.
Falls back to English-only cleaning for the original ML/LSTM models.

Original author: Soumyajit Behera
Ethiopian adaptation: Robel Tesfaye (https://github.com/Riter-Rob)
"""
import sys
import os
import re
from nltk.stem import WordNetLemmatizer

# ─── Import Ethiopian cleaner if available ────────────────────────────────────
_ethiopia_root = os.path.join(os.path.dirname(__file__), "..", "Ethiopia")
sys.path.insert(0, os.path.dirname(_ethiopia_root))

try:
    from Ethiopia.preprocessing.multilingual_cleaner import MultilingualCleaner
    from Ethiopia.preprocessing.amharic_normalizer import AmharicNormalizer
    _multilingual_available = True
    _eth_cleaner = MultilingualCleaner()
except ImportError:
    _multilingual_available = False

# ─── Original English pipeline (kept for backward compatibility) ──────────────
import spacy
sp = spacy.load("en_core_web_sm")
all_stopwords = sp.Defaults.stop_words

new_words = ["http", "www", "co", "u", "com", "t", "s", "m",
             "ve", "dy", "ll", "n", "r", "b", "wa", "y", "don", "ha"]
for words in new_words:
    all_stopwords.add(words)

wordnet = WordNetLemmatizer()


def clean_text(text: str, stopwords=all_stopwords) -> str:
    """
    Clean input text for model inference.

    If the text contains Ethiopic (Ge'ez) characters, routes through
    the multilingual Ethiopian cleaner.
    Otherwise, uses the original English preprocessing pipeline
    (lemmatization + TF-IDF stopword removal).

    Args:
        text:       Raw input text string.
        stopwords:  English stopword set (used only for Latin text).

    Returns:
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Detect if text contains Ethiopic Ge'ez characters
    has_geez = bool(re.search(r"[\u1200-\u137F\u1380-\u139F]", text))

    if has_geez and _multilingual_available:
        # Use multilingual cleaner (preserves Ge'ez characters)
        return _eth_cleaner.clean_text(text)

    # ── Original English-only pipeline ──────────────────────────────────────
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split(" ")
    text = [wordnet.lemmatize(word) for word in text]
    text = [word for word in text if word not in stopwords]
    text = " ".join(text)
    return text


def clean_text_ethiopian(text: str) -> dict:
    """
    Clean text and return full metadata (language, script, cleaned text).
    Used by the new Ethiopian Flask endpoint.

    Returns:
        dict: {"text": str, "lang": str, "script": str}
    """
    if _multilingual_available:
        return _eth_cleaner.clean(text)
    # Fallback
    return {"text": clean_text(text), "lang": "en", "script": "latin"}
