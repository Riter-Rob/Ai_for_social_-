"""
Ethiopia/preprocessing/multilingual_cleaner.py

Multilingual text cleaner for Ethiopian languages.
Supports: Amharic (am), Afan Oromo (om), Tigrinya (ti), English (en),
          and mixed/code-switched text (Fidelfon / Franco-Amharic).

Usage:
    cleaner = MultilingualCleaner()
    result = cleaner.clean("ህይወት መረረኝ ብቻ ሆኜ")
    # → {"text": "ህይወት መረረ ብቻ ሆ", "lang": "am", "script": "geez"}
"""

import re
import unicodedata
from typing import Tuple

from .amharic_normalizer import AmharicNormalizer


# ─── Unicode ranges ───────────────────────────────────────────────────────────
# Ethiopic (Ge'ez) block: U+1200–U+137F  (core block)
# Ethiopic Supplement:    U+1380–U+139F
# Ethiopic Extended:      U+2D80–U+2DDF
GEEZ_PATTERN = re.compile(
    r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]"
)

# Latin characters (for English and Fidelfon / Romanized Amharic)
LATIN_PATTERN = re.compile(r"[a-zA-Z]")

# URLs — common noise in social media text
URL_PATTERN = re.compile(r"http\S+|www\.\S+|t\.co/\S+")

# Mentions and hashtags
MENTION_PATTERN = re.compile(r"[@#]\w+")

# Repeated characters (e.g. "noooo" → "no", "ይሄዉዉዉ" → "ይሄዉ")
REPEAT_PATTERN = re.compile(r"(.)\1{2,}")

# Web/social noise tokens (Latin)
LATIN_NOISE_TOKENS = {
    "http", "https", "www", "com", "org", "net", "co", "io",
    "rt", "dm", "lol", "omg", "btw", "fyi", "imo", "tbh",
    "u", "ur", "r", "b", "n", "m", "s", "t",
}


class MultilingualCleaner:
    """
    Language-aware text cleaner for Ethiopian NLP tasks.

    Automatically detects whether text is written in:
      - Ge'ez script (Amharic / Tigrinya)
      - Latin script (English or Fidelfon/Romanized Ethiopian)
      - Mixed (code-switching, very common among Ethiopian youth)

    Then applies the appropriate cleaning strategy.
    """

    def __init__(self):
        self.amharic_normalizer = AmharicNormalizer()

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def clean(self, text: str) -> dict:
        """
        Clean and normalize text.

        Args:
            text: Raw input text (any language / script).

        Returns:
            dict with keys:
              "text"   — cleaned text string
              "lang"   — detected language code ("am", "om", "ti", "en", "mixed")
              "script" — detected script ("geez", "latin", "mixed")
        """
        if not text or not isinstance(text, str):
            return {"text": "", "lang": "unknown", "script": "unknown"}

        # Normalize unicode (NFC composition)
        text = unicodedata.normalize("NFC", text)

        # Remove URLs and mentions
        text = URL_PATTERN.sub(" ", text)
        text = MENTION_PATTERN.sub(" ", text)

        # Detect script composition
        script, lang = self._detect_script_and_language(text)

        if script == "geez":
            cleaned = self._clean_geez(text)
        elif script == "latin":
            cleaned = self._clean_latin(text)
        else:
            # Mixed: clean both parts
            cleaned = self._clean_mixed(text)

        return {"text": cleaned, "lang": lang, "script": script}

    def clean_text(self, text: str) -> str:
        """
        Simple interface returning only the cleaned string.
        Drop-in compatible with the existing Flask/preprocess_text.py API.
        """
        return self.clean(text)["text"]

    # ──────────────────────────────────────────────────────────────────────────
    #  Script Detection
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_script_and_language(self, text: str) -> Tuple[str, str]:
        """
        Detect the dominant script and infer language.

        Returns:
            (script, lang) tuple.
            script: "geez" | "latin" | "mixed"
            lang:   "am" | "om" | "ti" | "en" | "mixed"
        """
        geez_chars = len(GEEZ_PATTERN.findall(text))
        latin_chars = len(LATIN_PATTERN.findall(text))
        total = geez_chars + latin_chars

        if total == 0:
            return "unknown", "unknown"

        geez_ratio = geez_chars / total

        if geez_ratio > 0.75:
            # Predominantly Ge'ez — classify as Amharic (most common)
            # Note: Tigrinya also uses Ge'ez; for now we default to "am"
            # A language ID model would refine this in Phase 3.
            return "geez", "am"
        elif geez_ratio < 0.25:
            # Predominantly Latin
            lang = self._guess_latin_language(text)
            return "latin", lang
        else:
            return "mixed", "mixed"

    def _guess_latin_language(self, text: str) -> str:
        """
        Heuristic guess for Latin-script language.
        Checks for common Afan Oromo or Tigrinya words;
        defaults to English otherwise.
        """
        lower = text.lower()

        # Oromo marker words (common in formal and informal Oromo text)
        oromo_markers = {
            "fi", "yoo", "kan", "hin", "ta'u", "dha", "miti",
            "jira", "jirra", "namni", "lubbu", "du'uu",
            "gammachuu", "dhukkubsate", "biyya", "oromoo",
        }
        # Tigrinya Romanized markers
        tigrinya_markers = {
            "eti", "nay", "zelo", "koynu", "hagerey", "tigrinya",
            "tesfay", "habesha",
        }

        words = set(re.findall(r"\b\w+\b", lower))

        if words & oromo_markers:
            return "om"
        if words & tigrinya_markers:
            return "ti"
        return "en"

    # ──────────────────────────────────────────────────────────────────────────
    #  Cleaning Strategies
    # ──────────────────────────────────────────────────────────────────────────

    def _clean_geez(self, text: str) -> str:
        """
        Clean Ge'ez (Amharic / Tigrinya) text.
        Preserves all Ethiopic characters while removing noise.
        """
        # Apply Amharic normalization (homophones + stopwords)
        text = self.amharic_normalizer.normalize(text, remove_stopwords=True)

        # Remove any remaining pure-ASCII noise tokens (URLs, emoji text, etc.)
        # while preserving Ethiopic characters and words
        text = re.sub(
            r"[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\s]",
            " ",
            text,
        )

        # Collapse repeated whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _clean_latin(self, text: str) -> str:
        """
        Clean Latin-script text (English or Fidelfon/Romanized Ethiopian).
        """
        # Lowercase
        text = text.lower()

        # Remove numbers
        text = re.sub(r"\d+", " ", text)

        # Remove punctuation
        text = re.sub(r"[^a-z\s]", " ", text)

        # Normalize repeated characters ("noooo" → "no")
        text = REPEAT_PATTERN.sub(r"\1", text)

        # Remove noise tokens
        tokens = text.split()
        tokens = [
            t for t in tokens
            if t not in LATIN_NOISE_TOKENS and len(t) > 1
        ]
        text = " ".join(tokens)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _clean_mixed(self, text: str) -> str:
        """
        Clean mixed-script text (code-switching, Fidelfon, etc.).
        Applies Ge'ez normalization to Ethiopic words and
        Latin cleaning to English/Romanized words.
        """
        # Split into Ge'ez tokens and Latin tokens
        tokens = re.split(r"(\s+)", text)  # preserve spacing structure
        cleaned_tokens = []

        for token in tokens:
            if not token.strip():
                cleaned_tokens.append(token)
                continue

            geez_chars = len(GEEZ_PATTERN.findall(token))
            latin_chars = len(LATIN_PATTERN.findall(token))

            if geez_chars > latin_chars:
                # Clean as Ge'ez
                cleaned = self.amharic_normalizer.normalize(
                    token, remove_stopwords=True
                )
            else:
                # Clean as Latin
                cleaned = self._clean_latin(token)

            if cleaned:
                cleaned_tokens.append(cleaned)

        text = " ".join(t for t in cleaned_tokens if t.strip())
        return re.sub(r"\s+", " ", text).strip()


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    cleaner = MultilingualCleaner()

    test_cases = [
        # Amharic crisis phrases
        "ህይወት መረረኝ ብቻ ሆኜ ሰው አጣሁ",
        "ብሞት ይሻለኛል ዋጋ የለኝም",
        # Amharic safe
        "ዛሬ ጠዋት ቡና ጠጣሁ ቤተሰቦቼ ደህና ናቸው",
        # Afan Oromo crisis (Romanized)
        "lubbu hin barbaadu du'uun wayya",
        # English crisis
        "I want to die nobody cares about me I am worthless",
        # Mixed code-switch (Fidelfon + English)
        "life is too hard ህይወቴ I can't anymore",
        # Noise to strip
        "check this out http://t.co/abc123 @user #suicide follow me",
    ]

    print("=== MultilingualCleaner Tests ===\n")
    for text in test_cases:
        result = cleaner.clean(text)
        print(f"Input:  {text}")
        print(f"Output: {result['text']}")
        print(f"Lang:   {result['lang']} | Script: {result['script']}\n")
