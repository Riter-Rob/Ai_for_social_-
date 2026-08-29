"""
Ethiopia/preprocessing/amharic_normalizer.py

Amharic Ge'ez script normalizer.
Handles:
  1. Homophone character merging (fidel equivalents with same sound)
  2. Common Amharic / Tigrinya stopword removal
  3. Whitespace normalization

Usage:
    normalizer = AmharicNormalizer()
    clean = normalizer.normalize("ሀዋሳ ሐዋሳ ኀዋሳ")  # → "ሀዋሳ ሀዋሳ ሀዋሳ"
"""

import re
from typing import Optional


class AmharicNormalizer:
    """
    Normalizes Ethiopic (Ge'ez) script text for NLP tasks.

    Ge'ez has several sets of characters that are phonetically identical
    in modern Amharic and Tigrinya but written differently.
    Normalizing them reduces vocabulary size and improves model performance.
    """

    # ─── Homophone Normalization Map ──────────────────────────────────────────
    # Each tuple: (characters to replace, canonical replacement)
    # Source: Amharic linguistics & SERA standard
    HOMOPHONE_GROUPS = [
        # ሀ group (ha) — 4 variants, all same sound
        ("ሐሑሒሓሔሕ", "ሀሁሂሃሄህ"),
        ("ኀኁኂኃኄኅ", "ሀሁሂሃሄህ"),
        ("ኃ", "ሃ"),

        # ሰ group (se) — ሠ is archaic
        ("ሠሡሢሣሤሥሦ", "ሰሱሲሳሴስሶ"),

        # አ group (a) — ዐ is pharyngeal in classical but merged in modern Amharic
        ("ዐዑዒዓዔዕዖ", "አኡኢኣኤእኦ"),

        # ጸ group (tse) — ፀ is archaic
        ("ፀፁፂፃፄፅፆ", "ጸጹጺጻጼጽጾ"),
    ]

    # ─── Stopwords ────────────────────────────────────────────────────────────
    # Common Amharic functional words that carry no distress signal
    AMHARIC_STOPWORDS = {
        # Copula / auxiliary
        "ነው", "ናት", "ናቸው", "ነበር", "ነበረ", "ነበሩ", "ይሆናል", "ሆነ", "ሆኑ",
        # Conjunctions
        "እና", "ና", "ወይ", "ወይም", "ግን", "ስለዚህ", "ደግሞ", "አለዚያ",
        # Prepositions / postpositions
        "በ", "ለ", "ወደ", "ከ", "እስከ", "በላይ", "በታች", "ላይ", "ስር",
        # Articles / demonstratives
        "ይህ", "ይህን", "ይህንን", "ያ", "ያን", "እነዚህ", "እነዚያ",
        # Subject/object markers
        "ን", "ም", "ኮ", "ሳ", "ማ",
        # Question words (harmless)
        "ማን", "ምን", "የት", "መቼ", "እንዴት", "ስንት",
        # Common pronouns
        "እኔ", "አንተ", "አንቺ", "እሱ", "እሷ", "እኛ", "እናንተ", "እነሱ",
        # Filler / discourse
        "አሁን", "ደህና", "እዚህ", "እዚያ", "ዛሬ", "ትናንት", "ነገ",
    }

    # Common Afan Oromo stopwords
    OROMO_STOPWORDS = {
        "fi", "yoo", "kan", "hin", "ta'u", "ta'ee", "dha", "si", "na",
        "nu", "isaan", "isa", "ishee", "ani", "ati", "inni", "isheetu",
        "garuu", "akkasumas", "erga", "yommuu",
    }

    # Common Tigrinya stopwords
    TIGRINYA_STOPWORDS = {
        "ን", "ም", "ካ", "ዩ", "ኤ", "ኣ", "ብ", "ናብ", "ካብ", "ንዓ",
        "ምስ", "ዝ", "እዩ", "እያ", "እዮም", "ዘሎ", "ዘላ", "ዘለዉ",
    }

    # ─── Punctuation & noise ──────────────────────────────────────────────────
    # Ethiopic word separators and punctuation marks (U+1361–U+1368)
    ETHIOPIC_PUNCT = re.compile(r"[\u1361-\u1368።፤፥፦፧፨]")

    def __init__(self):
        # Build character-level translation tables for each homophone group
        self._translation_table = self._build_translation_table()
        self.all_stopwords = (
            self.AMHARIC_STOPWORDS
            | self.OROMO_STOPWORDS
            | self.TIGRINYA_STOPWORDS
        )

    def _build_translation_table(self) -> dict:
        """Build a unicode translation table from homophone groups."""
        table = {}
        for source_chars, target_chars in self.HOMOPHONE_GROUPS:
            for src, tgt in zip(source_chars, target_chars):
                table[ord(src)] = tgt
        return table

    def normalize(self, text: str, remove_stopwords: bool = True) -> str:
        """
        Normalize Ethiopic text.

        Args:
            text: Input text (may contain Ge'ez, Latin, or mixed characters).
            remove_stopwords: Whether to remove Ethiopian stopwords.

        Returns:
            Normalized text string.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Apply homophone normalization
        text = text.translate(self._translation_table)

        # 2. Remove Ethiopic punctuation (keep words only)
        text = self.ETHIOPIC_PUNCT.sub(" ", text)

        # 3. Remove common ASCII punctuation while preserving Ethiopic chars
        text = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]", " ", text)

        # 4. Collapse repeated whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # 5. Optionally remove stopwords
        if remove_stopwords:
            tokens = text.split()
            tokens = [t for t in tokens if t not in self.all_stopwords]
            text = " ".join(tokens)

        return text

    def get_amharic_stopwords(self) -> set:
        """Return the full set of Amharic stopwords."""
        return self.AMHARIC_STOPWORDS.copy()


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    normalizer = AmharicNormalizer()

    test_cases = [
        # Homophone normalization
        ("ሐዋሳ", "→ should become ሀዋሳ"),
        ("ዐይን", "→ should become አይን"),
        ("ሠላም", "→ should become ሰላም"),
        # Distress phrases to verify preservation
        ("ህይወት መረረኝ ዋጋ የለኝም", "→ should preserve crisis words"),
        ("ብሞት ይሻለኛል ሰው አጣሁ", "→ should preserve death reference"),
        # Stopword removal
        ("እኔ ደህና ነኝ ዛሬ", "→ stopwords removed"),
    ]

    print("=== AmharicNormalizer Tests ===\n")
    for text, note in test_cases:
        result = normalizer.normalize(text)
        print(f"Input:  {text}")
        print(f"Output: {result}  {note}\n")
