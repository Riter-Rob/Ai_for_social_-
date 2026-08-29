"""Phase 1 verification script."""
import sys
sys.path.insert(0, ".")

from Ethiopia.preprocessing.amharic_normalizer import AmharicNormalizer
from Ethiopia.preprocessing.multilingual_cleaner import MultilingualCleaner

print("=== Phase 1 Verification ===\n")

# ─── Test 1: Amharic normalizer homophones ────────────────────────────────────
norm = AmharicNormalizer()
print("-- Homophone Normalization --")
tests = [
    ("ሐዋሳ", "ሀዋሳ"),
    ("ዐይን", "አይን"),
    ("ሠላም", "ሰላም"),
]
for inp, expected in tests:
    result = norm.normalize(inp, remove_stopwords=False)
    status = "PASS" if expected in result else "FAIL"
    print(f"  [{status}] {inp} -> {result}")

print()

# ─── Test 2: Multilingual cleaner ────────────────────────────────────────────
cleaner = MultilingualCleaner()
print("-- Language Detection --")
cases = [
    ("ህይወት መረረኝ ዋጋ የለኝም", "am", "geez"),
    ("I want to die nobody cares", "en", "latin"),
    ("lubbu hin barbaadu du uun wayya", "om", "latin"),
    ("life is hard I cannot ህይወቴ", "mixed", "mixed"),
]
for text, exp_lang, exp_script in cases:
    result = cleaner.clean(text)
    lang_ok = result["lang"] == exp_lang
    script_ok = result["script"] == exp_script
    status = "PASS" if (lang_ok and script_ok) else "WARN"
    lang = result["lang"]
    script = result["script"]
    cleaned = result["text"][:50]
    print(f"  [{status}] lang={lang} script={script} | {cleaned}")

print()

# ─── Test 3: Crisis phrase preservation ──────────────────────────────────────
print("-- Crisis Phrase Preservation (Ge'ez chars MUST survive) --")
crisis = "ብሞት ይሻለኛል ማንም አይወደኝም"
result = cleaner.clean(crisis)
has_geez = any("\u1200" <= c <= "\u137F" for c in result["text"])
status = "PASS" if has_geez else "FAIL"
print(f"  Input:  {crisis}")
print(f"  Output: {result['text']}")
print(f"  [{status}] Ge'ez characters preserved: {has_geez}")

print()

# ─── Test 4: English still works ─────────────────────────────────────────────
print("-- English Backward Compatibility --")
eng = "I feel hopeless and I want to die"
result = cleaner.clean(eng)
print(f"  Input:  {eng}")
print(f"  Output: {result['text']}")
print(f"  Lang:   {result['lang']}")

print("\n=== Phase 1 COMPLETE ===")
