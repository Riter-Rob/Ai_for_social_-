"""
Ethiopia/Model/inference.py

Local inference engine for Ethiopian crisis detection.

Loads the fine-tuned afro-xlmr-base model and classifies new text.
Falls back to keyword heuristics if the model is not yet downloaded.

Usage:
    from Ethiopia.Model.inference import EthiopianCrisisDetector

    detector = EthiopianCrisisDetector()
    result = detector.predict("ህይወት መረረኝ ብቻ ሆኜ")
    # → {"label": 1, "confidence": 0.92, "risk_level": "HIGH", "lang": "am"}
"""

import os
import re
import json
from typing import Optional

# ─── Model path ───────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ethiopian_crisis_model")

# ─── Thresholds ───────────────────────────────────────────────────────────────
HIGH_RISK_THRESHOLD   = 0.75   # Definitely crisis
MEDIUM_RISK_THRESHOLD = 0.50   # Uncertain — flag for review


# ═════════════════════════════════════════════════════════════════════════════
#  Keyword fallback (used when model is not yet downloaded)
# ═════════════════════════════════════════════════════════════════════════════
CRISIS_KEYWORDS = {
    # Amharic
    "ብሞት", "ሞቼ", "ሞት", "ህይወቴ", "ሰው አጣሁ", "ዋጋ የለኝም",
    "ብቻዬን", "ስቃዬ", "ተስፋ", "ጠፍቻለሁ", "ምንም", "ለምን", "ፋይዳ",
    # Afan Oromo
    "du'uun", "lubbu", "hin barbaadu", "abdii kutadhe", "kophaa",
    # Tigrinya
    "ሞት ይሓይሽ", "ዋጋ የብለይን", "ካብ ዓለም ክጠፍእ",
    # English
    "want to die", "kill myself", "end my life", "worthless",
    "no reason to live", "disappear", "better off dead",
    "suicide", "self harm", "hurt myself",
}

SAFE_KEYWORDS = {
    # Amharic
    "ደስ", "ደህና", "ቡና", "ቤተሰብ", "ጓደኞ",
    # English
    "happy", "grateful", "great day", "feeling good", "proud",
}


class EthiopianCrisisDetector:
    """
    Detects crisis/suicidal ideation in Ethiopian multilingual text.

    Automatically loads the fine-tuned transformer model if available,
    or uses keyword heuristics as a fallback.
    """

    def __init__(self, model_dir: str = MODEL_DIR, threshold: float = HIGH_RISK_THRESHOLD):
        self.threshold = threshold
        self._model = None
        self._tokenizer = None
        self._device = None
        self._use_model = False

        self._try_load_model(model_dir)

        if not self._use_model:
            print("[CrisisDetector] No fine-tuned model found at:", model_dir)
            print("[CrisisDetector] Using keyword heuristic fallback.")
            print("[CrisisDetector] Run Ethiopia/Model/fine_tune_colab.py on Colab to get the model.")

    def _try_load_model(self, model_dir: str) -> None:
        """Attempt to load the fine-tuned HuggingFace model."""
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.isfile(config_path):
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self._model = self._model.to(self._device)
            self._model.eval()
            self._use_model = True

            # Load metadata if available
            meta_path = os.path.join(model_dir, "model_metadata.json")
            if os.path.isfile(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                print(f"[CrisisDetector] Loaded model: {meta.get('base_model', 'unknown')}")
            else:
                print("[CrisisDetector] Fine-tuned model loaded.")

        except Exception as e:
            print(f"[CrisisDetector] Could not load model: {e}")
            self._use_model = False

    def predict(self, text: str) -> dict:
        """
        Classify text for crisis/distress.

        Args:
            text: Input text in any supported language.

        Returns:
            dict with:
              "label"      — 0 (safe) or 1 (crisis)
              "confidence" — probability of crisis class (0.0 – 1.0)
              "risk_level" — "HIGH", "MEDIUM", or "SAFE"
              "lang"       — detected language code
              "method"     — "model" or "heuristic"
        """
        if not text or not isinstance(text, str) or not text.strip():
            return self._build_result(0, 0.0, "en", "heuristic")

        # Detect language
        lang = self._detect_lang(text)

        if self._use_model:
            return self._predict_with_model(text, lang)
        else:
            return self._predict_with_heuristic(text, lang)

    def _predict_with_model(self, text: str, lang: str) -> dict:
        """Run inference using the fine-tuned transformer model."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        crisis_prob = probs[0][1].item()
        label = 1 if crisis_prob >= self.threshold else 0

        return self._build_result(label, crisis_prob, lang, "model")

    def _predict_with_heuristic(self, text: str, lang: str) -> dict:
        """Keyword-based fallback classifier."""
        text_lower = text.lower()

        crisis_hits = sum(1 for kw in CRISIS_KEYWORDS if kw.lower() in text_lower)
        safe_hits   = sum(1 for kw in SAFE_KEYWORDS   if kw.lower() in text_lower)

        if crisis_hits > 0 and crisis_hits >= safe_hits:
            confidence = min(0.65 + (crisis_hits * 0.05), 0.90)
            label = 1
        else:
            confidence = max(0.10, 0.35 - (crisis_hits * 0.05))
            label = 0

        return self._build_result(label, confidence, lang, "heuristic")

    def _build_result(self, label: int, confidence: float, lang: str, method: str) -> dict:
        """Build the standard result dict."""
        if label == 1:
            if confidence >= HIGH_RISK_THRESHOLD:
                risk = "HIGH"
            else:
                risk = "MEDIUM"
        else:
            risk = "SAFE"

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "risk_level": risk,
            "lang":       lang,
            "method":     method,
        }

    def _detect_lang(self, text: str) -> str:
        """Quick script-based language detection."""
        geez_chars = len(re.findall(r"[\u1200-\u137F]", text))
        latin_chars = len(re.findall(r"[a-zA-Z]", text))
        total = geez_chars + latin_chars
        if total == 0:
            return "unknown"
        ratio = geez_chars / total
        if ratio > 0.75:
            return "am"
        elif ratio < 0.25:
            return "en"
        return "mixed"


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = EthiopianCrisisDetector()
    print()

    cases = [
        ("ህይወት መረረኝ ብቻ ሆኜ ዋጋ የለኝም",          "CRISIS (Amharic)"),
        ("ብሞት ይሻለኛል ሰው አጣሁ",                  "CRISIS (Amharic)"),
        ("du uun wayya lubbu hin barbaadu",       "CRISIS (Oromo)"),
        ("I want to die nobody cares about me",   "CRISIS (English)"),
        ("ዛሬ ቡና ጠጣሁ ቤተሰቦቼ ደህና ናቸው",           "SAFE   (Amharic)"),
        ("Today was a great day I feel happy",    "SAFE   (English)"),
    ]

    print("=== Inference Test ===\n")
    for text, note in cases:
        result = detector.predict(text)
        risk = result["risk_level"]
        conf = result["confidence"]
        method = result["method"]
        print(f"  [{risk:6}] conf={conf:.2f} [{method:10}] {note}")
        print(f"           Text: {text[:55]}")
        print()
