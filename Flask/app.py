"""
Flask/app.py — Upgraded Flask server for Ethiopian Multilingual Crisis Detection.

Endpoints:
  GET  /                    → Web chat UI
  POST /predict             → Original English model (RF + LSTM, preserved)
  POST /predict_ethiopian   → New multilingual Ethiopian model

Original author: Soumyajit Behera
Ethiopian adaptation: Robel Tesfaye (https://github.com/Riter-Rob)
"""

import sys
import os
import json
import pickle

from flask import Flask, jsonify, render_template, request

# ─── Add project root to path so Ethiopia/ package is importable ──────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, ".."))

import preprocess_text
from crisis_response import get_crisis_response, format_for_web

# Ethiopian inference engine (lazy-loaded on first request)
_crisis_detector = None

def get_detector():
    """Lazy-load the Ethiopian crisis detector to keep startup fast."""
    global _crisis_detector
    if _crisis_detector is None:
        from Ethiopia.Model.inference import EthiopianCrisisDetector
        _crisis_detector = EthiopianCrisisDetector()
    return _crisis_detector


app = Flask(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  Routes
# ════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Original English-only endpoint.
    Uses the pretrained Random Forest + BiLSTM models.
    Preserved for backward compatibility.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import load_model

        tf_idf    = pickle.load(open("./Models/tfidf_tokenizer.pkl", "rb"))
        rf_model  = pickle.load(open("./Models/random_forest.pkl",   "rb"))
        tokenizer = pickle.load(open("./Models/tf_tokenizer.pkl",    "rb"))
        model     = load_model("./Models/lstm.h5", compile=False)

        text     = request.form["text"]
        new_text = preprocess_text.clean_text(text)

        # ML prediction (Random Forest + TF-IDF)
        vec      = tf_idf.transform([new_text])
        ml_pred  = int(rf_model.predict(vec)[0])

        # DL prediction (BiLSTM + GloVe)
        sequence        = tokenizer.texts_to_sequences([new_text])
        padded_sequence = pad_sequences(sequence, padding="post")
        dl_pred         = model.predict(padded_sequence)
        dl              = int(dl_pred[0][0] > 0.5)

        return jsonify({
            "status":   200,
            "dl_pred":  json.dumps(dl),
            "ml_pred":  json.dumps(ml_pred),
        })

    except Exception as e:
        return jsonify({"status": 422, "message": f"Error: {str(e)}"}), 422


@app.route("/predict_ethiopian", methods=["POST"])
def predict_ethiopian():
    """
    Ethiopian multilingual crisis detection endpoint.
    Supports Amharic, Afan Oromo, Tigrinya, English, and mixed text.

    POST body (form or JSON):
      text — the message to analyze

    Response:
      {
        "status": 200,
        "is_crisis": true/false,
        "risk_level": "HIGH" | "MEDIUM" | "SAFE",
        "confidence": 0.92,
        "lang": "am",
        "method": "model" | "heuristic",
        "crisis_response": { ... }   // only present if is_crisis=true
      }
    """
    try:
        # Accept both form-data and JSON body
        if request.is_json:
            text = request.json.get("text", "")
        else:
            text = request.form.get("text", "")

        if not text or not text.strip():
            return jsonify({"status": 400, "message": "No text provided"}), 400

        # Run multilingual inference
        detector = get_detector()
        result   = detector.predict(text)

        response_payload = {
            "status":     200,
            "is_crisis":  result["label"] == 1,
            "risk_level": result["risk_level"],
            "confidence": result["confidence"],
            "lang":       result["lang"],
            "method":     result["method"],
        }

        # Attach crisis response if needed
        if result["label"] == 1:
            crisis = get_crisis_response(
                lang=result["lang"],
                risk_level=result["risk_level"],
            )
            response_payload["crisis_response"] = format_for_web(crisis)

        return jsonify(response_payload)

    except Exception as e:
        return jsonify({"status": 500, "message": f"Server error: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "Ethiopian Crisis Detection API"})


# ════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
