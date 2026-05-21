from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import torch

app = Flask(__name__)

print("Loading sentiment model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    top_k=None,
    device=-1  # always CPU — no GPU on Render free tier
)
print("Model ready.")

LABEL_MAP = {
    "NEGATIVE": "Negative",
    "POSITIVE": "Positive",
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if len(text) > 512:
        return jsonify({"error": "Text too long (max 512 characters)"}), 400

    try:
        raw = sentiment_pipeline(text)[0]
        scores = {
            LABEL_MAP.get(item["label"], item["label"]): round(item["score"] * 100, 2)
            for item in raw
        }
        top = max(scores, key=scores.get)
        return jsonify({"scores": scores, "top": top, "text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)