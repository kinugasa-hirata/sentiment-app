from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import torch

app = Flask(__name__)

# Load model once at startup
# cardiffnlp/twitter-roberta-base-sentiment-latest gives 3 labels:
# Negative / Neutral / Positive with fine-grained scores
print("Loading sentiment model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    return_all_scores=True,          # get ALL label scores, not just top-1
    device=0 if torch.cuda.is_available() else -1
)
print("Model ready.")


LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
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
        raw = sentiment_pipeline(text)[0]   # list of {label, score} dicts
        scores = {
            LABEL_MAP.get(item["label"], item["label"]): round(item["score"] * 100, 2)
            for item in raw
        }
        top = max(scores, key=scores.get)
        return jsonify({"scores": scores, "top": top, "text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
