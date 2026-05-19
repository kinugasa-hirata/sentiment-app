# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps (tokenizer binaries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model into the image so cold-start is instant
RUN python - <<'EOF'
from transformers import pipeline
pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    return_all_scores=True
)
EOF

# Copy application code
COPY app.py .
COPY templates/ templates/

# Hugging Face model cache lives here — mount a volume in prod to persist it
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 5000

# Use gunicorn for production (installed as dep of flask in this setup)
CMD ["python", "app.py"]
