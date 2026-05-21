# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps (tokenizer binaries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (saves ~2GB vs full CUDA version)
RUN pip install --no-cache-dir torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model into the image so cold-start is instant
RUN python - <<'EOF2'
from transformers import pipeline
pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None
)
EOF2

# Copy application code
COPY app.py .
COPY templates/ templates/

# Hugging Face model cache
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 10000

CMD ["python", "app.py"]