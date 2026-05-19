# SentimentLens 🔍

A sentiment analysis web app powered by **HuggingFace Transformers** (RoBERTa model).  
Returns three fine-grained scores: **Positive / Neutral / Negative**.

---

## Project Structure

```
sentiment-app/
├── app.py               ← Flask backend + HuggingFace model
├── templates/
│   └── index.html       ← Browser UI (textarea → analyse → score bars)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Run Locally (Python)

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server (model downloads on first run ~500MB)
python app.py

# 4. Open browser
open http://localhost:5000
```

---

## Run with Docker (Recommended)

```bash
# Build & start (model is baked into the image)
docker compose up --build

# Open browser
open http://localhost:5000
```

The `hf-cache` Docker volume persists the model between restarts.

---

## Deploy to the Cloud

### Option A — Railway (easiest, free tier)
```bash
# Install Railway CLI
npm install -g @railway/cli

railway login
railway init
railway up          # auto-detects Dockerfile
```
Your app gets a public HTTPS URL instantly.

### Option B — Fly.io (generous free tier)
```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

fly launch          # follows the Dockerfile
fly deploy
```

### Option C — Google Cloud Run (scales to zero)
```bash
# Build & push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/sentiment-app

# Deploy
gcloud run deploy sentiment-app \
  --image gcr.io/YOUR_PROJECT/sentiment-app \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi          # model needs ~1.5GB RAM
```

---

## Model Info

| Field | Value |
|---|---|
| Model | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Labels | Negative · Neutral · Positive |
| Max input | 512 tokens |
| Size | ~500 MB |

---

## API Endpoint

`POST /analyze`  
```json
// Request
{ "text": "This product is amazing!" }

// Response
{
  "top": "Positive",
  "scores": {
    "Positive": 97.43,
    "Neutral":   2.01,
    "Negative":  0.56
  },
  "text": "This product is amazing!"
}
```
