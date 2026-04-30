# PakSentinel — System Architecture Diagram

**Submitted before Task 7 implementation as required by rubric.**

---

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐    │
│   │  Web Browser  │  │  curl / httpx│  │  Python Client (pytest)│   │
│   └──────┬───────┘  └──────┬───────┘  └───────────┬───────────┘    │
│          │                 │                       │                 │
└──────────┼─────────────────┼───────────────────────┼─────────────────┘
           │                 │                       │
           └─────────────────┼───────────────────────┘
                             │  HTTP (port 8000)
┌────────────────────────────▼────────────────────────────────────────┐
│                    MIDDLEWARE LAYER                                   │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  RequestLoggingMiddleware                                 │      │
│   │  • Logs method, path, client IP, query params            │      │
│   │  • Logs response status code and processing time         │      │
│   │  • Console output + rotating file (10MB, 5 backups)      │      │
│   │  • Adds X-Processing-Time-Ms response header             │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  Rate Limiter (slowapi)                                   │      │
│   │  • /classify:       100 requests/minute                   │      │
│   │  • /classify/batch:  10 requests/minute                   │      │
│   │  • Key function: client IP address                        │      │
│   └──────────────────────────────────────────────────────────┘      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    APPLICATION LAYER (FastAPI)                        │
│                                                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐     │
│   │ GET /health   │  │POST /preproc │  │ POST /classify       │     │
│   │               │  │              │  │ (rate limited)       │     │
│   │ Model info,   │  │ Text →       │  │ Text → prediction,  │     │
│   │ version, F1,  │  │ tokens,      │  │ confidence, probs,  │     │
│   │ load time     │  │ stopwords,   │  │ top features        │     │
│   │               │  │ proc time    │  │                     │     │
│   └──────────────┘  └──────────────┘  └──────────────────────┘     │
│                                                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐     │
│   │POST /classify│  │POST /retrieve│  │ GET /model/          │     │
│   │     /batch   │  │    /similar  │  │    performance       │     │
│   │ (rate limited│  │              │  │                      │     │
│   │  ≤100 texts, │  │ top-k cosine │  │ Live metrics from    │     │
│   │  <500ms)     │  │ similarity   │  │ MLFlow registry      │     │
│   └──────────────┘  └──────────────┘  └──────────────────────┘     │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  Pydantic Validation                                      │      │
│   │  • text: 10–10,000 characters                             │      │
│   │  • top_k: 1–20                                            │      │
│   │  • batch texts: 1–100, each 10–10,000 chars               │      │
│   └──────────────────────────────────────────────────────────┘      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                 NLP PROCESSING LAYER                                 │
│                                                                      │
│   ┌────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────┐       │
│   │ Clean  │→│ Tokenize  │→│ Stopwords │→│ Lemmatize   │       │
│   │ (HTML, │  │ (NLTK     │  │ (Custom   │  │ (WordNet    │       │
│   │  URLs) │  │  word_tok) │  │  199 words)│  │  POS-aware) │       │
│   └────────┘  └───────────┘  └───────────┘  └─────────────┘       │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  TF-IDF Vectorizer (Sublinear TF, 10K features)          │      │
│   └──────────────────────────────────────────────────────────┘      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    MODEL LAYER                                       │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  Logistic Regression (L2, C=1.0)                          │      │
│   │  • Loaded once at startup via lifespan context manager    │      │
│   │  • 3-class output: Real, Fake, Satire                     │      │
│   │  • Outputs calibrated probabilities                       │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  Cosine Similarity Engine                                  │      │
│   │  • Pre-computed TF-IDF matrix (9891 × 10000)              │      │
│   │  • Returns top-k similar fact-checked claims               │      │
│   └──────────────────────────────────────────────────────────┘      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    STORAGE LAYER                                     │
│                                                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐     │
│   │ MinIO / Local │  │ MLFlow       │  │ Rotating Logs        │     │
│   │ DataLake      │  │ (mlruns/)    │  │ (logs/)              │     │
│   │               │  │              │  │                      │     │
│   │ ├── Raw       │  │ Experiments, │  │ api_requests.log     │     │
│   │ ├── Processed │  │ Runs,        │  │ (10MB × 5 backups)   │     │
│   │ └── Embeddings│  │ Registry     │  │                      │     │
│   └──────────────┘  └──────────────┘  └──────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Docker Deployment Architecture

```
docker-compose.yml
│
├── minio (port 9000/9001)
│   └── S3-compatible object storage for data lake
│
├── mlflow (port 5000)
│   └── Experiment tracking UI and model registry
│
└── paksentinel-api (port 8000)
    ├── Depends on: minio (service_healthy)
    ├── Mounts: ./data, ./mlruns, ./logs
    └── Command: uvicorn api.app:app --host 0.0.0.0 --port 8000
```

---

## Data Flow

```
Raw Data (LIAR, FakeNewsNet, Sarcasm)
    │
    ▼
DataLakeManager.upload_raw()     ──→  Raw Layer (with metadata sidecars)
    │
    ▼
Cleaning → Tokenization → Stopwords → Lemmatization
    │
    ▼
DataLakeManager.upload_processed() ──→  Processed Layer (Parquet + PKL)
    │
    ▼
Feature Extraction (TF-IDF, Word2Vec)
    │
    ▼
DataLakeManager.upload_embeddings() ──→  Embeddings Layer (versioned .model)
    │
    ▼
Model Training (NB, LR, Poly) ──→  MLFlow Tracking + Registry
    │
    ▼
Best Model (L2 LR) ──→  FastAPI Deployment Artifacts
    │
    ▼
API Serving (/classify, /retrieve/similar, etc.)
```
