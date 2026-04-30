# PakSentinel — Misinformation Detection Pipeline

An end-to-end NLP pipeline for misinformation detection across three classes: **Real**, **Fake**, and **Satire**.

---

## How to Run (Demo Walkthrough)

This section gives the exact commands to bring up every service from a clean clone. Run the steps in the order shown.

### 0. Prerequisites

- Python 3.11+
- Git
- Docker Desktop (only for the Docker path)
- ~3 GB free disk space (datasets + Word2Vec models)

### 1. Clone & enter the repo

```powershell
git clone https://github.com/jawadsaleem007/PakSentinel.git
cd PakSentinel
```

### 2. (Option A) Local Python environment

```powershell
# Create and activate virtual env
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# One-time NLP model downloads
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','averaged_perceptron_tagger','words','omw-1.4']]"
python -m spacy download en_core_web_sm
```

### 3. Run the full ML pipeline (Tasks 1–6)

```powershell
# End-to-end: data sourcing → cleaning → features → models → MLflow logging
python -m src.pipeline
```

Outputs:
- `data/processed/combined_dataset.csv` and `.parquet`
- `data/embeddings/word2vec_cbow.model`, `word2vec_skipgram.model`
- Trained model pickles in `data/processed/`
- MLflow runs under `mlruns/`

### 4. Start the FastAPI inference service (Task 7)

```powershell
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open the interactive Swagger UI:
- http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 5. Start the MLflow UI (for the demo screenshots)

In a **second terminal**:

```powershell
.\.venv\Scripts\Activate.ps1
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Open: http://localhost:5000

### 6. Run the API test suite (25 tests)

```powershell
python -m pytest tests/test_api.py -v
```

### 7. Quick API smoke tests

```powershell
# Health
curl http://localhost:8000/health

# Single classification
curl -X POST http://localhost:8000/classify `
  -H "Content-Type: application/json" `
  -d '{"text": "Breaking: Scientists discover cure for fake news using AI"}'

# Batch classification
curl -X POST http://localhost:8000/classify/batch `
  -H "Content-Type: application/json" `
  -d '{"texts": ["Pakistan wins cricket match", "Aliens land in Karachi", "PM announces new policy"]}'

# Similar claims retrieval
curl -X POST http://localhost:8000/retrieve/similar `
  -H "Content-Type: application/json" `
  -d '{"text": "election fraud allegations", "top_k": 5}'

# Live model performance
curl http://localhost:8000/model/performance
```

---

### 2. (Option B) Docker Compose — All services in one command

Brings up **MinIO + MLflow + FastAPI** together.

```powershell
# Build and start everything (detached)
docker compose up --build -d

# Watch logs
docker compose logs -f paksentinel-api

# Check container health
docker compose ps
```

Services exposed on the host:

| Service              | URL                        | Credentials              |
|----------------------|----------------------------|--------------------------|
| PakSentinel API      | http://localhost:8000/docs | —                        |
| MLflow UI            | http://localhost:5000      | —                        |
| MinIO Console        | http://localhost:9001      | `minioadmin/minioadmin`  |
| MinIO S3 endpoint    | http://localhost:9000      | `minioadmin/minioadmin`  |

Stop everything:

```powershell
docker compose down          # keep volumes
docker compose down -v       # also wipe MinIO + MLflow data
```

---

### Demo Runbook (5-minute order)

1. `docker compose up --build -d` (or local `uvicorn` + `mlflow ui`)
2. Open http://localhost:8000/docs → call `/health`, `/classify`, `/classify/batch`.
3. Open http://localhost:5000 → show experiment runs, parameters, metrics, parallel coordinates plot, and the Model Registry (3 registered models).
4. Open http://localhost:9001 → show MinIO buckets (`raw/`, `processed/`, `embeddings/`).
5. Run `python -m pytest tests/test_api.py -v` to show 25/25 passing.

---

## Project Structure

```
assignment2/
├── api/                          # Task 7: FastAPI Inference System
│   ├── app.py                    # 6 endpoints with lifespan context manager
│   ├── models.py                 # Pydantic request/response models
│   └── middleware.py             # Logging + rate limiting middleware
├── src/                          # Tasks 1-6: Pipeline modules
│   ├── pipeline.py               # Master orchestrator
│   ├── data_sourcing.py          # Task 1: Multi-source dataset construction
│   ├── data_lake_manager.py      # Task 2: MinIO/local data lake
│   ├── cleaning.py               # Task 3.1: Text cleaning
│   ├── tokenization.py           # Task 3.2: Tokenizer comparison
│   ├── stopwords.py              # Task 3.3: Custom stopword list
│   ├── normalization.py          # Task 3.4: Stemming vs lemmatization
│   ├── features.py               # Task 3.5: BoW, TF-IDF, Word2Vec
│   ├── ngram_models.py           # Task 4: N-gram language models
│   ├── naive_bayes.py            # Task 5.1: NB from scratch
│   ├── logistic_regression.py    # Task 5.2: LR with L1/L2/ElasticNet
│   ├── polynomial_lr.py          # Task 5.3: Polynomial features + LR
│   └── mlflow_tracking.py        # Task 6: MLFlow experiment tracking
├── tests/
│   └── test_api.py               # 25 tests (all passing)
├── reports/
│   ├── PakSentinel_Technical_Report.md
│   ├── Data_Sourcing_Declaration.md
│   ├── MLFlow_Hierarchy_Diagram.md
│   ├── System_Architecture_Diagram.md
│   └── figures/                  # All generated plots
├── data/
│   ├── raw/                      # Original dataset files
│   ├── processed/                # Cleaned Parquet + model artifacts
│   └── embeddings/               # Word2Vec models
├── mlruns/                       # MLFlow experiment data
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model name, version, stage, F1, load timestamp |
| `/preprocess` | POST | Text preprocessing with configurable steps |
| `/classify` | POST | Single text classification (100 req/min) |
| `/classify/batch` | POST | Batch classification ≤100 texts (10 req/min) |
| `/retrieve/similar` | POST | Top-k similar fact-checked claims |
| `/model/performance` | GET | Live metrics from MLFlow |

## Results Summary

| Model | F1 (weighted) |
|-------|---------------|
| Naive Bayes (BoW) | 0.6669 |
| Logistic Regression L2 (TF-IDF) | 0.6681 |
| TF-IDF + Word2Vec (combined) | 0.6926 |

## Dataset

- **9,904 samples** from LIAR, FakeNewsNet, and Sarcasm Headlines
- **3 classes:** Real (35.1%), Fake (34.6%), Satire (30.2%)
- **0% duplicate rate** after deduplication
