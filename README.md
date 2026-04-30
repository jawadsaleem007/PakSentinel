# PakSentinel — Misinformation Detection Pipeline

An end-to-end NLP pipeline for misinformation detection across three classes: **Real**, **Fake**, and **Satire**.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('words'); nltk.download('omw-1.4')"

# Download SpaCy model
python -m spacy download en_core_web_sm

# Run full pipeline (Tasks 1-6)
python -m src.pipeline

# Start API server (Task 7)
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# Run tests
python -m pytest tests/test_api.py -v
```

## Docker Deployment

```bash
docker-compose up --build
```

Services:
- **PakSentinel API:** http://localhost:8000
- **MLFlow UI:** http://localhost:5000
- **MinIO Console:** http://localhost:9001 (admin/minioadmin)

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
