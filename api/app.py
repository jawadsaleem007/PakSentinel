"""
Task 7 — FastAPI Inference System [30 Marks]

app.py — Main FastAPI application with 6 endpoints:
1. GET  /health            — Model info and health check
2. POST /preprocess        — Text preprocessing pipeline
3. POST /classify          — Single text classification
4. POST /classify/batch    — Batch classification (≤100 texts, <500ms)
5. POST /retrieve/similar  — Similar fact-checked claims retrieval
6. GET  /model/performance — Live metrics from MLFlow
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.models import (
    PreprocessRequest, PreprocessResponse,
    ClassifyRequest, ClassifyResponse,
    BatchClassifyRequest, BatchClassifyResponse,
    SimilarRequest, SimilarResponse, SimilarClaim,
    HealthResponse, PerformanceMetrics,
)
from api.middleware import RequestLoggingMiddleware, limiter

# ──────────────────────────────────────────────────────────
#  Global State (loaded once at startup)
# ──────────────────────────────────────────────────────────
app_state = {
    'model': None,
    'vectorizer': None,
    'label_encoder': None,
    'dataset': None,
    'tfidf_matrix': None,
    'load_timestamp': None,
    'model_name': 'PakSentinel_LR',
    'model_version': '1.0',
    'model_stage': 'Production',
    'f1_score': 0.0,
    'metrics': {},
}


# ──────────────────────────────────────────────────────────
#  Lifespan Context Manager (model loaded once at startup)
# ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and resources at startup, clean up on shutdown."""
    print("[STARTUP] Loading PakSentinel model and resources...")
    
    models_dir = PROJECT_ROOT / "data" / "processed"
    
    try:
        # Load TF-IDF vectorizer
        vectorizer_path = models_dir / "v1.0" / "tfidf_vectorizer.pkl"
        if vectorizer_path.exists():
            with open(vectorizer_path, 'rb') as f:
                app_state['vectorizer'] = pickle.load(f)
            print(f"  Loaded vectorizer from {vectorizer_path}")
        else:
            # Fallback: create a basic vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            app_state['vectorizer'] = TfidfVectorizer(max_features=10000, sublinear_tf=True)
            print("  [WARN] No saved vectorizer found, using default")
        
        # Load model
        model_path = models_dir / "v1.0" / "best_model.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                app_state['model'] = pickle.load(f)
            print(f"  Loaded model from {model_path}")
        else:
            from sklearn.linear_model import LogisticRegression
            app_state['model'] = LogisticRegression(max_iter=1000, random_state=42)
            print("  [WARN] No saved model found, using default")
        
        # Load label encoder
        le_path = models_dir / "v1.0" / "label_encoder.pkl"
        if le_path.exists():
            with open(le_path, 'rb') as f:
                app_state['label_encoder'] = pickle.load(f)
            print(f"  Loaded label encoder from {le_path}")
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.classes_ = np.array(['Fake', 'Real', 'Satire'])
            app_state['label_encoder'] = le
            print("  [WARN] No saved label encoder found, using default")
        
        # Load dataset for similarity retrieval
        dataset_path = models_dir / "combined_dataset.parquet"
        if dataset_path.exists():
            app_state['dataset'] = pd.read_parquet(dataset_path)
            print(f"  Loaded dataset ({len(app_state['dataset'])} samples)")
        
        # Load TF-IDF matrix for similarity search
        tfidf_path = models_dir / "v1.0" / "tfidf_matrix.pkl"
        if tfidf_path.exists():
            with open(tfidf_path, 'rb') as f:
                app_state['tfidf_matrix'] = pickle.load(f)
            print(f"  Loaded TF-IDF matrix")
        
        # Load metrics
        metrics_path = models_dir / "v1.0" / "metrics.pkl"
        if metrics_path.exists():
            with open(metrics_path, 'rb') as f:
                app_state['metrics'] = pickle.load(f)
                app_state['f1_score'] = app_state['metrics'].get('f1_weighted', 0.0)
        
        app_state['load_timestamp'] = datetime.now().isoformat()
        print(f"[STARTUP] Complete! Model loaded at {app_state['load_timestamp']}")
        
    except Exception as e:
        print(f"[STARTUP ERROR] {e}")
        app_state['load_timestamp'] = datetime.now().isoformat()
    
    yield  # App is running
    
    # Cleanup on shutdown
    print("[SHUTDOWN] Cleaning up resources...")
    app_state.clear()


# ──────────────────────────────────────────────────────────
#  FastAPI App
# ──────────────────────────────────────────────────────────
app = FastAPI(
    title="PakSentinel API",
    description="End-to-end NLP pipeline for Pakistani misinformation detection",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(RequestLoggingMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ──────────────────────────────────────────────────────────
#  Helper Functions
# ──────────────────────────────────────────────────────────
def preprocess_text(text: str, steps: list = None) -> dict:
    """Run preprocessing pipeline on text."""
    from src.cleaning import clean_text
    from src.tokenization import tokenize_nltk
    from src.stopwords import remove_stopwords, CUSTOM_STOPWORDS
    from src.normalization import apply_lemmatizer
    
    result = {
        'original_text': text,
        'tokens': [],
        'removed_stopwords': [],
        'steps_applied': [],
    }
    
    current_text = text
    
    if steps is None or 'clean' in steps:
        current_text = clean_text(current_text)
        result['steps_applied'].append('clean')
    
    if steps is None or 'tokenize' in steps:
        tokens = tokenize_nltk(current_text)
        result['tokens'] = tokens
        result['steps_applied'].append('tokenize')
    else:
        tokens = current_text.split()
        result['tokens'] = tokens
    
    if steps is None or 'remove_stopwords' in steps:
        before = set(t.lower() for t in tokens)
        filtered = remove_stopwords(tokens, CUSTOM_STOPWORDS)
        after = set(t.lower() for t in filtered)
        result['removed_stopwords'] = list(before - after)
        result['tokens'] = filtered
        result['steps_applied'].append('remove_stopwords')
    
    if steps is None or 'normalize' in steps:
        result['tokens'] = apply_lemmatizer(result['tokens'])
        result['steps_applied'].append('normalize')
    
    return result


def classify_single(text: str) -> dict:
    """Classify a single text."""
    # Preprocess
    processed = preprocess_text(text)
    processed_text = ' '.join(processed['tokens'])
    
    # Vectorize
    vectorizer = app_state['vectorizer']
    model = app_state['model']
    le = app_state['label_encoder']
    
    if hasattr(vectorizer, 'transform'):
        X = vectorizer.transform([processed_text])
    else:
        # Vectorizer not fitted, return default
        return {
            'prediction': 'Unknown',
            'confidence': 0.0,
            'class_probabilities': {},
            'top_features': [],
        }
    
    # Predict
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[0]
        prediction_idx = np.argmax(probabilities)
        prediction = le.inverse_transform([prediction_idx])[0]
        confidence = float(probabilities[prediction_idx])
        
        class_probs = {
            le.inverse_transform([i])[0]: float(p) 
            for i, p in enumerate(probabilities)
        }
    else:
        prediction_idx = model.predict(X)[0]
        prediction = le.inverse_transform([prediction_idx])[0]
        confidence = 1.0
        class_probs = {prediction: 1.0}
    
    # Top contributing features
    top_features = []
    if hasattr(model, 'coef_') and hasattr(vectorizer, 'get_feature_names_out'):
        feature_names = vectorizer.get_feature_names_out()
        coef = model.coef_[prediction_idx] if len(model.coef_.shape) > 1 else model.coef_[0]
        
        # Get non-zero features in input
        if hasattr(X, 'toarray'):
            x_dense = X.toarray()[0]
        else:
            x_dense = X[0]
        
        # Feature contribution = feature_value * coefficient
        contributions = x_dense * coef
        top_indices = np.argsort(np.abs(contributions))[-10:][::-1]
        
        for idx in top_indices:
            if contributions[idx] != 0:
                top_features.append({
                    feature_names[idx]: float(contributions[idx])
                })
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'class_probabilities': class_probs,
        'top_features': top_features[:10],
    }


# ──────────────────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """GET /health — Model name, version, stage, F1 score, and load timestamp."""
    return HealthResponse(
        model_name=app_state['model_name'],
        model_version=app_state['model_version'],
        model_stage=app_state['model_stage'],
        f1_score=app_state['f1_score'],
        load_timestamp=app_state['load_timestamp'] or datetime.now().isoformat(),
        status="healthy",
    )


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_endpoint(request: PreprocessRequest):
    """POST /preprocess — Preprocess text with configurable steps."""
    start = time.time()
    
    steps = [s.value for s in request.steps]
    result = preprocess_text(request.text, steps)
    
    elapsed = (time.time() - start) * 1000
    
    return PreprocessResponse(
        original_text=request.text,
        tokens=result['tokens'],
        removed_stopwords=result['removed_stopwords'],
        processing_steps=result['steps_applied'],
        processing_time_ms=round(elapsed, 2),
        token_count=len(result['tokens']),
    )


@app.post("/classify", response_model=ClassifyResponse)
@limiter.limit("100/minute")
async def classify_endpoint(request: Request, body: ClassifyRequest):
    """POST /classify -- Classify single text."""
    start = time.time()
    
    result = classify_single(body.text)
    
    elapsed = (time.time() - start) * 1000
    
    return ClassifyResponse(
        prediction=result['prediction'],
        confidence=result['confidence'],
        class_probabilities=result['class_probabilities'],
        top_features=result['top_features'],
        processing_time_ms=round(elapsed, 2),
    )


@app.post("/classify/batch", response_model=BatchClassifyResponse)
@limiter.limit("10/minute")
async def batch_classify_endpoint(request: Request, body: BatchClassifyRequest):
    """POST /classify/batch -- Classify up to 100 texts, <500ms total."""
    start = time.time()
    
    results = []
    for text in body.texts:
        text_start = time.time()
        result = classify_single(text)
        text_elapsed = (time.time() - text_start) * 1000
        
        results.append(ClassifyResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            class_probabilities=result['class_probabilities'],
            top_features=result['top_features'],
            processing_time_ms=round(text_elapsed, 2),
        ))
    
    total_elapsed = (time.time() - start) * 1000
    
    return BatchClassifyResponse(
        results=results,
        total_texts=len(body.texts),
        total_processing_time_ms=round(total_elapsed, 2),
    )


@app.post("/retrieve/similar", response_model=SimilarResponse)
async def similar_endpoint(request: SimilarRequest):
    """POST /retrieve/similar — Retrieve top-k similar fact-checked claims."""
    start = time.time()
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Preprocess query
    processed = preprocess_text(request.text)
    query_text = ' '.join(processed['tokens'])
    
    # Vectorize
    vectorizer = app_state['vectorizer']
    tfidf_matrix = app_state['tfidf_matrix']
    dataset = app_state['dataset']
    
    if vectorizer is None or tfidf_matrix is None or dataset is None:
        raise HTTPException(status_code=503, 
                          detail="Similarity search not available — dataset not loaded")
    
    query_vec = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    top_indices = similarities.argsort()[-request.top_k:][::-1]
    
    similar_claims = []
    for rank, idx in enumerate(top_indices):
        text_col = 'text_clean' if 'text_clean' in dataset.columns else 'text'
        similar_claims.append(SimilarClaim(
            text=str(dataset.iloc[idx][text_col])[:500],
            label=str(dataset.iloc[idx]['label']),
            similarity_score=round(float(similarities[idx]), 4),
            rank=rank + 1,
        ))
    
    elapsed = (time.time() - start) * 1000
    
    return SimilarResponse(
        query=request.text,
        similar_claims=similar_claims,
        processing_time_ms=round(elapsed, 2),
    )


@app.get("/model/performance", response_model=PerformanceMetrics)
async def model_performance():
    """GET /model/performance — Live metrics and version history from MLFlow."""
    metrics = app_state.get('metrics', {})
    
    # Try to get from MLFlow
    version_history = []
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        tracking_uri = f"file:///{(PROJECT_ROOT / 'mlruns').as_posix()}"
        client = MlflowClient(tracking_uri)
        
        # Get model versions
        try:
            versions = client.search_model_versions(f"name='PakSentinel_LR'")
            for v in versions:
                version_history.append({
                    'version': str(v.version),
                    'stage': v.current_stage or 'None',
                    'creation_timestamp': str(v.creation_timestamp or ''),
                })
        except Exception:
            pass
    except Exception:
        pass
    
    if not version_history:
        version_history = [
            {'version': '1.0', 'stage': 'Production', 
             'creation_timestamp': app_state.get('load_timestamp', '')}
        ]
    
    return PerformanceMetrics(
        current_model=app_state['model_name'],
        current_version=app_state['model_version'],
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        version_history=version_history,
        last_updated=datetime.now().isoformat(),
    )


# ──────────────────────────────────────────────────────────
#  Run Server
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
