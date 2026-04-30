"""
PakSentinel — Master Pipeline Orchestrator

Runs all tasks end-to-end in sequence:
1. Data Sourcing & Reliability Assessment
2. Data Storage Architecture  
3. NLP Processing Pipeline (Cleaning → Tokenization → Stopwords → Normalization → Features)
4. N-Gram Language Models
5. Machine Learning Models (NB → LR → Polynomial)
6. MLFlow Experiment Tracking
7. Save artifacts for FastAPI deployment

Ensures full reproducibility — no hardcoded metrics.
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Fix Windows encoding
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Run the complete PakSentinel pipeline."""
    total_start = time.time()
    
    print("=" * 60)
    print("   PAKSENTINEL - Misinformation Detection Pipeline")
    print("   End-to-End NLP Pipeline (Tasks 1-7)")
    print("=" * 60)
    
    # ------------------------------------------------------
    #  TASK 1: Data Sourcing [15 Marks]
    # ------------------------------------------------------
    from src.data_sourcing import run_task1
    df = run_task1()
    print(f"\n✅ Task 1 complete: {len(df)} samples")
    
    # ------------------------------------------------------
    #  TASK 2: Data Storage [10 Marks]
    # ------------------------------------------------------
    from src.data_lake_manager import DataLakeManager, run_task2
    dlm = run_task2()
    
    # Upload raw data files
    raw_dir = PROJECT_ROOT / "data" / "raw"
    for raw_file in raw_dir.glob("*"):
        if raw_file.is_file() and not raw_file.name.startswith('_'):
            try:
                dlm.upload_raw(str(raw_file), {
                    'source': 'pipeline_run',
                    'dataset': raw_file.stem,
                })
            except Exception as e:
                print(f"  [WARN] Could not upload {raw_file.name}: {e}")
    
    # Upload processed dataset
    dlm.upload_processed(df, 'combined_dataset', 'v1.0', data_type='dataframe')
    print(f"\n✅ Task 2 complete: DataLakeManager initialized")
    
    # ------------------------------------------------------
    #  TASK 3.1: Cleaning [5 Marks]
    # ------------------------------------------------------
    from src.cleaning import run_cleaning
    df = run_cleaning(df)
    print(f"\n✅ Task 3.1 complete: {len(df)} cleaned samples")
    
    # ------------------------------------------------------
    #  TASK 3.2: Tokenization [5 Marks]
    # ------------------------------------------------------
    from src.tokenization import run_tokenization
    token_results = run_tokenization(df)
    print(f"\n✅ Task 3.2 complete: Tokenization comparison done")
    
    # ------------------------------------------------------
    #  TASK 3.3: Stopword Removal [5 Marks]
    # ------------------------------------------------------
    from src.stopwords import run_stopword_analysis
    stopword_results = run_stopword_analysis(df)
    print(f"\n✅ Task 3.3 complete: Stopword analysis done")
    
    # ------------------------------------------------------
    #  TASK 3.4: Stemming vs. Lemmatization [5 Marks]
    # ------------------------------------------------------
    from src.normalization import run_normalization
    norm_results = run_normalization(df)
    print(f"\n✅ Task 3.4 complete: Normalization comparison done")
    
    # ------------------------------------------------------
    #  TASK 3.5: Feature Representation [15 Marks]
    # ------------------------------------------------------
    from src.features import run_features
    feature_results = run_features(df)
    print(f"\n✅ Task 3.5 complete: BoW, TF-IDF, Word2Vec done")
    
    # ------------------------------------------------------
    #  TASK 4: N-Gram Language Models [10 Marks]
    # ------------------------------------------------------
    from src.ngram_models import run_ngram_models
    ngram_results = run_ngram_models(df)
    print(f"\n✅ Task 4 complete: N-gram models trained")
    
    # ------------------------------------------------------
    #  TASK 5.1: Naive Bayes [8 Marks]
    # ------------------------------------------------------
    from src.naive_bayes import run_naive_bayes
    bow_matrix = feature_results['bow']['matrix']
    tfidf_matrix = feature_results['tfidf']['Sublinear TF']['matrix']
    nb_results = run_naive_bayes(df, bow_matrix, tfidf_matrix)
    print(f"\n✅ Task 5.1 complete: Naive Bayes (from scratch)")
    
    # ------------------------------------------------------
    #  TASK 5.2: Logistic Regression [9 Marks]
    # ------------------------------------------------------
    from src.logistic_regression import run_logistic_regression
    tfidf_result = feature_results['tfidf']['Sublinear TF']
    lr_results = run_logistic_regression(df, tfidf_result)
    print(f"\n✅ Task 5.2 complete: Logistic Regression")
    
    # ------------------------------------------------------
    #  TASK 5.3: Polynomial LR [8 Marks]
    # ------------------------------------------------------
    from src.polynomial_lr import run_polynomial_lr
    poly_results = run_polynomial_lr(df, tfidf_result)
    print(f"\n✅ Task 5.3 complete: Polynomial Features + LR")
    
    # ------------------------------------------------------
    #  TASK 6: MLFlow Tracking [25 Marks]
    # ------------------------------------------------------
    from src.mlflow_tracking import run_mlflow_tracking
    mlflow_results = run_mlflow_tracking(df)
    print(f"\n✅ Task 6 complete: MLFlow tracking done")
    
    # ------------------------------------------------------
    #  Save artifacts for FastAPI (Task 7)
    # ------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAVING DEPLOYMENT ARTIFACTS")
    print("=" * 60)
    
    deploy_dir = PROJECT_ROOT / "data" / "processed" / "v1.0"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best model (LR with best variant)
    best_variant = lr_results['best_variant']
    best_model = lr_results['variants'][best_variant]['model']
    with open(deploy_dir / "best_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    print(f"  Saved best model ({best_variant} LR)")
    
    # Save vectorizer
    vectorizer = tfidf_result['vectorizer']
    with open(deploy_dir / "tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  Saved TF-IDF vectorizer")
    
    # Save label encoder
    le = lr_results['label_encoder']
    with open(deploy_dir / "label_encoder.pkl", 'wb') as f:
        pickle.dump(le, f)
    print(f"  Saved label encoder")
    
    # Save TF-IDF matrix for similarity search
    with open(deploy_dir / "tfidf_matrix.pkl", 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    print(f"  Saved TF-IDF matrix")
    
    # Save metrics
    from sklearn.metrics import f1_score
    y_test = nb_results['y_test']
    y_pred = nb_results['y_pred']
    metrics = {
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'nb_f1_bow': nb_results['f1_bow'],
        'nb_f1_tfidf': nb_results['f1_tfidf'],
    }
    for name, result in lr_results['variants'].items():
        metrics[f'lr_{name}_f1'] = result['f1']
    
    with open(deploy_dir / "metrics.pkl", 'wb') as f:
        pickle.dump(metrics, f)
    print(f"  Saved metrics")
    
    # Upload embeddings
    embeddings_dir = PROJECT_ROOT / "data" / "embeddings"
    for model_file in embeddings_dir.glob("*.model"):
        dlm.upload_embeddings(str(model_file), 'v1.0', model_file.stem)
    
    # Upload processed data
    dlm.upload_processed(df, 'cleaned_dataset', 'v1.0', data_type='dataframe')
    
    # Save processed dataset for API
    df.to_parquet(deploy_dir.parent / "combined_dataset.parquet", index=False)
    
    # ------------------------------------------------------
    #  Summary
    # ------------------------------------------------------
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Dataset: {len(df)} samples across {df['label'].nunique()} classes")
    print(f"  Models trained: NB (from scratch), LR (3 variants), Poly LR (3 degrees)")
    print(f"  Plots saved to: {PROJECT_ROOT / 'reports' / 'figures'}")
    print(f"  MLFlow runs: {PROJECT_ROOT / 'mlruns'}")
    print(f"  Deployment artifacts: {deploy_dir}")
    
    print(f"\n  To start the API server:")
    print(f"    python -m uvicorn api.app:app --host 0.0.0.0 --port 8000")
    print(f"\n  To run tests:")
    print(f"    python -m pytest tests/test_api.py -v")
    print(f"\n  To view MLFlow UI:")
    print(f"    mlflow ui --backend-store-uri file:///{(PROJECT_ROOT / 'mlruns').as_posix()}")


if __name__ == "__main__":
    main()
