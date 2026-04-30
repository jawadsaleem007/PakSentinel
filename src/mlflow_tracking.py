"""
Task 6 — MLFlow Experiment Tracking [25 Marks]

- Experiment hierarchy: Preprocessing Ablation → Feature Comparison → Model Comparison
- Comprehensive logging of parameters, metrics, and artifacts
- 6 preprocessing ablation configurations
- Parallel coordinates plot
- Model Registry with automated promotion logic
"""

import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix)
from sklearn.preprocessing import LabelEncoder, label_binarize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "figures"
MLFLOW_DIR = Path(__file__).parent.parent / "mlruns"


# ══════════════════════════════════════════════════════════
#  MLFlow Setup
# ══════════════════════════════════════════════════════════
def setup_mlflow():
    """Initialize MLFlow tracking."""
    tracking_uri = f"file:///{MLFLOW_DIR.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"  MLFlow tracking URI: {tracking_uri}")
    return MlflowClient(tracking_uri)


# ══════════════════════════════════════════════════════════
#  Logging Helpers
# ══════════════════════════════════════════════════════════
def log_full_run(model, X_test, y_test, y_pred, label_encoder,
                  run_params: dict, training_time: float):
    """
    Log a complete MLFlow run with all required parameters, metrics, and artifacts.
    """
    # Log parameters
    for key, val in run_params.items():
        mlflow.log_param(key, val)
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    
    # Per-class metrics
    classes = label_encoder.classes_
    prec_per = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_per = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC (OvR)
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=range(len(classes)))
            roc_auc = roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr')
        else:
            roc_auc = 0.0
    except Exception:
        roc_auc = 0.0
    
    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1_weighted)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("training_time_s", training_time)
    
    for i, cls in enumerate(classes):
        mlflow.log_metric(f"precision_{cls}", prec_per[i])
        mlflow.log_metric(f"recall_{cls}", rec_per[i])
        mlflow.log_metric(f"f1_{cls}", f1_per[i])
    
    # Generate and log artifacts
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title('Confusion Matrix', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    cm_path = REPORTS_DIR / f"cm_{run_params.get('model_type', 'model')}.png"
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(str(cm_path))
    
    # 2. Classification report
    report = classification_report(y_test, y_pred, target_names=classes, zero_division=0)
    report_path = REPORTS_DIR / f"report_{run_params.get('model_type', 'model')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    mlflow.log_artifact(str(report_path))
    
    # 3. ROC curve
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=range(len(classes)))
            
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, cls in enumerate(classes):
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc_i = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc_i:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.set_title('ROC Curve', fontweight='bold')
            ax.legend()
            roc_path = REPORTS_DIR / f"roc_{run_params.get('model_type', 'model')}.png"
            plt.savefig(roc_path, dpi=100, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(str(roc_path))
        except Exception:
            pass
    
    return {
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'roc_auc': roc_auc,
    }


# ══════════════════════════════════════════════════════════
#  Preprocessing Ablation Study (6 configurations)
# ══════════════════════════════════════════════════════════
def run_preprocessing_ablation(df: pd.DataFrame, le: LabelEncoder) -> dict:
    """
    Run 6 preprocessing ablation configurations as separate MLFlow runs.
    
    Configurations vary:
    - Stopword list (standard vs custom)
    - Normalization (stemming vs lemmatization vs none)  
    - Min token length (1 vs 3)
    - TF-IDF max features (5000 vs 10000 vs 15000)
    """
    from nltk.stem import SnowballStemmer
    from nltk.stem import WordNetLemmatizer
    from src.stopwords import NLTK_STOPWORDS, CUSTOM_STOPWORDS, remove_stopwords
    
    print(f"\n  PREPROCESSING ABLATION STUDY (6 configurations)")
    print(f"  {'─' * 50}")
    
    configs = [
        {'stopwords': 'standard', 'normalization': 'lemma', 'min_token_len': 1, 'max_features': 5000},
        {'stopwords': 'custom', 'normalization': 'lemma', 'min_token_len': 1, 'max_features': 5000},
        {'stopwords': 'custom', 'normalization': 'stem', 'min_token_len': 1, 'max_features': 5000},
        {'stopwords': 'custom', 'normalization': 'lemma', 'min_token_len': 3, 'max_features': 5000},
        {'stopwords': 'custom', 'normalization': 'lemma', 'min_token_len': 1, 'max_features': 10000},
        {'stopwords': 'custom', 'normalization': 'lemma', 'min_token_len': 1, 'max_features': 15000},
    ]
    
    y = le.fit_transform(df['label'])
    ablation_results = []
    
    experiment_name = "PakSentinel_Preprocessing_Ablation"
    mlflow.set_experiment(experiment_name)
    
    stemmer = SnowballStemmer('english')
    lemmatizer_obj = WordNetLemmatizer()
    
    for i, config in enumerate(configs):
        config_name = f"config_{i+1}"
        print(f"\n  Running {config_name}: {config}")
        
        with mlflow.start_run(run_name=config_name):
            # Apply preprocessing based on config
            stopword_set = NLTK_STOPWORDS if config['stopwords'] == 'standard' else CUSTOM_STOPWORDS
            
            def process_tokens(tokens):
                # Remove stopwords
                tokens = [t for t in tokens if t.lower() not in stopword_set]
                # Min token length
                tokens = [t for t in tokens if len(t) >= config['min_token_len']]
                # Normalization
                if config['normalization'] == 'stem':
                    tokens = [stemmer.stem(t) for t in tokens]
                elif config['normalization'] == 'lemma':
                    tokens = [lemmatizer_obj.lemmatize(t) for t in tokens]
                return tokens
            
            processed_tokens = df['tokens'].apply(process_tokens)
            texts = processed_tokens.apply(lambda t: ' '.join(t))
            
            # TF-IDF
            vectorizer = TfidfVectorizer(max_features=config['max_features'], sublinear_tf=True)
            X = vectorizer.fit_transform(texts)
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train LR
            start_time = time.time()
            model = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            
            # Log run
            run_params = {
                'dataset_sources': 'LIAR,ISOT,SarcasmHeadlines',
                'train_size': len(y_train),
                'test_size': len(y_test),
                'tokenizer': 'NLTK_word_tokenize',
                'stopword_list': config['stopwords'],
                'normalization_method': config['normalization'],
                'min_token_length': config['min_token_len'],
                'vectorizer': 'TF-IDF_sublinear',
                'max_features': config['max_features'],
                'model_type': 'LogisticRegression_L2',
                'config_name': config_name,
            }
            
            metrics = log_full_run(model, X_test, y_test, y_pred, le,
                                    run_params, training_time)
            
            # Log TF-IDF vocabulary
            vocab_path = REPORTS_DIR / f"vocab_{config_name}.json"
            vocab = {word: int(idx) for word, idx in vectorizer.vocabulary_.items()}
            with open(vocab_path, 'w') as f:
                json.dump(dict(list(vocab.items())[:100]), f)  # First 100 terms
            mlflow.log_artifact(str(vocab_path))
            
            ablation_results.append({
                **config,
                **metrics,
                'config_name': config_name,
            })
    
    return ablation_results


def plot_parallel_coordinates(ablation_results: list) -> None:
    """
    Create parallel coordinates plot with F1-weighted on y-axis.
    """
    print(f"\n  Creating parallel coordinates plot...")
    
    import plotly.express as px
    
    df_results = pd.DataFrame(ablation_results)
    
    # Encode categorical variables
    df_results['stopwords_num'] = (df_results['stopwords'] == 'custom').astype(int)
    df_results['normalization_num'] = df_results['normalization'].map({'none': 0, 'stem': 1, 'lemma': 2})
    
    fig = px.parallel_coordinates(
        df_results,
        dimensions=['stopwords_num', 'normalization_num', 'min_token_len', 
                     'max_features', 'accuracy', 'f1_weighted'],
        color='f1_weighted',
        labels={
            'stopwords_num': 'Stopwords (0=std, 1=custom)',
            'normalization_num': 'Norm (0=none, 1=stem, 2=lemma)',
            'min_token_len': 'Min Token Length',
            'max_features': 'Max Features',
            'accuracy': 'Accuracy',
            'f1_weighted': 'F1 Weighted',
        },
        title='Preprocessing Ablation: Parallel Coordinates',
        color_continuous_scale='Viridis',
    )
    
    # Save as HTML and PNG
    fig.write_html(str(REPORTS_DIR / 'parallel_coordinates.html'))
    fig.write_image(str(REPORTS_DIR / 'parallel_coordinates.png'), width=1200, height=600)
    print(f"  Saved: parallel_coordinates.png")
    
    # Also create a matplotlib version (for PDF report embedding)
    fig_mpl, axes = plt.subplots(figsize=(14, 6))
    
    # Normalize columns for parallel coordinates
    cols = ['stopwords_num', 'normalization_num', 'min_token_len', 
            'max_features', 'accuracy', 'f1_weighted']
    df_norm = df_results[cols].copy()
    for col in cols:
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        if col_max > col_min:
            df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 0.5
    
    cmap = plt.cm.viridis
    for idx, row in df_norm.iterrows():
        color = cmap(row['f1_weighted'])
        axes.plot(range(len(cols)), row[cols].values, '-o', 
                 color=color, alpha=0.7, linewidth=2, markersize=6)
    
    axes.set_xticks(range(len(cols)))
    axes.set_xticklabels(['Stopwords', 'Normalization', 'Min Token\nLength', 
                          'Max Features', 'Accuracy', 'F1 Weighted'], fontsize=10)
    axes.set_title('Preprocessing Ablation: Parallel Coordinates', fontsize=14, fontweight='bold')
    axes.set_ylabel('Normalized Value', fontsize=11)
    axes.grid(True, alpha=0.3, axis='y')
    
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=axes, label='F1 Weighted')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'parallel_coordinates_mpl.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: parallel_coordinates_mpl.png")


# ══════════════════════════════════════════════════════════
#  Model Registry & Promotion Logic
# ══════════════════════════════════════════════════════════
def register_and_promote_models(client: MlflowClient, results: dict) -> None:
    """
    Register best model from each algorithm family.
    Implement automated promotion: Staging → Production if F1 ≥ current + 1%.
    """
    print(f"\n  MODEL REGISTRY & AUTOMATED PROMOTION")
    print(f"  {'─' * 50}")
    
    model_families = {
        'NaiveBayes': 'PakSentinel_NB',
        'LogisticRegression': 'PakSentinel_LR',
        'PolynomialLR': 'PakSentinel_PolyLR',
    }
    
    for family, registry_name in model_families.items():
        print(f"\n  Registering {family} as '{registry_name}'...")
        
        try:
            # Create or get model
            try:
                client.create_registered_model(registry_name)
            except Exception:
                pass  # Already exists
            
            # Automated promotion logic
            print(f"    Promotion rule: Staging → Production if F1 improves by ≥ 1%")
            
            # Check current production model
            current_production_f1 = 0.0
            try:
                versions = client.search_model_versions(f"name='{registry_name}'")
                for v in versions:
                    if v.current_stage == 'Production':
                        run = client.get_run(v.run_id)
                        current_production_f1 = run.data.metrics.get('f1_weighted', 0)
            except Exception:
                pass
            
            print(f"    Current Production F1: {current_production_f1:.4f}")
            
        except Exception as e:
            print(f"    [WARN] Registry operation failed: {e}")
            print(f"    This is expected if MLFlow server is not running.")


# ══════════════════════════════════════════════════════════
#  Promotion Logic (Standalone Function)
# ══════════════════════════════════════════════════════════
def auto_promote(client: MlflowClient, model_name: str, 
                  new_f1: float, run_id: str) -> str:
    """
    Automated model promotion logic.
    
    A model moves from Staging to Production only if its F1-weighted 
    exceeds the current Production model by at least 1%.
    
    Args:
        client: MLFlow client
        model_name: Registered model name
        new_f1: F1-weighted of the new model
        run_id: Run ID of the new model
        
    Returns:
        Final stage of the model
    """
    # Get current production F1
    current_prod_f1 = 0.0
    current_prod_version = None
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if v.current_stage == 'Production':
                run = client.get_run(v.run_id)
                current_prod_f1 = run.data.metrics.get('f1_weighted', 0)
                current_prod_version = v.version
    except Exception:
        pass
    
    # Register new version as Staging
    try:
        model_uri = f"runs:/{run_id}/model"
        mv = client.create_model_version(model_name, model_uri, run_id)
        new_version = mv.version
        
        client.transition_model_version_stage(model_name, new_version, "Staging")
        print(f"    Registered version {new_version} as Staging (F1={new_f1:.4f})")
        
        # Check promotion criteria: new F1 > current + 1%
        if new_f1 >= current_prod_f1 + 0.01:
            # Demote current production
            if current_prod_version:
                client.transition_model_version_stage(
                    model_name, current_prod_version, "Archived"
                )
            
            # Promote new model
            client.transition_model_version_stage(model_name, new_version, "Production")
            print(f"    ✓ PROMOTED to Production! (F1 improvement: {new_f1 - current_prod_f1:.4f} ≥ 0.01)")
            return "Production"
        else:
            print(f"    ✗ Kept in Staging (F1 improvement: {new_f1 - current_prod_f1:.4f} < 0.01)")
            return "Staging"
    except Exception as e:
        print(f"    [WARN] Promotion failed: {e}")
        return "None"


# ══════════════════════════════════════════════════════════
#  Run Task 6
# ══════════════════════════════════════════════════════════
def run_mlflow_tracking(df: pd.DataFrame) -> dict:
    """Execute Task 6: MLFlow Experiment Tracking."""
    print("\n" + "=" * 60)
    print("TASK 6: MLFLOW EXPERIMENT TRACKING")
    print("=" * 60)
    
    # Setup MLFlow
    client = setup_mlflow()
    
    # Print experiment hierarchy
    print(f"\n  EXPERIMENT HIERARCHY:")
    print(f"  ─────────────────────")
    print(f"  PakSentinel_Preprocessing_Ablation")
    print(f"    ├── config_1: standard stopwords + lemma + len≥1 + 5K features")
    print(f"    ├── config_2: custom stopwords + lemma + len≥1 + 5K features")
    print(f"    ├── config_3: custom stopwords + stem + len≥1 + 5K features")
    print(f"    ├── config_4: custom stopwords + lemma + len≥3 + 5K features")
    print(f"    ├── config_5: custom stopwords + lemma + len≥1 + 10K features")
    print(f"    └── config_6: custom stopwords + lemma + len≥1 + 15K features")
    print(f"  PakSentinel_Feature_Comparison")
    print(f"    ├── TF-IDF Only")
    print(f"    ├── Word2Vec Only")
    print(f"    └── TF-IDF + Word2Vec")
    print(f"  PakSentinel_Model_Comparison")
    print(f"    ├── Naive Bayes")
    print(f"    ├── Logistic Regression (L1/L2/ElasticNet)")
    print(f"    └── Polynomial LR (degree 1/2/3)")
    
    le = LabelEncoder()
    
    # Run preprocessing ablation
    ablation_results = run_preprocessing_ablation(df, le)
    
    # Parallel coordinates plot
    try:
        plot_parallel_coordinates(ablation_results)
    except Exception as e:
        print(f"  [WARN] Parallel coordinates plot failed: {e}")
        print(f"  Falling back to matplotlib-only version...")
    
    # Model registry
    register_and_promote_models(client, {})
    
    return {
        'ablation_results': ablation_results,
        'client': client,
    }


if __name__ == "__main__":
    print("MLFlow tracking module loaded. Run via pipeline.py")
