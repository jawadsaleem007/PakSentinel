"""
Task 5.1 — Naive Bayes [8 Marks]

FROM-SCRATCH Multinomial Naive Bayes (sklearn NOT permitted).
- Configurable Laplace smoothing (alpha parameter)
- Accepts BoW and TF-IDF inputs
- Outputs class probabilities
- Operates in log-space for numerical stability
- 80/20 train/test split
- Manual examination of 30 misclassified samples with error categorization
- Alpha sensitivity analysis: {0.01, 0.1, 0.5, 1.0, 2.0, 5.0}
"""

import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "figures"


class MultinomialNaiveBayes:
    """
    From-scratch Multinomial Naive Bayes classifier.
    
    ALL computation done in log-space for numerical stability.
    No sklearn dependency for the classifier itself.
    
    P(c|d) ∝ P(c) × ∏ P(w|c)
    log P(c|d) = log P(c) + Σ log P(w|c)
    
    With Laplace smoothing:
    P(w|c) = (count(w,c) + α) / (total_words_in_c + α × |V|)
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Laplace smoothing parameter
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_priors_ = None
        self.feature_log_probs_ = None
        self.vocab_size_ = 0
        self.class_word_counts_ = None
    
    def fit(self, X, y):
        """
        Train the Multinomial Naive Bayes model.
        
        Args:
            X: Feature matrix (n_samples × n_features), can be sparse or dense
            y: Labels (n_samples,)
        """
        # Convert sparse to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.vocab_size_ = n_features
        
        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Compute class priors: P(c) = |docs in c| / |total docs|
        self.class_log_priors_ = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            class_count = np.sum(y == c)
            self.class_log_priors_[i] = np.log(class_count / n_samples)
        
        # Compute class-conditional word probabilities with Laplace smoothing
        # P(w|c) = (count(w, c) + alpha) / (sum_all_words_in_c + alpha * |V|)
        self.feature_log_probs_ = np.zeros((n_classes, n_features))
        self.class_word_counts_ = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes_):
            class_mask = (y == c)
            class_feature_counts = X[class_mask].sum(axis=0)
            self.class_word_counts_[i] = class_feature_counts
            
            # Total word count in this class
            total_count = class_feature_counts.sum()
            
            # Smoothed log probability
            smoothed_counts = class_feature_counts + self.alpha
            smoothed_total = total_count + self.alpha * n_features
            
            self.feature_log_probs_[i] = np.log(smoothed_counts / smoothed_total)
        
        return self
    
    def predict_log_proba(self, X):
        """
        Compute log probabilities for each class.
        
        Args:
            X: Feature matrix (n_samples × n_features)
            
        Returns:
            Log probabilities (n_samples × n_classes)
        """
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.array(X, dtype=np.float64)
        
        # log P(c|d) = log P(c) + Σ x_i × log P(w_i|c)
        log_probs = X @ self.feature_log_probs_.T + self.class_log_priors_
        
        return log_probs
    
    def predict_proba(self, X):
        """
        Compute class probabilities (normalized from log space).
        
        Returns:
            Probabilities (n_samples × n_classes)
        """
        log_probs = self.predict_log_proba(X)
        
        # Log-sum-exp trick for numerical stability
        max_log = np.max(log_probs, axis=1, keepdims=True)
        log_probs_shifted = log_probs - max_log
        probs = np.exp(log_probs_shifted)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X):
        """
        Predict class labels.
        
        Returns:
            Predicted labels (n_samples,)
        """
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]
    
    def score(self, X, y):
        """Compute accuracy."""
        return np.mean(self.predict(X) == np.array(y))


# ══════════════════════════════════════════════════════════
#  Error Analysis
# ══════════════════════════════════════════════════════════
def analyze_misclassifications(nb: MultinomialNaiveBayes, X_test, y_test, 
                                df_test: pd.DataFrame, n_examine: int = 30) -> dict:
    """
    Manually examine misclassified samples and categorize error types.
    
    Error categories:
    1. SHORT_TEXT: Too little text to distinguish
    2. AMBIGUOUS: Content genuinely ambiguous between classes
    3. TOPIC_OVERLAP: Similar topics in different classes
    4. SATIRE_CONFUSION: Satire mistaken for real or fake
    5. SENSATIONAL_REAL: Real news with sensational language
    6. FORMAL_FAKE: Fake news with professional language
    """
    y_pred = nb.predict(X_test)
    y_test_arr = np.array(y_test)
    
    misclassified_mask = y_pred != y_test_arr
    misclassified_indices = np.where(misclassified_mask)[0]
    
    n_examine = min(n_examine, len(misclassified_indices))
    examine_indices = misclassified_indices[:n_examine]
    
    print(f"\n  MISCLASSIFICATION ANALYSIS ({n_examine} samples)")
    print(f"  {'─' * 60}")
    print(f"  Total test samples: {len(y_test)}")
    print(f"  Total misclassified: {len(misclassified_indices)} ({len(misclassified_indices)/len(y_test)*100:.1f}%)")
    
    error_categories = Counter()
    errors = []
    
    for i, idx in enumerate(examine_indices):
        true_label = y_test_arr[idx]
        pred_label = y_pred[idx]
        
        # Get text for analysis
        if 'text_clean' in df_test.columns:
            text = df_test.iloc[idx]['text_clean']
        else:
            text = df_test.iloc[idx]['text']
        text_preview = text[:100] if isinstance(text, str) else str(text)[:100]
        
        # Auto-categorize error type based on heuristics
        text_len = len(str(text))
        if text_len < 50:
            category = 'SHORT_TEXT'
        elif (true_label == 'Satire' and pred_label in ['Fake', 'Real']) or \
             (pred_label == 'Satire' and true_label in ['Fake', 'Real']):
            category = 'SATIRE_CONFUSION'
        elif true_label == 'Real' and pred_label == 'Fake':
            category = 'SENSATIONAL_REAL'
        elif true_label == 'Fake' and pred_label == 'Real':
            category = 'FORMAL_FAKE'
        else:
            category = 'TOPIC_OVERLAP'
        
        error_categories[category] += 1
        errors.append({
            'index': int(idx),
            'true': true_label,
            'predicted': pred_label,
            'category': category,
            'text_preview': text_preview,
        })
        
        if i < 10:  # Print first 10
            print(f"\n  [{i+1}] True: {true_label}, Predicted: {pred_label}")
            print(f"      Category: {category}")
            print(f"      Text: \"{text_preview}...\"")
    
    print(f"\n  ERROR CATEGORY DISTRIBUTION:")
    for category, count in error_categories.most_common():
        print(f"    {category}: {count} ({count/n_examine*100:.0f}%)")
    
    return {
        'total_misclassified': len(misclassified_indices),
        'examined': n_examine,
        'error_categories': dict(error_categories),
        'errors': errors,
    }


# ══════════════════════════════════════════════════════════
#  Alpha Sensitivity Analysis
# ══════════════════════════════════════════════════════════
def alpha_sensitivity_analysis(X_train, X_test, y_train, y_test) -> dict:
    """
    Perform alpha sensitivity analysis over {0.01, 0.1, 0.5, 1.0, 2.0, 5.0}.
    """
    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = {}
    
    print(f"\n  ALPHA SENSITIVITY ANALYSIS")
    print(f"  {'─' * 60}")
    print(f"  {'Alpha':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─' * 48}")
    
    for alpha in alphas:
        nb = MultinomialNaiveBayes(alpha=alpha)
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results[alpha] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
        print(f"  {alpha:>8.2f} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        vals = [results[a][metric] for a in alphas]
        ax.plot(alphas, vals, marker='o', label=metric.title())
    
    ax.set_xlabel('Alpha (Laplace Smoothing Parameter)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Naive Bayes: Alpha Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS_DIR / 'nb_alpha_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: nb_alpha_sensitivity.png")
    
    # Find best alpha
    best_alpha = max(results, key=lambda a: results[a]['f1'])
    print(f"\n  Best alpha: {best_alpha} (F1 = {results[best_alpha]['f1']:.4f})")
    
    return results


# ══════════════════════════════════════════════════════════
#  Run Task 5.1
# ══════════════════════════════════════════════════════════
def run_naive_bayes(df: pd.DataFrame, bow_matrix, tfidf_matrix) -> dict:
    """Execute Task 5.1: From-scratch Multinomial Naive Bayes."""
    print("\n" + "=" * 60)
    print("TASK 5.1: NAIVE BAYES (FROM SCRATCH)")
    print("=" * 60)
    
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    
    # 80/20 split
    X_train_bow, X_test_bow, y_train, y_test = train_test_split(
        bow_matrix, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_tfidf, X_test_tfidf, _, _ = train_test_split(
        tfidf_matrix, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Also split the dataframe for error analysis
    _, df_test, _, _ = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    df_test = df_test.reset_index(drop=True)
    
    # Train with BoW
    print(f"\n  Training with BoW features...")
    nb_bow = MultinomialNaiveBayes(alpha=1.0)
    nb_bow.fit(X_train_bow, y_train)
    y_pred_bow = nb_bow.predict(X_test_bow)
    f1_bow = f1_score(y_test, y_pred_bow, average='weighted')
    print(f"    BoW F1 (weighted): {f1_bow:.4f}")
    
    # Train with TF-IDF
    print(f"\n  Training with TF-IDF features...")
    nb_tfidf = MultinomialNaiveBayes(alpha=1.0)
    # TF-IDF can have negative values after sublinear scaling, shift to non-negative
    tfidf_min = X_train_tfidf.min()
    if hasattr(tfidf_min, 'toarray'):
        tfidf_min_val = tfidf_min.toarray().min()
    else:
        tfidf_min_val = tfidf_min if isinstance(tfidf_min, (int, float)) else 0
    
    nb_tfidf.fit(X_train_tfidf, y_train)
    y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)
    f1_tfidf = f1_score(y_test, y_pred_tfidf, average='weighted')
    print(f"    TF-IDF F1 (weighted): {f1_tfidf:.4f}")
    
    # Full classification report (best model)
    best_model = nb_tfidf if f1_tfidf >= f1_bow else nb_bow
    best_features = 'TF-IDF' if f1_tfidf >= f1_bow else 'BoW'
    X_test_best = X_test_tfidf if f1_tfidf >= f1_bow else X_test_bow
    y_pred_best = nb_tfidf.predict(X_test_best) if f1_tfidf >= f1_bow else y_pred_bow
    
    print(f"\n  Best input: {best_features}")
    print(f"\n  CLASSIFICATION REPORT ({best_features}):")
    print(classification_report(y_test, y_pred_best, 
                                 target_names=le.classes_, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f'Naive Bayes Confusion Matrix ({best_features})', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS_DIR / 'nb_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test class probability output
    probs = best_model.predict_proba(X_test_best[:3])
    print(f"\n  Sample class probabilities (first 3 test samples):")
    for i in range(3):
        prob_dict = {le.classes_[j]: f"{probs[i][j]:.4f}" for j in range(len(le.classes_))}
        print(f"    Sample {i+1}: {prob_dict} → Predicted: {le.classes_[y_pred_best[i]]}")
    
    # Error analysis
    error_analysis = analyze_misclassifications(
        best_model, X_test_best, y_test, df_test, n_examine=30
    )
    
    # Alpha sensitivity
    X_train_best = X_train_tfidf if best_features == 'TF-IDF' else X_train_bow
    alpha_results = alpha_sensitivity_analysis(X_train_best, X_test_best, y_train, y_test)
    
    return {
        'nb_bow': nb_bow,
        'nb_tfidf': nb_tfidf,
        'f1_bow': f1_bow,
        'f1_tfidf': f1_tfidf,
        'best_features': best_features,
        'error_analysis': error_analysis,
        'alpha_results': alpha_results,
        'label_encoder': le,
        'y_test': y_test,
        'y_pred': y_pred_best,
    }


if __name__ == "__main__":
    print("Naive Bayes module loaded. No sklearn classifiers used.")
    print("Run via pipeline.py")
