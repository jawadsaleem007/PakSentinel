"""
Task 5.2 — Logistic Regression [9 Marks]

- Train with TF-IDF features using L1, L2, and ElasticNet regularization (C=1.0)
- Extract top 20 weighted features per class for L2
- Plot ROC curves for all three variants on one figure with per-class AUC
- 250-word explanation: why LR handles correlated features better than NB
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder, label_binarize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "figures"


# ══════════════════════════════════════════════════════════
#  Logistic Regression with Regularization
# ══════════════════════════════════════════════════════════
def train_lr_variants(X_train, X_test, y_train, y_test, 
                       feature_names, label_encoder) -> dict:
    """
    Train LR with L1, L2, and ElasticNet regularization.
    """
    print(f"\n  LOGISTIC REGRESSION VARIANTS (C=1.0)")
    print(f"  {'─' * 50}")
    
    variants = {
        'L1': LogisticRegression(penalty='l1', C=1.0, solver='saga', 
                                  max_iter=2000, random_state=42, multi_class='multinomial'),
        'L2': LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', 
                                  max_iter=2000, random_state=42, multi_class='multinomial'),
        'ElasticNet': LogisticRegression(penalty='elasticnet', C=1.0, solver='saga',
                                          l1_ratio=0.5, max_iter=2000, random_state=42,
                                          multi_class='multinomial'),
    }
    
    results = {}
    
    for name, clf in variants.items():
        print(f"\n  Training {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"    Accuracy: {acc:.4f}")
        print(f"    F1 (weighted): {f1:.4f}")
        print(classification_report(y_test, y_pred, 
                                     target_names=label_encoder.classes_, zero_division=0))
        
        results[name] = {
            'model': clf,
            'accuracy': acc,
            'f1': f1,
            'y_pred': y_pred,
        }
    
    return results


def extract_top_features(lr_model, feature_names, label_encoder, top_k: int = 20) -> dict:
    """
    Extract top 20 weighted features per class from L2 model.
    """
    print(f"\n  TOP {top_k} WEIGHTED FEATURES PER CLASS (L2)")
    print(f"  {'─' * 50}")
    
    coef = lr_model.coef_
    classes = label_encoder.classes_
    
    top_features = {}
    for i, cls in enumerate(classes):
        weights = coef[i] if len(coef.shape) > 1 else coef[0]
        top_indices = np.argsort(np.abs(weights))[-top_k:][::-1]
        
        features = [(feature_names[idx], weights[idx]) for idx in top_indices]
        top_features[cls] = features
        
        print(f"\n    {cls}:")
        for j, (feat, weight) in enumerate(features[:10]):
            direction = "+" if weight > 0 else "-"
            print(f"      {j+1:>2}. {feat:<20} {direction}{abs(weight):.4f}")
        print(f"      ... ({top_k - 10} more)")
    
    return top_features


def plot_roc_curves(results: dict, X_test, y_test, label_encoder) -> None:
    """
    Plot ROC curves for all three LR variants on one figure with per-class AUC.
    """
    print(f"\n  Plotting ROC curves...")
    
    classes = label_encoder.classes_
    n_classes = len(classes)
    
    # Binarize labels
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    colors = {'L1': '#e74c3c', 'L2': '#2ecc71', 'ElasticNet': '#3498db'}
    linestyles = {'L1': '-', 'L2': '--', 'ElasticNet': '-.'}
    
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]
    
    for class_idx in range(n_classes):
        ax = axes[class_idx]
        
        for variant_name, result in results.items():
            model = result['model']
            
            if hasattr(X_test, 'toarray'):
                y_score = model.predict_proba(X_test)
            else:
                y_score = model.predict_proba(X_test)
            
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score[:, class_idx])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors[variant_name], 
                    linestyle=linestyles[variant_name],
                    label=f'{variant_name} (AUC={roc_auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'ROC Curve: {classes[class_idx]}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Logistic Regression ROC Curves (L1 vs L2 vs ElasticNet)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS_DIR / 'lr_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: lr_roc_curves.png")


def lr_vs_nb_explanation() -> str:
    """
    250-word explanation of why LR handles correlated features better than NB.
    """
    explanation = """
WHY LOGISTIC REGRESSION HANDLES CORRELATED FEATURES BETTER THAN NAIVE BAYES (250 words)
========================================================================================

Naive Bayes assumes conditional independence: P(x₁, x₂ | y) = P(x₁ | y) × P(x₂ | y). When 
features are correlated (e.g., "breaking" and "news" co-occur frequently), NB treats each 
independently, effectively double-counting their combined evidence. If both words strongly 
indicate "Fake" news, NB multiplies their individual probabilities, producing overconfident 
predictions. This independence violation causes NB to underestimate the variance of the 
posterior distribution, leading to poorly calibrated probabilities (Domingos & Pazzani, 1997).

Logistic Regression, by contrast, learns a single weight vector w where P(y|x) = σ(wᵀx). 
During optimization, LR jointly adjusts all feature weights to maximize conditional 
likelihood. When features x₁ and x₂ are correlated, gradient descent naturally distributes 
the predictive signal between them — if both convey redundant information, their individual 
weights decrease proportionally. This automatic regularization prevents double-counting.

L2 regularization (ridge) further addresses multicollinearity by adding a penalty term 
λ||w||² to the loss function. For perfectly correlated features, L2 distributes weight 
equally between them rather than assigning arbitrary large weights. L1 regularization 
(lasso) handles correlation more aggressively by driving redundant feature weights to 
exactly zero, performing implicit feature selection.

In misinformation detection, TF-IDF features exhibit significant correlation: "breaking 
news," "unnamed sources," and "exclusively learned" frequently co-occur in fake news. 
NB would overweight these patterns, while LR's joint optimization and regularization produce 
more robust, well-calibrated predictions. Empirically, LR's discriminative training — 
directly modeling P(y|x) rather than the generative P(x|y) — yields better decision 
boundaries when the independence assumption is violated (Ng & Jordan, 2002).
"""
    return explanation


# ══════════════════════════════════════════════════════════
#  Run Task 5.2
# ══════════════════════════════════════════════════════════
def run_logistic_regression(df: pd.DataFrame, tfidf_result: dict) -> dict:
    """Execute Task 5.2: Logistic Regression with regularization."""
    print("\n" + "=" * 60)
    print("TASK 5.2: LOGISTIC REGRESSION")
    print("=" * 60)
    
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    
    tfidf_matrix = tfidf_result['matrix']
    feature_names = tfidf_result['feature_names']
    
    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train all variants
    results = train_lr_variants(X_train, X_test, y_train, y_test, feature_names, le)
    
    # Extract top features from L2
    top_features = extract_top_features(results['L2']['model'], feature_names, le)
    
    # Plot ROC curves
    plot_roc_curves(results, X_test, y_test, le)
    
    # Confusion matrix for best variant
    best_variant = max(results, key=lambda k: results[k]['f1'])
    cm = confusion_matrix(y_test, results[best_variant]['y_pred'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f'LR ({best_variant}) Confusion Matrix', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.savefig(REPORTS_DIR / 'lr_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print explanation
    explanation = lr_vs_nb_explanation()
    print(explanation)
    
    return {
        'variants': results,
        'top_features': top_features,
        'best_variant': best_variant,
        'label_encoder': le,
    }


if __name__ == "__main__":
    print("Logistic Regression module loaded. Run via pipeline.py")
