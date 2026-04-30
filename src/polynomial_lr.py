"""
Task 5.3 — Polynomial Features + Logistic Regression [8 Marks]

- PCA reduction of TF-IDF to 2D
- Polynomial features degree {1, 2, 3} with decision boundary plots
- Train/test accuracy and F1 per degree
- Feature space size computation for degree-2 on full TF-IDF
- Alternative non-linear approach proposal
"""

import numpy as np
import pandas as pd
from pathlib import Path
from math import comb

from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "figures"


def run_polynomial_lr(df: pd.DataFrame, tfidf_result: dict) -> dict:
    """Execute Task 5.3: Polynomial Features + LR."""
    print("\n" + "=" * 60)
    print("TASK 5.3: POLYNOMIAL FEATURES + LOGISTIC REGRESSION")
    print("=" * 60)
    
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    
    tfidf_matrix = tfidf_result['matrix']
    
    # PCA to 2D
    print(f"\n  Reducing TF-IDF to 2D with PCA...")
    if hasattr(tfidf_matrix, 'toarray'):
        tfidf_dense = tfidf_matrix.toarray()
    else:
        tfidf_dense = np.array(tfidf_matrix)
    
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(tfidf_dense)
    
    print(f"    Original shape: {tfidf_dense.shape}")
    print(f"    PCA 2D shape: {X_2d.shape}")
    print(f"    Explained variance: {pca.explained_variance_ratio_}")
    print(f"    Total variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_2d, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Test degrees 1, 2, 3
    degrees = [1, 2, 3]
    results = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    print(f"\n  {'Degree':>8} {'Train Acc':>10} {'Test Acc':>10} {'F1':>10} {'Features':>10}")
    print(f"  {'─' * 48}")
    
    for idx, degree in enumerate(degrees):
        # Generate polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Train LR
        clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
        clf.fit(X_train_poly, y_train)
        
        y_pred_train = clf.predict(X_train_poly)
        y_pred_test = clf.predict(X_test_poly)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        
        results[degree] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'f1': f1,
            'n_features': X_train_poly.shape[1],
            'model': clf,
            'poly': poly,
        }
        
        print(f"  {degree:>8} {train_acc:>10.4f} {test_acc:>10.4f} {f1:>10.4f} {X_train_poly.shape[1]:>10}")
        
        # Plot decision boundaries
        ax = axes[idx]
        
        # Create mesh grid
        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_poly = poly.transform(grid_points)
        Z = clf.predict(grid_poly)
        Z = Z.reshape(xx.shape)
        
        # Plot decision regions
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set2)
        ax.contour(xx, yy, Z, alpha=0.5, colors='black', linewidths=0.5)
        
        # Scatter plot
        for i, label in enumerate(le.classes_):
            mask = y == i
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[i],
                      label=label, alpha=0.4, s=10, edgecolors='none')
        
        ax.set_title(f'Degree {degree}\nTrain: {train_acc:.3f}, Test: {test_acc:.3f}, F1: {f1:.3f}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(fontsize=8, loc='best')
    
    plt.suptitle('Polynomial Features + LR: Decision Boundaries (PCA 2D)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS_DIR / 'polynomial_decision_boundaries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: polynomial_decision_boundaries.png")
    
    # Feature space size for degree-2 on full TF-IDF
    n_original = tfidf_dense.shape[1]
    # For degree 2: C(n + d, d) - 1 (without bias) = n + C(n, 2)
    degree2_size = n_original + comb(n_original, 2)  # Linear + interaction terms
    # More precisely: PolynomialFeatures(degree=2) gives: n + n*(n+1)/2 features
    degree2_full = comb(n_original + 2, 2) - 1  # All terms up to degree 2 minus bias
    
    print(f"\n  FEATURE SPACE SIZE FOR DEGREE-2 ON FULL TF-IDF:")
    print(f"  {'─' * 50}")
    print(f"  Original TF-IDF dimensions: {n_original}")
    print(f"  Degree-2 polynomial features: {degree2_full:,}")
    print(f"  ")
    print(f"  Formula: C(n + d, d) - 1 = C({n_original} + 2, 2) - 1 = {degree2_full:,}")
    print(f"  This includes {n_original} linear terms + {comb(n_original, 2):,} interaction terms")
    print(f"  + {n_original} squared terms")
    print(f"  ")
    print(f"  This is computationally infeasible for training — a {degree2_full:,}-dimensional")
    print(f"  feature space would require enormous memory and computation time.")
    print(f"  This is why we reduce to 2D with PCA before applying polynomial features.")
    
    # Alternative non-linear approach
    print(f"\n  ALTERNATIVE NON-LINEAR APPROACH:")
    print(f"  {'─' * 50}")
    print(f"  KERNEL SVM (Support Vector Machine with RBF Kernel)")
    print(f"  ")
    print(f"  Instead of explicitly computing polynomial features (which explodes the")
    print(f"  feature space), Kernel SVM uses the 'kernel trick' to implicitly map data")
    print(f"  to a higher-dimensional space. The RBF (Radial Basis Function) kernel")
    print(f"  K(x, x') = exp(-γ||x - x'||²) maps to an infinite-dimensional space")
    print(f"  without ever computing the transformation explicitly.")
    print(f"  ")
    print(f"  Advantages over polynomial features + LR:")
    print(f"  1. Handles non-linear decision boundaries without dimensionality explosion")
    print(f"  2. The kernel matrix is O(n²) regardless of feature space dimension")
    print(f"  3. RBF kernel is universal — it can approximate any continuous function")
    print(f"  4. Built-in regularization through the C and γ hyperparameters")
    print(f"  5. Well-suited for text classification (Joachims, 1998)")
    print(f"  ")
    print(f"  This technique is covered in our course under 'Kernel Methods' and")
    print(f"  'Support Vector Machines' — an established approach for non-linear")
    print(f"  classification that avoids the curse of dimensionality inherent in")
    print(f"  explicit polynomial feature expansion.")
    
    return results


if __name__ == "__main__":
    print("Polynomial LR module loaded. Run via pipeline.py")
