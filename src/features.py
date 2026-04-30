"""
Task 3.5 — Feature Representation [15 Marks]

BoW [3 Marks]: Matrix dimensions, sparsity, top 30 terms per class, mathematical limitation
TF-IDF [4 Marks]: Standard, Smooth IDF, Sublinear TF variants. Top 15 discriminative terms.
                   Cosine similarity retrieval system tested on 10 examples.
Word2Vec [8 Marks]: CBOW and Skip-gram (window=5, dim=200, min_count=3).
                     Domain word pair similarity, top-5 neighbors, t-SNE visualization.
                     Classification F1 comparison: TF-IDF vs Word2Vec vs concatenated.
"""

import time
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "figures"
DATA_DIR = Path(__file__).parent.parent / "data"


# ══════════════════════════════════════════════════════════
#  3.5.1 — Bag of Words [3 Marks]
# ══════════════════════════════════════════════════════════
def build_bow(df: pd.DataFrame, max_features: int = 10000) -> dict:
    """
    Build Bag-of-Words representation.
    
    Reports: matrix dimensions, sparsity, top 30 terms per class.
    """
    print(f"\n  BoW REPRESENTATION")
    print(f"  {'─' * 40}")
    
    # Join tokens back to strings for CountVectorizer
    texts = df['tokens_normalized'].apply(lambda t: ' '.join(t))
    
    vectorizer = CountVectorizer(max_features=max_features)
    bow_matrix = vectorizer.fit_transform(texts)
    
    # Matrix dimensions and sparsity
    n_docs, n_features = bow_matrix.shape
    total_elements = n_docs * n_features
    non_zero = bow_matrix.nnz
    sparsity = (1 - non_zero / total_elements) * 100
    
    print(f"  Matrix dimensions: {n_docs} × {n_features}")
    print(f"  Non-zero elements: {non_zero:,}")
    print(f"  Sparsity: {sparsity:.2f}%")
    
    # Top 30 terms per class
    feature_names = vectorizer.get_feature_names_out()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for idx, label in enumerate(sorted(df['label'].unique())):
        class_mask = df['label'] == label
        class_bow = bow_matrix[class_mask.values]
        term_frequencies = np.array(class_bow.sum(axis=0)).flatten()
        top_indices = term_frequencies.argsort()[-30:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        top_freqs = [term_frequencies[i] for i in top_indices]
        
        axes[idx].barh(range(30), top_freqs[::-1], color=plt.cm.Set2(idx))
        axes[idx].set_yticks(range(30))
        axes[idx].set_yticklabels(top_terms[::-1], fontsize=8)
        axes[idx].set_title(f'Top 30 Terms: {label}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Frequency')
    
    plt.suptitle('Bag-of-Words: Top 30 Terms per Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS_DIR / 'bow_top30_terms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: bow_top30_terms.png")
    
    # Mathematical limitation of BoW
    bow_limitation = """
  MATHEMATICAL LIMITATION OF BoW FOR MISINFORMATION DETECTION:
  ─────────────────────────────────────────────────────────────
  BoW represents each document d as a vector x ∈ ℝ^|V| where x_i = count(w_i, d).
  
  1. LOSS OF WORD ORDER: BoW treats documents as unordered sets of words.
     For misinformation detection, word order is critical:
     "Pakistan did NOT deny the allegations" vs "Pakistan denied the allegations"
     produce identical BoW vectors (same word counts), yet carry opposite meanings.
     Formally: BoW(d) = BoW(π(d)) for any permutation π, losing sequential semantics.
  
  2. NO SEMANTIC SIMILARITY: BoW uses one-hot word representations where
     cos(e_i, e_j) = 0 for i ≠ j. Semantically related words like "fabricated"
     and "manufactured" have zero similarity, preventing generalization across 
     paraphrased misinformation.
  
  3. HIGH DIMENSIONALITY & SPARSITY: With |V| = {n_features}, the feature space is
     extremely high-dimensional. Most entries are zero (sparsity = {sparsity:.1f}%),
     leading to the curse of dimensionality. Distance metrics become unreliable
     in such sparse spaces (Aggarwal et al., 2001).
  
  4. NO CONTEXT: The same word has identical representation regardless of context.
     "Breaking news" in real journalism vs. "breaking news" as clickbait in fake 
     articles receive identical BoW treatment despite vastly different implications.
  
  These limitations motivate TF-IDF (which addresses term importance) and Word2Vec
  (which captures semantic relationships) as superior alternatives.
  """
    print(bow_limitation.format(n_features=n_features, sparsity=sparsity))
    
    return {
        'vectorizer': vectorizer,
        'matrix': bow_matrix,
        'dimensions': (n_docs, n_features),
        'sparsity': sparsity,
        'feature_names': feature_names,
    }


# ══════════════════════════════════════════════════════════
#  3.5.2 — TF-IDF [4 Marks]
# ══════════════════════════════════════════════════════════
def build_tfidf_variants(df: pd.DataFrame, max_features: int = 10000) -> dict:
    """
    Implement three TF-IDF variants:
    1. Standard TF-IDF
    2. Smooth IDF (add 1 to document frequency)
    3. Sublinear TF (replace tf with 1 + log(tf))
    
    Compare top 15 discriminative terms per class.
    """
    print(f"\n  TF-IDF VARIANTS")
    print(f"  {'─' * 40}")
    
    texts = df['tokens_normalized'].apply(lambda t: ' '.join(t))
    
    variants = {
        'Standard': TfidfVectorizer(max_features=max_features, smooth_idf=False, sublinear_tf=False),
        'Smooth IDF': TfidfVectorizer(max_features=max_features, smooth_idf=True, sublinear_tf=False),
        'Sublinear TF': TfidfVectorizer(max_features=max_features, smooth_idf=True, sublinear_tf=True),
    }
    
    results = {}
    labels = sorted(df['label'].unique())
    
    for variant_name, vectorizer in variants.items():
        matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        print(f"\n  {variant_name} TF-IDF:")
        print(f"    Shape: {matrix.shape}, Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.2f}%")
        
        # Top 15 discriminative terms per class
        print(f"\n    Top 15 Discriminative Terms:")
        class_top_terms = {}
        for label in labels:
            class_mask = (df['label'] == label).values
            class_mean = np.array(matrix[class_mask].mean(axis=0)).flatten()
            other_mean = np.array(matrix[~class_mask].mean(axis=0)).flatten()
            
            # Discriminative score: class mean - other mean
            discriminative = class_mean - other_mean
            top_indices = discriminative.argsort()[-15:][::-1]
            top_terms = [(feature_names[i], discriminative[i]) for i in top_indices]
            class_top_terms[label] = top_terms
            
            print(f"    {label}: {', '.join([t[0] for t in top_terms[:5]])} ...")
        
        results[variant_name] = {
            'vectorizer': vectorizer,
            'matrix': matrix,
            'feature_names': feature_names,
            'discriminative_terms': class_top_terms,
        }
    
    # Save the Sublinear TF variant as the primary TF-IDF (best practice)
    return results


def build_cosine_retrieval(tfidf_result: dict, df: pd.DataFrame, 
                            n_test: int = 10) -> dict:
    """
    Build a cosine similarity retrieval system using TF-IDF.
    Tests on n_test example queries.
    """
    print(f"\n  COSINE SIMILARITY RETRIEVAL SYSTEM")
    print(f"  {'─' * 40}")
    
    matrix = tfidf_result['matrix']
    
    # Use first n_test documents as queries
    np.random.seed(42)
    query_indices = np.random.choice(len(df), size=n_test, replace=False)
    
    results = []
    for i, q_idx in enumerate(query_indices):
        query_vec = matrix[q_idx]
        
        # Compute cosine similarity with all documents
        similarities = cosine_similarity(query_vec, matrix).flatten()
        similarities[q_idx] = -1  # Exclude self
        
        # Top 3 similar documents
        top_k = 3
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        query_text = df.iloc[q_idx]['text_clean'][:80] if 'text_clean' in df.columns else df.iloc[q_idx]['text'][:80]
        query_label = df.iloc[q_idx]['label']
        
        result = {
            'query_idx': int(q_idx),
            'query_label': query_label,
            'query_text': query_text,
            'matches': []
        }
        
        print(f"\n  Query {i+1} [{query_label}]: \"{query_text}...\"")
        for rank, idx in enumerate(top_indices):
            sim = similarities[idx]
            match_label = df.iloc[idx]['label']
            match_text = df.iloc[idx]['text_clean'][:60] if 'text_clean' in df.columns else df.iloc[idx]['text'][:60]
            result['matches'].append({
                'idx': int(idx),
                'label': match_label,
                'similarity': float(sim),
                'text': match_text,
            })
            print(f"    #{rank+1} [{match_label}] (sim={sim:.4f}): \"{match_text}...\"")
        
        results.append(result)
    
    return results


# ══════════════════════════════════════════════════════════
#  3.5.3 — Word2Vec [8 Marks]
# ══════════════════════════════════════════════════════════
def train_word2vec(df: pd.DataFrame, window: int = 5, vector_size: int = 200,
                   min_count: int = 3) -> dict:
    """
    Train CBOW and Skip-gram Word2Vec models.
    
    Reports:
    - Similarity scores for domain word pairs
    - Top-5 neighbors for key terms
    - t-SNE visualization
    """
    from gensim.models import Word2Vec
    
    print(f"\n  WORD2VEC MODELS")
    print(f"  {'─' * 40}")
    
    sentences = df['tokens_normalized'].tolist()
    
    # Train CBOW
    print(f"\n  Training CBOW (window={window}, dim={vector_size}, min_count={min_count})...")
    start = time.time()
    cbow_model = Word2Vec(
        sentences, vector_size=vector_size, window=window,
        min_count=min_count, sg=0, workers=4, epochs=10
    )
    cbow_time = time.time() - start
    print(f"    Trained in {cbow_time:.2f}s, vocabulary: {len(cbow_model.wv)} words")
    
    # Train Skip-gram
    print(f"\n  Training Skip-gram (window={window}, dim={vector_size}, min_count={min_count})...")
    start = time.time()
    skipgram_model = Word2Vec(
        sentences, vector_size=vector_size, window=window,
        min_count=min_count, sg=1, workers=4, epochs=10
    )
    skipgram_time = time.time() - start
    print(f"    Trained in {skipgram_time:.2f}s, vocabulary: {len(skipgram_model.wv)} words")
    
    # Domain word pairs similarity
    word_pairs = [
        ('fake', 'false'), ('real', 'true'), ('news', 'report'),
        ('claim', 'allegation'), ('fact', 'evidence'), ('misleading', 'deceptive'),
        ('government', 'administration'), ('election', 'vote'),
        ('propaganda', 'manipulation'), ('investigation', 'probe'),
    ]
    
    print(f"\n  DOMAIN WORD PAIR SIMILARITIES:")
    print(f"  {'Word Pair':<30} {'CBOW':>10} {'Skip-gram':>10}")
    print(f"  {'─' * 50}")
    
    for w1, w2 in word_pairs:
        try:
            cbow_sim = cbow_model.wv.similarity(w1, w2)
            sg_sim = skipgram_model.wv.similarity(w1, w2)
            print(f"  ({w1}, {w2}){' ' * (26 - len(w1) - len(w2))} {cbow_sim:>10.4f} {sg_sim:>10.4f}")
        except KeyError as e:
            print(f"  ({w1}, {w2}){' ' * (26 - len(w1) - len(w2))} {'N/A':>10} {'N/A':>10}  [{e}]")
    
    # Top-5 neighbors for key terms
    key_terms = ['fake', 'real', 'news', 'claim', 'evidence', 'election', 'government', 'truth']
    
    print(f"\n  TOP-5 NEIGHBORS (Skip-gram):")
    for term in key_terms:
        try:
            neighbors = skipgram_model.wv.most_similar(term, topn=5)
            neighbor_str = ', '.join([f"{w}({s:.3f})" for w, s in neighbors])
            print(f"    {term}: {neighbor_str}")
        except KeyError:
            print(f"    {term}: [not in vocabulary]")
    
    # Save models
    MODELS_DIR = DATA_DIR / "embeddings"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cbow_path = MODELS_DIR / "word2vec_cbow.model"
    skipgram_path = MODELS_DIR / "word2vec_skipgram.model"
    cbow_model.save(str(cbow_path))
    skipgram_model.save(str(skipgram_path))
    print(f"\n  Models saved to {MODELS_DIR}")
    
    return {
        'cbow': cbow_model,
        'skipgram': skipgram_model,
        'cbow_time': cbow_time,
        'skipgram_time': skipgram_time,
    }


def visualize_tsne(model, title: str = "Word2Vec t-SNE", n_words: int = 200,
                    perplexity: int = 30) -> None:
    """
    Visualize Word2Vec embeddings using t-SNE.
    """
    from sklearn.manifold import TSNE
    
    print(f"\n  Generating t-SNE visualization (perplexity={perplexity})...")
    
    words = list(model.wv.index_to_key[:n_words])
    vectors = np.array([model.wv[w] for w in words])
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(16, 12))
    plt.scatter(coords[:, 0], coords[:, 1], c='steelblue', alpha=0.6, s=20)
    
    # Label key terms
    key_terms = ['fake', 'real', 'news', 'true', 'false', 'claim', 'evidence',
                 'government', 'election', 'report', 'source', 'propaganda']
    
    for i, word in enumerate(words):
        if word in key_terms:
            plt.annotate(word, (coords[i, 0], coords[i, 1]),
                        fontsize=11, fontweight='bold', color='red',
                        arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
        elif i < 50:
            plt.annotate(word, (coords[i, 0], coords[i, 1]),
                        fontsize=7, alpha=0.7)
    
    plt.title(f'{title} (perplexity={perplexity}, top {n_words} words)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    
    filename = title.lower().replace(' ', '_').replace('-', '_') + '.png'
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def get_document_embedding(tokens: list, model) -> np.ndarray:
    """
    Get document embedding by averaging word vectors.
    
    Args:
        tokens: List of tokens
        model: Trained Word2Vec model
        
    Returns:
        Average word vector (vector_size,) or zero vector if no words found
    """
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.wv.vector_size)


def build_w2v_features(df: pd.DataFrame, model) -> np.ndarray:
    """Build document-level Word2Vec features by averaging word vectors."""
    features = np.array([
        get_document_embedding(tokens, model) 
        for tokens in df['tokens_normalized']
    ])
    return features


# ══════════════════════════════════════════════════════════
#  Classification F1 Comparison
# ══════════════════════════════════════════════════════════
def compare_feature_f1(df: pd.DataFrame, tfidf_matrix, w2v_features: np.ndarray) -> dict:
    """
    Compare classification F1 across:
    1. TF-IDF only
    2. Word2Vec only
    3. Concatenated (TF-IDF + Word2Vec)
    
    Uses Logistic Regression as a standard classifier.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, classification_report
    
    print(f"\n  FEATURE COMPARISON: CLASSIFICATION F1")
    print(f"  {'─' * 50}")
    
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    
    # Prepare feature sets
    tfidf_dense = tfidf_matrix.toarray() if hasattr(tfidf_matrix, 'toarray') else tfidf_matrix
    
    feature_sets = {
        'TF-IDF Only': tfidf_dense,
        'Word2Vec Only': w2v_features,
        'TF-IDF + Word2Vec': np.hstack([tfidf_dense, w2v_features]),
    }
    
    results = {}
    
    for name, X in feature_sets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        results[name] = {
            'f1_weighted': f1_weighted,
            'f1_per_class': {le.inverse_transform([i])[0]: f1 
                            for i, f1 in enumerate(f1_per_class)},
            'feature_dim': X.shape[1],
        }
        
        print(f"\n  {name} (dim={X.shape[1]}):")
        print(f"    F1 (weighted): {f1_weighted:.4f}")
        for cls, f1 in results[name]['f1_per_class'].items():
            print(f"    F1 ({cls}): {f1:.4f}")
    
    # Summary comparison
    print(f"\n  SUMMARY:")
    print(f"  {'Feature Set':<25} {'Dimension':>10} {'F1-Weighted':>12}")
    print(f"  {'─' * 47}")
    for name, r in results.items():
        print(f"  {name:<25} {r['feature_dim']:>10} {r['f1_weighted']:>12.4f}")
    
    return results


# ══════════════════════════════════════════════════════════
#  Run Task 3.5
# ══════════════════════════════════════════════════════════
def run_features(df: pd.DataFrame) -> dict:
    """Execute Task 3.5: Feature representation (BoW, TF-IDF, Word2Vec)."""
    print("\n" + "=" * 60)
    print("TASK 3.5: FEATURE REPRESENTATION")
    print("=" * 60)
    
    # 3.5.1 BoW
    bow_result = build_bow(df)
    
    # 3.5.2 TF-IDF
    tfidf_results = build_tfidf_variants(df)
    
    # Cosine similarity retrieval (use Sublinear TF variant)
    retrieval_results = build_cosine_retrieval(tfidf_results['Sublinear TF'], df)
    
    # 3.5.3 Word2Vec
    w2v_results = train_word2vec(df)
    
    # t-SNE visualization
    visualize_tsne(w2v_results['skipgram'], title="Word2Vec Skip-gram t-SNE")
    visualize_tsne(w2v_results['cbow'], title="Word2Vec CBOW t-SNE")
    
    # Classification F1 comparison
    w2v_features = build_w2v_features(df, w2v_results['skipgram'])
    tfidf_matrix = tfidf_results['Sublinear TF']['matrix']
    f1_comparison = compare_feature_f1(df, tfidf_matrix, w2v_features)
    
    return {
        'bow': bow_result,
        'tfidf': tfidf_results,
        'retrieval': retrieval_results,
        'word2vec': w2v_results,
        'f1_comparison': f1_comparison,
    }


if __name__ == "__main__":
    print("Features module loaded. Run via pipeline.py")
