"""
Task 4 — N-Gram Language Models [10 Marks]

- Unigram, bigram, trigram LMs for Fake and Real classes separately
- Unique n-gram counts and top 20 most probable n-grams per class
- Kneser-Ney smoothing from scratch for trigram model
- Perplexity-based classification on 100 held-out samples
- Accuracy, precision, recall, F1 comparison against Naive Bayes
"""

import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "figures"


# ══════════════════════════════════════════════════════════
#  N-Gram Language Model
# ══════════════════════════════════════════════════════════
class NgramLanguageModel:
    """
    N-gram language model with optional Kneser-Ney smoothing.
    """
    
    def __init__(self, n: int = 3, smoothing: str = 'none', discount: float = 0.75):
        """
        Args:
            n: Order of the n-gram model (1=unigram, 2=bigram, 3=trigram)
            smoothing: 'none', 'laplace', or 'kneser_ney'
            discount: Discount parameter for Kneser-Ney (default: 0.75)
        """
        self.n = n
        self.smoothing = smoothing
        self.discount = discount
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()
        self.total_tokens = 0
        
        # For Kneser-Ney
        self.continuation_counts = Counter()  # How many unique contexts each word follows
        self.unique_continuations = Counter()  # How many unique words follow each context
        self.total_bigram_types = 0
    
    def _get_ngrams(self, tokens: list) -> list:
        """Extract n-grams from token list with BOS/EOS markers."""
        padded = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        ngrams = []
        for i in range(len(padded) - self.n + 1):
            ngram = tuple(padded[i:i + self.n])
            ngrams.append(ngram)
        return ngrams
    
    def fit(self, documents: list):
        """
        Train the language model on a list of token lists.
        
        Args:
            documents: List of lists of tokens
        """
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()
        self.total_tokens = 0
        
        for tokens in documents:
            self.vocab.update(tokens)
            self.total_tokens += len(tokens)
            
            ngrams = self._get_ngrams(tokens)
            for ngram in ngrams:
                self.ngram_counts[ngram] += 1
                context = ngram[:-1]
                self.context_counts[context] += 1
        
        self.vocab.add('<s>')
        self.vocab.add('</s>')
        
        # Build Kneser-Ney statistics
        if self.smoothing == 'kneser_ney':
            self._build_kneser_ney_stats(documents)
    
    def _build_kneser_ney_stats(self, documents: list):
        """Build continuation and unique continuation counts for Kneser-Ney."""
        # For each word type, count how many distinct contexts it follows
        word_contexts = defaultdict(set)
        context_words = defaultdict(set)
        
        for tokens in documents:
            padded = ['<s>'] * (self.n - 1) + tokens + ['</s>']
            for i in range(len(padded) - 1):
                context = padded[i]
                word = padded[i + 1]
                word_contexts[word].add(context)
                context_words[context].add(word)
        
        # Continuation count: number of distinct contexts each word follows
        for word, contexts in word_contexts.items():
            self.continuation_counts[word] = len(contexts)
        
        # Unique continuations: number of distinct words following each context
        for context, words in context_words.items():
            self.unique_continuations[context] = len(words)
        
        # Total distinct bigram types
        self.total_bigram_types = sum(len(contexts) for contexts in word_contexts.values())
    
    def log_probability(self, ngram: tuple) -> float:
        """
        Compute log probability of an n-gram.
        
        Returns:
            Log probability (base e)
        """
        if self.n == 1:
            # Unigram model
            word = ngram[0]
            count = self.ngram_counts.get(ngram, 0)
            if self.smoothing == 'laplace':
                return math.log((count + 1) / (self.total_tokens + len(self.vocab)))
            elif count > 0:
                return math.log(count / self.total_tokens)
            else:
                return math.log(1e-10)  # Smoothing floor
        
        context = ngram[:-1]
        word = ngram[-1]
        context_count = self.context_counts.get(context, 0)
        ngram_count = self.ngram_counts.get(ngram, 0)
        
        if self.smoothing == 'laplace':
            return math.log((ngram_count + 1) / (context_count + len(self.vocab)))
        
        elif self.smoothing == 'kneser_ney':
            return self._kneser_ney_log_prob(ngram)
        
        else:
            # No smoothing
            if context_count == 0 or ngram_count == 0:
                return math.log(1e-10)
            return math.log(ngram_count / context_count)
    
    def _kneser_ney_log_prob(self, ngram: tuple) -> float:
        """
        Compute Kneser-Ney smoothed log probability.
        
        P_KN(w_i | w_{i-1}) = max(c(w_{i-1}, w_i) - d, 0) / c(w_{i-1})
                             + λ(w_{i-1}) * P_continuation(w_i)
        
        where:
            λ(w_{i-1}) = d * |{w : c(w_{i-1}, w) > 0}| / c(w_{i-1})
            P_continuation(w_i) = |{w : c(w, w_i) > 0}| / Σ_w' |{w : c(w, w') > 0}|
        """
        context = ngram[:-1]
        word = ngram[-1]
        
        context_count = self.context_counts.get(context, 0)
        ngram_count = self.ngram_counts.get(ngram, 0)
        
        if context_count == 0:
            # Back off to continuation probability
            cont = self.continuation_counts.get(word, 0)
            if self.total_bigram_types > 0:
                prob = max(cont, 1e-10) / self.total_bigram_types
            else:
                prob = 1e-10
            return math.log(max(prob, 1e-10))
        
        # First term: discounted probability
        first_term = max(ngram_count - self.discount, 0) / context_count
        
        # Lambda (interpolation weight)
        context_key = context[-1] if len(context) == 1 else str(context)
        unique_cont = self.unique_continuations.get(context_key if isinstance(context_key, str) else context[-1], 1)
        lambda_weight = (self.discount * unique_cont) / context_count
        
        # Continuation probability
        cont = self.continuation_counts.get(word, 0)
        if self.total_bigram_types > 0:
            p_continuation = max(cont, 1e-10) / self.total_bigram_types
        else:
            p_continuation = 1e-10
        
        prob = first_term + lambda_weight * p_continuation
        return math.log(max(prob, 1e-10))
    
    def perplexity(self, tokens: list) -> float:
        """
        Compute perplexity of a token sequence.
        
        PP(W) = exp(-1/N * Σ log P(w_i | context))
        """
        ngrams = self._get_ngrams(tokens)
        if not ngrams:
            return float('inf')
        
        total_log_prob = 0
        for ngram in ngrams:
            total_log_prob += self.log_probability(ngram)
        
        avg_log_prob = total_log_prob / len(ngrams)
        return math.exp(-avg_log_prob)
    
    def top_ngrams(self, k: int = 20) -> list:
        """Return top-k most probable n-grams."""
        sorted_ngrams = sorted(self.ngram_counts.items(), key=lambda x: x[1], reverse=True)
        total = sum(self.ngram_counts.values())
        return [(ngram, count, count/total) for ngram, count, in sorted_ngrams[:k]]


# ══════════════════════════════════════════════════════════
#  Classification using Perplexity
# ══════════════════════════════════════════════════════════
def classify_with_perplexity(text_tokens: list, class_models: dict) -> str:
    """
    Classify a document by finding the class whose LM assigns lowest perplexity.
    
    Args:
        text_tokens: Tokenized document
        class_models: Dict mapping class label → NgramLanguageModel
        
    Returns:
        Predicted class label
    """
    best_class = None
    best_perplexity = float('inf')
    
    for label, model in class_models.items():
        pp = model.perplexity(text_tokens)
        if pp < best_perplexity:
            best_perplexity = pp
            best_class = label
    
    return best_class


# ══════════════════════════════════════════════════════════
#  Run Task 4
# ══════════════════════════════════════════════════════════
def run_ngram_models(df: pd.DataFrame) -> dict:
    """Execute Task 4: N-Gram Language Models."""
    print("\n" + "=" * 60)
    print("TASK 4: N-GRAM LANGUAGE MODELS")
    print("=" * 60)
    
    # Build models for Real and Fake classes only (as specified)
    class_labels = ['Real', 'Fake']
    
    # Split data: use 100 held-out samples for evaluation
    from sklearn.model_selection import train_test_split
    
    # Filter to Real and Fake only for LM classification
    df_lm = df[df['label'].isin(class_labels)].copy().reset_index(drop=True)
    train_df, test_df = train_test_split(df_lm, test_size=100, random_state=42, stratify=df_lm['label'])
    
    results = {}
    
    for n, name in [(1, 'Unigram'), (2, 'Bigram'), (3, 'Trigram')]:
        print(f"\n  {'═' * 40}")
        print(f"  {name} Models (n={n})")
        print(f"  {'═' * 40}")
        
        smoothing = 'kneser_ney' if n == 3 else 'laplace'
        class_models = {}
        
        for label in class_labels:
            class_docs = train_df[train_df['label'] == label]['tokens_normalized'].tolist()
            
            model = NgramLanguageModel(n=n, smoothing=smoothing)
            model.fit(class_docs)
            class_models[label] = model
            
            # Report unique n-gram counts
            unique_count = len(model.ngram_counts)
            print(f"\n    {label} class:")
            print(f"      Unique {n}-grams: {unique_count:,}")
            
            # Top 20 most probable
            top = model.top_ngrams(20)
            print(f"      Top 20 {name.lower()}s:")
            for ngram, count, prob in top[:10]:
                ngram_str = ' '.join(ngram)
                print(f"        {ngram_str:<30} count={count:>5} p={prob:.6f}")
            print(f"        ... ({len(top) - 10} more)")
        
        # Classify held-out samples
        y_true = test_df['label'].tolist()
        y_pred = []
        
        for tokens in test_df['tokens_normalized']:
            pred = classify_with_perplexity(tokens, class_models)
            y_pred.append(pred)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n    CLASSIFICATION RESULTS ({name}, {len(test_df)} held-out samples):")
        print(f"      Accuracy:  {acc:.4f}")
        print(f"      Precision: {prec:.4f}")
        print(f"      Recall:    {rec:.4f}")
        print(f"      F1:        {f1:.4f}")
        
        results[name] = {
            'models': class_models,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'smoothing': smoothing,
        }
    
    # Justify Kneser-Ney over Laplace
    print(f"\n  KNESER-NEY vs LAPLACE JUSTIFICATION:")
    print(f"  {'─' * 50}")
    print(f"  Kneser-Ney smoothing is superior to Laplace (add-one) for several reasons:")
    print(f"  ")
    print(f"  1. MATHEMATICAL FOUNDATION: Laplace smoothing adds 1 to all n-gram counts,")
    print(f"     which steals too much probability mass from observed n-grams and distributes")
    print(f"     it uniformly to unseen n-grams. For vocabulary size V, Laplace effectively")
    print(f"     reduces observed probabilities by a factor of V/(N+V), which is severe for")
    print(f"     large vocabularies.")
    print(f"  ")
    print(f"  2. CONTINUATION PROBABILITY: Kneser-Ney uses continuation counts — how many")
    print(f"     distinct contexts a word appears in — rather than raw frequency. This means")
    print(f"     a word like 'Francisco' (which almost always follows 'San') gets low")
    print(f"     continuation probability as a unigram backoff, while versatile words get")
    print(f"     higher probability. This is linguistically motivated.")
    print(f"  ")
    print(f"  3. DISCOUNTING: Instead of adding to all counts, KN subtracts a fixed discount d")
    print(f"     (typically 0.75) from observed counts. This preserves the relative ordering of")
    print(f"     observed n-grams while redistributing a calibrated amount of probability mass.")
    print(f"  ")
    print(f"  4. EMPIRICAL SUPERIORITY: Chen & Goodman (1999) demonstrated that modified")
    print(f"     Kneser-Ney consistently achieves lower perplexity than any other smoothing")
    print(f"     method across multiple corpora and n-gram orders.")
    
    return results


if __name__ == "__main__":
    print("N-gram models module loaded. Run via pipeline.py")
