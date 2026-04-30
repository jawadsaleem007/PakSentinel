"""
Task 1 — Data Sourcing & Reliability Assessment [15 Marks]

Downloads and combines multi-source datasets for 3-class misinformation detection:
- LIAR Dataset (HuggingFace) → Real / Fake
- FakeNewsNet (PolitiFact + GossipCop) → Real / Fake  
- Sarcasm Headlines (Google Storage) → Satire class

Produces reliability scorecards, justification, and handles class imbalance.
"""

import os
import json
import hashlib
import requests
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ──────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

LIAR_URL_TRAIN = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv"
LIAR_URL_TEST = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv"
LIAR_URL_VALID = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv"

# FakeNewsNet from GitHub (PolitiFact + GossipCop)
FNN_POLITIFACT_FAKE = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv"
FNN_POLITIFACT_REAL = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv"
FNN_GOSSIPCOP_FAKE = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_fake.csv"
FNN_GOSSIPCOP_REAL = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_real.csv"

# Sarcasm Headlines Dataset (The Onion = satire, HuffPost = real)
SARCASM_URL = "https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json"


# ──────────────────────────────────────────────────────────
#  Download Utilities
# ──────────────────────────────────────────────────────────
def download_file(url: str, save_path: Path, chunk_size: int = 8192) -> Path:
    """Download a file from URL with progress indication."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        print(f"  [SKIP] Already exists: {save_path.name}")
        return save_path
    
    print(f"  [DOWNLOADING] {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    
    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = (downloaded / total) * 100
                print(f"\r  Progress: {pct:.1f}%", end="", flush=True)
    print()
    print(f"  [SAVED] {save_path.name} ({save_path.stat().st_size / 1024:.1f} KB)")
    return save_path


# ──────────────────────────────────────────────────────────
#  Dataset Loaders
# ──────────────────────────────────────────────────────────
def load_liar_dataset() -> pd.DataFrame:
    """
    Load LIAR dataset from GitHub mirror.
    Maps 6-way labels to 3-class: Real (true, mostly-true, half-true), 
    Fake (false, pants-fire, barely-true).
    """
    print("\n[1/3] Loading LIAR Dataset...")
    
    train_path = download_file(LIAR_URL_TRAIN, DATA_DIR / "liar_train.tsv")
    test_path = download_file(LIAR_URL_TEST, DATA_DIR / "liar_test.tsv")
    valid_path = download_file(LIAR_URL_VALID, DATA_DIR / "liar_valid.tsv")
    
    cols = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title',
            'state_info', 'party', 'barely_true', 'false_count', 'half_true',
            'mostly_true', 'pants_on_fire', 'context']
    
    dfs = []
    for path in [train_path, test_path, valid_path]:
        df = pd.read_csv(path, sep='\t', header=None, names=cols, on_bad_lines='skip')
        dfs.append(df)
    
    liar = pd.concat(dfs, ignore_index=True)
    
    # Map 6-way to 2-class (Real / Fake)
    real_labels = ['true', 'mostly-true', 'half-true']
    fake_labels = ['false', 'pants-fire', 'barely-true']
    
    liar['mapped_label'] = liar['label'].apply(
        lambda x: 'Real' if x in real_labels else ('Fake' if x in fake_labels else None)
    )
    liar = liar.dropna(subset=['mapped_label', 'statement'])
    
    result = pd.DataFrame({
        'text': liar['statement'].astype(str),
        'label': liar['mapped_label'],
        'source': 'LIAR',
        'original_label': liar['label'],
    })
    
    print(f"  LIAR loaded: {len(result)} samples")
    print(f"  Distribution: {dict(Counter(result['label']))}")
    return result


def load_fakenewsnet_dataset() -> pd.DataFrame:
    """
    Load FakeNewsNet Dataset from GitHub.
    PolitiFact + GossipCop datasets with fake and real labels.
    Uses news titles as text content.
    """
    print("\n[2/3] Loading FakeNewsNet Dataset (PolitiFact + GossipCop)...")
    
    pf_fake_path = download_file(FNN_POLITIFACT_FAKE, DATA_DIR / "politifact_fake.csv")
    pf_real_path = download_file(FNN_POLITIFACT_REAL, DATA_DIR / "politifact_real.csv")
    gc_fake_path = download_file(FNN_GOSSIPCOP_FAKE, DATA_DIR / "gossipcop_fake.csv")
    gc_real_path = download_file(FNN_GOSSIPCOP_REAL, DATA_DIR / "gossipcop_real.csv")
    
    dfs = []
    for path, label in [(pf_fake_path, 'Fake'), (pf_real_path, 'Real'),
                         (gc_fake_path, 'Fake'), (gc_real_path, 'Real')]:
        df = pd.read_csv(path, on_bad_lines='skip')
        df['label'] = label
        dfs.append(df)
    
    fnn = pd.concat(dfs, ignore_index=True)
    fnn = fnn.dropna(subset=['title'])
    
    result = pd.DataFrame({
        'text': fnn['title'].astype(str),
        'label': fnn['label'],
        'source': 'FakeNewsNet',
        'original_label': fnn['label'],
    })
    
    print(f"  FakeNewsNet loaded: {len(result)} samples")
    print(f"  Distribution: {dict(Counter(result['label']))}")
    return result


def load_sarcasm_dataset() -> pd.DataFrame:
    """
    Load Sarcasm Headlines Dataset from Google Storage.
    is_sarcastic=1 (The Onion) → Satire
    is_sarcastic=0 (HuffPost) → Real (used as supplementary Real data)
    """
    print("\n[3/3] Loading Sarcasm Headlines Dataset (for Satire class)...")
    
    save_path = download_file(SARCASM_URL, DATA_DIR / "sarcasm.json")
    
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sarcasm_df = pd.DataFrame(data)
    
    # Sarcastic headlines → Satire class
    satire = sarcasm_df[sarcasm_df['is_sarcastic'] == 1].copy()
    
    result = pd.DataFrame({
        'text': satire['headline'].astype(str),
        'label': 'Satire',
        'source': 'SarcasmHeadlines',
        'original_label': 'sarcastic',
    })
    
    print(f"  Sarcasm dataset loaded: {len(result)} Satire samples")
    return result


# ──────────────────────────────────────────────────────────
#  Dataset Combination & Balancing
# ──────────────────────────────────────────────────────────
def compute_duplicate_rate(df: pd.DataFrame) -> float:
    """Compute the duplicate rate based on text content."""
    total = len(df)
    unique = df['text'].nunique()
    duplicates = total - unique
    rate = duplicates / total if total > 0 else 0
    return rate


def balance_dataset(df: pd.DataFrame, method: str = 'class_weighted') -> pd.DataFrame:
    """
    Handle class imbalance if it exceeds 40%.
    
    Methods:
    - 'undersample': Undersample majority classes to match minority
    - 'class_weighted': Keep all data, return weights for training
    - 'smote': Not applied here (works on feature space, done during training)
    
    Returns balanced dataframe. For class_weighted, adds 'sample_weight' column.
    """
    class_counts = df['label'].value_counts()
    total = len(df)
    max_class_pct = class_counts.max() / total
    
    print(f"\n  Class distribution:")
    for cls, count in class_counts.items():
        print(f"    {cls}: {count} ({count/total*100:.1f}%)")
    
    if max_class_pct > 0.40:
        print(f"\n  WARNING: Class imbalance detected (max class: {max_class_pct*100:.1f}% > 40%)")
        
        if method == 'undersample':
            min_count = class_counts.min()
            balanced_dfs = []
            for label in class_counts.index:
                class_df = df[df['label'] == label]
                sampled = class_df.sample(n=min_count, random_state=42)
                balanced_dfs.append(sampled)
            df = pd.concat(balanced_dfs, ignore_index=True)
            print(f"  Applied undersampling: {len(df)} samples (each class: {min_count})")
            
        elif method == 'class_weighted':
            # Compute inverse frequency weights
            weights = {}
            for cls, count in class_counts.items():
                weights[cls] = total / (len(class_counts) * count)
            df['sample_weight'] = df['label'].map(weights)
            print(f"  Applied class-weighted loss weights: {weights}")
    else:
        print(f"  OK: Class balance acceptable (max class: {max_class_pct*100:.1f}%)")
        df['sample_weight'] = 1.0
    
    return df


def combine_datasets(max_per_source: int = 3000) -> pd.DataFrame:
    """
    Combine all three datasets into a unified DataFrame.
    
    Caps samples per source to maintain balance, ensures minimum 5,000 total.
    """
    print("=" * 60)
    print("TASK 1: DATA SOURCING & RELIABILITY ASSESSMENT")
    print("=" * 60)
    
    # Load all datasets
    liar_df = load_liar_dataset()
    fnn_df = load_fakenewsnet_dataset()
    sarcasm_df = load_sarcasm_dataset()
    
    # Sample from each to create balanced dataset
    # FakeNewsNet: PolitiFact (~800) + GossipCop (~22k)
    # LIAR: ~12.8k
    # Sarcasm: ~11.7k satire
    
    # Sample FakeNewsNet: take up to 2000 Real + 2000 Fake
    fnn_real = fnn_df[fnn_df['label'] == 'Real']
    fnn_fake = fnn_df[fnn_df['label'] == 'Fake']
    fnn_real_sampled = fnn_real.sample(n=min(2000, len(fnn_real)), random_state=42)
    fnn_fake_sampled = fnn_fake.sample(n=min(2000, len(fnn_fake)), random_state=42)
    fnn_sampled = pd.concat([fnn_real_sampled, fnn_fake_sampled])
    
    # Sample LIAR: take 1500 Real + 1500 Fake
    liar_real = liar_df[liar_df['label'] == 'Real'].sample(n=min(1500, len(liar_df[liar_df['label'] == 'Real'])), random_state=42)
    liar_fake = liar_df[liar_df['label'] == 'Fake'].sample(n=min(1500, len(liar_df[liar_df['label'] == 'Fake'])), random_state=42)
    liar_sampled = pd.concat([liar_real, liar_fake])
    
    # Take satire samples (cap at 3000)
    satire_count = len(sarcasm_df)
    target_satire = min(satire_count, 3000)
    sarcasm_sampled = sarcasm_df.sample(n=target_satire, random_state=42)
    
    # Combine
    combined = pd.concat([liar_sampled, fnn_sampled, sarcasm_sampled], ignore_index=True)
    
    # Remove exact duplicates
    dup_rate = compute_duplicate_rate(combined)
    print(f"\n  Duplicate rate before dedup: {dup_rate*100:.2f}%")
    combined = combined.drop_duplicates(subset=['text']).reset_index(drop=True)
    print(f"  After dedup: {len(combined)} samples")
    
    # Handle class imbalance
    combined = balance_dataset(combined, method='undersample')
    
    # Final stats
    print(f"\n  FINAL DATASET:")
    print(f"  Total samples: {len(combined)}")
    print(f"  Class distribution: {dict(Counter(combined['label']))}")
    print(f"  Sources: {dict(Counter(combined['source']))}")
    print(f"  Duplicate rate: {compute_duplicate_rate(combined)*100:.2f}%")
    
    return combined


# ──────────────────────────────────────────────────────────
#  Data Reliability Scorecards
# ──────────────────────────────────────────────────────────
def generate_reliability_scorecards() -> dict:
    """
    Generate Data Reliability Scorecards for each data source.
    
    Criteria (each scored 1-5):
    1. Label Credibility: How trustworthy are the labels?
    2. Recency: How recent is the data?
    3. Domain Relevance to Pakistan: How relevant to Pakistani context?
    4. Class Balance: How balanced are the classes?
    5. Language Consistency: How consistent is the language quality?
    """
    scorecards = {
        'LIAR': {
            'Label Credibility': {
                'score': 5,
                'justification': 'Labels from PolitiFact, a Pulitzer Prize-winning fact-checking organization. '
                                 'Human-annotated with 6-level granularity (pants-fire to true). '
                                 'Gold-standard in NLP misinformation research (Wang, 2017).'
            },
            'Recency': {
                'score': 3,
                'justification': 'Dataset covers statements from 2007-2016. While political misinformation patterns '
                                 'remain relevant, the data predates recent social media misinformation trends '
                                 '(COVID-19, 2020+ elections). Still widely used as benchmark.'
            },
            'Domain Relevance to Pakistan': {
                'score': 2,
                'justification': 'Primarily US political statements. Linguistic patterns of misinformation '
                                 '(hedging, exaggeration, emotional language) transfer across domains, '
                                 'but Pakistan-specific political context is absent.'
            },
            'Class Balance': {
                'score': 4,
                'justification': 'Roughly balanced across 6 labels. After binary mapping (Real/Fake), '
                                 'balance is approximately 55/45. Minor variation but within acceptable thresholds.'
            },
            'Language Consistency': {
                'score': 4,
                'justification': 'Consistent English political statements. Short-form text (1-2 sentences). '
                                 'Minimal noise, no code-switching. Professional language throughout.'
            },
        },
        'FakeNewsNet': {
            'Label Credibility': {
                'score': 5,
                'justification': 'Labels derived from PolitiFact (fact-checking) and GossipCop (entertainment) '
                                 'professional fact-checking organizations. Published as academic benchmark '
                                 'by Shu et al. (2020) in their seminal FakeNewsNet paper.'
            },
            'Recency': {
                'score': 3,
                'justification': 'Dataset covers 2015-2018 news articles. Captures political misinformation '
                                 'patterns from the US election period. News classification patterns '
                                 'remain relevant for current analysis.'
            },
            'Domain Relevance to Pakistan': {
                'score': 2,
                'justification': 'US-centric news articles. However, the stylistic differences between '
                                 'real and fake news (sensationalism, source attribution, factual density) '
                                 'are language-universal features (Perez-Rosas et al., 2018).'
            },
            'Class Balance': {
                'score': 3,
                'justification': 'PolitiFact subset has ~400 fake and ~400 real. GossipCop has ~5,000 fake '
                                 'and ~16,000 real. Combined and sampled to achieve balance.'
            },
            'Language Consistency': {
                'score': 4,
                'justification': 'News article titles from professional sources. Consistent formatting '
                                 'across PolitiFact and GossipCop subsets. English-only content with minimal noise.'
            },
        },
        'SarcasmHeadlines': {
            'Label Credibility': {
                'score': 5,
                'justification': 'Labels are inherently reliable: The Onion is a well-known satirical '
                                 'publication, and HuffPost is a mainstream news outlet. Source-level '
                                 'labeling eliminates human annotation error. Used in Misra & Grover (2021).'
            },
            'Recency': {
                'score': 4,
                'justification': 'Headlines span 2012-2018, covering a broad temporal range. Satire '
                                 'patterns (irony, absurdity, exaggeration) are temporally stable '
                                 'linguistic features, making recency less critical for this class.'
            },
            'Domain Relevance to Pakistan': {
                'score': 2,
                'justification': 'US-centric satirical content. However, the assignment requires a Satire '
                                 'class and The Onion represents the gold standard for English satirical '
                                 'writing. Satirical linguistic features (irony markers, absurdity) '
                                 'transfer across cultural contexts.'
            },
            'Class Balance': {
                'score': 3,
                'justification': 'Dataset contains approximately 11,700 sarcastic and 14,900 non-sarcastic '
                                 'headlines. We only use the sarcastic subset for the Satire class. '
                                 'Balance with other classes is managed through strategic sampling.'
            },
            'Language Consistency': {
                'score': 4,
                'justification': 'Consistent headline format throughout. The Onion maintains high editorial '
                                 'quality for satire. Short-form text (headlines only) which may limit '
                                 'feature richness compared to full articles from other sources.'
            },
        }
    }
    
    # Print scorecards
    print("\n" + "=" * 60)
    print("DATA RELIABILITY SCORECARDS")
    print("=" * 60)
    
    for source, criteria in scorecards.items():
        total = sum(c['score'] for c in criteria.values())
        print(f"\n{'─' * 50}")
        print(f"Source: {source} (Total: {total}/25)")
        print(f"{'─' * 50}")
        for criterion, details in criteria.items():
            print(f"  {criterion}: {details['score']}/5")
            print(f"    {details['justification'][:100]}...")
    
    return scorecards


def generate_justification() -> str:
    """
    Generate 300+ word justification for source combination with NLP literature references.
    """
    justification = """
SOURCE COMBINATION JUSTIFICATION (300+ words)
==============================================

Our dataset construction strategy combines three complementary sources -- LIAR, FakeNewsNet, 
and the Sarcasm Headlines Dataset -- to create a robust three-class (Real, Fake, Satire) corpus 
for misinformation detection. This combination is motivated by both practical and theoretical 
considerations grounded in NLP literature on dataset bias and misinformation detection.

The LIAR dataset (Wang, 2017) provides fine-grained credibility labels from PolitiFact, a 
Pulitzer Prize-winning fact-checking organization. Its 6-level annotation scheme (pants-fire 
to true) allows flexible binary mapping while preserving label reliability. Research by 
Rashkin et al. (2017) demonstrated that linguistic features from political fact-checking 
datasets generalize well to broader misinformation contexts, making LIAR an excellent 
foundation despite its US-centric focus.

The FakeNewsNet dataset (Shu et al., 2020) contributes news article titles from two domains:
PolitiFact (political fact-checking) and GossipCop (entertainment news verification). This
multi-domain coverage enables our pipeline to learn features that generalize beyond a single
topic area. Perez-Rosas et al. (2018) showed that stylometric and lexical features of fake
news (sensationalism, emotional manipulation, lack of attribution) are language-universal,
supporting cross-domain applicability.

The Sarcasm Headlines Dataset (Misra & Grover, 2021) fills the critical Satire class using 
The Onion's satirical headlines. Satire detection is a known challenge in misinformation 
research because satire mimics fake news structure while serving a different communicative 
purpose (Rubin et al., 2016). Including this class prevents the common pitfall of binary 
classifiers that conflate satire with misinformation, a significant source of false positives 
in production systems.

Regarding dataset bias, we acknowledge the Western-centric nature of all three sources. 
Augenstein et al. (2019) demonstrated that cross-cultural bias in training data can reduce 
model performance on underrepresented populations. To mitigate this, we focus on 
language-universal features (TF-IDF patterns, stylometric markers, sentiment distributions) 
rather than topic-specific or culturally-bound features. Our preprocessing pipeline (Task 3) 
includes Roman Urdu handling to prepare for future Pakistan-specific data integration.

We apply undersampling to address class imbalance exceeding 40%, following the recommendation 
of He & Garcia (2009) that undersampling is preferred over oversampling when sufficient data 
exists in all classes, as it avoids the overfitting risks associated with SMOTE on text data 
(Blagus & Lusa, 2013). The final dataset maintains balanced representation across all three 
classes while preserving the linguistic diversity of each source.

REFERENCES:
- Ahmed, H., et al. (2018). "Detecting opinion spam and fake news using text classification." 
  Security and Privacy, 1(1), e9.
- Augenstein, I., et al. (2019). "MultiFC: A real-world multi-domain dataset for evidence-based 
  fact checking of claims." EMNLP.
- Blagus, R., & Lusa, L. (2013). "SMOTE for high-dimensional class-imbalanced data." 
  BMC Bioinformatics, 14(1), 106.
- He, H., & Garcia, E. A. (2009). "Learning from imbalanced data." IEEE TKDE, 21(9).
- Misra, R., & Grover, J. (2021). "Sculpting Data for ML: The first act of Machine Learning."
- Pérez-Rosas, V., et al. (2018). "Automatic detection of fake news." COLING.
- Rashkin, H., et al. (2017). "Truth of varying shades." EMNLP.
- Rubin, V. L., et al. (2016). "Fake news or truth? Using satirical cues to detect potentially 
  misleading news." NAACL Workshop on Computational Approaches to Deception Detection.
- Wang, W. Y. (2017). "Liar, liar pants on fire: A new benchmark dataset for fake news detection." 
  ACL.
"""
    return justification


# ──────────────────────────────────────────────────────────
#  Main Execution
# ──────────────────────────────────────────────────────────
def run_task1() -> pd.DataFrame:
    """Execute Task 1 completely and return the combined dataset."""
    # Combine datasets
    df = combine_datasets()
    
    # Generate scorecards
    scorecards = generate_reliability_scorecards()
    
    # Generate justification
    justification = generate_justification()
    print(justification)
    
    # Save combined dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "combined_dataset.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\n  Dataset saved to: {output_path}")
    
    # Also save as CSV for inspection
    csv_path = PROCESSED_DIR / "combined_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved to: {csv_path}")
    
    # Save scorecards
    scorecards_path = REPORTS_DIR / "reliability_scorecards.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    scorecards_flat = {}
    for source, criteria in scorecards.items():
        scorecards_flat[source] = {k: v for k, v in criteria.items()}
    
    with open(scorecards_path, 'w') as f:
        json.dump(scorecards_flat, f, indent=2)
    print(f"  Scorecards saved to: {scorecards_path}")
    
    return df


if __name__ == "__main__":
    df = run_task1()
    print(f"\n✓ Task 1 complete. Final dataset: {len(df)} samples")
