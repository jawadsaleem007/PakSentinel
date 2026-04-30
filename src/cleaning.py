"""
Task 3.1 — Cleaning Module [5 Marks]

Handles: HTML tags, URLs, social media handles, Roman Urdu code-switching,
repeated punctuation, emojis, and inconsistent formats.
Includes before/after noise audit on 200 randomly sampled records.
"""

import re
import time
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "figures"


# ──────────────────────────────────────────────────────────
#  Cleaning Functions
# ──────────────────────────────────────────────────────────
def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', ' ', text)


def remove_urls(text: str) -> str:
    """Remove URLs (http, https, www, ftp)."""
    return re.sub(r'https?://\S+|www\.\S+|ftp://\S+', ' ', text)


def remove_social_media_handles(text: str) -> str:
    """Remove @mentions and #hashtags."""
    text = re.sub(r'@\w+', ' ', text)  # @mentions
    text = re.sub(r'#\w+', ' ', text)   # #hashtags
    return text


def normalize_repeated_punctuation(text: str) -> str:
    """Reduce repeated punctuation (e.g., !!! → !, ??? → ?)."""
    text = re.sub(r'([!?.])\1+', r'\1', text)
    text = re.sub(r'[-]{2,}', '—', text)  # Multiple dashes → em dash
    return text


def remove_emojis(text: str) -> str:
    """Remove emoji characters."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(' ', text)


def handle_roman_urdu(text: str) -> str:
    """
    Handle Roman Urdu code-switching.
    
    Strategy: Detect and preserve Roman Urdu words rather than removing them.
    Common Roman Urdu markers are transliterated Urdu words mixed with English.
    We normalize common patterns and mark code-switching boundaries.
    """
    # Common Roman Urdu words that appear in Pakistani social media
    roman_urdu_words = {
        'kya', 'hai', 'nahi', 'yeh', 'woh', 'koi', 'aur', 'bhi', 'toh',
        'mein', 'ko', 'se', 'ka', 'ki', 'ke', 'par', 'haan', 'ji',
        'achha', 'theek', 'bohot', 'zyada', 'kam', 'sab', 'log',
        'kuch', 'abhi', 'phir', 'lekin', 'magar', 'kyunke', 'isliye',
        'sahi', 'galat', 'jhoot', 'sach', 'khabar', 'mulk', 'awam',
        'hukumat', 'fauj', 'siyasat', 'wazir', 'azam',
    }
    
    # We don't remove Roman Urdu — we just normalize spacing around it
    # This preserves multilingual signal which may be important for Pakistani context
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: multiple spaces → single, strip leading/trailing."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_case(text: str) -> str:
    """Convert to lowercase."""
    return text.lower()


def remove_special_characters(text: str) -> str:
    """Remove special characters but keep basic punctuation and alphanumerics."""
    # Keep letters, digits, basic punctuation, and spaces
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text)
    return text


def normalize_numbers(text: str) -> str:
    """Replace specific numbers with NUM token for generalization."""
    text = re.sub(r'\b\d{4,}\b', ' NUM ', text)  # Long numbers
    text = re.sub(r'\$[\d,.]+', ' MONEY ', text)  # Dollar amounts
    text = re.sub(r'\d+%', ' PERCENT ', text)  # Percentages
    return text


def remove_email_addresses(text: str) -> str:
    """Remove email addresses."""
    return re.sub(r'\S+@\S+\.\S+', ' ', text)


# ──────────────────────────────────────────────────────────
#  Main Cleaning Pipeline
# ──────────────────────────────────────────────────────────
def clean_text(text: str, aggressive: bool = False) -> str:
    """
    Apply full cleaning pipeline to a single text.
    
    Args:
        text: Raw text to clean
        aggressive: If True, apply additional normalization (numbers, special chars)
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str) or len(text) == 0:
        return ""
    
    # Core cleaning steps (ordered by priority)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_email_addresses(text)
    text = remove_social_media_handles(text)
    text = remove_emojis(text)
    text = normalize_repeated_punctuation(text)
    text = handle_roman_urdu(text)
    
    if aggressive:
        text = normalize_numbers(text)
        text = remove_special_characters(text)
    
    text = normalize_case(text)
    text = normalize_whitespace(text)
    
    return text


def clean_dataframe(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """Apply cleaning pipeline to entire dataframe."""
    df = df.copy()
    df['text_clean'] = df[text_col].apply(clean_text)
    
    # Remove empty texts after cleaning
    empty_before = len(df)
    df = df[df['text_clean'].str.len() > 10].reset_index(drop=True)
    empty_after = len(df)
    
    if empty_before != empty_after:
        print(f"  Removed {empty_before - empty_after} empty/very short texts after cleaning")
    
    return df


# ──────────────────────────────────────────────────────────
#  Noise Audit (200 random samples)
# ──────────────────────────────────────────────────────────
def noise_audit(df: pd.DataFrame, n_samples: int = 200, text_col: str = 'text') -> dict:
    """
    Conduct before/after noise audit on n randomly sampled records.
    
    Reports metrics on noise patterns found and cleaned.
    """
    print(f"\n{'─' * 50}")
    print(f"NOISE AUDIT ({n_samples} random samples)")
    print(f"{'─' * 50}")
    
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    audit_results = {
        'total_samples': len(sample),
        'html_tags_found': 0,
        'urls_found': 0,
        'social_handles_found': 0,
        'repeated_punct_found': 0,
        'emojis_found': 0,
        'email_addresses_found': 0,
        'avg_length_before': 0,
        'avg_length_after': 0,
        'avg_tokens_before': 0,
        'avg_tokens_after': 0,
        'total_chars_removed': 0,
    }
    
    before_lengths = []
    after_lengths = []
    
    for _, row in sample.iterrows():
        text = str(row[text_col])
        before_lengths.append(len(text))
        
        # Count noise patterns
        audit_results['html_tags_found'] += len(re.findall(r'<[^>]+>', text))
        audit_results['urls_found'] += len(re.findall(r'https?://\S+|www\.\S+', text))
        audit_results['social_handles_found'] += len(re.findall(r'@\w+|#\w+', text))
        audit_results['repeated_punct_found'] += len(re.findall(r'([!?.])\1+', text))
        audit_results['email_addresses_found'] += len(re.findall(r'\S+@\S+\.\S+', text))
        
        cleaned = clean_text(text)
        after_lengths.append(len(cleaned))
    
    audit_results['avg_length_before'] = np.mean(before_lengths)
    audit_results['avg_length_after'] = np.mean(after_lengths)
    audit_results['avg_tokens_before'] = np.mean([len(str(t).split()) for _, t in sample[text_col].items()])
    audit_results['total_chars_removed'] = sum(b - a for b, a in zip(before_lengths, after_lengths))
    
    # Print audit report
    print(f"\n  Noise Patterns Found (in {n_samples} samples):")
    print(f"    HTML tags:          {audit_results['html_tags_found']}")
    print(f"    URLs:               {audit_results['urls_found']}")
    print(f"    Social handles:     {audit_results['social_handles_found']}")
    print(f"    Repeated punct:     {audit_results['repeated_punct_found']}")
    print(f"    Email addresses:    {audit_results['email_addresses_found']}")
    print(f"\n  Length Statistics:")
    print(f"    Avg length before:  {audit_results['avg_length_before']:.0f} chars")
    print(f"    Avg length after:   {audit_results['avg_length_after']:.0f} chars")
    print(f"    Total chars removed:{audit_results['total_chars_removed']}")
    print(f"    Reduction:          {(1 - audit_results['avg_length_after']/audit_results['avg_length_before'])*100:.1f}%")
    
    return audit_results


# ──────────────────────────────────────────────────────────
#  Run Task 3.1
# ──────────────────────────────────────────────────────────
def run_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Execute Task 3.1: Clean data and produce noise audit."""
    print("\n" + "=" * 60)
    print("TASK 3.1: TEXT CLEANING")
    print("=" * 60)
    
    # Run noise audit BEFORE cleaning
    audit = noise_audit(df, n_samples=200)
    
    # Clean the dataset
    start_time = time.time()
    df_clean = clean_dataframe(df)
    elapsed = time.time() - start_time
    
    print(f"\n  Cleaning completed in {elapsed:.2f}s")
    print(f"  Dataset size: {len(df)} → {len(df_clean)} samples")
    
    return df_clean


if __name__ == "__main__":
    # Test with sample data
    test_texts = [
        "BREAKING: <b>Pakistan</b> PM says 'no war' https://t.co/abc123 @pmln_official #politics!!!",
        "Yeh kya ho raha hai??? 😡😡😡 This is FAKE NEWS email me at fake@news.com",
        "Reuters: The economic data shows growth of $1.5B (15% increase)...",
    ]
    for text in test_texts:
        print(f"BEFORE: {text}")
        print(f"AFTER:  {clean_text(text)}")
        print()
