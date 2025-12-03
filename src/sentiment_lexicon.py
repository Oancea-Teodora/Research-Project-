"""
BNR Financial Stability Lexicon - Sentiment Analysis Module

This module implements lexicon-based sentiment analysis using the Romanian
Financial Stability dictionary developed by the National Bank of Romania (BNR).

Reference:
"A Natural Language Processing toolbox for the National Bank of Romania"
Occasional Papers no. 38, May 2025, National Bank of Romania

Key Innovation:
This is the first application of the BNR FS lexicon to Bucharest Stock Exchange
(BVB) equity prediction, specifically for Romgaz (SNG.RO) stock movements.
"""

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


def normalize_romanian_diacritics(text: str) -> str:
    """
    Normalize Romanian diacritics to ensure consistency.
    
    The BNR lexicon contains both cedilla (ş, ţ) and comma-below (ș, ț) forms.
    We standardize to comma-below which is the correct Romanian standard.
    
    Args:
        text: Input Romanian text
        
    Returns:
        Text with normalized diacritics (ș, ț instead of ş, ţ)
    """
    # Map cedilla forms to comma-below forms
    replacements = {
        'ş': 'ș',  # s-cedilla to s-comma
        'Ş': 'Ș',
        'ţ': 'ț',  # t-cedilla to t-comma
        'Ţ': 'Ț',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def load_fs_lexicon(
    positive_path: str = "FS_positive_AI_scores.txt",
    negative_path: str = "FS_negative_AI_scores.txt",
) -> Dict[str, float]:
    """
    Load the BNR Financial Stability lexicon from tab-separated files.
    
    The lexicon contains Romanian financial terms (lemmatized) with sentiment
    polarity and intensity scores. Scores range from -1 (most negative) to
    +1 (most positive) in steps of 0.25.
    
    Args:
        positive_path: Path to positive terms file (lemma<TAB>score, score > 0)
        negative_path: Path to negative terms file (lemma<TAB>score, score < 0)
        
    Returns:
        Dictionary mapping normalized lemma (lowercase) to sentiment score
        
    Note:
        - Lemmas are normalized (diacritics, lowercase)
        - If a lemma appears in both files (rare), we keep the one with
          higher absolute score (stronger signal)
    """
    lexicon: Dict[str, float] = {}
    
    # Load positive terms
    with open(positive_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split('\t')
                if len(parts) != 2:
                    print(f"Warning: Skipping malformed line {line_num} in {positive_path}")
                    continue
                    
                lemma, score_str = parts
                score = float(score_str)
                
                # Normalize lemma
                lemma = normalize_romanian_diacritics(lemma.lower())
                
                # Store or update if stronger signal
                if lemma in lexicon:
                    if abs(score) > abs(lexicon[lemma]):
                        lexicon[lemma] = score
                else:
                    lexicon[lemma] = score
                    
            except ValueError as e:
                print(f"Warning: Could not parse line {line_num} in {positive_path}: {e}")
    
    # Load negative terms
    with open(negative_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split('\t')
                if len(parts) != 2:
                    print(f"Warning: Skipping malformed line {line_num} in {negative_path}")
                    continue
                    
                lemma, score_str = parts
                score = float(score_str)
                
                # Normalize lemma
                lemma = normalize_romanian_diacritics(lemma.lower())
                
                # Store or update if stronger signal
                if lemma in lexicon:
                    if abs(score) > abs(lexicon[lemma]):
                        lexicon[lemma] = score
                else:
                    lexicon[lemma] = score
                    
            except ValueError as e:
                print(f"Warning: Could not parse line {line_num} in {negative_path}: {e}")
    
    print(f"Loaded BNR FS Lexicon: {len(lexicon)} unique lemmas")
    print(f"  - Positive terms: {sum(1 for s in lexicon.values() if s > 0)}")
    print(f"  - Negative terms: {sum(1 for s in lexicon.values() if s < 0)}")
    
    return lexicon


@dataclass
class LexiconSentiment:
    """
    Container for lexicon-based sentiment metrics.
    
    Attributes:
        n_pos: Count of positive lemmas matched
        n_neg: Count of negative lemmas matched
        sum_pos: Sum of positive scores
        sum_neg: Sum of negative scores (will be negative)
        simple_index: Unweighted sentiment: (n_pos - n_neg) / (n_pos + n_neg)
        score_index: Weighted sentiment: (sum_pos + sum_neg) / (|sum_pos| + |sum_neg|)
        total_lemmas: Total lemmas in text
        coverage: Fraction of lemmas matched in lexicon
    """
    n_pos: int
    n_neg: int
    sum_pos: float
    sum_neg: float
    simple_index: float
    score_index: float
    total_lemmas: int = 0
    coverage: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for easy DataFrame creation."""
        return {
            'n_pos': self.n_pos,
            'n_neg': self.n_neg,
            'sum_pos': self.sum_pos,
            'sum_neg': self.sum_neg,
            'simple_index': self.simple_index,
            'score_index': self.score_index,
            'total_lemmas': self.total_lemmas,
            'coverage': self.coverage,
        }


def compute_lexicon_sentiment(
    lemmas: List[str],
    lexicon: Dict[str, float],
) -> LexiconSentiment:
    """
    Compute lexicon-based sentiment from a list of lemmas.
    
    This implements two sentiment indices:
    
    1. Simple Index (unweighted):
       - Counts positive vs negative lemma matches
       - Formula: (n_pos - n_neg) / (n_pos + n_neg)
       - Range: [-1, +1]
       - Interpretation: What fraction of sentiment-bearing terms are positive?
       
    2. Score Index (intensity-weighted):
       - Sums sentiment scores of matched lemmas
       - Formula: (sum_pos + sum_neg) / (|sum_pos| + |sum_neg|)
       - Range: [-1, +1]
       - Interpretation: What is the net sentiment when accounting for intensity?
    
    Args:
        lemmas: List of lemmatized tokens (lowercase, normalized)
        lexicon: BNR FS lexicon dictionary
        
    Returns:
        LexiconSentiment object with computed metrics
        
    Note:
        - If no lexicon hits, both indices return 0.0
        - This is analogous to BNR's approach for financial stability text
    """
    n_pos = 0
    n_neg = 0
    sum_pos = 0.0
    sum_neg = 0.0
    
    # Count matches
    for lemma in lemmas:
        if lemma in lexicon:
            score = lexicon[lemma]
            if score > 0:
                n_pos += 1
                sum_pos += score
            elif score < 0:
                n_neg += 1
                sum_neg += score  # sum_neg will be negative
    
    # Compute indices
    total_hits = n_pos + n_neg
    
    if total_hits == 0:
        simple_index = 0.0
    else:
        simple_index = (n_pos - n_neg) / total_hits
    
    total_abs = abs(sum_pos) + abs(sum_neg)
    
    if total_abs == 0:
        score_index = 0.0
    else:
        score_index = (sum_pos + sum_neg) / total_abs
    
    # Compute coverage
    total_lemmas = len(lemmas)
    coverage = total_hits / total_lemmas if total_lemmas > 0 else 0.0
    
    return LexiconSentiment(
        n_pos=n_pos,
        n_neg=n_neg,
        sum_pos=sum_pos,
        sum_neg=sum_neg,
        simple_index=simple_index,
        score_index=score_index,
        total_lemmas=total_lemmas,
        coverage=coverage,
    )


if __name__ == "__main__":
    # Test the lexicon loading
    print("Testing BNR FS Lexicon Loading...")
    print("=" * 60)
    
    lexicon = load_fs_lexicon()
    
    print("\nSample positive terms:")
    pos_terms = {k: v for k, v in list(lexicon.items())[:5] if v > 0}
    for lemma, score in pos_terms.items():
        print(f"  {lemma}: {score:+.2f}")
    
    print("\nSample negative terms:")
    neg_terms = {k: v for k, v in list(lexicon.items())[:100] if v < 0}
    for i, (lemma, score) in enumerate(list(neg_terms.items())[:5]):
        print(f"  {lemma}: {score:+.2f}")
    
    print("\nScore distribution:")
    from collections import Counter
    score_dist = Counter(lexicon.values())
    for score in sorted(score_dist.keys()):
        print(f"  {score:+.2f}: {score_dist[score]} terms")
    
    print("\n" + "=" * 60)
    print("✓ Lexicon module test complete")

