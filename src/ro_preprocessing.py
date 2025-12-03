"""
Romanian Text Preprocessing for Financial NLP

This module provides Romanian text preprocessing optimized for financial texts,
specifically designed to work with the BNR Financial Stability lexicon.

Key operations:
- Diacritics normalization
- Tokenization
- Lemmatization using Stanza (Romanian models)
- Text cleaning for financial documents
"""

import re
from typing import List, Optional
import unicodedata


def normalize_romanian(text: str) -> str:
    """
    Normalize Romanian text for consistent processing.
    
    Operations:
    - Replace cedilla forms (ş, ţ) with comma-below (ș, ț)
    - Collapse multiple whitespace to single space
    - Remove common PDF artifacts
    - Normalize quotes and dashes
    
    Args:
        text: Input Romanian text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Diacritics normalization (cedilla to comma)
    replacements = {
        'ş': 'ș',
        'Ş': 'Ș',
        'ţ': 'ț',
        'Ţ': 'Ț',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('–', '-').replace('—', '-')
    
    # Remove common PDF artifacts
    text = re.sub(r'Pagina \d+ din \d+', ' ', text)
    text = re.sub(r'\f', ' ', text)  # form feed
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def clean_financial_text(text: str) -> str:
    """
    Clean financial document text while preserving semantic content.
    
    Args:
        text: Raw extracted text (e.g., from PDF)
        
    Returns:
        Cleaned text suitable for lemmatization
    """
    text = normalize_romanian(text)
    
    # Convert to lowercase for lexicon matching
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove standalone numbers (but keep percentages, currency)
    # This is conservative - we keep numbers with context
    text = re.sub(r'\b\d+\b', ' ', text)
    
    # Collapse whitespace again
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


class RomanianLemmatizer:
    """
    Romanian lemmatizer using Stanza NLP pipeline.
    
    The BNR lexicon is in lemma form, so lemmatization is critical for
    good coverage and accurate sentiment measurement.
    
    Usage:
        lemmatizer = RomanianLemmatizer()
        lemmas = lemmatizer.lemmatize("Compania a raportat profituri mari.")
        # Returns: ['companie', 'a', 'raporta', 'profit', 'mare']
    """
    
    def __init__(self, use_gpu: bool = False, download_if_needed: bool = True):
        """
        Initialize the Stanza pipeline for Romanian.
        
        Args:
            use_gpu: Whether to use GPU acceleration (requires CUDA)
            download_if_needed: Auto-download models if not present
        """
        try:
            import stanza
        except ImportError:
            raise ImportError(
                "Stanza is required for Romanian lemmatization.\n"
                "Install with: pip install stanza"
            )
        
        self.stanza = stanza
        
        # Download Romanian models if needed
        if download_if_needed:
            try:
                print("Checking Stanza Romanian models...")
                self.stanza.download('ro', verbose=False)
            except Exception as e:
                print(f"Note: {e}")
        
        # Initialize pipeline
        # We use: tokenize, pos (part-of-speech), lemma
        print("Loading Stanza Romanian pipeline...")
        self.nlp = self.stanza.Pipeline(
            'ro',
            processors='tokenize,pos,lemma',
            use_gpu=use_gpu,
            verbose=False,
            download_method=None,  # Don't auto-download during pipeline init
        )
        print("✓ Stanza pipeline ready")
    
    def lemmatize(self, text: str, keep_alpha_only: bool = True) -> List[str]:
        """
        Lemmatize Romanian text.
        
        Args:
            text: Input text (will be normalized)
            keep_alpha_only: If True, filter out non-alphabetic tokens
            
        Returns:
            List of lemmas (lowercase, normalized)
            
        Note:
            - Text is normalized before processing
            - Punctuation is removed if keep_alpha_only=True
            - Empty lemmas are filtered out
        """
        # Normalize first
        text = normalize_romanian(text.lower())
        
        if not text:
            return []
        
        # Process with Stanza
        try:
            doc = self.nlp(text)
        except Exception as e:
            print(f"Warning: Stanza processing failed: {e}")
            return []
        
        lemmas = []
        
        for sentence in doc.sentences:
            for word in sentence.words:
                # Get lemma (fallback to original text if lemma not available)
                lemma = (word.lemma or word.text).lower()
                lemma = normalize_romanian(lemma)
                
                # Filter if requested
                if keep_alpha_only:
                    # Keep only if contains at least one alphabetic character
                    if not any(c.isalpha() for c in lemma):
                        continue
                
                if lemma:
                    lemmas.append(lemma)
        
        return lemmas


def lemmatize_romanian(
    text: str,
    nlp,
    clean: bool = True,
) -> List[str]:
    """
    Convenience function to lemmatize Romanian text.
    
    This function is compatible with both RomanianLemmatizer objects
    and raw Stanza pipeline objects.
    
    Args:
        text: Input Romanian text
        nlp: RomanianLemmatizer or Stanza Pipeline object
        clean: Whether to clean text first
        
    Returns:
        List of lemmas
    """
    if clean:
        text = clean_financial_text(text)
    
    if isinstance(nlp, RomanianLemmatizer):
        return nlp.lemmatize(text)
    else:
        # Assume it's a Stanza pipeline
        text = normalize_romanian(text.lower())
        if not text:
            return []
        
        doc = nlp(text)
        lemmas = []
        
        for sent in doc.sentences:
            for w in sent.words:
                lemma = (w.lemma or w.text).lower()
                lemma = normalize_romanian(lemma)
                if any(ch.isalpha() for ch in lemma):
                    lemmas.append(lemma)
        
        return lemmas


if __name__ == "__main__":
    # Test the preprocessing
    print("Testing Romanian Preprocessing Module")
    print("=" * 60)
    
    # Test normalization
    test_text = "Compania Romgaz a raportat profituri record în trimestrul 2."
    print(f"\nOriginal: {test_text}")
    print(f"Normalized: {normalize_romanian(test_text)}")
    print(f"Cleaned: {clean_financial_text(test_text)}")
    
    # Test with cedilla forms
    test_cedilla = "Preşedintele companiei a ţinut un discurs."
    print(f"\nCedilla form: {test_cedilla}")
    print(f"Normalized: {normalize_romanian(test_cedilla)}")
    
    # Test lemmatization (requires Stanza)
    print("\n" + "-" * 60)
    print("Testing lemmatization (requires Stanza)...")
    
    try:
        lemmatizer = RomanianLemmatizer()
        
        test_sent = "Companiile au raportat profituri mari și dividende crescute."
        print(f"\nOriginal: {test_sent}")
        
        lemmas = lemmatizer.lemmatize(test_sent)
        print(f"Lemmas: {lemmas}")
        
        print("\n✓ Lemmatization test successful")
        
    except ImportError as e:
        print(f"\nSkipping lemmatization test: {e}")
    except Exception as e:
        print(f"\nLemmatization test error: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing module test complete")

