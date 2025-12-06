"""
Build News-Level Sentiment Dataset for Romgaz (SNG.RO)

This script:
1. Loads Romgaz news from Excel file
2. Reads full text from extracted .txt files
3. Computes BNR FS lexicon-based sentiment for each article
4. Saves enriched news-level dataset

Output: data/sng_news_with_sentiment.csv
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from sentiment_lexicon import load_fs_lexicon, compute_lexicon_sentiment
from ro_preprocessing import RomanianLemmatizer, clean_financial_text


def parse_bvb_datetime(dt_str: str) -> datetime:
    """
    Parse BVB datetime format: "DD.MM.YYYY HH:MM:SS"
    
    Args:
        dt_str: DateTime string from BVB
        
    Returns:
        datetime object
    """
    try:
        return datetime.strptime(dt_str, "%d.%m.%Y %H:%M:%S")
    except:
        try:
            return datetime.strptime(dt_str, "%d.%m.%Y %H:%M")
        except:
            # Try other formats
            return pd.to_datetime(dt_str)


def read_news_text(text_file_path: str, text_excerpt: str = "") -> str:
    """
    Read news text from file, with fallback to excerpt.
    
    Args:
        text_file_path: Relative path to .txt file
        text_excerpt: Excerpt from Excel (fallback)
        
    Returns:
        Full text or excerpt
    """
    # Normalize path separators
    text_file_path = text_file_path.replace('\\\\', '/').replace('\\', '/')
    
    # Try to read from file
    if text_file_path and text_file_path != "":
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if text.strip():
                    return text
        except FileNotFoundError:
            # Try with different base paths
            for base in ['', 'bvb_sng_stiri/', '../bvb_sng_stiri/']:
                full_path = base + text_file_path
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        if text.strip():
                            return text
                except:
                    continue
        except Exception as e:
            print(f"Warning: Could not read {text_file_path}: {e}")
    
    # Fallback to excerpt
    return text_excerpt if text_excerpt else ""


def main():
    print("=" * 70)
    print("Building Romgaz News Sentiment Dataset with BNR FS Lexicon")
    print("=" * 70)
    
    # Configuration
    EXCEL_PATH = "bvb_sng_stiri/SNG_stiri_full.xlsx"
    OUTPUT_PATH = "data/sng_news_with_sentiment.csv"
    
    # Step 1: Load BNR lexicon
    print("\n[1/5] Loading BNR Financial Stability Lexicon...")
    lexicon = load_fs_lexicon()
    
    # Step 2: Initialize Romanian lemmatizer
    print("\n[2/5] Initializing Romanian lemmatizer (Stanza)...")
    try:
        lemmatizer = RomanianLemmatizer(use_gpu=False, download_if_needed=True)
    except Exception as e:
        print(f"Error initializing lemmatizer: {e}")
        print("Please install Stanza: pip install stanza")
        return
    
    # Step 3: Load news data
    print("\n[3/5] Loading Romgaz news from Excel...")
    try:
        df_news = pd.read_excel(EXCEL_PATH)
        print(f"  Loaded {len(df_news)} news articles")
        print(f"  Columns: {df_news.columns.tolist()}")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return
    
    # Step 4: Process each news article
    print("\n[4/5] Processing news articles and computing sentiment...")
    
    results = []
    
    for idx, row in df_news.iterrows():
        if (idx + 1) % 10 == 0:
            print(f"  Processing article {idx + 1}/{len(df_news)}...")
        
        # Parse datetime
        try:
            pub_dt = parse_bvb_datetime(row['published_at'])
            pub_date = pub_dt.date()
        except Exception as e:
            print(f"  Warning: Could not parse datetime for row {idx}: {e}")
            pub_dt = pd.NaT
            pub_date = pd.NaT
        
        # Read full text
        text_file = row.get('text_file', '')
        text_excerpt = row.get('text_excerpt', '')
        
        full_text = read_news_text(text_file, text_excerpt)
        
        # Clean and lemmatize
        cleaned_text = clean_financial_text(full_text)
        
        try:
            lemmas = lemmatizer.lemmatize(cleaned_text)
        except Exception as e:
            print(f"  Warning: Lemmatization failed for row {idx}: {e}")
            lemmas = []
        
        # Compute sentiment
        sentiment = compute_lexicon_sentiment(lemmas, lexicon)
        
        # Build result row
        result = {
            'news_id': idx,
            'title': row.get('title', ''),
            'published_at': row['published_at'],
            'published_datetime': pub_dt,
            'published_date': pub_date,
            'pdf_url': row.get('pdf_url', ''),
            'pdf_file': row.get('pdf_file', ''),
            'text_file': text_file,
            'text_length': len(full_text),
            'status': row.get('status', ''),
        }
        
        # Add sentiment metrics
        result.update(sentiment.to_dict())
        
        # Add flags
        result['has_lex_hits'] = sentiment.n_pos + sentiment.n_neg > 0
        
        results.append(result)
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Step 5: Save results
    print("\n[5/5] Saving results...")
    Path("data").mkdir(exist_ok=True)
    df_results.to_csv(OUTPUT_PATH, index=False)
    print(f"  ✓ Saved to: {OUTPUT_PATH}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\nTotal articles: {len(df_results)}")
    print(f"Articles with lexicon hits: {df_results['has_lex_hits'].sum()} "
          f"({df_results['has_lex_hits'].sum() / len(df_results) * 100:.1f}%)")
    
    print(f"\nDate range: {df_results['published_date'].min()} to {df_results['published_date'].max()}")
    
    # Sentiment distribution
    df_with_hits = df_results[df_results['has_lex_hits']]
    
    if len(df_with_hits) > 0:
        print(f"\nFor articles with lexicon hits (n={len(df_with_hits)}):")
        print(f"  Simple Index - Mean: {df_with_hits['simple_index'].mean():.3f}, "
              f"Std: {df_with_hits['simple_index'].std():.3f}")
        print(f"  Score Index - Mean: {df_with_hits['score_index'].mean():.3f}, "
              f"Std: {df_with_hits['score_index'].std():.3f}")
        print(f"  Avg pos terms: {df_with_hits['n_pos'].mean():.1f}")
        print(f"  Avg neg terms: {df_with_hits['n_neg'].mean():.1f}")
        print(f"  Avg coverage: {df_with_hits['coverage'].mean() * 100:.1f}%")
    
    print("\n" + "=" * 70)
    print("✓ News-level sentiment dataset created successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

