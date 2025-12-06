"""
Build Final Modelling Dataset for Romgaz Stock Prediction

This script merges:
- Daily stock price data (returns, lags)
- Daily sentiment features from BNR FS lexicon

The result is a dataset where each row represents day t with:
- Features available at end of day t (price history + news sentiment from day t)
- Label: whether next trading day (t+1) closes up or down

Output: data/sng_modelling_dataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta


def load_price_data(price_file: str) -> pd.DataFrame:
    """
    Load and prepare Romgaz price data.
    
    Args:
        price_file: Path to price CSV
        
    Returns:
        DataFrame with date, close, returns, and lags
    """
    df = pd.read_csv(price_file)
    
    # Parse date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.rename(columns={'Date': 'date'})
    
    # Ensure we have Close price
    if 'Close' in df.columns:
        df['close'] = df['Close']
    elif 'close' not in df.columns and 'Price' in df.columns:
        df['close'] = df['Price']
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Compute returns if not present
    if 'Return_t' not in df.columns:
        df['return'] = df['close'].pct_change()
    else:
        df['return'] = df['Return_t']
    
    # Compute lagged returns
    df['r_t_minus_1'] = df['return'].shift(1)
    df['r_t_minus_2'] = df['return'].shift(2)
    
    # Compute next-day return (label)
    df['r_t_plus_1'] = df['return'].shift(-1)
    
    # Binary label: 1 if next day up, 0 otherwise
    df['y_next_up'] = (df['r_t_plus_1'] > 0).astype(int)
    
    # Volume if available
    if 'Vol.' in df.columns:
        df['volume'] = df['Vol.']
    
    print(f"  Loaded {len(df)} trading days")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def merge_price_and_sentiment(
    df_price: pd.DataFrame,
    df_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge price and sentiment data on calendar date.
    
    The merge logic:
    - For each trading day t, we use:
      * Price features from t and earlier (return, lags)
      * Sentiment from news published on calendar date t
      * Label: whether trading day t+1 is up
    
    Args:
        df_price: Price DataFrame with date, returns, label
        df_sentiment: Daily sentiment DataFrame
        
    Returns:
        Merged modelling dataset
    """
    # Merge on date
    df_merged = pd.merge(
        df_price,
        df_sentiment,
        on='date',
        how='left',  # Keep all trading days
    )
    
    # Fill missing sentiment values (days without news)
    sentiment_cols = [
        'news_count', 'lex_hits_docs',
        'simple_mean', 'simple_sum', 'simple_max', 'simple_min', 'simple_std',
        'score_mean', 'score_sum', 'score_max', 'score_min', 'score_std',
        'sum_pos_total', 'sum_neg_total',
        'n_pos_total', 'n_neg_total',
        'avg_coverage',
    ]
    
    for col in sentiment_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)
    
    # Fill boolean flags
    df_merged['has_news'] = df_merged['has_news'].fillna(False)
    df_merged['has_lex_hits'] = df_merged['has_lex_hits'].fillna(False)
    
    return df_merged


def prepare_modelling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final preparation of modelling dataset.
    
    - Remove rows with missing label (last day has no next-day return)
    - Remove rows with missing price features (first days have no lags)
    - Ensure proper ordering
    
    Args:
        df: Merged DataFrame
        
    Returns:
        Clean modelling dataset ready for ML
    """
    # Remove rows with missing label
    df = df[df['y_next_up'].notna()].copy()
    
    # Remove rows with missing price features
    df = df[df['r_t_minus_1'].notna()].copy()
    df = df[df['r_t_minus_2'].notna()].copy()
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def main():
    print("=" * 70)
    print("Building Romgaz Modelling Dataset")
    print("Price Data + BNR FS Lexicon Sentiment")
    print("=" * 70)
    
    # Configuration
    PRICE_FILE = "romgaz_prices_events_with_sentiment_for_model.csv"
    SENTIMENT_FILE = "data/sng_daily_sentiment.csv"
    OUTPUT_FILE = "data/sng_modelling_dataset.csv"
    
    # Step 1: Load price data
    print("\n[1/4] Loading price data...")
    try:
        df_price = load_price_data(PRICE_FILE)
    except FileNotFoundError:
        print(f"Error: Price file not found: {PRICE_FILE}")
        return
    
    # Step 2: Load sentiment data
    print("\n[2/4] Loading daily sentiment data...")
    try:
        df_sentiment = pd.read_csv(SENTIMENT_FILE)
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
        print(f"  Loaded sentiment for {len(df_sentiment)} days")
    except FileNotFoundError:
        print(f"Error: Sentiment file not found: {SENTIMENT_FILE}")
        print("Please run build_daily_sentiment.py first.")
        return
    
    # Step 3: Merge
    print("\n[3/4] Merging price and sentiment data...")
    df_merged = merge_price_and_sentiment(df_price, df_sentiment)
    print(f"  Merged dataset: {len(df_merged)} rows")
    
    # Check overlap
    days_with_both = df_merged[df_merged['has_news']].shape[0]
    print(f"  Days with both price and news: {days_with_both}")
    
    # Step 4: Prepare final dataset
    print("\n[4/4] Preparing final modelling dataset...")
    df_final = prepare_modelling_dataset(df_merged)
    print(f"  Final dataset: {len(df_final)} rows")
    print(f"  Removed {len(df_merged) - len(df_final)} rows (missing features/label)")
    
    # Save
    Path("data").mkdir(exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  OK - Saved to: {OUTPUT_FILE}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("MODELLING DATASET SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal observations: {len(df_final)}")
    print(f"Date range: {df_final['date'].min().date()} to {df_final['date'].max().date()}")
    
    print(f"\nTarget variable (y_next_up):")
    print(f"  Up days (1): {(df_final['y_next_up'] == 1).sum()} ({(df_final['y_next_up'] == 1).mean() * 100:.1f}%)")
    print(f"  Down/flat (0): {(df_final['y_next_up'] == 0).sum()} ({(df_final['y_next_up'] == 0).mean() * 100:.1f}%)")
    
    print(f"\nNews coverage:")
    print(f"  Days with news: {df_final['has_news'].sum()} ({df_final['has_news'].mean() * 100:.1f}%)")
    print(f"  Days with lexicon hits: {df_final['has_lex_hits'].sum()} ({df_final['has_lex_hits'].mean() * 100:.1f}%)")
    
    if df_final['has_lex_hits'].sum() > 0:
        df_with_hits = df_final[df_final['has_lex_hits']]
        print(f"\nSentiment on days with lexicon hits (n={len(df_with_hits)}):")
        print(f"  Simple Index - Mean: {df_with_hits['simple_mean'].mean():.3f}, Std: {df_with_hits['simple_mean'].std():.3f}")
        print(f"  Score Index - Mean: {df_with_hits['score_mean'].mean():.3f}, Std: {df_with_hits['score_mean'].std():.3f}")
    
    # Feature summary
    print(f"\nAvailable features:")
    price_features = ['return', 'r_t_minus_1', 'r_t_minus_2']
    print(f"  Price features: {price_features}")
    
    sentiment_features = ['simple_mean', 'score_mean', 'news_count', 'sum_pos_total', 'sum_neg_total']
    available_sent = [f for f in sentiment_features if f in df_final.columns]
    print(f"  Main sentiment features: {available_sent}")
    
    print("\n" + "=" * 70)
    print("SUCCESS - Modelling dataset ready!")
    print(f"NEXT: Run train_models_with_lexicon_sentiment.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

