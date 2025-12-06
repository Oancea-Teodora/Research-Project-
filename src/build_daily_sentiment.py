"""
Aggregate News-Level Sentiment to Daily Level

This script aggregates news-level sentiment metrics to daily (calendar date) level
for merging with stock price data.

Input: data/sng_news_with_sentiment.csv
Output: data/sng_daily_sentiment.csv
"""

import pandas as pd
from pathlib import Path
import numpy as np


def aggregate_daily_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate news-level sentiment to daily calendar date level.
    
    For each date with news, we compute:
    - news_count: number of news articles
    - lex_hits_docs: number of articles with at least one lexicon hit
    - Aggregated sentiment metrics: mean, sum, max, min of indices
    - Separate positive and negative totals
    
    Args:
        df_news: News-level DataFrame with sentiment metrics
        
    Returns:
        Daily-level DataFrame with aggregated sentiment
    """
    # Filter to rows with valid dates
    df_valid = df_news[df_news['published_date'].notna()].copy()
    
    print(f"Aggregating {len(df_valid)} news articles across dates...")
    
    # Group by date
    grouped = df_valid.groupby('published_date')
    
    daily_records = []
    
    for date, group_df in grouped:
        # Basic counts
        news_count = len(group_df)
        lex_hits_docs = group_df['has_lex_hits'].sum()
        
        # Filter to docs with hits for sentiment stats
        docs_with_hits = group_df[group_df['has_lex_hits']]
        
        if len(docs_with_hits) > 0:
            # Simple index stats
            simple_mean = docs_with_hits['simple_index'].mean()
            simple_sum = docs_with_hits['simple_index'].sum()
            simple_max = docs_with_hits['simple_index'].max()
            simple_min = docs_with_hits['simple_index'].min()
            simple_std = docs_with_hits['simple_index'].std()
            
            # Score index stats
            score_mean = docs_with_hits['score_index'].mean()
            score_sum = docs_with_hits['score_index'].sum()
            score_max = docs_with_hits['score_index'].max()
            score_min = docs_with_hits['score_index'].min()
            score_std = docs_with_hits['score_index'].std()
            
            # Raw positive/negative totals
            sum_pos_total = docs_with_hits['sum_pos'].sum()
            sum_neg_total = docs_with_hits['sum_neg'].sum()
            n_pos_total = docs_with_hits['n_pos'].sum()
            n_neg_total = docs_with_hits['n_neg'].sum()
            
            # Coverage
            avg_coverage = docs_with_hits['coverage'].mean()
            
        else:
            # No lexicon hits on this day
            simple_mean = simple_sum = simple_max = simple_min = simple_std = 0.0
            score_mean = score_sum = score_max = score_min = score_std = 0.0
            sum_pos_total = sum_neg_total = 0.0
            n_pos_total = n_neg_total = 0
            avg_coverage = 0.0
        
        daily_records.append({
            'date': date,
            'news_count': news_count,
            'lex_hits_docs': lex_hits_docs,
            'has_news': True,
            'has_lex_hits': lex_hits_docs > 0,
            # Simple index
            'simple_mean': simple_mean,
            'simple_sum': simple_sum,
            'simple_max': simple_max,
            'simple_min': simple_min,
            'simple_std': simple_std if pd.notna(simple_std) else 0.0,
            # Score index
            'score_mean': score_mean,
            'score_sum': score_sum,
            'score_max': score_max,
            'score_min': score_min,
            'score_std': score_std if pd.notna(score_std) else 0.0,
            # Raw totals
            'sum_pos_total': sum_pos_total,
            'sum_neg_total': sum_neg_total,
            'n_pos_total': n_pos_total,
            'n_neg_total': n_neg_total,
            'avg_coverage': avg_coverage,
        })
    
    df_daily = pd.DataFrame(daily_records)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.sort_values('date').reset_index(drop=True)
    
    return df_daily


def main():
    print("=" * 70)
    print("Aggregating News Sentiment to Daily Level")
    print("=" * 70)
    
    INPUT_PATH = "data/sng_news_with_sentiment.csv"
    OUTPUT_PATH = "data/sng_daily_sentiment.csv"
    
    # Load news-level sentiment
    print(f"\nLoading news-level sentiment from: {INPUT_PATH}")
    
    try:
        df_news = pd.read_csv(INPUT_PATH)
        df_news['published_date'] = pd.to_datetime(df_news['published_date'])
        print(f"  OK - Loaded {len(df_news)} news articles")
    except FileNotFoundError:
        print(f"Error: File not found: {INPUT_PATH}")
        print("Please run build_sng_sentiment_dataset.py first.")
        return
    
    # Aggregate
    print("\nAggregating to daily level...")
    df_daily = aggregate_daily_sentiment(df_news)
    
    # Save
    print(f"\nSaving to: {OUTPUT_PATH}")
    Path("data").mkdir(exist_ok=True)
    df_daily.to_csv(OUTPUT_PATH, index=False)
    print("  OK - Saved")
    
    # Summary
    print("\n" + "=" * 70)
    print("DAILY SENTIMENT SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal days with news: {len(df_daily)}")
    print(f"Days with lexicon hits: {df_daily['has_lex_hits'].sum()}")
    print(f"\nDate range: {df_daily['date'].min().date()} to {df_daily['date'].max().date()}")
    
    print(f"\nNews per day:")
    print(f"  Mean: {df_daily['news_count'].mean():.1f}")
    print(f"  Median: {df_daily['news_count'].median():.0f}")
    print(f"  Max: {df_daily['news_count'].max()}")
    
    days_with_hits = df_daily[df_daily['has_lex_hits']]
    if len(days_with_hits) > 0:
        print(f"\nFor days with lexicon hits (n={len(days_with_hits)}):")
        print(f"  Simple Index Mean: {days_with_hits['simple_mean'].mean():.3f} ± {days_with_hits['simple_mean'].std():.3f}")
        print(f"  Score Index Mean: {days_with_hits['score_mean'].mean():.3f} ± {days_with_hits['score_mean'].std():.3f}")
    
    print("\n" + "=" * 70)
    print("SUCCESS - Daily sentiment aggregation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

