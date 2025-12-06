"""
Full Pipeline Runner for Romgaz Stock Prediction Research

This script runs the complete pipeline:
1. Load BNR FS lexicon
2. Process news and compute sentiment
3. Aggregate to daily level
4. Merge with price data
5. Train and evaluate models

Run this after ensuring:
- BNR lexicon files (FS_positive_AI_scores.txt, FS_negative_AI_scores.txt) are present
- News data (bvb_sng_stiri/SNG_stiri_full.xlsx) is present
- Price data (romgaz_prices_events_with_sentiment_for_model.csv) is present
"""

import sys
import subprocess
from pathlib import Path


def run_script(script_path: str, description: str) -> bool:
    """
    Run a Python script and report success/failure.
    
    Args:
        script_path: Path to Python script
        description: Human-readable description
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed: {e}")
        return False


def main():
    print("=" * 70)
    print("ROMGAZ STOCK PREDICTION - FULL PIPELINE")
    print("BNR Financial Stability Lexicon Sentiment Analysis")
    print("=" * 70)
    
    # Define pipeline steps
    steps = [
        ("src/build_sng_sentiment_dataset.py", "Build News-Level Sentiment Dataset"),
        ("src/build_daily_sentiment.py", "Aggregate to Daily Sentiment"),
        ("src/build_sng_modelling_dataset.py", "Build Final Modelling Dataset"),
        ("src/train_models_with_lexicon_sentiment.py", "Train and Evaluate Models"),
    ]
    
    # Run each step
    success_count = 0
    
    for script_path, description in steps:
        if not Path(script_path).exists():
            print(f"\n✗ Script not found: {script_path}")
            break
        
        success = run_script(script_path, description)
        
        if success:
            success_count += 1
        else:
            print(f"\n✗ Pipeline stopped at: {description}")
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    
    print(f"\nCompleted {success_count}/{len(steps)} steps")
    
    if success_count == len(steps):
        print("\n✓ ✓ ✓ FULL PIPELINE COMPLETED SUCCESSFULLY! ✓ ✓ ✓")
        print("\nGenerated files:")
        print("  - data/sng_news_with_sentiment.csv")
        print("  - data/sng_daily_sentiment.csv")
        print("  - data/sng_modelling_dataset.csv")
        print("  - results/results_summary_*.csv")
        print("  - results/results_detailed_*.txt")
        print("\nNext steps:")
        print("  1. Review results in results/ folder")
        print("  2. Update research report in report/ folder with findings")
        print("  3. Create visualizations if needed")
    else:
        print("\n✗ Pipeline incomplete. Please check errors above.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

