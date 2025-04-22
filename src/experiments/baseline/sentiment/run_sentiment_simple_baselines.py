#!/usr/bin/env python
# coding: utf-8

"""
Script to run simple sentiment analysis baselines:
1. Random Guess (among positive, negative, neutral)
2. Majority Class / Single Label Guess (predicting only positive, only negative, or only neutral)

Usage:
    python run_sentiment_simple_baselines.py [--balanced]
    
Arguments:
    --balanced: Optional flag to use balanced sampling (equal samples per class)
"""

import sys
import os
import pandas as pd
import numpy as np
import argparse

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Project specific imports
from src.utils.data_loaders.load_afrisenti import load_afrisenti_samples
from evaluation.sentiment_metrics import calculate_sentiment_metrics

# Define constants
EXPECTED_LABELS = ["positive", "negative", "neutral"]
DATASET_NAME = "afrisenti"
BASE_RESULTS_PATH = "/work/bbd6522/results/sentiment/simple_baselines" # New distinct path

def run_and_save_summary(
    samples_df: pd.DataFrame,
    lang_code: str,
    baseline_type: str, # e.g., "RandomGuess", "AlwaysPositive"
    predictions: pd.Series,
    is_balanced: bool = False
):
    """Calculates metrics and saves the summary CSV."""
    print(f"\n--- Calculating Metrics for: {baseline_type} ({lang_code}) ---")
    
    # Create a temporary DataFrame for metric calculation
    results_df = pd.DataFrame({
        'ground_truth_label': samples_df['label'],
        'predicted_label': predictions
    })
    
    metrics = calculate_sentiment_metrics(results_df)
    
    # Prepare summary dictionary
    summary = {
        'baseline_type': baseline_type,
        'language': lang_code,
        'dataset': DATASET_NAME,
        'pipeline': 'simple_baseline',
        'balanced_sampling': is_balanced,
        'accuracy': metrics.get('accuracy', float('nan')) ,
        'macro_f1': metrics.get('macro_f1', float('nan')),
        'positive_precision': metrics.get('positive_precision', float('nan')),
        'positive_recall': metrics.get('positive_recall', float('nan')),
        'negative_precision': metrics.get('negative_precision', float('nan')),
        'negative_recall': metrics.get('negative_recall', float('nan')),
        'neutral_precision': metrics.get('neutral_precision', float('nan')),
        'neutral_recall': metrics.get('neutral_recall', float('nan'))
    }
    
    print(f"Results for {baseline_type} ({lang_code}):")
    print(f"  Accuracy: {summary['accuracy']:.4f}")
    print(f"  Macro F1: {summary['macro_f1']:.4f}")
    print(f"  Positive (Prec/Recall): {summary['positive_precision']:.4f} / {summary['positive_recall']:.4f}")
    print(f"  Negative (Prec/Recall): {summary['negative_precision']:.4f} / {summary['negative_recall']:.4f}")
    print(f"  Neutral (Prec/Recall): {summary['neutral_precision']:.4f} / {summary['neutral_recall']:.4f}")
    
    # Save summary metrics
    summary_df = pd.DataFrame([summary])
    
    # Include balanced status in the path for easier analysis
    sampling_type = "balanced" if is_balanced else "original"
    summary_path = os.path.join(BASE_RESULTS_PATH, baseline_type, "summaries", sampling_type)
    os.makedirs(summary_path, exist_ok=True)
    
    summary_filename = f"summary_{baseline_type}_sentiment_{DATASET_NAME}_{lang_code}.csv"
    summary_df.to_csv(os.path.join(summary_path, summary_filename), index=False)
    print(f"Summary metrics saved to {summary_path}/{summary_filename}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run simple sentiment analysis baselines")
    parser.add_argument("--balanced", action="store_true", help="Use balanced sampling (equal samples per class)")
    args = parser.parse_args()
    
    # Languages to process
    afrisenti_langs = {
        "swahili": "sw",
        "hausa": "ha"
    }

    # Load data 
    print(f"\n--- Loading {DATASET_NAME.capitalize()} Data ---")
    print(f"Balanced sampling: {'ENABLED' if args.balanced else 'DISABLED'}")
    
    sentiment_samples = {}
    for name, code in afrisenti_langs.items():
        print(f"\nLoading samples for {name} ({code})...")
        # Use balanced sampling if requested, otherwise load all samples
        if args.balanced:
            # For balanced sampling, load with equal number from each class
            sentiment_samples[code] = load_afrisenti_samples(
                code, 
                num_samples=None, 
                balanced=True
            )
            sampling_desc = "balanced (equal per class)"
        else:
            # For normal sampling, load all samples (still shuffled)
            sentiment_samples[code] = load_afrisenti_samples(
                code, 
                num_samples=None, 
                balanced=False
            )
            sampling_desc = "original (all samples)"
            
        if sentiment_samples[code].empty:
             print(f"WARNING: No samples loaded for {code}. Skipping this language.")
        else:
            # Display label distribution to verify randomness
            label_counts = sentiment_samples[code]['label'].value_counts()
            print(f"\nLabel distribution for {name} ({code}) - {sampling_desc}:")
            for label, count in label_counts.items():
                percentage = 100 * count / len(sentiment_samples[code])
                print(f"  {label}: {count} samples ({percentage:.1f}%)")
                
            # Check the first few and last few samples to verify they're not sorted by label
            print(f"\nChecking randomness in {name} samples:")
            print("First 5 samples' labels:")
            print(sentiment_samples[code]['label'].head(5).values)
            print("Last 5 samples' labels:")
            print(sentiment_samples[code]['label'].tail(5).values)
            print("\n" + "-"*50)  
    
    # --- Run Baselines --- 
    print(f"\n--- Running Simple Baselines ({sampling_desc}) ---")
    for lang_code, samples_df in sentiment_samples.items():
        if samples_df.empty:
            continue
            
        print(f"\n== Processing Language: {lang_code} ==")
        n_samples = len(samples_df)
        
        # 1. Random Guess Baseline
        random_predictions = np.random.choice(EXPECTED_LABELS, size=n_samples)
        run_and_save_summary(samples_df, lang_code, "RandomGuess", random_predictions, args.balanced)
        
        # 2. Single Label Baselines
        for label_to_predict in EXPECTED_LABELS:
            baseline_name = f"Always{label_to_predict.capitalize()}" # e.g., AlwaysPositive
            single_label_predictions = [label_to_predict] * n_samples
            run_and_save_summary(samples_df, lang_code, baseline_name, single_label_predictions, args.balanced)

if __name__ == "__main__":
    main() 