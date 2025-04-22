#!/usr/bin/env python
# coding: utf-8

"""
Script to run simple classification baselines for MasakhaNEWS:
1. Random Guess (among all unique labels found in the data)
2. Single Label Guess (predicting only one specific label for all samples)
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Project specific imports
from src.utils.data_loaders.load_masakhanews import load_masakhanews_samples
# Use the updated metrics calculation function
from evaluation.classification_metrics import calculate_classification_metrics 

# Define constants
DATASET_NAME = "masakhanews"
# Save results in a dedicated folder within classification results
BASE_RESULTS_PATH = "/work/bbd6522/results/classification/simple_baselines" 

def run_and_save_summary(
    samples_df: pd.DataFrame,
    lang_code: str,
    baseline_type: str, # e.g., "RandomGuess", "AlwaysSports"
    predictions: pd.Series,
    all_labels: list # Pass the list of all possible labels for context
):
    """Calculates metrics and saves the summary CSV."""
    print(f"\n--- Calculating Metrics for: {baseline_type} ({lang_code}) ---")
    
    # Create a temporary DataFrame for metric calculation
    results_df = pd.DataFrame({
        'ground_truth_label': samples_df['label'],
        'predicted_label': predictions
    })
    
    metrics = calculate_classification_metrics(results_df)
    
    # Prepare summary dictionary - include all potential labels for consistent columns
    summary = {
        'baseline_type': baseline_type,
        'language': lang_code,
        'dataset': DATASET_NAME,
        'pipeline': 'simple_baseline',
        'accuracy': metrics.get('accuracy', float('nan')) ,
        'macro_f1': metrics.get('macro_f1', float('nan'))
    }
    
    # Add per-class metrics, ensuring all possible labels have a column
    for label in sorted(all_labels):
        summary[f'{label}_precision'] = metrics.get(f'{label}_precision', 0.0) # Default to 0 if not in this specific result
        summary[f'{label}_recall'] = metrics.get(f'{label}_recall', 0.0)
        
    print(f"Results for {baseline_type} ({lang_code}):")
    print(f"  Accuracy: {summary['accuracy']:.4f}")
    print(f"  Macro F1: {summary['macro_f1']:.4f}")
    # Print per-class (optional, can be long)
    # for label in sorted(all_labels):
    #     print(f"  {label.capitalize()} (Prec/Recall): {summary[f'{label}_precision']:.4f} / {summary[f'{label}_recall']:.4f}")

    # Save summary metrics
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(BASE_RESULTS_PATH, baseline_type, "summaries")
    os.makedirs(summary_path, exist_ok=True)
    summary_filename = f"summary_{baseline_type}_classification_{DATASET_NAME}_{lang_code}.csv"
    summary_df.to_csv(os.path.join(summary_path, summary_filename), index=False)
    print(f"Summary metrics saved to {summary_path}/{summary_filename}")

def main():
    # Languages to process
    masakhanews_langs = {
        "swahili": "swa",
        "hausa": "hau",
        "english": "eng"
    }

    # Load data (load all available test samples for accurate baseline stats)
    print(f"\n--- Loading {DATASET_NAME.capitalize()} Data ---")
    classification_samples = {}
    all_available_labels = set() # Collect all labels across languages
    
    for name, code in masakhanews_langs.items():
        print(f"Loading all test samples for {name} ({code})...")
        # Load full test split from local TSV
        df = load_masakhanews_samples(code, num_samples=None, split='test') 
        if not df.empty:
            classification_samples[code] = df
            # Update the set of all unique labels found
            all_available_labels.update(df['label'].unique())
            print(f"  Loaded {len(df)} samples for {code}. Unique labels so far: {len(all_available_labels)}")
        else:
             print(f"WARNING: No samples loaded for {code}. Skipping this language.")
             
    if not all_available_labels:
        print("ERROR: No valid labels found in any loaded dataset. Cannot run simple baselines.")
        return
        
    all_labels_list = sorted(list(all_available_labels))
    print(f"\nFound the following unique labels across all loaded languages: {all_labels_list}")
    
    # --- Run Baselines --- 
    print(f"\n--- Running Simple Baselines --- ")
    for lang_code, samples_df in classification_samples.items():
        if samples_df.empty:
            continue
            
        print(f"\n== Processing Language: {lang_code} ==")
        n_samples = len(samples_df)
        
        # 1. Random Guess Baseline (using all discovered labels)
        random_predictions = np.random.choice(all_labels_list, size=n_samples)
        run_and_save_summary(samples_df, lang_code, "RandomGuess", random_predictions, all_labels_list)
        
        # 2. Single Label Baselines (for each discovered label)
        for label_to_predict in all_labels_list:
            baseline_name = f"Always{label_to_predict.capitalize()}" # e.g., AlwaysSports
            single_label_predictions = [label_to_predict] * n_samples
            run_and_save_summary(samples_df, lang_code, baseline_name, single_label_predictions, all_labels_list)

if __name__ == "__main__":
    main() 