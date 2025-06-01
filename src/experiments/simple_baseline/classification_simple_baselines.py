import argparse
import pandas as pd
import os
import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from typing import List, Dict, Any
import numpy as np # Added for potential future use and consistency

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the specific data loader for MasakhaNEWS
from src.utils.data_loaders.load_masakhanews import load_masakhanews_samples

# Define the known labels for MasakhaNEWS - this should be the single source of truth
MASAKHANEWS_LABELS = ['health', 'religion', 'politics', 'sports', 'local', 'business', 'entertainment']

def main():
    parser = argparse.ArgumentParser(description="Run Simple Text Classification Baselines (Fixed Prediction for each class).")
    parser.add_argument("--langs", nargs='+', default=['en', 'sw', 'ha', 'te'], 
                        help="Languages to evaluate (default: en, sw, ha, te).")
    parser.add_argument("--split", type=str, default="test", 
                        help="Dataset split to use (default: test). Ensure consistency with main experiments.")
    parser.add_argument("--sample_percentage", type=float, default=10.0,
                        help="Percentage of samples to use from the specified split (default: 10.0 for 10%%).")
    parser.add_argument("--output_dir", type=str, default="/work/bbd6522/results/classification/simple_baselines_fixed", 
                        help="Directory for results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    overall_summary_list = []

    for lang_code in args.langs:
        print(f"--- Processing Classification Fixed Prediction Baselines for Language: {lang_code} ---")

        try:
            # Load all samples for the given language and split to determine total size for percentage sampling
            print(f"Determining total number of '{args.split}' MasakhaNEWS samples for {lang_code}...")
            temp_all_samples_df = load_masakhanews_samples(lang_code=lang_code, split=args.split, num_samples=None) # Load all
            
            if temp_all_samples_df.empty:
                print(f"No MasakhaNEWS samples found for {lang_code}, split '{args.split}' when checking total size. Skipping.")
                continue
            total_available_samples = len(temp_all_samples_df)
            del temp_all_samples_df # free memory

            num_to_sample = int(total_available_samples * (args.sample_percentage / 100.0))
            if num_to_sample == 0 and total_available_samples > 0 : # Ensure at least 1 sample if percentage is too low for small datasets
                num_to_sample = 1 
            
            if num_to_sample == 0:
                 print(f"Calculated 0 samples to select for {lang_code} with {args.sample_percentage}%% from {total_available_samples}. Skipping.")
                 continue

            print(f"Loading {num_to_sample} ({args.sample_percentage}%% of {total_available_samples}) '{args.split}' MasakhaNEWS samples for {lang_code}...")
            # Now load the exact number of samples required.
            # load_masakhanews_samples should handle the seed for reproducible sampling if num_samples is given.
            samples_df = load_masakhanews_samples(
                lang_code=lang_code, 
                split=args.split, 
                num_samples=num_to_sample
            )
            
            # The manual sampling `all_samples_df.sample(n=num_to_sample, random_state=args.seed)` is replaced
            # by passing num_samples directly to the loader.

            if samples_df.empty:
                print(f"No MasakhaNEWS samples loaded for {lang_code} on split '{args.split}' after requesting {num_to_sample} samples. Skipping.")
                continue
            
            if 'label' not in samples_df.columns:
                print(f"No 'label' column found after loading MasakhaNEWS samples for {lang_code}. Skipping.")
                continue
            print(f"Successfully loaded {len(samples_df)} MasakhaNEWS samples for {lang_code}.")

        except Exception as e:
            print(f"Error loading or sampling data for {lang_code}: {e}")
            continue

        ground_truth_labels = samples_df['label'].astype(str).str.lower().str.strip().tolist()
        
        if not ground_truth_labels:
            print(f"No ground truth labels found for {lang_code} after processing. Skipping.")
            continue
        
        # Iterate through each possible MasakhaNEWS label and predict it for all samples
        for fixed_label_to_predict in MASAKHANEWS_LABELS:
            print(f"  Evaluating fixed prediction of: '{fixed_label_to_predict}' for {lang_code}")
            
            predicted_labels = [fixed_label_to_predict] * len(ground_truth_labels)

            # Using MASAKHANEWS_LABELS as the `labels` parameter for scikit-learn metrics
            # ensures all classes are considered, even if not present in ground_truth_labels or predicted_labels for a small sample.

            accuracy = accuracy_score(ground_truth_labels, predicted_labels)
            macro_f1 = f1_score(ground_truth_labels, predicted_labels, labels=MASAKHANEWS_LABELS, average='macro', zero_division=0)
            macro_precision = precision_score(ground_truth_labels, predicted_labels, labels=MASAKHANEWS_LABELS, average='macro', zero_division=0)
            macro_recall = recall_score(ground_truth_labels, predicted_labels, labels=MASAKHANEWS_LABELS, average='macro', zero_division=0)
        
            weighted_f1 = f1_score(ground_truth_labels, predicted_labels, labels=MASAKHANEWS_LABELS, average='weighted', zero_division=0)
            weighted_precision = precision_score(ground_truth_labels, predicted_labels, labels=MASAKHANEWS_LABELS, average='weighted', zero_division=0)
            weighted_recall = recall_score(ground_truth_labels, predicted_labels, labels=MASAKHANEWS_LABELS, average='weighted', zero_division=0)

            print(f"    Lang: {lang_code}, Fixed Prediction: '{fixed_label_to_predict}'")
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    Macro F1: {macro_f1:.4f}, Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}")
            print(f"    Weighted F1: {weighted_f1:.4f}, Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}")

            # Detailed classification report (per-class P, R, F1)
            report_dict = classification_report(ground_truth_labels, predicted_labels, labels=MASAKHANEWS_LABELS, output_dict=True, zero_division=0)

            summary_entry = {
            "language": lang_code,
                "baseline_strategy": f"fixed_predict_{fixed_label_to_predict}",
            "num_samples": len(samples_df),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "weighted_f1": weighted_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
        }
            
            # Add per-class metrics from the report
            for label_name in MASAKHANEWS_LABELS:
                metrics = report_dict.get(label_name, {'precision': 0, 'recall': 0, 'f1-score': 0})
                summary_entry[f"{label_name}_precision"] = metrics['precision']
                summary_entry[f"{label_name}_recall"] = metrics['recall']
                summary_entry[f"{label_name}_f1-score"] = metrics['f1-score']
            
            overall_summary_list.append(summary_entry)

    summary_df = pd.DataFrame(overall_summary_list)
    summary_file_path = os.path.join(args.output_dir, "classification_fixed_prediction_summary.csv")
    summary_df.to_csv(summary_file_path, index=False, float_format='%.4f')
    print(f"Overall summary for Text Classification fixed prediction baselines saved to {summary_file_path}")
    if not summary_df.empty:
        print("Summary Table:")
        print(summary_df.to_string())

if __name__ == "__main__":
    main() 