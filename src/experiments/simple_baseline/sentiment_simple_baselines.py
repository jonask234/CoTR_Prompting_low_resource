import argparse
import pandas as pd
import os
import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from typing import List, Dict, Any 
import numpy as np 
from collections import Counter # Added for potential majority baseline if needed in future, though not for fixed predict

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loaders.load_afrisenti import load_afrisenti_samples

SENTIMENT_LABELS = ['positive', 'neutral', 'negative'] # Standard sentiment labels

def main():
    parser = argparse.ArgumentParser(description="Run Simple Sentiment Analysis Baselines (Fixed Prediction for each class).")
    parser.add_argument("--langs", nargs='+', default=["sw", "ha", "pt"], 
                        help="Languages to evaluate.")
    parser.add_argument("--split", type=str, default="test", 
                        help="Dataset split to use (default: test). Ensure consistency.")
    parser.add_argument("--num_samples", type=int, default=80,
                        help="Number of samples to use from the specified split (default: 80).")
    parser.add_argument("--output_dir", type=str, default="/work/bbd6522/results/sentiment/simple_baselines_fixed", 
                        help="Directory for results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    overall_summary_list = []

    for lang_code in args.langs:
        print(f"--- Processing Sentiment Fixed Prediction Baselines for Language: {lang_code} ---")

        try:
            print(f"Loading up to {args.num_samples} samples for {lang_code} ({args.split} split) with seed {args.seed}...")
            samples_df = load_afrisenti_samples(
                lang_code=lang_code, 
                split=args.split, 
                num_samples=args.num_samples,
                seed=args.seed
            )
            
            if samples_df.empty:
                print(f"No sentiment samples loaded for {lang_code} on split '{args.split}'. Skipping.")
                continue
            
            if 'label' not in samples_df.columns:
                print(f"No 'label' column found after loading sentiment samples for {lang_code}. Skipping.")
                continue
            print(f"Successfully loaded {len(samples_df)} sentiment samples for {lang_code}.")

        except Exception as e:
            print(f"Error loading or sampling sentiment data for {lang_code}: {e}")
            continue

        ground_truth_labels = samples_df['label'].astype(str).str.lower().str.strip().tolist()
        
        if not ground_truth_labels:
            print(f"No ground truth labels found for {lang_code} after processing. Skipping.")
            continue
        
        # This baseline always predicts one of the known SENTIMENT_LABELS
        # It does not have an 'unable to answer' state, as it's a fixed prediction.
        for fixed_label_to_predict in SENTIMENT_LABELS:
            print(f"  Evaluating fixed prediction of: '{fixed_label_to_predict}' for {lang_code}")
            
            predicted_labels = [fixed_label_to_predict] * len(ground_truth_labels)

            accuracy = accuracy_score(ground_truth_labels, predicted_labels)
            # For metrics, ensure all SENTIMENT_LABELS are considered, even if not present in predictions or GT for a small sample
            macro_f1 = f1_score(ground_truth_labels, predicted_labels, labels=SENTIMENT_LABELS, average='macro', zero_division=0)
            macro_precision = precision_score(ground_truth_labels, predicted_labels, labels=SENTIMENT_LABELS, average='macro', zero_division=0)
            macro_recall = recall_score(ground_truth_labels, predicted_labels, labels=SENTIMENT_LABELS, average='macro', zero_division=0)
        
            weighted_f1 = f1_score(ground_truth_labels, predicted_labels, labels=SENTIMENT_LABELS, average='weighted', zero_division=0)
            weighted_precision = precision_score(ground_truth_labels, predicted_labels, labels=SENTIMENT_LABELS, average='weighted', zero_division=0)
            weighted_recall = recall_score(ground_truth_labels, predicted_labels, labels=SENTIMENT_LABELS, average='weighted', zero_division=0)

            print(f"    Lang: {lang_code}, Fixed Prediction: '{fixed_label_to_predict}'")
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    Macro F1: {macro_f1:.4f}, Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}")
            print(f"    Weighted F1: {weighted_f1:.4f}, Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}")

            report_dict = classification_report(ground_truth_labels, predicted_labels, labels=SENTIMENT_LABELS, output_dict=True, zero_division=0)

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
            
            for label_name in SENTIMENT_LABELS:
                metrics_for_label = report_dict.get(label_name, {'precision': 0, 'recall': 0, 'f1-score': 0})
                summary_entry[f"{label_name}_precision"] = metrics_for_label['precision']
                summary_entry[f"{label_name}_recall"] = metrics_for_label['recall']
                summary_entry[f"{label_name}_f1-score"] = metrics_for_label['f1-score']
            
            overall_summary_list.append(summary_entry)

    summary_df = pd.DataFrame(overall_summary_list)
    if not summary_df.empty:
        summary_file_path = os.path.join(args.output_dir, "sentiment_fixed_prediction_summary.csv")
        summary_df.to_csv(summary_file_path, index=False, float_format='%.4f')
        print(f"\nOverall summary for Sentiment fixed prediction baselines saved to {summary_file_path}")
        print("\nSummary Table:")
        print(summary_df.to_string())
    else:
        print("\nNo sentiment summary data generated.")

if __name__ == "__main__":
    main() 