# -*- coding: utf-8 -*-
import sys
import os
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Fügt das Projektverzeichnis zum Python-Pfad hinzu
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Lädt die Daten
from src.utils.data_loaders.load_xnli import load_xnli_samples

NLI_LABELS = ['entailment', 'neutral', 'contradiction']
# Mappt numerische XNLI-Labels
XNLI_NUMERIC_TO_STR_LABEL_MAP = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

def parse_args():
    # Parst Kommandozeilenargumente
    parser = argparse.ArgumentParser(description="Run NLI Fixed Prediction Baselines.")
    parser.add_argument("--langs", nargs='+', default=['en', 'ur', 'sw', 'fr'], 
                        help="Languages to test (default: en, ur, sw, fr for XNLI).")
    parser.add_argument("--split", type=str, default="test", 
                        help="Dataset split to use (default: test). XNLI typically uses 'validation' or 'test'.")
    parser.add_argument("--num_samples", type=int, default=80,
                        help="Number of samples to use from the specified split (default: 80).")
    parser.add_argument("--output_dir", type=str, default="/work/bbd6522/results/nli/simple_baselines_fixed",
                        help="Base output directory for results and summaries.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility in sampling.")
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    overall_summary_list = []

    for lang_code in args.langs:
        print(f"--- Processing NLI Fixed Prediction Baselines for Language: {lang_code} ---")

        try:
            print(f"Loading up to {args.num_samples} samples for {lang_code} ({args.split} split) with seed {args.seed}...")
            samples_df = load_xnli_samples(
                lang_code=lang_code,
                split=args.split,
                num_samples=args.num_samples,
                seed=args.seed
            )

            if samples_df.empty:
                print(f"No XNLI samples loaded for {lang_code} on split '{args.split}'. Skipping.")
                continue
            
            # Data cleaning and label normalization
            if 'label' not in samples_df.columns:
                print(f"No 'label' column found after loading XNLI samples for {lang_code}. Skipping.")
                continue

            samples_df['label'] = samples_df['label'].replace(XNLI_NUMERIC_TO_STR_LABEL_MAP)
            samples_df.dropna(subset=['label'], inplace=True)
            print(f"Successfully loaded and cleaned {len(samples_df)} XNLI samples for {lang_code}.")

        except Exception as e:
            print(f"Error loading or sampling data for {lang_code}: {e}")
            continue

        ground_truth_labels = samples_df['label'].astype(str).str.lower().str.strip().tolist()
        
        if not ground_truth_labels:
            print(f"No ground truth labels found for {lang_code} after processing. Skipping.")
            continue
        
        # Iterate through each possible NLI label and predict it for all samples
        for fixed_label_to_predict in NLI_LABELS:
            print(f"  Evaluating fixed prediction of: '{fixed_label_to_predict}' for {lang_code}")
            
            predicted_labels = [fixed_label_to_predict] * len(ground_truth_labels)

            # Using NLI_LABELS as the `labels` parameter for scikit-learn metrics
            # ensures all classes are considered, even if not present in ground_truth_labels for a small sample.
            accuracy = accuracy_score(ground_truth_labels, predicted_labels)
            macro_f1 = f1_score(ground_truth_labels, predicted_labels, labels=NLI_LABELS, average='macro', zero_division=0)
            macro_precision = precision_score(ground_truth_labels, predicted_labels, labels=NLI_LABELS, average='macro', zero_division=0)
            macro_recall = recall_score(ground_truth_labels, predicted_labels, labels=NLI_LABELS, average='macro', zero_division=0)
        
            weighted_f1 = f1_score(ground_truth_labels, predicted_labels, labels=NLI_LABELS, average='weighted', zero_division=0)
            weighted_precision = precision_score(ground_truth_labels, predicted_labels, labels=NLI_LABELS, average='weighted', zero_division=0)
            weighted_recall = recall_score(ground_truth_labels, predicted_labels, labels=NLI_LABELS, average='weighted', zero_division=0)

            print(f"    Lang: {lang_code}, Fixed Prediction: '{fixed_label_to_predict}'")
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    Macro F1: {macro_f1:.4f}, Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}")
            print(f"    Weighted F1: {weighted_f1:.4f}, Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}")

            # Detailed classification report (per-class P, R, F1)
            report_dict = classification_report(ground_truth_labels, predicted_labels, labels=NLI_LABELS, output_dict=True, zero_division=0)

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
            for label_name in NLI_LABELS:
                metrics = report_dict.get(label_name, {'precision': 0, 'recall': 0, 'f1-score': 0})
                summary_entry[f"{label_name}_precision"] = metrics['precision']
                summary_entry[f"{label_name}_recall"] = metrics['recall']
                summary_entry[f"{label_name}_f1-score"] = metrics['f1-score']
            
            overall_summary_list.append(summary_entry)

    summary_df = pd.DataFrame(overall_summary_list)
    if not summary_df.empty:
        summary_file_path = os.path.join(args.output_dir, "nli_fixed_prediction_summary.csv")
        summary_df.to_csv(summary_file_path, index=False, float_format='%.4f')
        print(f"\nOverall summary for NLI fixed prediction baselines saved to {summary_file_path}")
        print("\nSummary Table:")
        print(summary_df.to_string())
        
        # Fügt den Plot-Aufruf hinzu
        create_distribution_plot(summary_df, args.output_dir)
    else:
        print("\nNo NLI summary data generated.")

def create_distribution_plot(summary_df, output_dir):
    # Erstellt und speichert ein gruppiertes Balkendiagramm
    summary_df['Fixed Prediction Strategy'] = summary_df['baseline_strategy'].str.replace('fixed_predict_', '')
    summary_df['Accuracy (%)'] = summary_df['accuracy'] * 100

    plt.figure(figsize=(12, 7))
    
    sns.barplot(
        data=summary_df,
        x='language',
        y='Accuracy (%)',
        hue='Fixed Prediction Strategy',
        hue_order=NLI_LABELS
    )

    plt.ylabel("Accuracy (%)")
    plt.xlabel("Language")
    plt.title("NLI Fixed Prediction Accuracy by Language")
    plt.tight_layout()

    # Speichert das Diagramm
    plot_file_path = os.path.join(output_dir, "nli_label_distribution_clustered_viz.png")
    plt.savefig(plot_file_path, dpi=300)
    plt.close()

    print(f"New clustered visualization saved to {plot_file_path}")


if __name__ == "__main__":
    main() 