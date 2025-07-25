import argparse
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Fügt das Projektverzeichnis zum Python-Pfad hinzu
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Lädt die Daten
from src.utils.data_loaders.load_masakhanews import load_masakhanews_samples

# Definiert die Labels für MasakhaNEWS
MASAKHANEWS_LABELS = ['business', 'entertainment', 'health', 'politics', 'religion', 'sports', 'technology']

def create_distribution_plot(summary_df, output_dir):
    """Create distribution plot with consistent formatting matching QA visualization style."""
    
    # Set consistent formatting parameters
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'axes.titlesize': 16,
        'font.family': 'sans-serif'
    })
    
    # Color scheme matching QA visualization
    colors = ['#2d5477', '#4e8a86', '#76a990', '#316e7e', '#40726f', '#254562', '#628c77', '#285b68']
    
    # Prepare data for plotting
    summary_df['Fixed Prediction Strategy'] = summary_df['baseline_strategy'].str.replace('fixed_predict_', '')
    summary_df['Accuracy (%)'] = summary_df['accuracy'] * 100
    
    # Language mapping for better labels
    lang_map = {'en': 'English', 'sw': 'Swahili', 'ha': 'Hausa', 'te': 'Telugu'}
    summary_df['Language'] = summary_df['language'].map(lang_map)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.set_facecolor('white')
    
    # Create grouped bar plot
    languages = summary_df['Language'].unique()
    x = np.arange(len(languages))
    width = 0.12  # Thinner bars to prevent overlap
    multiplier = 0
    
    for i, label in enumerate(MASAKHANEWS_LABELS):
        offset = width * multiplier
        data = summary_df[summary_df['Fixed Prediction Strategy'] == label]['Accuracy (%)']
        if len(data) > 0:
            bars = ax.bar(x + offset, data, width, label=label.capitalize(), 
                         color=colors[i % len(colors)], alpha=0.8)
            # Add value labels on bars with 45 degree rotation
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10, rotation=45)
        multiplier += 1
    
    ax.set_ylabel('Accuracy (%)', fontsize=16)
    ax.set_xticks(x + width * (len(MASAKHANEWS_LABELS) - 1) / 2)
    ax.set_xticklabels(languages, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=False)
    ax.set_ylim(0, 20)  # Set y-axis range to 20% for classification
    
    # Remove grid and set clean background
    ax.grid(False)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file_path = os.path.join(output_dir, "classification_label_distribution_clustered_viz.png")
    fig.savefig(plot_file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Classification distribution visualization saved to {plot_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Simple Text Classification Baselines (Fixed Prediction for each class).")
    parser.add_argument("--langs", nargs='+', default=['en', 'sw', 'ha', 'te'], 
                        help="Languages to evaluate (default: en, sw, ha, te).")
    parser.add_argument("--split", type=str, default="test", 
                        help="Dataset split to use (default: test). Ensure consistency with main experiments.")
    parser.add_argument("--num_samples", type=int, default=80,
                        help="Number of samples to use from the specified split (default: 80).")
    parser.add_argument("--output_dir", type=str, default="/work/bbd6522/results/classification/simple_baselines_fixed", 
                        help="Directory for results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    overall_summary_list = []

    for lang_code in args.langs:
        print(f"--- Processing Classification Fixed Prediction Baselines for Language: {lang_code} ---")

        try:
            print(f"Loading up to {args.num_samples} samples for {lang_code} ({args.split} split) with seed {args.seed}...")
            samples_df = load_masakhanews_samples(
                lang_code=lang_code,
                split=args.split,
                num_samples=args.num_samples,
                seed=args.seed
            )

            if samples_df.empty:
                print(f"No MasakhaNEWS samples loaded for {lang_code} on split '{args.split}'. Skipping.")
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
        
        # Iteriert durch jedes mögliche MasakhaNEWS-Label
        for fixed_label_to_predict in MASAKHANEWS_LABELS:
            print(f"  Evaluating fixed prediction of: '{fixed_label_to_predict}' for {lang_code}")
            
            predicted_labels = [fixed_label_to_predict] * len(ground_truth_labels)

            # Stellt sicher, dass alle Klassen berücksichtigt werden
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

            # Detaillierter Klassifikationsbericht
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
            
            # Metriken pro Klasse aus dem Bericht hinzufügen
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
        
        # Add plotting functionality
        create_distribution_plot(summary_df, args.output_dir)
    else:
        print("No classification summary data generated.")

if __name__ == "__main__":
    main() 