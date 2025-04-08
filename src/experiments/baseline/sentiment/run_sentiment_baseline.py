import sys
import os

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from huggingface_hub import login
from config import get_token

# Project specific imports
from src.utils.data_loaders.load_afrisenti import load_afrisenti_samples
from src.experiments.baseline.sentiment.sentiment_baseline import evaluate_sentiment_baseline
from evaluation.sentiment_metrics import calculate_sentiment_metrics

def run_sentiment_experiment_baseline(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str,
    dataset_name: str, # e.g., 'afrisenti'
    base_results_path: str
):
    """
    Run the baseline sentiment analysis experiment for a specific model and language.
    """
    if samples_df.empty:
        print(f"WARNING: No samples provided for {lang_code} ({dataset_name}). Skipping baseline experiment for {model_name}.")
        return

    try:
        print(f"\nProcessing {lang_code} baseline sentiment ({dataset_name}) with {model_name}...")
        results_df = evaluate_sentiment_baseline(model_name, samples_df, lang_code)

        if results_df.empty:
            print(f"WARNING: No baseline results generated for {lang_code} ({dataset_name}) with {model_name}. Skipping metrics.")
            return

        # Calculate metrics (Dataset-wide)
        print("\nCalculating sentiment metrics...")
        metrics = calculate_sentiment_metrics(results_df)
        avg_accuracy = metrics.get('accuracy', float('nan'))
        avg_macro_f1 = metrics.get('macro_f1', float('nan'))

        # Add metrics to each row (optional, for detailed CSV, but less common for summary)
        # results_df['accuracy'] = results_df.apply(lambda row: 1.0 if row['ground_truth_label'] == row['predicted_label'] else 0.0, axis=1)

        print(f"\nOverall Metrics for {lang_code} ({dataset_name}) ({model_name}, Baseline):")
        print(f"  Accuracy: {avg_accuracy:.4f}")
        print(f"  Macro F1-Score: {avg_macro_f1:.4f}")

        # --- Save Results --- 
        lang_path = os.path.join(base_results_path, lang_code)
        os.makedirs(lang_path, exist_ok=True)

        # Save detailed results per sample
        model_name_short = model_name.split('/')[-1]
        output_filename = f"baseline_sentiment_{dataset_name}_{lang_code}_{model_name_short}.csv"
        results_df.to_csv(os.path.join(lang_path, output_filename), index=False)
        print(f"Detailed results saved to {lang_path}/{output_filename}")

        # Save summary metrics
        summary = {
            'model': model_name,
            'language': lang_code,
            'dataset': dataset_name,
            'pipeline': 'baseline',
            'accuracy': avg_accuracy,
            'macro_f1': avg_macro_f1
        }
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(base_results_path, "summaries")
        os.makedirs(summary_path, exist_ok=True)
        summary_filename = f"summary_baseline_sentiment_{dataset_name}_{lang_code}_{model_name_short}.csv"
        summary_df.to_csv(os.path.join(summary_path, summary_filename), index=False)
        print(f"Summary metrics saved to {summary_path}/{summary_filename}")

    except Exception as e:
        print(f"ERROR during baseline sentiment experiment for {model_name}, {lang_code} ({dataset_name}): {e}")
        import traceback
        traceback.print_exc()

def main():
    # Setup
    token = get_token()
    login(token=token)

    models = [
        "Qwen/Qwen2-7B",
        "CohereForAI/aya-23-8B"
    ]

    # AfriSenti languages
    afrisenti_langs = {
        "swahili": "sw",
        "hausa": "ha"
    }
    dataset_name = "afrisenti"
    # num_samples = 500 # Remove fixed number

    # Define sample sizes per language (approx 10%)
    sample_sizes = {
        "sw": 2516, # Reverted to ~10%
        "ha": 1510  # Reverted to ~10%
    }

    # Load data
    print(f"\n--- Loading {dataset_name.capitalize()} Data ---")
    sentiment_samples = {}
    for name, code in afrisenti_langs.items():
        num_samples = sample_sizes.get(code) # Get specific size
        print(f"Loading {num_samples if num_samples else 'all'} samples for {name} ({code})...")
        sentiment_samples[code] = load_afrisenti_samples(code, num_samples)

    # Define results path - Point to subfolder within existing results
    base_results_path = "/work/bbd6522/results/sentiment/baseline" # Updated path
    os.makedirs(base_results_path, exist_ok=True)

    # Run experiments
    print(f"\n--- Running {dataset_name.capitalize()} Baseline Sentiment Experiments ---")
    for model_name in models:
        for lang_code, samples_df in sentiment_samples.items():
            run_sentiment_experiment_baseline(
                model_name,
                samples_df,
                lang_code,
                dataset_name,
                base_results_path
            )

if __name__ == "__main__":
    main() 