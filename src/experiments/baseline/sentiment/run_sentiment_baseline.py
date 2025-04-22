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
    base_results_path: str,
    prompt_type: str # 'EnPrompt' or 'LrlPrompt'
):
    """
    Run the baseline sentiment analysis experiment for a specific model, language, and prompt type.
    """
    if samples_df.empty:
        print(f"WARNING: No samples provided for {lang_code} ({dataset_name}). Skipping experiment.")
        return

    # Determine prompt_in_lrl flag based on prompt_type
    prompt_in_lrl = True if prompt_type == 'LrlPrompt' else False

    try:
        print(f"\nProcessing {lang_code} baseline ({prompt_type}) sentiment ({dataset_name}) with {model_name}...")
        results_df = evaluate_sentiment_baseline(model_name, samples_df, lang_code, prompt_in_lrl=prompt_in_lrl)

        if results_df.empty:
            print(f"WARNING: No baseline results generated for {lang_code} ({dataset_name}, {prompt_type}) with {model_name}. Skipping metrics.")
            return

        # Calculate sentiment metrics (Dataset-wide)
        print("\nCalculating sentiment metrics...")
        metrics = calculate_sentiment_metrics(results_df)
        avg_accuracy = metrics.get('accuracy', float('nan'))
        avg_macro_f1 = metrics.get('macro_f1', float('nan'))
        # --- Retrieve per-class metrics --- 
        positive_precision = metrics.get('positive_precision', float('nan'))
        positive_recall = metrics.get('positive_recall', float('nan'))
        negative_precision = metrics.get('negative_precision', float('nan'))
        negative_recall = metrics.get('negative_recall', float('nan'))
        neutral_precision = metrics.get('neutral_precision', float('nan'))
        neutral_recall = metrics.get('neutral_recall', float('nan'))
        # --- End retrieve --- 

        print(f"\nOverall Metrics for {lang_code} ({dataset_name}) ({model_name}, Baseline - {prompt_type}):")
        print(f"  Accuracy: {avg_accuracy:.4f}")
        print(f"  Macro F1-Score: {avg_macro_f1:.4f}")
        print(f"  Positive (Prec/Recall): {positive_precision:.4f} / {positive_recall:.4f}")
        print(f"  Negative (Prec/Recall): {negative_precision:.4f} / {negative_recall:.4f}")
        print(f"  Neutral (Prec/Recall): {neutral_precision:.4f} / {neutral_recall:.4f}")

        # --- Save Results --- 
        # Include prompt type in subdirectory for organization
        results_subdir = os.path.join(base_results_path, prompt_type, lang_code)
        os.makedirs(results_subdir, exist_ok=True)

        # Save detailed results per sample
        model_name_short = model_name.split('/')[-1]
        # Include prompt type in filename
        output_filename = f"baseline_sentiment_{dataset_name}_{lang_code}_{prompt_type}_{model_name_short}.csv"
        
        # Add prompt_language column if it doesn't exist
        if 'prompt_language' not in results_df.columns:
            results_df['prompt_language'] = 'EN' if prompt_type == 'EnPrompt' else 'LRL'
            
        cols_to_save = ['id', 'text', 'ground_truth_label', 'predicted_label', 'language', 'prompt_language']
        cols_to_save = [col for col in cols_to_save if col in results_df.columns]
        results_df[cols_to_save].to_csv(os.path.join(results_subdir, output_filename), index=False)
        print(f"Detailed results saved to {results_subdir}/{output_filename}")

        # Save summary metrics
        summary = {
            'model': model_name,
            'language': lang_code,
            'dataset': dataset_name,
            'pipeline': 'baseline',
            'prompt_type': prompt_type,
            'accuracy': avg_accuracy,
            'macro_f1': avg_macro_f1,
            'positive_precision': positive_precision,
            'positive_recall': positive_recall,
            'negative_precision': negative_precision,
            'negative_recall': negative_recall,
            'neutral_precision': neutral_precision,
            'neutral_recall': neutral_recall
        }
        summary_df = pd.DataFrame([summary])
        # Include prompt type in summary path
        summary_path = os.path.join(base_results_path, prompt_type, "summaries")
        os.makedirs(summary_path, exist_ok=True)
        # Include prompt type in summary filename
        summary_filename = f"summary_baseline_sentiment_{dataset_name}_{lang_code}_{prompt_type}_{model_name_short}.csv"
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

    # Define sample sizes per language (use full dataset for now)
    sample_sizes = {
        "sw": None, # Set to None to load all
        "ha": None  # Set to None to load all
    }

    # Load data
    print(f"\n--- Loading {dataset_name.capitalize()} Data ---")
    sentiment_samples = {}
    for name, code in afrisenti_langs.items():
        # Load the full dataset first by setting num_samples=None
        print(f"Loading ALL samples for {name} ({code}) from validation split to calculate 10%...")
        full_samples_df = load_afrisenti_samples(code, num_samples=None, split='validation')
        
        if not full_samples_df.empty:
            total_loaded = len(full_samples_df)
            # Calculate 10% of samples, ensuring at least 1 sample if dataset is very small
            num_to_sample = max(1, int(total_loaded * 0.1)) 
            print(f"  Loaded {total_loaded} total samples. Sampling {num_to_sample} (10%)...")
            # Sample 10% of the data
            sentiment_samples[code] = full_samples_df.sample(n=num_to_sample, random_state=42) 
            print(f"  Finished sampling {len(sentiment_samples[code])} samples for {code}.")
        else:
            print(f"  No samples loaded for {code}, cannot sample.")
            sentiment_samples[code] = pd.DataFrame() # Store empty DataFrame

    # Define results path 
    base_results_path = "/work/bbd6522/results/sentiment/baseline" # Adjusted path
    os.makedirs(base_results_path, exist_ok=True)

    # Define prompt types to run
    prompt_types_to_run = ['EnPrompt', 'LrlPrompt']

    # Run experiments for both prompt types
    print(f"\n--- Running {dataset_name.capitalize()} Baseline Sentiment Experiments ---")
    for model_name in models:
        for lang_code, samples_df in sentiment_samples.items():
            for prompt_type in prompt_types_to_run:
                if samples_df.empty:
                    print(f"WARNING: No samples loaded for {lang_code} ({dataset_name}). Skipping baseline experiment for {model_name} ({prompt_type}).")
                    continue
                run_sentiment_experiment_baseline(
                    model_name,
                    samples_df,
                    lang_code,
                    dataset_name,
                    base_results_path,
                    prompt_type=prompt_type # Pass the prompt type
                )

if __name__ == "__main__":
    main() 