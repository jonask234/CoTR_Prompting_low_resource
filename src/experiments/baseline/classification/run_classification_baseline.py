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
from datasets import load_dataset

# Project specific imports
from src.utils.data_loaders.load_news_classification import load_swahili_news, load_hausa_news
from src.experiments.baseline.classification.classification_baseline import evaluate_classification_baseline
# Reuse sentiment metrics (Acc, F1) but under classification name
from evaluation.sentiment_metrics import calculate_sentiment_metrics

def run_classification_experiment_baseline(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str,
    dataset_name: str, # e.g., 'masakhanews'
    base_results_path: str
):
    """
    Run the baseline text classification experiment for a specific model and language.
    """
    if samples_df.empty:
        print(f"WARNING: No samples provided for {lang_code} ({dataset_name}). Skipping baseline experiment for {model_name}.")
        return

    try:
        print(f"\nProcessing {lang_code} baseline classification ({dataset_name}) with {model_name}...")
        results_df = evaluate_classification_baseline(model_name, samples_df, lang_code)

        if results_df.empty:
            print(f"WARNING: No baseline results generated for {lang_code} ({dataset_name}) with {model_name}. Skipping metrics.")
            return

        # Calculate metrics (Dataset-wide)
        print("\nCalculating classification metrics...")
        metrics = calculate_sentiment_metrics(results_df) # Reuse function
        avg_accuracy = metrics.get('accuracy', float('nan'))
        avg_macro_f1 = metrics.get('macro_f1', float('nan'))

        print(f"\nOverall Metrics for {lang_code} ({dataset_name}) ({model_name}, Baseline):")
        print(f"  Accuracy: {avg_accuracy:.4f}")
        print(f"  Macro F1-Score: {avg_macro_f1:.4f}")

        # --- Save Results --- 
        lang_path = os.path.join(base_results_path, lang_code)
        os.makedirs(lang_path, exist_ok=True)

        # Save detailed results per sample
        model_name_short = model_name.split('/')[-1]
        # Ensure filename consistency
        output_filename = f"baseline_classification_{dataset_name}_{lang_code}_{model_name_short}.csv"
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
            # No translation quality for baseline
        }
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(base_results_path, "summaries")
        os.makedirs(summary_path, exist_ok=True)
        summary_filename = f"summary_baseline_classification_{dataset_name}_{lang_code}_{model_name_short}.csv"
        summary_df.to_csv(os.path.join(summary_path, summary_filename), index=False, float_format='%.4f')
        print(f"Summary metrics saved to {summary_path}/{summary_filename}")

    except Exception as e:
        print(f"ERROR during baseline classification experiment for {model_name}, {lang_code} ({dataset_name}): {e}")
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

    # --- Data Loading ---
    # Using MasakhaNEWS dataset for both languages
    datasets_to_run = {
        "masakhane_news_swa": {"lang_code": "swa", "loader": load_swahili_news, "dataset_hf_id": "masakhane/masakhane_news"},
        "masakhane_news_hau": {"lang_code": "hau", "loader": load_hausa_news, "dataset_hf_id": "masakhane/masakhane_news"}
    }
    
    # Define samples per language (~10% of total available samples)
    # Swahili: 476 total samples → 10% = ~48 samples
    # Hausa: 637 total samples → 10% = ~64 samples
    samples_per_lang = {
        "swa": 48,  # 10% of 476 Swahili samples
        "hau": 64   # 10% of 637 Hausa samples
    }
    
    hf_token = get_token()  # Still needed for model loading

    print(f"\n--- Loading MasakhaNEWS Classification Data (10% sample) ---")
    classification_samples = {} # Dictionary to store samples: {lang_code: (df, dataset_name)}
    
    for dataset_name, config in datasets_to_run.items():
        lang_code = config["lang_code"]
        loader_func = config["loader"]
        hf_id = config["dataset_hf_id"]
        
        # Get language-specific sample size
        num_samples = samples_per_lang.get(lang_code, 50)  # Default to 50 if not specified
        
        print(f"Loading samples for {lang_code} from {dataset_name} ({hf_id}) - Target: {num_samples} samples (~10%)")
        try:
            # Pass token to the loader and the appropriate number of samples
            samples_df = loader_func(num_samples=num_samples, token=hf_token)
            if samples_df is not None and not samples_df.empty:
                 # Store DataFrame and its corresponding dataset name
                classification_samples[lang_code] = (samples_df, dataset_name) 
                print(f"  Loaded {len(samples_df)} samples for {lang_code} ({dataset_name}).")
            else:
                print(f"  No samples loaded or returned for {lang_code} ({dataset_name}).")
        except Exception as e:
            print(f"ERROR loading data for {lang_code} ({dataset_name}): {e}")
            import traceback
            traceback.print_exc()

    # Define base results path for classification baseline results
    base_results_path = "/work/bbd6522/results/classification/baseline"
    os.makedirs(base_results_path, exist_ok=True)

    # Run experiments
    print(f"\n--- Running Baseline Classification Experiments ---")
    for model_name in models:
        # Iterate through the loaded samples
        for lang_code, (samples_df, dataset_name) in classification_samples.items():
            print(f"\nStarting experiment for model: {model_name}, language: {lang_code}, dataset: {dataset_name}")
            run_classification_experiment_baseline(
                model_name,
                samples_df,
                lang_code,
                dataset_name, # Pass the specific dataset name for this language
                base_results_path
            )

if __name__ == "__main__":
    main() 