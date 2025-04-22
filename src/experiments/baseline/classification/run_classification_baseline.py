import sys
import os

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Add project root to Python path for evaluation module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from huggingface_hub import login

# Try to import config from various locations
try:
    from utils.config import get_token
except ImportError:
    try:
        from config import get_token
    except ImportError:
        # Fallback to define a basic get_token function if the module can't be found
        def get_token():
            print("WARNING: Could not import get_token, using environment variable HF_TOKEN instead")
            return os.environ.get("HF_TOKEN", "")

# Project specific imports
from utils.data_loaders.load_masakhanews import load_masakhanews_samples
from src.experiments.baseline.classification.classification_baseline import evaluate_classification_baseline
from evaluation.classification_metrics import calculate_classification_metrics

# Define the run_classification_experiment_baseline function here, rather than importing it
def run_classification_experiment_baseline(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str,
    dataset_name: str,
    base_results_path: str,
    prompt_type: str, # 'EnPrompt' or 'LrlPrompt'
    # Keep use_lrl_ground_truth even if not directly used by this func
    use_lrl_ground_truth: bool = False 
):
    """
    Run the baseline text classification experiment for a specific model and language.
    """
    print(f"Running classification baseline for {model_name} on {dataset_name} ({lang_code})")
    print(f"LRL ground truth evaluation: {use_lrl_ground_truth}")
    
    if samples_df.empty:
        print(f"WARNING: No samples provided for {lang_code} ({dataset_name}). Skipping baseline experiment.")
        return

    prompt_in_lrl = True if prompt_type == 'LrlPrompt' else False

    try:
        print(f"\nProcessing {lang_code} baseline ({prompt_type}) classification ({dataset_name}) with {model_name}...")
        results_df = evaluate_classification_baseline(
            model_name, samples_df, lang_code, 
            use_lrl_ground_truth=use_lrl_ground_truth, # Pass this through
            prompt_in_lrl=prompt_in_lrl
        )

        if results_df.empty:
            print(f"WARNING: No results generated for {lang_code}, {prompt_type} with {model_name}.")
            return

        # Calculate metrics (Dataset-wide)
        print("\nCalculating classification metrics...")
        metrics = calculate_classification_metrics(results_df)
        avg_accuracy = metrics.get('accuracy', float('nan'))
        avg_macro_f1 = metrics.get('macro_f1', float('nan'))
        
        # Collect per-class metrics dynamically
        per_class_metrics = {k: v for k, v in metrics.items() if k not in ['accuracy', 'macro_f1']}

        print(f"\nOverall Metrics for {lang_code} ({dataset_name}) ({model_name}, Baseline - {prompt_type}):")
        print(f"  LRL Ground Truth Eval Flag: {use_lrl_ground_truth}")
        print(f"  Accuracy: {avg_accuracy:.4f}")
        print(f"  Macro F1-Score: {avg_macro_f1:.4f}")
        # Print per-class metrics
        for label, value in per_class_metrics.items():
             # Simple formatting assuming label_precision/label_recall pattern
             metric_name = label.split('_')[-1].capitalize()
             class_name = label.replace(f'_{metric_name.lower()}', '') 
             if metric_name == 'Precision':
                 print(f"  {class_name.capitalize()} (Prec): {value:.4f}", end='')
             elif metric_name == 'Recall':
                 print(f" / (Recall): {value:.4f}")
        print() # Newline after per-class metrics

        # Save results
        results_subdir = os.path.join(base_results_path, prompt_type, lang_code)
        os.makedirs(results_subdir, exist_ok=True)

        model_name_short = model_name.split('/')[-1]
        lrl_suffix = "_lrleval" if use_lrl_ground_truth else "" # Keep suffix based on flag
        # Include prompt type in filename
        output_filename = f"baseline_classification_{dataset_name}_{lang_code}_{prompt_type}{lrl_suffix}_{model_name_short}.csv"
        
        # Add prompt_language column if it doesn't exist
        if 'prompt_language' not in results_df.columns:
            results_df['prompt_language'] = 'EN' if prompt_type == 'EnPrompt' else 'LRL'
            
        cols_to_save = ['id', 'original_text', 'ground_truth_label', 'predicted_label', 'language', 'lrl_evaluation', 'prompt_language']
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
            'lrl_evaluation': use_lrl_ground_truth, # Keep flag in summary
            'accuracy': avg_accuracy,
            'macro_f1': avg_macro_f1,
            **per_class_metrics 
        }
        
        summary_df = pd.DataFrame([summary])
        # Include prompt type in summary path
        summary_path = os.path.join(base_results_path, prompt_type, "summaries")
        os.makedirs(summary_path, exist_ok=True)
        # Include prompt type and lrl suffix in summary filename
        summary_filename = f"summary_baseline_{dataset_name}_{lang_code}_{prompt_type}{lrl_suffix}_{model_name_short}.csv"
        summary_df.to_csv(os.path.join(summary_path, summary_filename), index=False)
        print(f"Summary metrics saved to {summary_path}/{summary_filename}")

    except Exception as e:
        print(f"ERROR during baseline classification experiment for {model_name}, {lang_code} ({dataset_name}): {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run classification baselines."""
    # Setup
    token = get_token()
    login(token=token)
    
    models = [
        "Qwen/Qwen2-7B",
        "CohereForAI/aya-23-8B"
    ]
    
    # --- MasakhaNEWS Dataset (English ground truth) ---
    masakhanews_langs = {
        "swahili": "swa",
        "hausa": "hau"
    }
    masakhanews_name = "masakhanews"
    masakhanews_samples = {}
    
    # Load MasakhaNEWS data (English ground truth labels) from local files
    print(f"\n--- Loading {masakhanews_name.capitalize()} Data ---")
    for name, code in masakhanews_langs.items():
        print(f"Loading ALL samples for {name} ({code}) from local test split to calculate 10%...")
        # Load the full dataset first by setting num_samples=None
        full_samples_df = load_masakhanews_samples(code, num_samples=None, split='test')
        
        if not full_samples_df.empty:
            total_loaded = len(full_samples_df)
            # Calculate 10% of samples, ensuring at least 1 sample if dataset is very small
            num_to_sample = max(1, int(total_loaded * 0.1)) 
            print(f"  Loaded {total_loaded} total samples. Sampling {num_to_sample} (10%)...")
            # Sample 10% of the data
            masakhanews_samples[code] = full_samples_df.sample(n=num_to_sample, random_state=42) 
            print(f"  Finished sampling {len(masakhanews_samples[code])} samples for {code}.")
        else:
            print(f"  No samples loaded for {code}, cannot sample.")
            masakhanews_samples[code] = pd.DataFrame() # Store empty DataFrame
    
    # Define results path 
    base_results_path = "/work/bbd6522/results/classification/baseline"
    os.makedirs(base_results_path, exist_ok=True)
    
    # Define prompt types to run
    prompt_types_to_run = ['EnPrompt', 'LrlPrompt']

    # Run experiments on MasakhaNEWS (English ground truth)
    print(f"\n--- Running {masakhanews_name.capitalize()} Classification Baseline Experiments ---")
    print(f"Languages being evaluated: {', '.join(masakhanews_langs.keys())} (English removed)")
    for model_name in models:
        for lang_code, samples_df in masakhanews_samples.items():
            for prompt_type in prompt_types_to_run:
                if samples_df.empty:
                    print(f"WARNING: No samples loaded for {lang_code} ({masakhanews_name}). Skipping baseline experiment for {model_name} ({prompt_type}).")
                    continue
                run_classification_experiment_baseline(
                    model_name,
                    samples_df,
                    lang_code,
                    masakhanews_name,
                    base_results_path, 
                    prompt_type=prompt_type, # Pass prompt type
                    # Pass use_lrl_ground_truth - always False for MasakhaNEWS
                    use_lrl_ground_truth=False 
                )

if __name__ == "__main__":
    main() 