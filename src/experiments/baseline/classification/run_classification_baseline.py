import sys
import os
import argparse
import time
import pandas as pd
import numpy as np
import torch
import json # Added for parameter logging if needed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List # Added List

# Add project root to Python path BEFORE other project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import baseline classification functions
from src.experiments.baseline.classification.classification_baseline import (
    initialize_model,
    evaluate_classification_baseline
)

# Import data loader (e.g., for a news classification dataset like MasakhaNEWS)
# Corrected import to use load_masakhanews
from src.utils.data_loaders.load_masakhanews import load_masakhanews_samples 

# Import metrics calculator
from evaluation.classification_metrics import calculate_classification_metrics

# Import HuggingFace utilities
from huggingface_hub import login
from config import get_token

# --- Define Default Parameters (consistent with CoTR runner where applicable) ---
DEFAULT_GENERATION_PARAMS = {
    "temperature": 0.05,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 20,
    "repetition_penalty": 1.1
}

# Language-specific overrides for baseline (if different from CoTR defaults)
LANGUAGE_SPECIFIC_BASELINE_PARAMS = {
    "sw": {"temperature": 0.05, "repetition_penalty": 1.15},
    "ha": {"temperature": 0.05, "repetition_penalty": 1.15},
    "te": {"temperature": 0.05, "repetition_penalty": 1.2}
}

# Model-specific adjustments for baseline
MODEL_SPECIFIC_BASELINE_ADJUSTMENTS = {
    "CohereLabs/aya-expanse-8b": {"temperature_factor": 1.0},
    "Qwen/Qwen2.5-7B-Instruct": {"top_p": 0.85, "top_k": 35, "temperature": 0.05}
}

# Define expected labels - these should match what the model is trained/prompted to output
# For MasakhaNEWS, these are the actual categories:
POSSIBLE_LABELS_EN = ['health', 'religion', 'politics', 'sports', 'local', 'business', 'entertainment']
# OLD: POSSIBLE_LABELS_EN = ["topic_a", "topic_b", "topic_c", "other"]

# Define language names dictionary
LANG_NAMES = {
    "en": "English",
    "sw": "Swahili",
    "ha": "Hausa",
    "te": "Telugu"
}

def get_effective_baseline_params(lang_code: str, model_name: str) -> Dict:
    """Merge default, language-specific, and model-specific parameters for baseline."""
    effective_params = DEFAULT_GENERATION_PARAMS.copy()
    if lang_code in LANGUAGE_SPECIFIC_BASELINE_PARAMS:
        effective_params.update(LANGUAGE_SPECIFIC_BASELINE_PARAMS[lang_code])
    
    model_adjustments = MODEL_SPECIFIC_BASELINE_ADJUSTMENTS.get(model_name, {})
    for key, value in model_adjustments.items():
        if key.endswith("_factor") and key[:-len("_factor")] in effective_params:
            param_to_adjust = key[:-len("_factor")]
            effective_params[param_to_adjust] *= value
            if param_to_adjust == "temperature":
                effective_params[param_to_adjust] = max(0.01, effective_params[param_to_adjust])
        else:
            effective_params[key] = value # Direct override
    return effective_params

def run_baseline_classification_experiment(
    model_name: str,
    tokenizer: Any, 
    model: Any, 
    samples_df: pd.DataFrame,
    lang_code: str,
    possible_labels_en: List[str],
    prompt_in_lrl: bool, 
    use_few_shot: bool, 
    base_results_path: str,
    generation_params: Dict
) -> Optional[Dict[str, Any]]:
    """Run a single baseline classification experiment configuration."""
    if samples_df.empty:
        print(f"No samples for {lang_code}, skipping baseline experiment.")
        return None

    model_short_name = model_name.split('/')[-1]
    shot_type_str = "fs" if use_few_shot else "zs"
    prompt_lang_str = "lrl" if prompt_in_lrl and lang_code != 'en' else "en"

    results_dir = os.path.join(base_results_path, prompt_lang_str, shot_type_str, lang_code, model_short_name)
    summaries_dir = os.path.join(base_results_path, "summaries", prompt_lang_str, shot_type_str)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"results_baseline_classification_{lang_code}.csv")
    summary_file = os.path.join(summaries_dir, f"summary_baseline_classification_{lang_code}_{model_short_name}.csv")

    results_df_computed = pd.DataFrame() # Initialize to empty
    if os.path.exists(results_file):
        print(f"Results file {results_file} already exists. Skipping computation, loading for summary.")
        try:
            results_df_computed = pd.read_csv(results_file)
            if 'ground_truth_label' in results_df_computed.columns:
                label_map = {
                    0: 'health', 1: 'religion', 2: 'politics', 3: 'sports',
                    4: 'local', 5: 'business', 6: 'entertainment'
                }
                results_df_computed['ground_truth_label'] = pd.to_numeric(results_df_computed['ground_truth_label'], errors='coerce')
                results_df_computed.dropna(subset=['ground_truth_label'], inplace=True)
                results_df_computed['ground_truth_label'] = results_df_computed['ground_truth_label'].astype(int).map(label_map)
                results_df_computed.dropna(subset=['ground_truth_label'], inplace=True)
                print(f"Applied label mapping to 'ground_truth_label' from loaded CSV: {results_file}")
        except Exception as e:
            print(f"Could not load existing results {results_file}: {e}. Will recompute if summary also missing.")
            if not os.path.exists(summary_file): results_df_computed = pd.DataFrame() # Mark for recomputation
            else:
                try: 
                    existing_summary_df = pd.read_csv(summary_file)
                    return existing_summary_df.to_dict('records')[0]
                except Exception as es:
                    print(f"Could not load existing summary {summary_file}: {es}. Recomputing.")
                    results_df_computed = pd.DataFrame() # Mark for recomputation
    
    if results_df_computed.empty: # Needs computation
        print(f"Running Baseline Classification: {model_name} on {lang_code} (Prompt: {prompt_lang_str}, Shot: {shot_type_str})")
        start_time = time.time()
        try:
            results_df_computed = evaluate_classification_baseline(
                model_name, tokenizer, model, samples_df, lang_code, 
                possible_labels_en, prompt_in_lrl, use_few_shot, generation_params
            )
            runtime = time.time() - start_time
            if not results_df_computed.empty:
                results_df_computed['runtime_seconds_total'] = runtime
                results_df_computed['runtime_per_sample'] = runtime / len(results_df_computed)
                results_df_computed.to_csv(results_file, index=False)
                print(f"Results saved to {results_file}")
            else:
                print("Baseline evaluation returned empty DataFrame.")
                return None
        except Exception as e:
            logging.error(f"Error during baseline classification for {model_name}, {lang_code}, {prompt_lang_str}, {shot_type_str}: {e}", exc_info=True)
            return None
    
    if results_df_computed.empty:
        print("No results to calculate metrics from for baseline.")
        return None

    # Ensure 'predicted_label' is used for metrics against 'ground_truth_label'
    # Baseline evaluation should output English labels if prompt_in_lrl=False or lang_code='en'
    # If prompt_in_lrl=True for LRL, the baseline function itself should ensure EN output or mapping.
    metrics = calculate_classification_metrics(results_df_computed)

    summary_data = {
        'model': model_short_name,
            'language': lang_code,
        'prompt_language': prompt_lang_str,
        'shot_type': shot_type_str,
        'accuracy': metrics.get('accuracy', 0.0),
        'macro_f1': metrics.get('macro_f1', 0.0),
        'samples_processed': len(results_df_computed),
        'runtime_total_s': results_df_computed['runtime_seconds_total'].iloc[0] if 'runtime_seconds_total' in results_df_computed.columns and not results_df_computed.empty else 0,
    }
    summary_data.update({k: v for k,v in generation_params.items()})
    for label in possible_labels_en:
        summary_data[f'{label}_precision'] = metrics.get(f'{label}_precision', 0.0)
        summary_data[f'{label}_recall'] = metrics.get(f'{label}_recall', 0.0)
        
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(summary_file, index=False, float_format='.4f')
    print(f"Summary saved to {summary_file}")
    print(summary_df.to_string())
    return summary_data

def main():
    parser = argparse.ArgumentParser(description="Run Classification Baseline experiments.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-expanse-8b,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated models")
    parser.add_argument("--langs", type=str, default="en,sw,ha", help="Comma-separated language codes")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples per language")
    parser.add_argument("--prompt_instructions", nargs='+', default=['en', 'lrl'], choices=['en', 'lrl'], help="Instruction language for prompts (en or lrl)")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Shot settings")
    parser.add_argument("--dataset_name", type=str, default="masakhanews", 
                        help="Name of the classification dataset (e.g., masakhanews)")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/classification/baseline", help="Base output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    token = get_token()
    login(token=token)
    
    model_list = [m.strip() for m in args.models.split(',')]
    lang_list = [l.strip() for l in args.langs.split(',')]

    os.makedirs(args.base_output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.base_output_dir, "summaries"), exist_ok=True)

    possible_labels_en = POSSIBLE_LABELS_EN
    print(f"Using English labels for classification: {possible_labels_en}")

    all_experiment_summaries = []

    for model_name in model_list:
        print(f"\n{'='*20} Initializing Model: {model_name} {'='*20}")
        tokenizer, model = None, None
        try:
            tokenizer, model = initialize_model(model_name)
        except Exception as e_init:
            logging.error(f"Failed to initialize model {model_name}: {e_init}. Skipping.", exc_info=True)
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        for lang_code in lang_list:
            print(f"\n--- Processing Language: {lang_code} for model {model_name} (Baseline) ---")
            samples_df = load_masakhanews_samples(lang_code=lang_code, num_samples=args.samples)
            if samples_df.empty or 'text' not in samples_df.columns or 'label' not in samples_df.columns:
                print(f"Skipping {lang_code} for {model_name} due to missing data/columns ('text', 'label').")
                continue
            samples_df['label'] = samples_df['label'].astype(str).str.lower().str.strip()

            for prompt_instr_lang in args.prompt_instructions:
                prompt_in_lrl = (prompt_instr_lang == 'lrl')
                if lang_code == 'en' and prompt_in_lrl:
                    print(f"Skipping LRL instructions for English language text. Using EN instructions.")
                    current_prompt_in_lrl = False
                else:
                    current_prompt_in_lrl = prompt_in_lrl

                for shot_setting in args.shot_settings:
                    use_few_shot = (shot_setting == 'few_shot')
                    effective_gen_params = get_effective_baseline_params(lang_code, model_name)
                    print(f"\n  Running config: Lang={lang_code}, PromptInstr={'LRL' if current_prompt_in_lrl else 'EN'}, Shot={shot_setting}")
                    print(f"    Effective Gen Params: {json.dumps(effective_gen_params, indent=2)}")

                    summary_data = run_baseline_classification_experiment(
                        model_name, tokenizer, model, samples_df, lang_code, 
                        possible_labels_en, current_prompt_in_lrl, use_few_shot,
                        args.base_output_dir, effective_gen_params
                    )
                    if summary_data:
                        # Ensure numeric types for all expected float columns before appending
                        float_cols = ['accuracy', 'macro_f1', 'runtime_total_s', 'temperature', 'top_p', 'repetition_penalty']
                        # Add per-class precision and recall columns
                        for label_name in POSSIBLE_LABELS_EN: # Use the globally defined POSSIBLE_LABELS_EN
                            float_cols.append(f'{label_name}_precision')
                            float_cols.append(f'{label_name}_recall')
                        
                        for col in float_cols:
                            if col in summary_data:
                                try:
                                    summary_data[col] = float(summary_data[col])
                                except (ValueError, TypeError):
                                    print(f"Warning: Could not convert {col} to float. Value: {summary_data[col]}. Setting to NaN.")
                                    summary_data[col] = np.nan # Use NaN for unconvertible values
                        
                        # Ensure integer types for relevant columns
                        int_cols = ['samples_processed', 'top_k', 'max_tokens']
                        for col in int_cols:
                            if col in summary_data:
                                try:
                                    summary_data[col] = int(summary_data[col])
                                except (ValueError, TypeError):
                                     print(f"Warning: Could not convert {col} to int. Value: {summary_data[col]}. Setting to 0 or NaN.")
                                     summary_data[col] = 0 # Or np.nan, depending on desired handling

                        all_experiment_summaries.append(summary_data)
        
        print(f"Finished all baseline experiments for model {model_name}. Unloading...")
        del model
        del tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logging.info(f"GPU memory cache cleared for {model_name}.")

    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        
        # --- Debugging DataFrame before saving ---
        print("\n--- Debugging overall_summary_df before saving ---")
        print("DataFrame Info:")
        overall_summary_df.info()
        print("\nDataFrame Head:")
        print(overall_summary_df.head().to_string())
        
        float_check_cols = ['accuracy', 'macro_f1', 'temperature', 'health_precision'] # Check a subset
        for col_to_check in float_check_cols:
            if col_to_check in overall_summary_df.columns and not overall_summary_df.empty:
                first_val = overall_summary_df[col_to_check].iloc[0]
                print(f"Sample value for {col_to_check} (1st row): {first_val}, type: {type(first_val)}")
        print("---------------------------------------------------\n")
        # --- End Debugging ---

        overall_summary_filename = os.path.join(args.base_output_dir, "summaries", f"classification_baseline_ALL_experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        try:
            # Define the order of columns for the CSV
            ordered_columns = [
                'model', 'language', 'prompt_language', 'shot_type', 'accuracy', 'macro_f1', 
                'samples_processed', 'runtime_total_s', 'temperature', 'top_p', 'top_k', 'max_tokens', 'repetition_penalty'
            ]
            # Add per-class metrics to the ordered list
            for label_name in POSSIBLE_LABELS_EN: # Use the globally defined POSSIBLE_LABELS_EN
                ordered_columns.append(f'{label_name}_precision')
                ordered_columns.append(f'{label_name}_recall')
            
            # Ensure all columns in ordered_columns exist in the DataFrame, add if missing (with NaN)
            for col in ordered_columns:
                if col not in overall_summary_df.columns:
                    overall_summary_df[col] = np.nan
            
            overall_summary_df = overall_summary_df[ordered_columns] # Reorder and select columns

            overall_summary_df.to_csv(overall_summary_filename, index=False, float_format='%.4f', na_rep='NaN')
            print(f"Overall summary saved to: {overall_summary_filename}")
            print(overall_summary_df.to_string(float_format='%.4f', na_rep='NaN'))
        except Exception as e_save:
            print(f"ERROR saving overall classification baseline summary to {overall_summary_filename}: {e_save}")
    else:
        print("No classification baseline experiments were successfully summarized.")

    print("\n====== Classification Baseline Script Finished ======")

if __name__ == "__main__":
    # Define POSSIBLE_LABELS_EN globally for access in main and other functions if needed
    # This should be the source of truth for MasakhaNEWS categories.
    POSSIBLE_LABELS_EN = ['health', 'religion', 'politics', 'sports', 'local', 'business', 'entertainment']
    print(f"Using English labels for classification: {POSSIBLE_LABELS_EN}")
    main() 