import sys
import os
import logging
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from typing import Any, Tuple, Dict, List, Optional
import re
import json

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from your project structure
from src.utils.data_loaders.load_masakhanews import load_masakhanews_samples # Specific loader for MasakhaNEWS
from src.experiments.cotr.classification.classification_cotr import (
    evaluate_classification_cotr_multi_prompt,
    evaluate_classification_cotr_single_prompt,
    initialize_model,
    CLASS_LABELS_ENGLISH # Ensure this is imported if used, or defined if not
)
# from src.experiments.cotr.language_information import NEWS_CATEGORIES_EN # Commented out
from huggingface_hub import login
from config import get_token
from sklearn.metrics import accuracy_score, f1_score # For calculating task metrics

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# --- Define Default Parameters (align with baseline where applicable) ---
DEFAULT_GENERATION_PARAMS = {
    "lrl_to_eng_translation": {"temperature": 0.3, "top_p": 0.9, "max_new_tokens": 300, "repetition_penalty": 1.0, "top_k": 40, "do_sample": True},
    "eng_classification": {"temperature": 0.1, "top_p": 0.85, "max_new_tokens": 50, "repetition_penalty": 1.05, "top_k": 30, "do_sample": True},
    "eng_to_lrl_label_translation": {"temperature": 0.3, "top_p": 0.9, "max_new_tokens": 30, "repetition_penalty": 1.0, "top_k": 40, "do_sample": True},
    "single_prompt_chain": {"temperature": 0.1, "top_p": 0.85, "max_new_tokens": 400, "repetition_penalty": 1.05, "top_k": 30, "do_sample": True}
}
MAX_INPUT_LENGTH = 2048

# Placeholder for language-specific parameter overrides if needed
LANGUAGE_SPECIFIC_PARAMS = {
    "sw": {
        "text_translation": {"temperature": 0.35, "max_new_tokens": 450},
        "english_classification": {"temperature": 0.05, "repetition_penalty": 1.15},
        "label_translation": {"temperature": 0.35},
        "single_prompt_cotr": {"temperature": 0.05, "repetition_penalty": 1.15}
    },
    "ha": {
        "english_classification": {"temperature": 0.05, "repetition_penalty": 1.15},
        "single_prompt_cotr": {"temperature": 0.05, "repetition_penalty": 1.15}
    },
    "te": { # Retain for QA if this script is adapted
        "english_classification": {"temperature": 0.05, "repetition_penalty": 1.2},
        "single_prompt_cotr": {"temperature": 0.05, "repetition_penalty": 1.2}
    }
}

# Placeholder for model-specific parameter adjustments
MODEL_SPECIFIC_ADJUSTMENTS = {
    "CohereLabs/aya-expanse-8b": {
        "english_classification": {"temperature_factor": 1.0}, # Example: no change from default
        "single_prompt_cotr": {"temperature_factor": 1.0}
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "english_classification": {"top_p": 0.85, "top_k": 35, "temperature": 0.05},
        "single_prompt_cotr": {"top_p": 0.85, "top_k": 35, "temperature": 0.05}
    }
}

# Define the correct MasakhaNEWS labels
POSSIBLE_LABELS_EN = ['business', 'entertainment', 'health', 'politics', 'religion', 'sports', 'technology']

# Base parameters configuration (Model-specific overrides)
# This section (MODEL_SPECIFIC_OVERRIDES) is replaced by MODEL_SPECIFIC_ADJUSTMENTS from the old script.
# The UNIFIED_GENERATION_PARAMETERS_CORE and task-specific max_tokens are replaced by DEFAULT_GENERATION_PARAMS.
# MODEL_SPECIFIC_OVERRIDES = {
#     "CohereLabs/aya-expanse-8b": { # Example model-specific base
#         "temperature": 0.05, # Stricter for Aya
#         "repetition_penalty": 1.15,
#         "max_tokens_classification_label": 22, # Slight adjust for Aya labels
#         "max_tokens_translation_step": 160,
#         "max_tokens_single_prompt_chain": 270,
#     },
#     "Qwen/Qwen2.5-7B-Instruct": {
#         "temperature": 0.15,
#         "top_p": 0.85,
#         "max_tokens_classification_label": 25, # Qwen might need more for labels
#         "max_tokens_translation_step": 170,
#         "max_tokens_single_prompt_chain": 280,
#     }
# }

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run Text Classification CoTR experiments with MasakhaNEWS.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated model names.")
    parser.add_argument("--langs", type=str, default="en,ha,sw", help="Comma-separated MasakhaNEWS language codes (e.g., sw,am,ha,yo,pcm,ig,en,pt).")
    parser.add_argument("--num_samples", type=int, default=80, help="Number of samples per language. Default: 80")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split.")
    parser.add_argument("--pipeline_types", nargs='+', default=['multi_prompt', 'single_prompt'], choices=['multi_prompt', 'single_prompt'])
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'])
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/classification/cotr_masakhanews")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token for gated models.")
    parser.add_argument("--overwrite_results", action='store_true', help="Overwrite existing result files if they exist.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    parser.add_argument("--test_mode", action='store_true', help="Run in test mode (uses a very small subset of data and fewer iterations).")
    parser.add_argument("--sample_percentage", type=float, default=10.0, help="Percentage of MasakhaNEWS samples per language (0-100). Default: 10%% of the split.")
    parser.add_argument("--max_samples_per_lang", type=int, default=None, 
                        help="Maximum number of samples to load per language. If --sample_percentage is used, this acts as a cap after percentage. If --sample_percentage is not used, this is the direct number of samples. For 10%% behavior, this should typically not be set, or set very high if --sample_percentage logic is separate.")
    
    # General generation parameters that can be overridden for any step via get_effective_params
    parser.add_argument("--temperature", type=float, default=None, help="Global override for temperature if not set by specific step params.")
    parser.add_argument("--top_p", type=float, default=None, help="Global override for top_p.")
    parser.add_argument("--top_k", type=int, default=None, help="Global override for top_k.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Global override for repetition_penalty.")
    parser.add_argument("--do_sample", type=lambda x: (str(x).lower() == 'true'), default=None, help="Global override for do_sample (True/False).")

    # Step-specific max_new_tokens overrides (these are used by get_effective_params)
    parser.add_argument("--max_translation_tokens", type=int, default=None, help="Override max_new_tokens for text translation step")
    parser.add_argument("--max_label_translation_tokens", type=int, default=None, help="Override max_new_tokens for label translation step")
    parser.add_argument("--max_single_prompt_tokens", type=int, default=None, help="Override max_new_tokens for single_prompt_cotr step")
    parser.add_argument("--max_tokens", type=int, default=None, help="Global override for max_new_tokens for english_classification step")
    return parser.parse_args()

def get_effective_params(base_params_config_key: str, lang_code: str, model_name: str, cli_overrides: Dict) -> Dict:
    """
    Merge default, language-specific, model-specific, and CLI override parameters
    for a specific step of the CoTR pipeline (e.g., "english_classification").
    """
    # Start with a copy of the default parameters for the given step (e.g., "english_classification")
    effective_params = DEFAULT_GENERATION_PARAMS.get(base_params_config_key, {}).copy()

    # Layer 1: Language-specific overrides for the current step
    lang_step_params = LANGUAGE_SPECIFIC_PARAMS.get(lang_code, {}).get(base_params_config_key, {})
    effective_params.update(lang_step_params)

    # Layer 2: Model-specific adjustments for the current step
    model_step_adjustments = MODEL_SPECIFIC_ADJUSTMENTS.get(model_name, {}).get(base_params_config_key, {})
    for adj_key, adj_value in model_step_adjustments.items():
        if adj_key.endswith("_factor"):
            param_to_adjust = adj_key[:-len("_factor")]
            if param_to_adjust in effective_params: # Apply factor only if the base param exists
                effective_params[param_to_adjust] *= adj_value
                if param_to_adjust == "temperature": # Ensure temp doesn't go too low
                    effective_params[param_to_adjust] = max(0.01, effective_params[param_to_adjust])
            # If param_to_adjust is not in effective_params, we do not add the _factor key.
        else: # This is the else for adj_key.endswith("_factor")
            effective_params[adj_key] = adj_value # Direct override for non-factor keys

    # Layer 3: CLI overrides (highest priority)
    cli_general_overrides = {}
    cli_max_new_tokens_overrides = {}

    if cli_overrides.get("temperature") is not None: cli_general_overrides["temperature"] = cli_overrides["temperature"]
    if cli_overrides.get("top_p") is not None: cli_general_overrides["top_p"] = cli_overrides["top_p"]
    if cli_overrides.get("top_k") is not None: cli_general_overrides["top_k"] = cli_overrides["top_k"]
    if cli_overrides.get("repetition_penalty") is not None: cli_general_overrides["repetition_penalty"] = cli_overrides["repetition_penalty"]
    if cli_overrides.get("do_sample") is not None: cli_general_overrides["do_sample"] = cli_overrides["do_sample"]
    
    # Handle max_new_tokens specifically based on step
    if base_params_config_key == "english_classification" and cli_overrides.get("max_tokens") is not None:
        cli_max_new_tokens_overrides["max_new_tokens"] = cli_overrides["max_tokens"]
    elif base_params_config_key == "text_translation" and cli_overrides.get("max_translation_tokens") is not None:
        cli_max_new_tokens_overrides["max_new_tokens"] = cli_overrides["max_translation_tokens"]
    elif base_params_config_key == "label_translation" and cli_overrides.get("max_label_translation_tokens") is not None:
        cli_max_new_tokens_overrides["max_new_tokens"] = cli_overrides["max_label_translation_tokens"]
    elif base_params_config_key == "single_prompt_cotr" and cli_overrides.get("max_single_prompt_tokens") is not None:
        cli_max_new_tokens_overrides["max_new_tokens"] = cli_overrides["max_single_prompt_tokens"]

    effective_params.update(cli_general_overrides) # Apply general CLI overrides
    effective_params.update(cli_max_new_tokens_overrides) # Apply specific max_new_tokens CLI overrides

    # Final logic for do_sample: if not explicitly set by CLI, derive from temperature.
    # If CLI did set do_sample, cli_general_overrides would have applied it.
    if cli_overrides.get("do_sample") is None:
        effective_params["do_sample"] = effective_params.get("temperature", 0.0) > 0.01 # Default to True if temp > 0.01
    
    # Ensure no _factor keys are left in the effective_params from earlier stages
    # (e.g. if they were in DEFAULT_GENERATION_PARAMS or LANGUAGE_SPECIFIC_PARAMS)
    keys_to_remove = [key for key in effective_params if key.endswith("_factor")]
    for key in keys_to_remove:
        del effective_params[key]

    logging.debug(f"Effective CoTR params for {base_params_config_key} on {model_name} ({lang_code}): {effective_params}")
    return effective_params

def run_classification_experiment(
    model_name_str: str, tokenizer: Any, model: Any, samples_df: pd.DataFrame,
    lang_code: str, possible_labels_en: List[str], pipeline_type: str, 
    use_few_shot: bool, base_results_path: str, generation_args: Dict, # This will now be the structured dict from main
    overwrite_results: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Runs a classification experiment for a given configuration (multi or single prompt).
    Handles loading existing results, running evaluations, calculating metrics, and saving.
    Returns a dictionary with summary metrics for this specific run, or None on failure.
    (Adapted from old script's run_classification_experiment)
    """
    if samples_df.empty:
        print(f"No samples for {lang_code}, skipping {model_name_str}.")
        return None

    shot_type_str = "fs" if use_few_shot else "zs"
    model_short_name = model_name_str.split('/')[-1]

    results_dir = os.path.join(base_results_path, pipeline_type, shot_type_str, lang_code, model_short_name)
    summaries_dir = os.path.join(base_results_path, "summaries", pipeline_type, shot_type_str)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"results_cotr_classification_{lang_code}.csv")
    summary_file = os.path.join(summaries_dir, f"summary_cotr_classification_{lang_code}_{model_short_name}.csv")

    results_df_current_run = pd.DataFrame()

    if os.path.exists(results_file) and not overwrite_results:
        print(f"Results file {results_file} already exists. Skipping computation, attempting to load for summary.")
        try:
            results_df_current_run = pd.read_csv(results_file)
        except Exception as e:
            print(f"Could not load existing results {results_file}: {e}. Will recompute if summary also missing or force overwrite is enabled.")
            results_df_current_run = pd.DataFrame()
        
        # Check for summary file only if results were loaded or attempted to be loaded
        if os.path.exists(summary_file): # This check was a bit misplaced before
            try:
                existing_summary_df = pd.read_csv(summary_file)
                print(f"Loaded existing summary: {summary_file}") # Corrected indentation
                return existing_summary_df.to_dict('records')[0]
            except Exception as es:
                print(f"Could not load existing summary {summary_file}: {es}. Recomputing if results_df_current_run is empty or overwrite is True.")
                # If summary loading fails, we might still proceed if results_df_current_run has data
                # and re-generate the summary later. If results_df_current_run is also empty, we must recompute.
                if results_df_current_run.empty: # Ensure recomputation if both failed
                    pass # Will fall through to recomputation logic
        
        # If results_df_current_run is not empty here, it means we loaded existing results
        # and will proceed to metric calculation with them, as summary wasn't returned.
        # If both failed or results_df was marked for recompute, it will be empty.

    if overwrite_results and os.path.exists(results_file): # Explicit recompute if overwrite
        print(f"Overwrite_results is True. Recomputing for {results_file}")
        results_df_current_run = pd.DataFrame()

    if results_df_current_run.empty:
        print(f"Running CoTR Classification: {model_name_str} on {lang_code} (Pipeline: {pipeline_type}, Shot: {shot_type_str})")
        start_time = time.time()
        try: # THIS IS THE TRY FOR THE EVALUATION BLOCK
            eval_func_params = { # <-- This line and subsequent lines in the block should be indented
                "model_name": model_name_str,
                "model": model,
                "tokenizer": tokenizer,
        "samples_df": samples_df,
        "lang_code": lang_code,
        "possible_labels_en": possible_labels_en,
        "use_few_shot": use_few_shot,
            }
            if pipeline_type == 'multi_prompt':
                eval_func_params.update({
                    "text_translation_params": generation_args.get("text_translation", {}),
                    "classification_params": generation_args.get("english_classification", {}),
                    "label_translation_params": generation_args.get("label_translation", {})
                })
                results_df_current_run = evaluate_classification_cotr_multi_prompt(**eval_func_params)
            else: # single_prompt
                eval_func_params["generation_params"] = generation_args.get("single_prompt_cotr", {})
                results_df_current_run = evaluate_classification_cotr_single_prompt(**eval_func_params)

            runtime = time.time() - start_time
            if not results_df_current_run.empty:
                results_df_current_run['runtime_seconds_total'] = runtime
                results_df_current_run['runtime_per_sample'] = runtime / len(results_df_current_run)
                # Note: Save results CSV after metrics calculation to include is_correct column
                print(f"Evaluation completed successfully.")
            else:
                print("Evaluation returned empty DataFrame.")
        except Exception as e: # This except pairs with the try above
            logging.error(f"Error during CoTR classification for {lang_code}, {model_short_name}, {pipeline_type}, {shot_type_str}: {e}", exc_info=True)

    # Calculate summary metrics from results_df_current_run
    # First, create the is_correct column if it doesn't exist
    if not results_df_current_run.empty:
        # Determine which column contains the predicted labels for accuracy calculation
        if pipeline_type == 'multi_prompt':
            predicted_col = 'predicted_label_eng_model'
            ground_truth_col = 'ground_truth_label_eng'
        else:  # single_prompt
            predicted_col = 'predicted_label_accuracy'  # This contains the mapped English label
            ground_truth_col = 'label_lrl_ground_truth'
        
        if predicted_col in results_df_current_run.columns and ground_truth_col in results_df_current_run.columns:
            # Create is_correct column for accuracy calculation
            results_df_current_run['is_correct'] = (
                results_df_current_run[predicted_col].str.lower().str.strip() == 
                results_df_current_run[ground_truth_col].str.lower().str.strip()
            )
            
            # Calculate additional metrics using sklearn
            try:
                from sklearn.metrics import accuracy_score, f1_score
                
                y_true = results_df_current_run[ground_truth_col].str.lower().str.strip()
                y_pred = results_df_current_run[predicted_col].str.lower().str.strip()
                
                # Filter out any samples with missing predictions or ground truth
                # Use case-insensitive filtering to catch all variations of error labels
                missing_gt_mask = y_true.isin(['[missing ground truth]', ''])
                error_pred_mask = (y_pred.str.contains(r'\[unknown label\]', case=False, na=False) | 
                                  y_pred.str.contains(r'\[classification error\]', case=False, na=False) |
                                  y_pred.isin(['']))
                valid_mask = ~(missing_gt_mask | error_pred_mask)
                y_true_valid = y_true[valid_mask]
                y_pred_valid = y_pred[valid_mask]
                
                if len(y_true_valid) > 0:
                    accuracy = accuracy_score(y_true_valid, y_pred_valid)
                    macro_f1 = f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
                    weighted_f1 = f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
                else:
                    accuracy = 0.0
                    macro_f1 = 0.0
                    weighted_f1 = 0.0
                    
            except Exception as e:
                logging.warning(f"Error calculating metrics with sklearn: {e}. Using basic accuracy only.")
                accuracy = results_df_current_run['is_correct'].mean()
                macro_f1 = 0.0
                weighted_f1 = 0.0
        else:
            logging.warning(f"Missing columns for accuracy calculation. Expected {predicted_col} and {ground_truth_col}")
            accuracy = 0.0
            macro_f1 = 0.0
            weighted_f1 = 0.0
            
        # Now save the results CSV with all calculated columns including is_correct
        if 'runtime_seconds_total' in results_df_current_run.columns:
            results_df_current_run.to_csv(results_file, index=False)
            print(f"Results saved to {results_file}")
    else:
        accuracy = 0.0
        macro_f1 = 0.0
        weighted_f1 = 0.0

    avg_accuracy = accuracy

    summary_data = {
        'model': model_name_str.split('/')[-1], 'language': lang_code, 'pipeline': pipeline_type, 'shot_type': 'few-shot' if use_few_shot else 'zero-shot',
        'samples': len(results_df_current_run), 'accuracy': avg_accuracy, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
        'generation_params': generation_args  # Log the generation_args dict for this experiment
    }

    summary_df_to_save = pd.DataFrame([summary_data])
    summary_df_to_save.to_csv(summary_file, index=False, float_format='%.4f')
    print(f"Summary saved to {summary_file}")
    print(summary_df_to_save.to_string())
    return summary_data

def plot_classification_metrics(summary_df: pd.DataFrame, plots_dir: str, metric_col: str, metric_name: str, task_name: str = "Classification CoTR"):
    if summary_df.empty or metric_col not in summary_df.columns:
        print(f"Summary DataFrame empty or missing column '{metric_col}'. Skipping plot for {metric_name}.")
        return
    
    # Ensure numeric type for metric column before plotting, converting errors to NaN
    summary_df[metric_col] = pd.to_numeric(summary_df[metric_col], errors='coerce')
    plot_df = summary_df.dropna(subset=[metric_col]) # Drop rows where the metric is NaN after conversion

    if plot_df.empty:
        print(f"No valid data to plot for {metric_name} after dropping NaNs from column '{metric_col}'.")
        return

    plt.figure(figsize=(20, 12)) # Increased figure size for better readability
    try:
        # Create a unique configuration identifier for x-axis labels if not already present
        if 'experiment_config' not in plot_df.columns:
             plot_df['experiment_config'] = plot_df['language'] + '-' + plot_df['model'] + '-' + plot_df['pipeline'] + '-' + plot_df['shot_type']
        
        # Use seaborn for potentially better aesthetics and hue handling
        sns.barplot(data=plot_df, x='experiment_config', y=metric_col, hue='language', dodge=False) # dodge=False if too cluttered
        plt.xticks(rotation=60, ha='right', fontsize=10) # Rotate more for long names
        plt.yticks(fontsize=10)
        plt.title(f'Average {metric_name} for {task_name} Experiments', fontsize=16)
        plt.ylabel(f'Average {metric_name}', fontsize=12)
        plt.xlabel('Experiment Configuration (Lang-Model-Pipeline-Shot)', fontsize=12)
        plt.legend(title='Language', bbox_to_anchor=(1.02, 1), loc='upper left') # Adjust legend position
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout to make space for legend
        
        # Sanitize metric_col for filename (remove special chars, spaces)
        safe_metric_name = re.sub(r'[^a-zA-Z0-9_]', '', metric_col.replace('avg_comet_', '').replace('->', 'To'))
        plot_filename = os.path.join(plots_dir, f"cotr_classification_{safe_metric_name}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"{metric_name} plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error generating {metric_name} plot: {e}")
        if plt.get_fignums(): # Close figure if an error occurred mid-plot
            plt.close()

def main():
    args = parse_cli_args()
    # HF Login
    if args.hf_token:
        token = args.hf_token
    else:
        token = get_token() # Call without arguments

    if token:
        login(token=token)

    models_list = [m.strip() for m in args.models.split(',')]
    lang_list = [l.strip() for l in args.langs.split(',')] # From old script, for iteration
    all_experiment_summaries = []
    
    # Use the correct MasakhaNEWS labels
    possible_labels_en_for_exp = POSSIBLE_LABELS_EN
    print(f"Using English labels for classification: {possible_labels_en_for_exp}")

    # Define overall summary and plots directories
    # Old script structure for overall summary:
    overall_summary_base_dir = os.path.join(args.base_output_dir, "summaries_overall") # Old script puts general summary here
    os.makedirs(overall_summary_base_dir, exist_ok=True)
    # Plots can go into a general plots dir as per new script, or parallel to summaries
    overall_plots_dir = os.path.join(args.base_output_dir, "plots_overall")
    os.makedirs(overall_plots_dir, exist_ok=True)

    print(f"All Classification CoTR experiment outputs will be saved under: {args.base_output_dir}")
    print(f"Individual summaries in: {args.base_output_dir}/summaries/[pipeline]/[shot]") # Adjusted to old structure
    print(f"Overall summary in: {overall_summary_base_dir}")
    print(f"Overall plots in: {overall_plots_dir}")

    # Use a fixed seed for the 10% sampling for reproducibility across runs if args.seed is not defined
    sampling_seed = args.seed # from old script

    for model_name_str in models_list:
        print(f"\n{'='*20} Initializing Model: {model_name_str} {'='*20}")
        tokenizer_main, model_main = None, None
        try:
            tokenizer_main, model_main = initialize_model(model_name_str)
        except Exception as e:
            logging.error(f"Failed to initialize model {model_name_str}: {e}. Skipping this model.")
            if model_main is not None: del model_main
            if tokenizer_main is not None: del tokenizer_main
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue # Skip to the next model

        for lang_code in lang_list: # Iterate using lang_list from old script
            print(f"\n--- Processing Language: {lang_code} for model {model_name_str} ---")
            
            # Simplified sample loading:
            # Directly request args.num_samples (default 80) from the loader.
            # The loader itself handles returning min(args.num_samples, total_available_for_lang_split).
            samples_df_for_run = load_masakhanews_samples(
                lang_code, 
                split=args.data_split, 
                num_samples=args.num_samples, # Default 80, from args
                seed=sampling_seed
            )

            if samples_df_for_run.empty:
                print(f"    No MasakhaNEWS samples found or loaded for {lang_code} (split {args.data_split}, requested up to {args.num_samples}), skipping.")
                continue
            else:
                print(f"    Successfully loaded {len(samples_df_for_run)} MasakhaNEWS samples for {lang_code} (split {args.data_split}, requested up to {args.num_samples}).")
            
            # Note: The CLI arguments args.sample_percentage and args.max_samples_per_lang 
            # will no longer directly influence the sample count with this simplified logic,
            # as args.num_samples (default 80) becomes the primary controller passed to the loader.

            # Ground truth label processing (from old script context)

            # Ensure 'label' column contains English ground truth, as expected by old script's metric flow
            # This might need mapping if the loader provides LRL labels.
            # The new data loader (`load_masakhanews.py`) already maps numerical labels to English strings.
            # So, samples_df_for_run['label'] should be English.
            if 'label' not in samples_df_for_run.columns or not all(isinstance(x, str) for x in samples_df_for_run['label']):
                 print(f"Warning: 'label' column for {lang_code} is missing or not all strings. Downstream metrics might fail.")
            else:
                samples_df_for_run['label'] = samples_df_for_run['label'].astype(str).str.lower().str.strip()
                # Optional: Filter for labels in possible_labels_en_for_exp if necessary (old script had commented out similar logic)
                # initial_count = len(samples_df_for_run)
                # samples_df_for_run = samples_df_for_run[samples_df_for_run['label'].isin([l.lower() for l in possible_labels_en_for_exp])]
                # if len(samples_df_for_run) < initial_count:
                #    print(f"    Filtered samples for {lang_code} to only include labels in {possible_labels_en_for_exp}. Count changed from {initial_count} to {len(samples_df_for_run)}")
                # if samples_df_for_run.empty:
                #    print(f"    No samples remaining for {lang_code} after filtering for valid labels. Skipping.")
                #    continue

            if args.test_mode: # test_mode further reduces the sampled data
                print("    Running in TEST MODE with first 5 samples only.")
                samples_df_for_run = samples_df_for_run.head(5)
            
            if samples_df_for_run.empty: # Re-check after test_mode
                print(f"    No samples for {lang_code} after test_mode reduction. Skipping.")
                continue

            for pipeline_type in args.pipeline_types:
                for shot_setting_val_str in args.shot_settings:
                    use_few_shot_bool = (shot_setting_val_str == 'few_shot')
                    
                    # CLI overrides structure from old script
                    cli_gen_overrides_for_step = {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "max_tokens": args.max_tokens, # For english_classification in get_effective_params
                        "repetition_penalty": args.repetition_penalty,
                        # Pass specific max_new_tokens overrides directly to get_effective_params
                        "max_translation_tokens": args.max_translation_tokens,
                        "max_label_translation_tokens": args.max_label_translation_tokens,
                        "max_single_prompt_tokens": args.max_single_prompt_tokens
                    }

                    # Generation args structure from old script
                    generation_args_for_run_structured = {
                        "text_translation": get_effective_params("text_translation", lang_code, model_name_str, cli_gen_overrides_for_step),
                        "english_classification": get_effective_params("english_classification", lang_code, model_name_str, cli_gen_overrides_for_step),
                        "label_translation": get_effective_params("label_translation", lang_code, model_name_str, cli_gen_overrides_for_step),
                        "single_prompt_cotr": get_effective_params("single_prompt_cotr", lang_code, model_name_str, cli_gen_overrides_for_step)
                    }
                    
                    # This specific override from old script's main loop is now handled inside get_effective_params by "max_tokens"
                    # if args.max_tokens is not None and "english_classification" in generation_args_for_run_structured:
                    #    generation_args_for_run_structured["english_classification"]["max_new_tokens"] = args.max_tokens

                    print(f"\n  Running config: Model={model_name_str}, Lang={lang_code}, Pipeline={pipeline_type}, Shot={shot_setting_val_str}")
                    print(f"    Effective Gen Params: {json.dumps(generation_args_for_run_structured, indent=2)}")

                    # Call the experiment runner
                    summary = run_classification_experiment(
                        model_name_str, tokenizer_main, model_main, samples_df_for_run, # Use samples_df_for_run
                        lang_code, possible_labels_en_for_exp, pipeline_type,
                        use_few_shot_bool, args.base_output_dir, generation_args_for_run_structured, # Pass structured args
                        overwrite_results=args.overwrite_results
                    )
                    if summary:
                        all_experiment_summaries.append(summary)
        
        print(f"Finished all experiments for model {model_name_str}. Unloading...")
        del model_main; del tokenizer_main
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        if not overall_summary_df.empty:
            # Save overall summary (old script's naming and location)
            from datetime import datetime # Ensure datetime is imported
            summary_filename_overall = f"cotr_classification_ALL_experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            overall_summary_path = os.path.join(overall_summary_base_dir, summary_filename_overall)
            overall_summary_df.to_csv(overall_summary_path, index=False, float_format='%.4f')
            print(f"\nOverall summary of Classification CoTR experiments saved to: {overall_summary_path}")
            print(overall_summary_df.to_string())

            # Plot various metrics
            try:
                plot_classification_metrics(overall_summary_df, overall_plots_dir, 'accuracy', 'Accuracy')
                plot_classification_metrics(overall_summary_df, overall_plots_dir, 'macro_f1', 'Macro F1')
                plot_classification_metrics(overall_summary_df, overall_plots_dir, 'weighted_f1', 'Weighted F1')
            except Exception as e_plot_acc:
                logging.error(f"Error generating accuracy plot: {e_plot_acc}")
        else:
            print("Overall summary DataFrame is empty. No plots generated.")
    else:
        print("No summaries collected. Skipping overall summary and plot generation for Classification CoTR.")

    print("\nAll Classification CoTR experiments completed!")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main() 