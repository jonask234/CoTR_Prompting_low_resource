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
    "text_translation": {"temperature": 0.4, "top_p": 0.9, "max_new_tokens": 512, "repetition_penalty": 1.0, "top_k": 40},
    "english_classification": {"temperature": 0.05, "top_p": 0.9, "max_new_tokens": 20, "repetition_penalty": 1.1, "top_k": 40},
    "label_translation": {"temperature": 0.4, "top_p": 0.9, "max_new_tokens": 75, "repetition_penalty": 1.0, "top_k": 40},
    "single_prompt_cotr": {"temperature": 0.05, "top_p": 0.9, "max_new_tokens": 400, "repetition_penalty": 1.1, "top_k": 40}
}

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

# Define the English news categories, aligning with POSSIBLE_LABELS_EN from classification_cotr.py
# CLASS_LABELS_ENGLISH is imported, so this can be removed if it's identical
# NEWS_CATEGORIES_EN = ['health', 'religion', 'politics', 'sports', 'local', 'business', 'entertainment', 'unknown'] # Ensure 'unknown' is present if used

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

def parse_args():
    parser = argparse.ArgumentParser(description="Run Classification CoTR experiments with MasakhaNEWS.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-expanse-8b,Qwen/Qwen2.5-7B-Instruct",
                        help="Comma-separated model names.")
    parser.add_argument("--langs", type=str, default="en,sw,ha", # Changed from nargs='+'
                        help="Comma-separated language codes (default: en,sw,ha for MasakhaNEWS).")
    parser.add_argument("--sample_percentage", type=float, default=10.0, help="Percentage of samples to use (0.0 to 1.0). Default is 10.0 (10%%). Should be used as num_samples effectively based on 10% of dataset.")
    parser.add_argument("--max_samples_per_lang", type=int, default=None, 
                        help="Maximum number of samples to load per language. If --sample_percentage is used, this acts as a cap after percentage. If --sample_percentage is not used, this is the direct number of samples. For 10% behavior, this should typically not be set, or set very high if --sample_percentage logic is separate.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples per language (from old script, maps to 10% logic).")

    parser.add_argument("--data_split", type=str, default="test",
                        choices=["train", "validation", "test"], help="Dataset split to use. Default: test.")

    parser.add_argument("--pipeline_types", nargs='+', default=['multi_prompt', 'single_prompt'], choices=['multi_prompt', 'single_prompt'], help="CoTR pipeline types.")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'])
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/classification/cotr_masakhanews", help="Base directory to save results.")
    parser.add_argument("--hf_token", type=str, help="HuggingFace API token.")
    parser.add_argument("--overwrite_results", action='store_true', help="Overwrite existing result files if they exist.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    parser.add_argument("--test_mode", action='store_true', help="Run in test mode (uses a very small subset of data and fewer iterations).")

    # CLI overrides for generation parameters (applied to all relevant steps)
    # These match the old script's general overrides.
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    # The old script had max_tokens for classification, and specific max_new_tokens for other steps.
    # We need to ensure the get_effective_params can correctly use these.
    parser.add_argument("--max_tokens", type=int, default=None, help="Global override for max_new_tokens for english_classification step")
    parser.add_argument("--max_translation_tokens", type=int, default=None, help="Override max_new_tokens for text translation step")
    parser.add_argument("--max_label_translation_tokens", type=int, default=None, help="Override max_new_tokens for label translation step")
    parser.add_argument("--max_single_prompt_tokens", type=int, default=None, help="Override max_new_tokens for single_prompt_cotr step")
    # do_sample is not explicitly in old script's CLI, but handled by temperature > 0 in its logic.
    # We can remove the direct --do_sample CLI arg if we follow that.
    # parser.add_argument("--do_sample", type=lambda x: (str(x).lower() == 'true'), default=None)

    # Max tokens specific to CoTR steps from new script are now covered by the above args from old script.
    # parser.add_argument("--max_tokens_classification_label", type=int, default=None, help="(Multi-Prompt) Max new tokens for English classification step.")
    # parser.add_argument("--max_tokens_translation", type=int, default=None, help="(Multi-Prompt) Max new tokens for translation steps (LRL text to English, English label to LRL).")
    # parser.add_argument("--max_tokens_single_chain", type=int, default=None, help="(Single-Prompt) Max new tokens for the entire single prompt CoTR output chain.")
    return parser.parse_args()

def get_effective_params(base_params_config_key: str, lang_code: str, model_name: str, cli_overrides: Dict) -> Dict:
    """
    Merge default, language-specific, model-specific, and CLI override parameters
    for a specific step of the CoTR pipeline (e.g., "english_classification").
    (This function is taken from the old script: run_classification_cotr_old.py)
    """
    # Start with the default for the specific configuration key
    effective_params = DEFAULT_GENERATION_PARAMS.get(base_params_config_key, {}).copy()

    # Apply language-specific overrides for that key
    lang_step_params = LANGUAGE_SPECIFIC_PARAMS.get(lang_code, {}).get(base_params_config_key, {})
    effective_params.update(lang_step_params)

    # Apply model-specific adjustments for that key
    model_step_adjustments = MODEL_SPECIFIC_ADJUSTMENTS.get(model_name, {}).get(base_params_config_key, {})
    for adj_key, adj_value in model_step_adjustments.items():
        if adj_key.endswith("_factor") and adj_key[:-len("_factor")] in effective_params:
            param_to_adjust = adj_key[:-len("_factor")]
            effective_params[param_to_adjust] *= adj_value
            if param_to_adjust == "temperature": # Ensure temp doesn't go too low from factor
                effective_params[param_to_adjust] = max(0.01, effective_params[param_to_adjust])
        else:
            effective_params[adj_key] = adj_value # Direct override

    # Apply CLI overrides (highest priority)
    # Note: CLI overrides are general, so we apply them if the key matches
    for cli_key, cli_value in cli_overrides.items():
        if cli_value is not None and cli_key in effective_params:
            effective_params[cli_key] = cli_value
        # Allow CLI to override max_new_tokens even if key is just max_tokens (specific for english_classification)
        if cli_key == "max_tokens" and cli_value is not None and "max_new_tokens" in effective_params:
             if base_params_config_key == "english_classification": # only for classification step
                effective_params["max_new_tokens"] = cli_value
        # Allow specific max_new_tokens overrides for translation and single_prompt steps
        if cli_key == "max_translation_tokens" and base_params_config_key == "text_translation" and cli_value is not None:
            effective_params["max_new_tokens"] = cli_value
        if cli_key == "max_label_translation_tokens" and base_params_config_key == "label_translation" and cli_value is not None:
            effective_params["max_new_tokens"] = cli_value
        if cli_key == "max_single_prompt_tokens" and base_params_config_key == "single_prompt_cotr" and cli_value is not None:
            effective_params["max_new_tokens"] = cli_value
            
    # Ensure do_sample is set based on temperature, if not explicitly set by CLI (which it isn't here)
    effective_params["do_sample"] = effective_params.get("temperature", 0) > 0.01

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
            eval_func_params = {
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
                results_df_current_run.to_csv(results_file, index=False)
                print(f"Results saved to {results_file}")
            else:
                print("Evaluation returned empty DataFrame.")
                return None
        except Exception as e: # This except pairs with the try above
            logging.error(f"Error during CoTR classification for {lang_code}, {model_short_name}, {pipeline_type}, {shot_type_str}: {e}", exc_info=True)
            return None

    # Calculate metrics (adapted from old script, using calculate_classification_metrics if available and desired)
    if results_df_current_run.empty:
        print("No results to calculate metrics from.")
        return None

    if pipeline_type == "multi_prompt":
        if 'predicted_label_eng_model' in results_df_current_run.columns:
            results_df_current_run.rename(columns={'predicted_label_eng_model': 'final_predicted_label'}, inplace=True)
    elif pipeline_type == "single_prompt":
        if 'final_predicted_label_eng' in results_df_current_run.columns:
            results_df_current_run.rename(columns={'final_predicted_label_eng': 'final_predicted_label'}, inplace=True)

    y_true = []
    if 'label' in results_df_current_run.columns:
        y_true = results_df_current_run['label'].astype(str).str.lower().str.strip().tolist()
    elif 'ground_truth_label_eng' in results_df_current_run.columns:
         y_true = results_df_current_run['ground_truth_label_eng'].astype(str).str.lower().str.strip().tolist()
    else:
        print("ERROR: Cannot determine ground truth column for metrics. Expected 'label' (if English GT) or 'ground_truth_label_eng'.")
        return None

    if not y_true:
        print("ERROR: Ground truth list for metrics is empty.")
        return None

    y_pred_col = 'final_predicted_label'
    if y_pred_col in results_df_current_run.columns:
        y_pred = results_df_current_run[y_pred_col].astype(str).str.lower().str.strip().tolist()
    else:
        print(f"ERROR: Prediction column '{y_pred_col}' not found for metrics.")
        return None

    if not y_pred:
        print("ERROR: Prediction list for metrics is empty after loading.")
        return None

    try:
        avg_accuracy = accuracy_score(y_true, y_pred)
        avg_macro_f1 = f1_score(y_true, y_pred, labels=possible_labels_en, average='macro', zero_division=0)
        avg_weighted_f1 = f1_score(y_true, y_pred, labels=possible_labels_en, average='weighted', zero_division=0)
    except Exception as e_metrics:
        logging.error(f"Error calculating sklearn metrics: {e_metrics}")
        avg_accuracy = 0.0
        avg_macro_f1 = 0.0
        avg_weighted_f1 = 0.0

    avg_comet_lrl_text_to_en = np.mean(results_df_current_run['comet_lrl_text_to_en'].dropna()) if 'comet_lrl_text_to_en' in results_df_current_run.columns and not results_df_current_run['comet_lrl_text_to_en'].dropna().empty else None
    avg_comet_en_label_to_lrl = np.mean(results_df_current_run['comet_en_label_to_lrl'].dropna()) if 'comet_en_label_to_lrl' in results_df_current_run.columns and not results_df_current_run['comet_en_label_to_lrl'].dropna().empty else None

    summary_data = {
        'model': model_short_name,
        'language': lang_code,
        'pipeline': pipeline_type,
        'shot_type': shot_type_str,
        'accuracy': avg_accuracy,
        'macro_f1': avg_macro_f1,
        'weighted_f1': avg_weighted_f1,
        'samples_processed': len(results_df_current_run),
        'runtime_total_s': results_df_current_run['runtime_seconds_total'].iloc[0] if 'runtime_seconds_total' in results_df_current_run.columns and not results_df_current_run.empty else 0,
        'avg_comet_lrl_text_to_en': avg_comet_lrl_text_to_en,
        'avg_comet_en_label_to_lrl': avg_comet_en_label_to_lrl,
    }
    if pipeline_type == 'multi_prompt':
        summary_data.update({f"text_trans_{k}": v for k,v in generation_args.get("text_translation", {}).items()})
        summary_data.update({f"en_class_{k}": v for k,v in generation_args.get("english_classification", {}).items()})
        summary_data.update({f"label_trans_{k}": v for k,v in generation_args.get("label_translation", {}).items()})
    else: # single_prompt
        summary_data.update({f"single_prompt_{k}": v for k,v in generation_args.get("single_prompt_cotr", {}).items()})

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
    args = parse_args()
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
    
    # Use CLASS_LABELS_ENGLISH from classification_cotr.py for possible English labels
    # This is imported at the top now.
    # masakhanews_en_labels = CLASS_LABELS_ENGLISH
    # Ensure 'unknown' is handled if it's part of CLASS_LABELS_ENGLISH and relevant
    possible_labels_en_for_exp = CLASS_LABELS_ENGLISH
    if "unknown" not in possible_labels_en_for_exp: # Ensure 'unknown' is included if experiments depend on it
        possible_labels_en_for_exp = possible_labels_en_for_exp + ["unknown"]
    print(f"Using English labels for classification: {possible_labels_en_for_exp}")

    # Define overall summary and plots directories
    # Old script structure for overall summary:
    overall_summary_base_dir = os.path.join(args.base_output_dir, "summaries") # Old script puts general summary here
    os.makedirs(overall_summary_base_dir, exist_ok=True)
    # Plots can go into a general plots dir as per new script, or parallel to summaries
    overall_plots_dir = os.path.join(args.base_output_dir, "overall_plots")
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
            # Load MasakhaNEWS samples - old script uses args.num_samples for this.
            # New script uses percentage. We need to reconcile.
            # The user wants "10% of all samples". load_masakhanews_samples should handle this.
            # The args.num_samples from old script is likely a fixed number, not percentage.
            # Let's assume load_masakhanews_samples can take num_samples=None to mean "all for split"
            # and then we sample 10% from that.
            
            # Fetch all samples for the split first
            # The loader now directly handles num_samples, mapping to its internal logic.
            # For "10% of all samples", the data loader needs to be aware of this or we do it here.
            # The current `load_masakhanews_samples` takes `num_samples`. If this is to be 10%,
            # we need to calculate that first.
            
            # For "10% of all samples per language per dataset"
            # 1. Load all samples for the split
            # 2. Calculate 10% of that count
            # 3. Sample that many, or use a direct argument if the loader supports percentage.
            # The current data loader doesn't seem to take percentage directly.
            
            # Using args.num_samples from old script as the target number of samples *per language*.
            # This matches the old script's behavior. The "10%" is a general guideline but implemented
            # via a fixed number in the old script's --samples arg (default 20).
            # If the user wants exactly 10%, then args.num_samples should be dynamically calculated.
            # For now, using args.num_samples as the fixed count from the old script.
            
            # Reconciling sample loading:
            # Old script: fixed `args.num_samples` per lang.
            # New script: `args.sample_percentage` then `args.max_samples_per_lang` cap.
            # User request: "10% of all samples per language per dataset".
            # This implies loading all, then taking 10%. `args.num_samples` (default 20) might not be 10%.

            # Let's try to implement the 10% logic here if the loader returns full data.
            temp_full_samples_df = load_masakhanews_samples(lang_code, split=args.data_split, num_samples=None) # Request all
            
            num_to_sample_10_percent = 0
            if not temp_full_samples_df.empty:
                num_total_for_lang_split = len(temp_full_samples_df)
                num_to_sample_10_percent = max(1, int(args.sample_percentage / 100.0 * num_total_for_lang_split))
                print(f"    Total available for {lang_code} ({args.data_split}): {num_total_for_lang_split}. Taking {args.sample_percentage}% -> {num_to_sample_10_percent} samples.")
                samples_df_for_run = temp_full_samples_df.sample(n=num_to_sample_10_percent, random_state=sampling_seed)
            else:
                samples_df_for_run = pd.DataFrame()

            if args.max_samples_per_lang is not None and not samples_df_for_run.empty:
                 if len(samples_df_for_run) > args.max_samples_per_lang:
                    samples_df_for_run = samples_df_for_run.sample(n=args.max_samples_per_lang, random_state=sampling_seed)
                    print(f"    Capped samples to {args.max_samples_per_lang} for {lang_code}.")

            if samples_df_for_run.empty:
                print(f"    No MasakhaNEWS samples found or sampled for {lang_code} (split {args.data_split}), skipping.")
                continue
            
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

            # Plotting task performance metrics using overall_plots_dir
            plot_classification_metrics(overall_summary_df, overall_plots_dir, 'accuracy', 'Accuracy')
            plot_classification_metrics(overall_summary_df, overall_plots_dir, 'macro_f1', 'Macro F1-Score')
            plot_classification_metrics(overall_summary_df, overall_plots_dir, 'weighted_f1', 'Weighted F1-Score')
            # Plot COMET scores
            plot_classification_metrics(overall_summary_df, overall_plots_dir, 'avg_comet_lrl_text_to_en', 'COMET Text LRL->EN')
            plot_classification_metrics(overall_summary_df, overall_plots_dir, 'avg_comet_en_label_to_lrl', 'COMET Label EN->LRL')
            print(f"Overall plots saved to: {overall_plots_dir}")
        else:
            print("Overall summary DataFrame is empty. No plots generated.")
    else:
        print("No summaries collected. Skipping overall summary and plot generation for Classification CoTR.")

    print("\nAll Classification CoTR experiments completed!")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main() 