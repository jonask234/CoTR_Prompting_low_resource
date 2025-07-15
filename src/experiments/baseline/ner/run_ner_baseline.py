#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import torch
import time
from tqdm import tqdm
import numpy as np
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import ast
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path: # Prevent duplicates
    sys.path.insert(0, project_root)

# Import necessary functions
from src.utils.data_loaders.load_masakhaner import load_masakhaner_samples
from src.experiments.baseline.ner.ner_baseline import (
    initialize_model,
    evaluate_ner_baseline,
    calculate_ner_metrics,
    create_dummy_ner_data
)
from config import get_token
from huggingface_hub import login

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Unified Generation Parameters ---
# Default values for core generation parameters
UNIFIED_GENERATION_PARAMETERS_CORE = {
    "temperature": 0.2,  # Slightly lower for more deterministic NER
    "top_p": 0.85,
    "top_k": 35,
    "repetition_penalty": 1.1,
    "do_sample": True # Default to sampling, temp will control it
}

# Default task-specific max_tokens for Baseline NER (for the generated entity list)
MAX_TOKENS_BASELINE_NER_OUTPUT = 150 

# --- Model-Specific Overrides ---
# These can override UNIFIED_GENERATION_PARAMETERS_CORE or MAX_TOKENS_BASELINE_NER_OUTPUT
MODEL_SPECIFIC_OVERRIDES_BASELINE = {
    "CohereLabs/aya-expanse-8b": {
        "temperature": 0.225, # Stricter for Aya (example)
        "repetition_penalty": 1.15,
        "max_tokens": 180, 
        "top_p": 0.85,
        "top_k": 35,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "temperature": 0.15,
        "top_p": 0.80,
        "max_tokens": 160,
    }
}

logger = logging.getLogger(__name__)

# Model-specific adjustments (can be expanded)
# These are base settings that can be overridden by CLI arguments.
MODEL_SPECIFIC_ADJUSTMENTS = {
    "CohereLabs/aya-expanse-8b": {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 30,
        "repetition_penalty": 1.1,
        "max_tokens": 150 
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "temperature": 0.1,
        "top_p": 0.85,
        "top_k": 35,
        "repetition_penalty": 1.05,
        "max_tokens": 160
    }
    # Add other models as needed
}

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run Baseline NER Experiments with MasakhaNER.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated model names.")
    parser.add_argument("--langs", type=str, default="swa,hau", help="Comma-separated MasakhaNER language codes (e.g., swa, hau for Swahili, Hausa from MasakhaNER). Ensure these match dataset keys.")
    parser.add_argument("--num_samples", type=int, default=80, help="Number of samples per language. Default: 80")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split.")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/ner/baseline_masakhaner", help="Base directory to save results and summaries.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and reproducibility.")
    parser.add_argument(
        "--prompt_in_lrl", 
        action=argparse.BooleanOptionalAction, # Allows --prompt_in_lrl and --no-prompt_in_lrl
        default=True, # Set default to True
        help="If set, prompt instructions will be in LRL (few-shot examples remain in English). Default: True (LRL instructions)."
    )
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Prompting strategies.")
    parser.add_argument(
        "--compare_shots", 
        action="store_true", 
        help="Run both few-shot (English examples) and zero-shot evaluations. Overrides --few_shot."
    )
    parser.add_argument(
        "--few_shot", 
        action="store_true", 
        help="Enable few-shot prompting (English examples). If --compare_shots is also set, this is ignored as both are run."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/work/bbd6522/results/ner/baseline", 
        help="Base output directory for results, summaries, and plots."
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default=None, 
        help="HuggingFace API token (optional, reads from config if not provided)."
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="If set, overwrite existing result files instead of skipping experiments."
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run in test mode: uses a very small number of samples (e.g., 5) for quick testing."
    )
    # Generation parameter arguments
    parser.add_argument("--temperature", type=float, default=None, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling top_p.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Global override for max_new_tokens for generation, affecting the NER output length.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Global override for repetition_penalty.")
    parser.add_argument(
        "--do_sample", 
        type=lambda x: (str(x).lower() == 'true'), 
        default=None,  # Default to None, so it can be derived from temperature if not set
        help="Global override for do_sample (True/False). If not set, derived from temperature."
    )
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")

    args = parser.parse_args()
    return args

# --- Standardized Parameters (Aligned with CoTR) ---
STANDARD_PARAMETERS = {
    "temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_tokens": 200, "repetition_penalty": 1.1
}

# Language-specific parameters (Aligned with CoTR)
LANGUAGE_PARAMETERS = {
    "sw": { # Swahili (Example - Adjust as needed)
        "temperature": 0.25, "top_p": 0.85, "top_k": 35, "max_tokens": 180, "repetition_penalty": 1.15
    },
    "ha": { # Hausa (Example - Adjust as needed)
        "temperature": 0.25, "top_p": 0.85, "top_k": 35, "max_tokens": 180, "repetition_penalty": 1.15
    }
    # Add other languages if tuning is done
}

# --- Model Specific Adjustments (Function) ---
def apply_model_specific_adjustments(params: Dict, model_name: str) -> Dict:
    """Applies model-specific adjustments to parameters."""
    adjusted_params = params.copy()
    if "aya" in model_name.lower():
        adjusted_params["temperature"] = max(0.1, adjusted_params["temperature"] * 0.9)
        print(f"  Applied Aya adjustments: Temp={adjusted_params['temperature']:.2f}")
    elif "qwen" in model_name.lower():
        adjusted_params["top_p"] = max(0.7, adjusted_params["top_p"] * 0.9)
        adjusted_params["top_k"] = 35
        print(f"  Applied Qwen adjustments: TopP={adjusted_params['top_p']:.2f}, TopK={adjusted_params['top_k']}")
    return adjusted_params

# --- Refactored Experiment Runner Function ---
def run_experiment(
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    use_few_shot: bool,
    base_results_path: str,
    effective_params: Dict[str, Any],
    overwrite_results: bool,
    prompt_in_lrl_cli: bool
) -> Optional[Dict[str, Any]]: # Return summary dict or None
    """Run a single NER experiment for a given model, language, and configuration."""
    experiment_start_time = time.time() # Initialize at the start of the function

    if samples_df.empty:
        logger.warning(f"No samples for {lang_code}, skipping {model_name}.")
        return None

    model_short_name = model_name.split('/')[-1]

    # Simplified: Baseline now always uses EN-instruct prompts
    shot_setting_str = "few_shot" if use_few_shot else "zero_shot"
    # Determine prompt language string based on the passed-in flag
    if prompt_in_lrl_cli:
        prompt_lang_str = f"{lang_code}-instruct"
    else:
        prompt_lang_str = "EN-instruct" 
    logger.info(f"Prompt language for this run: {prompt_lang_str}") # Added log
    
    results_dir_for_run = os.path.join(base_results_path, "results")
    summaries_dir_for_run = os.path.join(base_results_path, "summaries")
    os.makedirs(results_dir_for_run, exist_ok=True)
    os.makedirs(summaries_dir_for_run, exist_ok=True)

    # Define file paths (consistent naming)
    results_file_for_run = os.path.join(results_dir_for_run, f"results_baseline_{shot_setting_str[0]}s_ner_{lang_code}_{model_short_name}.csv")
    summary_file_for_run = os.path.join(summaries_dir_for_run, f"summary_baseline_{shot_setting_str[0]}s_ner_{lang_code}_{model_short_name}.csv")

    # --- Loading/Running Logic ---
    results_df = pd.DataFrame() # Initialize as empty
    run_evaluation_and_save = False

    if overwrite_results:
        logging.info(f"Overwrite_results is True. Will run experiment for {model_name}, {lang_code}, {shot_setting_str}.")
        run_evaluation_and_save = True
    elif not os.path.exists(results_file_for_run):
        logging.info(f"Results file does not exist: {results_file_for_run}. Will run experiment for {model_name}, {lang_code}, {shot_setting_str}.")
        run_evaluation_and_save = True
    else: # File exists and overwrite_results is False
        logging.info(f"Results file exists: {results_file_for_run} and overwrite_results is False. Attempting to load.")
        try:
            loaded_df = pd.read_csv(results_file_for_run)
            logging.info(f"Successfully loaded existing results from {results_file_for_run}.")
            # Check for the target column 'ground_truth_entities'
            if 'ground_truth_entities' in loaded_df.columns:
                results_df = loaded_df
            elif 'entities' in loaded_df.columns: # Fallback 1: 'entities'
                logging.info("Found 'entities' column in existing CSV; renaming to 'ground_truth_entities'.")
                results_df = loaded_df.rename(columns={'entities': 'ground_truth_entities'})
            elif 'gold_entities' in loaded_df.columns: # Fallback 2: 'gold_entities'
                logging.info("Found 'gold_entities' column in existing CSV; renaming to 'ground_truth_entities'.")
                results_df = loaded_df.rename(columns={'gold_entities': 'ground_truth_entities'})
            else:
                logging.warning(
                    f"'ground_truth_entities' (nor fallbacks 'entities', 'gold_entities') column not found in {results_file_for_run}. "
                    f"Columns present: {loaded_df.columns.tolist()}. Will re-run experiment for this configuration as data seems incomplete."
                )
                run_evaluation_and_save = True # Mark for re-run due to missing critical column
        except Exception as e:
            logging.error(f"Could not load or properly process existing results {results_file_for_run}: {e}. Will re-run experiment.")
            run_evaluation_and_save = True # Mark for re-run
    
    if run_evaluation_and_save:
        logging.info(f"Proceeding to run evaluation for {model_name}, {lang_code}, {shot_setting_str}.")
        # Run the core evaluation if results_df is not loaded
        if results_df.empty:
            logger.info(f"Running NER Baseline: Model={model_name}, Lang={lang_code}, Shot Type={shot_setting_str}, Prompt Lang={prompt_lang_str}")
            try:
                results_df = evaluate_ner_baseline(
                    tokenizer=tokenizer,
                    model=model,
                    model_name=model_name, # Pass model_name for potential internal use/logging
                    samples_df=samples_df.copy(), # Pass a copy of the original samples_df
                    lang_code=lang_code,
                    use_few_shot=use_few_shot,
                    prompt_in_lrl=prompt_in_lrl_cli,
                    temperature=effective_params["temperature"],
                    top_p=effective_params["top_p"],
                    top_k=effective_params["top_k"],
                    max_tokens=effective_params["max_tokens"],
                    repetition_penalty=effective_params["repetition_penalty"],
                    do_sample=effective_params["do_sample"] # Pass do_sample flag
                )

                if results_df is None or results_df.empty:
                    logging.error("Baseline evaluation returned None or empty DataFrame. No results to save.")
                    # results_df remains empty or as loaded if partial load was successful before deciding to re-run
                else:
                    results_df.to_csv(results_file_for_run, index=False)
                    logging.info(f"Results saved to {results_file_for_run}")

            except Exception as e: 
                logging.error(f"Error during NER Baseline evaluation for {lang_code}, {model_short_name}, {shot_setting_str}: {e}", exc_info=True)
                # results_df remains as it was (empty or loaded) - essentially the run failed
                # No specific return None here, will be caught by the check below.
            
    # --- Calculate Metrics ---
    if results_df.empty: # Check if results_df is empty after load/run attempts
        logging.error(f"No results DataFrame available for {model_name}, {lang_code}, {shot_setting_str} after attempting load/run. Skipping summary.")
        return None

    logging.info(f"Calculating metrics from results for {model_name}, {lang_code}, {shot_setting_str}...")
    precisions, recalls, f1s = [], [], []
    successful_samples = 0

    # Convert string representation of lists/tuples back if loaded from CSV
    needs_conversion = False
    # Check the type of the first non-null element
    first_gold = results_df['ground_truth_entities'].dropna().iloc[0] if not results_df['ground_truth_entities'].dropna().empty else None
    if first_gold is not None and isinstance(first_gold, str):
        needs_conversion = True
        print("Converting entity strings from CSV back to lists/tuples...")
        def safe_literal_eval(val):
            # Handles potential errors during eval more gracefully
            if pd.isna(val): return []
            try:
                evaluated = ast.literal_eval(val)
                # Ensure it's a list after eval
                return evaluated if isinstance(evaluated, list) else []
            except (ValueError, SyntaxError, TypeError):
                print(f"Warning: Could not evaluate entity string: {str(val)[:50]}...")
                return [] # Return empty list on error

    # Check for predicted_entities column and its type for conversion
    needs_pred_conversion = False
    if 'predicted_entities' in results_df.columns:
        first_pred = results_df['predicted_entities'].dropna().iloc[0] if not results_df['predicted_entities'].dropna().empty else None
        if first_pred is not None and isinstance(first_pred, str):
            needs_pred_conversion = True
            if not needs_conversion: # Only print this if not already printed for gold entities
                print("Converting predicted entity strings from CSV back to lists/tuples...")
    else:
        # Fallback for 'predicted_entities' if missing
        if 'predictions' in results_df.columns:
            logging.info("Found 'predictions' column in existing CSV; renaming to 'predicted_entities'.")
            results_df.rename(columns={'predictions': 'predicted_entities'}, inplace=True)
            # Re-check for conversion after renaming
            first_pred = results_df['predicted_entities'].dropna().iloc[0] if not results_df['predicted_entities'].dropna().empty else None
            if first_pred is not None and isinstance(first_pred, str):
                needs_pred_conversion = True
        else:
            logging.warning(
                f"'predicted_entities' (and fallback 'predictions') column not found in results_df. "
                f"Columns present: {results_df.columns.tolist()}. Metrics for predictions will be 0."
            )
            results_df['predicted_entities'] = [[] for _ in range(len(results_df))] # Fill with empty lists

    for idx, row in results_df.iterrows():
        try:
            # Apply conversion if needed
            gold = safe_literal_eval(row['ground_truth_entities']) if needs_conversion else row['ground_truth_entities']
            
            # Handle predicted_entities conversion or missing column
            if 'predicted_entities' in row:
                pred = safe_literal_eval(row['predicted_entities']) if needs_pred_conversion else row['predicted_entities']
            else: # Should be handled by the column creation above, but as a safeguard
                pred = []

            # Ensure data types are list before passing to metrics
            if not isinstance(gold, list): gold = []
            if not isinstance(pred, list): pred = []

            # Use the metrics function from ner_baseline.py
            metrics = calculate_ner_metrics(gold, pred)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            # Ensure the key matches what calculate_ner_metrics returns ('f1_score')
            f1s.append(metrics.get('f1_score', 0.0)) # Use .get() for safety
            successful_samples += 1
        except Exception as e_metric:
            print(f"Error calculating metrics for row {idx}: {e_metric}")
            # Append default values on error to avoid crashing
            precisions.append(0.0); recalls.append(0.0); f1s.append(0.0)

    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    avg_f1 = np.mean(f1s) if f1s else 0.0
    total_runtime = time.time() - experiment_start_time # Calculate total runtime

    summary_data = {
        "model_name": model_short_name, 
        "language": lang_code,
        "prompt_lang": prompt_lang_str, # This is now dynamic
        "shot_setting": shot_setting_str,
        "num_samples_processed": len(results_df),
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1,
        "runtime_seconds": total_runtime,
        **effective_params # Log all effective generation params used for this run
    }

    summary_df = pd.DataFrame([summary_data])
    try:
        summary_path = os.path.join(summaries_dir_for_run, f"summary_baseline_{shot_setting_str[0]}s_ner_{lang_code}_{model_short_name}.csv")
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        logging.info(f"Summary metrics saved to {summary_path}")
    except Exception as e_save_sum:
        print(f"ERROR saving summary file {summary_path}: {e_save_sum}")

    return summary_data # Return the dictionary for aggregation

# --- Main Function ---
def main():
    args = parse_cli_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Setup seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # HF Login
    token = args.hf_token if args.hf_token else get_token()
    if token:
        try:
            login(token=token)
            logging.info("Successfully logged into Hugging Face Hub.")
        except Exception as e:
            logging.error(f"Failed to login to Hugging Face Hub: {e}. Gated models might be inaccessible.")
    else:
        logging.warning("No Hugging Face token provided. Gated models might be inaccessible.")

    model_names_list = [m.strip() for m in args.models.split(',')]
    lang_codes_list = [lang.strip() for lang in args.langs.split(',')]

    all_experiment_summaries = []
    overall_summary_dir = os.path.join(args.base_output_dir, "summaries_overall") 
    os.makedirs(overall_summary_dir, exist_ok=True)
    overall_plots_dir = os.path.join(args.base_output_dir, "plots_overall")
    os.makedirs(overall_plots_dir, exist_ok=True)

    logging.info(f"NER Baseline Outputs Target: {args.base_output_dir}")
    logging.info(f"Overall Summaries in: {overall_summary_dir}")
    logging.info(f"Overall Plots in: {overall_plots_dir}")

    num_samples_to_load = args.num_samples
    current_seed = args.seed

    for model_name_str in model_names_list:
        logging.info(f"\n===== Initializing Model: {model_name_str} =====")
        
        # Explicitly delete previous model and tokenizer, and clear cache
        # This ensures resources from a previous model (even if it failed to load fully)
        # are released before attempting to load the next one.
        model = None
        tokenizer = None
        del model
        del tokenizer
        gc.collect() # Suggest garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"Cleared CUDA cache and called gc.collect() before initializing {model_name_str}")
            
        current_model_tokenizer, current_model_object = None, None # Initialize for this iteration
        try:
            current_model_tokenizer, current_model_object = initialize_model(model_name_str)
        except Exception as e_init:
            logging.error(f"Failed to initialize model {model_name_str}: {e_init}. Skipping this model.", exc_info=True)
            if current_model_object is not None: del current_model_object
            if current_model_tokenizer is not None: del current_model_tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect() # Ensure garbage collection after del and cache clear
            continue # Skip to the next model

        # Defensive check: Ensure model and tokenizer are loaded before proceeding
        if current_model_tokenizer is None or current_model_object is None:
            logging.critical(f"Model or tokenizer is None for {model_name_str} even after initialize_model call did not raise an exception. This should not happen. Skipping model.")
            # Perform cleanup similar to the exception block
            if current_model_object is not None: del current_model_object
            if current_model_tokenizer is not None: del current_model_tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
            continue # Skip to the next model
            
        # If we are here, initialize_model succeeded, and model/tokenizer are not None.
        effective_gen_params = get_effective_generation_params(model_name_str, args)

        for lang_code in lang_codes_list:
            logger.info(f"\n--- Processing Language: {lang_code} for model {model_name_str} ---")
            
            current_samples_df = load_masakhaner_samples(
                lang_code=lang_code, 
                split=args.data_split, 
                num_samples=args.num_samples, 
                seed=args.seed
            )

            if current_samples_df.empty:
                logger.warning(f"No MasakhaNER samples for {lang_code} (split {args.data_split}). Skipping config.")
                continue
            else:
                logger.info(f"Loaded {len(current_samples_df)} samples for {lang_code} (split {args.data_split}).")

            # Fixed prompt language configuration: EN-instruct only for baseline simplification
            prompt_lang_code_fixed = "EN"
            use_lrl_prompt_bool_fixed = False
            prompt_lang_str_fixed = "EN-instruct" # This is determined by use_lrl_prompt_bool_fixed

            # Iterate through specified shot settings from CLI
            for shot_setting_str in args.shot_settings: # e.g., ["zero_shot", "few_shot"]
                use_few_shot_bool = (shot_setting_str == "few_shot")
                
                # Determine the actual prompt language string for logging/summary based on args.prompt_in_lrl
                actual_prompt_lang_for_log = f"{lang_code}-instruct" if args.prompt_in_lrl else "EN-instruct"
                
                exp_config_tuple = (lang_code, model_name_str, shot_setting_str, prompt_lang_str_fixed)
                logger.info(f"\n--- Running NER Baseline: Model={model_name_str}, Lang={lang_code}, Shot={shot_setting_str}, PromptLang={actual_prompt_lang_for_log} ---")
                logger.info(f"  Effective Generation Params: {effective_gen_params}")

                experiment_details_for_run = {
                    "model_name": model_name_str,
                    "tokenizer": current_model_tokenizer,
                    "model": current_model_object,
                    "samples_df": current_samples_df,
                    "lang_code": lang_code,
                    "use_few_shot": use_few_shot_bool,
                    "base_results_path": args.base_output_dir,
                    "effective_params": effective_gen_params,
                    "overwrite_results": args.overwrite_results,
                    "prompt_in_lrl_cli": args.prompt_in_lrl
                }
                
                # Call run_experiment
                summary_data = run_experiment(**experiment_details_for_run)
                
                if summary_data:
                    all_experiment_summaries.append(summary_data)
            # End of shot_setting_str loop
        # End of lang_code loop
        
        # Correctly indented model cleanup block:
        logging.info(f"Finished all experiments for model {model_name_str}. Model and tokenizer unloaded, cache cleared.")
        if current_model_object is not None:
            del current_model_object
            current_model_object = None # Ensure it's None after del
        if current_model_tokenizer is not None:
            del current_model_tokenizer
            current_model_tokenizer = None # Ensure it's None after del
        
        # Suggest garbage collection more aggressively
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"CUDA cache cleared after processing model {model_name_str}")

    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        if not overall_summary_df.empty:
            # Ensure numeric columns are numeric for plotting and display
            numeric_cols_for_summary = ['num_samples_processed', 'avg_runtime_seconds_per_sample', 'precision', 'recall', 'f1_score',
                                 'temperature', 'top_p', 'top_k', 'max_tokens', 'repetition_penalty']
            for nc in numeric_cols_for_summary:
                if nc in overall_summary_df.columns:
                    overall_summary_df[nc] = pd.to_numeric(overall_summary_df[nc], errors='coerce')
            
            summary_filename_overall = f"ner_baseline_ALL_experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            overall_summary_path_csv = os.path.join(overall_summary_dir, summary_filename_overall)
            overall_summary_df.to_csv(overall_summary_path_csv, index=False, float_format='%.4f')
            logging.info(f"\nOverall summary of NER Baseline experiments saved to: {overall_summary_path_csv}")
            print("\nOverall NER Baseline Summary:")
            try:
                print(overall_summary_df.to_string(index=False))
            except Exception as e_print_df:
                logging.error(f"Could not print overall_summary_df to string: {e_print_df}")
                print(overall_summary_df.head())

            # Plotting (example for F1 score)
            if 'f1_score' in overall_summary_df.columns:
                # Create 'experiment_key' for a unique x-axis identifier
                overall_summary_df['experiment_key'] = overall_summary_df['model_name'] + "_" + \
                                                       overall_summary_df['language'] + "_" + \
                                                       overall_summary_df['shot_setting']
                
                plt.figure(figsize=(15, 8))
                sns.barplot(data=overall_summary_df, x='experiment_key', y='f1_score', hue='language')
                plt.xticks(rotation=45, ha='right')
                plt.title('NER Baseline F1 Scores by Experiment Configuration')
                plt.ylabel('F1 Score')
                plt.xlabel('Experiment Configuration')
                plt.tight_layout()
                plot_path = os.path.join(overall_plots_dir, "ner_baseline_f1_scores.png")
                try:
                    plt.savefig(plot_path)
                    logging.info(f"F1 score plot saved to {plot_path}")
                except Exception as e_plot:
                    logging.error(f"Failed to save F1 plot: {e_plot}")
                finally:
                    plt.close() # Ensure plot is closed
            else:
                logging.warning("'f1_score' column not found in overall summary. Skipping plot.")
        else:
            logging.info("Overall summary DataFrame is empty. No CSV or plots generated.")
    else:
        logging.info("No summaries were collected from any experiment. No overall summary or plots.")

    logging.info("\nAll NER Baseline experiments completed.")

def get_effective_generation_params(model_name_str: str, cli_args: argparse.Namespace) -> Dict[str, Any]:
    """Get effective generation parameters, prioritizing CLI, then model-specific, then defaults."""
    
    # Start with model-specific settings, or an empty dict if model not listed
    params = MODEL_SPECIFIC_ADJUSTMENTS.get(model_name_str, {}).copy()
    
    # Fallback defaults if not in model-specific (or if model-specific is empty)
    # These are general defaults if a model isn't in MODEL_SPECIFIC_ADJUSTMENTS
    # or if certain params are missing from its specific config.
    default_fallbacks = {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.0,
        "max_tokens": 150, # Default max_tokens for NER output
        "do_sample": True # Default do_sample, will be adjusted by temperature later
    }
    for key, value in default_fallbacks.items():
        params.setdefault(key, value)

    # Apply CLI overrides (highest priority)
    cli_overrides = {
        "temperature": cli_args.temperature,
        "top_p": cli_args.top_p,
        "top_k": cli_args.top_k,
        "repetition_penalty": cli_args.repetition_penalty,
        "max_tokens": cli_args.max_tokens # This now refers to max_new_tokens for generation
    }
    if cli_args.do_sample is not None: # Explicitly handle do_sample from CLI
        cli_overrides["do_sample"] = cli_args.do_sample

    for key, value in cli_overrides.items():
        if value is not None:
            params[key] = value
            
    # If do_sample was not set by CLI, derive it from the final temperature
    if cli_args.do_sample is None:
        if params["temperature"] is not None and params["temperature"] <= 0.01: # or exactly 0
            params["do_sample"] = False
        else:
            params["do_sample"] = True # Explicitly set if temperature suggests sampling
    
    # Ensure max_tokens is suitable for NER (short responses generally)
    # This also acts as a final default if no other max_tokens was set.
    params["max_tokens"] = params.get("max_tokens", 150) 

    logging.debug(f"Effective generation parameters for {model_name_str}: {params}")
    return params

def plot_ner_metrics(summary_df: pd.DataFrame, plots_dir: str):
    """Generate and save plots for NER metrics."""
    if summary_df.empty:
        logger.info("Summary DataFrame is empty. Skipping plotting.")
        return

    # Define required columns for plotting
    # 'experiment_config' is typically generated by config_to_str, so ensure source columns are present.
    required_cols = ['model_name', 'language', 'shot_setting', 'prompt_lang', 'f1_score'] 

    # Check if necessary columns are present
    if not all(col in summary_df.columns for col in required_cols):
        logger.warning(f"One or more required columns ({required_cols}) not found in summary_df. Skipping plotting.")
        logger.debug(f"Available columns: {summary_df.columns.tolist()}")
        return

    # Ensure f1_score is numeric
    # ... (rest of the function remains unchanged)

# --- Entry Point ---
if __name__ == "__main__":
    main() 