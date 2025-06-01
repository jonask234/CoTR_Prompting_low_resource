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

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run NER baseline experiments with various configurations.")
    parser.add_argument(
        "--models", 
        nargs='+', 
        default=["CohereLabs/aya-expanse-8b", "Qwen/Qwen2.5-7B-Instruct"], 
        help="List of Hugging Face model names or paths (e.g., 'google/gemma-7b-it' or 'CohereLabs/aya-expanse-8b')."
    )
    parser.add_argument(
        "--langs", 
        type=str, 
        default="ha,sw", 
        help="Comma-separated language codes (e.g., ha,sw,yo,pcm,amh,लग). Default is ha,sw."
    )
    parser.add_argument(
        "--sample_percentage", 
        type=float, 
        default=10.0,
        help="Percentage of samples per language from MasakhaNER test split (e.g., 10 for 10%%). Default: 10."
    )
    parser.add_argument(
        "--prompt_in_lrl", 
        action="store_true", 
        help="Use LRL-specific instructions in the prompt. Default is English instructions."
    )
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
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum new tokens to generate for NER output.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty.")

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
    overwrite_results: bool # Added overwrite_results parameter
) -> Optional[Dict[str, Any]]: # Return summary dict or None
    """
    Runs the NER baseline evaluation for one config, saves results, calculates metrics,
    and returns a summary dictionary.
    """
    shot_type_str = "few-shot" if use_few_shot else "zero-shot"
    model_short = model_name.split('/')[-1].replace("/", "_") # Sanitize for filename
    prompt_lang = lang_code # Baseline uses LRL prompts by default now

    print(f"Running experiment: {model_name} on {lang_code} ({shot_type_str})")
    print(f"Effective Params: {effective_params}")

    # Initialize experiment_start_time at the beginning of the function
    experiment_start_time = time.time()

    # Create directories
    results_dir = os.path.join(base_results_path, "results")
    summaries_dir = os.path.join(base_results_path, "summaries")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    # Define file paths (consistent naming)
    results_file = os.path.join(results_dir, f"results_baseline_{shot_type_str[0]}s_ner_{lang_code}_{model_short}.csv")
    summary_file = os.path.join(summaries_dir, f"summary_baseline_{shot_type_str[0]}s_ner_{lang_code}_{model_short}.csv")

    # --- Loading/Running Logic ---
    results_df = pd.DataFrame() # Initialize as empty
    run_evaluation_and_save = False

    if overwrite_results:
        logging.info(f"Overwrite_results is True. Will run experiment for {model_name}, {lang_code}, {shot_type_str}.")
        run_evaluation_and_save = True
    elif not os.path.exists(results_file):
        logging.info(f"Results file does not exist: {results_file}. Will run experiment for {model_name}, {lang_code}, {shot_type_str}.")
        run_evaluation_and_save = True
    else: # File exists and overwrite_results is False
        logging.info(f"Results file exists: {results_file} and overwrite_results is False. Attempting to load.")
        try:
            loaded_df = pd.read_csv(results_file)
            logging.info(f"Successfully loaded existing results from {results_file}.")
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
                    f"'ground_truth_entities' (nor fallbacks 'entities', 'gold_entities') column not found in {results_file}. "
                    f"Columns present: {loaded_df.columns.tolist()}. Will re-run experiment for this configuration as data seems incomplete."
                )
                run_evaluation_and_save = True # Mark for re-run due to missing critical column
        except Exception as e:
            logging.error(f"Could not load or properly process existing results {results_file}: {e}. Will re-run experiment.")
            run_evaluation_and_save = True # Mark for re-run
    
    if run_evaluation_and_save:
        logging.info(f"Proceeding to run evaluation for {model_name}, {lang_code}, {shot_type_str}.")
        # experiment_start_time is already initialized at the function start
        try:
            # Call the core evaluation function from ner_baseline.py
            evaluated_results_df = evaluate_ner_baseline(
                tokenizer=tokenizer,
                model=model,
                model_name=model_name, # Pass model_name for potential internal use/logging
                samples_df=samples_df.copy(), # Pass a copy of the original samples_df
                lang_code=lang_code,
                use_few_shot=use_few_shot,
                prompt_in_lrl=effective_params.get('prompt_in_lrl', False),
                temperature=effective_params["temperature"],
                top_p=effective_params["top_p"],
                top_k=effective_params["top_k"],
                max_tokens=effective_params["max_tokens"],
                repetition_penalty=effective_params["repetition_penalty"],
                do_sample=effective_params["do_sample"] # Pass do_sample flag
            )

            if evaluated_results_df is None or evaluated_results_df.empty:
                logging.error("Baseline evaluation returned None or empty DataFrame. No results to save.")
                # results_df remains empty or as loaded if partial load was successful before deciding to re-run
            else:
                results_df = evaluated_results_df # Use the newly evaluated results
                results_df.to_csv(results_file, index=False)
                logging.info(f"Results saved to {results_file}")

        except Exception as e: 
            logging.error(f"Error during NER Baseline evaluation for {lang_code}, {model_short}, {shot_type_str}: {e}", exc_info=True)
            # results_df remains as it was (empty or loaded) - essentially the run failed
            # No specific return None here, will be caught by the check below.
            
    # --- Calculate Metrics ---
    if results_df.empty: # Check if results_df is empty after load/run attempts
        logging.error(f"No results DataFrame available for {model_name}, {lang_code}, {shot_type_str} after attempting load/run. Skipping summary.")
        return None

    logging.info(f"Calculating metrics from results for {model_name}, {lang_code}, {shot_type_str}...")
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
        'model': model_short,
        'language': lang_code,
        'pipeline': 'baseline', # Indicate baseline
        'shot_type': shot_type_str,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1, # Use 'f1' for consistency
        'num_samples': len(samples_df),
        'num_successful': successful_samples,
        'runtime_seconds': total_runtime,
        **effective_params # Log all effective generation params used for this run
    }

    summary_df = pd.DataFrame([summary_data])
    try:
        summary_path = os.path.join(summaries_dir, f"summary_baseline_{shot_type_str[0]}s_ner_{lang_code}_{model_short}.csv")
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        logging.info(f"Summary metrics saved to {summary_path}")
    except Exception as e_save_sum:
        print(f"ERROR saving summary file {summary_path}: {e_save_sum}")

    return summary_data # Return the dictionary for aggregation

# --- Main Function ---
def main():
    args = parse_cli_args()
    all_experiment_summaries = [] # Ensure initialized at the very top of main
    
    # Setup HuggingFace login
    token = args.hf_token or get_token()
    if token:
        login(token=token)
        logging.info("Successfully logged in to HuggingFace Hub.")
    else:
        logging.warning("No HuggingFace token provided. Some models might be inaccessible.")

    # Ensure base output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    summaries_path = os.path.join(args.output_dir, "summaries")
    plots_path = os.path.join(args.output_dir, "plots")
    os.makedirs(summaries_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    logging.info(f"All Baseline NER experiment outputs will be saved under: {args.output_dir}")

    # Load MasakhaNER samples
    logging.info("\n--- Loading MasakhaNER Data ---")
    masakhaner_samples = {}
    
    # Determine languages to process
    langs_to_process = [lang.strip() for lang in args.langs.split(',') if lang.strip()]

    for lang_code in langs_to_process:
        logging.info(f"Loading MasakhaNER data for {lang_code} (test split, {args.sample_percentage}%) ...")
        try:
            # load_masakhaner_samples handles the percentage internally.
            # It uses split='test' and seed=42 by default.
            df = load_masakhaner_samples(
                lang_code=lang_code,
                sample_percentage=args.sample_percentage
            )
            if df.empty:
                logging.warning(f"No samples returned by load_masakhaner_samples for {lang_code}. Trying fallback.")
                df = create_dummy_ner_data(lang_code=lang_code, num_samples=5 if args.test_mode else 2) # Smaller dummy for non-test
                if df.empty:
                    logging.error(f"Fallback dummy data also empty for {lang_code}. Skipping this language.")
                    continue
                else:
                    logging.info(f"Using fallback dummy data for {lang_code} ({len(df)} samples).")
            else:
                logging.info(f"  Successfully loaded {len(df)} samples for {lang_code} via load_masakhaner_samples.")

        except TypeError as e: # Catch the specific error observed
            if "got an unexpected keyword argument 'num_samples'" in str(e) or \
               "got an unexpected keyword argument 'split'" in str(e) or \
               "got an unexpected keyword argument 'seed'" in str(e):
                logging.error(f"Error calling load_masakhaner_samples for {lang_code} due to argument mismatch: {e}")
                logging.info("Attempting to call load_masakhaner_samples with corrected arguments (lang_code, sample_percentage).")
                try:
                    df = load_masakhaner_samples(lang_code=lang_code, sample_percentage=args.sample_percentage)
                    if df.empty:
                         logging.warning(f"Corrected call to load_masakhaner_samples still returned no samples for {lang_code}. Using fallback.")
                         df = create_dummy_ner_data(lang_code=lang_code, num_samples=5 if args.test_mode else 2)
                         if df.empty:
                             logging.error(f"Fallback dummy data also empty for {lang_code}. Skipping.")
                             continue
                         else:
                             logging.info(f"Using fallback dummy data for {lang_code} ({len(df)} samples).")
                    else:
                        logging.info(f"  Successfully loaded {len(df)} samples for {lang_code} with corrected call.")
                except Exception as e_corrected_call:
                    logging.error(f"Error with corrected call to load_masakhaner_samples for {lang_code}: {e_corrected_call}. Using fallback.")
                    df = create_dummy_ner_data(lang_code=lang_code, num_samples=5 if args.test_mode else 2) # Fallback
                    if df.empty:
                        logging.error(f"Fallback dummy data also empty for {lang_code} after corrected call error. Skipping.")
                        continue
                    else:
                        logging.info(f"Using fallback dummy data for {lang_code} ({len(df)} samples).")

            else: # Other TypeErrors
                logging.error(f"TypeError loading MasakhaNER data for {lang_code}: {e}. Using fallback.")
                df = create_dummy_ner_data(lang_code=lang_code, num_samples=5 if args.test_mode else 2) # Fallback
                if df.empty:
                    logging.error(f"Fallback dummy data also empty for {lang_code}. Skipping this language.")
                    continue
                else:
                    logging.info(f"Using fallback dummy data for {lang_code} ({len(df)} samples).")

        except Exception as e: # Catch any other exception during data loading
            logging.error(f"General error loading MasakhaNER data for {lang_code}: {e}. Using fallback.", exc_info=True)
            df = create_dummy_ner_data(lang_code=lang_code, num_samples=5 if args.test_mode else 2) # Fallback
            if df.empty:
                logging.error(f"Fallback dummy data also empty for {lang_code}. Skipping this language.")
                continue
            else:
                logging.info(f"Using fallback dummy data for {lang_code} ({len(df)} samples).")
        
        # Apply test_mode sample reduction AFTER loading the initial percentage or dummy data
        if args.test_mode and not df.empty:
            num_samples_test_mode = min(len(df), 5)
            if num_samples_test_mode < len(df): # Only sample if we need to reduce
                logging.info(f"Test mode enabled: Reducing samples for {lang_code} from {len(df)} to {num_samples_test_mode}.")
                df = df.sample(n=num_samples_test_mode, random_state=42).reset_index(drop=True)
            else:
                logging.info(f"Test mode enabled: Using all {len(df)} available samples for {lang_code} as it's <= 5.")
        
        if not df.empty:
            masakhaner_samples[lang_code] = df
        else:
            logging.warning(f"Final DataFrame for {lang_code} is empty. Skipping this language.")


    if not masakhaner_samples:
        logging.error("No data loaded for any language. Exiting NER baseline script.")
        return

    # Main experiment loop
    for model_name in args.models:
        logging.info(f"\n====== Starting experiments for model: {model_name} ======")
        tokenizer, model = None, None # Initialize to allow cleanup even if init fails
        try:
            tokenizer, model = initialize_model(model_name) # Defined in ner_baseline.py
            logging.info(f"Model {model_name} initialized successfully.")
        except Exception as e_init:
            logging.error(f"Failed to initialize model {model_name}: {e_init}. Skipping this model.", exc_info=True)
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue # Skip to the next model

        for lang_code in masakhaner_samples: # Iterate over all languages with data
            current_samples_df = masakhaner_samples[lang_code]
            if current_samples_df.empty: # Should not happen if available_languages is built correctly
                logging.warning(f"Unexpected: No samples for {lang_code} at experiment stage. Skipping.")
                continue

            logging.info(f"\n--- Processing Language: {lang_code} for model {model_name} ---")

            # Determine shot settings for the current model and language
            if args.compare_shots:
                shot_configs_to_run = [True, False] # Few-shot and Zero-shot
                logging.info(f"Evaluating {model_name} - {lang_code}: Both Few-shot and Zero-shot.")
            elif args.few_shot:
                shot_configs_to_run = [True] # Few-shot only
                logging.info(f"Evaluating {model_name} - {lang_code}: Few-shot only.")
            else:
                shot_configs_to_run = [False] # Zero-shot only (default)
                logging.info(f"Evaluating {model_name} - {lang_code}: Zero-shot only (default).")
            
            for use_few_shot_config in shot_configs_to_run:
                current_params_for_run = UNIFIED_GENERATION_PARAMETERS_CORE.copy()
                current_params_for_run["do_sample"] = current_params_for_run.get("do_sample", True) and current_params_for_run.get("temperature", 0) > 0
                
                # Apply model-specific overrides (can override core or task-specific max_tokens)
                if model_name in MODEL_SPECIFIC_OVERRIDES_BASELINE:
                    model_overrides = MODEL_SPECIFIC_OVERRIDES_BASELINE[model_name]
                    current_params_for_run.update(model_overrides) # Update with all keys from model_overrides
                    logging.info(f"  Applied model-specific adjustments for {model_name}: {model_overrides}")

                # Apply CLI overrides (highest precedence)
                cli_override_dict = {
                    k: v for k, v in vars(args).items() 
                    if v is not None and k in current_params_for_run # Only override existing keys
                }
                if cli_override_dict:
                    current_params_for_run.update(cli_override_dict)
                    logging.info(f"  Applied CLI overrides: {cli_override_dict}")

                # Ensure do_sample is consistent with temperature
                current_params_for_run["do_sample"] = current_params_for_run.get("do_sample", True) and current_params_for_run.get("temperature", 0) > 0
                
                logging.info(f"  Final Effective Params for {use_few_shot_config}: {json.dumps(current_params_for_run, indent=2)}")
                logging.info(f"  Do Sample: {current_params_for_run['do_sample']}")

                logging.info(f"\n  Starting evaluation: Shot='{'few-shot' if use_few_shot_config else 'zero-shot'}'")
                
                # Call the experiment runner function
                summary_for_run = run_experiment(
                    model_name=model_name,
                    tokenizer=tokenizer,
                    model=model,
                    samples_df=current_samples_df.copy(), # Pass a copy to avoid modification issues
                    lang_code=lang_code,
                    use_few_shot=use_few_shot_config,
                    base_results_path=args.output_dir,
                    effective_params=current_params_for_run, # Pass the fully resolved params
                    overwrite_results=args.overwrite_results # Pass overwrite_results
                )
                if summary_for_run:
                    all_experiment_summaries.append(summary_for_run)
        
        # Clean up model and cache after all its experiments
        logging.info(f"Finished all experiments for model {model_name}. Unloading...")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info(f"GPU memory cache cleared for {model_name}.")

    # After all models and languages, save overall summary and plot
    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        
        # Ensure correct dtypes for plotting and saving, handling potential mixed types
        numeric_cols_expected = ['f1', 'precision', 'recall', 'runtime_seconds', 'samples_processed', 
                                 'temperature', 'top_p', 'top_k', 'max_tokens', 'repetition_penalty']
        for col in numeric_cols_expected:
            if col in overall_summary_df.columns:
                overall_summary_df[col] = pd.to_numeric(overall_summary_df[col], errors='coerce')

        if 'do_sample' in overall_summary_df.columns:
             overall_summary_df['do_sample'] = overall_summary_df['do_sample'].astype(bool)
        
        # Save overall summary CSV
        overall_summary_filename = os.path.join(summaries_path, "baseline_ner_ALL_experiments_summary.csv")
        try:
            overall_summary_df.to_csv(overall_summary_filename, index=False, float_format='%.4f', na_rep='NaN')
            logging.info(f"\nOverall summary of Baseline NER experiments saved to: {overall_summary_filename}")
            logging.info("Overall Summary Table:\n" + overall_summary_df.to_string())
        except Exception as e_csv:
            logging.error(f"Error saving overall summary CSV: {e_csv}")

        # Plotting F1 scores
        try:
            plot_ner_f1_scores(overall_summary_df, plots_path) # Function defined in this script
            logging.info(f"Overall F1 score plot saved to: {plots_path}")
        except Exception as e_plot:
            logging.error(f"Error generating F1 score plot: {e_plot}")
    else:
        logging.info("No NER baseline experiments were successfully summarized. Skipping overall summary and plot generation.")

    logging.info("\n====== NER Baseline Script Finished ======")

# --- Plotting Function (Optional) ---
def plot_ner_f1_scores(summary_df, plots_dir):
    """Generates and saves a bar plot of F1 scores."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Plotting requires matplotlib and seaborn. Please install them.")
        return

    if summary_df.empty or 'f1' not in summary_df.columns:
        print("Summary DataFrame is empty or missing 'f1' column, skipping plot.")
        return

    plt.figure(figsize=(15, 8))
    try:
        # Create a combined categorical column for plotting
        summary_df['experiment_config'] = summary_df['language'] + '-' + summary_df['model'] + '-' + summary_df['shot_type']
        
        sns.barplot(data=summary_df, x='experiment_config', y='f1', palette='viridis') # Use 'f1' column
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Average F1 Scores for Baseline NER Experiments', fontsize=14)
        plt.ylabel('Average F1 Score', fontsize=12)
        plt.xlabel('Experiment Configuration (Lang-Model-Shot)', fontsize=12)
        plt.ylim(0, max(summary_df['f1'].max() * 1.1, 0.1)) # Adjust y-axis limit
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        
        plot_filename = os.path.join(plots_dir, "baseline_ner_f1_scores.png")
        plt.savefig(plot_filename)
        plt.close() # Close the plot figure
        print(f"F1 score plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error generating F1 plot: {e}")

# --- Entry Point ---
if __name__ == "__main__":
    main() 