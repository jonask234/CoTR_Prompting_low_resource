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

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path: # Prevent duplicates
    sys.path.insert(0, project_root)

# Import necessary functions
from src.experiments.baseline.ner.ner_baseline import (
    initialize_model,
    evaluate_ner_baseline,
    calculate_ner_metrics,
    load_masakhaner_samples
)
from config import get_token
from huggingface_hub import login

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    # Add all generation parameters
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
    do_sample: bool # Added do_sample flag
) -> Optional[Dict[str, Any]]: # Return summary dict or None
    """
    Runs the NER baseline evaluation for one config, saves results, calculates metrics,
    and returns a summary dictionary.
    """
    shot_type_str = "few-shot" if use_few_shot else "zero-shot"
    model_short = model_name.split('/')[-1].replace("/", "_") # Sanitize for filename
    prompt_lang = lang_code # Baseline uses LRL prompts by default now

    print(f"Running experiment: {model_name} on {lang_code} ({shot_type_str})")
    print(f"Effective Params: temp={temperature:.2f}, top_p={top_p:.2f}, top_k={top_k}, max_tokens={max_tokens}, rep_penalty={repetition_penalty:.2f}, do_sample={do_sample}")

    # Create directories
    results_dir = os.path.join(base_results_path, "results")
    summaries_dir = os.path.join(base_results_path, "summaries")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    # Define file paths (consistent naming)
    results_file = os.path.join(results_dir, f"results_baseline_{shot_type_str[0]}s_ner_{lang_code}_{model_short}.csv")
    summary_file = os.path.join(summaries_dir, f"summary_baseline_{shot_type_str[0]}s_ner_{lang_code}_{model_short}.csv")

    # --- Loading/Running Logic ---
    results_df = None
    experiment_start_time = time.time() # Start timer before potential loading/running

    if os.path.exists(results_file):
        print(f"Results file exists: {results_file}")
        if os.path.exists(summary_file):
            try:
                existing_summary_df = pd.read_csv(summary_file)
                print(f"Loaded existing summary from {summary_file}")
                # Return the summary data to avoid re-running
                # Convert DataFrame row back to dictionary
                return existing_summary_df.iloc[0].to_dict()
            except Exception as e:
                print(f"Could not load existing summary file {summary_file}: {e}. Will regenerate results.")
                results_df = None # Force regeneration
        else:
            # Summary missing, but results exist. Load results to generate summary.
            try:
                print(f"Attempting to load existing results from {results_file} to generate summary...")
                results_df = pd.read_csv(results_file)
            except Exception as e:
                print(f"Could not load existing results: {e}. Cannot generate summary.")
                return None
    
    # If results_df is still None, run the evaluation
    if results_df is None:
        try:
            # Call the core evaluation function from ner_baseline.py
        results_df = evaluate_ner_baseline(
                tokenizer=tokenizer,
                model=model,
                model_name=model_name, # Pass model_name for potential internal use/logging
            samples_df=samples_df,
            lang_code=lang_code,
                use_few_shot=use_few_shot,
                prompt_in_lrl=True, # Baseline defaults to LRL prompt
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample # Pass do_sample flag
            )

            if results_df is None or results_df.empty:
                print("ERROR: Baseline evaluation returned empty DataFrame.")
                return None # Return None if evaluation failed

            results_df.to_csv(results_file, index=False)
            print(f"Results saved to {results_file}")

        except Exception as e: # Added except block
            logging.error(f"Error during NER Baseline evaluation for {lang_code}, {model_short}, {shot_type_str}: {e}", exc_info=True)
            return None # Indicate failure
            
    # --- Calculate Metrics ---
    if results_df is None or results_df.empty:
        print("Cannot calculate metrics: No results available.")
        return None

    print(f"Calculating metrics from results...")
    precisions, recalls, f1s = [], [], []
    successful_samples = 0

    # Convert string representation of lists/tuples back if loaded from CSV
    needs_conversion = False
    # Check the type of the first non-null element
    first_gold = results_df['gold_entities'].dropna().iloc[0] if not results_df['gold_entities'].dropna().empty else None
    if first_gold is not None and isinstance(first_gold, str):
        needs_conversion = True
        print("Converting entity strings from CSV back to lists/tuples...")
        import ast
        def safe_literal_eval(val):
            # Handles potential errors during eval more gracefully
            if pd.isna(val): return []
            try:
                evaluated = ast.literal_eval(val)
                # Ensure it's a list after eval
                return evaluated if isinstance(evaluated, list) else []
            except (ValueError, SyntaxError, TypeError):
                print(f"Warning: Could not evaluate entity string: {val[:50]}...")
                return [] # Return empty list on error

    for idx, row in results_df.iterrows():
        try:
            # Apply conversion if needed
            gold = safe_literal_eval(row['gold_entities']) if needs_conversion else row['gold_entities']
            pred = safe_literal_eval(row['predicted_entities']) if needs_conversion else row['predicted_entities']

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
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'max_tokens': max_tokens,
        'repetition_penalty': repetition_penalty,
        'do_sample': do_sample
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
    parser = argparse.ArgumentParser(description="Run NER baseline experiments with standardized parameters.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-expanse-8b,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated list of models")
    parser.add_argument("--langs", type=str, default="sw,ha", help="Comma-separated list of languages")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per language")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Shot settings to evaluate")
    parser.add_argument("--hf-token", type=str, help='HuggingFace API token (if needed)')
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/ner/baseline", help="Base directory for outputs.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    # Add CLI overrides for generation parameters
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Override top_p")
    parser.add_argument("--top_k", type=int, default=None, help="Override top_k")
    parser.add_argument("--max_tokens", type=int, default=None, help="Override max_tokens")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Override repetition_penalty")
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True, help="Override do_sample (use --do-sample or --no-do-sample)") # Use BooleanOptionalAction
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    token = get_token()
    if token: login(token=token)
    
    models_list = [m.strip() for m in args.models.split(',')]
    langs_list = [l.strip() for l in args.langs.split(',')]
    
    # Ensure output directories exist
    os.makedirs(args.base_output_dir, exist_ok=True)
    summaries_output_dir = os.path.join(args.base_output_dir, "summaries")
    plots_output_dir = os.path.join(args.base_output_dir, "plots") # Keep plots dir separate
    os.makedirs(summaries_output_dir, exist_ok=True)
    os.makedirs(plots_output_dir, exist_ok=True)
    print(f"All Baseline NER experiment outputs will be saved under: {args.base_output_dir}")

    all_experiment_summaries = [] # To collect summaries from all runs

    # --- Load Data Once ---
    print("--- Loading MasakhaNER Data ---")
    masakhaner_samples = {}
    for lang_code in langs_list:
        print(f"Loading {args.split} data for {lang_code}...") # Specify split
        samples = load_masakhaner_samples(lang_code, num_samples=args.samples, split=args.split, seed=args.seed)
        if not samples.empty:
            # Ensure 'text' and 'entities' columns exist after loading
            if 'text' not in samples.columns or 'entities' not in samples.columns:
                 print(f"ERROR: Loaded data for {lang_code} missing required columns ('text', 'entities'). Skipping.")
                 masakhaner_samples[lang_code] = pd.DataFrame() # Mark as empty
            else:
                masakhaner_samples[lang_code] = samples
                print(f"Loaded and prepared {len(samples)} samples for {lang_code}.")
        else:
            print(f"WARNING: No samples loaded for {lang_code} from split '{args.split}'.")
            masakhaner_samples[lang_code] = pd.DataFrame() # Mark as empty

    # --- Main Experiment Loop ---
    for model_name_str in models_list:
        print(f"\n====== Starting experiments for model: {model_name_str} ======")
        model_initialized = False
        tokenizer, model = None, None
        try:
            tokenizer, model = initialize_model(model_name_str)
            model_initialized = True
            print(f"Model {model_name_str} initialized successfully.")
        except Exception as e_init:
            print(f"ERROR: Failed to initialize model {model_name_str}: {e_init}. Skipping this model.")
            continue # Skip to the next model if initialization fails

        for lang_code in langs_list:
            print(f"\n--- Processing Language: {lang_code} for model {model_name_str} ---")
            # Check if samples are valid for this language
            if lang_code not in masakhaner_samples or masakhaner_samples[lang_code].empty:
                print(f"  Skipping {lang_code} due to missing or invalid data.")
                continue # Skip to the next language

            current_samples_df = masakhaner_samples[lang_code]

            # Determine effective parameters for this lang/model, considering CLI overrides
            # Start with standard defaults
            base_params = STANDARD_PARAMETERS.copy()
            # Update with language-specific defaults
            base_params.update(LANGUAGE_PARAMETERS.get(lang_code, {}))
            # Apply model-specific adjustments
            effective_params = apply_model_specific_adjustments(base_params, model_name_str)
            
            # Apply CLI overrides
            if args.temperature is not None: effective_params["temperature"] = args.temperature
            if args.top_p is not None: effective_params["top_p"] = args.top_p
            if args.top_k is not None: effective_params["top_k"] = args.top_k
            if args.max_tokens is not None: effective_params["max_tokens"] = args.max_tokens
            if args.repetition_penalty is not None: effective_params["repetition_penalty"] = args.repetition_penalty
            effective_do_sample = args.do_sample # Use the value from BooleanOptionalAction

            print(f"  Final Effective Params: {effective_params}")
            print(f"  Do Sample: {effective_do_sample}")

            # Loop through shot settings
            for shot_setting_str in args.shot_settings:
                use_few_shot_to_run = (shot_setting_str == 'few_shot')
                print(f"\n    Starting evaluation: Shot='{shot_setting_str}'")

                # Call the refactored run_experiment function
                summary_result = run_experiment(
                    model_name=model_name_str,
                    tokenizer=tokenizer,
                    model=model,
                    samples_df=current_samples_df,
            lang_code=lang_code,
                    use_few_shot=use_few_shot_to_run,
                    base_results_path=args.base_output_dir,
                    # Pass all effective parameters
                    temperature=effective_params["temperature"],
                    top_p=effective_params["top_p"],
                    top_k=effective_params["top_k"],
                    max_tokens=effective_params["max_tokens"],
                    repetition_penalty=effective_params["repetition_penalty"],
                    do_sample=effective_do_sample
                )

                if summary_result is not None:
                    all_experiment_summaries.append(summary_result)
                else:
                    print(f"    WARNING: Experiment returned None for {lang_code}, {shot_setting_str}.")
            # End shot_setting loop
        # End lang_code loop

        # Clean up model and tokenizer after processing all languages for it
        if model_initialized:
            print(f"\n====== Finished experiments for model {model_name_str}. Unloading... ======")
            del model
            del tokenizer
            model, tokenizer = None, None # Prevent potential use after del
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("GPU memory cache cleared.")
        else:
            print(f"Model {model_name_str} was not initialized, no cleanup needed.")
    # End model loop

    # --- Aggregate and Save Overall Summary ---
    if all_experiment_summaries:
        print("\n--- Aggregating Overall Summary --- ")
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        overall_summary_path = os.path.join(summaries_output_dir, f'ner_baseline_ALL_experiments_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        try:
            overall_summary_df.to_csv(overall_summary_path, index=False, float_format='%.4f')
            logging.info(f"\n=== Overall Baseline NER Summary Saved ===")
            logging.info(f"Path: {overall_summary_path}")
            print(overall_summary_df.to_string()) # Print to console as well
        except Exception as e_save:
            logging.error(f"ERROR saving overall summary to {overall_summary_path}: {e_save}")

        # Optional: Add plotting here based on overall_summary_df
        try:
            plot_ner_f1_scores(overall_summary_df, plots_output_dir) # Call plotting function
            print(f"Plots saved to: {plots_output_dir}")
        except ImportError:
             print("Plotting skipped: matplotlib or seaborn not found.")
        except Exception as e_plot:
            print(f"Error generating plots: {e_plot}")

    else:
        logging.warning("No successful experiments were completed. No overall summary generated.")

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