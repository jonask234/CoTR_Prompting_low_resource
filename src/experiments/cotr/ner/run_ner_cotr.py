import sys
import os
import argparse
import time
import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import itertools
import logging
from typing import Any, Dict, Optional

# Add project root to Python path BEFORE other project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path: # Optional: prevent adding multiple times
    sys.path.insert(0, project_root)

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import utility functions
from src.experiments.cotr.ner.ner_cotr import (
    evaluate_ner_cotr,
    evaluate_ner_cotr_single_prompt,
    initialize_model,
    calculate_ner_metrics
)
from src.experiments.baseline.ner.ner_baseline import load_masakhaner_samples
from src.evaluation.cotr.translation_metrics import calculate_comet_score
from evaluation.cotr.qa_metrics_cotr import COMET_AVAILABLE
from huggingface_hub import login
from config import get_token
from src.experiments.cotr.language_information import get_language_information

# Set of standardized parameters that will be consistent across baseline and CoTR
# IMPORTANT: These must match the parameters in baseline/ner/run_ner_baseline.py
STANDARD_PARAMETERS = {
    "sw": {  # Swahili
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 200,
        "repetition_penalty": 1.1,
        "trans_temp": 0.3,
        "trans_top_p": 0.9,
        "trans_top_k": 40,
        "max_trans_tokens": 256,
        "trans_repetition_penalty": 1.0
    },
    "ha": {  # Hausa
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 200,
        "repetition_penalty": 1.1,
        "trans_temp": 0.3,
        "trans_top_p": 0.9,
        "trans_top_k": 40,
        "max_trans_tokens": 256,
        "trans_repetition_penalty": 1.0
    }
}

LANGUAGE_PARAMETERS = {
    "sw": {  # Swahili (Example - Adjust as needed)
        "temperature": 0.25,
        "top_p": 0.85,
        "top_k": 35,
        "max_tokens": 180,
        "repetition_penalty": 1.15,
        "trans_temp": 0.25,
        "trans_top_p": 0.85,
        "trans_top_k": 35,
        "max_trans_tokens": 256,
        "trans_repetition_penalty": 1.05
    },
    "ha": {  # Hausa (Example - Adjust as needed)
        "temperature": 0.25,
        "top_p": 0.85,
        "top_k": 35,
        "max_tokens": 180,
        "repetition_penalty": 1.15,
        "trans_temp": 0.25,
        "trans_top_p": 0.85,
        "trans_top_k": 35,
        "max_trans_tokens": 256,
        "trans_repetition_penalty": 1.05
    }
    # Add other languages if tuning is done
}

def run_experiment(
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    base_results_path: str,
    pipeline_type: str = 'multi_prompt',
    use_few_shot: bool = True,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 40,
    max_tokens: int = 200,
    repetition_penalty: float = 1.1,
    trans_temp: float = 0.35,
    trans_top_p: float = 0.9,
    trans_top_k: int = 40,
    max_trans_tokens: int = 256,
    trans_repetition_penalty: float = 1.0
) -> Optional[Dict[str, Any]]:
    """
    Run a specific NER CoTR experiment configuration.
    """
    # Apply language-specific parameter overrides
    if lang_code in LANGUAGE_PARAMETERS:
        lang_params = LANGUAGE_PARAMETERS[lang_code]
        temperature = lang_params.get("temperature", temperature)
        top_p = lang_params.get("top_p", top_p)
        top_k = lang_params.get("top_k", top_k)
        max_tokens = lang_params.get("max_tokens", max_tokens)
        repetition_penalty = lang_params.get("repetition_penalty", repetition_penalty)
        trans_temp = lang_params.get("trans_temp", trans_temp)
        trans_top_p = lang_params.get("trans_top_p", trans_top_p)
        trans_top_k = lang_params.get("trans_top_k", trans_top_k)
        max_trans_tokens = lang_params.get("max_trans_tokens", max_trans_tokens)
        trans_repetition_penalty = lang_params.get("trans_repetition_penalty", trans_repetition_penalty)
        print(f"Using language-specific parameters for {lang_code}")

    # Apply model-specific adjustments (on top of language/standard)
    # Ensure consistency with baseline adjustments
    ner_temp_final = temperature
    ner_top_p_final = top_p
    ner_top_k_final = top_k
    ner_rep_penalty_final = repetition_penalty
    trans_temp_final = trans_temp
    trans_top_p_final = trans_top_p
    trans_top_k_final = trans_top_k
    trans_rep_penalty_final = trans_repetition_penalty
    
    if "aya" in model_name.lower():
        ner_temp_final = max(0.1, ner_temp_final * 0.9) # Example adjustment
        trans_temp_final = max(0.1, trans_temp_final * 0.9) # Example adjustment
        print(f"Applied Aya adjustments: NER Temp={ner_temp_final}, Trans Temp={trans_temp_final}")
    elif "qwen" in model_name.lower():
        ner_top_p_final = max(0.7, ner_top_p_final * 0.9) # Example adjustment
        ner_top_k_final = 35 # Example adjustment
        trans_top_p_final = max(0.7, trans_top_p_final * 0.9) # Example adjustment
        trans_top_k_final = 35 # Example adjustment
        print(f"Applied Qwen adjustments: NER TopP={ner_top_p_final}, NER TopK={ner_top_k_final}, Trans TopP={trans_top_p_final}, Trans TopK={trans_top_k_final}")
    
    # Create directories if they don't exist
    results_dir = os.path.join(base_results_path, "results")
    summaries_dir = os.path.join(base_results_path, "summaries")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    # Define file paths
    model_short = model_name.split('/')[-1]
    shot_type_str = "fs" if use_few_shot else "zs"
    pipeline_short = "mp" if pipeline_type == "multi_prompt" else "sp"
    results_file = os.path.join(results_dir, f"results_cotr_{pipeline_short}_{shot_type_str}_ner_{lang_code}_{model_short}.csv")
    summary_file = os.path.join(summaries_dir, f"summary_cotr_{pipeline_short}_{shot_type_str}_ner_{lang_code}_{model_short}.csv")

    # Skip if results exist (optional, add --overwrite flag later if needed)
    if os.path.exists(results_file):
        print(f"Results file exists, skipping: {results_file}")
        # Try to load summary if results exist but summary doesn't
        if not os.path.exists(summary_file):
            try:
                print(f"Attempting to load existing results from {results_file} to generate summary...")
                results_df = pd.read_csv(results_file)
                # Recalculate metrics from loaded data
            except Exception as e:
                print(f"Could not load existing results to regenerate summary: {e}")
                return None # Cannot proceed
        else:
            # Load existing summary data if both exist
            try:
                existing_summary_df = pd.read_csv(summary_file)
                print(f"Loaded existing summary from {summary_file}")
                return existing_summary_df.to_dict('records')[0] # Return first row as dict
            except Exception as e:
                print(f"Could not load existing summary file {summary_file}: {e}")
                return None
    else:
        # Run the experiment
        print(f"Running experiment: {model_name} on {lang_code} ({pipeline_type}, {shot_type_str})")
        try:
            if pipeline_type == 'multi_prompt':
        results_df = evaluate_ner_cotr(
            model_name=model_name,
                    tokenizer=tokenizer,
                    model=model,
            samples_df=samples_df,
            lang_code=lang_code,
            use_few_shot=use_few_shot,
                    temperature=ner_temp_final,
                    top_p=ner_top_p_final,
                    top_k=ner_top_k_final,
            max_tokens=max_tokens,
                    max_translation_tokens=max_trans_tokens
                )
            else: # single_prompt
                # Single prompt uses NER params for the overall generation control
                results_df = evaluate_ner_cotr_single_prompt(
                    model_name=model_name,
                    tokenizer=tokenizer,
                    model=model,
                    samples_df=samples_df,
                    lang_code=lang_code,
                    use_few_shot=use_few_shot,
                    temperature=ner_temp_final,
                    top_p=ner_top_p_final,
                    top_k=ner_top_k_final,
                    max_tokens=max_tokens
        )
        
        if results_df.empty:
                print("ERROR: Evaluation returned empty DataFrame.")
            return None
        
            results_df.to_csv(results_file, index=False)
            print(f"Results saved to {results_file}")

        except Exception as e:
            logging.error(f"Error during NER CoTR evaluation for {lang_code}, {model_short}, {pipeline_type}, {shot_type_str}: {e}", exc_info=True)
            return None # Indicate failure
        
    # Calculate metrics from results_df (whether loaded or newly generated)
    print(f"Calculating metrics for {results_file}...")
    precisions, recalls, f1s = [], [], []
    comet_scores = []
    successful_samples = 0

    # Convert string representation of lists/tuples back if loaded from CSV
    # This is crucial if loading existing results
    needs_conversion = False
    # Check the type of the first non-null element using the correct column name
    first_gold = results_df['ground_truth_entities'].dropna().iloc[0] if not results_df['ground_truth_entities'].dropna().empty else None
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
            # Apply conversion if needed, using the correct column name
            gold = safe_literal_eval(row['ground_truth_entities']) if needs_conversion else row['ground_truth_entities']
            pred = safe_literal_eval(row['predicted_entities']) if needs_conversion else row['predicted_entities']

            # Ensure data types are list before passing to metrics
            if not isinstance(gold, list): gold = []
            if not isinstance(pred, list): pred = []
            
            # Use the metrics function from ner_cotr.py (or baseline if identical)
            metrics = calculate_ner_metrics(gold, pred) # Ensure this function is defined/imported correctly
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1s.append(metrics.get('f1_score', 0.0)) # Use .get() for safety
            successful_samples += 1
            
            # Collect COMET scores if available (only for multi-prompt)
            if pipeline_type == 'multi_prompt' and 'comet_score_en_lrl_entities' in row and pd.notna(row['comet_score_en_lrl_entities']):
                comet_scores.append(row['comet_score_en_lrl_entities'])

        except Exception as e_metric:
            print(f"Error calculating metrics for row {idx}: {e_metric}")
            # Append default values on error to avoid crashing
            precisions.append(0.0); recalls.append(0.0); f1s.append(0.0)
    
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    avg_f1 = np.mean(f1s) if f1s else 0.0
    avg_comet = np.mean(comet_scores) if comet_scores else None # Use None if no scores

    # Prepare summary data
    summary_data = {
        'model': model_short,
        'language': lang_code,
        'pipeline': pipeline_type,
        'shot_type': shot_type_str,
        'precision': float(avg_precision),
        'recall': float(avg_recall),
        'f1': float(avg_f1),
        'comet_score_entities': avg_comet,
        'num_samples': len(samples_df),
        'num_successful': successful_samples,
        # Store final effective parameters
        'ner_temp': ner_temp_final,
        'ner_top_p': ner_top_p_final,
        'ner_top_k': ner_top_k_final,
        'ner_rep_penalty': ner_rep_penalty_final,
        'trans_temp': trans_temp_final if pipeline_type == 'multi_prompt' else None,
        'trans_top_p': trans_top_p_final if pipeline_type == 'multi_prompt' else None,
        'trans_top_k': trans_top_k_final if pipeline_type == 'multi_prompt' else None,
        'trans_rep_penalty': trans_rep_penalty_final if pipeline_type == 'multi_prompt' else None,
        'max_tokens': max_tokens,
        'max_trans_tokens': max_trans_tokens if pipeline_type == 'multi_prompt' else None
    }

    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(summary_file, index=False, float_format='%.4f')
    print(f"Summary metrics saved to {summary_file}")
    print(summary_df.to_string())
        
    return summary_data # Return dict for overall aggregation

def run_grid_search(args, models, lang_codes, base_results_path, masakhaner_samples, pipeline_type_for_grid, use_few_shot_for_grid):
    """
    Run a grid search over different parameter combinations.
    
    Args:
        args: Command-line arguments
        models: List of models to evaluate
        lang_codes: List of language codes
        base_results_path: Base path for saving results
        masakhaner_samples: Dictionary of DataFrames with samples
        pipeline_type_for_grid: Specific pipeline type for this grid search
        use_few_shot_for_grid: Specific shot setting for this grid search
    """
    # Define parameter grid - MAKE SURE THIS MATCHES the baseline grid search
    param_grid = {
        "temperature": [0.2, 0.3, 0.4],
        "top_p": [0.8, 0.9, 1.0],
        "top_k": [40, 60],
        "max_tokens": [150, 200, 250],
        "num_beams": [1, 2]
    }
    
    # Optional: Reduce grid size for faster searching
    if args.compact_grid:
        param_grid = {
            "temperature": [0.3],
            "top_p": [0.9],
            "top_k": [40],
            "max_tokens": [200],
            "num_beams": [1]
        }
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        param_grid["temperature"],
        param_grid["top_p"],
        param_grid["top_k"],
        param_grid["max_tokens"],
        param_grid["num_beams"]
    ))
    
    print(f"Running grid search with {len(param_combinations)} parameter combinations")
    
    # Track best parameters
    best_params = {}
    
    for lang_code in lang_codes:
        best_params[lang_code] = {
            "model": None,
            "params": None,
            "f1": -1  # Initialize with a low value
        }
    
    # Run experiments with each parameter combination
    for model_name in models:
        print(f"\n====== Starting experiments for model: {model_name} ======")
        model_initialized = False
        tokenizer, model = None, None
        try:
            print(f"  Attempting to initialize {model_name}...")
            tokenizer, model = initialize_model(model_name)
            model_initialized = True
            print(f"Model {model_name} initialized successfully.")
        except Exception as e:
            print(f"ERROR: Failed to initialize model {model_name}: {e}. Skipping this model for grid search.")
            if 'model' in locals() and model is not None: del model
            if 'tokenizer' in locals() and tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        for lang_code in lang_codes:
            if lang_code in masakhaner_samples and not masakhaner_samples[lang_code].empty:
                print(f"\n--- Running grid search for {model_name} on {lang_code} (Pipeline: {pipeline_type_for_grid}, Shot: {'Few-shot' if use_few_shot_for_grid else 'Zero-shot'}) ---")
                
                for temp, top_p, top_k, max_tokens, num_beams in tqdm(param_combinations, desc="Parameter combinations"):
                    result_df = run_experiment(
                        model_name=model_name,
                        tokenizer=tokenizer,
                        model=model,
                        samples_df=masakhaner_samples[lang_code],
                        lang_code=lang_code,
                        base_results_path=base_results_path,
                        pipeline_type=pipeline_type_for_grid,
                        use_few_shot=use_few_shot_for_grid,
                        temperature=temp,
                        top_p=top_p,
                        top_k=top_k,
                        max_tokens=max_tokens,
                        max_trans_tokens=max_tokens,
                        num_beams=num_beams
                    )
                    
                    if result_df is not None and not result_df.empty:
                        avg_f1 = result_df["f1"].mean()
                        
                        # Check if this is the best result so far for this language
                        if avg_f1 > best_params[lang_code]["f1"]:
                            best_params[lang_code] = {
                                "model": model_name,
                                "params": {
                                    "temperature": temp,
                                    "top_p": top_p,
                                    "top_k": top_k,
                                    "max_tokens": max_tokens,
                                    "num_beams": num_beams
                                },
                                "f1": avg_f1
                            }
                            
                            print(f"\nNew best parameters for {lang_code}: F1={avg_f1:.4f}")
                            print(f"  Model: {model_name}")
                            print(f"  Parameters: temp={temp}, top_p={top_p}, top_k={top_k}, max_tokens={max_tokens}, num_beams={num_beams}")
        
        print(f"Finished grid search for model {model_name}. Unloading...")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save best parameters
    best_params_path = os.path.join(base_results_path, "best_params")
    os.makedirs(best_params_path, exist_ok=True)
    
    with open(os.path.join(best_params_path, "best_params_cotr.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    
    print("\nBest parameters saved to best_params_cotr.json")
    
    # Print summary of best parameters
    print("\n--- Best parameters summary ---")
    for lang_code, params in best_params.items():
        print(f"\n{lang_code}:")
        print(f"  Model: {params['model']}")
        print(f"  F1: {params['f1']:.4f}")
        print(f"  Parameters: {params['params']}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run NER CoTR experiments")
    parser.add_argument("--model", type=str, default="CohereLabs/aya-expanse-8b,Qwen/Qwen2.5-7B-Instruct", help="Model to use, comma-separated if multiple models")
    parser.add_argument("--lang", type=str, default="sw,ha", help="Language code(s), comma-separated if multiple")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per language")
    parser.add_argument("--pipeline_types", nargs='+', default=['multi_prompt', 'single_prompt'], choices=['multi_prompt', 'single_prompt'], help="CoTR pipeline types to run")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Shot settings to evaluate (zero_shot, few_shot)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for generation (overrides standard if set)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p for generation (overrides standard if set)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k for generation (overrides standard if set)")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens for generation (overrides standard if set)")
    parser.add_argument("--max_translation_tokens", type=int, default=200, help="Max tokens for translation steps (used in multi_prompt)")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty (overrides standard if set)")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/ner/cotr", help="Base directory for all outputs.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (e.g., test, dev)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    args = parser.parse_args()
    
    # Determine effective generation parameters from args or defaults
    # These will be used by run_experiment if not overridden by language/model logic inside it
    current_temp = args.temperature
    current_top_p = args.top_p
    current_top_k = args.top_k
    current_max_tokens = args.max_tokens
    current_repetition_penalty = args.repetition_penalty
    
    # Get Hugging Face token
    token = get_token()
    login(token=token)
    
    # Define models to test
    if "," in args.model:
        models = args.model.split(",")
    else:
        models = [args.model]
    
    # Define language codes
    if "," in args.lang:
        lang_codes = args.lang.split(",")
    else:
        lang_codes = [args.lang]
    
    # Define base results path
    base_results_path = args.base_output_dir
    os.makedirs(base_results_path, exist_ok=True)
    os.makedirs(os.path.join(base_results_path, 'summaries'), exist_ok=True)
    
    # Load MasakhaNER samples - Load ONCE outside the model loop
    print("\n--- Loading MasakhaNER Data ---")
    masakhaner_samples = {}
    for lang_code in lang_codes:
        print(f"Loading data for {lang_code}...")
        samples_df_full = load_masakhaner_samples(
            lang_code=lang_code,
            split=args.split,
            num_samples=None, # Load all first
            seed=args.seed
        )
        if samples_df_full.empty:
            print(f"WARNING: No samples found for {lang_code} in split '{args.split}'. Skipping this language.")
            masakhaner_samples[lang_code] = pd.DataFrame()
        else:
            actual_loaded = len(samples_df_full)
            if args.samples is not None and actual_loaded > args.samples:
                masakhaner_samples[lang_code] = samples_df_full.sample(n=args.samples, random_state=args.seed)
            else:
                masakhaner_samples[lang_code] = samples_df_full
            print(f"Using {len(masakhaner_samples[lang_code])} samples for {lang_code}.")

            # --- Check for required columns after loading --- #
            required_cols = ['text', 'entities'] # Assuming 'text' is now created by loader if needed
            if not all(col in masakhaner_samples[lang_code].columns for col in required_cols):
                print(f"ERROR: Loaded data for {lang_code} is missing required columns ({required_cols}). Skipping this language.")
                masakhaner_samples[lang_code] = pd.DataFrame() # Mark as empty

    all_experiment_summaries = []

    # --- Main Experiment Loop Structure --- #
    for model_name_str in models:
        print(f"\n====== Starting experiments for model: {model_name_str} ======")
        model_initialized = False
        tokenizer, model = None, None
        try:
            print(f"  Attempting to initialize {model_name_str}...")
            tokenizer, model = initialize_model(model_name_str)
            model_initialized = True
            print(f"Model {model_name_str} initialized successfully.")
        except Exception as e_init:
            print(f"ERROR: Failed to initialize model {model_name_str}: {e_init}. Skipping this model.")
            continue # Skip to the next model if initialization fails

        model_language_results = [] # Store results for this specific model

            for lang_code in lang_codes:
            print(f"\n--- Processing Language: {lang_code} for model {model_name_str} ---")
            # Check if samples are valid for this language
            if lang_code not in masakhaner_samples or masakhaner_samples[lang_code].empty:
                print(f"  Skipping {lang_code} due to missing or invalid data.")
                continue # Skip to the next language

            current_samples_df = masakhaner_samples[lang_code]

            # Determine effective parameters for this lang, considering command-line overrides
            # Use STANDARD_PARAMETERS as the ultimate fallback if lang not in LANGUAGE_PARAMETERS
            default_lang_params = STANDARD_PARAMETERS.get(lang_code, STANDARD_PARAMETERS.get('sw', {})) # Default to sw if lang unknown
            lang_specific_params = LANGUAGE_PARAMETERS.get(lang_code, default_lang_params) # Get lang specific, or default

            # NER step parameters
            effective_temp = current_temp if current_temp is not None else lang_specific_params.get("temperature", default_lang_params.get("temperature"))
            effective_top_p = current_top_p if current_top_p is not None else lang_specific_params.get("top_p", default_lang_params.get("top_p"))
            effective_top_k = current_top_k if current_top_k is not None else lang_specific_params.get("top_k", default_lang_params.get("top_k"))
            effective_max_tokens = current_max_tokens if current_max_tokens is not None else lang_specific_params.get("max_tokens", default_lang_params.get("max_tokens"))
            effective_rep_penalty = current_repetition_penalty if current_repetition_penalty is not None else lang_specific_params.get("repetition_penalty", default_lang_params.get("repetition_penalty"))

            # Translation step parameters (relevant for multi-prompt)
            effective_trans_temp = args.temperature if args.temperature is not None else lang_specific_params.get("trans_temp", default_lang_params.get("trans_temp"))
            effective_trans_top_p = args.top_p if args.top_p is not None else lang_specific_params.get("trans_top_p", default_lang_params.get("trans_top_p"))
            effective_trans_top_k = args.top_k if args.top_k is not None else lang_specific_params.get("trans_top_k", default_lang_params.get("trans_top_k"))
            effective_max_trans_tokens = args.max_translation_tokens # Always use arg for trans max tokens
            effective_trans_rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else lang_specific_params.get("trans_repetition_penalty", default_lang_params.get("trans_repetition_penalty"))

            print(f"  Base Params: Temp={effective_temp}, TopP={effective_top_p}, TopK={effective_top_k}, MaxTok={effective_max_tokens}, RepPen={effective_rep_penalty}")
            print(f"  Trans Params: Temp={effective_trans_temp}, TopP={effective_trans_top_p}, TopK={effective_trans_top_k}, MaxTok={effective_max_trans_tokens}, RepPen={effective_trans_rep_penalty}")

            # Loop through pipeline types and shot settings
            for pipeline_type_to_run in args.pipeline_types:
                for shot_setting_str in args.shot_settings:
                    use_few_shot_to_run = (shot_setting_str == 'few_shot')
                    print(f"\n    Attempting to run: Model='{model_name_str}', Lang='{lang_code}', Pipeline='{pipeline_type_to_run}', Shot='{shot_setting_str}'")
                    print(f"      Effective NER Params: temp={effective_temp:.2f}, top_p={effective_top_p:.2f}, top_k={effective_top_k}, max_tokens={effective_max_tokens}, rep_penalty={effective_rep_penalty:.2f}")
                    if pipeline_type_to_run == 'multi_prompt':
                        print(f"      Effective Translation Params: temp={effective_trans_temp:.2f}, top_p={effective_trans_top_p:.2f}, top_k={effective_trans_top_k}, max_tokens={effective_max_trans_tokens}, rep_penalty={effective_trans_rep_penalty:.2f}")
                    
                    # Call run_experiment with all determined parameters
                    summary_result = run_experiment(
                        model_name=model_name_str,
                        tokenizer=tokenizer,
                        model=model,
                        samples_df=current_samples_df,
                        lang_code=lang_code,
                        base_results_path=base_results_path,
                        pipeline_type=pipeline_type_to_run,
                        use_few_shot=use_few_shot_to_run,
                        # NER Params
                        temperature=effective_temp,
                        top_p=effective_top_p,
                        top_k=effective_top_k,
                        max_tokens=effective_max_tokens,
                        repetition_penalty=effective_rep_penalty,
                        # Translation Params
                        trans_temp=effective_trans_temp,
                        trans_top_p=effective_trans_top_p,
                        trans_top_k=effective_trans_top_k,
                        max_trans_tokens=effective_max_trans_tokens,
                        trans_repetition_penalty=effective_trans_rep_penalty
                    )

                    if summary_result is not None:
                        print(f"    SUCCESS: Experiment completed for {lang_code}, {pipeline_type_to_run}, {shot_setting_str}.")
                        model_language_results.append(summary_result)
                        all_experiment_summaries.append(summary_result)
                    else:
                        print(f"    FAILURE/SKIP: Experiment returned None for {lang_code}, {pipeline_type_to_run}, {shot_setting_str}.")
                # End shot_setting loop
            # End pipeline_type loop
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

    # --- Aggregate and Save Overall Summary --- #
    if all_experiment_summaries:
        print("\n--- Aggregating Overall Summary --- ")
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        overall_summary_path = os.path.join(base_results_path, 'summaries', f'ner_cotr_ALL_experiments_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        try:
            overall_summary_df.to_csv(overall_summary_path, index=False, float_format='%.4f')
            print(f"\n=== Overall CoTR NER Summary Saved ===")
            print(f"Path: {overall_summary_path}")
            print(overall_summary_df.to_string())
        except Exception as e_save:
            print(f"ERROR saving overall summary to {overall_summary_path}: {e_save}")

        # Optional: Add plotting here based on overall_summary_df
        # try:
        #     plot_ner_metrics(overall_summary_df, os.path.join(base_results_path, 'plots'))
        # except Exception as e_plot:
        #     print(f"Error generating plots: {e_plot}")

                else:
        print("\nNo successful experiments were completed. No overall summary generated.")

    print("\n====== NER CoTR Script Finished ======")

if __name__ == "__main__":
    main() 