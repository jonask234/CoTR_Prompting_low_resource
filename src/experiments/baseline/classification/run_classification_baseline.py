import sys
import os
import argparse
import time
import pandas as pd
import numpy as np
import torch
import json # Added for parameter logging if needed
import csv # Add csv import
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add project root to Python path BEFORE other project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import baseline classification functions
from src.experiments.baseline.classification.classification_baseline import (
    initialize_model,
    evaluate_classification_baseline,
    POSSIBLE_LABELS_EN # For metrics and prompt generation
)

# Import data loader (e.g., for a news classification dataset like MasakhaNEWS)
# Corrected import to use load_masakhanews
from src.utils.data_loaders.load_masakhanews import load_masakhanews_samples 

# Import metrics calculator
from evaluation.classification_metrics import calculate_classification_metrics

# Import HuggingFace utilities
from huggingface_hub import login
from config import get_token

# Define the language codes available in MasakhaNEWS
MASAKHANEWS_LANG_CODES_WITH_ALL = ["en", "sw", "ha", "yo", "pcm", "ibo", "amh", "fra", "lin", "lug", "orm", "run", "sna", "som", "tir", "xho"]

# --- Unified Generation Parameters ---
# Core parameters shared between baseline and CoTR
UNIFIED_GENERATION_PARAMETERS_CORE = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.1,
    "do_sample": True
}

# Task-specific max_tokens for Baseline Classification
MAX_TOKENS_BASELINE_CLASSIFICATION_LABEL = 20

# Model-specific overrides for Baseline
# These will override the UNIFIED_GENERATION_PARAMETERS_CORE or task-specific max_tokens
MODEL_SPECIFIC_OVERRIDES_BASELINE = {
    "CohereLabs/aya-expanse-8b": {
        "temperature": 0.05, # Stricter for Aya
        "repetition_penalty": 1.15,
        "max_tokens": 22, # Slight adjust for Aya labels (overrides MAX_TOKENS_BASELINE_CLASSIFICATION_LABEL)
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "temperature": 0.15,
        "top_p": 0.85,
        "max_tokens": 25, # Qwen might need more for labels
    }
}

# Define language names dictionary
LANG_NAMES = {
    "en": "English",
    "sw": "Swahili",
    "ha": "Hausa",
    "te": "Telugu"
}

def get_effective_baseline_params(lang_code: str, model_name: str, cli_overrides: Dict = None) -> Dict:
    """Gets effective parameters for baseline classification.
    Starts with unified core, adds baseline-specific max_tokens, then model-specific, then CLI.
    """
    params = UNIFIED_GENERATION_PARAMETERS_CORE.copy()
    params["max_tokens"] = MAX_TOKENS_BASELINE_CLASSIFICATION_LABEL

    # Apply model-specific overrides from MODEL_SPECIFIC_OVERRIDES_BASELINE
    if model_name in MODEL_SPECIFIC_OVERRIDES_BASELINE:
        model_overrides = MODEL_SPECIFIC_OVERRIDES_BASELINE[model_name]
        for key, value in model_overrides.items():
            params[key] = value # This can override core params or max_tokens

    # Override with any CLI parameters
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None and k in params:
                params[k] = v
            elif v is not None: # If it's a new param from CLI not in defaults
                logging.warning(f"CLI param '{k}' is not a standard generation param for baseline. It will be added.")
                params[k] = v
    
    # Ensure 'do_sample' is explicitly present, respecting overrides.
    params["do_sample"] = params.get("do_sample", True)

    logging.debug(f"Effective baseline params for {model_name} ({lang_code}): {params}")
    return params

def run_baseline_classification_experiment(
    model_name: str,
    tokenizer: Any, 
    model: Any, 
    samples_df: pd.DataFrame,
    lang_code: str,
    possible_labels_en: List[str],
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
    # Determine prompt language string based directly on lang_code
    prompt_lang_str = "lrl" if lang_code != 'en' else "en"

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
                    0: 'business', 1: 'entertainment', 2: 'health', 3: 'politics',
                    4: 'religion', 5: 'sports', 6: 'technology'
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
                model_name=model_name,
                tokenizer=tokenizer,
                model=model,
                samples_df=samples_df,
                lang_code=lang_code,
                possible_labels=possible_labels_en,
                use_few_shot=use_few_shot,
                temperature=generation_params["temperature"],
                top_p=generation_params["top_p"],
                top_k=generation_params["top_k"],
                max_tokens=generation_params["max_tokens"],
                repetition_penalty=generation_params["repetition_penalty"],
                do_sample=generation_params["do_sample"]
            )
            runtime = time.time() - start_time
            if not results_df_computed.empty:
                results_df_computed['runtime_seconds_total'] = runtime
                results_df_computed['runtime_per_sample'] = runtime / len(results_df_computed)
                results_df_computed.to_csv(results_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
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
    # Ensure correct float formatting and NaN representation for individual summary CSVs
    summary_df.to_csv(summary_file, index=False, float_format='%.4f', na_rep='NaN') 
    print(f"Summary saved to {summary_file}")
    print(summary_df.to_string())
    return summary_data

def parse_args():
    parser = argparse.ArgumentParser(description="Run Baseline Classification Experiments with MasakhaNEWS.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated model names.")
    parser.add_argument("--langs", type=str, default="en,ha,sw", help="Comma-separated MasakhaNEWS language codes.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per language.")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split.")
    parser.add_argument("--prompt_in_lrl", action='store_true', help="If set, prompt instructions and examples will be in LRL, not English.")
    
    # Control flow and experiment settings
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Prompting strategies to evaluate: zero-shot or few-shot.")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/classification/baseline_masakhanews", 
                        help="Base directory to save results and summaries.")
    parser.add_argument("--overwrite_results", action='store_true', help="Overwrite existing detailed results and summary files if they exist.")
    parser.add_argument("--test_mode", action='store_true', help="Run in test mode (uses a very small subset of data and fewer iterations).")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token for gated models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and reproducibility.")

    # Sampling parameters
    parser.add_argument("--sample_percentage", type=float, default=1.0, 
                        help="Percentage of initially loaded samples (e.g., from num_samples or all if num_samples is None) to actually use. E.g., 0.1 for 10%%. Default: 1.0 (all loaded).")
    parser.add_argument("--max_samples_per_lang", type=int, default=None, 
                        help="Absolute maximum number of samples per language to process. Acts as a cap *after* initial loading and percentage sampling. Default: No cap.")

    # Generation parameters (with defaults that can be overridden by get_effective_baseline_params)
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature for generation.")
    parser.add_argument("--top_p", type=float, default=None, help="Override top-p for generation.")
    parser.add_argument("--top_k", type=int, default=None, help="Override top-k for generation.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Override max_tokens for classification label generation (e.g., 10-20).")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Override repetition penalty.")
    parser.add_argument("--do_sample", type=lambda x: (str(x).lower() == 'true'), default=None, 
                        help="Override do_sample (True/False). Default determined by temperature > 0.")

    return parser.parse_args()

def main():
    args = parse_args()

    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' # Added name for clarity
    logging.basicConfig(level=args.log_level.upper(), format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__) # Initialize logger for this main script

    token = get_token()
    login(token=token)
    
    model_list = [m.strip() for m in args.models.split(',')]
    lang_list = [l.strip() for l in args.langs.split(',')]

    os.makedirs(args.base_output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.base_output_dir, "summaries"), exist_ok=True)

    possible_labels_en = POSSIBLE_LABELS_EN
    print(f"Using English labels for classification: {possible_labels_en}")

    all_experiment_summaries = []
    # Use a fixed seed for the 10% sampling for reproducibility across runs if args.seed is not defined
    # args.seed is already defined with a default of 42 by the parser
    sampling_seed = args.seed 

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
            # Step 1: Load samples, potentially capped by max_samples_per_lang
            # Pass args.num_samples (which is the renamed max_samples_per_lang in this context based on previous edits)
            # or more directly args.max_samples_per_lang if that argument name is settled.
            # Assuming args.max_samples_per_lang is the correct argument name now.
            loaded_df = load_masakhanews_samples(
                lang_code=lang_code,
                split=args.data_split,
                num_samples=args.max_samples_per_lang # Load up to this many, or all if None
            )

            samples_df_for_lang = pd.DataFrame() # Initialize as empty
            if not loaded_df.empty:
                # Step 2: Apply percentage sampling to the loaded_df
                num_total_loaded = len(loaded_df)
                if args.sample_percentage < 1.0 and args.sample_percentage > 0.0: # Ensure percentage is valid
                    num_to_sample_percent = max(1, int(args.sample_percentage * num_total_loaded))
                    logger.info(f"    Loaded {num_total_loaded} samples. Applying {args.sample_percentage*100:.1f}% sampling: {num_to_sample_percent} samples.")
                    samples_df_for_lang = loaded_df.sample(n=num_to_sample_percent, random_state=sampling_seed)
                else:
                    logger.info(f"    Loaded {num_total_loaded} samples. Using all (sample_percentage is {args.sample_percentage}).")
                    samples_df_for_lang = loaded_df # Use all loaded samples
            
            # Step 3: Apply the --num_samples limit as the final cap
            if not samples_df_for_lang.empty:
                if args.num_samples is not None and args.num_samples > 0 and len(samples_df_for_lang) > args.num_samples:
                    logger.info(f"    Currently have {len(samples_df_for_lang)} samples. Capping to --num_samples: {args.num_samples}.")
                    # Ensure consistent sampling if reducing, even if already sampled by percentage
                    samples_df_for_lang = samples_df_for_lang.sample(n=args.num_samples, random_state=sampling_seed).reset_index(drop=True)
                elif args.num_samples is not None and args.num_samples > 0:
                    logger.info(f"    Currently have {len(samples_df_for_lang)} samples. --num_samples is {args.num_samples}, no capping needed or already met.")
            
            if samples_df_for_lang.empty:
                print(f"Skipping {lang_code} for {model_name} due to missing data/columns ('text', 'label') after loading/sampling.")
                continue
            samples_df_for_lang['label'] = samples_df_for_lang['label'].astype(str).str.lower().str.strip()

            # Determine prompt_lang_str for logging/path purposes, reflecting the automatic choice
            actual_prompt_lang_description = 'LRL' if lang_code != 'en' else 'EN'

            for shot_setting in args.shot_settings:
                use_few_shot = (shot_setting == 'few_shot')
                effective_gen_params = get_effective_baseline_params(lang_code, model_name)
            
                print(f"\n  Running config: Lang={lang_code}, PromptInstr={actual_prompt_lang_description}, Shot={shot_setting}")
                print(f"    Effective Gen Params: {json.dumps(effective_gen_params, indent=2)}")

                summary_data = run_baseline_classification_experiment(
                    model_name, tokenizer, model, samples_df_for_lang, lang_code, 
                    possible_labels_en,
                    use_few_shot,
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
        
        # Define summaries_dir before using it
        summaries_dir = os.path.join(args.base_output_dir, "summaries")
        os.makedirs(summaries_dir, exist_ok=True)

        # Explicitly replace problematic strings like '.4f' with np.nan
        # This helps ensure that subsequent to_numeric conversions work as expected.
        for col in overall_summary_df.columns:
            if overall_summary_df[col].dtype == 'object':
                # Replace the exact string '.4f' if it exists.
                overall_summary_df[col] = overall_summary_df[col].replace('.4f', np.nan)
        
        # Ensure correct dtypes, especially for numeric columns that might be object
        numeric_cols = ['accuracy', 'macro_f1', 'runtime_total_s', 'temperature', 'top_p', 'top_k', 'max_tokens', 'repetition_penalty', 'samples_processed'] # ensure samples_processed is numeric
        
        # Dynamically add per-class metric columns to numeric_cols if they exist
        if not overall_summary_df.empty:
            # Use the keys from the first dictionary in all_experiment_summaries as a proxy for all possible columns
            # This is safer if overall_summary_df might have columns not in the first dict due to varying outputs
            potential_metric_keys = set()
            for summary_dict in all_experiment_summaries:
                potential_metric_keys.update(summary_dict.keys())

            for key in potential_metric_keys:
                if (key.endswith("_precision") or key.endswith("_recall") or key.endswith("_f1-score")) and key not in numeric_cols:
                    numeric_cols.append(key)
        
        for col in numeric_cols:
            if col in overall_summary_df.columns:
                # Convert to numeric, coercing errors. Strings like ".4f" (now np.nan) or other non-numeric strings will become NaN.
                overall_summary_df[col] = pd.to_numeric(overall_summary_df[col], errors='coerce')
        
        # Convert 'do_sample' to boolean if it's not already
        if 'do_sample' in overall_summary_df.columns:
            # Handle potential string "True"/"False" from dict conversion or np.nan
            overall_summary_df['do_sample'] = overall_summary_df['do_sample'].apply(
                lambda x: str(x).lower() == 'true' if pd.notna(x) else None
            ).astype('boolean') # Use pandas nullable boolean type

        overall_summary_filename = os.path.join(summaries_dir, "baseline_classification_ALL_experiments_summary.csv")
        
        # Save the cleaned DataFrame to CSV
        # Using float_format for actual floats, and na_rep for NaN values (which includes coerced '.4f' strings)
        overall_summary_df.to_csv(overall_summary_filename, index=False, float_format='%.4f', na_rep='NaN')
        
        logging.info(f"Overall summary saved to {overall_summary_filename}")
        logging.info("\n" + overall_summary_df.to_string())
    else:
        print("No classification baseline experiments were successfully summarized.")

    print("\n====== Classification Baseline Script Finished ======")

if __name__ == "__main__":
    # Define POSSIBLE_LABELS_EN globally for access in main and other functions if needed
    # This should be the source of truth for MasakhaNEWS categories.
    POSSIBLE_LABELS_EN = ['business', 'entertainment', 'health', 'politics', 'religion', 'sports', 'technology']
    print(f"Using English labels for classification: {POSSIBLE_LABELS_EN}")
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main() 