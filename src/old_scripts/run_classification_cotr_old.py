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
import logging
from typing import Dict, Any, Optional, List

# Add project root to Python path BEFORE other project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import CoTR classification functions
from src.experiments.cotr.classification.classification_cotr import (
    initialize_model,
    evaluate_classification_cotr_multi_prompt,
    evaluate_classification_cotr_single_prompt,
    CLASS_LABELS_ENGLISH # Import the English labels
)

# Import data loader for MasakhaNEWS
from src.utils.data_loaders.load_masakhanews import load_masakhanews_samples 

# Import metrics calculator
from evaluation.classification_metrics import calculate_classification_metrics

# Import HuggingFace utilities
from huggingface_hub import login
from config import get_token

# --- Define Default Parameters (align with baseline where applicable) ---
# These are general defaults; language/model specific might override them.
DEFAULT_GENERATION_PARAMS = {
    "text_translation": {"temperature": 0.4, "top_p": 0.9, "max_new_tokens": 512, "repetition_penalty": 1.0, "top_k": 40},
    "english_classification": {"temperature": 0.05, "top_p": 0.9, "max_new_tokens": 20, "repetition_penalty": 1.1, "top_k": 40},
    "label_translation": {"temperature": 0.4, "top_p": 0.9, "max_new_tokens": 75, "repetition_penalty": 1.0, "top_k": 40},
    "single_prompt_cotr": {"temperature": 0.05, "top_p": 0.9, "max_new_tokens": 200, "repetition_penalty": 1.1, "top_k": 40}
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
    "te": {
        "english_classification": {"temperature": 0.05, "repetition_penalty": 1.2},
        "single_prompt_cotr": {"temperature": 0.05, "repetition_penalty": 1.2}
    }
}

# Placeholder for model-specific parameter adjustments
MODEL_SPECIFIC_ADJUSTMENTS = {
    "CohereLabs/aya-expanse-8b": {
        "english_classification": {"temperature_factor": 1.0},
        "single_prompt_cotr": {"temperature_factor": 1.0}
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "english_classification": {"top_p": 0.85, "top_k": 35, "temperature": 0.05},
        "single_prompt_cotr": {"top_p": 0.85, "top_k": 35, "temperature": 0.05}
    }
}

def get_effective_params(base_params_config_key: str, lang_code: str, model_name: str, cli_overrides: Dict) -> Dict:
    """
    Merge default, language-specific, model-specific, and CLI override parameters
    for a specific step of the CoTR pipeline (e.g., "english_classification").
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
        # Allow CLI to override max_new_tokens even if key is just max_tokens
        if cli_key == "max_tokens" and cli_value is not None and "max_new_tokens" in effective_params:
             if base_params_config_key == "english_classification": # only for classification step
                effective_params["max_new_tokens"] = cli_value

    return effective_params

def run_classification_experiment(
    model_name: str,
    tokenizer: Any, 
    model: Any, 
    samples_df: pd.DataFrame,
    lang_code: str,
    possible_labels_en: List[str],
    pipeline_type: str, 
    use_few_shot: bool, 
    base_results_path: str,
    generation_args: Dict, # Contains all specific generation params for each step
    overwrite_results: bool = False # Added overwrite_results argument
) -> Optional[Dict[str, Any]]:
    """Run a single classification experiment configuration."""
    if samples_df.empty:
        print(f"No samples for {lang_code}, skipping experiment.")
        return None

    model_short_name = model_name.split('/')[-1]
    shot_type_str = "fs" if use_few_shot else "zs"
    pipeline_short = "mp" if pipeline_type == "multi_prompt" else "sp"

    results_dir = os.path.join(base_results_path, pipeline_short, shot_type_str, lang_code, model_short_name)
    summaries_dir = os.path.join(base_results_path, "summaries", pipeline_short, shot_type_str)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"results_cotr_classification_{lang_code}.csv")
    summary_file = os.path.join(summaries_dir, f"summary_cotr_classification_{lang_code}_{model_short_name}.csv")

    results_df = pd.DataFrame() # Initialize results_df

    if os.path.exists(results_file) and not overwrite_results: # Use passed argument
        print(f"Results file {results_file} already exists. Skipping computation, attempting to load for summary.")
        try:
            results_df = pd.read_csv(results_file)
        except Exception as e:
            print(f"Could not load existing results {results_file}: {e}. Will recompute if summary also missing or force overwrite is enabled.")
            if not os.path.exists(summary_file): # If summary is also missing, recompute
                results_df = pd.DataFrame() # Mark for recomputation
            else: # Summary exists, try to load it
                try:
                    existing_summary_df = pd.read_csv(summary_file)
                    return existing_summary_df.to_dict('records')[0]
                except Exception as es:
                    print(f"Could not load existing summary {summary_file}: {es}. Recomputing.")
                    results_df = pd.DataFrame() # Mark for recomputation
    else: # This covers if results_file does not exist OR if overwrite_results is True
        results_df = pd.DataFrame() # Ensure it's empty to trigger computation if overwrite

    if results_df.empty: # Needs computation (either not exists, failed to load, or overwrite)
        print(f"Running CoTR Classification: {model_name} on {lang_code} (Pipeline: {pipeline_type}, Shot: {shot_type_str})")
        start_time = time.time()
        try:
            if pipeline_type == 'multi_prompt':
                results_df = evaluate_classification_cotr_multi_prompt(
                    model, tokenizer, samples_df, lang_code, possible_labels_en, use_few_shot,
                    text_translation_params=generation_args.get("text_translation", {}),
                    classification_params=generation_args.get("english_classification", {}),
                    label_translation_params=generation_args.get("label_translation", {})
                )
            else: # single_prompt
                results_df = evaluate_classification_cotr_single_prompt(
                    model, tokenizer, samples_df, lang_code, possible_labels_en, use_few_shot,
                    generation_params=generation_args.get("single_prompt_cotr", {})
                )
            runtime = time.time() - start_time
            if not results_df.empty:
                results_df['runtime_seconds_total'] = runtime
                results_df['runtime_per_sample'] = runtime / len(results_df)
                results_df.to_csv(results_file, index=False)
                print(f"Results saved to {results_file}")
            else:
                print("Evaluation returned empty DataFrame.")
                return None
        except Exception as e:
            logging.error(f"Error during CoTR classification for {lang_code}, {model_short_name}, {pipeline_type}, {shot_type_str}: {e}", exc_info=True)
            return None

    # Calculate metrics
    if results_df.empty:
        print("No results to calculate metrics from.")
        return None
        
    # Ensure 'final_predicted_label' is used for metrics against 'ground_truth_label' (which should be EN)
    # The 'final_predicted_label' from CoTR functions should already be mapped to English.
    metrics = calculate_classification_metrics(results_df) 

    summary_data = {
        'model': model_short_name,
            'language': lang_code,
        'pipeline': pipeline_type,
        'shot_type': shot_type_str,
        'accuracy': metrics.get('accuracy', 0.0),
        'macro_f1': metrics.get('macro_f1', 0.0),
        'samples_processed': len(results_df),
        'runtime_total_s': results_df['runtime_seconds_total'].iloc[0] if 'runtime_seconds_total' in results_df.columns and not results_df.empty else 0,
        # Add other generation params to summary for reference
    }
    if pipeline_type == 'multi_prompt':
        summary_data.update({f"text_trans_{k}": v for k,v in generation_args.get("text_translation", {}).items()})
        summary_data.update({f"en_class_{k}": v for k,v in generation_args.get("english_classification", {}).items()})
        summary_data.update({f"label_trans_{k}": v for k,v in generation_args.get("label_translation", {}).items()})
    else:
        summary_data.update({f"single_prompt_{k}": v for k,v in generation_args.get("single_prompt_cotr", {}).items()})

    for label in possible_labels_en:
        summary_data[f'{label}_precision'] = metrics.get(f'{label}_precision', 0.0)
        summary_data[f'{label}_recall'] = metrics.get(f'{label}_recall', 0.0)

    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(summary_file, index=False, float_format='%.4f')
    print(f"Summary saved to {summary_file}")
    print(summary_df.to_string())
    return summary_data

def main():
    parser = argparse.ArgumentParser(description="Run Classification CoTR experiments.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-expanse-8b,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated models")
    parser.add_argument("--langs", type=str, default="en,sw,ha", help="Comma-separated language codes")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples per language")
    parser.add_argument("--pipeline_types", nargs='+', default=['multi_prompt', 'single_prompt'], choices=['multi_prompt', 'single_prompt'], help="CoTR pipeline types")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Shot settings")
    parser.add_argument("--dataset_name", type=str, default="masakhanews", 
                        help="Name of the classification dataset to load (e.g., masakhanews, news_cat_sw, etc.)")
    parser.add_argument("--temperature", type=float, default=None, help="Global override for temperature if set")
    parser.add_argument("--top_p", type=float, default=None, help="Global override for top_p if set")
    parser.add_argument("--top_k", type=int, default=None, help="Global override for top_k if set")
    parser.add_argument("--max_tokens", type=int, default=None, help="Global override for max_new_tokens for classification step")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Global override for repetition_penalty if set")
    parser.add_argument("--max_translation_tokens", type=int, default=None, help="Override max_new_tokens for text translation step")
    parser.add_argument("--max_label_translation_tokens", type=int, default=None, help="Override max_new_tokens for label translation step")
    parser.add_argument("--max_single_prompt_tokens", type=int, default=None, help="Override max_new_tokens for single_prompt_cotr step")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/classification/cotr", help="Base output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite_results", action="store_true", help="Overwrite existing detailed results files.")
    args = parser.parse_args()

    # Setup HF Token
    token = get_token()
    login(token=token)

    model_list = [m.strip() for m in args.models.split(',')]
    lang_list = [l.strip() for l in args.langs.split(',')]

    # Ensure base output directory exists
    os.makedirs(args.base_output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.base_output_dir, "summaries"), exist_ok=True)

    # --- DEFINE YOUR ENGLISH LABELS HERE ---
    # This should come from your dataset's characteristics or be a predefined set
    # For XL-Sum Topic, it might be like: ["news", "sports", "business", "health", "technology", "entertainment"]
    # For this example, using the placeholder from classification_cotr.py
    possible_labels_en = CLASS_LABELS_ENGLISH 
    print(f"Using the following English labels for classification: {possible_labels_en}")

    all_experiment_summaries = []

    for model_name in model_list:
        print(f"\n{'='*20} Initializing Model: {model_name} {'='*20}")
        tokenizer_main, model_main = None, None # Initialize to None
        try:
            # Initialize model and tokenizer once per model string
            tokenizer_main, model_main = initialize_model(model_name) # CHANGED unpacking order
        except Exception as e_init:
            logging.error(f"Failed to initialize model {model_name}: {e_init}. Skipping this model.", exc_info=True)
            if model_main is not None: del model_main
            if tokenizer_main is not None: del tokenizer_main
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        for lang_code in lang_list:
            print(f"\n--- Processing Language: {lang_code} for model {model_name} ---")
            # Load data - ensure your data loader returns 'text' and 'label' (English ground truth)
            samples_df = load_masakhanews_samples(lang_code=lang_code, num_samples=args.samples)
            if samples_df.empty or 'text' not in samples_df.columns or 'label' not in samples_df.columns:
                print(f"Skipping {lang_code} for {model_name} due to missing data or columns ('text', 'label').")
                continue
            
            # Ensure ground truth labels are among the possible_labels_en (or map them)
            # For simplicity, we assume GT labels are already in English and match possible_labels_en
            # If not, preprocessing/mapping would be needed here.
            samples_df['label'] = samples_df['label'].astype(str).str.lower().str.strip()
            # Filter out samples with labels not in our defined set, if necessary
            # samples_df = samples_df[samples_df['label'].isin([l.lower() for l in possible_labels_en])]
            # if samples_df.empty:
            #     print(f"No samples remaining for {lang_code} after filtering for valid labels. Skipping.")
            #     continue

            for pipeline_type in args.pipeline_types:
                for shot_setting in args.shot_settings:
                    use_few_shot = (shot_setting == 'few_shot')
                    
                    # Create a structured generation_args dictionary
                    cli_gen_overrides = {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "max_tokens": args.max_tokens, # For english_classification max_new_tokens
                        "repetition_penalty": args.repetition_penalty
                        # Note: CLI overrides for translation steps (e.g., max_translation_tokens) are handled separately if needed
                    }

                    generation_args_for_run = {
                        "text_translation": get_effective_params("text_translation", lang_code, model_name, 
                                                                 {"max_new_tokens": args.max_translation_tokens} # Specific CLI for trans tokens
                                                                ),
                        "english_classification": get_effective_params("english_classification", lang_code, model_name, cli_gen_overrides),
                        "label_translation": get_effective_params("label_translation", lang_code, model_name, 
                                                                  {"max_new_tokens": args.max_label_translation_tokens} # Specific CLI for label trans tokens
                                                                 ),
                        "single_prompt_cotr": get_effective_params("single_prompt_cotr", lang_code, model_name, 
                                                                   {"max_new_tokens": args.max_single_prompt_tokens, **cli_gen_overrides} # single prompt uses general overrides + its own max
                                                                  )
                    }
                    
                    # Ensure max_tokens from CLI overrides the one for english_classification step
                    if args.max_tokens is not None:
                        generation_args_for_run["english_classification"]["max_new_tokens"] = args.max_tokens

                    print(f"\n  Running config: Lang={lang_code}, Pipeline={pipeline_type}, Shot={shot_setting}")
                    print(f"    Effective Gen Params: {json.dumps(generation_args_for_run, indent=2)}")

                    summary_data = run_classification_experiment(
                        model_name, tokenizer_main, model_main, samples_df, lang_code, 
                        possible_labels_en, pipeline_type, use_few_shot, 
                        args.base_output_dir, generation_args_for_run,
                        overwrite_results=args.overwrite_results # Pass here
                    )
                    if summary_data:
                        all_experiment_summaries.append(summary_data)
        
        # Clean up model after all its languages/configs are done
        print(f"Finished all experiments for model {model_name}. Unloading...")
        del model_main
        del tokenizer_main
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logging.info(f"GPU memory cache cleared for {model_name}.")

    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        summary_filename = f"cotr_classification_ALL_experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        overall_summary_path = os.path.join(args.base_output_dir, "summaries", summary_filename)
        overall_summary_df.to_csv(overall_summary_path, index=False, float_format='%.4f')
        print(f"\nOverall CoTR Classification Summary saved to: {overall_summary_path}")
        print(overall_summary_df.to_string())

        # Optional: Add plotting based on overall_summary_df
        # try:
        #     # plot_classification_metrics(overall_summary_df, os.path.join(args.base_output_dir, "plots"))
        # except Exception as e_plot:
        #     print(f"Error generating plots: {e_plot}")
    else:
        print("No CoTR classification summaries collected.")

    print("\n====== CoTR Classification Script Finished ======")

if __name__ == "__main__":
    main() 