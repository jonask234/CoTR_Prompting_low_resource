import sys
import os
import argparse
import time
import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional, List

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from huggingface_hub import login
from config import get_token

# Functions from the new baseline script
from src.experiments.baseline.sentiment.sentiment_baseline_new import (
    initialize_model,
    evaluate_sentiment_baseline,
    EXPECTED_LABELS as SENTIMENT_LABELS # Import EXPECTED_LABELS and alias it
    # process_sentiment_baseline, # Not directly called by runner, but by evaluate_sentiment_baseline
    # extract_label,
    # generate_sentiment_prompt,
    # generate_lrl_instruct_sentiment_prompt
)
from src.utils.data_loaders.load_afrisenti import load_afrisenti_samples
from evaluation.sentiment_metrics import calculate_sentiment_metrics # Keep this for the function

# --- Standardized Parameters ---
STANDARD_PARAMETERS = {
    "temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_tokens": 30, "repetition_penalty": 1.1
}
LANGUAGE_PARAMETERS = {
    "sw": {"temperature": 0.25, "max_tokens": 15, "repetition_penalty": 1.15},
    "ha": {"temperature": 0.25, "max_tokens": 15, "repetition_penalty": 1.15},
    # Add other LRLs with specific parameters if needed
}
# Model-specific overrides can be added here if necessary
MODEL_SPECIFIC_OVERRIDES = {
    "CohereLabs/aya-expanse-8b": {
        "temperature": 0.225, # Example override
        "repetition_penalty": 1.15,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "temperature": 0.250,
        "top_p": 0.81, # Example Qwen specific
        "top_k": 35
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run Sentiment Baseline experiments with AfriSenti.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-expanse-8b", 
                        help="Comma-separated model names (e.g., 'CohereLabs/aya-expanse-8b,Qwen/Qwen2.5-7B-Instruct').")
    parser.add_argument("--langs", nargs='+', default=['sw', 'ha'], 
                        choices=['am', 'dz', 'ha', 'ig', 'ma', 'pcm', 'pt', 'sw', 'yo', 'twi', 'kr', 'ti', 'en', 'ar', 'multi', 'haus', 'ibo', 'swa', 'yor', 'mar'], # Expanded choices for flexibility
                        help="Languages to evaluate. Default: sw, ha.")
    parser.add_argument("--prompt_instructions", nargs='+', default=['en', 'lrl'], choices=['en', 'lrl'], 
                        help="Instruction language for prompts: 'en' for English, 'lrl' for LRL-specific.")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], 
                        help="Shot settings to evaluate.")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/sentiment/baseline", 
                        help="Base directory to save results and summaries.")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "validation", "test"], 
                        help="Dataset split to use. Default: test.")
    
    # Sampling options
    parser.add_argument("--sample_percentage", type=float, default=10.0, 
                        help="Percentage of samples to use *per language* (0.0 to 1.0) from the chosen split and num_samples. Default: 10.0 (10%%).")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Maximum number of samples to load per language and split *before* applying sample_percentage. If None, all available are considered.")
    parser.add_argument("--balanced_sampling", action='store_true', 
                        help="Enable balanced sampling (equal numbers per class). `samples_per_class` will be used if set.")
    parser.add_argument("--samples_per_class", type=int, default=None, 
                        help="Number of samples per class for balanced sampling. Effective if --balanced_sampling is used.")

    # Generation parameters with CLI overrides
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature for generation.")
    parser.add_argument("--top_p", type=float, default=None, help="Override top_p for generation.")
    parser.add_argument("--top_k", type=int, default=None, help="Override top_k for generation.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Override max_tokens for sentiment label generation (should be small).")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Override repetition penalty.")
    parser.add_argument("--do_sample", type=lambda x: (str(x).lower() == 'true'), default=None, help="Override do_sample (True/False). Default determined by temperature.")
    
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for gated models.")
    parser.add_argument("--overwrite_results", action='store_true', help="Overwrite existing detailed results and summary files if they exist.")
    parser.add_argument("--test_mode", action='store_true', help="Run with a very small subset of data (e.g., 5 samples) for quick testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and reproducibility.")

    return parser.parse_args()

def get_effective_generation_params(lang_code: str, model_name: str, cli_args: argparse.Namespace) -> Dict[str, Any]:
    """Determines effective generation parameters incorporating defaults, language-specific, model-specific, and CLI overrides."""
    params = {**STANDARD_PARAMETERS} # Start with standard defaults

    if lang_code in LANGUAGE_PARAMETERS:
        params.update(LANGUAGE_PARAMETERS[lang_code]) # Apply language-specific overrides

    # Apply model-specific overrides (if any for this model name)
    # Extract short model name for matching keys in MODEL_SPECIFIC_OVERRIDES
    short_model_name_key = model_name.split('/')[-1] 
    if short_model_name_key in MODEL_SPECIFIC_OVERRIDES:
        params.update(MODEL_SPECIFIC_OVERRIDES[short_model_name_key])
    # Check for full model name key as well
    elif model_name in MODEL_SPECIFIC_OVERRIDES:
         params.update(MODEL_SPECIFIC_OVERRIDES[model_name])


    # Apply CLI overrides last, as they have the highest precedence
    if cli_args.temperature is not None: params["temperature"] = cli_args.temperature
    if cli_args.top_p is not None: params["top_p"] = cli_args.top_p
    if cli_args.top_k is not None: params["top_k"] = cli_args.top_k
    if cli_args.max_tokens is not None: params["max_tokens"] = cli_args.max_tokens
    if cli_args.repetition_penalty is not None: params["repetition_penalty"] = cli_args.repetition_penalty
    if cli_args.do_sample is not None: params["do_sample"] = cli_args.do_sample
    
    # Ensure do_sample is consistent with temperature
    params["do_sample"] = True if params.get("temperature", 0) > 0.01 else False

    return params

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # HF Login
    token = args.hf_token if args.hf_token else get_token()
    if token:
        try:
            login(token=token)
            logging.info("Successfully logged in to Hugging Face Hub.")
        except Exception as e:
            logging.error(f"Hugging Face login failed: {e}")
    else:
        logging.warning("HuggingFace token not provided. Access to gated models may be restricted.")

    models_list = [m.strip() for m in args.models.split(',')]
    all_experiment_summaries = []

    for model_name_str in models_list:
        logging.info(f"\n{'='*20} Initializing Model: {model_name_str} {'='*20}")
        tokenizer, model = None, None
        try:
            tokenizer, model = initialize_model(model_name_str)
            logging.info(f"Model {model_name_str} initialized successfully.")
        except Exception as e_init:
            logging.error(f"Failed to initialize model {model_name_str}: {e_init}. Skipping this model.", exc_info=True)
            if "qwen" in model_name_str.lower():
                logging.error(f"QWEN_DEBUG: Initialization failed for {model_name_str}. Error: {e_init}")
            if model: del model
            if tokenizer: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        for lang_code in args.langs:
            logging.info(f"\n--- Processing Language: {lang_code} for model {model_name_str} ---")
            
            # Load all samples for the split first to determine dataset size
            # The load_afrisenti_samples function uses its own seed for internal shuffling,
            # but we also apply sampling with args.seed later.
            try:
                full_samples_df = load_afrisenti_samples(lang_code=lang_code, split=args.data_split, num_samples=None)
            except Exception as e_load:
                logging.error(f"Error loading data for {lang_code} split {args.data_split}: {e_load}. Skipping this language.", exc_info=True)
                continue

            samples_df_for_lang = pd.DataFrame()
            if not full_samples_df.empty:
                num_total_samples = len(full_samples_df)
                
                if args.test_mode:
                    num_to_sample = min(5, num_total_samples) # Max 5 samples in test mode
                    logging.info(f"    TEST MODE: Sampling {num_to_sample} samples.")
                else:
                    sample_prop = args.sample_percentage / 100.0
                    num_to_sample = max(1, int(num_total_samples * sample_prop)) if num_total_samples > 0 else 0
                    logging.info(f"    Total available samples for {lang_code} ('{args.data_split}' split): {num_total_samples}")
                    logging.info(f"    Sampling {args.sample_percentage:.1f}%%: {num_to_sample} samples using seed {args.seed}.")
                
                if num_to_sample > 0 :
                    samples_df_for_lang = full_samples_df.sample(n=num_to_sample, random_state=args.seed)
                else:
                    logging.warning(f"    Calculated 0 samples to select for {lang_code}. Skipping.")
            else:
                logging.warning(f"    No samples loaded for {lang_code} from split '{args.data_split}'.")
                
            if samples_df_for_lang.empty:
                logging.warning(f"No samples found for {lang_code} (split '{args.data_split}') after sampling. Skipping.")
                continue

            for shot_setting_str in args.shot_settings:
                use_few_shot = (shot_setting_str == 'few_shot')
                
                # Determine prompt_in_lrl (True if lang_code is not 'en')
                # This simplifies the logic as sentiment_baseline_new.py's evaluate function takes prompt_in_lrl
                prompt_in_lrl_bool = (lang_code != 'en')
                prompt_lang_description = "LRL-instruct" if prompt_in_lrl_bool else "EN-instruct"
                
                logging.info(f"  Running: Shot Setting='{shot_setting_str}', Prompt Instruction Language='{prompt_lang_description}'")

                current_generation_params = get_effective_generation_params(lang_code, model_name_str, args)
                logging.info(f"    Effective Generation Parameters: {current_generation_params}")

                # Create unique subdirectories for this specific configuration
                model_folder_name = model_name_str.replace("/", "_")
                results_path_prefix = os.path.join(args.base_output_dir, "detailed_results", model_folder_name, lang_code, prompt_lang_description, shot_setting_str)
                summaries_path_prefix = os.path.join(args.base_output_dir, "summary_reports", model_folder_name, lang_code, prompt_lang_description, shot_setting_str)
                os.makedirs(results_path_prefix, exist_ok=True)
                os.makedirs(summaries_path_prefix, exist_ok=True)
                
                detailed_results_filename = os.path.join(results_path_prefix, f"results_sentiment_baseline_new_{lang_code}.csv")
                current_summary_filename = os.path.join(summaries_path_prefix, f"summary_sentiment_baseline_new_{lang_code}.csv")

                if not args.overwrite_results and os.path.exists(detailed_results_filename) and os.path.exists(current_summary_filename):
                    logging.info(f"Results and summary for this configuration already exist and overwrite_results is False. Loading existing summary: {current_summary_filename}")
                    try:
                        existing_summary_df = pd.read_csv(current_summary_filename)
                        if not existing_summary_df.empty:
                            all_experiment_summaries.append(existing_summary_df.iloc[0].to_dict())
                            continue # Skip to next configuration
                    except Exception as e_load_sum:
                         logging.warning(f"Could not load existing summary {current_summary_filename}: {e_load_sum}. Will re-run experiment.")

                try:
                    exp_results_df = evaluate_sentiment_baseline(
                        model_name=model_name_str,
                        tokenizer=tokenizer,
                        model=model,
                        samples_df=samples_df_for_lang,
                        lang_code=lang_code,
                        prompt_in_lrl=prompt_in_lrl_bool,
                        use_few_shot=use_few_shot,
                        generation_params=current_generation_params
                    )

                    if not exp_results_df.empty:
                        exp_results_df.to_csv(detailed_results_filename, index=False, float_format='%.4f')
                        logging.info(f"Detailed results saved to {detailed_results_filename}")

                        # Ensure ground_truth_label and predicted_label are strings for metrics
                        exp_results_df['ground_truth_label'] = exp_results_df['ground_truth_label'].astype(str).str.lower()
                        exp_results_df['predicted_label'] = exp_results_df['predicted_label'].astype(str).str.lower()
                        
                        metrics = calculate_sentiment_metrics(exp_results_df, labels=SENTIMENT_LABELS) # Pass SENTIMENT_LABELS to `labels` parameter
                        
                        summary_data = {
                            'model': model_name_str,
                            'language': lang_code,
                            'prompt_language': prompt_lang_description,
                            'shot_type': shot_setting_str,
                            **metrics, # Includes accuracy, macro_f1, precision, recall, and per-class if returned
                            **current_generation_params,
                            'samples_processed': len(exp_results_df)
                        }
                        # Save current experiment's summary
                        pd.DataFrame([summary_data]).to_csv(current_summary_filename, index=False, float_format='%.4f')
                        logging.info(f"Summary for current experiment saved to {current_summary_filename}")
                        all_experiment_summaries.append(summary_data)
                    else:
                        logging.error(f"Evaluation for {model_name_str}, lang {lang_code}, shot {shot_setting_str} returned an EMPTY DataFrame. No results will be saved for this configuration.")
                        if "qwen" in model_name_str.lower():
                            logging.error(f"QWEN_DEBUG: evaluate_sentiment_baseline returned an empty DataFrame for {model_name_str}, lang {lang_code}, shot {shot_setting_str}.")

                except Exception as e_eval:
                    logging.error(f"Error during sentiment evaluation for {model_name_str}, lang {lang_code}, shot {shot_setting_str}: {e_eval}", exc_info=True)
                    if "qwen" in model_name_str.lower():
                        logging.error(f"QWEN_DEBUG: Exception during evaluate_sentiment_baseline for {model_name_str}, lang {lang_code}, shot {shot_setting_str}. Error: {e_eval}")
                    # Clean up potentially partial files if error occurs during evaluation
                    if os.path.exists(detailed_results_filename):
                        try:
                            os.remove(detailed_results_filename)
                            logging.info(f"Removed partial results file: {detailed_results_filename}")
                        except OSError as oe_rem_det:
                            logging.warning(f"Could not remove partial results file {detailed_results_filename}: {oe_rem_det}")
                    if os.path.exists(current_summary_filename):
                        try:
                            os.remove(current_summary_filename)
                            logging.info(f"Removed partial summary file: {current_summary_filename}")
                        except OSError as oe_rem_sum:
                            logging.warning(f"Could not remove partial summary file {current_summary_filename}: {oe_rem_sum}")
        
        logging.info(f"Finished processing all configurations for model {model_name_str}.")
        # Clean up model and tokenizer from memory
        logging.info(f"Finished all experiments for model {model_name_str}. Unloading model...")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"GPU memory cache cleared for {model_name_str}.")

    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        # Define overall summary directory (one level up from model-specific summaries)
        overall_summary_dir = os.path.join(args.base_output_dir, "overall_summaries")
        os.makedirs(overall_summary_dir, exist_ok=True)
        overall_summary_file_path = os.path.join(overall_summary_dir, "sentiment_baseline_new_ALL_experiments_summary.csv")
        
        # Ensure numeric columns are correctly formatted
        float_cols = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 
                      'weighted_f1', 'weighted_precision', 'weighted_recall',
                      'temperature', 'top_p', 'repetition_penalty']
        for label in SENTIMENT_LABELS: # Add per-class metrics
            float_cols.extend([f"{label}_precision", f"{label}_recall", f"{label}_f1-score"])

        for col in float_cols:
            if col in overall_summary_df.columns:
                overall_summary_df[col] = pd.to_numeric(overall_summary_df[col], errors='coerce')

        overall_summary_df.to_csv(overall_summary_file_path, index=False, float_format='%.4f')
        logging.info(f"Overall baseline sentiment summary saved to {overall_summary_file_path}")
        print("\nOverall Summary Table:")
        print(overall_summary_df.to_string())
    else:
        logging.info("No baseline sentiment experiments were successfully completed to summarize.")

    logging.info("\nAll Sentiment Baseline (New) experiments completed!")

if __name__ == "__main__":
    main() 