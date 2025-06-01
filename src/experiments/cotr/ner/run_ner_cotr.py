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
from typing import Any, Dict, Optional, List

# Add project root to Python path BEFORE other project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from refactored ner_cotr.py
from src.experiments.cotr.ner.ner_cotr import (
    initialize_model,
    evaluate_ner_cotr_multi_prompt, # Renamed
    evaluate_ner_cotr_single_prompt,
    calculate_ner_metrics_for_sample # Using this for per-sample metrics
)
from src.utils.data_loaders.load_masakhaner import load_masakhaner_samples as utils_load_masakhaner_samples
# COMET calculation is now handled within ner_cotr.py evaluation functions
# from evaluation.cotr.translation_metrics import calculate_comet_score, COMET_AVAILABLE
from huggingface_hub import login
from config import get_token

# Define logger at the module level
logger = logging.getLogger(__name__)

# TEST_MODE_SAMPLES for quick runs during development
TEST_MODE_SAMPLES = 3 # Reduced for faster testing if test_mode is on

def run_experiment(
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    base_results_path: str,
    pipeline_type: str,
    use_few_shot: bool,
    # Dictionaries of generation parameters, fully resolved by main()
    text_translation_params: Dict[str, Any], # For LRL Text -> EN Text
    eng_ner_params: Dict[str, Any],          # For EN Text -> EN Entities
    entity_translation_params: Dict[str, Any],# For EN Entities -> LRL Entities
    chain_params: Dict[str, Any],            # For Single-Prompt CoT full chain
    max_input_length: int,                   # General max input length for tokenizer
    overwrite_results: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Run a specific NER CoTR experiment configuration.
    Receives fully resolved generation parameter dictionaries.
    """
    logger.info(f"Executing run_experiment for {model_name}, lang:{lang_code}, pipeline:{pipeline_type}, few_shot:{use_few_shot}")
    if pipeline_type == 'multi_prompt':
        logger.info(f"  Text Translation Params: {text_translation_params}")
        logger.info(f"  English NER Params: {eng_ner_params}")
        logger.info(f"  Entity Translation Params: {entity_translation_params}")
    elif pipeline_type == 'single_prompt':
        logger.info(f"  Single Chain Params: {chain_params}")
    logger.info(f"  Max Input Length (Tokenizer): {max_input_length}")

    # Create directories
    results_dir = os.path.join(base_results_path, "results_per_sample") # More descriptive
    summaries_dir = os.path.join(base_results_path, "summaries_per_config")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    # Define file paths
    model_short_name = model_name.split('/')[-1].replace('.', '_') # Ensure fs safe
    shot_type_str = "fs" if use_few_shot else "zs"
    pipeline_short = "mp" if pipeline_type == "multi_prompt" else "sp"
    
    # Consistent naming convention
    base_filename = f"ner_cotr_{lang_code}_{model_short_name}_{pipeline_short}_{shot_type_str}"
    results_file = os.path.join(results_dir, f"{base_filename}_detailed.csv")
    summary_file = os.path.join(summaries_dir, f"{base_filename}_summary.csv")

    # Check for existing results
    if os.path.exists(summary_file) and not overwrite_results:
        logger.info(f"Summary file {summary_file} already exists and overwrite is False. Skipping experiment and loading summary.")
        try:
            existing_summary_df = pd.read_csv(summary_file)
            return existing_summary_df.to_dict('records')[0]
        except Exception as e:
            logger.warning(f"Could not load existing summary {summary_file}: {e}. Will recompute if overwrite is enabled or results_file is missing.")
            # Fall through to check results_file or recompute
            
    if os.path.exists(results_file) and not overwrite_results:
        logger.info(f"Detailed results file {results_file} exists and overwrite is False. Attempting to load and compute summary.")
            try:
                results_df = pd.read_csv(results_file)
            # If we loaded results, but summary was missing or failed to load, we need to compute metrics.
            except Exception as e:
            logger.error(f"Could not load existing results file {results_file}: {e}. Will recompute if overwrite is enabled.")
            results_df = None # Force recompute
        else:
        results_df = None # No existing detailed results or overwrite is True

    # Run the core evaluation if results_df is not loaded
    if results_df is None: 
            logger.info(f"Running CoTR NER experiment: Model={model_name}, Lang={lang_code}, Pipeline={pipeline_type}, Shot={'Few' if use_few_shot else 'Zero'}")
        
        # Add max_input_length to all relevant param dicts if not already there (for ner_cotr.py functions)
        text_translation_params['max_input_length'] = text_translation_params.get('max_input_length', max_input_length)
        eng_ner_params['max_input_length'] = eng_ner_params.get('max_input_length', max_input_length)
        entity_translation_params['max_input_length'] = entity_translation_params.get('max_input_length', max_input_length)
        chain_params['max_input_length'] = chain_params.get('max_input_length', max_input_length)

        start_time_experiment = time.time()
            if pipeline_type == 'multi_prompt':
            results_df = evaluate_ner_cotr_multi_prompt(
                model=model, tokenizer=tokenizer, samples_df=samples_df, lang_code=lang_code, 
                    use_few_shot=use_few_shot,
                ner_generation_params=eng_ner_params,
                translation_generation_params=text_translation_params, # Used for text LRL->EN
                # The evaluate_ner_cotr_multi_prompt in ner_cotr.py will use entity_translation_params for EN->LRL entities
                # We need to ensure ner_cotr.py's translate_entities_to_lrl gets the right dict.
                # Let's pass it explicitly if the structure of evaluate_ner_cotr_multi_prompt allows,
                # or ensure ner_cotr.py uses the correct sub-dictionary.
                # For now, assuming evaluate_ner_cotr_multi_prompt correctly uses what it needs from these.
                # It might be cleaner to pass `entity_translation_params` as a separate arg if ner_cotr.py is updated.
                # **Current ner_cotr.py's evaluate_ner_cotr_multi_prompt doesn't have a slot for separate entity_translation_params.**
                # **It uses `translation_generation_params` for BOTH text and entity translation.**
                # **THIS IS A KEY POINT - ensure consistency or update ner_cotr.py**
                # For now, assuming text_translation_params will be used for entity trans too.
                # Revisit: The new ner_cotr.py was designed to take separate dicts at the top level,
                # but the evaluate_ner_cotr_multi_prompt function in it takes translation_generation_params.
                # This implies the same params for text and entity translation. If different CLI args for entity trans are given, they are not used here.
                # This run_experiment needs to pass what evaluate_ner_cotr_multi_prompt expects.
                # So, for multi_prompt, `text_translation_params` will be used for all translations.
                model_name=model_name
                )
            elif pipeline_type == 'single_prompt':
                results_df = evaluate_ner_cotr_single_prompt(
                model=model, tokenizer=tokenizer, samples_df=samples_df, lang_code=lang_code, 
                    use_few_shot=use_few_shot,
                chain_generation_params=chain_params,
                model_name=model_name
            )
        else:
            logger.error(f"Unknown pipeline_type: {pipeline_type}")
            return None
        
        total_runtime_sec = time.time() - start_time_experiment
        logger.info(f"Experiment completed in {total_runtime_sec:.2f} seconds. DataFrame shape: {results_df.shape if results_df is not None else 'None'}")

        if results_df is None or results_df.empty:
            logger.warning(f"No results returned for {lang_code}, {model_name}, {pipeline_type}, {'Few' if use_few_shot else 'Zero'}. Skipping save and summary.")
                return None
        
        # Add runtime to the DataFrame (can be added in ner_cotr.py too)
        results_df['total_experiment_runtime_sec'] = total_runtime_sec 
        results_df['model_name'] = model_name # Add model_name for clarity in combined CSVs
        
        try:
            results_df.to_csv(results_file, index=False)
            logger.info(f"Detailed per-sample results saved to {results_file}")
        except Exception as e_save_detailed:
            logger.error(f"Error saving detailed results to {results_file}: {e_save_detailed}", exc_info=True)
            # Decide if this is fatal or if we can proceed with in-memory df for summary

    # --- Calculate Metrics and Summary (applies if results_df is loaded or computed) ---
    if results_df is None or results_df.empty: # Should not happen if computation was successful
        logger.error("Results DataFrame is unexpectedly empty before metric calculation.")
        return None
    
    metrics_list = []
    for _, row in results_df.iterrows():
        try:
            gt_entities_str = row.get('ground_truth_lrl_entities', '[]')
            pred_entities_str = row.get('final_predicted_lrl_entities', '[]')
            
            gt_entities = json.loads(gt_entities_str) if isinstance(gt_entities_str, str) else (gt_entities_str if isinstance(gt_entities_str, list) else [])
            pred_entities = json.loads(pred_entities_str) if isinstance(pred_entities_str, str) else (pred_entities_str if isinstance(pred_entities_str, list) else [])

            if not isinstance(gt_entities, list): gt_entities = []
            if not isinstance(pred_entities, list): pred_entities = []

            sample_metrics = calculate_ner_metrics_for_sample(gt_entities, pred_entities)
            metrics_list.append(sample_metrics)
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError processing entities for metrics (ID: {row.get('id', 'N/A')}): {e}. GT: '{gt_entities_str[:50]}...', Pred: '{pred_entities_str[:50]}...'. Assigning zero metrics.")
            metrics_list.append({'precision': 0, 'recall': 0, 'f1': 0})
        except Exception as e_metric:
            logger.error(f"Unexpected error calculating metrics for sample (ID: {row.get('id', 'N/A')}): {e_metric}", exc_info=True)
            metrics_list.append({'precision': 0, 'recall': 0, 'f1': 0})

    metrics_df = pd.DataFrame(metrics_list)
    results_df_with_metrics = pd.concat([results_df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)
    
    # Save the detailed results again, this time with metrics
    try:
        results_df_with_metrics.to_csv(results_file, index=False) # Overwrite with metrics
        logger.info(f"Detailed results (with metrics) re-saved to {results_file}")
    except Exception as e_save_detailed_metrics:
        logger.error(f"Error saving detailed results with metrics to {results_file}: {e_save_detailed_metrics}", exc_info=True)

    # Calculate average metrics for summary
    avg_precision = results_df_with_metrics['precision'].mean()
    avg_recall = results_df_with_metrics['recall'].mean()
    avg_f1 = results_df_with_metrics['f1'].mean()
    
    # Aggregate COMET scores (handle potential NaNs from errors or non-translation cases)
    avg_comet_lrl_to_en = results_df_with_metrics[f'comet_lrl_text_to_en{"_sp" if pipeline_type=="single_prompt" else ""}'].mean(skipna=True)
    avg_comet_en_entity_to_lrl = results_df_with_metrics[f'comet_en_entity_text_to_lrl{"_sp" if pipeline_type=="single_prompt" else ""}'].mean(skipna=True)

    summary_data = {
        'model_name': model_name,
        'language': lang_code,
        'pipeline_type': pipeline_type,
        'shot_setting': "few-shot" if use_few_shot else "zero-shot",
        'num_samples': len(samples_df),
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'avg_comet_lrl_text_to_en': avg_comet_lrl_to_en,
        'avg_comet_en_entity_text_to_lrl': avg_comet_en_entity_to_lrl,
        'total_runtime_sec': results_df_with_metrics['total_experiment_runtime_sec'].iloc[0] if 'total_experiment_runtime_sec' in results_df_with_metrics.columns and not results_df_with_metrics.empty else None,
        'detailed_results_file': results_file,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    # Include all generation params used in the summary for traceability
    if pipeline_type == 'multi_prompt':
        summary_data.update({f"param_text_trans_{k}": v for k,v in text_translation_params.items()})
        summary_data.update({f"param_eng_ner_{k}": v for k,v in eng_ner_params.items()})
        summary_data.update({f"param_entity_trans_{k}": v for k,v in entity_translation_params.items()}) # Assuming these were used
    elif pipeline_type == 'single_prompt':
        summary_data.update({f"param_chain_{k}": v for k,v in chain_params.items()})
    summary_data['param_max_input_length'] = max_input_length

    summary_df = pd.DataFrame([summary_data])
    try:
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Experiment summary saved to {summary_file}")
    except Exception as e_save_summary:
        logger.error(f"Error saving summary to {summary_file}: {e_save_summary}", exc_info=True)

    return summary_data


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run CoTR NER experiments for specified models and languages.")
    
    # Core experiment setup
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of Hugging Face model names (e.g., 'CohereLabs/aya-expanse-8b,Qwen/Qwen2.5-7B-Instruct')")
    parser.add_argument("--langs", type=str, default="sw,ha", help="Comma-separated list of language codes (e.g., 'sw,ha') for MasakhaNER.")
    parser.add_argument("--pipeline_types", type=str, default="multi_prompt,single_prompt", help="Comma-separated CoTR pipeline types (multi_prompt, single_prompt).")
    parser.add_argument("--shot_settings", type=str, default="zero_shot,few_shot", help="Comma-separated shot settings (zero_shot, few_shot).")
    
    # Dataset and Sampling
    parser.add_argument("--dataset_name", type=str, default="masakhaner", help="Dataset to use (currently only 'masakhaner' supported for NER).")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use (test, validation, train).")
    parser.add_argument("--sample_percentage", type=float, default=None, help="Percentage of samples to use per language (e.g., 10 for 10%). Set to None for all samples.")
    parser.add_argument("--num_samples_direct", type=int, default=None, help="Direct number of samples to use per language. Set to None for percentage-based sampling.")
    parser.add_argument("--max_samples_per_lang", type=int, default=None, help="Absolute maximum number of samples per language, overrides percentage or direct number if specified and lower.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # Paths and Control
    parser.add_argument("--output_dir", type=str, default="/work/bbd6522/results/ner", 
                        help="Base directory to save all results for this experiment run (a timestamped subfolder will be created here).")
    parser.add_argument("--cache_dir", type=str, default="/work/bbd6522/cache_dir", help="Directory for HuggingFace model cache.")
    parser.add_argument("--overwrite_results", action='store_true', help="Overwrite existing result files instead of skipping.")
    parser.add_argument("--test_mode", action="store_true", help=f"Run in test mode with only {TEST_MODE_SAMPLES} samples per configuration.")
    parser.add_argument("--hf_token_file", type=str, default=None, help="Path to Hugging Face auth token file if needed for gated models.")

    # General Generation Parameters
    parser.add_argument("--max_input_length", type=int, default=2048, help="Maximum input sequence length for tokenizer.")

    # --- Multi-Prompt Step-Specific Generation Parameters ---
    # 1. LRL Text -> English Text Translation
    parser.add_argument("--trans_text_temp", type=float, default=0.3, help="Temperature for LRL text to English text translation.")
    parser.add_argument("--trans_text_top_p", type=float, default=0.9, help="Top-p for LRL text to English text translation.")
    parser.add_argument("--trans_text_top_k", type=int, default=50, help="Top-k for LRL text to English text translation.")
    parser.add_argument("--trans_text_max_new_tokens", type=int, default=512, help="Max new tokens for LRL text to English text translation.")
    parser.add_argument("--trans_text_rep_penalty", type=float, default=1.0, help="Repetition penalty for LRL text to English text translation.")
    parser.add_argument("--trans_text_do_sample", type=lambda x: (str(x).lower() == 'true'), default=None, help="Explicitly set do_sample for text translation (True/False). If None, derived from temperature.")

    # 2. English Text -> English NER
    parser.add_argument("--ner_en_temp", type=float, default=0.2, help="Temperature for English NER.")
    parser.add_argument("--ner_en_top_p", type=float, default=0.85, help="Top-p for English NER.")
    parser.add_argument("--ner_en_top_k", type=int, default=40, help="Top-k for English NER.")
    parser.add_argument("--ner_en_max_new_tokens", type=int, default=300, help="Max new tokens for English NER output (JSON list).")
    parser.add_argument("--ner_en_rep_penalty", type=float, default=1.1, help="Repetition penalty for English NER.")
    parser.add_argument("--ner_en_do_sample", type=lambda x: (str(x).lower() == 'true'), default=None, help="Explicitly set do_sample for English NER (True/False). If None, derived from temperature.")

    # 3. English Entities -> LRL Entities Translation
    # NOTE: As per current ner_cotr.py's evaluate_ner_cotr_multi_prompt, these might not be used directly.
    # It uses the same `translation_generation_params` (derived from --trans_text_*) for both text and entity translation.
    # Keeping these args for future flexibility or if ner_cotr.py is updated to take separate entity trans params.
    parser.add_argument("--trans_entity_temp", type=float, default=0.3, help="Temperature for English entity text to LRL entity text translation.")
    parser.add_argument("--trans_entity_top_p", type=float, default=0.9, help="Top-p for English entity text to LRL entity text translation.")
    parser.add_argument("--trans_entity_top_k", type=int, default=50, help="Top-k for English entity text to LRL entity text translation.")
    parser.add_argument("--trans_entity_max_new_tokens", type=int, default=256, help="Max new tokens for English entity to LRL entity translation (per list).")
    parser.add_argument("--trans_entity_rep_penalty", type=float, default=1.0, help="Repetition penalty for English entity to LRL entity translation.")
    parser.add_argument("--trans_entity_do_sample", type=lambda x: (str(x).lower() == 'true'), default=None, help="Explicitly set do_sample for entity translation (True/False). If None, derived from temperature.")

    # --- Single-Prompt Chain Generation Parameters ---
    parser.add_argument("--chain_temp", type=float, default=0.25, help="Temperature for the single-prompt CoT chain.")
    parser.add_argument("--chain_top_p", type=float, default=0.85, help="Top-p for the single-prompt CoT chain.")
    parser.add_argument("--chain_top_k", type=int, default=40, help="Top-k for the single-prompt CoT chain.")
    parser.add_argument("--chain_max_new_tokens", type=int, default=768, help="Max new tokens for the entire single-prompt CoT chain response.")
    parser.add_argument("--chain_rep_penalty", type=float, default=1.1, help="Repetition penalty for the single-prompt CoT chain.")
    parser.add_argument("--chain_do_sample", type=lambda x: (str(x).lower() == 'true'), default=None, help="Explicitly set do_sample for single chain (True/False). If None, derived from temperature.")

    args = parser.parse_args()
    
    # Process comma-separated string arguments into lists
    args.models = [m.strip() for m in args.models.split(',')]
    args.langs = [lang.strip() for lang in args.langs.split(',')]
    args.pipeline_types = [p.strip() for p in args.pipeline_types.split(',')]
    args.shot_settings = [s.strip() for s in args.shot_settings.split(',')]

    return args

def main():
    args = parse_cli_args()

    # --- Configure Logging ---
    log_level = logging.DEBUG if args.test_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Silence less important loggers
    logging.getLogger("transformers.generation.utils").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)

    logger.info(f"Starting NER CoTR experiments with arguments: {args}")

    # Hugging Face Login
    if args.hf_token_file:
        try:
            token = get_token(args.hf_token_file)
            login(token=token)
            logger.info("Successfully logged into Hugging Face Hub.")
        except Exception as e:
            logger.warning(f"Could not log into Hugging Face Hub using token from {args.hf_token_file}: {e}. Gated models may fail.")

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Prepare output directory
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_experiment_path = os.path.join(args.output_dir, timestamp_dir)
    os.makedirs(current_experiment_path, exist_ok=True)
    logger.info(f"All outputs for this run will be saved under: {current_experiment_path}")

    # Store all individual experiment summaries
    all_experiment_summaries = []

    # Construct generation parameter dictionaries from CLI args
    # These will be passed to run_experiment
    text_translation_params = {
        "temperature": args.trans_text_temp, "top_p": args.trans_text_top_p, "top_k": args.trans_text_top_k,
        "max_new_tokens": args.trans_text_max_new_tokens, "repetition_penalty": args.trans_text_rep_penalty,
        "do_sample": args.trans_text_do_sample if args.trans_text_do_sample is not None else (args.trans_text_temp > 1e-5)
    }
    eng_ner_params = {
        "temperature": args.ner_en_temp, "top_p": args.ner_en_top_p, "top_k": args.ner_en_top_k,
        "max_new_tokens": args.ner_en_max_new_tokens, "repetition_penalty": args.ner_en_rep_penalty,
        "do_sample": args.ner_en_do_sample if args.ner_en_do_sample is not None else (args.ner_en_temp > 1e-5)
    }
    # Note: These entity_translation_params might be used if ner_cotr.py's multi-prompt path is updated.
    # Currently, text_translation_params are used for entity translation too in multi-prompt.
    entity_translation_params = {
        "temperature": args.trans_entity_temp, "top_p": args.trans_entity_top_p, "top_k": args.trans_entity_top_k,
        "max_new_tokens": args.trans_entity_max_new_tokens, "repetition_penalty": args.trans_entity_rep_penalty,
        "do_sample": args.trans_entity_do_sample if args.trans_entity_do_sample is not None else (args.trans_entity_temp > 1e-5)
    }
    chain_params = {
        "temperature": args.chain_temp, "top_p": args.chain_top_p, "top_k": args.chain_top_k,
        "max_new_tokens": args.chain_max_new_tokens, "repetition_penalty": args.chain_rep_penalty,
        "do_sample": args.chain_do_sample if args.chain_do_sample is not None else (args.chain_temp > 1e-5)
    }

    for model_name in args.models:
        logger.info(f"===== Initializing Model: {model_name} =====")
        try:
            tokenizer, model = initialize_model(model_name, cache_path=args.cache_dir)
        except Exception as e_init:
            logger.critical(f"Failed to initialize model {model_name}: {e_init}", exc_info=True)
            continue # Skip to next model

        for lang_code in args.langs:
            logger.info(f"---- Processing Language: {lang_code} for Model: {model_name} ----")
            
            # Load data for the current language
            if args.dataset_name.lower() == "masakhaner":
                # Load all samples for the split first, then the runner script will handle percentage sampling.
                full_samples_df = utils_load_masakhaner_samples(
                    lang_code=lang_code,
                    split=args.dataset_split,
                    sample_percentage=None, # Load all samples from the loader
                    seed=args.seed
                )
            else:
                logger.error(f"Dataset {args.dataset_name} not supported for NER. Skipping lang {lang_code}.")
                continue

            if full_samples_df.empty:
                logger.warning(f"No samples loaded for {lang_code} from {args.dataset_name} (split: {args.dataset_split}). Skipping.")
                continue

            # Determine number of samples
            num_to_sample = 0
            source_of_sample_size = "default (all samples)"

            if args.sample_percentage is not None and not full_samples_df.empty:
                num_to_sample = max(1, int(args.sample_percentage / 100.0 * len(full_samples_df)))
                source_of_sample_size = f"{args.sample_percentage}% of {len(full_samples_df)}"
                logger.info(f"Initial num_to_sample from percentage: {num_to_sample} (source: {source_of_sample_size})")
            elif args.num_samples_direct is not None: # If a direct number is given
                num_to_sample = args.num_samples_direct
                source_of_sample_size = f"direct value {args.num_samples_direct}"
                logger.info(f"Initial num_to_sample from direct value: {num_to_sample} (source: {source_of_sample_size})")
            else: # Default fallback if neither percentage nor direct number is specified
                num_to_sample = len(full_samples_df) # Use all if no specific sampling instruction
                logger.info(f"Initial num_to_sample (all samples): {num_to_sample} (source: {source_of_sample_size})")

            if args.max_samples_per_lang is not None:
                if num_to_sample > args.max_samples_per_lang:
                    logger.info(f"Capping num_to_sample from {num_to_sample} to {args.max_samples_per_lang} due to max_samples_per_lang.")
                    num_to_sample = args.max_samples_per_lang
                    source_of_sample_size += f", capped by max_samples_per_lang to {num_to_sample}"
            
            if args.test_mode: # Override for test mode
                # TEST_MODE_SAMPLES should be defined, e.g., TEST_MODE_SAMPLES = 5 at the script start
                if num_to_sample > TEST_MODE_SAMPLES:
                    logger.info(f"Capping num_to_sample from {num_to_sample} to {TEST_MODE_SAMPLES} due to test_mode.")
                    num_to_sample = min(TEST_MODE_SAMPLES, len(full_samples_df)) 
                    source_of_sample_size += f", capped by test_mode to {num_to_sample}"
            
            if num_to_sample == 0 and len(full_samples_df) > 0 : 
                 logger.info(f"num_to_sample was 0, setting to 1 as minimum.")
                 num_to_sample = 1
                 source_of_sample_size += f", adjusted to 1 as minimum"
            
            logger.info(f"Final num_to_sample for {lang_code}: {num_to_sample} (derived from: {source_of_sample_size})")

            if num_to_sample > 0 and not full_samples_df.empty:
                 current_samples_df = full_samples_df.sample(n=num_to_sample, random_state=args.seed)
            elif num_to_sample > 0 and full_samples_df.empty:
                 logger.warning(f"full_samples_df is empty for {lang_code}, cannot sample {num_to_sample} samples.")
                 current_samples_df = pd.DataFrame()
            else: # If no samples to select (e.g. full_samples_df was empty or num_to_sample became 0)
                 current_samples_df = pd.DataFrame()


            if current_samples_df.empty:
                logger.warning(f"No samples selected for {lang_code} after sampling (target: {num_to_sample}). Skipping.")
                continue

            logger.info(f"Loaded and sampled {len(current_samples_df)} examples for {lang_code} (target: {num_to_sample} from {len(full_samples_df)}).")
            # Ensure 'text' and 'entities' columns are present. MasakhaNER loader should provide them.
            if 'text' not in current_samples_df.columns or 'entities' not in current_samples_df.columns:
                logger.error(f"Loaded data for {lang_code} is missing 'text' or 'entities' column. Columns: {current_samples_df.columns}. Skipping.")
                continue


            for pipeline_type in args.pipeline_types:
                for shot_setting_str in args.shot_settings:
                    use_few_shot = (shot_setting_str == 'few_shot')
                    logger.info(f"--- Configuration: Lang={lang_code}, Model={model_name}, Pipeline={pipeline_type}, Shot={'Few' if use_few_shot else 'Zero'} ---")

                    experiment_summary = run_experiment(
                        model_name=model_name,
                        tokenizer=tokenizer,
                        model=model,
                        samples_df=current_samples_df,
                        lang_code=lang_code,
                        base_results_path=current_experiment_path, # Save under timestamped dir
                        pipeline_type=pipeline_type,
                        use_few_shot=use_few_shot,
                        text_translation_params=text_translation_params,
                        eng_ner_params=eng_ner_params,
                        entity_translation_params=entity_translation_params,
                        chain_params=chain_params,
                        max_input_length=args.max_input_length,
                        overwrite_results=args.overwrite_results
                    )
                    if experiment_summary:
                        all_experiment_summaries.append(experiment_summary)
        
        # Clean up model and tokenizer from memory after processing all its languages/configs
        del model
        del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info(f"===== Model {model_name} unloaded. =====")

    # After all models and configurations are processed
    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        overall_summary_file = os.path.join(current_experiment_path, "ner_cotr_ALL_CONFIGS_summary.csv")
        try:
            overall_summary_df.to_csv(overall_summary_file, index=False)
            logger.info(f"Overall summary of all configurations saved to: {overall_summary_file}")
            logger.info("--- Overall Summary Table ---")
            print(overall_summary_df.to_string()) # Print to console
        except Exception as e:
            logger.error(f"Failed to save overall summary CSV: {e}")
        
        # Basic plotting example (can be expanded)
        if not overall_summary_df.empty and 'avg_f1' in overall_summary_df.columns:
            try:
                plt.figure(figsize=(12, 7))
                sns.barplot(data=overall_summary_df, x='model_name', y='avg_f1', hue='language', dodge=True)
                plt.title('Average F1 Score by Model and Language (NER CoTR)')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Average F1 Score')
                plt.xlabel('Model Name')
                plt.tight_layout()
                plot_file = os.path.join(current_experiment_path, "ner_cotr_avg_f1_scores.png")
                plt.savefig(plot_file)
                logger.info(f"Summary plot saved to {plot_file}")
                plt.close()
            except Exception as e_plot:
                logger.warning(f"Could not generate summary plot: {e_plot}")
    else:
        logger.info("No experiments were successfully completed to generate an overall summary.")

    logger.info("NER CoTR experiment run finished.")

if __name__ == "__main__":
    main() 