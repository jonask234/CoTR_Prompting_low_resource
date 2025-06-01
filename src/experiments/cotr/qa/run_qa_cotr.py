import sys
import os
import logging
import re
import json # Added for COMET score list storage

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to the Python path
# Go up four levels from the script's directory: qa -> cotr -> experiments -> src -> project_root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from src.utils.data_loaders.load_tydiqa import load_tydiqa_samples
from src.experiments.cotr.qa.qa_cotr import (
    evaluate_qa_cotr, 
    evaluate_qa_cotr_single_prompt,
    initialize_model
)
from evaluation.cotr.translation_metrics import COMET_AVAILABLE, calculate_comet_score as calculate_translation_quality
from src.experiments.baseline.qa.qa_baseline import calculate_qa_f1
from huggingface_hub import login
from config import get_token
import argparse
import torch
from typing import Any, Dict, Optional, List
from tqdm import tqdm
import time

# Define standard parameters for both baseline and CoTR - EXACTLY matching baseline parameters
STANDARD_PARAMETERS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 50
}

# Language-specific standard parameters
LANGUAGE_PARAMETERS = {
    "sw": {  # Swahili
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": STANDARD_PARAMETERS["top_k"],
        "max_tokens": STANDARD_PARAMETERS["max_tokens"]
    },
    "te": {  # Telugu
        "temperature": 0.15,
        "top_p": 0.75,
        "top_k": STANDARD_PARAMETERS["top_k"],
        "max_tokens": STANDARD_PARAMETERS["max_tokens"]
    }
}

# --- Standardized Base Generation Parameters --- #
# These are the most general defaults.
BASE_GEN_PARAMS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.0, # Default repetition penalty
    # do_sample will be derived from temperature (temp > 0)
}

# --- Task-Specific Defaults (overriding or adding to BASE_GEN_PARAMS) --- #
# For the English QA step in multi-prompt, or the whole chain in single-prompt
DEFAULT_QA_PARAMS = {
    **BASE_GEN_PARAMS, # Inherit base
    "max_tokens": 50, # Max tokens for the answer itself
    "temperature": 0.2, # Often lower for more factual QA
    "repetition_penalty": 1.1 
}

# For all translation steps (LRL->EN Q, LRL->EN C, EN->LRL A)
DEFAULT_TRANSLATION_PARAMS = {
    **BASE_GEN_PARAMS, # Inherit base
    "max_tokens": 200, # Max tokens for translated text segments
    "temperature": 0.35, # Can be slightly higher for translation fluidity
    "repetition_penalty": 1.05
}

# --- Language-Specific Overrides --- #
# These override task-specific defaults for certain languages.
LANG_QA_PARAM_OVERRIDES = {
    "sw": { "temperature": 0.18, "top_p": 0.85},
    "te": { "temperature": 0.15, "top_p": 0.75}
}
LANG_TRANSLATION_PARAM_OVERRIDES = {
    "sw": { "temperature": 0.3, "max_tokens": 220},
    "te": { "temperature": 0.3, "max_tokens": 220}
}

# --- Model-Specific Adjustments (applied AFTER lang/CLI overrides) --- #
# This function can adjust parameters based on the model_name.
def apply_model_specific_adjustments(params: Dict, model_name: str, step_type: str) -> Dict:
    adj_params = params.copy()
    if "aya" in model_name.lower():
        if step_type == "qa":
            adj_params["temperature"] = max(0.1, adj_params.get("temperature", 0.2) * 0.9)
            # adj_params["repetition_penalty"] = adj_params.get("repetition_penalty", 1.1) * 1.05
        elif step_type == "translation":
            adj_params["temperature"] = max(0.1, adj_params.get("temperature", 0.35) * 0.9)
    elif "qwen" in model_name.lower():
        if step_type == "qa":
            adj_params["top_p"] = max(0.7, adj_params.get("top_p", 0.85) * 0.9)
            adj_params["top_k"] = min(adj_params.get("top_k",40), 35) # Example: Qwen prefers lower top_k for QA
        # Qwen translation might also benefit from specific settings
    # Add more model-specific rules here
    return adj_params

def calculate_exact_match_score(ground_truth_data: Any, predicted_answer: str) -> float:
    """Calculate Exact Match score. Ground_truth_data can be str, list of str, or dict with 'text'."""
    references = []
    if isinstance(ground_truth_data, str):
        references = [ground_truth_data.strip()]
    elif isinstance(ground_truth_data, list):
            references = [str(ref).strip() for ref in ground_truth_data]
    elif isinstance(ground_truth_data, dict) and 'text' in ground_truth_data and isinstance(ground_truth_data['text'], list):
        references = [str(ref).strip() for ref in ground_truth_data['text']]
    elif isinstance(ground_truth_data, dict) and 'text' in ground_truth_data and isinstance(ground_truth_data['text'], str):
        references = [ground_truth_data['text'].strip()]
    else:
        logging.warning(f"Unexpected ground_truth_data format for EM: {type(ground_truth_data)}, value: {str(ground_truth_data)[:100]}")
        # Attempt to convert to string and use as a single reference
        try:
            references = [str(ground_truth_data).strip()]
        except: # If str conversion fails, no valid reference
            return 0.0 

    prediction = str(predicted_answer).strip()

    def normalize_text(s):
        s = s.lower()
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    normalized_prediction = normalize_text(prediction)
    for ref in references:
        if normalize_text(ref) == normalized_prediction:
            return 1.0
    return 0.0

def run_experiment(
    model_name: str, 
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame, 
    lang_code: str, 
    base_results_path: str,
    pipeline_type: str,
    use_few_shot: bool,
    # NEW: Pass parameter dictionaries directly
    qa_params_dict: Dict[str, Any],
    trans_params_dict: Dict[str, Any],
    chain_params_dict: Dict[str, Any],
    overwrite_results: bool = False,
    max_input_length: int = 4096
):
    logger = logging.getLogger(__name__) # Define logger for this function
    if samples_df.empty:
        logging.warning(f"Empty samples dataframe for {lang_code}, {model_name}. Skipping experiment.")
        return None # Return None to indicate failure or skip
    
    results_dir = os.path.join(base_results_path, "results")
    summaries_dir = os.path.join(base_results_path, "summaries")
    # plots_dir = os.path.join(base_results_path, "plots") # Plots generated from overall summary
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    # os.makedirs(plots_dir, exist_ok=True)
    
    model_short = model_name.split('/')[-1]
    shot_type = "fs" if use_few_shot else "zs"
    pipeline_short = "mp" if pipeline_type == "multi_prompt" else "sp"
    
    results_file = os.path.join(results_dir, f"results_cotr_{pipeline_short}_{shot_type}_qa_tydiqa_{lang_code}_{model_short}.csv")
    summary_file = os.path.join(summaries_dir, f"summary_cotr_{pipeline_short}_{shot_type}_qa_tydiqa_{lang_code}_{model_short}.csv")
    
    results_df_for_metrics = pd.DataFrame() # Initialize

    if os.path.exists(results_file) and not overwrite_results:
        logging.info(f"Results file {results_file} already exists and overwrite_results is False. Loading existing results to compute summary.")
        try:
            results_df_for_metrics = pd.read_csv(results_file)
            if results_df_for_metrics.empty:
                logging.warning(f"Loaded results file {results_file} is empty. Will re-run experiment.")
            else:
                logging.info(f"Successfully loaded {len(results_df_for_metrics)} results from {results_file}.")
        except pd.errors.EmptyDataError:
            logging.warning(f"Results file {results_file} is empty. Will re-run experiment.")
        except Exception as e_load:
            logging.error(f"Error loading results file {results_file}: {e_load}. Will re-run experiment.")
    
    # If results_df_for_metrics is still empty (file didn't exist, was empty, overwrite=True, or load failed)
    if results_df_for_metrics.empty:
        logging.info(f"Running CoTR QA experiment: Model={model_name}, Lang={lang_code}, Pipeline={pipeline_type}, Shot={'Few' if use_few_shot else 'Zero'}")
        if pipeline_type == 'multi_prompt':
            # Parameters for multi-prompt, derived from qa_temp, trans_temp etc. and model/lang overrides
            # These dictionaries (qa_params_dict, trans_params_dict) should be defined before this block based on args and defaults.
            logging.info(f"  Multi-Prompt QA Params (from dict): {qa_params_dict}")
            logging.info(f"  Multi-Prompt Translation Params (from dict): {trans_params_dict}")
            
            # Ensure qa_params_dict has 'max_new_tokens' instead of 'max_tokens' for the QA step
            if 'max_tokens' in qa_params_dict and 'max_new_tokens' not in qa_params_dict:
                qa_params_dict['max_new_tokens'] = qa_params_dict.pop('max_tokens')
            elif 'max_tokens' in qa_params_dict and 'max_new_tokens' in qa_params_dict:
                pass # Assume max_new_tokens is correctly set if both exist

            # Ensure trans_params_dict has 'max_new_tokens' instead of 'max_tokens' for translation steps
            if 'max_tokens' in trans_params_dict and 'max_new_tokens' not in trans_params_dict:
                trans_params_dict['max_new_tokens'] = trans_params_dict.pop('max_tokens')
            elif 'max_tokens' in trans_params_dict and 'max_new_tokens' in trans_params_dict:
                pass # Assume max_new_tokens is correctly set if both exist

            results_df_for_metrics = evaluate_qa_cotr(
                model_name=model_name,
                tokenizer=tokenizer,
                model=model,
                samples_df=samples_df,
                lang_code=lang_code,
                use_few_shot=use_few_shot,
                qa_params=qa_params_dict, # Pass the correctly populated qa_params_dict
                translation_params=trans_params_dict, # Pass the correctly populated trans_params_dict
                max_input_length=max_input_length
            )
        elif pipeline_type == 'single_prompt':
            # Parameters for single-prompt, derived from qa_temp etc. and model/lang overrides
            # This dictionary (chain_params_dict) should be defined before this block.
            # For single prompt, all generation happens in one go, so use "chain" parameters
            # This dict comes from get_effective_qa_params which uses 'max_tokens' from defaults/CLI.
            # The evaluate_qa_cotr_single_prompt function expects 'max_new_tokens'.
            
            # Ensure chain_params_dict has 'max_new_tokens' instead of 'max_tokens'
            if 'max_tokens' in chain_params_dict and 'max_new_tokens' not in chain_params_dict:
                chain_params_dict['max_new_tokens'] = chain_params_dict.pop('max_tokens')
            elif 'max_tokens' in chain_params_dict and 'max_new_tokens' in chain_params_dict:
                # If both exist, 'max_new_tokens' is preferred by the evaluation function.
                # The get_effective_qa_params function populates 'max_tokens' from CLI if specified.
                # To ensure CLI override for 'max_tokens' is respected and used as 'max_new_tokens':
                chain_params_dict['max_new_tokens'] = chain_params_dict.pop('max_tokens')

            logger.info(f"  CoTR Chain Params (for single_prompt after potential rename): {chain_params_dict}")
            results_df_for_metrics = evaluate_qa_cotr_single_prompt(
                model_name=model_name,
                tokenizer=tokenizer,
                model=model,
                samples_df=samples_df,
                lang_code=lang_code,
                use_few_shot=use_few_shot,
                temperature=chain_params_dict['temperature'],
                do_sample=chain_params_dict['do_sample'],
                top_p=chain_params_dict['top_p'],
                top_k=chain_params_dict['top_k'],
                repetition_penalty=chain_params_dict['repetition_penalty'],
                max_new_tokens=chain_params_dict['max_new_tokens'], # Ensure this key exists in chain_params_dict
                max_input_length=max_input_length
            )
        else:
            logging.error(f"Unknown pipeline_type: {pipeline_type}")
            return None
        
        if results_df_for_metrics.empty:
            logging.warning(f"No results returned by evaluation function for {lang_code}, {model_name}, {pipeline_type}, {shot_type}.")
            return None # Critical failure if evaluation returns empty
        
        results_df_for_metrics.to_csv(results_file, index=False)
        logging.info(f"Results saved to {results_file}")
        
    # --- Calculate Metrics from results_df_for_metrics --- 
    if results_df_for_metrics.empty:
        logging.warning(f"results_df_for_metrics is empty for {results_file} even after trying to run/load. Skipping summary.")
        return None

    f1_scores, em_scores = [], []
    comet_q_lrl_en_list, comet_c_lrl_en_list, comet_a_en_lrl_list = [], [], []
    num_samples_with_results = len(results_df_for_metrics)

    for _, row in results_df_for_metrics.iterrows():
        # The qa_cotr.py script now stores lrl_ground_truth_answers_list as a direct list of strings.
        gt_text_list = row.get('lrl_ground_truth_answers_list', []) # Default to empty list if missing
        if not isinstance(gt_text_list, list): # Ensure it's a list, fallback if not
            logging.warning(f"Expected 'lrl_ground_truth_answers_list' to be a list, got {type(gt_text_list)}. Using empty list for metrics.")
            gt_text_list = []
        
        # Ensure all items in gt_text_list are strings, or handle/log if not
        processed_gt_text_list = []
        for item in gt_text_list:
            if isinstance(item, str):
                processed_gt_text_list.append(item)
            else:
                logging.warning(f"Non-string item '{item}' found in ground truth list for row {row.get('id', 'N/A')}. Skipping this item for metrics.")
        gt_text_list = processed_gt_text_list

        if not gt_text_list: # If list is empty after processing
            logging.warning(f"Empty or invalid 'lrl_ground_truth_answers_list' for row {row.get('id', 'N/A')}. Metrics for this sample may be 0.")
            # gt_text_list will be empty, calculate_qa_f1 and calculate_exact_match_score should handle this (e.g. return 0)

        predicted_answer = str(row.get('lrl_answer_model_final', "")) # Ensure it's a string
        
        # F1 score expects a dict like {'text': [list_of_answers]} for reference from TyDiQA official script style.
        # calculate_exact_match_score expects the list of reference strings directly.
        f1_scores.append(calculate_qa_f1({'text': gt_text_list if gt_text_list else [""]}, predicted_answer)) # Pass empty string in list if gt_text_list is empty
        em_scores.append(calculate_exact_match_score(gt_text_list, predicted_answer))
        
        # COMET scores processing remains the same if column names for COMET scores are correct in results_df_for_metrics
        if 'comet_lrl_q_to_en' in row and pd.notna(row['comet_lrl_q_to_en']): # Changed key from comet_score_q_lrl_en
            comet_q_lrl_en_list.append(row['comet_lrl_q_to_en'])
        if 'comet_lrl_c_to_en' in row and pd.notna(row['comet_lrl_c_to_en']): # Changed key from comet_score_c_lrl_en
            comet_c_lrl_en_list.append(row['comet_lrl_c_to_en'])
        if 'comet_en_a_to_lrl' in row and pd.notna(row['comet_en_a_to_lrl']): # Changed key from comet_score_a_en_lrl
            comet_a_en_lrl_list.append(row['comet_en_a_to_lrl'])
    
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    avg_comet_q = np.mean(comet_q_lrl_en_list) if comet_q_lrl_en_list else None
    avg_comet_c = np.mean(comet_c_lrl_en_list) if comet_c_lrl_en_list else None
    avg_comet_a = np.mean(comet_a_en_lrl_list) if comet_a_en_lrl_list else None
    
    summary_data = {
        'model': model_short, 'language': lang_code, 'pipeline': pipeline_type, 'shot_type': shot_type,
        'samples': num_samples_with_results, 'f1_score': avg_f1,
        # Log the dictionaries
        'qa_params': qa_params_dict,
        'trans_params': trans_params_dict,
        'chain_params': chain_params_dict, # Relevant for single_prompt
        'avg_comet_q_lrl_en': avg_comet_q, 'avg_comet_c_lrl_en': avg_comet_c, 'avg_comet_a_en_lrl': avg_comet_a
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(summary_file, index=False, float_format='%.4f') # Use float_format
    logging.info(f"Summary saved to {summary_file}")
    logging.info(f"Summary for {model_name}, {lang_code}, {pipeline_type}, {shot_type}:\n{summary_df.to_string()}")
    return summary_data

def parse_args():
    parser = argparse.ArgumentParser(description='Run QA Chain of Translation Prompting (CoTR) experiments')
    parser.add_argument('--models', type=str, default="CohereLabs/aya-expanse-8b,Qwen/Qwen2.5-7B-Instruct", help='Comma-separated list of models')
    parser.add_argument('--langs', nargs='+', default=['en', 'sw', 'te'], help='Languages to evaluate')
    parser.add_argument('--sample_percentage', type=float, default=10.0, help='Percentage of TyDiQA GoldP validation samples (0-100)')
    parser.add_argument('--pipeline_types', nargs='+', default=['multi_prompt', 'single_prompt'], choices=['multi_prompt', 'single_prompt'])
    parser.add_argument('--shot_settings', nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'])
    parser.add_argument('--test_mode', action='store_true', help='Run with first 5 samples only')
    parser.add_argument('--hf_token', type=str, help='HuggingFace API token')
    parser.add_argument('--base_output_dir', type=str, default="/work/bbd6522/results/qa/cotr_v2", help="Base directory for all outputs.")
    parser.add_argument("--overwrite_results", action="store_true", help="Overwrite existing results files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and reproducibility.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")

    # QA Step specific CLI overrides (for multi-prompt English QA or single-prompt chain)
    parser.add_argument("--qa_temp", type=float, default=None, help="Temperature for QA step/chain.")
    parser.add_argument("--qa_top_p", type=float, default=None, help="Top-p for QA step/chain.")
    parser.add_argument("--qa_top_k", type=int, default=None, help="Top-k for QA step/chain.")
    parser.add_argument("--qa_max_tokens", type=int, default=None, help="Max new tokens for QA step/chain output.")
    parser.add_argument("--qa_rep_penalty", type=float, default=None, help="Repetition penalty for QA step/chain.")

    # Translation Step specific CLI overrides (for multi-prompt translation steps)
    parser.add_argument("--trans_temp", type=float, default=None, help="Temperature for translation steps.")
    parser.add_argument("--trans_top_p", type=float, default=None, help="Top-p for translation steps.")
    parser.add_argument("--trans_top_k", type=int, default=None, help="Top-k for translation steps.")
    parser.add_argument("--trans_max_tokens", type=int, default=None, help="Max new tokens for translation steps.")
    parser.add_argument("--trans_rep_penalty", type=float, default=None, help="Repetition penalty for translation steps.")
    
    parser.add_argument("--dataset_version", type=str, default="goldp", help="Dataset version to use (e.g., goldp, minimal for TyDiQA).")
    parser.add_argument("--data_split", type=str, default="validation",
                        choices=["train", "validation", "test"], # TyDiQA GoldP has 'train' and 'validation'
                        help="Dataset split to use. Default: validation.")
    parser.add_argument("--max_samples_per_lang", type=int, default=None,
                        help="Maximum number of samples per language to process after percentage sampling. Default: None (no cap).")
    parser.add_argument("--max_input_length", type=int, default=4096, help="Maximum input token length for the tokenizer globally.")

    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    logger = logging.getLogger(__name__) # Initialize logger

    # HF Login (from new script, more robust)

    models_list = [m.strip() for m in args.models.split(',') if m.strip()]
    summaries_output_dir = os.path.join(args.base_output_dir, "summaries")
    plots_output_dir = os.path.join(args.base_output_dir, "plots")
    os.makedirs(summaries_output_dir, exist_ok=True)
    os.makedirs(plots_output_dir, exist_ok=True) 
    logging.info(f"All CoTR QA experiment outputs will be saved under: {args.base_output_dir}")

    all_experiment_summaries = []

    for model_name_str in models_list:
        logging.info(f"\n{'='*20} Initializing Model: {model_name_str} {'='*20}")
        model, tokenizer = None, None
        try:
            tokenizer, model = initialize_model(model_name_str) 
            logging.info(f"Model {model_name_str} initialized successfully.")
        except Exception as e_init:
            logging.error(f"Failed to initialize model {model_name_str}: {e_init}. Skipping.", exc_info=True)
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        for lang_code in args.langs:
            logging.info(f"QA CoTR Main: Loading TyDiQA samples for language: {lang_code}, version: {args.dataset_version}")
            
            # Use the data_split argument directly
            split_to_use = args.data_split
            
            logging.info(f"QA CoTR Main: Using split '{split_to_use}' for dataset version '{args.dataset_version}'")
            all_samples_df = load_tydiqa_samples(lang_code=lang_code, split=split_to_use, sample_percentage=args.sample_percentage)
            
            if all_samples_df.empty:
                logging.warning(f"QA CoTR Main: No samples loaded for {lang_code} (split: {split_to_use}, sample_percentage: {args.sample_percentage}%). Skipping this language for {model_name_str}.")
                continue

            # ---- ADD DEBUG LOGGING HERE ----
            logging.info(f"QA CoTR Main: After load_tydiqa_samples for {lang_code} ({len(all_samples_df)} samples loaded initially):")
            if 'question' in all_samples_df.columns:
                logging.info(f"  Sample of 'question' types: {all_samples_df['question'].apply(type).value_counts().to_string(index=True)}")
                logging.info(f"  Head of 'question': \n{all_samples_df['question'].head().to_string(index=True)}")
                q_is_empty_str_loaded = (all_samples_df['question'].fillna('').astype(str).str.strip() == "")
                logging.info(f"  Count of empty/whitespace in loaded 'question': {q_is_empty_str_loaded.sum()}")
                q_is_none_or_nan_loaded = all_samples_df['question'].isnull()
                logging.info(f"  Count of None/NaN in loaded 'question': {q_is_none_or_nan_loaded.sum()}")
            else:
                logging.warning("  'question' column NOT FOUND in loaded samples.")

            if 'context' in all_samples_df.columns:
                logging.info(f"  Sample of 'context' types: {all_samples_df['context'].apply(type).value_counts().to_string(index=True)}")
                logging.info(f"  Head of 'context': \n{all_samples_df['context'].head().to_string(index=True)}")
                c_is_empty_str_loaded = (all_samples_df['context'].fillna('').astype(str).str.strip() == "")
                logging.info(f"  Count of empty/whitespace in loaded 'context': {c_is_empty_str_loaded.sum()}")
                c_is_none_or_nan_loaded = all_samples_df['context'].isnull()
                logging.info(f"  Count of None/NaN in loaded 'context': {c_is_none_or_nan_loaded.sum()}")
            else:
                logging.warning("  'context' column NOT FOUND in loaded samples.")
            # ---- END DEBUG LOGGING ----

            # Check for essential columns immediately after loading
            required_columns = ['question', 'context', 'answers'] # 'id' is also good but not strictly used by core logic downstream

            # Filter out rows where question or context might be NaN or empty strings BEFORE test_mode slicing
            # Convert to string and strip to catch whitespace-only entries, then check for emptiness
            all_samples_df = all_samples_df[
                all_samples_df['question'].astype(str).str.strip().ne('') &
                all_samples_df['context'].astype(str).str.strip().ne('')
            ]
            if all_samples_df.empty:
                logging.warning(f"QA CoTR Main: All samples for {lang_code} had empty questions or contexts after filtering. Skipping.")
                continue
            logging.info(f"QA CoTR Main: {len(all_samples_df)} samples remaining for {lang_code} after filtering empty Q/C.")

            if args.test_mode:
                all_samples_df = all_samples_df.head(5)
                logging.info(f"Test mode: Using first {len(all_samples_df)} samples for {lang_code}.")
            
            # Check for ground truth column and log count (as in new script)
            if 'answers' not in all_samples_df.columns: # TyDiQA loader typically provides 'answers'
                logger.error(f"Ground truth column 'answers' not found in data for {lang_code}. Check data loader. Skipping.")
                continue
            else:
                # Ensure 'answers' column is mapped to 'lrl_ground_truth_answers_list' as expected by qa_cotr.py functions
                # The TyDiQA loader returns 'answers' as a dict: {'text': ['ans1', ...], 'answer_start': [...]} 
                try:
                    all_samples_df['lrl_ground_truth_answers_list'] = all_samples_df['answers'].apply(
                        lambda x: x['text'] if isinstance(x, dict) and 'text' in x and isinstance(x['text'], list) else ([str(x['text'])] if isinstance(x, dict) and 'text' in x and isinstance(x['text'], str) else ([str(x)] if pd.notna(x) else []))
                    )
                    # Ensure all items in the list are strings
                    all_samples_df['lrl_ground_truth_answers_list'] = all_samples_df['lrl_ground_truth_answers_list'].apply(
                        lambda L: [str(item) for item in L] if isinstance(L, list) else []
                    )
                except Exception as e_gt_map:
                    logger.error(f"Error mapping ground truth 'answers' to 'lrl_ground_truth_answers_list' for {lang_code}: {e_gt_map}", exc_info=True)
                    continue # Skip if GT cannot be processed

                logger.info(f"Loaded {len(all_samples_df)} samples for {lang_code} with ground truth processing.")

            # --- Determine Effective Parameters for this run --- #
            # QA Parameters
            eff_qa_params = DEFAULT_QA_PARAMS.copy()
            eff_qa_params.update(LANG_QA_PARAM_OVERRIDES.get(lang_code, {}))
            if args.qa_temp is not None: eff_qa_params["temperature"] = args.qa_temp
            if args.qa_top_p is not None: eff_qa_params["top_p"] = args.qa_top_p
            if args.qa_top_k is not None: eff_qa_params["top_k"] = args.qa_top_k
            if args.qa_max_tokens is not None: eff_qa_params["max_new_tokens"] = args.qa_max_tokens
            if args.qa_rep_penalty is not None: eff_qa_params["repetition_penalty"] = args.qa_rep_penalty
            eff_qa_params = apply_model_specific_adjustments(eff_qa_params, model_name_str, "qa")
            eff_qa_params["do_sample"] = eff_qa_params.get("do_sample", eff_qa_params.get("temperature", 0) > 1e-5)

            # Translation Parameters
            eff_trans_params = DEFAULT_TRANSLATION_PARAMS.copy()
            eff_trans_params.update(LANG_TRANSLATION_PARAM_OVERRIDES.get(lang_code, {}))
            if args.trans_temp is not None: eff_trans_params["temperature"] = args.trans_temp
            if args.trans_top_p is not None: eff_trans_params["top_p"] = args.trans_top_p
            if args.trans_top_k is not None: eff_trans_params["top_k"] = args.trans_top_k
            if args.trans_max_tokens is not None: eff_trans_params["max_new_tokens"] = args.trans_max_tokens
            if args.trans_rep_penalty is not None: eff_trans_params["repetition_penalty"] = args.trans_rep_penalty
            eff_trans_params = apply_model_specific_adjustments(eff_trans_params, model_name_str, "translation")
            eff_trans_params["do_sample"] = eff_trans_params.get("do_sample", eff_trans_params.get("temperature", 0) > 1e-5)

            # Single-Prompt Chain Parameters (often similar to QA params but can be distinct)
            # For QA, the CLI args --qa_... are used for single_prompt chain as well.
            eff_chain_params = DEFAULT_QA_PARAMS.copy() # Start with QA defaults for the chain
            eff_chain_params.update(LANG_QA_PARAM_OVERRIDES.get(lang_code, {})) # Apply lang overrides for QA/chain
            if args.qa_temp is not None: eff_chain_params["temperature"] = args.qa_temp
            if args.qa_top_p is not None: eff_chain_params["top_p"] = args.qa_top_p
            if args.qa_top_k is not None: eff_chain_params["top_k"] = args.qa_top_k
            if args.qa_max_tokens is not None: eff_chain_params["max_new_tokens"] = args.qa_max_tokens
            if args.qa_rep_penalty is not None: eff_chain_params["repetition_penalty"] = args.qa_rep_penalty
            eff_chain_params = apply_model_specific_adjustments(eff_chain_params, model_name_str, "chain") # Use a "chain" type for model adjustments if specific needed
            eff_chain_params["do_sample"] = eff_chain_params.get("do_sample", eff_chain_params.get("temperature", 0) > 1e-5)

            for pipeline_type in args.pipeline_types:
                for shot_setting in args.shot_settings:
                    use_few_shot = (shot_setting == 'few_shot')
                    logging.info(f"\nEvaluating {model_name_str} on {lang_code} ({pipeline_type}, {shot_setting})...")

                    summary_result = run_experiment(
                        model_name=model_name_str, tokenizer=tokenizer, model=model,
                        samples_df=all_samples_df, lang_code=lang_code, base_results_path=args.base_output_dir,
                        pipeline_type=pipeline_type, use_few_shot=use_few_shot,
                        qa_params_dict=eff_qa_params,
                        trans_params_dict=eff_trans_params,
                        chain_params_dict=eff_chain_params if pipeline_type == 'single_prompt' else {},
                        overwrite_results=args.overwrite_results, max_input_length=4096
                    )
                    if summary_result:
                        all_experiment_summaries.append(summary_result)

        if model is not None: del model
        if tokenizer is not None: del tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logging.info(f"Model {model_name_str} and its tokenizer unloaded. GPU cache cleared.")

    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        overall_summary_filename = os.path.join(summaries_output_dir, f"cotr_qa_ALL_experiments_summary_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        overall_summary_df.to_csv(overall_summary_filename, index=False, float_format='%.4f')
        logging.info(f"\nOverall summary saved to: {overall_summary_filename}")
        print(overall_summary_df.to_string())
        try:
            plot_qa_metrics(overall_summary_df, plots_output_dir)
        except ImportError:
            logging.warning("matplotlib or seaborn not installed. Skipping plot generation.")
        except Exception as e_plot:
            logging.error(f"Error during plot generation: {e_plot}", exc_info=True)
    else:
        logging.info("No summaries collected. Skipping overall summary and plots.")
    logging.info("\nAll CoTR QA experiments completed!")
            
def plot_qa_metrics(summary_df, plots_dir):
    if summary_df.empty:
        logging.warning("Summary DataFrame is empty, skipping plots.")
        return

    metrics_to_plot = ['f1_score', 'avg_comet_q_lrl_en', 'avg_comet_c_lrl_en', 'avg_comet_a_en_lrl']
    for metric in metrics_to_plot:
        if metric not in summary_df.columns or summary_df[metric].isnull().all():
            logging.info(f"Metric '{metric}' not found or all NaN in summary. Skipping this plot.")
            continue
        plt.figure(figsize=(18, 10))
        try:
            # Create a unique identifier for each experiment configuration for x-axis labels
            # Ensure all components are strings and handle potential missing dict keys gracefully
            summary_df['experiment_config'] = summary_df['model'] + '-' + \
                                          summary_df['language'] + '-' + \
                                          summary_df['pipeline'] + '-' + \
                                          summary_df['shot_type'] + '-Q' + \
                                          summary_df['qa_params'].apply(lambda x: str(round(x.get('temperature', 0), 2)) if isinstance(x, dict) else 'N/A') + '-T' + \
                                          summary_df['trans_params'].apply(lambda x: str(round(x.get('temperature', 0), 2)) if isinstance(x, dict) else 'N/A')
            
            plot_data = summary_df.dropna(subset=[metric]) # Drop rows where the current metric is NaN
            if plot_data.empty:
                logging.info(f"No valid data to plot for '{metric}' after dropping NaNs.")
                continue

            sns.barplot(data=plot_data, x='experiment_config', y=metric, hue='language', dodge=False)
            plt.xticks(rotation=75, ha='right', fontsize=8)
            plt.title(f'Average {metric.replace("_", " ").title()} for CoTR QA Experiments', fontsize=14)
            plt.ylabel(f'Average {metric.replace("_", " ").title()}', fontsize=12)
            plt.xlabel('Experiment Configuration (Model-Lang-Pipeline-Shot-QATemp-TransTemp)', fontsize=10)
            plt.legend(title='Language', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout(rect=[0,0,0.88,1]) # Adjust layout to make space for legend
            plot_filename = os.path.join(plots_dir, f"cotr_qa_{metric}_scores.png")
            plt.savefig(plot_filename)
            plt.close()
            logging.info(f"{metric.replace('_', ' ').title()} plot saved to {plot_filename}")
        except Exception as e:
            logging.error(f"Error generating {metric} plot: {e}", exc_info=True)
            if plt.get_fignums(): # Check if a figure is open
                plt.close() # Ensure figure is closed on error

if __name__ == "__main__":
    main()