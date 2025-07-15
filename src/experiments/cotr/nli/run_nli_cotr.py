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
import random
import json
from datetime import datetime
import itertools
import copy

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loaders.load_xnli import load_xnli_samples
from src.experiments.cotr.nli.nli_cotr import (
    initialize_model,
    evaluate_nli_cotr,
    evaluate_nli_cotr_single_prompt,
    NLI_LABELS, # This is defined and can be used if needed by the runner
    # NLI_LABELS_MAP_TO_INT, # Not available for import
    # NLI_TRANSLATIONS # Not directly used by runner
)
from src.experiments.baseline.nli.nli_baseline import calculate_nli_metrics # ADDED IMPORT
from huggingface_hub import login
from config import get_token
# from src.utils.summarization_utils import save_overall_summary, append_to_overall_summary_and_plot # Commented out

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Define the integer to string label map directly in the runner
NLI_INT_TO_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
# Define English NLI labels for metrics, using the values from NLI_INT_TO_LABEL_MAP
NLI_LABELS_EN_FOR_METRICS = list(NLI_INT_TO_LABEL_MAP.values())

# --- Parameter Definitions from OLD SCRIPT (run_nli_cotr_old.py) ---
# These will be set by argparse. No global dicts like UNIFIED_NLI_PARAMS_CORE_COTR needed here
# if we are directly using args.

logger = logging.getLogger(__name__)

# REMOVE the get_effective_nli_cotr_params function from the current script
# def get_effective_nli_cotr_params(model_name: str, cli_args: argparse.Namespace, pipeline_type: str) -> Dict:
#    ... (entire function removed) ...

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run NLI CoTR experiments")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct", help="Model to use, comma-separated if multiple models")
    parser.add_argument("--languages", type=str, default="en,ur,sw,fr", help="Language code(s), comma-separated if multiple")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per language")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "validation", "test"], 
                        help="Dataset split to use. Default: test")
    parser.add_argument("--pipeline_types", nargs='+', default=['single_prompt', 'multi_prompt'], 
                        choices=['single_prompt', 'multi_prompt'], help="CoTR pipeline types to run.")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], 
                        choices=['zero_shot', 'few_shot'], help="Prompting strategies (zero-shot or few-shot).")
    parser.add_argument("--test_mode", action='store_true', help="Run with first 5 samples only for quick testing.")
    parser.add_argument("--hf_token", type=str, help="HuggingFace API token for model access.")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/nli/cotr_xnli", 
                        help="Base directory to save results, summaries, and plots.")
    parser.add_argument("--overwrite_results", action="store_true", 
                        help="If set, overwrite existing detailed result and summary files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--max_input_length", type=int, default=2048, 
                        help="Maximum input length for the tokenizer. Default 2048.") # From old script, seems sensible
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")

    # Generation Parameters - Common (used by multi-prompt steps and potentially overridden for single-prompt if not specified there)
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for text generation. Default: 0.3. Used for multi-prompt steps unless overridden by single-prompt specific flags.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling. Default: 0.9. Used for multi-prompt steps unless overridden by single-prompt specific flags.")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling. Default: 40. Used for multi-prompt steps unless overridden by single-prompt specific flags.")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty. Default: 1.1. Used for multi-prompt steps unless overridden by single-prompt specific flags.")
    parser.add_argument("--do_sample", type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to use sampling. Default: True. Used for multi-prompt steps unless overridden by single-prompt specific flags.")

    # Generation Parameters - Max New Tokens for Multi-Prompt Steps
    parser.add_argument("--max_tokens_text_translation", type=int, default=256, help="Max new tokens for LRL->EN text translation (premise/hypothesis). Default: 256") # Adjusted from 512 to 256
    parser.add_argument("--max_tokens_nli_processing", type=int, default=30, help="Max new tokens for English NLI processing (label generation). Default: 30") # Adjusted from 50 to 30
    parser.add_argument("--max_tokens_label_translation", type=int, default=30, help="Max new tokens for EN->LRL label translation. Default: 30") # Adjusted from 50 to 30

    # Generation Parameters - For Single-Prompt CoT Chain (can override common params if desired for the whole chain)
    # These allow distinct settings for the single prompt if needed, otherwise it might fall back to common ones or have its own defaults in nli_cotr.py
    parser.add_argument("--temperature_single_prompt", type=float, default=None, help="Override temperature for the single-prompt CoT chain. Default: Uses common --temperature.")
    parser.add_argument("--top_p_single_prompt", type=float, default=None, help="Override top-p for the single-prompt CoT chain. Default: Uses common --top_p.")
    parser.add_argument("--top_k_single_prompt", type=int, default=None, help="Override top-k for the single-prompt CoT chain. Default: Uses common --top_k.")
    parser.add_argument("--repetition_penalty_single_prompt", type=float, default=None, help="Override repetition penalty for the single-prompt CoT chain. Default: Uses common --repetition_penalty.")
    parser.add_argument("--do_sample_single_prompt", type=lambda x: (str(x).lower() == 'true'), default=None, help="Override do_sample for the single-prompt CoT chain. Default: Uses common --do_sample.")
    parser.add_argument("--max_tokens_single_prompt_chain", type=int, default=350, help="Max new tokens for the entire single-prompt CoT chain output. Default: 350") # Adjusted from 768 to 350

    # Deprecated/Old args from run_nli_cotr_old.py (commented out for now, can be re-added if logic is ported)
    # parser.add_argument("--multi_prompt_temp_translation", type=float, default=0.3)
    return parser.parse_args()

def run_nli_cotr_experiment( # Signature adapted from run_nli_cotr_old.py
    model_name_str: str, # Full model name string
    tokenizer: Any,      # Initialized tokenizer
    model: Any,          # Initialized model
    samples_df: pd.DataFrame, 
    lang_code: str, 
    base_results_path: str, # This is cotr_path from old script
    pipeline_type_to_run: str, # 'multi_prompt' or 'single_prompt'
    shot_setting_to_run: str, # 'zero_shot' or 'few_shot'
    cli_args: argparse.Namespace # Pass all parsed CLI args for parameters
):
    """
    Run a single NLI CoTR experiment configuration.
    Structure adapted from run_nli_cotr_old.py.
    """
    if samples_df.empty:
        logging.warning(f"No samples for {lang_code}, skipping experiment for {model_name_str}.")
        return None

    use_few_shot = (shot_setting_to_run == 'few_shot')
    
    # model_name_short for paths, from old script
    model_name_short = model_name_str.split('/')[-1].replace("-", "_").replace(".", "_") # Sanitize for paths

    # Directory structure from old script
    # base_results_path is equivalent to cotr_path in old script
    # pipeline_type_to_run is 'multi-prompt' or 'single-prompt'
    # shot_setting_to_run is 'zero-shot' or 'few-shot'
    
    # Path adjustments to match old script more closely
    # Results: base_path / pipeline_type / shot_name / lang_code / specific_results_file.csv
    # Summaries: base_path / summaries / pipeline_type / shot_name / specific_summary_file.csv (old script summary path was a bit different)
    # Let's follow the old structure for detailed results and a similar one for summaries.
    
    # Old script summary path: base_path / "summaries" / pipeline_type / shot_type / lang_code / summary_file
    # Old script results path: base_path / pipeline_type / shot_type / lang_code / results_file
    # The current script run_nli_cotr.py uses:
    # results_dir = os.path.join(base_path, "results", pipeline_type, shot_type_for_path, lang_code, model_short_name)
    # summaries_dir = os.path.join(base_path, "summaries", pipeline_type, shot_type_for_path, lang_code, model_short_name)
    # This is more granular with model_short_name, which is good. Let's keep this from current.

    shot_type_for_path = "fs" if use_few_shot else "zs"
    results_dir = os.path.join(base_results_path, "results", pipeline_type_to_run, shot_type_for_path, lang_code, model_name_short)
    summaries_dir = os.path.join(base_results_path, "summaries", pipeline_type_to_run, shot_type_for_path, lang_code, model_name_short)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    
    # Filename from old script (adapted slightly for clarity)
    # Old: f"{prefix}_{pipeline_suffix}_{shot_suffix}_nli_{lang_code}_{model_name_short}.csv"
    # prefix = "cotr", pipeline_suffix = "mp" or "sp", shot_suffix = "fs" or "zs"
    results_file = os.path.join(results_dir, f"results_cotr_{pipeline_type_to_run}_{shot_type_for_path}_nli_{lang_code}_{model_name_short}.csv")
    summary_file = os.path.join(summaries_dir, f"summary_cotr_{pipeline_type_to_run}_{shot_type_for_path}_nli_{lang_code}_{model_name_short}.csv")


    if os.path.exists(summary_file) and not cli_args.overwrite_results:
            try:
                existing_summary_df = pd.read_csv(summary_file)
                if not existing_summary_df.empty:  # Line ~162
                    logging.info(f"Loaded existing summary: {summary_file} (overwrite_results=False)")
                    summary_dict_reloaded = existing_summary_df.to_dict('records')[0]
                    for comet_col in ['avg_comet_lrl_prem_to_en', 'avg_comet_lrl_hyp_to_en', 'avg_comet_en_label_to_lrl']:
                        if comet_col not in summary_dict_reloaded:
                            summary_dict_reloaded[comet_col] = None
                    return summary_dict_reloaded
            except Exception as e:
                logging.warning(f"Could not load or parse existing summary {summary_file}: {e}. Will attempt to generate from results or re-run.")

    if os.path.exists(results_file) and not cli_args.overwrite_results:
        logging.info(f"Results file {results_file} exists and overwrite_results is False. Attempting to generate summary from it.")
        try:
            results_df_existing = pd.read_csv(results_file)
            # Proceed to metric calculation and summary generation (logic adapted from current script)
            if 'original_gt_label_int' in results_df_existing.columns and \
               ('ground_truth_label_str' not in results_df_existing.columns or results_df_existing['ground_truth_label_str'].isnull().any()):
                 results_df_existing['ground_truth_label_str'] = results_df_existing['original_gt_label_int'].map(NLI_INT_TO_LABEL_MAP)

            df_for_metrics_existing = results_df_existing.copy()
            df_for_metrics_existing.rename(columns={
                'ground_truth_label_str': 'gold_label',
                'predicted_label_for_accuracy': 'predicted_label'
            }, inplace=True)

            metrics = {'accuracy': 0.0, 'macro_f1': 0.0} # Simplified default
            if 'gold_label' in df_for_metrics_existing.columns and 'predicted_label' in df_for_metrics_existing.columns:
                if not df_for_metrics_existing['gold_label'].isnull().all() and not df_for_metrics_existing['predicted_label'].isnull().all():
                    metrics = calculate_nli_metrics(df_for_metrics_existing) # Using imported function
                else: logging.warning("Metrics not calculated from existing results: gold_label or predicted_label columns are all nulls.")
            else: logging.warning("Metrics not calculated from existing results: gold_label or predicted_label columns missing after rename.")
            
            avg_comet_prem = np.mean(results_df_existing['comet_lrl_prem_to_en'].dropna()) if 'comet_lrl_prem_to_en' in results_df_existing.columns and not results_df_existing['comet_lrl_prem_to_en'].dropna().empty else None
            avg_comet_hyp = np.mean(results_df_existing['comet_lrl_hyp_to_en'].dropna()) if 'comet_lrl_hyp_to_en' in results_df_existing.columns and not results_df_existing['comet_lrl_hyp_to_en'].dropna().empty else None
            avg_comet_label = np.mean(results_df_existing['comet_en_label_to_lrl'].dropna()) if 'comet_en_label_to_lrl' in results_df_existing.columns and not results_df_existing['comet_en_label_to_lrl'].dropna().empty else None
            
            # Construct generation_params_for_summary from cli_args for logging
            # This should reflect parameters relevant to the specific pipeline
            generation_params_for_summary = {
                "temperature": cli_args.temperature, "top_p": cli_args.top_p, "top_k": cli_args.top_k,
                "repetition_penalty": cli_args.repetition_penalty, "do_sample": cli_args.do_sample,
                "max_tokens_nli_label": cli_args.max_tokens_nli_processing,
                "max_tokens_translation": cli_args.max_tokens_translation,
                "max_tokens_single_prompt_chain": cli_args.max_tokens_single_prompt_chain,
            }

            current_summary_data = {
                'model': model_name_short, 'language': lang_code, 'pipeline_type': pipeline_type_to_run,
                'shot_type': shot_type_for_path, 'samples': len(results_df_existing),
                'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
                'avg_comet_lrl_prem_to_en': avg_comet_prem,
                'avg_comet_lrl_hyp_to_en': avg_comet_hyp,
                'avg_comet_en_label_to_lrl': avg_comet_label,
                **generation_params_for_summary
            }
            summary_df_new = pd.DataFrame([current_summary_data])
            summary_df_new.to_csv(summary_file, index=False, float_format='%.4f')
            logging.info(f"Summary generated from existing results and saved to {summary_file}")
            return current_summary_data
        except Exception as e:
            logging.error(f"Could not process existing results file {results_file}: {e}. Re-running experiment.")

    logging.info(f"Running NLI CoTR experiment: Model={model_name_str}, Lang={lang_code}, Pipeline={pipeline_type_to_run}, Shot={shot_setting_to_run}")
    
    # Prepare generation parameters for the core evaluation functions
    # These will be dictionaries derived from cli_args

    results_df_computed = None
    
    if pipeline_type_to_run == 'multi_prompt':
        # Parameters for multi-prompt, derived from cli_args
        # Text Translation (LRL -> EN Premise/Hypothesis) Parameters
        text_translation_step_params = {
            "temperature": cli_args.temperature, "top_p": cli_args.top_p, "top_k": cli_args.top_k,
            "repetition_penalty": cli_args.repetition_penalty, "do_sample": cli_args.do_sample,
            "max_new_tokens": cli_args.max_tokens_text_translation
        }
        # NLI Processing (EN Premise/Hypothesis -> EN Label) Parameters
        nli_eval_step_params = {
            "temperature": cli_args.temperature, "top_p": cli_args.top_p, "top_k": cli_args.top_k,
            "repetition_penalty": cli_args.repetition_penalty, "do_sample": cli_args.do_sample,
            "max_new_tokens": cli_args.max_tokens_nli_processing # Max tokens for the NLI label itself
        }
        # Label Translation (EN -> LRL Label) Parameters (only if lang_code != 'en')
        label_translation_step_params = {}
        if lang_code != 'en':
            label_translation_step_params = {
            "temperature": cli_args.temperature, "top_p": cli_args.top_p, "top_k": cli_args.top_k,
            "repetition_penalty": cli_args.repetition_penalty, "do_sample": cli_args.do_sample,
                "max_new_tokens": cli_args.max_tokens_label_translation # Often similar to text translation length
        }
        logging.info(f"  Multi-Prompt Params: NLI={nli_eval_step_params}, TextTrans={text_translation_step_params}, LabelTrans={label_translation_step_params}")
        results_df_computed = evaluate_nli_cotr( # This is the multi-prompt eval func from nli_cotr.py
            model_name=model_name_str, tokenizer=tokenizer, model=model,
            samples_df=samples_df, lang_code=lang_code, use_few_shot=use_few_shot,
            nli_params=nli_eval_step_params, # Pass the dicts
            text_translation_params=text_translation_step_params,
            label_translation_params=label_translation_step_params
        )
    elif pipeline_type_to_run == 'single_prompt':
        single_prompt_chain_params = {
            "temperature": cli_args.temperature, "top_p": cli_args.top_p, "top_k": cli_args.top_k,
            "repetition_penalty": cli_args.repetition_penalty, "do_sample": cli_args.do_sample,
            "max_tokens": cli_args.max_tokens_single_prompt_chain # Max tokens for the entire CoT output
        }
        logging.info(f"  Single-Prompt Params: Chain={single_prompt_chain_params}")
        results_df_computed = evaluate_nli_cotr_single_prompt( # This is the single-prompt eval func from nli_cotr.py
            model_name=model_name_str, tokenizer=tokenizer, model=model,
            samples_df=samples_df, lang_code=lang_code, use_few_shot=use_few_shot,
            # Pass individual params from the prepared dict
            temperature=single_prompt_chain_params["temperature"],
            top_p=single_prompt_chain_params["top_p"],
            top_k=single_prompt_chain_params["top_k"],
            max_tokens=single_prompt_chain_params["max_tokens"],
            repetition_penalty=single_prompt_chain_params["repetition_penalty"],
            do_sample=single_prompt_chain_params["do_sample"]
            )
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type_to_run}")
    
    if results_df_computed is None or results_df_computed.empty:
        logging.warning(f"No results returned from evaluation for {model_name_str}, {lang_code}, {pipeline_type_to_run}, {shot_setting_to_run}")
        return None

    results_df_computed.to_csv(results_file, index=False) # Line ~280
    logging.info(f"Results saved to {results_file}")       # Line ~281 - Ensure this line starts at the same column as the one above

    # Prepare DataFrame for metrics (consistent with how existing results are handled)
    if 'original_gt_label_int' in results_df_computed.columns and \
       ('ground_truth_label_str' not in results_df_computed.columns or results_df_computed['ground_truth_label_str'].isnull().any()):
        results_df_computed['ground_truth_label_str'] = results_df_computed['original_gt_label_int'].map(NLI_INT_TO_LABEL_MAP)
    
    df_for_metrics_computed = results_df_computed.copy()

    # Rename columns to match what calculate_nli_metrics expects
    # Expected: 'premise', 'hypothesis', 'gold_label', 'predicted_label'
    rename_map = {}
    if 'premise_en' in df_for_metrics_computed.columns:
        rename_map['premise_en'] = 'premise'
    elif 'premise_lrl' in df_for_metrics_computed.columns: # Fallback if EN not generated due to error
        rename_map['premise_lrl'] = 'premise'
    
    if 'hypothesis_en' in df_for_metrics_computed.columns:
        rename_map['hypothesis_en'] = 'hypothesis'
    elif 'hypothesis_lrl' in df_for_metrics_computed.columns: # Fallback
        rename_map['hypothesis_lrl'] = 'hypothesis'

    if 'ground_truth_eng_label_str' in df_for_metrics_computed.columns:
        rename_map['ground_truth_eng_label_str'] = 'gold_label'
    
    if 'predicted_label_for_accuracy' in df_for_metrics_computed.columns: # This is the English predicted label
        rename_map['predicted_label_for_accuracy'] = 'predicted_label'
    
    df_for_metrics_computed = df_for_metrics_computed.rename(columns=rename_map)

    # Ensure all expected columns are present after renaming, or skip metrics
    expected_cols_for_metrics = ['premise', 'hypothesis', 'gold_label', 'predicted_label']
    if not all(col in df_for_metrics_computed.columns for col in expected_cols_for_metrics):
        logger.error(f"Metric calculation skipped: DataFrame is missing one or more expected columns after renaming: {expected_cols_for_metrics}. Available: {df_for_metrics_computed.columns.tolist()}")
        # Create a dummy metrics dict to avoid downstream errors if summary still expects these keys
        metrics = {
            'accuracy': 0.0, 'macro_f1': 0.0, 'weighted_f1': 0.0, 
            'report_dict': {lbl: {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0} for lbl in NLI_LABELS_EN_FOR_METRICS}
        }
    else:
        logger.info(f"Columns for metric calculation: {df_for_metrics_computed.columns.tolist()}")
        try:
            metrics = calculate_nli_metrics(df_for_metrics_computed) # Using imported function
        except Exception as e_metrics_calc:
            logger.error(f"Error during calculate_nli_metrics: {e_metrics_calc}", exc_info=True)
            metrics = {
            'accuracy': 0.0, 'macro_f1': 0.0, 'weighted_f1': 0.0, 
            'report_dict': {lbl: {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0} for lbl in NLI_LABELS_EN_FOR_METRICS}
        }

    # Prepare summary_data dictionary (aligned with old script + new additions)

    # Define generation_params_for_summary for logging
    generation_params_for_summary = {}
    if pipeline_type_to_run == 'multi_prompt':
        generation_params_for_summary.update({f"text_trans_{k}": v for k,v in text_translation_step_params.items()})
        generation_params_for_summary.update({f"nli_step_{k}": v for k,v in nli_eval_step_params.items()})
        if lang_code != 'en': # label translation params only relevant if lang is not English
            generation_params_for_summary.update({f"label_trans_{k}": v for k,v in label_translation_step_params.items()})
    else: # single_prompt, params are directly from cli_args for the single chain
        generation_params_for_summary.update({
            f"single_chain_temp": cli_args.temperature, 
            f"single_chain_top_p": cli_args.top_p,
            f"single_chain_top_k": cli_args.top_k,
            f"single_chain_rep_pen": cli_args.repetition_penalty,
            f"single_chain_max_tok": cli_args.max_tokens_single_prompt_chain,
            f"single_chain_do_sample": cli_args.do_sample
        })

    summary_data = {
        'model': model_name_short, 'language': lang_code, 'pipeline_type': pipeline_type_to_run,
            'shot_type': shot_type_for_path, 'samples': len(results_df_computed),
            'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
        **generation_params_for_summary
    }
    # Adding class metrics to summary if available (from old script style)
    if 'class_metrics' in metrics:
        for label_val in NLI_LABELS_EN_FOR_METRICS: # ["entailment", "neutral", "contradiction"]
            if label_val in metrics['class_metrics']:
                summary_data[f'{label_val}_precision'] = metrics['class_metrics'][label_val].get('precision', 0.0)
                summary_data[f'{label_val}_recall'] = metrics['class_metrics'][label_val].get('recall', 0.0)
                summary_data[f'{label_val}_f1'] = metrics['class_metrics'][label_val].get('f1-score', 0.0) # Note: key is 'f1-score' in sklearn report

    # Save individual summary
    summary_df = pd.DataFrame([summary_data])
    model_short_name = model_name_str.split('/')[-1]

    # Define the directory path for this specific run's summary
    summary_dir = os.path.join(
        base_results_path,
        "summaries",
        pipeline_type_to_run, # e.g., 'multi_prompt' or 'single_prompt'
        shot_setting_to_run,  # e.g., 'zero_shot' or 'few_shot'
        lang_code,
        model_short_name
    )
    os.makedirs(summary_dir, exist_ok=True)

    summary_file = os.path.join(summary_dir, f"summary_{pipeline_type_to_run}_{shot_setting_to_run}_{lang_code}_{model_short_name}.csv")
    try:
        summary_df.to_csv(summary_file, index=False, float_format='%.4f')
        logging.info(f"Summary saved to {summary_file}") # This line should be aligned with the line above
    except Exception as e_save:
        logger.error(f"Error saving summary file {summary_file}: {e_save}")
        return None # Indicate failure if summary can't be saved

    return summary_data


def plot_nli_metrics(summary_df, plots_dir, metric_col, metric_name, task_name="NLI CoTR"):
    if summary_df.empty or metric_col not in summary_df.columns:
        print(f"Summary DataFrame is empty or missing '{metric_col}', skipping {metric_name} plot.")
        return
    plt.figure(figsize=(18, 10)) # Increased figure size from new script
    try:
        # Create a unique configuration identifier for x-axis labels (from new script)
        if 'experiment_config' not in summary_df.columns:
             summary_df['experiment_config'] = summary_df['language'] + '-' + summary_df['model'] + '-' + summary_df['pipeline_type'] + '-' + summary_df['shot_type']
        
        plot_df = summary_df.dropna(subset=[metric_col])
        if plot_df.empty:
            print(f"No valid data to plot for {metric_name} after dropping NaNs from column '{metric_col}'.")
            plt.close()
            return

        # Using sns.barplot from new script's plotting for consistency if desired, or simpler plot from old
        # Old script used sns.catplot for more complex faceting. Let's simplify for now or use new script's barplot.
        # For simplicity, using a direct barplot:
        sns.barplot(data=plot_df, x='experiment_config', y=metric_col, hue='language', dodge=False)
        plt.xticks(rotation=60, ha='right', fontsize=10) # Rotated more, from new script
        plt.yticks(fontsize=10)
        plt.title(f'Average {metric_name} for {task_name} Experiments', fontsize=16)
        plt.ylabel(f'Average {metric_name}', fontsize=12)
        plt.xlabel('Experiment Configuration (Lang-Model-Pipeline-Shot)', fontsize=12) # More descriptive X label
        plt.legend(title='Language', bbox_to_anchor=(1.02, 1), loc='upper left') # Better legend placement
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout for legend
        
        safe_metric_col_name = metric_col.replace('avg_comet_', '').replace('->', '_To_') # Sanitize
        plot_filename = os.path.join(plots_dir, f"cotr_nli_{safe_metric_col_name}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"{metric_name} plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error generating {metric_name} plot: {e}", exc_info=True)
        if plt.get_fignums(): plt.close()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # HF Login
    if args.hf_token:
        token = args.hf_token
    else:
        token = get_token() # Call without arguments

    if token:
            login(token=token)
    else:
        logger.warning("Hugging Face token not provided. Downloads may be restricted for some models.")

    models_list = [m.strip() for m in args.models.split(',')]
    lang_list = [l.strip() for l in args.languages.split(',')]
    all_experiment_summaries = [] # From old script, for collecting all summaries

    # Define overall summary and plots directories (from old script structure)
    overall_summary_base_dir = os.path.join(args.base_output_dir, "summaries_overall") # General summary location
    os.makedirs(overall_summary_base_dir, exist_ok=True)
    overall_plots_dir = os.path.join(args.base_output_dir, "plots_overall") # General plots location
    os.makedirs(overall_plots_dir, exist_ok=True)

    logger.info(f"All NLI CoTR experiment outputs will be saved under: {args.base_output_dir}")
    logger.info(f"Individual run summaries in: {args.base_output_dir}/summaries/[pipeline]/[shot]") # Per-run summaries
    logger.info(f"Overall summary in: {overall_summary_base_dir}")
    logger.info(f"Overall plots in: {overall_plots_dir}")

    # Use args.seed for sampling reproducibility (from old script)
    sampling_seed = args.seed
    # Use args.num_samples for fixed number of samples per language (from old script)
    num_samples_to_load = args.num_samples

    for model_name_str in models_list:
        logger.info(f"\n========== Initializing Model: {model_name_str} ==========")
        tokenizer, model = None, None # Initialize to allow cleanup
        try:
            tokenizer, model = initialize_model(model_name_str) # from nli_cotr.py
            logging.info(f"Model {model_name_str} initialized successfully.")
        except Exception as e_init:
            logging.error(f"Failed to initialize model {model_name_str}: {e_init}. Skipping this model.", exc_info=True)
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue # Skip to the next model

        for lang_code in lang_list: # Iterate using lang_list from args (old script style)
            logger.info(f"\n--- Processing Language: {lang_code} for model {model_name_str} ---")

            # Load XNLI samples using the utility, now with num_samples and seed
            logger.info(f"Loading XNLI samples for {lang_code} ({args.data_split} split), num_samples={num_samples_to_load}, seed={sampling_seed}")
            samples_df_for_lang = load_xnli_samples(
                lang_code=lang_code, 
                num_samples=num_samples_to_load, 
                split=args.data_split, 
                seed=sampling_seed
            )

            if samples_df_for_lang.empty:
                logging.warning(f"No XNLI samples loaded for {lang_code} ('{args.data_split}' split, {args.num_samples} samples). Skipping.")
                continue
            
            if args.test_mode:
                logging.info(f"Test mode: Using first 5 samples for {lang_code}.")
                samples_df_for_lang = samples_df_for_lang.head(5)
                if samples_df_for_lang.empty:
                    logging.warning(f"No samples remaining for {lang_code} after test_mode filtering. Skipping.")
                    continue

            # Ensure 'label' column exists and is string (load_xnli_samples should provide this as 'label')
            # Also ensure 'original_gt_label_int' is present for mapping if needed.
            if 'label' not in samples_df_for_lang.columns or 'original_label_int' not in samples_df_for_lang.columns:
                logging.error(f"CRITICAL: 'label' or 'original_label_int' column missing from loaded XNLI samples for {lang_code}.")
                continue

            for pipeline_to_run in ['multi_prompt', 'single_prompt']:
                for shot_setting_to_run in ['zero_shot', 'few_shot']:
                    summary_data = run_nli_cotr_experiment(
                        model_name_str, tokenizer, model, samples_df_for_lang, lang_code,
                        args.base_output_dir, # This is the main output dir for individual results/summaries
                        pipeline_to_run, shot_setting_to_run, args
                    )
                    if summary_data:
                        all_experiment_summaries.append(summary_data)
        
        logging.info(f"Finished all experiments for model {model_name_str}. Unloading...")
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logging.info(f"Model {model_name_str} unloaded and CUDA cache cleared (if applicable).")

    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        if not overall_summary_df.empty:
            # Ensure all columns that should be numeric are, coercing errors for robustness.
            for col in ['accuracy', 'macro_f1', 'weighted_f1', 'comet_premise_lrl_en', 
                        'comet_hypothesis_lrl_en', 'comet_label_en_lrl', 'runtime_total_s', 'runtime_per_sample_s']:
                if col in overall_summary_df.columns:
                    overall_summary_df[col] = pd.to_numeric(overall_summary_df[col], errors='coerce')

            summary_filename_overall = f"cotr_nli_ALL_experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            overall_summary_path = os.path.join(overall_summary_base_dir, summary_filename_overall)
            overall_summary_df.to_csv(overall_summary_path, index=False, float_format='%.4f')
            logger.info(f"\nOverall summary of NLI CoTR experiments saved to: {overall_summary_path}")
            logger.info(f"Overall Summary:\n{overall_summary_df.to_string()}")

            # Plotting (from old script)
            plot_nli_metrics(overall_summary_df, overall_plots_dir, 'accuracy', 'Accuracy')
            plot_nli_metrics(overall_summary_df, overall_plots_dir, 'macro_f1', 'Macro F1-Score')
            plot_nli_metrics(overall_summary_df, overall_plots_dir, 'weighted_f1', 'Weighted F1-Score')
            # Add COMET plots if columns exist and are numeric
            if 'comet_premise_lrl_en' in overall_summary_df.columns: # Name from old script
                plot_nli_metrics(overall_summary_df, overall_plots_dir, 'comet_premise_lrl_en', 'COMET Premise LRL->EN')
            if 'comet_hypothesis_lrl_en' in overall_summary_df.columns: # Name from old script
                plot_nli_metrics(overall_summary_df, overall_plots_dir, 'comet_hypothesis_lrl_en', 'COMET Hypothesis LRL->EN')
            if 'comet_label_en_lrl' in overall_summary_df.columns: # Name from old script
                plot_nli_metrics(overall_summary_df, overall_plots_dir, 'comet_label_en_lrl', 'COMET Label EN->LRL')
            logger.info(f"Overall plots saved to: {overall_plots_dir}")
        else:
            logger.info("Overall summary DataFrame is empty. No plots generated.")
    else:
        logger.info("No summaries collected. Skipping overall summary and plot generation for NLI CoTR.")

    logger.info("\nAll NLI CoTR experiments completed!")

if __name__ == "__main__":
    # Setup logging (basicConfig should be called early, before any logging commands)
    # The level will be set by args later, but good to have a basic config.
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main() 