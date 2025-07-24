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
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, f1_score

# Fügt das Projekt-Stammverzeichnis zum Python-Pfad hinzu, BEVOR andere Projekt-Importe erfolgen
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Deaktiviert die Tokenizer-Parallelität, um Warnungen zu vermeiden
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Importiert CoTR-Sentiment-Funktionen
from src.experiments.cotr.sentiment.sentiment_cotr import (
    initialize_model,
    evaluate_sentiment_cotr_multi_prompt,
    evaluate_sentiment_cotr_single_prompt,
    SENTIMENT_LABELS_EN, # Import für Konsistenz
    LANG_NAMES as LANG_NAMES_SENTIMENT 
)

from src.utils.data_loaders.load_afrisenti import load_afrisenti_samples 
from datasets import load_dataset as hf_load_dataset, get_dataset_split_names 

# Importiert die Metrikberechnung
from evaluation.sentiment_metrics import calculate_sentiment_metrics

# Hugging Face Login
from huggingface_hub import login
from config import get_token


SENTIMENT_LABELS_LRL = {
    "sw": {"Positive": "Chanya", "Negative": "Hasi", "Neutral": "Kati"},
    "ha": {"Positive": "Tabbatacce", "Negative": "Korau", "Neutral": "Tsaka-tsaki"}, 
    "yo": {"Positive": "Rere", "Negative": "Buburu", "Neutral": "Dede"}, 
    "am": {"Positive": "አዎንታዊ", "Negative": "አሉታዊ", "Neutral": "ገለልተኛ"}, 
    "pcm": {"Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral"}, 
}


STANDARD_PARAMETERS = {
    "text_translation": {"temperature": 0.5, "top_p": 0.9, "top_k": 40, "max_new_tokens": 512, "repetition_penalty": 1.05},
    "sentiment_classification": {"temperature": 0.2, "top_p": 0.9, "top_k": 30, "max_new_tokens": 10, "repetition_penalty": 1.1},
    "label_translation": {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_new_tokens": 20, "repetition_penalty": 1.0},
    "single_prompt_cotr": {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_new_tokens": 150, "repetition_penalty": 1.1}
}

LANGUAGE_PARAMETERS = {
    "sw": {
        "text_translation": {"temperature": 0.45, "max_new_tokens": 400},
        "sentiment_classification": {"temperature": 0.15, "max_new_tokens": 8},
        "label_translation": {"temperature": 0.25, "max_new_tokens": 15},
        "single_prompt_cotr": {"temperature": 0.25, "max_new_tokens": 120}
    },
    "ha": {
        "text_translation": {"temperature": 0.45, "max_new_tokens": 400},
        "sentiment_classification": {"temperature": 0.15, "max_new_tokens": 8},
        "label_translation": {"temperature": 0.25, "max_new_tokens": 15},
        "single_prompt_cotr": {"temperature": 0.25, "max_new_tokens": 120}
    },
}

MODEL_ADJUSTMENTS = {
    "aya": {
        "text_translation": {"temperature_factor": 0.9},
        "sentiment_classification": {"temperature_factor": 0.85},
        "label_translation": {"temperature_factor": 0.9},
        "single_prompt_cotr": {"temperature_factor": 0.9}
    },
    "qwen": {
        "text_translation": {"top_p_factor": 0.95, "top_k_set": 35},
        "sentiment_classification": {"top_p_factor": 0.9, "top_k_set": 25},
        "label_translation": {"top_p_factor": 0.95, "top_k_set": 35},
        "single_prompt_cotr": {"top_p_factor": 0.95, "top_k_set": 35}
    }
}

AFRISENTI_LANGS = ['am', 'dz', 'ha', 'ig', 'ma', 'pcm', 'pt', 'sw', 'yo', 'multi'] 


def get_effective_params(step_name: str, lang_code: str, model_name_str: str, cli_args: argparse.Namespace) -> Dict:
    """Determines the effective generation parameters for a specific step."""
    base_params = STANDARD_PARAMETERS.get(step_name, {}).copy()
    # Correct identification of the model short name for MODEL_ADJUSTMENTS
    model_short_name = model_name_str.split('/')[-1].split('-')[0].lower() # e.g., "aya" or "qwen"
    lang_params = LANGUAGE_PARAMETERS.get(lang_code, {}).get(step_name, {}).copy()
    model_adjust = MODEL_ADJUSTMENTS.get(model_short_name, {}).get(step_name, {}).copy()

    # Begin with standard, override with language-specific, then apply model adjustments
    effective_params = {**base_params, **lang_params}

    # Apply model-specific multiplicative factors or direct settings
    if 'temperature_factor' in model_adjust:
        effective_params["temperature"] *= model_adjust['temperature_factor']
    if 'top_p_factor' in model_adjust:
        effective_params["top_p"] *= model_adjust['top_p_factor']
    if 'top_k_set' in model_adjust:
        effective_params["top_k"] = model_adjust['top_k_set']
    
    
    if cli_args.top_p is not None: effective_params["top_p"] = cli_args.top_p
    if cli_args.top_k is not None: effective_params["top_k"] = cli_args.top_k
    if cli_args.max_sentiment_tokens is not None and step_name == "sentiment_classification":
        effective_params["max_new_tokens"] = cli_args.max_sentiment_tokens
    elif step_name == "label_translation" and cli_args.max_label_trans_tokens is not None:
        effective_params["max_new_tokens"] = cli_args.max_label_trans_tokens
    elif step_name == "single_prompt_cotr" and cli_args.max_single_prompt_tokens is not None:
        effective_params["max_new_tokens"] = cli_args.max_single_prompt_tokens
    if cli_args.max_text_trans_tokens is not None and step_name == "text_translation":
        effective_params["max_new_tokens"] = cli_args.max_text_trans_tokens
    if cli_args.repetition_penalty is not None: effective_params["repetition_penalty"] = cli_args.repetition_penalty
    
    # Ensure do_sample is consistent with temperature
    effective_params["do_sample"] = True if effective_params.get("temperature", 0) > 0.01 else False

    return effective_params

def run_single_experiment_config(
    model_name_str: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    pipeline_type: str,
    use_few_shot: bool,
    base_output_dir: str,
    generation_params: Dict,
    overwrite_results: bool
) -> Optional[Dict[str, Any]]:
    
    if samples_df.empty:
        print(f"No samples for {lang_code}, skipping {model_name_str} for {pipeline_type}/{'fs' if use_few_shot else 'zs'}.")
        return None

    shot_type_str = "fs" if use_few_shot else "zs"
    pipeline_short = "mp" if pipeline_type == "multi_prompt" else "sp"
    model_short_name = model_name_str.split('/')[-1]

    results_subdir = os.path.join(base_output_dir, "results", pipeline_type, shot_type_str, lang_code, model_short_name)
    summaries_subdir = os.path.join(base_output_dir, "summaries", pipeline_type, shot_type_str, lang_code, model_short_name)
    os.makedirs(results_subdir, exist_ok=True)
    os.makedirs(summaries_subdir, exist_ok=True)

    results_file = os.path.join(results_subdir, f"results_sentiment_cotr_{pipeline_short}_{shot_type_str}_{lang_code}_{model_short_name}.csv")
    summary_file = os.path.join(summaries_subdir, f"summary_sentiment_cotr_{pipeline_short}_{shot_type_str}_{lang_code}_{model_short_name}.csv")

    results_df_loaded_from_file = False
    if not overwrite_results and os.path.exists(results_file):
        print(f"Results file {results_file} exists. Attempting to load summary or results.")
        if os.path.exists(summary_file):
            try:
                summary_df_existing = pd.read_csv(summary_file)
                if not summary_df_existing.empty:
                    print(f"Loaded existing summary: {summary_file}")
                    summary_dict_reloaded = summary_df_existing.to_dict('records')[0]

                    return summary_dict_reloaded
            except Exception as e:
                print(f"Could not load/parse summary {summary_file}: {e}. Will try to use results file.")
        
        try: 
            results_df = pd.read_csv(results_file)
            results_df_loaded_from_file = True
            print(f"Successfully loaded results from existing file: {results_file}")
        except Exception as e:
            print(f"Could not read results file {results_file} ({e}). Will run experiment.")
            results_df_loaded_from_file = False # Explizit zurücksetzen
    
    if not results_df_loaded_from_file:
        print(f"Running Sentiment CoTR: M={model_name_str}, L={lang_code}, Pipe={pipeline_type}, Shot={shot_type_str}")
        print(f"  Effective Params: {generation_params}")

        eval_params = {
            "model_name": model_name_str, "model": model, "tokenizer": tokenizer,
            "samples_df": samples_df, "lang_code": lang_code, "use_few_shot": use_few_shot,
            **generation_params
        }

        if pipeline_type == 'multi_prompt':
            results_df = evaluate_sentiment_cotr_multi_prompt(**eval_params)
        elif pipeline_type == 'single_prompt':
            results_df = evaluate_sentiment_cotr_single_prompt(**eval_params)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

        if results_df.empty:
            print(f"No results returned from {pipeline_type} CoTR evaluation for {lang_code}, {shot_type_str}.")
            return None
            
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")

    # AfriSenti provides labels directly in English and lowercase (e.g., "positive", "negative")
    # The 'ground_truth_english_label' in the CoTR evaluation functions expects this already.
    # If the loaded 'label' column is not already the English GT, it needs to be mapped.
    # We assume that 'label' from load_afrisenti_samples is the English GT.
    if 'label' in results_df.columns:
        results_df['ground_truth_english_label'] = results_df['label'].astype(str).str.lower().str.strip()
    else:
        # This case should ideally not occur if data loading is correct.
        # Fallback or error if the 'label' column is missing in the CoTR output DF.
        # The CoTR functions save 'ground_truth_english_label' based on the 'label' input from samples_df.
        # Therefore, this reprocessing might be redundant if the CoTR functions handle it,
        # but good for verification if results_df is loaded from a file.
        if 'ground_truth_english_label' not in results_df.columns:
            print(f"CRITICAL: 'label' column missing from initial CoTR output and 'ground_truth_english_label' not set. Metrics will be incorrect.")
            results_df['ground_truth_english_label'] = "unknown" # Placeholder

    if pipeline_type == 'multi_prompt':
        predicted_label_col = 'predicted_en_label'
    else: 
        predicted_label_col = 'intermediate_en_label'

    if predicted_label_col not in results_df.columns:
        print(f"Warning: Predicted label column '{predicted_label_col}' not found in results. Metrics will be 0.")
        avg_accuracy, avg_macro_f1, avg_weighted_f1 = 0.0, 0.0, 0.0
    else:
        y_true_clean = results_df['ground_truth_english_label'].fillna("Unknown").astype(str).tolist()
        y_pred_clean = results_df[predicted_label_col].fillna("Unknown").astype(str).tolist()
        
        valid_labels_for_f1 = [l for l in SENTIMENT_LABELS_EN if l in y_true_clean or l in y_pred_clean]

        try:
            avg_accuracy = accuracy_score(y_true_clean, y_pred_clean)
            avg_macro_f1 = f1_score(y_true_clean, y_pred_clean, labels=valid_labels_for_f1, average='macro', zero_division=0)
            avg_weighted_f1 = f1_score(y_true_clean, y_pred_clean, labels=valid_labels_for_f1, average='weighted', zero_division=0)
        except Exception as e_metrics:
            print(f"Error calculating sklearn metrics: {e_metrics}. Setting metrics to 0.")
            avg_accuracy, avg_macro_f1, avg_weighted_f1 = 0.0, 0.0, 0.0


    
    # Calculate metrics
    sentiment_metrics = calculate_sentiment_metrics(results_df)
    logging.info(f"Results: Accuracy: {sentiment_metrics['accuracy']:.4f}, Macro F1: {sentiment_metrics['macro_f1']:.4f}")
    
    summary_data = {
        'model': model_name_str.split('/')[-1], 'language': lang_code, 'pipeline': pipeline_type, 'shot_type': shot_type_str,
        'samples': len(results_df), 'accuracy': sentiment_metrics['accuracy'], 'macro_f1': sentiment_metrics['macro_f1'],
        'weighted_f1': sentiment_metrics['weighted_f1'],  # Add Weighted F1 for completeness
        'generation_params': generation_params
    }
    
    pd.DataFrame([summary_data]).to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")
    print(f"  Metrics: Acc={sentiment_metrics['accuracy']:.4f}, Macro F1={sentiment_metrics['macro_f1']:.4f}, Weighted F1={sentiment_metrics['weighted_f1']:.4f}")
    
    return summary_data

def plot_sentiment_metrics(summary_df: pd.DataFrame, plots_dir: str, metric_col: str, metric_name: str):
    if summary_df.empty or metric_col not in summary_df.columns or summary_df[metric_col].dropna().empty:
        print(f"Not enough data to plot {metric_name}. Skipping plot.")
        return

    plt.figure(figsize=(18, 10))
    summary_df['config_id'] = summary_df['language'] + '_' + summary_df['model'] + '_' + \
                              summary_df['pipeline'] + '_' + summary_df['shot_type']
    
    sns.barplot(data=summary_df.dropna(subset=[metric_col]), x='config_id', y=metric_col, hue='language', dodge=False)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=10)
    plt.title(f'Sentiment CoTR: Average {metric_name}', fontsize=16)
    plt.ylabel(f'Average {metric_name}', fontsize=12)
    plt.xlabel('Experiment Configuration (Language_Model_Pipeline_Shot)', fontsize=12)
    plt.legend(title='Language', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    plot_filename = os.path.join(plots_dir, f"sentiment_cotr_avg_{metric_col.replace('avg_', '')}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"{metric_name} plot saved to {plot_filename}")

def main():
    parser = argparse.ArgumentParser(description="Run Sentiment Analysis CoTR experiments with detailed parameter control.")
    
    # Core settings
    parser.add_argument("--models", type=str, default="CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated model names from Hugging Face.")
    parser.add_argument("--languages", type=str, default="sw,ha,pt", help="Comma-separated language codes for AfriSenti (e.g., sw, ha, pt).")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per language. Use a small number for testing.")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split.")
    parser.add_argument("--pipeline_types", nargs='+', default=['multi_prompt', 'single_prompt'], choices=['multi_prompt', 'single_prompt'])
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Shot settings to evaluate.")
    
    parser.add_argument("--temperature", type=float, default=None, help="Global temperature for generation.")
    parser.add_argument("--top_p", type=float, default=None, help="Global top-p for generation.")
    parser.add_argument("--top_k", type=int, default=None, help="Global top-k for generation.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Global repetition penalty.")
    
    parser.add_argument("--max_sentiment_tokens", type=int, default=None, help="Maximum tokens for sentiment classification step.")
    parser.add_argument("--max_text_trans_tokens", type=int, default=None, help="Maximum tokens for LRL text to English translation.")
    parser.add_argument("--max_label_trans_tokens", type=int, default=None, help="Maximum tokens for English label to LRL translation.")
    parser.add_argument("--max_single_prompt_tokens", type=int, default=None, help="Maximum tokens for the entire single CoTR generation.")

    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for closed models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--test_mode", action='store_true', help="Run with a small subset of data (5 samples).")
    parser.add_argument("--overwrite_results", action='store_true', help="Overwrite existing results and summary files.")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/sentiment/cotr_afrisenti", help="Base directory to save results and summaries.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.hf_token:
        login(token=args.hf_token)
    else:
        login(token=get_token())
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    models_list = [m.strip() for m in args.models.split(',')]
    langs_list = [l.strip() for l in args.languages.split(',')]

    all_summaries = []
    
    for model_name_str in models_list:
        logging.info(f"\n===== Initializing model: {model_name_str} =====")
        model_initialized = False
        try:
            tokenizer, model = initialize_model(model_name_str)
            model_initialized = True
            logging.info(f"Model {model_name_str} successfully initialized.")
        except Exception as e_init:
            logging.error(f"Model {model_name_str} could not be initialized: {e_init}. Skipping model.")
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        for lang_code in langs_list:
            logging.info(f"\n--- Loading data for {lang_code} (Sentiment CoTR) ---")
            
            target_num_samples = args.num_samples
            if args.test_mode: # This line was moved
                logging.info(f"    TEST MODE: Overriding sample count to 5 for {lang_code}.") # This line was moved
                target_num_samples = 5 # This line was moved
            
            # Data split is passed to load_afrisenti_samples
            samples_df_for_lang = load_afrisenti_samples(
                lang_code=lang_code, 
                split=args.data_split, 
                num_samples=target_num_samples, 
                seed=args.seed
            ) 

            if samples_df_for_lang.empty:
                logging.warning(f"No samples for {lang_code} after attempting to load data/samples for split '{args.data_split}'. Skipping model {model_name_str}.")
                continue
            
            if 'label' in samples_df_for_lang.columns:
                samples_df_for_lang['label'] = samples_df_for_lang['label'].astype(str).str.lower()

            for pipeline_type in args.pipeline_types:
                for shot_setting in args.shot_settings:
                    use_few_shot = (shot_setting == 'few_shot')
                    logging.info(f"Running: Model={model_name_str}, Language={lang_code}, Pipeline={pipeline_type}, Shot={shot_setting}")
                    
                    if pipeline_type == "multi_prompt":
                        text_trans_params = get_effective_params("text_translation", lang_code, model_name_str, args)
                        sentiment_params = get_effective_params("sentiment_classification", lang_code, model_name_str, args)
                        label_trans_params = get_effective_params("label_translation", lang_code, model_name_str, args)
                        current_generation_params = {
                            "text_translation_params": text_trans_params,
                            "sentiment_classification_params": sentiment_params,
                            "label_translation_params": label_trans_params
                        }
                    else: 
                        current_generation_params = get_effective_params("single_prompt_cotr", lang_code, model_name_str, args)

                    summary = run_single_experiment_config(
                        model_name_str, tokenizer, model, samples_df_for_lang, 
                        lang_code, pipeline_type, use_few_shot, 
                        args.base_output_dir, current_generation_params, args.overwrite_results
                    )
                    if summary:
                        all_summaries.append(summary)
        
        if model_initialized:
            logging.info(f"===== All experiments for model {model_name_str} completed. Unloading... =====")
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("GPU memory cache cleared.")

    # Preparation for overall summary and plotting
    if all_summaries:
        overall_summary_df = pd.DataFrame(all_summaries)
        if not overall_summary_df.empty:
            # Save overall summary
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            overall_summary_filename = os.path.join(args.base_output_dir, "summaries", f"sentiment_cotr_ALL_experiments_summary_{timestamp}.csv")
            # Ensure directory exists
            os.makedirs(os.path.join(args.base_output_dir, "summaries"), exist_ok=True)
            overall_summary_df.to_csv(overall_summary_filename, index=False, float_format='%.4f')
            logging.info(f"\nOverall Sentiment CoTR experiment summary saved to: {overall_summary_filename}")
            print(overall_summary_df.to_string(max_rows=None, max_cols=None)) # Print full summary

            # Generate and save plots
            plots_output_dir = os.path.join(args.base_output_dir, "plots")
            os.makedirs(plots_output_dir, exist_ok=True)
            plot_sentiment_metrics(overall_summary_df, plots_output_dir, 'accuracy', 'Accuracy')
            plot_sentiment_metrics(overall_summary_df, plots_output_dir, 'macro_f1', 'Macro F1-Score')
            plot_sentiment_metrics(overall_summary_df, plots_output_dir, 'weighted_f1', 'Weighted F1-Score')
        else:
            logging.info("Overall Sentiment CoTR summary DataFrame is empty. No plots generated.")
    else:
        logging.info("No summaries collected for Sentiment CoTR. Skipping overall summary and plot generation.")

    logging.info("\nAll Sentiment CoTR experiments completed!")

if __name__ == "__main__":
    main() 