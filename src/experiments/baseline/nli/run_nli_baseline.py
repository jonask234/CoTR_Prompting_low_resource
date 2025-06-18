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
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, classification_report
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to Python path BEFORE other project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.baseline.nli.nli_baseline import (
    initialize_model,
    evaluate_nli_baseline,
    EXPECTED_NLI_LABELS, # For metrics calculation
    NLI_LABEL_MAP_FROM_NUMERIC # For data loading if labels are numeric
)
# Import the actual XNLI data loader
from src.utils.data_loaders.load_xnli import load_xnli_samples

# Hugging Face Login (if needed, adapt from other scripts)
# from huggingface_hub import login
# from config import get_token

# --- Default Generation Parameters for NLI Baseline ---
# NLI typically requires concise, specific answers.
# Low temperature and small max_tokens are often preferred.
STANDARD_NLI_PARAMETERS = {
    "temperature": 0.1,  # Lower for more deterministic output
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 10,    # Small, expecting just one word label
    "repetition_penalty": 1.0, # Usually 1.0 for NLI is fine
    "do_sample": False   # Often greedy decoding (do_sample=False) is good for NLI
}

# Language-specific overrides can be added here if needed
LANGUAGE_NLI_PARAMETERS = {
    "sw": {"temperature": 0.15, "max_tokens": 10},
    "ur": {"temperature": 0.15, "max_tokens": 10},
    # Add other LRLs like 'ha' if you plan to run NLI for them
}

# Model-specific adjustments (e.g., slightly higher temp for more expressive models)
MODEL_NLI_ADJUSTMENTS = {
    "aya": {
        "temperature_factor": 1.2, # e.g., make Aya slightly more creative if needed
        "max_tokens_set": 12
    },
    "qwen": {
        "temperature_factor": 1.1,
        "top_k_set": 35,
        "max_tokens_set": 12
    }
}

# Placeholder NLI Data Loader - REPLACE THIS with your actual data loader
def load_nli_samples(lang_code: str, split: str = "validation", num_samples: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """
    Placeholder for loading NLI samples (e.g., from XNLI dataset).
    This function should return a pandas DataFrame with at least three columns:
    'premise', 'hypothesis', and 'label' (numeric 0,1,2 or string "entailment", "neutral", "contradiction").
    It should also handle language-specific loading if your NLI dataset has multiple languages.
    """
    logging.warning(f"Using MOCK NLI data loader for {lang_code}, split '{split}'. REPLACE with your actual loader.")
    
    # Example data structure (replace with actual data loading)
    data = []
    if lang_code == 'en':
        data = [
            {"premise": "A man inspects the uniform of a figure in some East Asian country.", "hypothesis": "The man is sleeping.", "label": 2}, # contradiction
            {"premise": "An older and younger man smiling.", "hypothesis": "Two men are smiling and laughing at the cats playing on the floor.", "label": 1}, # neutral
            {"premise": "A soccer game with multiple males playing.", "hypothesis": "Some men are playing a sport.", "label": 0}, # entailment
            {"premise": "The cat sat on the mat.", "hypothesis": "A feline was on a rug.", "label": 0},
            {"premise": "The weather is sunny.", "hypothesis": "It is raining.", "label": 2}
        ] * ( (num_samples // 5) if num_samples else 50) # Create more samples
    elif lang_code == 'sw': # Swahili examples (replace with actual Swahili XNLI data)
        data = [
            {"premise": "Mwanamume anayekagua sare ya mtu katika nchi fulani ya Asia Mashariki.", "hypothesis": "Mwanamume huyo amelala.", "label": 2},
            {"premise": "Mzee na kijana wakitabasamu.", "hypothesis": "Wanaume wawili wanatabasamu na kucheka paka wanaocheza sakafuni.", "label": 1},
            {"premise": "Mchezo wa soka na wanaume wengi wanacheza.", "hypothesis": "Baadhi ya wanaume wanacheza mchezo.", "label": 0}
        ] * ( (num_samples // 3) if num_samples else 50)
    elif lang_code == 'ur': # Urdu examples (replace with actual Urdu XNLI data)
        data = [
            {"premise": "ایک شخص مشرقی ایشیا کے کسی ملک میں ایک شخصیت کی وردی کا معائنہ کر رہا ہے۔", "hypothesis": "وہ شخص سو رہا ہے۔", "label": 2},
            {"premise": "ایک بوڑھا اور جوان مسکرا رہے ہیں۔", "hypothesis": "دو آدمی بلیوں کو فرش پر کھیلتے ہوئے دیکھ کر مسکرا رہے ہیں اور ہنس رہے ہیں۔", "label": 1},
            {"premise": "ایک فٹ بال کا کھیل جس میں کئی مرد کھیل رہے ہیں۔", "hypothesis": "کچھ مرد ایک کھیل کھیل رہے ہیں۔", "label": 0}
        ] * ( (num_samples // 3) if num_samples else 50)
    else:
        logging.warning(f"No mock NLI data defined for language: {lang_code}. Returning empty DataFrame.")
        return pd.DataFrame(columns=['premise', 'hypothesis', 'label'])

    full_df = pd.DataFrame(data)
    if num_samples and num_samples < len(full_df):
        return full_df.sample(n=num_samples, random_state=seed).reset_index(drop=True)
    return full_df.sample(frac=1, random_state=seed).reset_index(drop=True) # Shuffle all if not sampling a subset


def get_effective_nli_params(lang_code: str, model_name_str: str, cli_args: argparse.Namespace) -> Dict:
    """Determines effective generation parameters for NLI, applying overrides."""
    base_params = STANDARD_NLI_PARAMETERS.copy()
    lang_params = LANGUAGE_NLI_PARAMETERS.get(lang_code, {}).copy()
    
    model_short_name = "other"
    if "aya" in model_name_str.lower(): model_short_name = "aya"
    elif "qwen" in model_name_str.lower(): model_short_name = "qwen"
    # Add other model short names if you have specific adjustments for them
    
    model_adjust = MODEL_NLI_ADJUSTMENTS.get(model_short_name, {}).copy()

    # Start with standard, override with lang-specific, then apply model adjustments
    effective_params = {**base_params, **lang_params}

    # Apply model-specific multiplicative factors or direct sets
    if 'temperature_factor' in model_adjust and effective_params.get("temperature") is not None:
        effective_params["temperature"] *= model_adjust['temperature_factor']
    if 'top_p_factor' in model_adjust and effective_params.get("top_p") is not None:
        effective_params["top_p"] *= model_adjust['top_p_factor']
    if 'top_k_set' in model_adjust:
        effective_params["top_k"] = model_adjust['top_k_set']
    if 'max_tokens_set' in model_adjust:
        effective_params["max_tokens"] = model_adjust['max_tokens_set']
    
    # Override with CLI arguments if provided
    if cli_args.temperature is not None: effective_params["temperature"] = cli_args.temperature
    if cli_args.top_p is not None: effective_params["top_p"] = cli_args.top_p
    if cli_args.top_k is not None: effective_params["top_k"] = cli_args.top_k
    if cli_args.max_tokens is not None: effective_params["max_tokens"] = cli_args.max_tokens
    if cli_args.repetition_penalty is not None: effective_params["repetition_penalty"] = cli_args.repetition_penalty
    if cli_args.do_sample is not None: effective_params["do_sample"] = cli_args.do_sample
    
    # Ensure do_sample is consistent with temperature if not explicitly set by CLI
    if cli_args.do_sample is None: # Only if CLI didn't set it
        effective_params["do_sample"] = True if effective_params.get("temperature", 0) > 0.01 else False
    
    # Ensure temperature is not excessively low if do_sample is True
    if effective_params["do_sample"] and effective_params.get("temperature",0) < 0.01:
        logging.warning(f"Temperature is {effective_params['temperature']} but do_sample is True. Adjusting temperature to 0.01 for NLI.")
        effective_params["temperature"] = 0.01
        
    return effective_params

def calculate_nli_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate accuracy, F1, and per-class metrics for NLI predictions.
    Adapted from the user's provided NLI script.
    Assumes 'ground_truth_label' and 'predicted_label' columns contain string labels
    (e.g., "entailment", "neutral", "contradiction").
    """
    if results_df.empty or 'ground_truth_label' not in results_df.columns or 'predicted_label' not in results_df.columns:
        logging.error("Metrics calculation failed: DataFrame is empty or missing required label columns.")
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'class_metrics': {label: {'precision': 0, 'recall': 0, 'f1': 0} for label in EXPECTED_NLI_LABELS}}

    y_true = results_df['ground_truth_label'].astype(str).fillna("unknown").tolist()
    y_pred = results_df['predicted_label'].astype(str).fillna("unknown").tolist()
    
    # Labels for sklearn metrics (ensure they are the ones present in the data + expected)
    active_labels = sorted(list(set(y_true + y_pred) & set(EXPECTED_NLI_LABELS)))
    if not active_labels:
        active_labels = EXPECTED_NLI_LABELS # Fallback if no overlap, though unlikely with proper extraction

    try:
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=active_labels, average='macro', zero_division=0)
        
        report_dict = classification_report(y_true, y_pred, labels=active_labels, output_dict=True, zero_division=0)
        
        class_metrics = {}
        for label_name in EXPECTED_NLI_LABELS: # Report on all expected labels
            metrics = report_dict.get(label_name, {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0})
            class_metrics[label_name] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1-score'],
                'support': metrics['support']
            }
        logging.info(f"NLI Classification Report:\n{classification_report(y_true, y_pred, labels=active_labels, zero_division=0)}")

    except Exception as e_metrics:
        logging.error(f"Error calculating NLI metrics: {e_metrics}", exc_info=True)
        accuracy = 0.0
        macro_f1 = 0.0
        class_metrics = {label: {'precision': 0, 'recall': 0, 'f1': 0, 'support':0} for label in EXPECTED_NLI_LABELS}

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'class_metrics': class_metrics # This will be a dict of dicts
    }

def main():
    parser = argparse.ArgumentParser(description="Run NLI Baseline Experiments (e.g., with XNLI)")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated list of model names from HuggingFace.")
    parser.add_argument("--langs", type=str, default="en,ur,sw,fr", help="Comma-separated XNLI language codes (e.g., en,sw,ha,fr for English, Swahili, Hausa, French from XNLI).")
    parser.add_argument("--num_samples", type=int, default=80, help="Number of samples per language. Default: 80. Max for XNLI 'test' is 5000, 'validation' is 2500.")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split.")
    parser.add_argument("--prompt_in_lrl", action=argparse.BooleanOptionalAction, default=True, help="Use LRL instructions for prompts (few-shot examples remain English). Default is LRL instructions.")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Prompting strategies.")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/nli/baseline_xnli", 
                        help="Base directory to save results and summaries.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and reproducibility.")
    
    # Generation parameters (CLI overrides)
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature for generation.")
    parser.add_argument("--top_p", type=float, default=None, help="Override top-p for generation.")
    parser.add_argument("--top_k", type=int, default=None, help="Override top-k for generation.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Override max_tokens for NLI label generation (should be small).")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Override repetition penalty.")
    parser.add_argument("--do_sample", type=lambda x: (str(x).lower() == 'true'), default=None, help="Override do_sample (True/False). Default determined by temperature.")

    # parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for gated models.") # Add if needed
    parser.add_argument("--overwrite_results", action='store_true', help="Overwrite existing detailed results and summary files if they exist.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # For more detailed logs from the nli_baseline.py, set its logger to DEBUG if needed, e.g., by:
    # logging.getLogger('src.experiments.baseline.nli.nli_baseline').setLevel(logging.DEBUG)

    # HF Login (if needed)
    # token = args.hf_token if args.hf_token else get_token()
    # if token: login(token=token)
    # else: logging.warning("HF token not provided. Some models might be inaccessible.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_names = [m.strip() for m in args.models.split(',')]
    lang_codes = [lang.strip() for lang in args.langs.split(',')]

    # Use fixed number of samples directly, seed from args
    num_samples_to_load = args.num_samples
    current_seed = args.seed

    all_summaries = []

    # Create base output directories if they don't exist
    summaries_dir = os.path.join(args.base_output_dir, "summaries")
    results_dir = os.path.join(args.base_output_dir, "results")
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    for model_name_str in model_names:
        logging.info(f"\n===== Initializing Model for NLI: {model_name_str} =====")
        tokenizer, model = None, None # Ensure they are reset
        try:
            tokenizer, model = initialize_model(model_name_str)
            logging.info(f"Model {model_name_str} initialized successfully.")
        except Exception as e_init:
            logging.error(f"Failed to initialize model {model_name_str}: {e_init}. Skipping this model.", exc_info=True)
            if model: del model
            if tokenizer: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        for lang_code in lang_codes:
            logging.info(f"\n--- Loading NLI data for Language: {lang_code}, Model: {model_name_str} ---")
            # Load data (using the fixed num_samples and seed)
            logging.info(f"Loading data for language: {lang_code}, split: {args.data_split}, num_samples: {num_samples_to_load}, seed: {current_seed}")
            samples_df = load_xnli_samples(
                lang_code=lang_code, 
                split=args.data_split, 
                num_samples=num_samples_to_load, 
                seed=current_seed
            )

            if samples_df.empty:
                logging.warning(f"No XNLI samples loaded for {lang_code} ('test' split, 10% sample). Skipping this language for model {model_name_str}.")
                continue
            logging.info(f"Loaded {len(samples_df)} NLI samples for {lang_code} using 10% of 'test' split.")

            for shot_setting in args.shot_settings:
                use_few_shot_current = (shot_setting == 'few_shot')
                
                # Determine prompt_in_lrl based on lang_code (True if LRL, False if EN)
                current_prompt_in_lrl = lang_code.lower() != 'en'
                prompt_lang_str_display = "LRL-instruct" if current_prompt_in_lrl else "EN-instruct"

                logging.info(f"Running NLI Baseline: Model={model_name_str}, Lang={lang_code}, Shot={shot_setting}, Prompt-Instruct={prompt_lang_str_display}")

                effective_gen_params = get_effective_nli_params(lang_code, model_name_str, args)
                logging.info(f"  Effective NLI Generation Parameters: {effective_gen_params}")

                model_name_file_safe = model_name_str.replace("/", "_")
                exp_key = f"{lang_code}_{model_name_file_safe}_{shot_setting}_{prompt_lang_str_display}"
                
                # Define file paths for detailed results and summary for this specific config
                detailed_results_file = os.path.join(results_dir, f"results_nli_baseline_{exp_key}.csv")
                # Individual summary not typically needed if we make an overall one, but can be added.

                if not args.overwrite_results and os.path.exists(detailed_results_file):
                    logging.info(f"Results file {detailed_results_file} exists and overwrite is False. Attempting to load existing results.")
                    try:
                        exp_results_df = pd.read_csv(detailed_results_file)
                        logging.info(f"Successfully loaded existing results from {detailed_results_file}.")
                    except Exception as e_load:
                        logging.warning(f"Could not load existing results from {detailed_results_file}: {e_load}. Re-running experiment.")
                        exp_results_df = None # Ensure it runs
                else:
                    exp_results_df = None # Needs to run

                if exp_results_df is None: # Run the evaluation
                    exp_results_df = evaluate_nli_baseline(
                        model_name=model_name_str,
                        tokenizer=tokenizer,
                        model=model,
                        samples_df=samples_df,
                        data_lang_code=lang_code, 
                        prompt_in_lrl=current_prompt_in_lrl, 
                        use_few_shot=use_few_shot_current,
                        generation_params=effective_gen_params
                    )
                    if not exp_results_df.empty:
                        # Ensure the directory for the detailed_results_file exists
                        os.makedirs(os.path.dirname(detailed_results_file), exist_ok=True)
                        exp_results_df.to_csv(detailed_results_file, index=False, float_format='%.4f')
                        logging.info(f"Detailed NLI results saved to {detailed_results_file}")
                    else:
                        logging.error(f"NLI evaluation for {exp_key} (model: {model_name_str}, lang: {lang_code}, shot: {shot_setting}, prompt: {prompt_lang_str_display}) produced NO RESULTS. This might indicate an error during the evaluation process for this specific configuration.")

                if not exp_results_df.empty:
                    metrics = calculate_nli_metrics(exp_results_df)
                    summary_data = {
                        'model': model_name_str,
                        'language': lang_code,
                        'shot_type': shot_setting,
                        'prompt_language': prompt_lang_str_display, # EN-instruct or LRL-instruct
                        'accuracy': metrics['accuracy'],
                        'macro_f1': metrics['macro_f1'],
                        'class_metrics': json.dumps(metrics['class_metrics']), # Store dict as JSON string
                        **effective_gen_params, # Store the actual generation params used
                        'samples_processed': len(exp_results_df)
                    }
                    all_summaries.append(summary_data)
                    logging.info(f"Summary for {exp_key}: Acc={metrics['accuracy']:.4f}, MacroF1={metrics['macro_f1']:.4f}")
                else:
                    logging.warning(f"Skipping metrics calculation for {exp_key} as no results were generated or loaded.")
        
        # Clean up model from memory after processing all its languages/configs
        logging.info(f"Finished all NLI experiments for model {model_name_str}. Unloading model.")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU memory cache cleared after model unload.")

    if all_summaries:
        overall_summary_df = pd.DataFrame(all_summaries)
        # Ensure all columns that should be numeric are, coercing errors for robustness.
        numeric_cols = ['accuracy', 'macro_f1', 'temperature', 'top_p', 'top_k', 'max_tokens', 'repetition_penalty', 'samples_processed']
        # Add class specific metrics if they exist
        for label_class in EXPECTED_NLI_LABELS:
            numeric_cols.extend([f'{label_class}_precision', f'{label_class}_recall', f'{label_class}_f1'])

        for col in numeric_cols:
            if col in overall_summary_df.columns:
                overall_summary_df[col] = pd.to_numeric(overall_summary_df[col], errors='coerce')

        summary_file_path = os.path.join(summaries_dir, f"nli_baseline_ALL_experiments_summary_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        
        try:
            overall_summary_df.to_csv(summary_file_path, index=False, float_format='%.4f')
            logging.info(f"Overall NLI baseline summary saved to {summary_file_path}")
            print("\n===== Overall NLI Baseline Summary =====")
            # Print relevant columns for quick overview
            cols_to_show = ['model', 'language', 'shot_type', 'prompt_language', 'accuracy', 'macro_f1', 'temperature', 'max_tokens', 'samples_processed']
            cols_present = [col for col in cols_to_show if col in overall_summary_df.columns]
            print(overall_summary_df[cols_present].to_string())
        except Exception as e_save:
            logging.error(f"Error saving overall NLI summary to {summary_file_path}: {e_save}")
    else:
        logging.info("No NLI baseline experiments were successfully completed to summarize.")

    logging.info("\nAll NLI Baseline experiments completed!")

if __name__ == "__main__":
    main()