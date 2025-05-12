import sys
import os

# Add the project root to the Python path - MUST come before other project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)
print(f"Added project root to Python path: {project_root}")

import argparse
import pandas as pd
import numpy as np
import time
from huggingface_hub import login

# Import utility functions using absolute imports
from src.utils.data_loaders.load_xnli import load_xnli_samples
from src.experiments.baseline.nli.nli_baseline import evaluate_nli_baseline, calculate_nli_metrics, initialize_model
from config import get_token

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define STANDARD_PARAMETERS and LANGUAGE_PARAMETERS
STANDARD_PARAMETERS = {
    "temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_tokens": 10, "repetition_penalty": 1.0
}
LANGUAGE_PARAMETERS = { # Specific overrides for NLI baseline
    "en": {"temperature": 0.2, "max_tokens": 10, "repetition_penalty": 1.05},
    "sw": {"temperature": 0.25, "max_tokens": 10, "repetition_penalty": 1.1},
    "ur": {"temperature": 0.25, "max_tokens": 10, "repetition_penalty": 1.1},
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run NLI baseline experiments using XNLI dataset.")
    parser.add_argument("--langs", nargs='+', default=['en', 'sw', 'ur'], 
                        help="Languages to test from XNLI (default: en, sw, ur).")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-expanse-8b",
                        help="Comma-separated list of models to use (e.g., CohereLabs/aya-expanse-8b,Qwen/Qwen2.5-7B-Instruct).")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'],
                        help="Shot settings to evaluate (zero_shot, few_shot).")
    parser.add_argument("--prompt_in_lrl_settings", nargs='+', type=lambda x: (str(x).lower() == 'true'), default=[False, True],
                        help="Prompt in LRL (True) or EN (False). Provide as 'True' or 'False'.")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to use per language (default: 100).")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/nli/baseline",
                        help="Base directory for results and summaries.")
    # Generation parameters with defaults that can be overridden
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None) # Max new tokens for label
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--do_sample", action='store_true', help="Enable sampling (temperature > 0 implies this).")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--data_split", type=str, default="test", help="Dataset split for XNLI (e.g., train, validation, test). Default: test")

    return parser.parse_args()

def run_nli_baseline_experiment(
    model_name: str,
    tokenizer: Any, # Pass tokenizer
    model: Any,     # Pass model
    samples_df: pd.DataFrame,
    lang_code: str,
    base_path: str,
    args_exp: argparse.Namespace # Pass all parsed args for consistency
):
    """Run a single NLI baseline experiment configuration and save results."""
    shot_type_str = args_exp.shot_type # This will be 'zero_shot' or 'few_shot'
    use_few_shot_run = (shot_type_str == 'few_shot')
    prompt_in_lrl_run = args_exp.prompt_in_lrl

    # Skip LRL prompts for English
    if lang_code == 'en' and prompt_in_lrl_run:
        print(f"INFO: Skipping LRL prompt for English language with {shot_type_str} for model {model_name}.")
        return None

    prompt_lang_str = "LRL-instruct" if prompt_in_lrl_run and lang_code != 'en' else "EN-instruct"
    print(f"\nRunning NLI Baseline: Model={model_name}, Lang={lang_code}, Shot={shot_type_str}, PromptLang={prompt_lang_str}")

    # Determine effective generation parameters
    current_params = LANGUAGE_PARAMETERS.get(lang_code, STANDARD_PARAMETERS)
    effective_params = {
        "temperature": args_exp.temperature if args_exp.temperature is not None else current_params.get("temperature"),
        "top_p": args_exp.top_p if args_exp.top_p is not None else current_params.get("top_p"),
        "top_k": args_exp.top_k if args_exp.top_k is not None else current_params.get("top_k"),
        "max_tokens": args_exp.max_tokens if args_exp.max_tokens is not None else current_params.get("max_tokens"),
        "repetition_penalty": args_exp.repetition_penalty if args_exp.repetition_penalty is not None else current_params.get("repetition_penalty"),
        "do_sample": args_exp.do_sample or (args_exp.temperature is not None and args_exp.temperature > 0.01) # Ensure do_sample is true if temp > 0.01
    }
    # Model-specific overrides on top
    if "aya" in model_name.lower():
        effective_params["temperature"] = max(0.05, effective_params["temperature"] * 0.85 if effective_params["temperature"] else 0.05)
        effective_params["repetition_penalty"] = (effective_params["repetition_penalty"] or 1.0) * 1.05
    elif "qwen" in model_name.lower():
        effective_params["top_p"] = max(0.7, effective_params["top_p"] * 0.9 if effective_params["top_p"] else 0.7)
        if args_exp.top_k is None and LANGUAGE_PARAMETERS.get(lang_code, {}).get("top_k") is None : effective_params["top_k"] = 35

    print(f"  Effective generation parameters: {effective_params}")
    
    results_df = evaluate_nli_baseline(
        model_name=model_name, # Pass model_name for internal use
        tokenizer=tokenizer,    # Pass tokenizer
        model=model,            # Pass model
        samples_df=samples_df,
        lang_code=lang_code,
        prompt_in_lrl=prompt_in_lrl_run,
        use_few_shot=use_few_shot_run,
        temperature=effective_params["temperature"],
        top_p=effective_params["top_p"],
        top_k=effective_params["top_k"],
        max_new_tokens=effective_params["max_tokens"],
        repetition_penalty=effective_params["repetition_penalty"],
        do_sample=effective_params["do_sample"]
    )
    
    if results_df.empty:
        print(f"WARNING: No results returned for {model_name}, {lang_code}, {shot_type_str}, {prompt_lang_str}")
        return None
    
    # Calculate metrics
    metrics = calculate_nli_metrics(results_df)
    print(f"  Metrics for {model_name}, {lang_code}, {shot_type_str}, {prompt_lang_str}: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")
    
    # Save detailed results
    model_name_file = model_name.replace("/", "_")
    results_dir = os.path.join(base_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    detailed_results_filename = f"results_nli_baseline_{lang_code}_{model_name_file}_{shot_type_str}_{prompt_lang_str}.csv"
    detailed_results_path = os.path.join(results_dir, detailed_results_filename)
    results_df.to_csv(detailed_results_path, index=False)
    print(f"  Detailed results saved to: {detailed_results_path}")
    
    # Prepare summary data
    summary_data = {
        'model': model_name,
        'language': lang_code,
        'shot_type': shot_type_str,
        'prompt_language': prompt_lang_str,
        **metrics,
        **effective_params,
        'samples_processed': len(results_df)
    }
    return summary_data

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # HF Login
    token = args.hf_token if args.hf_token else get_token()
    if token:
    login(token=token)
    else:
        logging.warning("HuggingFace token not provided. Some models might be inaccessible.")

    # Ensure base output directory exists
    summaries_dir = os.path.join(args.base_output_dir, "summaries")
    os.makedirs(summaries_dir, exist_ok=True)

    all_experiment_summaries = []
    models_to_run = args.models.split(',')

    for model_name_str in models_to_run:
        logging.info(f"\n{'='*20} Initializing Model: {model_name_str} {'='*20}")
        tokenizer_main, model_main = None, None # Use different names to avoid scope issues
        try:
            tokenizer_main, model_main = initialize_model(model_name_str)
        except Exception as e_init:
            logging.error(f"Failed to initialize model {model_name_str}: {e_init}. Skipping this model.")
            if model_main: del model_main
            if tokenizer_main: del tokenizer_main
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

    for lang_code in args.langs:
            print(f"\n--- Loading XNLI Data for {lang_code} (Split: {args.data_split}) ---")
            samples_df_for_lang = load_xnli_samples(lang_code, num_samples=args.samples, split=args.data_split)
        
            if samples_df_for_lang.empty:
                print(f"WARNING: No samples loaded for {lang_code} using split '{args.data_split}'. Skipping experiments for {model_name_str}.")
                continue

            # Iterate through shot_settings specified by user
            for shot_setting_str_arg in args.shot_settings:
                current_args_exp = argparse.Namespace(**vars(args)) # Create a mutable copy for this iteration
                current_args_exp.shot_type = shot_setting_str_arg # Set current shot_type for run_nli_baseline_experiment

                # Iterate through prompt_in_lrl_settings specified by user
                for prompt_in_lrl_arg in args.prompt_in_lrl_settings:
                    current_args_exp.prompt_in_lrl = prompt_in_lrl_arg # Set current prompt_in_lrl

                    summary = run_nli_baseline_experiment(
                        model_name=model_name_str,
                        tokenizer=tokenizer_main,
                        model=model_main,
                        samples_df=samples_df_for_lang,
                        lang_code=lang_code,
                        base_path=args.base_output_dir,
                        args_exp=current_args_exp # Pass the copied and modified args
                    )
                    if summary:
                        all_experiment_summaries.append(summary)
        
        # Clean up model after all its experiments
        del model_main
        del tokenizer_main
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logging.info(f"Model {model_name_str} unloaded.")

    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        summary_file_path = os.path.join(summaries_dir, "nli_baseline_ALL_XNLI_experiments_summary.csv")
        overall_summary_df.to_csv(summary_file_path, index=False, float_format='%.4f')
        logging.info(f"Overall NLI baseline summary for XNLI saved to {summary_file_path}")
        print(overall_summary_df.to_string())
    else:
        logging.info("No NLI baseline experiments were successfully completed to summarize.")

    logging.info("\nAll NLI Baseline (XNLI) experiments completed!")

if __name__ == "__main__":
    main() 