# In src/experiments/baseline/sentiment/run_sentiment_baseline.py

import sys
import os

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to the Python path
# Assuming the script is in src/experiments/baseline/sentiment/
# Project root is four levels up.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import argparse
import torch
import pandas as pd # Make sure pandas is imported
from huggingface_hub import login # For HF token
from config import get_token # For HF token

from src.utils.data_loaders.load_afrisenti import load_afrisenti_samples # Corrected import
from src.experiments.baseline.sentiment.sentiment_baseline import evaluate_sentiment_baseline, initialize_model
from evaluation.sentiment_metrics import calculate_sentiment_metrics # Assuming this exists

# Define STANDARD_PARAMETERS and LANGUAGE_PARAMETERS (similar to run_qa_cotr.py)
# Example:
STANDARD_PARAMETERS = {
    "temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_tokens": 10, "repetition_penalty": 1.1
}
LANGUAGE_PARAMETERS = {
    "sw": {"temperature": 0.25, "max_tokens": 10, "repetition_penalty": 1.15},
    "ha": {"temperature": 0.25, "max_tokens": 10, "repetition_penalty": 1.15}
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sentiment Analysis baseline experiments.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-expanse-8b,Qwen/Qwen2.5-7B-Instruct", help="Models to evaluate.")
    parser.add_argument("--langs", nargs='+', default=['sw', 'ha'], help="Languages to test (e.g., sw ha te).")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples per language.")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Shot settings.")
    parser.add_argument("--prompt_in_lrl_settings", nargs='+', type=lambda x: (str(x).lower() == 'true'), default=[False, True], help="Prompt in LRL (True) or EN (False). Provide as 'True' or 'False'.")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/sentiment/baseline", help="Base output directory.")
    # Add generation parameters
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None) # Max new tokens for label
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--data_split", type=str, default="train", help="Dataset split to use (e.g., train, validation, test)") # Added argument for data split
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # HF Login
    token = args.hf_token if args.hf_token else get_token()
    if token:
    login(token=token)
    else:
        logging.warning("HuggingFace token not provided. Some models might be inaccessible.")

    # Create output directories
    summaries_dir = os.path.join(args.base_output_dir, "summaries")
    results_dir = os.path.join(args.base_output_dir, "results")
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    all_experiment_summaries = []

    for model_name_str in args.models.split(','):
        logging.info(f"\n{'='*20} Initializing Model: {model_name_str} {'='*20}")
        tokenizer, model = None, None
        try:
            tokenizer, model = initialize_model(model_name_str)
        except Exception as e_init:
            logging.error(f"Failed to initialize model {model_name_str}: {e_init}. Skipping.")
            if model: del model
            if tokenizer: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        for lang_code in args.langs:
            logging.info(f"\n--- Processing Language: {lang_code} for model {model_name_str} ---")
            # Use the corrected function name and pass the data_split argument
            samples_df_for_lang = load_afrisenti_samples(lang_code, num_samples=args.samples, split=args.data_split)
            if samples_df_for_lang.empty:
                logging.warning(f"No samples found for {lang_code} using split '{args.data_split}'. Skipping.")
                continue

            for use_few_shot in [False, True]: # Iterate zero-shot then few-shot
                shot_setting_str = "few_shot" if use_few_shot else "zero_shot"
                if shot_setting_str not in args.shot_settings:
                    continue

                for prompt_in_lrl in args.prompt_in_lrl_settings:
                    if lang_code == 'en' and prompt_in_lrl: # English text should use English prompts
                        logging.info(f"Skipping LRL prompt for English language with {shot_setting_str}.")
                        continue
                    
                    prompt_lang_str = "LRL-instruct" if prompt_in_lrl and lang_code != 'en' else "EN-instruct"
                    logging.info(f"Running: {shot_setting_str}, {prompt_lang_str}")

                    # Determine effective generation parameters
                    current_params = LANGUAGE_PARAMETERS.get(lang_code, STANDARD_PARAMETERS)
                    effective_params = {
                        "temperature": args.temperature if args.temperature is not None else current_params.get("temperature"),
                        "top_p": args.top_p if args.top_p is not None else current_params.get("top_p"),
                        "top_k": args.top_k if args.top_k is not None else current_params.get("top_k"),
                        "max_tokens": args.max_tokens if args.max_tokens is not None else current_params.get("max_tokens"),
                        "repetition_penalty": args.repetition_penalty if args.repetition_penalty is not None else current_params.get("repetition_penalty"),
                    }
                    # Model-specific overrides on top
                    if "aya" in model_name_str.lower():
                        effective_params["temperature"] = max(0.1, effective_params["temperature"] * 0.9 if effective_params["temperature"] else 0.1) # Adjust if not None
                    elif "qwen" in model_name_str.lower():
                         effective_params["top_p"] = max(0.7, effective_params["top_p"] * 0.9 if effective_params["top_p"] else 0.7)
                         if args.top_k is None and LANGUAGE_PARAMETERS.get(lang_code, {}).get("top_k") is None : effective_params["top_k"] = 35


                    logging.info(f"Effective generation parameters: {effective_params}")

                    exp_results_df = evaluate_sentiment_baseline(
                        model_name=model_name_str,
                        tokenizer=tokenizer,
                        model=model,
                        samples_df=samples_df_for_lang,
                        lang_code=lang_code,
                        prompt_in_lrl=prompt_in_lrl,
                        use_few_shot=use_few_shot,
                        generation_params=effective_params
                    )

                    if not exp_results_df.empty:
                        # Save detailed results
                        model_name_file = model_name_str.replace("/", "_")
                        detailed_results_path = os.path.join(results_dir, f"results_baseline_sentiment_{lang_code}_{model_name_file}_{shot_setting_str}_{prompt_lang_str}.csv")
                        exp_results_df.to_csv(detailed_results_path, index=False)
                        logging.info(f"Detailed results saved to {detailed_results_path}")

                        # Calculate and store summary
                        metrics = calculate_sentiment_metrics(exp_results_df) # Assuming this exists
                        summary_data = {
                            'model': model_name_str, 'language': lang_code,
                            'shot_type': shot_setting_str, 'prompt_language': prompt_lang_str,
                            **metrics, **effective_params,
                            'samples_processed': len(exp_results_df)
                        }
                        all_experiment_summaries.append(summary_data)
                    else:
                        logging.warning(f"No results for {model_name_str}, {lang_code}, {shot_setting_str}, {prompt_lang_str}")
        
        # Clean up model after all its experiments
        del model
        del tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logging.info(f"Model {model_name_str} unloaded.")

    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        summary_file_path = os.path.join(summaries_dir, "sentiment_baseline_ALL_experiments_summary.csv")
        overall_summary_df.to_csv(summary_file_path, index=False, float_format='%.4f')
        logging.info(f"Overall baseline sentiment summary saved to {summary_file_path}")
        print(overall_summary_df)
    else:
        logging.info("No baseline sentiment experiments were successfully completed to summarize.")

    logging.info("\nAll Sentiment Baseline experiments completed!")

if __name__ == "__main__":
    main() 