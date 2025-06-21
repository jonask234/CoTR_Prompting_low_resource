# thesis_project/run_qa_baseline.py

import sys
import os
import logging
import argparse
import torch
from typing import Any, Optional, Dict
import random
import numpy as np
import pandas as pd
from huggingface_hub import login
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import gc

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to the Python path
# Go up four levels from the script's directory: qa -> baseline -> experiments -> src -> project_root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loaders.load_tydiqa import load_tydiqa_samples, TYDIQA_LANG_CONFIG_MAP
from src.experiments.baseline.qa.qa_baseline import evaluate_qa_baseline, initialize_model
from config import get_token

SUPPORTED_LANGS = list(TYDIQA_LANG_CONFIG_MAP.keys()) # From tydiqa_loader

logger = logging.getLogger(__name__)

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run Baseline QA Experiments with TyDiQA.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-expanse-8b,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated model names.")
    parser.add_argument("--langs", type=str, default="en,sw,fi", help="Comma-separated TyDiQA language codes (e.g., en,sw,fi).")
    parser.add_argument("--num_samples", type=int, default=80, help="Number of samples per language. Default: 80")
    parser.add_argument("--data_split", type=str, default="validation", choices=["train", "validation"], help="Dataset split (TyDiQA 'validation' corresponds to dev set).")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/qa/baseline_tydiqa", help="Base directory to save results and summaries.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and reproducibility.")
    parser.add_argument(
        "--prompt_in_lrl",
        action="store_true",
        default=True,
        help="Use LRL for main prompt instructions, with English few-shot examples. Default: True (LRL instructions, English few-shot)."
    )
    parser.add_argument(
        "--no-prompt_in_lrl",
        action="store_false",
        dest="prompt_in_lrl",
        help="Use English for main prompt instructions and English few-shot examples."
    )
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Prompting strategies.")
    parser.add_argument("--temperature", type=float, default=None, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling top_p.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Global override for max_new_tokens for generation.")
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="If set, overwrite existing result files instead of skipping experiments."
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default=None, 
        help="HuggingFace API token (optional, reads from config if not provided)."
    )
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    
    args = parser.parse_args()
    # Convert comma-separated strings to lists
    args.models = [m.strip() for m in args.models.split(',')]
    args.langs = [l.strip() for l in args.langs.split(',')]
    return args

# Define LANGUAGE_PARAMETERS globally or pass it appropriately
LANGUAGE_PARAMETERS = {
    "en": {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_tokens": 50},
    "sw": {"temperature": 0.25, "top_p": 0.85, "top_k": 35, "max_tokens": 55}, # Example for Swahili
    "fi": {"temperature": 0.28, "top_p": 0.88, "top_k": 38, "max_tokens": 52}  # Example for Finnish
}

def run_experiment(
    model_name: str, 
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame, 
    lang_code: str, 
    base_results_path: str,
    use_few_shot: bool, # Add few-shot flag
    prompt_in_lrl: bool, # Add prompt_in_lrl flag
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int
) -> Optional[Dict[str, Any]]:
    """
    Runs a single QA baseline experiment configuration.
    
    Args:
        model_name: Name of the model to use
        tokenizer: Tokenizer for the model
        model: Model to use
        samples_df: DataFrame containing the samples
        lang_code: Language code
        base_results_path: Base path for saving results
        use_few_shot: Whether to use few-shot prompting
        prompt_in_lrl: Whether to use LRL for main prompt instructions
        temperature: Generation temperature
        top_p: Nucleus sampling top_p
        top_k: Top-k filtering
        max_tokens: Maximum new tokens to generate
    """
    # Check if samples_df is empty
    if samples_df.empty:
        print(f"WARNING: No samples provided for {lang_code}. Skipping experiment for {model_name}.")
        return
        
    try:
        shot_desc = "FewShot" if use_few_shot else "ZeroShot"
        print(f"\nProcessing {lang_code} samples with {model_name} (Baseline, {shot_desc})...")
        
        # --- Call the evaluation function --- 
        # Pass the use_few_shot flag and prompt_in_lrl flag
        results_df = evaluate_qa_baseline(
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            samples_df=samples_df.copy(), # Pass a copy
            lang_code=lang_code,
            use_few_shot=use_few_shot,
            prompt_in_lrl=prompt_in_lrl, # Pass the flag
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens
            # repetition_penalty and do_sample will use defaults in evaluate_qa_baseline
        )
        
        # --- Check if results were generated ---
        if results_df.empty:
            print(f"WARNING: evaluate_qa_baseline returned empty DataFrame for {lang_code} with {model_name}. Skipping further processing.")
            return
            
        # --- Process the results --- 
        # First check for column format compatibility (handle both formats)
        # Fallback data has 'ground_truth' directly, while regular dataset has 'ground_truth_answers'
        if 'ground_truth' in results_df.columns and not results_df['ground_truth'].isnull().all():
            # We already have ground_truth column, no need to extract
            print(f"Using 'ground_truth' column from fallback data for {lang_code}")
        elif 'ground_truth_answers' in results_df.columns and not results_df['ground_truth_answers'].isnull().all():
            # We have ground_truth_answers, need to convert to ground_truth
            print(f"Converting ground_truth_answers to ground_truth for {lang_code}")
            results_df['ground_truth'] = results_df['ground_truth_answers'].apply(
                lambda answers: answers[0] if isinstance(answers, list) and answers else None
            )
        else:
            print(f"WARNING: Neither 'ground_truth' nor 'ground_truth_answers' column available for {lang_code} with {model_name}. Cannot calculate F1.")
            return
        
        # Drop rows where ground_truth is missing
        results_df.dropna(subset=['ground_truth'], inplace=True)
        if results_df.empty:
            print(f"WARNING: No valid samples remaining after checking ground truth for {lang_code} with {model_name}. Cannot calculate F1.")
            return

        # Ensure few_shot column exists
        if 'few_shot' not in results_df.columns:
            results_df['few_shot'] = use_few_shot

        avg_f1 = results_df["f1_score"].mean()
        print(f"\nAverage F1 Score for {lang_code} ({model_name}, Baseline, {shot_desc}): {avg_f1:.4f}")
        
        # --- Save Detailed Results (DataFrame) ---
        # Adjust path to include few-shot status
        results_subdir = os.path.join(base_results_path, shot_desc, lang_code)
        os.makedirs(results_subdir, exist_ok=True)
        
        # Save detailed results - adjust filename format
        model_name_short = model_name.split('/')[-1]  # Get just the model name
        shot_suffix = "fs" if use_few_shot else "zs"
        output_filename = f"baseline_{shot_suffix}_qa_tydiqa_{lang_code}_{model_name_short}.csv" 
        
        # Define columns to save
        columns_to_save = ['question', 'ground_truth', 'predicted_answer', 'f1_score', 'language', 'few_shot']
        # Add context_used if it exists
        if 'context_used' in results_df.columns:
             columns_to_save.append('context_used')
             
        # Filter columns to ensure they exist
        columns_to_save = [col for col in columns_to_save if col in results_df.columns]
             
        results_df[columns_to_save].to_csv(os.path.join(results_subdir, output_filename), index=False)
        print(f"Results saved to {results_subdir}/{output_filename}")

        # --- Save Summary Metrics ---
        summary = {
            'model': model_name,
            'language': lang_code,
            'pipeline': 'baseline',
            'few_shot': use_few_shot,
            'prompt_in_lrl': prompt_in_lrl,
            'f1_score': avg_f1
        }
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(base_results_path, "summaries") 
        os.makedirs(summary_path, exist_ok=True)
        # Update summary filename
        summary_filename = f"summary_baseline_{shot_suffix}_qa_tydiqa_{lang_code}_{model_name_short}.csv" 
        summary_df.to_csv(os.path.join(summary_path, summary_filename), index=False, float_format='%.4f')
        print(f"Summary metrics saved to {summary_path}/{summary_filename}")

        return summary

    except Exception as e:
        print(f"Error during run_experiment for {model_name} for {lang_code}: {str(e)}")
        # Keep the specific error handling for restricted models
        if "Access to model" in str(e) and "is restricted" in str(e):
            print("\nTo use the Aya model, you need to:")
            print("1. Create a Hugging Face account at https://huggingface.co/join")
            print("2. Accept the model's terms of use at https://huggingface.co/CohereLabs/aya-expanse-8b")
            print("3. Generate an access token at https://huggingface.co/settings/tokens")
            print("4. Run this script again with your token")
            sys.exit(1) 
        
def main():
    """
    Main function to run QA baseline experiments with optimized parameters.
    """
    # Define standard parameters for both baseline and CoTR
    STANDARD_PARAMETERS = {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 50
    }
    
    # Get arguments
    args = parse_cli_args()
    
    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Define the results directory using the --base_output_dir argument
    base_results_dir = args.base_output_dir
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Define the languages to test
    languages = {
        "english": "en",
        "swahili": "sw",
        "finnish": "fi"  # Added Finnish
    }
    
    # Load the samples for each language
    tydiqa_samples = {}
    for name, code in languages.items():
        logging.info(f"Loading samples for {name} ({code})...")
        
        # Load appropriate number of samples based on percentage
        num_samples_to_load = args.num_samples
        current_seed = args.seed
        logger.info(f"Loading TyDiQA samples for {code}, split: {args.data_split}, num_samples: {num_samples_to_load}, seed: {current_seed}")
        # Ensure load_tydiqa_samples from the utils is used
        samples_df_for_lang = load_tydiqa_samples(
            lang_code=code, 
            num_samples=num_samples_to_load,
            split=args.data_split,
            seed=current_seed
        )
        
        if not samples_df_for_lang.empty:
            logging.info(f"  Loaded {len(samples_df_for_lang)} samples for {name}")
            tydiqa_samples[code] = samples_df_for_lang
        else:
            logging.warning(f"  No samples loaded for {name}")
    
    # Run the experiments
    all_experiment_summaries = [] # ADDED: To collect all summaries

    model_names_list = args.models
    lang_codes_list = args.langs

    for model_name_str in model_names_list:
        logger.info(f"\n===== Initializing Model: {model_name_str} =====")
        tokenizer, model = None, None # Initialize here for the current model scope
        try:
            # Initialize model once per model string
            tokenizer, model = initialize_model(model_name_str) 
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name_str}: {e}. Skipping this model.")
            # Ensure cleanup if partial initialization occurred before error
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue # Next model in args.models

        # THIS LOOP IS NOW NESTED CORRECTLY
        for lang_code, samples_df in tydiqa_samples.items():
            if samples_df.empty:
                continue
            
            logger.info(f"Processing {lang_code} samples with {model_name_str}...")
        
            # Determine effective generation parameters
            # Start with language-specific or standard parameters
            current_params = LANGUAGE_PARAMETERS.get(lang_code, STANDARD_PARAMETERS).copy()

            # Apply model-specific adjustments (ensure this block is correctly indented)
            if "aya" in model_name_str.lower():
                current_params["temperature"] = max(0.01, current_params.get("temperature", 0.3) * 0.85) 
                current_params["top_p"] = max(0.7, current_params.get("top_p", 0.9) * 0.95)
                current_params["max_tokens"] = current_params.get("max_tokens", 50) + 10 # Allow a bit more for Aya
            elif "qwen" in model_name_str.lower():
                current_params["temperature"] = max(0.01, current_params.get("temperature", 0.3) * 0.9)
                current_params["top_k"] = current_params.get("top_k", 40) - 5 # Qwen might be better with slightly lower top_k
                current_params["max_tokens"] = current_params.get("max_tokens", 50) + 5
            # Add more model-specific adjustments as needed
            # else: (No specific adjustments for other models, uses lang/standard)

            # Override with CLI arguments if provided
            effective_params = {
                "temperature": args.temperature if args.temperature is not None else current_params.get("temperature"),
                "top_p": args.top_p if args.top_p is not None else current_params.get("top_p"),
                "top_k": args.top_k if args.top_k is not None else current_params.get("top_k"),
                "max_tokens": args.max_tokens if args.max_tokens is not None else current_params.get("max_tokens")
            }
        
            # MODIFIED: Determine shot_settings based on command-line arguments
            if args.shot_settings:
                shot_settings = args.shot_settings
                logger.info(f"Running specified shot settings: {shot_settings}")
            else:
                shot_settings = ['zero_shot', 'few_shot'] # Default: zero-shot and few-shot
                logger.info("Running both zero-shot and few-shot evaluations (default). Use --shot_settings for other modes.")
        
            for use_few_shot in shot_settings:
                shot_type = "few-shot" if use_few_shot == 'few_shot' else "zero-shot"
                logger.info(f"Running {shot_type} evaluation with parameters: {effective_params}")
                
                # MODIFIED: Call run_experiment which now returns a summary dict
                experiment_summary = run_experiment(
                    model_name_str,
                    tokenizer,
                    model,
                    samples_df,
                    lang_code,
                    base_results_dir, # Pass base_results_dir
                    use_few_shot == 'few_shot',
                    args.prompt_in_lrl, # Pass the prompt_in_lrl flag
                    effective_params["temperature"],
                    effective_params["top_p"],
                    effective_params["top_k"],
                    effective_params["max_tokens"]
                )

                if experiment_summary:
                    all_experiment_summaries.append(experiment_summary)
                
                # This block for detailed results and individual summary saving is now part of run_experiment
                # So, it's removed from here to avoid duplication.

        # Clean up model and tokenizer after all languages and shot settings for it are done
        logger.info(f"Finished all experiments for model {model_name_str}. Unloading...")
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared.")

    # ADDED: Generate overall summary and plots
    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        overall_summary_filename = os.path.join(base_results_dir, "summaries", "baseline_qa_ALL_experiments_summary.csv")
        overall_summary_df.to_csv(overall_summary_filename, index=False, float_format='%.4f')
        logger.info(f"Overall summary of all baseline QA experiments saved to: {overall_summary_filename}")
        print("\nOverall Summary:")
        print(overall_summary_df)

        # Generate plots
        plots_dir = os.path.join(base_results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_qa_f1_scores(overall_summary_df, plots_dir)
    else:
        logger.info("No experiment summaries collected. Skipping overall summary and plot generation.")

# ADDED: Plotting function for F1 scores
def plot_qa_f1_scores(summary_df: pd.DataFrame, plots_dir: str):
    """Generate and save plots for QA F1 scores."""
    if summary_df.empty:
        logger.info("Summary DataFrame is empty. Skipping plotting.")
        return

    required_cols = ['model', 'language', 'few_shot', 'prompt_in_lrl', 'f1_score']
    if not all(col in summary_df.columns for col in required_cols if col in summary_df.columns): # check only for existing required cols
        logger.warning(f"One or more required columns for plotting not found in summary_df. Skipping plotting.")
        logger.debug(f"Available columns: {summary_df.columns.tolist()}")
        return

    # Ensure f1_score is numeric
    summary_df['f1_score'] = pd.to_numeric(summary_df['f1_score'], errors='coerce')

    # Create an experiment configuration string for unique x-axis labels
    if 'prompt_in_lrl' in summary_df.columns and summary_df['prompt_in_lrl'].nunique() > 1:
        summary_df['experiment_config'] = summary_df['model'] + "_" + \
                                          summary_df['language'] + "_" + \
                                          summary_df['few_shot'].astype(str) + "_" + \
                                          summary_df['prompt_in_lrl'].astype(str)
    else:
        summary_df['experiment_config'] = summary_df['model'] + "_" + \
                                      summary_df['language'] + "_" + \
                                      summary_df['few_shot'].astype(str)
    
    plt.figure(figsize=(15, 8))
    sns.barplot(data=summary_df, x='experiment_config', y='f1_score', hue='language', dodge=True)
    plt.xticks(rotation=45, ha='right')
    plt.title('QA Baseline F1 Scores by Experiment Configuration')
    plt.ylabel('F1 Score')
    plt.xlabel('Experiment Configuration (Model_Language_ShotSetting[_PromptLRL])')
    plt.tight_layout()
    
    plot_filename = "qa_baseline_f1_scores_summary.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    try:
        plt.savefig(plot_path)
        logger.info(f"F1 score plot saved to {plot_path}")
    except Exception as e_plot:
        logger.error(f"Failed to save F1 plot: {e_plot}")
    finally:
        plt.close()

if __name__ == "__main__":
    main()