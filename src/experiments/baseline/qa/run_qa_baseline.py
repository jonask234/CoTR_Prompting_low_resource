# thesis_project/run_qa_baseline.py

import sys
import os
import logging
import argparse
import torch
from typing import Any, Optional, Dict

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to the Python path
# Go up four levels from the script's directory: qa -> baseline -> experiments -> src -> project_root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from src.utils.data_loaders.load_tydiqa import load_tydiqa_samples
from src.experiments.baseline.qa.qa_baseline import evaluate_qa_baseline, initialize_model
from huggingface_hub import login
from config import get_token

# ADDED: Import for plotting
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run QA baseline experiments with optimized parameters.")
    parser.add_argument("--models", nargs='+', default=["CohereLabs/aya-expanse-8b", "Qwen/Qwen2.5-7B-Instruct"], 
                        help="List of model names or paths for QA.")
    parser.add_argument("--langs", type=str, default="en,sw,te",
                        help="Comma-separated language codes (e.g., en,sw,te). Default is en,sw,te.")
    parser.add_argument("--sample_percentage", type=float, default=10.0,
                        help="Percentage of samples per language for TyDiQA (e.g., 10 for 10%). Default: 10.")
    parser.add_argument("--few_shot", action="store_true", 
                        help="Enable few-shot prompting if not comparing shots.")
    parser.add_argument("--compare_shots", action="store_true", 
                        help="Run both few-shot and zero-shot evaluations.")
    # Add any other arguments that might be needed based on script usage
    parser.add_argument("--output", type=str, default="/work/bbd6522/results/qa/baseline",
                        help="Base output directory for results and summaries.")
    # Add missing generation parameters to parser
    parser.add_argument("--temperature", type=float, default=None, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling top_p.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum new tokens to generate.")
    return parser.parse_args()

# Define LANGUAGE_PARAMETERS globally or pass it appropriately
LANGUAGE_PARAMETERS = {
    "en": {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_tokens": 50},
    "sw": {"temperature": 0.25, "top_p": 0.85, "top_k": 35, "max_tokens": 55}, # Example for Swahili
    "te": {"temperature": 0.28, "top_p": 0.88, "top_k": 38, "max_tokens": 52}  # Example for Telugu
}

def run_experiment(
    model_name: str, 
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame, 
    lang_code: str, 
    base_results_path: str,
    use_few_shot: bool, # Add few-shot flag
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
        # Pass the use_few_shot flag
        results_df = evaluate_qa_baseline(
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            samples_df=samples_df,
            lang_code=lang_code,
            use_few_shot=use_few_shot,
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
    args = parse_args()
    
    # Process models argument to handle comma-separated strings
    processed_models_list = []
    for model_arg in args.models:
        if ',' in model_arg:
            processed_models_list.extend([m.strip() for m in model_arg.split(',')])
        else:
            processed_models_list.append(model_arg.strip())
    args.models = processed_models_list
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define the results directory using the --output argument
    base_results_dir = args.output
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Define the languages to test
    languages = {
        "english": "en",
        "swahili": "sw",
        "telugu": "te"  # Keep Telugu, remove Hausa
    }
    
    # Load the samples for each language
    tydiqa_samples = {}
    for name, code in languages.items():
        logging.info(f"Loading samples for {name} ({code})...")
        
        # Load appropriate number of samples based on percentage
        samples = load_tydiqa_samples(code, sample_percentage=args.sample_percentage)
        
        if not samples.empty:
            logging.info(f"  Loaded {len(samples)} samples for {name}")
            tydiqa_samples[code] = samples
        else:
            logging.warning(f"  No samples loaded for {name}")
    
    # Run the experiments
    all_experiment_summaries = [] # ADDED: To collect all summaries

    for model_name_str in args.models:
        logging.info(f"\n{'='*20} Initializing Model: {model_name_str} {'='*20}")
        tokenizer, model = None, None # Initialize here for the current model scope
        try:
            # Initialize model once per model string
            tokenizer, model = initialize_model(model_name_str) 
        except Exception as e:
            logging.error(f"Failed to initialize model {model_name_str}: {e}. Skipping this model.")
            # Ensure cleanup if partial initialization occurred before error
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue # Next model in args.models

        # THIS LOOP IS NOW NESTED CORRECTLY
        for lang_code, samples_df in tydiqa_samples.items():
            if samples_df.empty:
                continue
            
            logging.info(f"Processing {lang_code} samples with {model_name_str}...")
        
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
            if args.compare_shots:
                shot_settings = [True, False] # Few-shot and Zero-shot
                logging.info("Running both few-shot and zero-shot evaluations as per --compare_shots flag.")
            elif args.few_shot:
                shot_settings = [True] # Few-shot only
                logging.info("Running only few-shot evaluation as per --few_shot flag.")
            else:
                shot_settings = [False] # Zero-shot only (default)
                logging.info("Running only zero-shot evaluation (default). Use --compare_shots or --few_shot for other modes.")
        
            for use_few_shot in shot_settings:
                shot_type = "few-shot" if use_few_shot else "zero-shot"
                logging.info(f"Running {shot_type} evaluation with parameters: {effective_params}")
                
                # MODIFIED: Call run_experiment which now returns a summary dict
                experiment_summary = run_experiment(
                    model_name_str,
                    tokenizer,
                    model,
                    samples_df,
                    lang_code,
                    base_results_dir, # Pass base_results_dir
                    use_few_shot,
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
        logging.info(f"Finished all experiments for model {model_name_str}. Unloading...")
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU memory cache cleared.")

    # ADDED: Generate overall summary and plots
    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        overall_summary_filename = os.path.join(base_results_dir, "summaries", "baseline_qa_ALL_experiments_summary.csv")
        overall_summary_df.to_csv(overall_summary_filename, index=False, float_format='%.4f')
        logging.info(f"Overall summary of all baseline QA experiments saved to: {overall_summary_filename}")
        print("\nOverall Summary:")
        print(overall_summary_df)

        # Generate plots
        plots_dir = os.path.join(base_results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_qa_f1_scores(overall_summary_df, plots_dir)
    else:
        logging.info("No experiment summaries collected. Skipping overall summary and plot generation.")

# ADDED: Plotting function for F1 scores
def plot_qa_f1_scores(summary_df: pd.DataFrame, plots_dir: str):
    """Generate and save a bar plot of F1 scores."""
    if summary_df.empty:
        logging.warning("Summary DataFrame is empty. Skipping F1 score plot generation.")
        return

    plt.figure(figsize=(12, 7))
    try:
        # Create a unique identifier for each experiment run for plotting
        summary_df['experiment_id'] = summary_df['model'] + "-" + summary_df['language'] + "-" + summary_df['shot_type']
        
        sns.barplot(data=summary_df, x='experiment_id', y='f1_score', hue='language')
        plt.xticks(rotation=45, ha='right')
        plt.title('Average F1 Scores for QA Baseline Experiments')
        plt.ylabel('Average F1 Score')
        plt.xlabel('Experiment Configuration (Model-Language-ShotType)')
        plt.tight_layout()
        plot_filename = os.path.join(plots_dir, "baseline_qa_f1_scores.png")
        plt.savefig(plot_filename)
        plt.close()
        logging.info(f"F1 score plot saved to {plot_filename}")
    except Exception as e:
        logging.error(f"Error generating F1 plot: {e}", exc_info=True)

if __name__ == "__main__":
    main()