# thesis_project/run_qa_baseline.py

import sys
import os
import logging
import argparse
import torch

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
from evaluation.baseline.qa_metrics_baseline import calculate_qa_f1
from huggingface_hub import login
from config import get_token

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run QA baseline experiments with optimized parameters.")
    parser.add_argument("--models", nargs='+', default=["CohereLabs/aya-expanse-8b", "Qwen/Qwen2.5-7B-Instruct"], 
                        help="List of model names or paths for QA.")
    parser.add_argument("--samples", type=int, default=10, 
                        help="Number of samples per language for TyDiQA.")
    parser.add_argument("--few_shot", action="store_true", 
                        help="Enable few-shot prompting if not comparing shots.")
    parser.add_argument("--compare_shots", action="store_true", 
                        help="Run both few-shot and zero-shot evaluations.")
    # Add any other arguments that might be needed based on script usage
    parser.add_argument("--output", type=str, default="/work/bbd6522/results/qa/baseline",
                        help="Base output directory for results and summaries.")
    return parser.parse_args()

def run_experiment(
    model_name: str, 
    samples_df: pd.DataFrame, 
    lang_code: str, 
    base_results_path: str,
    use_few_shot: bool # Add few-shot flag
):
    """
    Run the baseline experiment for a specific model and language.
    
    Args:
        model_name: Name of the model to use
        samples_df: DataFrame containing the samples
        lang_code: Language code
        base_results_path: Base path for saving results
        use_few_shot: Whether to use few-shot prompting
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
        results_df = evaluate_qa_baseline(model_name, samples_df, lang_code, use_few_shot=use_few_shot)
        
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

        # Calculate F1 scores using the DataFrame
        print(f"\nCalculating F1 scores (Predicted LRL/EN vs Ground Truth LRL/EN) - {shot_desc}...")
        results_df["f1_score"] = results_df.apply(calculate_qa_f1, axis=1)
        
        # Calculate average F1 score
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
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Add the absolute path to the root directory to the sys.path
    # PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")) # No longer strictly needed for output path
    # sys.path.insert(0, PROJECT_ROOT) # sys.path modification for imports is still at the top of the file
    
    # Define the results directory using the --output argument
    base_results_dir = args.output
    os.makedirs(base_results_dir, exist_ok=True)
    # os.makedirs(os.path.join(base_results_dir, "summaries"), exist_ok=True) # Not creating subdirectories anymore
    # os.makedirs(os.path.join(base_results_dir, "results"), exist_ok=True)   # Not creating subdirectories anymore
    
    # Load custom functions for calculating metrics
    # from src.utils.evaluation import calculate_qa_f1, calculate_exact_match_score # Commented out due to ModuleNotFoundError
    from src.experiments.baseline.qa.qa_baseline import calculate_qa_f1 # Import F1 from qa_baseline
    # calculate_exact_match_score needs to be defined or imported from its correct location
    
    # Define the languages to test
    languages = {
        "english": "en",
        "swahili": "sw",
        "hausa": "ha",
        "telugu": "te"  # Added Telugu
    }
    
    # Load the samples for each language
    tydiqa_samples = {}
    for name, code in languages.items():
        logging.info(f"Loading samples for {name} ({code})...")
        
        # Load appropriate number of samples
        samples = load_tydiqa_samples(code, args.samples)
        
        if not samples.empty:
            logging.info(f"  Loaded {len(samples)} samples for {name}")
            tydiqa_samples[code] = samples
        else:
            logging.warning(f"  No samples loaded for {name}")
    
    # Run the experiments
    for model_name_str in args.models:
        logging.info(f"\n{'='*20} Initializing Model: {model_name_str} {'='*20}")
        tokenizer, model = None, None
        try:
            # Initialize model once per model string
            tokenizer, model = initialize_model(model_name_str) 
        except Exception as e:
            logging.error(f"Failed to initialize model {model_name_str}: {e}. Skipping this model.")
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue # Next model

    for lang_code, samples_df in tydiqa_samples.items():
        if samples_df.empty:
            continue
        
            logging.info(f"Processing {lang_code} samples with {model_name_str}...")
        
        # Apply optimized parameters
        params = {
            "temperature": STANDARD_PARAMETERS["temperature"],
            "top_p": STANDARD_PARAMETERS["top_p"], 
            "top_k": STANDARD_PARAMETERS["top_k"],
            "max_tokens": STANDARD_PARAMETERS["max_tokens"]
        }
        
        # Language-specific optimizations
        if lang_code == "sw":  # Swahili
            params["temperature"] = 0.2
            params["top_p"] = 0.8
            elif lang_code == "ha":  # Hausa
                params["temperature"] = 0.15 # Example: Adjust if needed for Hausa
                params["top_p"] = 0.75   # Example: Adjust if needed for Hausa
        elif lang_code == "te":  # Telugu
                params["temperature"] = 0.15 # Example: Placeholder for Telugu
                params["top_p"] = 0.75   # Example: Placeholder for Telugu
        
        # Model-specific optimizations
            if "aya" in model_name_str.lower():
            # Aya-specific parameters
            params["temperature"] = max(0.1, params["temperature"] - 0.05)
            elif "qwen" in model_name_str.lower():
            # Qwen-specific parameters
            params["top_p"] = max(0.7, params["top_p"] - 0.05)
            params["top_k"] = 35
        
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
            logging.info(f"Running {shot_type} evaluation with parameters: {params}")
            
            # Run the evaluation with optimized parameters
            results_df = evaluate_qa_baseline(
                        model_name_str,  # Pass model name string for logging
                        tokenizer,       # Pass initialized tokenizer
                        model,           # Pass initialized model
                samples_df, 
                lang_code, 
                use_few_shot=use_few_shot,
                temperature=params["temperature"],
                top_p=params["top_p"],
                top_k=params["top_k"],
                max_tokens=params["max_tokens"]
            )
            
            if results_df.empty:
                    logging.warning(f"No results returned for {lang_code} with {model_name_str} ({shot_type})")
                    continue # Correctly indented to continue to the next shot_setting
            
            # Calculate metrics
            f1_scores = []
                # exact_match_scores = [] # Commented out
            
                for _, row_data in results_df.iterrows(): # Renamed row to row_data to avoid conflict
                    ground_truth = row_data["ground_truth"]
                    predicted = row_data["predicted_answer"]
                
                # Calculate F1 and Exact Match
                f1 = calculate_qa_f1({"text": [ground_truth]}, predicted)
                    # em = calculate_exact_match_score({"text": [ground_truth]}, predicted) # Commented out
                
                f1_scores.append(f1)
                    # exact_match_scores.append(em) # Commented out
            
            # Add metrics to results
            results_df["f1_score"] = f1_scores
                # results_df["exact_match"] = exact_match_scores # Commented out
            
            # Calculate average metrics
            avg_f1 = results_df["f1_score"].mean()
                # avg_em = results_df["exact_match"].mean() # Commented out
            
                logging.info(f"Results for {lang_code} with {model_name_str} ({shot_type}):")
            logging.info(f"  Average F1: {avg_f1:.4f}")
                # logging.info(f"  Average Exact Match: {avg_em:.4f}") # Commented out
            
            # Save the results
                model_name_short = model_name_str.split("/")[-1]
            shot_suffix = "fs" if use_few_shot else "zs"
            
            # Save detailed results
            results_filename = f"qa_baseline_{shot_suffix}_{lang_code}_{model_name_short}.csv"
                results_path = os.path.join(base_results_dir, results_filename) # Save directly into base_results_dir
            results_df.to_csv(results_path, index=False)
            logging.info(f"Detailed results saved to {results_path}")
            
            # Save summary
            summary = {
                "language": lang_code,
                    "model": model_name_str,
                "shot_type": shot_type,
                "f1_score": avg_f1,
                    # "exact_match": avg_em, # Commented out
                "temperature": params["temperature"],
                "top_p": params["top_p"],
                "top_k": params["top_k"],
                "max_tokens": params["max_tokens"],
                "samples_processed": len(results_df)
            }
            
            summary_df = pd.DataFrame([summary])
            summary_filename = f"summary_baseline_{shot_suffix}_qa_tydiqa_{lang_code}_{model_name_short}.csv"
                summary_path = os.path.join(base_results_dir, summary_filename) # Save directly into base_results_dir
                summary_df.to_csv(summary_path, index=False, float_format='%.4f')
            logging.info(f"Summary saved to {summary_path}")

        # Clean up model and tokenizer after all languages and shot settings for it are done
        logging.info(f"Finished all experiments for model {model_name_str}. Unloading...")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU memory cache cleared.")

if __name__ == "__main__":
    main()