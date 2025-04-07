# thesis_project/run_qa_baseline.py

import sys
import os

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
from src.experiments.baseline.qa.qa_baseline import evaluate_qa_baseline
from evaluation.baseline.qa_metrics_baseline import calculate_qa_f1
from huggingface_hub import login
from config import get_token

def run_experiment(model_name, samples_df, lang_code, base_results_path):
    """
    Run the baseline experiment for a specific model and language.
    
    Args:
        model_name: Name of the model to use
        samples_df: DataFrame containing the samples
        lang_code: Language code
        base_results_path: Base path for saving results
    """
    # Check if samples_df is empty
    if samples_df.empty:
        print(f"WARNING: No samples provided for {lang_code}. Skipping experiment for {model_name}.")
        return
        
    try:
        print(f"\nProcessing {lang_code} samples with {model_name}...")
        results = evaluate_qa_baseline(model_name, samples_df, lang_code)
        
        # Check if results are empty (could happen if model fails on all samples)
        if results.empty:
            print(f"WARNING: No results generated for {lang_code} with {model_name}. Skipping metrics calculation.")
            return
            
        # Calculate F1 scores
        print("\nCalculating F1 scores...")
        results["f1_score"] = results.apply(calculate_qa_f1, axis=1)
        
        # Calculate average F1 score
        avg_f1 = results["f1_score"].mean()
        print(f"\nAverage F1 Score for {lang_code} ({model_name}): {avg_f1:.4f}")
        
        # Create results directory if it doesn't exist
        lang_path = os.path.join(base_results_path, lang_code)
        os.makedirs(lang_path, exist_ok=True)
        
        # Save results - revert filename format
        model_name_short = model_name.split('/')[-1]  # Get just the model name without the organization
        output_filename = f"baseline_qa_{lang_code}_{model_name_short}.csv" # Reverted filename
        results.to_csv(os.path.join(lang_path, output_filename), index=False)
        print(f"Results saved to {lang_path}/{output_filename}")
    except Exception as e:
        print(f"Error processing {model_name} for {lang_code}: {str(e)}")
        # Keep the specific error handling for restricted models
        if "Access to model" in str(e) and "is restricted" in str(e):
            print("\nTo use the Aya model, you need to:")
            print("1. Create a Hugging Face account at https://huggingface.co/join")
            print("2. Accept the model's terms of use at https://huggingface.co/CohereForAI/aya-23-8B")
            print("3. Generate an access token at https://huggingface.co/settings/tokens")
            print("4. Run this script again with your token")
            sys.exit(1) # Exit specifically for this auth error
        # For other errors, maybe just log and continue if possible, or re-raise
        # Depending on desired behavior

def main():
    # Get Hugging Face token
    token = get_token()
    login(token=token)
    
    # Define models to test
    models = [
        "Qwen/Qwen2-7B",
        "CohereForAI/aya-23-8B"
    ]
    
    # Language codes for TyDi QA
    lang_codes = { # Renamed back
        "swahili": "sw", 
        "indonesian": "id" 
    }
    
    num_samples_per_lang = 50 # Number of samples to load
    
    # Load TyDiQA data samples
    print("\n--- Loading TyDiQA Data ---")
    swahili_samples = load_tydiqa_samples(lang_codes["swahili"], num_samples_per_lang)
    indonesian_samples = load_tydiqa_samples(lang_codes["indonesian"], num_samples_per_lang)
    
    # Define base results paths - USE ONLY ONE PATH
    base_results_path = "/work/bbd6522/results/baseline" 
    
    # Run TyDiQA experiments 
    print("\n--- Running TyDiQA Baseline Experiments ---")
    for model_name in models:
        run_experiment(model_name, swahili_samples, lang_codes["swahili"], base_results_path)
        run_experiment(model_name, indonesian_samples, lang_codes["indonesian"], base_results_path)

if __name__ == "__main__":
    main()