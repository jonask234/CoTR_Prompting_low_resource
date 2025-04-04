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
        
        # Save results
        model_name_short = model_name.split('/')[-1]  # Get just the model name without the organization
        results.to_csv(os.path.join(lang_path, f"baseline_qa_{lang_code}_{model_name_short}.csv"), index=False)
        print(f"Results saved to {lang_path}/baseline_qa_{lang_code}_{model_name_short}.csv")
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
    lang_codes = {
        "swahili": "sw", 
        "indonesian": "id" # Changed from Bengali to Indonesian
    }
    
    # Load data samples (50 samples each)
    swahili_samples = load_tydiqa_samples(lang_codes["swahili"], 50)
    indonesian_samples = load_tydiqa_samples(lang_codes["indonesian"], 50) # Changed from Bengali to Indonesian
    
    # Define base results path
    base_results_path = "/work/bbd6522/results/baseline"
    
    # Run experiments for each model and language
    for model_name in models:
        # Process Swahili samples
        run_experiment(model_name, swahili_samples, "sw", base_results_path)
        
        # Process Indonesian samples
        run_experiment(model_name, indonesian_samples, "id", base_results_path) # Changed from Bengali to Indonesian

if __name__ == "__main__":
    main()