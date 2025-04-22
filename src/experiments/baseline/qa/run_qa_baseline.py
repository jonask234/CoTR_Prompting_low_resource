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
        
        # --- Call the evaluation function --- 
        # This function handles model init, looping, and calling process_qa_baseline
        results_df = evaluate_qa_baseline(model_name, samples_df, lang_code)
        
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

        # Calculate F1 scores using the DataFrame
        print("\nCalculating F1 scores (Predicted LRL/EN vs Ground Truth LRL/EN)...")
        results_df["f1_score"] = results_df.apply(calculate_qa_f1, axis=1)
        
        # Calculate average F1 score
        avg_f1 = results_df["f1_score"].mean()
        print(f"\nAverage F1 Score for {lang_code} ({model_name}): {avg_f1:.4f}")
        
        # --- Save Detailed Results (DataFrame) ---
        # Create results directory if it doesn't exist
        lang_path = os.path.join(base_results_path, lang_code)
        os.makedirs(lang_path, exist_ok=True)
        
        # Save detailed results - adjust filename format to include 'tydiqa'
        model_name_short = model_name.split('/')[-1]  # Get just the model name without the organization
        output_filename = f"baseline_qa_tydiqa_{lang_code}_{model_name_short}.csv" 
        # Save the DataFrame (now includes 'ground_truth' and 'f1_score')
        results_df.to_csv(os.path.join(lang_path, output_filename), index=False)
        print(f"Results saved to {lang_path}/{output_filename}")

        # --- Save Summary Metrics ---
        summary = {
            'model': model_name,
            'language': lang_code,
            'f1_score': avg_f1
        }
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(base_results_path, "summaries") 
        os.makedirs(summary_path, exist_ok=True)
        # Update summary filename to use 'tydiqa'
        summary_filename = f"summary_qa_tydiqa_{lang_code}_{model_name_short}.csv" 
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
    # Get Hugging Face token
    token = get_token()
    login(token=token)
    
    # Define models to test
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "CohereLabs/aya-expanse-8b"
    ]
    
    # Language codes for TyDiQA - focusing on Swahili and Telugu (LRLs) with ground truths, plus English (HRL)
    lang_codes = { 
        "swahili": "sw", 
        "telugu": "te",
        "english": "en"  # Added English (HRL) for comparison
    }
    
    # Load TyDiQA data samples (with LRL ground truths)
    print("\n--- Loading TyDiQA-GoldP Data with LRL Ground Truths ---")
    tydiqa_samples = {}
    for name, code in lang_codes.items():
        # Use validation split which is more appropriate for evaluation
        print(f"Loading ALL samples for {name} ({code}) from validation split to calculate 10%...")
        # Load the full dataset first by setting num_samples=None
        full_samples_df = load_tydiqa_samples(code, num_samples=None, split='validation')
        
        if not full_samples_df.empty:
            total_loaded = len(full_samples_df)
            # Calculate 10% of samples, ensuring at least 1 sample if dataset is very small
            num_to_sample = max(1, int(total_loaded * 0.1)) 
            print(f"  Loaded {total_loaded} total samples. Sampling {num_to_sample} (10%)...")
            # Sample 10% of the data
            tydiqa_samples[code] = full_samples_df.sample(n=num_to_sample, random_state=42) 
            print(f"  Finished sampling {len(tydiqa_samples[code])} samples for {code}.")
        else:
            print(f"  WARNING: No TyDiQA samples loaded for {name} ({code}).")
            tydiqa_samples[code] = pd.DataFrame()
    
    # Define base results paths - USE ONLY ONE PATH
    base_results_path = "/work/bbd6522/results/qa/baseline" 
    
    # Run TyDiQA experiments with LRL ground truths
    print("\n--- Running TyDiQA Baseline Experiments ---")
    for model_name in models:
        for code, samples_df in tydiqa_samples.items():
            if not samples_df.empty:
                run_experiment(model_name, samples_df, code, base_results_path)
            else:
                print(f"  Skipping {code} due to empty dataset.")

if __name__ == "__main__":
    main()