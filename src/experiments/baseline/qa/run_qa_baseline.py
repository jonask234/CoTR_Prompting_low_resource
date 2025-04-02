# thesis_project/run_qa_baseline.py

import pandas as pd
import numpy as np
import os
from src.utils.data_loaders.load_mlqa import load_mlqa_samples
from src.experiments.baseline.qa.qa_baseline import evaluate_qa_baseline
from evaluation.baseline.qa_metrics import calculate_qa_f1

def main():
    # Choose model - you can switch to "CohereForAI/aya-23-8B" if preferred
    model_name = "Qwen/Qwen2-7B"
    
    # Load data samples (100 samples each)
    hindi_samples = load_mlqa_samples("mlqa.hi.en", 50)
    vietnamese_samples = load_mlqa_samples("mlqa.vi.en", 50)
    
    # Process baseline for Hindi
    print("\nProcessing Hindi samples with baseline approach...")
    hindi_results = evaluate_qa_baseline(model_name, hindi_samples, "hi")
    
    # Process baseline for Vietnamese
    print("\nProcessing Vietnamese samples with baseline approach...")
    vietnamese_results = evaluate_qa_baseline(model_name, vietnamese_samples, "vi")
    
    # Calculate F1 scores
    print("\nCalculating F1 scores...")
    hindi_results["f1_score"] = hindi_results.apply(calculate_qa_f1, axis=1)
    vietnamese_results["f1_score"] = vietnamese_results.apply(calculate_qa_f1, axis=1)
    
    # Combine results
    all_results = pd.concat([hindi_results, vietnamese_results])
    
    # Calculate average F1 scores
    avg_f1_hindi = hindi_results["f1_score"].mean()
    avg_f1_vietnamese = vietnamese_results["f1_score"].mean()
    avg_f1_overall = all_results["f1_score"].mean()
    
    print(f"\nAverage F1 Score for Hindi: {avg_f1_hindi:.4f}")
    print(f"Average F1 Score for Vietnamese: {avg_f1_vietnamese:.4f}")
    print(f"Average F1 Score Overall: {avg_f1_overall:.4f}")
    
    # Define base results path
    base_results_path = "results/baseline"

    # Create results directories for each language if they don't exist
    hindi_path = os.path.join(base_results_path, "hi")
    vietnamese_path = os.path.join(base_results_path, "vi")
    os.makedirs(hindi_path, exist_ok=True)
    os.makedirs(vietnamese_path, exist_ok=True)

    # Save results
    hindi_results.to_csv(os.path.join(hindi_path, "baseline_qa_hindi.csv"), index=False)
    vietnamese_results.to_csv(os.path.join(vietnamese_path, "baseline_qa_vietnamese.csv"), index=False)

    print(f"\nResults saved to CSV files in the '{base_results_path}' directory, organized by language.")

if __name__ == "__main__":
    main()