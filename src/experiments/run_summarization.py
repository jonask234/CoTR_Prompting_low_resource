import sys
import os
import argparse

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from src.utils.data_loaders.load_xlsum import load_xlsum_samples, get_xlsum_stats
from src.experiments.baseline.summarization.summarization_baseline import evaluate_summarization_baseline
from src.experiments.cotr.summarization.summarization_cotr import evaluate_summarization_cotr
from huggingface_hub import login
from config import get_token
from rouge_score import rouge_scorer

def calculate_rouge_scores(predictions, references):
    """Calculate ROUGE scores for generated summaries."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)
    
    # Calculate average scores
    avg_scores = {
        'rouge1': np.mean(scores['rouge1']),
        'rouge2': np.mean(scores['rouge2']),
        'rougeL': np.mean(scores['rougeL'])
    }
    
    return avg_scores

def run_summarization_experiment(experiment_type, model_name, samples_df, lang_code, dataset_name, base_path):
    """
    Run summarization experiment of specified type.
    
    Args:
        experiment_type: 'baseline' or 'cotr'
        model_name: Name of the model to use
        samples_df: DataFrame containing the samples to use
        lang_code: Language code
        dataset_name: Name of the dataset
        base_path: Base path for storing results
    """
    # Create output directory
    os.makedirs(base_path, exist_ok=True)
    
    if samples_df.empty:
        print(f"ERROR: No samples provided for {lang_code}.")
        return
    
    print(f"Running {experiment_type} experiment with {len(samples_df)} samples")
    
    try:
        # Run appropriate experiment
        if experiment_type == 'baseline':
            prompt_tag = "_en_prompt"  # Always use English prompts now
            results_df = evaluate_summarization_baseline(model_name, samples_df, lang_code)
            output_path = f"{base_path}/{lang_code}"
            method_name = "English Prompt"
        else:  # cotr
            prompt_tag = ""
            results_df = evaluate_summarization_cotr(model_name, samples_df, lang_code)
            output_path = f"{base_path}/{lang_code}"
            method_name = "CoTR"
        
        # Create lang directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        if results_df.empty:
            print(f"ERROR: No results generated for {lang_code} with {model_name} ({experiment_type}).")
            return
        
        # Calculate ROUGE scores
        rouge_scores = calculate_rouge_scores(
            results_df['predicted_summary'].tolist(),
            results_df['reference_summary'].tolist()
        )
        
        # Print results summary
        print(f"\nResults for {lang_code} with {model_name} ({method_name}):")
        print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        # Save detailed results
        model_name_short = model_name.split('/')[-1]
        results_filename = f"{experiment_type}_summarization_{lang_code}_{model_name_short}{prompt_tag}.csv"
        results_df.to_csv(f"{output_path}/{results_filename}", index=False)
        print(f"Detailed results saved to {output_path}/{results_filename}")
        
        # Save summary metrics
        summary = {
            'model': model_name,
            'language': lang_code,
            'method': method_name,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'num_samples': len(results_df)
        }
        
        summary_df = pd.DataFrame([summary])
        summary_path = f"{base_path}/summaries"
        os.makedirs(summary_path, exist_ok=True)
        
        summary_filename = f"summary_{experiment_type}_{lang_code}_{model_name_short}{prompt_tag}.csv"
        summary_df.to_csv(f"{summary_path}/{summary_filename}", index=False, float_format='%.4f')
        print(f"Summary metrics saved to {summary_path}/{summary_filename}")
    
    except Exception as e:
        print(f"ERROR during {experiment_type} experiment for {lang_code} with {model_name}: {str(e)}")
        print("Continuing with next experiment...")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run summarization experiments with reduced sample size')
    parser.add_argument('--lang', type=str, choices=['en', 'sw', 'te'], help='Language code to process')
    parser.add_argument('--model', type=str, choices=['qwen', 'aya'], help='Model to use (short name)')
    parser.add_argument('--approach', type=str, choices=['baseline', 'cotr', 'both'], default='both', 
                        help='Approach to run (baseline, cotr, or both)')
    parser.add_argument('--samples', type=int, default=15, help='Number of samples to process')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Login to HuggingFace
    token = get_token()
    login(token=token)
    
    # Define model mapping
    model_mapping = {
        'qwen': "Qwen/Qwen2.5-7B-Instruct",
        'aya': "CohereLabs/aya-expanse-8b"
    }
    
    # Set up language codes
    if args.lang:
        lang_codes = [args.lang]
    else:
        lang_codes = ["en", "sw", "te"]
    
    # Set up models
    if args.model:
        models = [model_mapping[args.model]]
    else:
        models = [model_mapping['qwen'], model_mapping['aya']]
    
    # Load datasets
    datasets = {}
    for lang_code in lang_codes:
        # Load dataset
        samples_df = load_xlsum_samples(lang_code)
        print(f"Loaded {len(samples_df)} total samples for {lang_code}")
        
        # Use very small sample size (10-15) for fast processing
        # Use even smaller sample size for Telugu with Aya
        if lang_code == "te" and "aya" in " ".join(models).lower():
            num_to_sample = min(len(samples_df), 10)
            print(f"Using very small sample size (10) for Telugu to avoid memory issues")
        else:
            num_to_sample = min(len(samples_df), args.samples)
        
        samples_df = samples_df.sample(n=num_to_sample, random_state=42)
        print(f"Sampling {num_to_sample} samples for resource efficiency for {lang_code}")
        
        datasets[lang_code] = samples_df
    
    # Print dataset statistics
    for lang_code, df in datasets.items():
        print(f"{lang_code}: {len(df)} samples")
    
    # Create directories for results
    baseline_path = "/work/bbd6522/results/summarization/baseline"
    cotr_path = "/work/bbd6522/results/summarization/cotr"
    os.makedirs(baseline_path, exist_ok=True)
    os.makedirs(cotr_path, exist_ok=True)
    
    # Run experiments according to specified approach
    for model_name in models:
        for lang_code in lang_codes:
            if datasets[lang_code].empty:
                print(f"Skipping {lang_code} for {model_name} - no samples available")
                continue
            
            # Run baseline if specified
            if args.approach in ['baseline', 'both']:
                print(f"\n{'='*50}")
                print(f"Running baseline for {model_name} on {lang_code}")
                print(f"{'='*50}")
                run_summarization_experiment("baseline", model_name, datasets[lang_code], 
                                           lang_code, "xlsum", baseline_path)
            
            # Run CoTR if specified
            if args.approach in ['cotr', 'both']:
                print(f"\n{'='*50}")
                print(f"Running CoTR for {model_name} on {lang_code}")
                print(f"{'='*50}")
                run_summarization_experiment("cotr", model_name, datasets[lang_code],
                                           lang_code, "xlsum", cotr_path)

if __name__ == "__main__":
    main() 