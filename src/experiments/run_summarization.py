import sys
import os

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

def run_summarization_experiment(experiment_type, model_name, lang_code, prompt_in_lrl=False):
    """
    Run summarization experiment of specified type.
    
    Args:
        experiment_type: 'baseline' or 'cotr'
        model_name: Name of the model to use
        lang_code: Language code
        prompt_in_lrl: For baseline, whether to use prompts in LRL
    """
    # Skip if trying to use LRL prompts (we've removed this option)
    if prompt_in_lrl:
        print(f"Skipping {lang_code} for {model_name} with LRL prompts (option removed)")
        return
        
    # Create output directory
    base_path = f"/work/bbd6522/results/summarization/{experiment_type}"
    os.makedirs(base_path, exist_ok=True)
    
    # Load XL-Sum data - using 10% of samples
    samples_df = load_xlsum_samples(lang_code, num_samples="10%")
    
    if samples_df.empty:
        print(f"ERROR: No samples loaded for {lang_code}.")
        return
    
    print(f"Loaded {len(samples_df)} samples for {lang_code}")
    
    # For Aya model, reduce sample size for Telugu to avoid CUDA memory issues
    aya_issue = "aya" in model_name.lower() and lang_code == "te"
    if aya_issue:
        # For Aya + Telugu, use half the samples to reduce CUDA memory pressure
        original_len = len(samples_df)
        samples_df = samples_df.sample(frac=0.5, random_state=42)
        print(f"🔄 Reduced samples for Aya + Telugu from {original_len} to {len(samples_df)} to avoid memory issues")
    
    try:
        # Run appropriate experiment
        if experiment_type == 'baseline':
            prompt_tag = "_en_prompt"  # Always use English prompts now
            results_df = evaluate_summarization_baseline(model_name, samples_df, lang_code, prompt_in_lrl=False)
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

def main():
    # Get Hugging Face token
    token = get_token()
    login(token=token)
    
    # Define models to test
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "CohereLabs/aya-expanse-8b"
    ]
    
    # Language codes to test
    languages = ["sw", "te", "en"]
    
    # Print dataset info
    print("\n--- XL-Sum Dataset Statistics ---")
    stats = get_xlsum_stats()
    for lang_code in languages:
        if lang_code in stats:
            lang_stats = stats[lang_code]
            print(f"{lang_code} ({lang_stats.get('name', 'unknown')}): "
                f"Train: {lang_stats.get('train', 'N/A')}, "
                f"Validation: {lang_stats.get('validation', 'N/A')}, "
                f"Test: {lang_stats.get('test', 'N/A')}")
    
    # Run baseline experiments with only English prompts
    print("\n--- Running Baseline Experiments (English prompts) ---")
    for model_name in models:
        for lang_code in languages:
            run_summarization_experiment('baseline', model_name, lang_code, prompt_in_lrl=False)
    
    # Run CoTR experiments
    print("\n--- Running CoTR Experiments ---")
    for model_name in models:
        for lang_code in languages:
            run_summarization_experiment('cotr', model_name, lang_code)

if __name__ == "__main__":
    main() 