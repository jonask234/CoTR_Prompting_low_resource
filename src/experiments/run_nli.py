import sys
import os
import argparse
import pandas as pd
import numpy as np
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import utility functions using absolute imports
from src.utils.data_loaders.load_xnli import load_xnli_samples
from config import get_token

# Import COMET for translation quality if available
from evaluation.cotr.qa_metrics_cotr import COMET_AVAILABLE # Assuming qa_metrics_cotr has COMET_AVAILABLE
from src.evaluation.cotr.translation_metrics import calculate_comet_score

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run NLI experiments with baseline and CoTR approaches.")
    parser.add_argument("--langs", nargs='+', default=['en', 'sw', 'ha'], 
                        help="Languages to test (default: en, sw, ha)")
    parser.add_argument("--model", type=str, choices=['aya', 'qwen', 'both'], default='both',
                        help="Model to use (default: both)")
    parser.add_argument("--approach", type=str, choices=['baseline', 'cotr', 'both'], default='both',
                        help="Approach to use (default: both)")
    parser.add_argument("--pipeline-type", type=str, choices=['multi-prompt', 'single-prompt', 'both'], default='both',
                        help="Pipeline type to use for CoTR (default: both)")
    parser.add_argument("--shot-type", type=str, choices=['zero-shot', 'few-shot', 'both'], default='both',
                        help="Shot type to use (default: both)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples to use per language (default: 100)")
    # Add new arguments for temperature and token count
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for generation (default: 0.3 from standard)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximum tokens to generate for NLI label (default: 20 from standard)")
    parser.add_argument("--translation-temp", type=float, default=None, # RENAMED from --translation-temp
                        help="Temperature for translation in CoTR (default: 0.3 from standard)")
    parser.add_argument("--max-trans-tokens", type=int, default=None,
                        help="Maximum tokens to generate for translations (default: 512 from standard)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p for generation")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k for generation")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty for generation")
    parser.add_argument("--do-sample", action="store_true",
                        help="Use sampling instead of greedy decoding")
    return parser.parse_args()

# Define STANDARD_PARAMETERS for NLI (can be tuned)
STANDARD_PARAMETERS = {
    "temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_tokens": 5, "repetition_penalty": 1.1,
    "trans_temp": 0.3, "trans_top_p": 0.9, "trans_top_k": 40, "max_trans_tokens": 256, "trans_repetition_penalty": 1.0
}

# Define LANGUAGE_PARAMETERS for NLI (can be tuned, examples shown)
LANGUAGE_PARAMETERS = {
    "sw": { # Swahili
        "temperature": 0.25, "top_p": 0.85, "top_k": 35, "max_tokens": 5, "repetition_penalty": 1.15,
        "trans_temp": 0.25, "trans_top_p": 0.85, "trans_top_k": 35, "max_trans_tokens": 200, "trans_repetition_penalty": 1.05
    },
    "ha": { # Hausa
        "temperature": 0.25, "top_p": 0.85, "top_k": 35, "max_tokens": 5, "repetition_penalty": 1.15,
        "trans_temp": 0.25, "trans_top_p": 0.85, "trans_top_k": 35, "max_trans_tokens": 200, "trans_repetition_penalty": 1.05
    },
    "ur": { # Urdu
        "temperature": 0.2, "top_p": 0.8, "top_k": 30, "max_tokens": 5, "repetition_penalty": 1.2,
        "trans_temp": 0.2, "trans_top_p": 0.8, "trans_top_k": 30, "max_trans_tokens": 220, "trans_repetition_penalty": 1.1
    }
    # Add other languages like 'en', 'te' if specific NLI tuning is desired
}

def run_nli_experiment(experiment_type, model_name, samples_df, lang_code, base_path, args):
    """
    Run an NLI experiment of the specified type.
    
    Args:
        experiment_type: 'baseline' or 'cotr'
        model_name: Name of the model to use
        samples_df: DataFrame containing NLI samples
        lang_code: Language code
        base_path: Base path for saving results
        args: Command line arguments with temperature and token settings
        
    Returns:
        DataFrame with results, or None if no results were obtained
    """
    start_time = time.time()
    
    # Import the appropriate module based on experiment type
    if experiment_type == 'baseline':
        from src.experiments.baseline.nli.nli_baseline import evaluate_nli_baseline, calculate_nli_metrics
        method_name = "Baseline"
        prefix = "baseline"
        
        # Always use English prompt instructions regardless of shot type
        use_lrl_prompt = False
        if args.shot_type == 'zero-shot':
            shot_name = "zero-shot"
        else:  # few-shot
            shot_name = "few-shot"
    else:  # cotr
        from src.experiments.cotr.nli.nli_cotr import (
            evaluate_nli_cotr, evaluate_nli_cotr_single_prompt, 
            calculate_nli_metrics, calculate_translation_quality
        )
        method_name = "CoTR"
        prefix = "cotr"
        
        # Determine pipeline type and shot type names
        if args.pipeline_type == 'single-prompt':
            pipeline_name = "single-prompt"
        else:  # multi-prompt
            pipeline_name = "multi-prompt"
            
        if args.shot_type == 'zero-shot':
            shot_name = "zero-shot"
        else:  # few-shot
            shot_name = "few-shot"
    
    print(f"\nRunning {method_name} NLI for {lang_code} with {model_name}")
    print(f"Settings: temp={args.temperature}, max_tokens={args.max_tokens}, do_sample={args.do_sample}")
    if experiment_type == 'cotr':
        print(f"Pipeline: {args.pipeline_type if experiment_type == 'cotr' else 'N/A'}, Shot Type: {args.shot_type}")
        print(f"Translation temperature: {args.translation_temp}")
    
    # Determine effective parameters
    # NLI specific (for baseline and NLI step in CoTR)
    eff_temp = args.temperature if args.temperature is not None else (LANGUAGE_PARAMETERS.get(lang_code, {}).get("temperature", STANDARD_PARAMETERS["temperature"]))
    eff_top_p = args.top_p if args.top_p is not None else (LANGUAGE_PARAMETERS.get(lang_code, {}).get("top_p", STANDARD_PARAMETERS["top_p"]))
    eff_top_k = args.top_k if args.top_k is not None else (LANGUAGE_PARAMETERS.get(lang_code, {}).get("top_k", STANDARD_PARAMETERS["top_k"]))
    eff_max_tokens = args.max_tokens if args.max_tokens is not None else (LANGUAGE_PARAMETERS.get(lang_code, {}).get("max_tokens", STANDARD_PARAMETERS["max_tokens"]))
    eff_rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else (LANGUAGE_PARAMETERS.get(lang_code, {}).get("repetition_penalty", STANDARD_PARAMETERS["repetition_penalty"]))

    # Translation specific (for CoTR multi-prompt)
    eff_trans_temp = args.translation_temp if args.translation_temp is not None else (LANGUAGE_PARAMETERS.get(lang_code, {}).get("trans_temp", STANDARD_PARAMETERS["trans_temp"]))
    eff_trans_top_p = args.top_p if args.top_p is not None else (LANGUAGE_PARAMETERS.get(lang_code, {}).get("trans_top_p", STANDARD_PARAMETERS["trans_top_p"])) # Assuming top_p for trans can be same as NLI if not specified
    eff_trans_top_k = args.top_k if args.top_k is not None else (LANGUAGE_PARAMETERS.get(lang_code, {}).get("trans_top_k", STANDARD_PARAMETERS["trans_top_k"])) # Assuming top_k for trans can be same as NLI if not specified
    eff_max_trans_tokens = args.max_trans_tokens if args.max_trans_tokens is not None else (LANGUAGE_PARAMETERS.get(lang_code, {}).get("max_trans_tokens", STANDARD_PARAMETERS["max_trans_tokens"])) 
    eff_trans_rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else (LANGUAGE_PARAMETERS.get(lang_code, {}).get("trans_repetition_penalty", STANDARD_PARAMETERS["trans_repetition_penalty"])) # Assuming rep_penalty for trans can be same as NLI if not specified

    # Model-specific adjustments (applied on top of defaults/lang-specific)
    if "aya" in model_name.lower():
        eff_temp = max(0.1, eff_temp - 0.05)
        eff_trans_temp = max(0.1, eff_trans_temp - 0.05)
        print(f"Applied Aya-specific adjustments: NLI temp={eff_temp}, Trans temp={eff_trans_temp}")
    elif "qwen" in model_name.lower():
        eff_top_p = max(0.7, eff_top_p - 0.05)
        eff_top_k = 35 # Qwen specific NLI top_k
        eff_trans_top_p = max(0.7, eff_trans_top_p - 0.05)
        eff_trans_top_k = 35 # Qwen specific trans top_k
        print(f"Applied Qwen-specific adjustments: NLI top_p={eff_top_p}, NLI top_k={eff_top_k}, Trans top_p={eff_trans_top_p}, Trans top_k={eff_trans_top_k}")

    print(f"Effective NLI Params: temp={eff_temp}, top_p={eff_top_p}, top_k={eff_top_k}, max_tokens={eff_max_tokens}, rep_penalty={eff_rep_penalty}")
    if experiment_type == 'cotr' and args.pipeline_type == 'multi-prompt':
        print(f"Effective Translation Params: temp={eff_trans_temp}, top_p={eff_trans_top_p}, top_k={eff_trans_top_k}, max_tokens={eff_max_trans_tokens}, rep_penalty={eff_trans_rep_penalty}")
    
    model_name_short = model_name.split('/')[-1].replace("-", "-").replace("_", "-")
    
    # Create directory structure and filenames based on experiment parameters
    if experiment_type == 'baseline':
        # Baseline results path: baseline/{lang}/{shot_name}/baseline_nli_{lang}_{model}.csv
        results_dir = os.path.join(base_path, lang_code, shot_name)
        summaries_dir = os.path.join(base_path, "summaries", shot_name)
    else:  # cotr
        # CoTR results path: cotr/{lang}/{pipeline_name}/{shot_name}/cotr_nli_{lang}_{model}.csv
        results_dir = os.path.join(base_path, lang_code, pipeline_name, shot_name)
        summaries_dir = os.path.join(base_path, "summaries", pipeline_name, shot_name)
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    
    results_filename = f"{prefix}_nli_{lang_code}_{model_name_short}.csv"
    results_path = os.path.join(results_dir, results_filename)
    
    summary_filename = f"summary_nli_{lang_code}_{model_name_short}.csv"
    summary_path = os.path.join(summaries_dir, summary_filename)
    
    # Run the appropriate experiment with generation parameters
    if experiment_type == 'baseline':
        use_few_shot = args.shot_type == 'few-shot'
        results_df = evaluate_nli_baseline(
            model_name, samples_df, lang_code,
            prompt_in_lrl=use_lrl_prompt,
            temperature=eff_temp,
            max_new_tokens=eff_max_tokens,
            do_sample=args.do_sample,
            use_few_shot=use_few_shot,
            top_p=eff_top_p,
            top_k=eff_top_k,
            repetition_penalty=eff_rep_penalty
        )
    else:  # cotr
        use_few_shot = args.shot_type == 'few-shot'
        if args.pipeline_type == 'single-prompt':
            # Single prompt approach
            results_df = evaluate_nli_cotr_single_prompt(
                model_name, samples_df, lang_code,
                temperature=eff_temp,
                max_new_tokens=eff_max_tokens,
                do_sample=args.do_sample,
                use_few_shot=use_few_shot,
                top_p=eff_top_p,
                top_k=eff_top_k,
                repetition_penalty=eff_rep_penalty
            )
        else:  # multi-prompt
        results_df = evaluate_nli_cotr(
            model_name, samples_df, lang_code,
                temperature=eff_temp,
                max_new_tokens=eff_max_tokens,
                max_translation_tokens=eff_max_trans_tokens,
            do_sample=args.do_sample,
                use_few_shot=use_few_shot,
                top_p=eff_top_p,
                top_k=eff_top_k,
                repetition_penalty=eff_rep_penalty
        )
    
    if results_df.empty:
        print(f"Error: No results obtained for {lang_code} with {model_name} using {method_name}.")
        return None
    
    # Calculate metrics
    metrics = calculate_nli_metrics(results_df)
    
    # Calculate translation quality metrics for CoTR
    if experiment_type == 'cotr' and args.pipeline_type == 'multi-prompt':
        try:
            results_df = calculate_translation_quality(results_df)
        except Exception as e:
            print(f"Error calculating translation quality: {e}")
    
    # Save detailed results
    results_df.to_csv(results_path, index=False)
    print(f"Detailed results saved to {results_path}")
    
    # Create summary DataFrame with key metrics
    summary = {
        'model': model_name,
        'language': lang_code,
        'approach': method_name,
        'pipeline_type': args.pipeline_type if experiment_type == 'cotr' else 'N/A',
        'shot_type': args.shot_type,
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'samples_processed': len(results_df),
        'runtime_seconds': results_df['runtime_seconds'].iloc[0] if 'runtime_seconds' in results_df.columns else 0,
        'runtime_per_sample': results_df['runtime_per_sample'].iloc[0] if 'runtime_per_sample' in results_df.columns else 0,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'do_sample': args.do_sample,
        'eff_temp': eff_temp,
        'eff_top_p': eff_top_p,
        'eff_top_k': eff_top_k,
        'eff_repetition_penalty': eff_rep_penalty
    }
    
    # Add class metrics
    for label in ["entailment", "neutral", "contradiction"]:
        if label in metrics['class_metrics']:
            summary[f'{label}_precision'] = metrics['class_metrics'][label]['precision']
            summary[f'{label}_recall'] = metrics['class_metrics'][label]['recall']
            summary[f'{label}_f1'] = metrics['class_metrics'][label]['f1']
    
    # Add translation quality metrics for CoTR
    if experiment_type == 'cotr' and args.pipeline_type == 'multi-prompt':
        summary['eff_trans_temp'] = eff_trans_temp
        summary['eff_trans_top_p'] = eff_trans_top_p
        summary['eff_trans_top_k'] = eff_trans_top_k
        summary['eff_max_trans_tokens'] = eff_max_trans_tokens
        summary['eff_trans_repetition_penalty'] = eff_trans_rep_penalty
        
        # Calculate COMET score for premise and hypothesis translations
        if COMET_AVAILABLE and 'premise_en' in results_df.columns and 'hypothesis_en' in results_df.columns:
            premise_translations = results_df[['premise', 'premise_en']].dropna()
            hypothesis_translations = results_df[['hypothesis', 'hypothesis_en']].dropna()
            
            avg_comet_premise = 0.0
            if not premise_translations.empty:
                avg_comet_premise = calculate_comet_score(premise_translations['premise'].tolist(), premise_translations['premise_en'].tolist())
            
            avg_comet_hypothesis = 0.0
            if not hypothesis_translations.empty:
                avg_comet_hypothesis = calculate_comet_score(hypothesis_translations['hypothesis'].tolist(), hypothesis_translations['hypothesis_en'].tolist())
                
            summary['avg_comet_premise_lrl_en'] = avg_comet_premise
            summary['avg_comet_hypothesis_lrl_en'] = avg_comet_hypothesis
            summary['avg_comet_overall_lrl_en'] = (avg_comet_premise + avg_comet_hypothesis) / 2 if (avg_comet_premise + avg_comet_hypothesis) > 0 else 0
            print(f"Avg COMET (Premise LRL->EN): {avg_comet_premise:.4f}")
            print(f"Avg COMET (Hypothesis LRL->EN): {avg_comet_hypothesis:.4f}")
    
    # Save summary metrics
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary metrics saved to {summary_path}")
    
    # Print key metrics
    print(f"\nResults for {lang_code} with {model_name} ({method_name}):")
    print(f"Pipeline: {args.pipeline_type if experiment_type == 'cotr' else 'N/A'}, Shot Type: {args.shot_type}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    for label in ["entailment", "neutral", "contradiction"]:
        if label in metrics['class_metrics']:
            print(f"{label.capitalize()} F1: {metrics['class_metrics'][label]['f1']:.4f}")
    
    if experiment_type == 'cotr' and args.pipeline_type == 'multi-prompt':
        if 'avg_comet_overall_lrl_en' in summary:
            print(f"Avg Overall COMET (LRL->EN): {summary['avg_comet_overall_lrl_en']:.4f}")
    
    runtime = time.time() - start_time
    print(f"Total runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    
    # Return the results DataFrame for inclusion in consolidated results
    return results_df

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup
    token = get_token()
    login(token=token)
    
    # Define models based on user choice
    if args.model == 'aya':
        models = ["CohereLabs/aya-expanse-8b"]
    elif args.model == 'qwen':
        models = ["Qwen/Qwen2.5-7B-Instruct"]
    else:  # both
        models = [
            "Qwen/Qwen2.5-7B-Instruct",
            "CohereLabs/aya-expanse-8b"
        ]
    
    # Load AfriNLI data for each language (function kept as load_xnli_samples for compatibility)
    datasets = {}
    for lang_code in args.langs:
        print(f"\n--- Loading AfriNLI Data for {lang_code} ---")
        
        # Load all samples first to get total count
        full_samples_df = load_xnli_samples(lang_code, num_samples=None, split='test')
        
        if not full_samples_df.empty:
            total_loaded = len(full_samples_df)
            
            # For this experiment, use the specified number of samples
            num_to_sample = min(total_loaded, args.samples)
            print(f"  Loaded {total_loaded} total samples. Sampling {num_to_sample} samples...")
            
            # Sample the data
            datasets[lang_code] = full_samples_df.sample(n=num_to_sample, random_state=42)
            print(f"  Finished sampling {len(datasets[lang_code])} samples for {lang_code}.")
        else:
            print(f"  No samples loaded for {lang_code}, cannot sample.")
            datasets[lang_code] = pd.DataFrame()  # Store empty DataFrame
    
    # Define paths for results
    baseline_path = "/work/bbd6522/results/nli/baseline"
    cotr_path = "/work/bbd6522/results/nli/cotr"
    
    # Determine which pipeline types to run
    pipeline_types = []
    if args.pipeline_type == 'both':
        pipeline_types = ['multi-prompt', 'single-prompt']
    else:
        pipeline_types = [args.pipeline_type]
    
    # Determine which shot types to run
    shot_types = []
    if args.shot_type == 'both':
        shot_types = ['zero-shot', 'few-shot']
    else:
        shot_types = [args.shot_type]
    
    # Run experiments
    for model_name in models:
        for lang_code, samples_df in datasets.items():
            if samples_df.empty:
                print(f"WARNING: No samples loaded for {lang_code}. Skipping experiments for {model_name}.")
                continue
            
            if args.approach in ['baseline', 'both']:
                for shot_type in shot_types:
                    args.shot_type = shot_type
                    args.pipeline_type = 'N/A'  # Not applicable for baseline
                run_nli_experiment('baseline', model_name, samples_df, lang_code, baseline_path, args)
            
            if args.approach in ['cotr', 'both']:
                for pipeline_type in pipeline_types:
                    args.pipeline_type = pipeline_type
                    for shot_type in shot_types:
                        args.shot_type = shot_type
                run_nli_experiment('cotr', model_name, samples_df, lang_code, cotr_path, args)

if __name__ == "__main__":
    main() 