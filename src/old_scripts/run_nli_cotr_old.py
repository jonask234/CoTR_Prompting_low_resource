import sys
import os

# Add the project root to the Python path - MUST come before other project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)
print(f"Added project root to Python path: {project_root}")

import argparse
import pandas as pd
import numpy as np
import time
from huggingface_hub import login

# Import utility functions using absolute imports
from src.utils.data_loaders.load_xnli import load_xnli_samples
from src.experiments.cotr.nli.nli_cotr import (
    evaluate_nli_cotr, 
    evaluate_nli_cotr_single_prompt,
    calculate_nli_metrics
)
from evaluation.cotr.qa_metrics_cotr import COMET_AVAILABLE
from config import get_token

# Import the new metrics function
from src.evaluation.cotr.translation_metrics import calculate_comet_score

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run NLI CoTR (Chain of Thought Translation Reasoning) experiments using XNLI dataset.")
    parser.add_argument("--langs", nargs='+', default=['en', 'sw', 'ur'], 
                        help="Languages to test from XNLI (default: en, sw, ur)")
    parser.add_argument("--model", type=str, choices=['aya', 'qwen', 'both'], default='both',
                        help="Model to use (default: both)")
    parser.add_argument("--shot-type", type=str, choices=['zero-shot', 'few-shot', 'both'], default='both',
                        help="Shot type to use (default: both)")
    parser.add_argument("--pipeline", type=str, choices=['multi-prompt', 'single-prompt', 'both'], default='both',
                        help="Pipeline type to use (default: both)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples to use per language (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Temperature for generation (default: 0.3)")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Maximum tokens to generate for final classification (default: 20)")
    parser.add_argument("--max-trans-tokens", type=int, default=512,
                        help="Maximum tokens to generate for translations (default: 512)")
    parser.add_argument("--do-sample", action="store_true",
                        help="Use sampling instead of greedy decoding")
    return parser.parse_args()

def run_nli_cotr_experiment(
    model_name: str, 
    samples_df: pd.DataFrame, 
    lang_code: str, 
    base_path: str, 
    pipeline_type: str,
    args
):
    """
    Run an NLI CoTR experiment.
    
    Args:
        model_name: Name of the model to use
        samples_df: DataFrame containing NLI samples
        lang_code: Language code
        base_path: Base path for saving results
        pipeline_type: 'multi-prompt' or 'single-prompt'
        args: Command line arguments with temperature and token settings
        
    Returns:
        DataFrame with results, or None if no results were obtained
    """
    start_time = time.time()
    method_name = "CoTR"
    prefix = "cotr"
    
    if args.shot_type == 'zero-shot':
        shot_name = "zero-shot"
    else:  # few-shot
        shot_name = "few-shot"
    
    print(f"\nRunning {method_name} NLI for {lang_code} with {model_name}")
    print(f"Settings: temp={args.temperature}, max_tokens={args.max_tokens}, max_trans_tokens={args.max_trans_tokens}, do_sample={args.do_sample}")
    print(f"Shot Type: {args.shot_type}, Pipeline: {pipeline_type}")
    
    model_name_short = model_name.split('/')[-1].replace("-", "-").replace("_", "-")
    
    # Create directory structure and filenames based on experiment parameters
    results_dir = os.path.join(base_path, pipeline_type, shot_name, lang_code)
    summaries_dir = os.path.join(base_path, "summaries", pipeline_type, shot_name)
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    
    pipeline_suffix = "mp" if pipeline_type == "multi-prompt" else "sp"
    shot_suffix = "fs" if args.shot_type == "few-shot" else "zs"
    
    results_filename = f"{prefix}_{pipeline_suffix}_{shot_suffix}_nli_{lang_code}_{model_name_short}.csv"
    results_path = os.path.join(results_dir, results_filename)
    
    summary_filename = f"summary_{prefix}_{pipeline_suffix}_{shot_suffix}_nli_{lang_code}_{model_name_short}.csv"
    summary_path = os.path.join(summaries_dir, summary_filename)
    
    # Run the CoTR experiment with generation parameters
    use_few_shot = args.shot_type == 'few-shot'
    
    if pipeline_type == "multi-prompt":
        # Multi-prompt approach: separate translation and reasoning steps
        results_df = evaluate_nli_cotr(
            model_name, 
            samples_df, 
            lang_code,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            max_translation_tokens=args.max_trans_tokens,
            do_sample=args.do_sample,
            use_few_shot=use_few_shot
        )
    else:
        # Single-prompt approach: all in one step
        results_df = evaluate_nli_cotr_single_prompt(
            model_name, 
            samples_df, 
            lang_code,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            do_sample=args.do_sample,
            use_few_shot=use_few_shot
        )
    
    if results_df is None or results_df.empty:
        print(f"Error: No results obtained for {lang_code} with {model_name} using {method_name} ({pipeline_type}).")
        return None
    
    # --- Calculate COMET score for LRL -> EN translation --- 
    # (Applicable only for multi-prompt pipeline where translations are stored)
    avg_comet_score = 0.0
    if pipeline_type == 'multi_prompt' and 'premise' in results_df.columns and 'premise_en' in results_df.columns and 'hypothesis' in results_df.columns and 'hypothesis_en' in results_df.columns:
        premise_src = results_df['premise'].tolist()
        premise_trans = results_df['premise_en'].tolist()
        hypothesis_src = results_df['hypothesis'].tolist()
        hypothesis_trans = results_df['hypothesis_en'].tolist()

        # Combine premise and hypothesis for scoring
        source_texts = premise_src + hypothesis_src
        translated_texts = premise_trans + hypothesis_trans

        # Filter valid pairs
        valid_pairs = [(s, t) for s, t in zip(source_texts, translated_texts) if isinstance(s, str) and isinstance(t, str)]
        if valid_pairs:
            avg_comet_score = calculate_comet_score([p[0] for p in valid_pairs], [p[1] for p in valid_pairs])
        else:
             print("Warning: No valid source/translation pairs found for COMET calculation.")
    elif pipeline_type == 'multi_prompt':
        print("Warning: Missing premise/hypothesis translation columns for COMET calculation.")
    # ------------------------------------------------------

    # Calculate metrics
    metrics = calculate_nli_metrics(results_df)
    
    # Save detailed results
    results_df.to_csv(results_path, index=False)
    print(f"Detailed results saved to {results_path}")
    
    # Create summary DataFrame with key metrics
    summary = {
        'model': model_name,
        'language': lang_code,
        'approach': method_name,
        'pipeline_type': pipeline_type,
        'shot_type': args.shot_type,
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'samples_processed': len(results_df),
        'runtime_seconds': results_df['runtime_seconds'].iloc[0] if 'runtime_seconds' in results_df.columns else 0,
        'runtime_per_sample': results_df['runtime_per_sample'].iloc[0] if 'runtime_per_sample' in results_df.columns else 0,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'do_sample': args.do_sample,
        'comet_score': avg_comet_score
    }
    
    # Add class metrics
    for label in ["entailment", "neutral", "contradiction"]:
        if label in metrics['class_metrics']:
            summary[f'{label}_precision'] = metrics['class_metrics'][label]['precision']
            summary[f'{label}_recall'] = metrics['class_metrics'][label]['recall']
            summary[f'{label}_f1'] = metrics['class_metrics'][label]['f1']
    
    # Add translation quality metrics for multi-prompt pipeline
    if pipeline_type == "multi-prompt" and 'average_translation_quality' in results_df.columns:
        # Translation quality
        summary['premise_translation_quality'] = results_df['premise_translation_quality'].mean() if 'premise_translation_quality' in results_df.columns else 0
        summary['hypothesis_translation_quality'] = results_df['hypothesis_translation_quality'].mean() if 'hypothesis_translation_quality' in results_df.columns else 0
        summary['average_translation_quality'] = results_df['average_translation_quality'].mean() if 'average_translation_quality' in results_df.columns else 0
        
        # COMET scores if available
        if COMET_AVAILABLE:
            if 'comet_source_to_en' in results_df.columns and not results_df['comet_source_to_en'].isnull().all():
                summary['comet_source_to_en'] = np.nanmean(results_df['comet_source_to_en'].astype(float))
            if 'comet_en_to_source' in results_df.columns and not results_df['comet_en_to_source'].isnull().all():
                summary['comet_en_to_source'] = np.nanmean(results_df['comet_en_to_source'].astype(float))
    
    # Save summary metrics
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary metrics saved to {summary_path}")
    
    # Print key metrics
    print(f"\nResults for {lang_code} with {model_name} ({method_name}, {pipeline_type}):")
    print(f"Shot Type: {args.shot_type}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    for label in ["entailment", "neutral", "contradiction"]:
        if label in metrics['class_metrics']:
            print(f"{label.capitalize()} F1: {metrics['class_metrics'][label]['f1']:.4f}")
    
    # Print translation metrics for multi-prompt
    if pipeline_type == "multi-prompt" and 'average_translation_quality' in results_df.columns:
        print("\nTranslation Quality Metrics:")
        print(f"  Premise Translation Quality: {summary.get('premise_translation_quality', 0):.4f}")
        print(f"  Hypothesis Translation Quality: {summary.get('hypothesis_translation_quality', 0):.4f}")
        print(f"  Average Translation Quality: {summary.get('average_translation_quality', 0):.4f}")
        
        if COMET_AVAILABLE and 'comet_source_to_en' in summary:
            print(f"  COMET Source→English: {summary.get('comet_source_to_en', 0):.4f}")
            print(f"  COMET English→Source: {summary.get('comet_en_to_source', 0):.4f}")
    
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
    
    # Load AfriNLI data for each language
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
    
    # Define path for results
    cotr_path = "/work/bbd6522/results/nli/cotr"
    
    # Create plots directory
    plots_path = os.path.join(cotr_path, "plots")
    os.makedirs(plots_path, exist_ok=True)
    
    # Store all results for visualization
    all_results = []
    
    # Determine which shot types to run
    shot_types = []
    if args.shot_type == 'both':
        shot_types = ['zero-shot', 'few-shot']
    else:
        shot_types = [args.shot_type]
    
    # Determine which pipeline types to run
    pipeline_types = []
    if args.pipeline == 'both':
        pipeline_types = ['multi-prompt', 'single-prompt']
    else:
        pipeline_types = [args.pipeline]
    
    # Run experiments
    for model_name in models:
        for lang_code, samples_df in datasets.items():
            if samples_df.empty:
                print(f"WARNING: No samples loaded for {lang_code}. Skipping experiments for {model_name}.")
                continue
            
            for shot_type in shot_types:
                args.shot_type = shot_type
                for pipeline_type in pipeline_types:
                    result_df = run_nli_cotr_experiment(
                        model_name, 
                        samples_df, 
                        lang_code, 
                        cotr_path, 
                        pipeline_type, 
                        args
                    )
                    
                    # Load the summary results for plotting
                    if result_df is not None:
                        # Get the model short name for the filename
                        model_name_short = model_name.split('/')[-1].replace("-", "_")
                        
                        # Load summary from the summaries directory
                        summary_path = os.path.join(cotr_path, "summaries", shot_type)
                        summary_filename = f"summary_nli_{lang_code}_{model_name_short}.csv"
                        full_path = os.path.join(summary_path, summary_filename)
                        
                        if os.path.exists(full_path):
                            try:
                                summary_df = pd.read_csv(full_path)
                                # Add shot_type and pipeline_type explicitly for visualization
                                summary_df['shot_type'] = shot_type
                                summary_df['pipeline_type'] = pipeline_type
                                all_results.append(summary_df)
                            except Exception as e:
                                print(f"Error reading summary file {full_path}: {e}")
    
    # Create visualizations
    if all_results:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Combine all results
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Plot accuracy scores by language, model, shot type and pipeline type
            plt.figure(figsize=(14, 8))
            g = sns.catplot(
                x='language', 
                y='accuracy', 
                hue='model', 
                col='shot_type',
                row='pipeline_type',
                data=combined_results,
                kind='bar',
                height=4,
                aspect=1.2
            )
            g.fig.suptitle('NLI Accuracy by Language, Model, Shot Type and Pipeline', y=1.02)
            # Adjust layout
            plt.tight_layout()
            # Save the figure
            plt.savefig(os.path.join(plots_path, 'nli_accuracy_scores.png'))
            plt.close()
            
            # Plot COMET scores if available
            if 'comet_score' in combined_results.columns:
                plt.figure(figsize=(14, 8))
                g = sns.catplot(
                    x='language', 
                    y='comet_score', 
                    hue='model', 
                    col='shot_type',
                    row='pipeline_type',
                    data=combined_results,
                    kind='bar',
                    height=4,
                    aspect=1.2
                )
                g.fig.suptitle('Translation Quality (COMET Score) by Language, Model, Shot Type and Pipeline', y=1.02)
                # Adjust layout
                plt.tight_layout()
                # Save the figure
                plt.savefig(os.path.join(plots_path, 'nli_comet_scores.png'))
                plt.close()
                
            print(f"Visualizations saved to {plots_path}")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 