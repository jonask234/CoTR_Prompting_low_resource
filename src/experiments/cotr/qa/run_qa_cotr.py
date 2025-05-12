import sys
import os
import logging

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to the Python path
# Go up four levels from the script's directory: qa -> cotr -> experiments -> src -> project_root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from src.utils.data_loaders.load_tydiqa import load_tydiqa_samples
from src.experiments.cotr.qa.qa_cotr import (
    evaluate_qa_cotr, 
    evaluate_qa_cotr_single_prompt,
    initialize_model
)
from evaluation.cotr.qa_metrics_cotr import calculate_qa_f1, calculate_translation_quality, COMET_AVAILABLE
from huggingface_hub import login
from config import get_token
import argparse
import torch
from typing import Any

# Define standard parameters for both baseline and CoTR - EXACTLY matching baseline parameters
STANDARD_PARAMETERS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 50
}

# Language-specific standard parameters
LANGUAGE_PARAMETERS = {
    "sw": {  # Swahili
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": STANDARD_PARAMETERS["top_k"],
        "max_tokens": STANDARD_PARAMETERS["max_tokens"]
    },
    "te": {  # Telugu
        "temperature": 0.15,
        "top_p": 0.75,
        "top_k": STANDARD_PARAMETERS["top_k"],
        "max_tokens": STANDARD_PARAMETERS["max_tokens"]
    }
}

def run_experiment(
    model_name: str, 
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame, 
    lang_code: str, 
    base_results_path: str,
    pipeline_type: str = 'multi_prompt',
    use_few_shot: bool = True,
    temperature: float = STANDARD_PARAMETERS["temperature"],
    top_p: float = STANDARD_PARAMETERS["top_p"],
    top_k: float = STANDARD_PARAMETERS["top_k"],
    max_tokens: int = STANDARD_PARAMETERS["max_tokens"],
    max_translation_tokens: int = 200
):
    """
    Run the Chain of Translation Prompting (CoTR) experiment for QA.
    
    Args:
        model_name: Name of the model to use
        tokenizer: Initialized tokenizer
        model: Initialized model
        samples_df: DataFrame containing the samples
        lang_code: Language code
        base_results_path: Base path for saving results
        pipeline_type: Type of CoTR pipeline ('multi_prompt' or 'single_prompt')
        use_few_shot: Whether to use few-shot prompting
        temperature: Temperature for generation (this value is now expected to be final)
        top_p: Top-p for generation (this value is now expected to be final)
        top_k: Top-k for generation (this value is now expected to be final)
        max_tokens: Maximum tokens for generation (this value is now expected to be final)
        max_translation_tokens: Maximum tokens for translation
    """
    # The parameters temperature, top_p, top_k, max_tokens received by this function
    # are now considered final and pre-calculated by the main() function to include
    # CLI overrides, language-specific values, and model-specific adjustments.
    # Thus, this function will directly use these passed-in values for those specific args.
    # Other parameters like max_translation_tokens are handled as before.
    
    # Check if samples_df is empty
    if samples_df.empty:
        print(f"Empty samples dataframe for {lang_code}, skipping...")
        return
    
    # Create directories for results
    results_dir = os.path.join(base_results_path, "results")
    summaries_dir = os.path.join(base_results_path, "summaries")
    plots_dir = os.path.join(base_results_path, "plots")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a short model name for file naming
    model_short = model_name.split('/')[-1]
    shot_type = "fs" if use_few_shot else "zs"  # few-shot or zero-shot
    pipeline_short = "mp" if pipeline_type == "multi_prompt" else "sp"  # multi-prompt or single-prompt
    
    # File paths
    results_file = os.path.join(results_dir, f"results_cotr_{pipeline_short}_{shot_type}_qa_tydiqa_{lang_code}_{model_short}.csv")
    summary_file = os.path.join(summaries_dir, f"summary_cotr_{pipeline_short}_{shot_type}_qa_tydiqa_{lang_code}_{model_short}.csv")
    
    # Check if results already exist to avoid duplicate computation
    if os.path.exists(results_file):
        print(f"Results file {results_file} already exists. Skipping...")
        return
    
    # Print experiment settings
    print(f"Running CoTR QA experiment:")
    print(f"  Model: {model_name}")
    print(f"  Language: {lang_code}")
    print(f"  Pipeline: {pipeline_type}")
    print(f"  Shot type: {'few-shot' if use_few_shot else 'zero-shot'}")
    print(f"  Parameters: temp={temperature}, top_p={top_p}, top_k={top_k}, max_tokens={max_tokens}")
    print(f"  Samples: {len(samples_df)}")
    
    # Run the appropriate evaluation based on pipeline type
    try:
        if pipeline_type == 'multi_prompt':
            # Multi-prompt approach (translate -> process -> translate)
            results_df = evaluate_qa_cotr(
                model_name=model_name, 
                tokenizer=tokenizer,
                model=model,
                samples_df=samples_df, 
                lang_code=lang_code,
                use_few_shot=use_few_shot,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                max_translation_tokens=max_translation_tokens  # Only needed for multi-prompt
            )
        else:
            # Single-prompt approach
            results_df = evaluate_qa_cotr_single_prompt(
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
            )
        
        if results_df.empty:
            print(f"No results were returned for {lang_code} with {model_name}.")
            return
            
        # Save the results dataframe
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
        
        print(f"DEBUG: Attempting to calculate metrics for {lang_code}, {model_name}, {pipeline_type}, {shot_type}") # DEBUG
        f1_scores = []
        comet_scores_en_to_lrl_list = []

        for _, row in results_df.iterrows():
            try: # DEBUG: Add try-except for individual F1 calculation
                f1_score_val = calculate_qa_f1({'ground_truth': row['ground_truth'], 'predicted_answer': row['predicted_answer']})
                f1_scores.append(f1_score_val)
            except Exception as e_f1:
                print(f"DEBUG: Error calculating F1 for a row: {e_f1}")
                f1_scores.append(0.0) # Append a default value or handle as appropriate

            if 'comet_score_en_to_lrl' in row and pd.notna(row['comet_score_en_to_lrl']):
                comet_scores_en_to_lrl_list.append(row['comet_score_en_to_lrl'])
        
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_comet_en_to_lrl = np.mean(comet_scores_en_to_lrl_list) if comet_scores_en_to_lrl_list else None
        print(f"DEBUG: Avg F1 calculated: {avg_f1}, Avg COMET: {avg_comet_en_to_lrl}") # DEBUG
        
        # Create a summary dataframe
        summary_data = {
            'model': [model_short],
            'language': [lang_code],
            'pipeline': [pipeline_type],
            'shot_type': ['few-shot' if use_few_shot else 'zero-shot'],
            'samples': [len(results_df)],
            'f1_score': [avg_f1],
            'temperature': [temperature],
            'top_p': [top_p],
            'top_k': [top_k],
            'max_tokens': [max_tokens]
        }
        # NEW: Add COMET score to summary if available
        if avg_comet_en_to_lrl is not None:
            summary_data['avg_comet_en_to_lrl'] = [avg_comet_en_to_lrl]
        
        summary_df = pd.DataFrame(summary_data)
        print(f"DEBUG: Individual summary_data for {summary_file}:\\n{summary_data}") # DEBUG
        print(f"DEBUG: Individual summary_df for {summary_file}:\\n{summary_df.head()}") # DEBUG
        
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")
        
        # Print the F1 score
        print(f"\nAverage F1 score for {lang_code} with {model_name} ({pipeline_type}, {'few-shot' if use_few_shot else 'zero-shot'}): {avg_f1:.4f}")
        # NEW: Print COMET score if available
        if avg_comet_en_to_lrl is not None:
            print(f"Average COMET score (EN->LRL Answer Back-Translation) for {lang_code} with {model_name} ({pipeline_type}, {'few-shot' if use_few_shot else 'zero-shot'}): {avg_comet_en_to_lrl:.4f}")
        
        return summary_data # MODIFIED: Return the summary data dictionary
        
    except Exception as e:
        print(f"Error in experiment for {lang_code} with {model_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run the QA-CoTR experiments."""
    parser = argparse.ArgumentParser(description='Run QA Chain of Translation Prompting (CoTR) experiments')
    parser.add_argument('--models', type=str, default="CohereLabs/aya-expanse-8b,Qwen/Qwen2.5-7B-Instruct", help='Comma-separated list of models to use')
    parser.add_argument('--langs', nargs='+', default=['en', 'sw', 'ha'], help='Languages to evaluate')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to use (default: 10)')
    parser.add_argument('--pipeline_types', nargs='+', default=['multi_prompt', 'single_prompt'], choices=['multi_prompt', 'single_prompt'], help='Pipeline types to run')
    parser.add_argument('--shot_settings', nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help='Shot settings to evaluate')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode (first 5 samples only)')
    parser.add_argument('--hf-token', type=str, help='HuggingFace API token (if needed)')
    parser.add_argument('--base_output_dir', type=str, default="/work/bbd6522/results/qa/cotr", help="Base directory for all outputs.")

    # ADDED: Arguments for generation parameters (to override defaults if needed for non-grid search runs)
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for generation (overrides standard if set)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p for generation (overrides standard if set)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k for generation (overrides standard if set)")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens for generation (overrides standard if set)")
    parser.add_argument("--max_translation_tokens", type=int, default=200, help="Max new tokens for translation steps (used in multi_prompt)")
    
    args = parser.parse_args()
    
    # Disable tokenizers parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set up HuggingFace token
    token = get_token()
    login(token=token)
    
    # Define models to test
    if args.models:
        models_list = [m.strip() for m in args.models.split(',')]
    else:
        models_list = []
    
    # Create a results directory if it doesn't exist
    # base_results_path = f"results/cotr/qa/{'-'.join(args.langs)}" # This was a bit too specific and might not align with base_output_dir intent
    # os.makedirs(base_results_path, exist_ok=True) # This would create a subdir named after langs
    
    # Ensure the overall base output directory and its subdirectories for summaries and plots exist
    summaries_output_dir = os.path.join(args.base_output_dir, "summaries")
    plots_output_dir = os.path.join(args.base_output_dir, "plots")
    os.makedirs(summaries_output_dir, exist_ok=True)
    os.makedirs(plots_output_dir, exist_ok=True) 
    print(f"All CoTR QA experiment outputs will be saved under: {args.base_output_dir}")

    all_experiment_summaries = [] # MODIFIED: To collect all summaries

    for model_name_str in models_list:
        print(f"\n{'='*20} Initializing Model: {model_name_str} {'='*20}")
        tokenizer, model = None, None
        try:
            model, tokenizer = initialize_model(model_name_str)
        except Exception as e:
            logging.error(f"Failed to initialize model {model_name_str}: {e}. Skipping this model.")
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue # Next model

    for lang_code in args.langs:
        print(f"\n--- Processing Language: {lang_code} for model {model_name_str} ---")
        samples_df = load_tydiqa_samples(lang_code, num_samples=args.samples)
        
        if samples_df.empty:
                print(f"No samples found for {lang_code}, skipping for this model.")
                continue # Correctly indented inside lang_code loop
            
        if args.test_mode:
                print("Running in TEST MODE with first 5 samples only for this language-model pair.")
            samples_df = samples_df.head(5)
            
        # Determine effective generation parameters for this language and potential CLI overrides
        # These are the FINAL parameters that will be passed down.
        effective_temp = args.temperature
        effective_top_p = args.top_p
        effective_top_k = args.top_k
        effective_max_tokens = args.max_tokens

         # Fallback to language-specific, then to general standard if CLI override is not present
        if effective_temp is None:
            effective_temp = LANGUAGE_PARAMETERS.get(lang_code, {}).get("temperature", STANDARD_PARAMETERS["temperature"])
        if effective_top_p is None:
            effective_top_p = LANGUAGE_PARAMETERS.get(lang_code, {}).get("top_p", STANDARD_PARAMETERS["top_p"])
        if effective_top_k is None:
            effective_top_k = LANGUAGE_PARAMETERS.get(lang_code, {}).get("top_k", STANDARD_PARAMETERS["top_k"])
        if effective_max_tokens is None:
            effective_max_tokens = LANGUAGE_PARAMETERS.get(lang_code, {}).get("max_tokens", STANDARD_PARAMETERS["max_tokens"])
        
            # Apply model-specific adjustments AFTER resolving language/CLI overrides
        if "aya" in model_name_str.lower():
            effective_temp = max(0.1, effective_temp - 0.05) # Example: Aya specific adjustment
            print(f"Applied Aya-specific adjustment to temperature: {effective_temp}")
        elif "qwen" in model_name_str.lower():
            effective_top_p = max(0.7, effective_top_p - 0.05) # Example: Qwen specific
            if effective_top_k == STANDARD_PARAMETERS["top_k"]: # Only override top_k if it wasn't set by CLI or lang
                effective_top_k = 35 
            print(f"Applied Qwen-specific adjustments: top_p={effective_top_p}, top_k={effective_top_k}")

        for pipeline_type_to_run in args.pipeline_types:
            for shot_setting_str in args.shot_settings:
                use_few_shot_to_run = (shot_setting_str == 'few_shot')
                    
                print(f"\nEvaluating {model_name_str} on {lang_code} using {pipeline_type_to_run} ({shot_setting_str})...")
                    # Pass the FINALIZED effective parameters to run_experiment
                print(f"Final Effective params for run_experiment: temp={effective_temp}, top_p={effective_top_p}, top_k={effective_top_k}, max_tokens={effective_max_tokens}, max_translation_tokens={args.max_translation_tokens}")

                summary_result = run_experiment(
                    model_name=model_name_str,
                    tokenizer=tokenizer_main, # Pass initialized tokenizer
                    model=model_main,         # Pass initialized model
                samples_df=samples_df,
                lang_code=lang_code,
                    base_results_path=args.base_output_dir,
                    pipeline_type=pipeline_type_to_run,
                    use_few_shot=use_few_shot_to_run,
                    temperature=effective_temp,         # Pass final temp
                    top_p=effective_top_p,             # Pass final top_p
                    top_k=effective_top_k,             # Pass final top_k
                    max_tokens=effective_max_tokens,     # Pass final max_tokens
                    max_translation_tokens=args.max_translation_tokens
                )
                if summary_result:
                    all_experiment_summaries.append(summary_result)

        print(f"\nFinished all experiments for model {model_name_str}. Unloading...")
        del model_main # Clean up
        del tokenizer_main # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU memory cache cleared.")
            
    if all_experiment_summaries:
        print(f"DEBUG: Collected {len(all_experiment_summaries)} individual summaries.")
        print(f"DEBUG: First collected summary (if any): {all_experiment_summaries[0] if all_experiment_summaries else 'None'}")
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        overall_summary_filename = os.path.join(summaries_output_dir, "cotr_qa_ALL_experiments_summary.csv")
        print(f"DEBUG: Overall summary_df to be saved to {overall_summary_filename}:\\n{overall_summary_df.head()}")
        overall_summary_df.to_csv(overall_summary_filename, index=False)
        print(f"\nOverall summary of all experiments saved to: {overall_summary_filename}")
        print(overall_summary_df)

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            print("Attempting to generate plots...")
            plot_qa_f1_scores(overall_summary_df, plots_output_dir)
            if 'avg_comet_en_to_lrl' in overall_summary_df.columns:
                 plot_qa_comet_scores(overall_summary_df, plots_output_dir)
            print(f"Plots saved to: {plots_output_dir}")
        except ImportError:
            print("matplotlib or seaborn not installed. Skipping plot generation.")
        except Exception as e:
            print(f"Error during plot generation: {e}")
    else:
        print("No summaries collected, skipping overall summary and plot generation.")
            
    print("\nAll CoTR QA experiments completed!")

# Placeholder for plotting functions - to be defined properly
def plot_qa_f1_scores(summary_df, plots_dir):
    if summary_df.empty:
        print("Summary DataFrame is empty, skipping F1 plot.")
        return
    plt.figure(figsize=(15, 8))
    try:
        # Create a combined categorical column for better plotting if many categories
        summary_df['experiment_config'] = summary_df['language'] + '-' + summary_df['model'] + '-' + summary_df['pipeline'] + '-' + summary_df['shot_type']
        sns.barplot(data=summary_df, x='experiment_config', y='f1_score')
        plt.xticks(rotation=45, ha='right')
        plt.title('Average F1 Scores for CoTR QA Experiments')
        plt.ylabel('Average F1 Score')
        plt.xlabel('Experiment Configuration')
        plt.tight_layout()
        plot_filename = os.path.join(plots_dir, "cotr_qa_f1_scores.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"F1 score plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error generating F1 plot: {e}")

def plot_qa_comet_scores(summary_df, plots_dir):
    if summary_df.empty or 'avg_comet_en_to_lrl' not in summary_df.columns:
        print("Summary DataFrame is empty or missing COMET scores, skipping COMET plot.")
        return
    plt.figure(figsize=(15, 8))
    try:
        # Ensure a unique identifier if not already created
        if 'experiment_config' not in summary_df.columns:
             summary_df['experiment_config'] = summary_df['language'] + '-' + summary_df['model'] + '-' + summary_df['pipeline'] + '-' + summary_df['shot_type']
        sns.barplot(data=summary_df.dropna(subset=['avg_comet_en_to_lrl']), x='experiment_config', y='avg_comet_en_to_lrl') # Drop NA for plotting COMET
        plt.xticks(rotation=45, ha='right')
        plt.title('Average COMET Scores (EN->LRL Answer Back-Translation)')
        plt.ylabel('Average COMET Score')
        plt.xlabel('Experiment Configuration')
        plt.tight_layout()
        plot_filename = os.path.join(plots_dir, "cotr_qa_comet_scores.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"COMET score plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error generating COMET plot: {e}")

if __name__ == "__main__":
    main()