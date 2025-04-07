import sys
import os

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
from src.experiments.cotr.qa.qa_cotr import evaluate_qa_cotr
from evaluation.cotr.qa_metrics_cotr import calculate_qa_f1, calculate_translation_quality
from huggingface_hub import login
from config import get_token

def run_experiment(model_name, samples_df, lang_code, base_results_path):
    """
    Run the CoTR experiment for a specific model and language.
    
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
        print(f"\nProcessing {lang_code} samples with {model_name} (CoTR)...")
        results = evaluate_qa_cotr(model_name, samples_df, lang_code)
        
        # Check if results are empty
        if results.empty:
            print(f"WARNING: No results generated for {lang_code} with {model_name}. Skipping metrics calculation.")
            return
            
        # Calculate F1 scores
        print("\nCalculating F1 scores...")
        results["f1_score"] = results.apply(calculate_qa_f1, axis=1)
        
        # Calculate translation quality metrics
        print("Calculating translation quality metrics...")
        translation_metrics = results.apply(calculate_translation_quality, axis=1)
        results['question_translation_quality'] = [m['question_translation_quality'] for m in translation_metrics]
        results['answer_translation_quality'] = [m['answer_translation_quality'] for m in translation_metrics]
        results['average_translation_quality'] = [m['average_translation_quality'] for m in translation_metrics]
        
        # Check if COMET metrics are available
        # Ensure translation_metrics is not empty before accessing index 0
        has_comet = False
        if not translation_metrics.empty:
             has_comet = 'comet_source_to_en' in translation_metrics.iloc[0]
        
        if has_comet:
            results['comet_source_to_en'] = [m.get('comet_source_to_en', 0.0) for m in translation_metrics]
            results['comet_en_to_source'] = [m.get('comet_en_to_source', 0.0) for m in translation_metrics]
        
        # Calculate average metrics
        avg_f1 = results["f1_score"].mean()
        avg_q_trans = results["question_translation_quality"].mean()
        avg_a_trans = results["answer_translation_quality"].mean()
        avg_trans = results["average_translation_quality"].mean()
        
        print(f"\nResults for {lang_code} ({model_name}, CoTR):")
        print(f"Average F1 Score: {avg_f1:.4f}")
        print(f"Average Translation Quality Metrics:")
        
        if has_comet:
            # Ensure columns exist before calculating mean
            avg_comet_src_en = results['comet_source_to_en'].mean() if 'comet_source_to_en' in results.columns else float('nan')
            avg_comet_en_src = results['comet_en_to_source'].mean() if 'comet_en_to_source' in results.columns else float('nan')
            print(f"  COMET Source→English: {avg_comet_src_en:.4f}")
            print(f"  COMET English→Source: {avg_comet_en_src:.4f}")
            print(f"  Normalized Question Translation Quality: {avg_q_trans:.4f}")
            print(f"  Normalized Answer Translation Quality: {avg_a_trans:.4f}")
            print(f"  Normalized Average Translation Quality: {avg_trans:.4f}")
        else:
            print(f"  Question Translation Quality: {avg_q_trans:.4f}")
            print(f"  Answer Translation Quality: {avg_a_trans:.4f}")
            print(f"  Average Translation Quality: {avg_trans:.4f}")
            print(f"  (Using token overlap metrics as COMET was not available)")
        
        # Create results directory if it doesn't exist
        lang_path = os.path.join(base_results_path, lang_code)
        os.makedirs(lang_path, exist_ok=True)
        
        # Save results - revert filename format
        model_name_short = model_name.split('/')[-1]  # Get just the model name without the organization
        output_filename = f"cotr_qa_{lang_code}_{model_name_short}.csv" # Reverted filename
        results.to_csv(os.path.join(lang_path, output_filename), index=False)
        print(f"Results saved to {lang_path}/{output_filename}")
        
        # Save summary metrics to a separate file - revert filename format
        summary = {
            'model': model_name,
            'language': lang_code,
            'f1_score': avg_f1,
            'question_translation_quality': avg_q_trans,
            'answer_translation_quality': avg_a_trans,
            'average_translation_quality': avg_trans
        }
        
        if has_comet:
             # Use the calculated averages, handle potential NaN
            summary['comet_source_to_en'] = avg_comet_src_en
            summary['comet_en_to_source'] = avg_comet_en_src
        
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(base_results_path, "summaries")
        os.makedirs(summary_path, exist_ok=True)
        summary_filename = f"summary_{lang_code}_{model_name_short}.csv" # Reverted filename
        summary_df.to_csv(os.path.join(summary_path, summary_filename), index=False)
        print(f"Summary metrics saved to {summary_path}/{summary_filename}")
        
    except Exception as e:
        print(f"Error processing {model_name} for {lang_code}: {str(e)}") # Reverted error message format
        # Keep the specific error handling for restricted models
        if "Access to model" in str(e) and "is restricted" in str(e):
            print("\nTo use the Aya model, you need to:")
            print("1. Create a Hugging Face account at https://huggingface.co/join")
            print("2. Accept the model's terms of use at https://huggingface.co/CohereForAI/aya-23-8B")
            print("3. Generate an access token at https://huggingface.co/settings/tokens")
            print("4. Run this script again with your token")
            sys.exit(1) # Exit specifically for this auth error
        # For other errors, maybe just log and continue if possible, or re-raise

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
    base_results_path = "/work/bbd6522/results/cotr" 
    
    # Run TyDiQA experiments 
    print("\n--- Running TyDiQA CoTR Experiments ---")
    for model_name in models:
        run_experiment(model_name, swahili_samples, lang_codes["swahili"], base_results_path)
        run_experiment(model_name, indonesian_samples, lang_codes["indonesian"], base_results_path)

if __name__ == "__main__":
    main()