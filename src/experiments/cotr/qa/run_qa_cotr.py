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
        
        # --- Call the main CoTR evaluation function --- 
        # This function handles model loading, translations, QA, back-translation, 
        # and extracting the ground truth string. It returns the results DataFrame.
        results_df = evaluate_qa_cotr(model_name, samples_df, lang_code)

        # --- Check if results were generated --- 
        if results_df.empty:
            print(f"WARNING: evaluate_qa_cotr returned empty DataFrame for {lang_code} with {model_name}. Skipping further processing.")
            return

        # --- Process the results ---
        # Make sure ground_truth column exists and is populated
        if 'ground_truth' not in results_df.columns or results_df['ground_truth'].isnull().all():
            print(f"WARNING: 'ground_truth' column missing or empty in results for {lang_code} with {model_name}. Cannot calculate F1.")
            return

        # Drop rows with missing ground truth
        results_df.dropna(subset=['ground_truth'], inplace=True)
        if results_df.empty:
            print(f"WARNING: No valid samples remaining after checking ground truth for {lang_code} with {model_name}. Cannot calculate F1.")
            return

        # --- Calculate Metrics --- 
        # The results_df returned by evaluate_qa_cotr should already have:
        # 'predicted_answer', 'ground_truth' (as string), 'question', 
        # 'question_en', 'answer_en' etc.

        # Calculate F1 scores using the DataFrame
        print("\nCalculating F1 scores...")
        # calculate_qa_f1 expects row['ground_truth'] and row['predicted_answer']
        results_df["f1_score"] = results_df.apply(calculate_qa_f1, axis=1)
        
        # Calculate translation quality metrics
        print("Calculating translation quality metrics...")
        # Apply the function that returns a dictionary of metrics
        translation_metrics_series = results_df.apply(calculate_translation_quality, axis=1)
        
        # Populate DataFrame columns from the Series of metric dictionaries
        # Use .get() with default values for safety
        results_df['question_translation_quality'] = translation_metrics_series.apply(lambda m: m.get('question_translation_quality', 0.0))
        results_df['answer_translation_quality'] = translation_metrics_series.apply(lambda m: m.get('answer_translation_quality', 0.0))
        results_df['average_translation_quality'] = translation_metrics_series.apply(lambda m: m.get('average_translation_quality', 0.0))
        results_df['comet_source_to_en'] = translation_metrics_series.apply(lambda m: m.get('comet_source_to_en', float('nan')))
        results_df['comet_en_to_source'] = translation_metrics_series.apply(lambda m: m.get('comet_en_to_source', float('nan')))

        # --- Calculate Average Metrics --- 
        avg_f1 = results_df["f1_score"].mean()
        avg_q_trans = results_df["question_translation_quality"].mean()
        avg_a_trans = results_df["answer_translation_quality"].mean()
        avg_trans = results_df["average_translation_quality"].mean()
        # Use nanmean for COMET scores as they might be NaN if COMET failed/unavailable
        avg_comet_src_en = np.nanmean(results_df['comet_source_to_en'].astype(float))
        avg_comet_en_src = np.nanmean(results_df['comet_en_to_source'].astype(float))

        # --- Print Summary --- 
        print(f"\nResults for {lang_code} ({model_name}, CoTR):")
        print(f"Average F1 Score: {avg_f1:.4f}")
        print(f"Average Translation Quality Metrics:")
        # Check if COMET scores are actually present (not all NaN)
        has_comet = not results_df['comet_source_to_en'].isnull().all()
        if has_comet:
            print(f"  COMET Source→English: {avg_comet_src_en:.4f}")
            print(f"  COMET English→Source: {avg_comet_en_src:.4f}")
            # Print normalized scores alongside raw COMET if desired, but averages are calculated above
            print(f"  Normalized Question Translation Quality (0-1 scale): {avg_q_trans:.4f}")
            print(f"  Normalized Answer Translation Quality (0-1 scale): {avg_a_trans:.4f}")
            print(f"  Normalized Average Translation Quality (0-1 scale): {avg_trans:.4f}")
        else:
            # If no COMET scores, just show the fallback/normalized quality
            print(f"  Question Translation Quality: {avg_q_trans:.4f}")
            print(f"  Answer Translation Quality: {avg_a_trans:.4f}")
            print(f"  Average Translation Quality: {avg_trans:.4f}")
            print(f"  (Using token overlap or COMET failed/unavailable)")
        
        # --- Save Detailed Results (DataFrame) --- 
        lang_path = os.path.join(base_results_path, lang_code)
        os.makedirs(lang_path, exist_ok=True)
        model_name_short = model_name.split('/')[-1]  
        output_filename = f"cotr_qa_tydiqa_{lang_code}_{model_name_short}.csv"
        # Save the DataFrame which now includes all metrics
        results_df.to_csv(os.path.join(lang_path, output_filename), index=False)
        print(f"Results saved to {lang_path}/{output_filename}")
        
        # --- Save Summary Metrics --- 
        summary = {
            'model': model_name,
            'language': lang_code,
            'f1_score': avg_f1,
            'question_translation_quality': avg_q_trans, 
            'answer_translation_quality': avg_a_trans,
            'average_translation_quality': avg_trans
        }
        if has_comet:
            summary['comet_source_to_en'] = avg_comet_src_en
            summary['comet_en_to_source'] = avg_comet_en_src
        
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(base_results_path, "summaries")
        os.makedirs(summary_path, exist_ok=True)
        summary_filename = f"summary_qa_tydiqa_{lang_code}_{model_name_short}.csv" 
        summary_df.to_csv(os.path.join(summary_path, summary_filename), index=False, float_format='%.4f')
        print(f"Summary metrics saved to {summary_path}/{summary_filename}")
        
    except Exception as e:
        print(f"Error during run_experiment for {model_name} for {lang_code} (CoTR): {str(e)}") 
        # Keep specific error handling for restricted models
        if "Access to model" in str(e) and "is restricted" in str(e):
            print("\nTo use the Aya model, you need to:")
            print("1. Create a Hugging Face account at https://huggingface.co/join")
            print("2. Accept the model's terms of use at https://huggingface.co/CohereLabs/aya-expanse-8b")
            print("3. Generate an access token at https://huggingface.co/settings/tokens")
            print("4. Run this script again with your token")
            sys.exit(1) 
        # Consider re-raising or logging other exceptions
        # raise e # Uncomment to stop execution on other errors

def main():
    # Get Hugging Face token
    token = get_token()
    login(token=token)
    
    # Define models to test
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "CohereLabs/aya-expanse-8b"
    ]
    
    # Language codes for TyDiQA - focusing on Swahili and Telugu (with LRL ground truths), plus English (HRL)
    lang_codes = { 
        "swahili": "sw", 
        "telugu": "te",
        "english": "en"  # Added English (HRL) for comparison
    }
    
    # Load TyDiQA data samples (validation split, all samples)
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
    
    # Define base results path
    base_results_path = "/work/bbd6522/results/qa/cotr" 
    
    # Run TyDiQA CoTR experiments
    print("\n--- Running TyDiQA CoTR Experiments with LRL Ground Truth Evaluation ---")
    for model_name in models:
        for code, samples_df in tydiqa_samples.items():
            if not samples_df.empty:
                run_experiment(model_name, samples_df, code, base_results_path)
            else:
                print(f"  Skipping {code} due to empty dataset.")

if __name__ == "__main__":
    main()