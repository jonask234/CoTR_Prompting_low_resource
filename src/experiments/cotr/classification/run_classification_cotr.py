import sys
import os

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Add project root to Python path for evaluation module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from huggingface_hub import login

# Try to import config from various locations
try:
    from utils.config import get_token
except ImportError:
    try:
        from config import get_token
    except ImportError:
        # Fallback to define a basic get_token function if the module can't be found
        def get_token():
            print("WARNING: Could not import get_token, using environment variable HF_TOKEN instead")
            return os.environ.get("HF_TOKEN", "")

# Project specific imports
from utils.data_loaders.load_masakhanews import load_masakhanews_samples
from experiments.cotr.classification.classification_cotr import evaluate_classification_cotr
from evaluation.classification_metrics import calculate_classification_metrics
# Import COMET calculation function and availability flag from QA metrics
from evaluation.cotr.qa_metrics_cotr import calculate_comet_score, COMET_AVAILABLE

def run_classification_experiment_cotr(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str,
    dataset_name: str, # e.g., 'masakhanews' or 'xlsum'
    base_results_path: str,
    use_lrl_ground_truth: bool = False
):
    """
    Run the CoTR text classification experiment for a specific model and language.
    
    Args:
        model_name: Name of the model to use
        samples_df: DataFrame with input samples
        lang_code: Language code
        dataset_name: Name of the dataset
        base_results_path: Base path for results
        use_lrl_ground_truth: Flag indicating if ground truths are in LRL (for XL-Sum)
    """
    if samples_df.empty:
        print(f"WARNING: No samples provided for {lang_code} ({dataset_name}). Skipping CoTR experiment for {model_name}.")
        return

    try:
        print(f"\nProcessing {lang_code} CoTR classification ({dataset_name}) with {model_name}...")
        # This function performs translation and English classification
        results_df = evaluate_classification_cotr(model_name, samples_df, lang_code, use_lrl_ground_truth)

        if results_df.empty:
            print(f"WARNING: No CoTR results generated for {lang_code} ({dataset_name}) with {model_name}. Skipping metrics.")
            return

        # Calculate classification metrics (Accuracy, Macro F1)
        print("\nCalculating classification metrics...")
        metrics = calculate_classification_metrics(results_df)
        avg_accuracy = metrics.get('accuracy', float('nan'))
        avg_macro_f1 = metrics.get('macro_f1', float('nan'))
        
        # Collect per-class metrics dynamically
        per_class_metrics = {k: v for k, v in metrics.items() if k not in ['accuracy', 'macro_f1']}

        # --- Calculate Translation Quality using COMET (if available) ---
        avg_comet_source_to_en = np.nan
        avg_translation_quality = np.nan
        # Add metrics for back-translation
        avg_comet_en_to_lrl = np.nan
        avg_backtranslation_quality = np.nan

        if COMET_AVAILABLE and 'original_text' in results_df.columns and 'text_en' in results_df.columns:
            print("Calculating translation quality metrics using COMET...")
            try:
                # Calculate forward translation quality (LRL to English)
                comet_scores = calculate_comet_score(
                    results_df['original_text'].tolist(), 
                    results_df['text_en'].tolist()
                )
                results_df['comet_source_to_en'] = comet_scores
                
                # Normalize COMET score to 0-1 range for easier interpretation
                results_df['translation_quality'] = results_df['comet_source_to_en'].apply(
                    lambda score: max(0, (score + 1) / 2) if pd.notna(score) else np.nan
                )
                
                # Calculate averages, ignoring NaNs
                avg_comet_source_to_en = np.nanmean(results_df['comet_source_to_en'])
                avg_translation_quality = np.nanmean(results_df['translation_quality'])

                # If we have back-translation data (English to LRL), calculate quality for that too
                if 'predicted_label_en' in results_df.columns and 'predicted_label_lrl' in results_df.columns:
                    # We can only measure translation quality for non-unknown predictions
                    valid_translations = results_df[results_df['predicted_label_en'] != "[Unknown]"]
                    
                    if not valid_translations.empty:
                        print("Calculating back-translation quality metrics (English to LRL)...")
                        # For back-translation, we're translating single words, so we'll measure differently
                        # We'll check if the translation was successful by comparing if they're different
                        # (if they're the same, it suggests the model didn't translate it)
                        results_df['en_to_lrl_different'] = results_df.apply(
                            lambda row: 0 if row['predicted_label_en'] == row['predicted_label_lrl'] else 1, 
                            axis=1
                        )
                        # Calculate the percentage of successful back-translations
                        avg_backtranslation_success = results_df['en_to_lrl_different'].mean()
                        print(f"Back-translation success rate: {avg_backtranslation_success:.4f}")
                
            except Exception as e:
                print(f"WARN: COMET score calculation failed: {e}")
                # Ensure columns exist even if calculation fails, filled with NaN
                if 'comet_source_to_en' not in results_df.columns:
                    results_df['comet_source_to_en'] = np.nan
                if 'translation_quality' not in results_df.columns:
                    results_df['translation_quality'] = np.nan
        elif not COMET_AVAILABLE:
             print("WARN: COMET model not available. Skipping translation quality calculation.")
             # Ensure columns exist filled with NaN if COMET is unavailable
             results_df['comet_source_to_en'] = np.nan
             results_df['translation_quality'] = np.nan
        else:
            print("WARN: Could not calculate translation quality (missing 'original_text' or 'text_en' columns).")
            # Ensure columns exist filled with NaN if columns missing
            results_df['comet_source_to_en'] = np.nan
            results_df['translation_quality'] = np.nan
        # --- End Translation Quality Calculation ---

        print(f"\nOverall Metrics for {lang_code} ({dataset_name}) ({model_name}, CoTR):")
        print(f"  LRL Ground Truth Eval: {use_lrl_ground_truth}")
        print(f"  Accuracy: {avg_accuracy:.4f}")
        print(f"  Macro F1-Score: {avg_macro_f1:.4f}")
        # Print per-class metrics
        for label, value in per_class_metrics.items():
             metric_name = label.split('_')[-1].capitalize()
             class_name = label.replace(f'_{metric_name.lower()}', '') 
             if metric_name == 'Precision':
                 print(f"  {class_name.capitalize()} (Prec): {value:.4f}", end='')
             elif metric_name == 'Recall':
                 print(f" / (Recall): {value:.4f}")
        print() # Newline after per-class metrics
        # Print COMET score if available
        if not np.isnan(avg_comet_source_to_en):
             print(f"  Avg. Raw COMET Score (Source -> En): {avg_comet_source_to_en:.4f}")
             print(f"  Avg. Normalized Translation Quality (0-1): {avg_translation_quality:.4f}")
        else:
             print(f"  Avg. Translation Quality: Not Available")
        
        # Print back-translation information if available
        if 'en_to_lrl_different' in results_df.columns:
            print(f"  Avg. Back-translation Success Rate: {avg_backtranslation_success:.4f}")

        # --- Save Results --- 
        lang_path = os.path.join(base_results_path, lang_code)
        os.makedirs(lang_path, exist_ok=True)

        # Save detailed results per sample
        model_name_short = model_name.split('/')[-1]
        lrl_suffix = "_lrleval" if use_lrl_ground_truth else ""
        output_filename = f"cotr_classification_{dataset_name}_{lang_code}_{model_name_short}{lrl_suffix}.csv"
        
        # Select columns to save (include translation quality metrics)
        cols_to_save = ['id', 'original_text', 'text_en', 'ground_truth_label', 'predicted_label', 
                        'predicted_label_en', 'predicted_label_lrl', 'language', 'lrl_evaluation']
        if 'comet_source_to_en' in results_df.columns:
            cols_to_save.append('comet_source_to_en')
        if 'translation_quality' in results_df.columns:
            cols_to_save.append('translation_quality')
        if 'en_to_lrl_different' in results_df.columns:
            cols_to_save.append('en_to_lrl_different')
            
        # Filter columns that exist in the DataFrame
        cols_to_save = [col for col in cols_to_save if col in results_df.columns]
            
        results_df[cols_to_save].to_csv(os.path.join(lang_path, output_filename), index=False)
        print(f"Detailed results saved to {lang_path}/{output_filename}")

        # Save summary metrics
        summary = {
            'model': model_name,
            'language': lang_code,
            'dataset': dataset_name,
            'pipeline': 'cotr',
            'lrl_evaluation': use_lrl_ground_truth,
            'accuracy': avg_accuracy,
            'macro_f1': avg_macro_f1,
            # Add per-class metrics dynamically
            **per_class_metrics,
            # Add translation metrics (will be NaN if not calculated)
            'avg_comet_source_to_en': avg_comet_source_to_en,
            'avg_translation_quality': avg_translation_quality,
            'avg_comet_en_to_lrl': avg_comet_en_to_lrl,
            'avg_backtranslation_quality': avg_backtranslation_quality
        }
        # Add back-translation success if calculated
        if 'en_to_lrl_different' in results_df.columns:
            summary['avg_backtranslation_success'] = avg_backtranslation_success
            
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(base_results_path, "summaries")
        os.makedirs(summary_path, exist_ok=True)
        summary_filename = f"summary_cotr_{dataset_name}_{lang_code}{lrl_suffix}_{model_name_short}.csv"
        summary_df.to_csv(os.path.join(summary_path, summary_filename), index=False)
        print(f"Summary metrics saved to {summary_path}/{summary_filename}")

    except Exception as e:
        print(f"ERROR during CoTR classification experiment for {model_name}, {lang_code} ({dataset_name}): {e}")
        import traceback
        traceback.print_exc()

def main():
    # Setup
    token = get_token()
    login(token=token)

    models = [
        "Qwen/Qwen2-7B",
        "CohereForAI/aya-23-8B"
    ]

    # --- MasakhaNEWS Dataset (English ground truth) ---
    masakhanews_langs = {
        "swahili": "swa",
        "hausa": "hau"
    }
    masakhanews_name = "masakhanews"
    masakhanews_samples = {}
    
    # Load MasakhaNEWS data (English ground truth labels) from local files
    print(f"\n--- Loading {masakhanews_name.capitalize()} Data ---")
    for name, code in masakhanews_langs.items():
        print(f"Loading ALL samples for {name} ({code}) from local test split to calculate 10%...")
        # Load the full dataset first by setting num_samples=None
        full_samples_df = load_masakhanews_samples(code, num_samples=None, split='test')
        
        if not full_samples_df.empty:
            total_loaded = len(full_samples_df)
            # Calculate 10% of samples, ensuring at least 1 sample if dataset is very small
            num_to_sample = max(1, int(total_loaded * 0.1)) 
            print(f"  Loaded {total_loaded} total samples. Sampling {num_to_sample} (10%)...")
            # Sample 10% of the data
            masakhanews_samples[code] = full_samples_df.sample(n=num_to_sample, random_state=42) 
            print(f"  Finished sampling {len(masakhanews_samples[code])} samples for {code}.")
        else:
            print(f"  No samples loaded for {code}, cannot sample.")
            masakhanews_samples[code] = pd.DataFrame() # Store empty DataFrame
    
    # Define single results path for CoTR
    base_results_path = "/work/bbd6522/results/classification/cotr"
    os.makedirs(base_results_path, exist_ok=True)
    
    # Run experiments on MasakhaNEWS (English ground truth)
    print(f"\n--- Running {masakhanews_name.capitalize()} Classification CoTR Experiments ---")
    print(f"Languages being evaluated: {', '.join(masakhanews_langs.keys())} (English removed)")
    for model_name in models:
        for lang_code, samples_df in masakhanews_samples.items():
            if samples_df.empty:
                print(f"WARNING: No samples loaded for {lang_code} ({masakhanews_name}). Skipping CoTR experiment for {model_name}.")
                continue
            run_classification_experiment_cotr(
                model_name,
                samples_df,
                lang_code,
                masakhanews_name,
                base_results_path # Use consolidated path
            )

if __name__ == "__main__":
    main() 