import sys
import os

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from huggingface_hub import login
from config import get_token
from datasets import load_dataset

# Project specific imports
from src.utils.data_loaders.load_news_classification import load_swahili_news, load_hausa_news
from src.experiments.cotr.classification.classification_cotr import evaluate_classification_cotr
from evaluation.sentiment_metrics import calculate_sentiment_metrics # Reusing for Acc/F1
# Import COMET calculation function and availability flag from QA metrics
from evaluation.cotr.qa_metrics_cotr import calculate_comet_score, COMET_AVAILABLE

def run_classification_experiment_cotr(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str,
    dataset_name: str, # e.g., 'masakhanews'
    base_results_path: str
):
    """
    Run the CoTR text classification experiment for a specific model and language.
    """
    if samples_df.empty:
        print(f"WARNING: No samples provided for {lang_code} ({dataset_name}). Skipping CoTR experiment for {model_name}.")
        return

    try:
        print(f"\nProcessing {lang_code} CoTR classification ({dataset_name}) with {model_name}...")
        # This function performs translation and English classification
        results_df = evaluate_classification_cotr(model_name, samples_df, lang_code)

        if results_df.empty:
            print(f"WARNING: No CoTR results generated for {lang_code} ({dataset_name}) with {model_name}. Skipping metrics.")
            return

        # Calculate classification metrics (Accuracy, Macro F1)
        print("\nCalculating classification metrics...")
        # Reusing the sentiment metrics function as it calculates standard Acc/F1
        metrics = calculate_sentiment_metrics(results_df)
        avg_accuracy = metrics.get('accuracy', float('nan'))
        avg_macro_f1 = metrics.get('macro_f1', float('nan'))

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
        print(f"  Accuracy: {avg_accuracy:.4f}")
        print(f"  Macro F1-Score: {avg_macro_f1:.4f}")
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
        output_filename = f"cotr_classification_{dataset_name}_{lang_code}_{model_name_short}.csv"
        # Select columns to save (include translation quality metrics)
        cols_to_save = ['id', 'original_text', 'text_en', 'ground_truth_label', 'predicted_label', 
                        'predicted_label_en', 'predicted_label_lrl', 'language']
        if 'comet_source_to_en' in results_df.columns:
            cols_to_save.append('comet_source_to_en')
        if 'translation_quality' in results_df.columns:
            cols_to_save.append('translation_quality')
        if 'en_to_lrl_different' in results_df.columns:
            cols_to_save.append('en_to_lrl_different')
            
        results_df[cols_to_save].to_csv(os.path.join(lang_path, output_filename), index=False)
        print(f"Detailed results saved to {lang_path}/{output_filename}")

        # Save summary metrics
        summary = {
            'model': model_name,
            'language': lang_code,
            'dataset': dataset_name,
            'pipeline': 'cotr',
            'accuracy': avg_accuracy,
            'macro_f1': avg_macro_f1,
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
        summary_filename = f"summary_cotr_classification_{dataset_name}_{lang_code}_{model_name_short}.csv"
        # Ensure consistent float formatting
        summary_df.to_csv(os.path.join(summary_path, summary_filename), index=False, float_format='%.4f')
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

    # Using MasakhaNEWS dataset for both languages
    datasets_to_run = {
        "masakhane_news_swa": {"lang_code": "swa", "loader": load_swahili_news, "dataset_hf_id": "masakhane/masakhane_news"},
        "masakhane_news_hau": {"lang_code": "hau", "loader": load_hausa_news, "dataset_hf_id": "masakhane/masakhane_news"}
    }
    
    # Define samples per language (~10% of total available samples)
    # Swahili: 476 total samples → 10% = ~48 samples
    # Hausa: 637 total samples → 10% = ~64 samples
    samples_per_lang = {
        "swa": 48,  # 10% of 476 Swahili samples
        "hau": 64   # 10% of 637 Hausa samples
    }
    
    hf_token = get_token()  # Still needed for model loading

    print(f"\n--- Loading MasakhaNEWS Classification Data (10% sample) for CoTR ---")
    classification_samples = {} # Dictionary to store samples: {lang_code: (df, dataset_name)}
    
    for dataset_name, config in datasets_to_run.items():
        lang_code = config["lang_code"]
        loader_func = config["loader"]
        hf_id = config["dataset_hf_id"]
        
        # Get language-specific sample size
        num_samples = samples_per_lang.get(lang_code, 50)  # Default to 50 if not specified
        
        print(f"Loading samples for {lang_code} from {dataset_name} ({hf_id}) - Target: {num_samples} samples (~10%)")
        try:
            # Pass token to the loader
            samples_df = loader_func(num_samples=num_samples, token=hf_token)
            if samples_df is not None and not samples_df.empty:
                classification_samples[lang_code] = samples_df
                print(f"  Loaded {len(samples_df)} samples for {lang_code} ({dataset_name}).")
            else:
                print(f"  No samples loaded or returned for {lang_code} ({dataset_name}).")
        except Exception as e:
            print(f"ERROR loading data for {lang_code} ({dataset_name}): {e}")
            import traceback
            traceback.print_exc()

    # Define base results path for classification CoTR results
    base_results_path = "/work/bbd6522/results/classification/cotr"
    os.makedirs(base_results_path, exist_ok=True)

    # Run experiments
    print(f"\n--- Running MasakhaNEWS CoTR Classification Experiments ---")
    for model_name in models:
        for lang_code, samples_df in classification_samples.items():
            # Get the dataset name for this language code
            dataset_name = next((name for name, cfg in datasets_to_run.items() 
                               if cfg["lang_code"] == lang_code), f"masakhane_news_{lang_code}")
            run_classification_experiment_cotr(
                model_name,
                samples_df,
                lang_code,
                dataset_name,
                base_results_path
            )

if __name__ == "__main__":
    main() 