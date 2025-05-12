import sys
import os
import argparse
import time
import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, Any, Optional

# Add project root to Python path BEFORE other project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import CoTR sentiment functions
from src.experiments.cotr.sentiment.sentiment_cotr import (
    initialize_model,
    evaluate_sentiment_cotr_multi_prompt,
    evaluate_sentiment_cotr_single_prompt
)

# Import data loader (assuming a generic loader for now, replace with actual if different)
# For example, if using AfriSenti, it might be from src.utils.data_loaders.load_afrisenti
from src.utils.data_loaders.load_afrisenti import load_afrisenti_samples # Corrected import

# Import metrics calculation
from evaluation.sentiment_metrics import calculate_sentiment_metrics
from evaluation.cotr.qa_metrics_cotr import COMET_AVAILABLE # For checking if COMET can be used for text_en
from src.evaluation.cotr.translation_metrics import calculate_comet_score # For translation quality

# Hugging Face Login
from huggingface_hub import login
from config import get_token

# --- Standardized Parameters --- (To be consistent with baseline if applicable)
# These are defaults if not overridden by CLI or language-specific settings.
STANDARD_PARAMETERS = {
    "text_translation": {"temperature": 0.5, "top_p": 0.9, "top_k": 40, "max_new_tokens": 512, "repetition_penalty": 1.05},
    "sentiment_classification": {"temperature": 0.2, "top_p": 0.9, "top_k": 30, "max_new_tokens": 10, "repetition_penalty": 1.1},
    "label_translation": {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_new_tokens": 20, "repetition_penalty": 1.0},
    "single_prompt_cotr": {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_new_tokens": 150, "repetition_penalty": 1.1}
}

LANGUAGE_PARAMETERS = {
    "sw": {
        "text_translation": {"temperature": 0.45, "max_new_tokens": 400},
        "sentiment_classification": {"temperature": 0.15, "max_new_tokens": 8},
        "label_translation": {"temperature": 0.25, "max_new_tokens": 15},
        "single_prompt_cotr": {"temperature": 0.25, "max_new_tokens": 120}
    },
    "ha": {
        "text_translation": {"temperature": 0.45, "max_new_tokens": 400},
        "sentiment_classification": {"temperature": 0.15, "max_new_tokens": 8},
        "label_translation": {"temperature": 0.25, "max_new_tokens": 15},
        "single_prompt_cotr": {"temperature": 0.25, "max_new_tokens": 120}
    },
    # Add other LRLs like 'te' if specific tuning is done
}

MODEL_ADJUSTMENTS = {
    "aya": {
        "text_translation": {"temperature_factor": 0.9},
        "sentiment_classification": {"temperature_factor": 0.85},
        "label_translation": {"temperature_factor": 0.9},
        "single_prompt_cotr": {"temperature_factor": 0.9}
    },
    "qwen": {
        "text_translation": {"top_p_factor": 0.95, "top_k_set": 35},
        "sentiment_classification": {"top_p_factor": 0.9, "top_k_set": 25},
        "label_translation": {"top_p_factor": 0.95, "top_k_set": 35},
        "single_prompt_cotr": {"top_p_factor": 0.95, "top_k_set": 35}
    }
}

def get_effective_params(step_name: str, lang_code: str, model_name_str: str, cli_args: argparse.Namespace) -> Dict:
    """Determines effective generation parameters for a given step."""
    base_params = STANDARD_PARAMETERS.get(step_name, {}).copy()
    # Correctly identify model short name for MODEL_ADJUSTMENTS
    model_short_name = model_name_str.split('/')[-1].split('-')[0].lower() # e.g. "aya" or "qwen"
    lang_params = LANGUAGE_PARAMETERS.get(lang_code, {}).get(step_name, {}).copy()
    model_adjust = MODEL_ADJUSTMENTS.get(model_short_name, {}).get(step_name, {}).copy()

    # Start with standard, override with lang-specific, then apply model adjustments
    effective_params = {**base_params, **lang_params}

    # Apply model-specific multiplicative factors or direct sets
    if 'temperature_factor' in model_adjust:
        effective_params["temperature"] *= model_adjust['temperature_factor']
    if 'top_p_factor' in model_adjust:
        effective_params["top_p"] *= model_adjust['top_p_factor']
    if 'top_k_set' in model_adjust:
        effective_params["top_k"] = model_adjust['top_k_set']
    
    # Override with CLI arguments if provided for the specific step
    # Example for text_translation step (needs to be adapted for how CLI args are named)
    # This part needs careful mapping from general CLI args to step-specific ones.
    # For simplicity, this example assumes CLI args might be step-specific or general.
    # A more robust way is to have CLI args like --text_translation_temp, --sentiment_temp etc.
    # Or, apply general CLI args as overrides to all steps if that's the intent.
    
    # Simplified: If a general CLI param exists, it overrides the current effective param for that key
    if cli_args.temperature is not None: effective_params["temperature"] = cli_args.temperature
    if cli_args.top_p is not None: effective_params["top_p"] = cli_args.top_p
    if cli_args.top_k is not None: effective_params["top_k"] = cli_args.top_k
    if cli_args.max_sentiment_tokens is not None and step_name == "sentiment_classification":
        effective_params["max_new_tokens"] = cli_args.max_sentiment_tokens
    elif step_name == "label_translation" and cli_args.max_label_trans_tokens is not None:
        effective_params["max_new_tokens"] = cli_args.max_label_trans_tokens
    elif step_name == "single_prompt_cotr" and cli_args.max_single_prompt_tokens is not None:
        effective_params["max_new_tokens"] = cli_args.max_single_prompt_tokens
    if cli_args.max_text_trans_tokens is not None and step_name == "text_translation":
        effective_params["max_new_tokens"] = cli_args.max_text_trans_tokens
    if cli_args.repetition_penalty is not None: effective_params["repetition_penalty"] = cli_args.repetition_penalty
    
    # Ensure do_sample is consistent with temperature
    effective_params["do_sample"] = True if effective_params.get("temperature", 0) > 0.01 else False

    return effective_params

def run_single_experiment_config(
    model_name_str: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    pipeline_type: str,
    use_few_shot: bool,
    base_output_dir: str,
    args: argparse.Namespace # Pass all CLI args
) -> Optional[Dict[str, Any]]:
    
    experiment_params_log = {}
    
    # Determine effective generation parameters for each step
    text_trans_params = get_effective_params("text_translation", lang_code, model_name_str, args)
    sentiment_class_params = get_effective_params("sentiment_classification", lang_code, model_name_str, args)
    label_trans_params = get_effective_params("label_translation", lang_code, model_name_str, args)

    try:
        if pipeline_type == 'multi_prompt':
            results_df = evaluate_sentiment_cotr_multi_prompt(
                model=model,         # Correct: pass model object to 'model' param
                tokenizer=tokenizer, # Correct: pass tokenizer object to 'tokenizer' param
                samples_df=samples_df,
                lang_code=lang_code,
                use_few_shot=use_few_shot,
                text_translation_params=text_trans_params,
                sentiment_classification_params=sentiment_class_params,
                label_translation_params=label_trans_params
            )
        elif pipeline_type == 'single_prompt':
            single_prompt_params = get_effective_params("single_prompt_cotr", lang_code, model_name_str, args)
            experiment_params_log.update({"single_prompt": single_prompt_params})
            results_df = evaluate_sentiment_cotr_single_prompt(
                model, tokenizer, samples_df, lang_code, use_few_shot,
                generation_params=single_prompt_params
            )
        else:
            logging.error(f"Unknown pipeline type: {pipeline_type}")
            return None

        if results_df is None or results_df.empty:
            logging.warning(f"No results for {model_name_str}, {lang_code}, {pipeline_type}, {use_few_shot}.")
            return None

        # --- Calculate Metrics ---
        # Decide which predicted label to use for metrics
        # For multi-prompt, 'final_predicted_label' (which defaults to predicted_label_en) is a good choice.
        # For single-prompt, 'final_predicted_label' (which is predicted_label_lrl) needs careful handling with GT.
        # Let's assume calculate_sentiment_metrics handles ground_truth in LRL and predicted_label_lrl if that's the target.
        # Or, map LRL predictions to English if GT is always English.
        # For now, let's assume GT label is in English & use 'predicted_label_en' for multi and map LRL for single.

        if pipeline_type == 'multi_prompt':
            results_df['eval_prediction'] = results_df['predicted_label_en']
            # --- Add COMET Scoring for Multi-Prompt ---
            if COMET_AVAILABLE:
                logging.info(f"Calculating COMET scores for {model_name_str}, {lang_code}, {pipeline_type}...")
                comet_scores_text_lrl_en = []
                comet_scores_label_en_lrl = []
                for _, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Calculating COMET"):
                    # Text LRL -> EN COMET (if original_text and text_en are present and not None/NaN)
                    if pd.notna(row.get('original_text')) and pd.notna(row.get('text_en')):
                        score_text = calculate_comet_score(sources=[row['original_text']], predictions=[row['text_en']], references=[[row['original_text']]]) # Using original as reference for now
                        comet_scores_text_lrl_en.append(score_text if isinstance(score_text, (float, int)) else 0.0)
                    else:
                        comet_scores_text_lrl_en.append(0.0) # Or np.nan

                    # Label EN -> LRL COMET (if predicted_label_en and predicted_label_lrl are present and not None/NaN)
                    # For label translation, we don't have a direct "source" in the LRL ground truth for the EN label.
                    # We are evaluating the translation of the *predicted* English label to LRL.
                    # A true reference for this step would be the LRL equivalent of the *predicted* English label.
                    # If ground_truth_label is LRL, and predicted_label_en is the EN equivalent of that GT, then GT is a ref.
                    # This is a bit circular for evaluation of translation quality itself.
                    # For now, we'll score the translation of the *predicted* EN label to LRL against the LRL ground truth if the EN prediction was "correct"
                    # This is an approximation.
                    if pd.notna(row.get('predicted_label_en')) and pd.notna(row.get('predicted_label_lrl')) and pd.notna(row.get('ground_truth_label')):
                        # We need to be careful here. The ground_truth_label is in LRL.
                        # predicted_label_en is what the model thought the sentiment was in English.
                        # predicted_label_lrl is the translation of predicted_label_en into LRL.
                        # So, we are checking how well predicted_label_lrl matches ground_truth_label, given predicted_label_en was the source.
                        score_label = calculate_comet_score(sources=[row['predicted_label_en']], predictions=[row['predicted_label_lrl']], references=[[row['ground_truth_label']]])
                        comet_scores_label_en_lrl.append(score_label if isinstance(score_label, (float, int)) else 0.0)
                    else:
                        comet_scores_label_en_lrl.append(0.0) # Or np.nan
                
                results_df['comet_text_lrl_en'] = comet_scores_text_lrl_en
                results_df['comet_label_en_lrl'] = comet_scores_label_en_lrl
                logging.info(f"COMET - Avg LRL-EN Text: {np.mean(comet_scores_text_lrl_en):.4f}, Avg EN-LRL Label: {np.mean(comet_scores_label_en_lrl):.4f}")
        else:
                logging.warning("COMET not available, skipping COMET score calculation.")
                results_df['comet_text_lrl_en'] = np.nan
                results_df['comet_label_en_lrl'] = np.nan
        else: # single_prompt - map LRL prediction to English if possible for metrics
            # This mapping should be robust or metrics function should handle LRL predictions
            # Simplified: if predicted_label_lrl is a known LRL sentiment word, map it.
            # Otherwise, if predicted_label_en_intermediate is valid, use it.
            # For now, we will assume that calculate_sentiment_metrics can handle the 'final_predicted_label' column
            # and that it contains labels (potentially LRL) that can be compared to 'ground_truth_label' (likely EN).
            # The `extract_lrl_sentiment_from_single_prompt` was updated to return English standard labels if it detects LRL keywords.
            results_df['eval_prediction'] = results_df['final_predicted_label'] 

        # Ensure ground_truth_label is also standardized if necessary (e.g. all lowercase)
        results_df['ground_truth_label'] = results_df['ground_truth_label'].astype(str).str.lower()
        results_df['eval_prediction'] = results_df['eval_prediction'].astype(str).str.lower()

        metrics = calculate_sentiment_metrics(results_df.rename(columns={'eval_prediction': 'predicted_label'}))
        logging.info(f"Metrics for {model_name_str}, {lang_code}, {pipeline_type}, {use_few_shot}: Accuracy: {metrics.get('accuracy', 0.0):.4f}, Macro F1: {metrics.get('macro_f1', 0.0):.4f}")

        # --- File Saving --- 
        shot_str = "fs" if use_few_shot else "zs"
        model_file_name = model_name_str.replace("/", "_")
        results_sub_dir = os.path.join(base_output_dir, "results", pipeline_type, shot_str, lang_code)
        summaries_sub_dir = os.path.join(base_output_dir, "summaries", pipeline_type, shot_str, lang_code)
        os.makedirs(results_sub_dir, exist_ok=True)
        os.makedirs(summaries_sub_dir, exist_ok=True)

        detailed_results_file = os.path.join(results_sub_dir, f"detailed_{model_file_name}.csv")
        summary_file = os.path.join(summaries_sub_dir, f"summary_{model_file_name}.csv")

        results_df.to_csv(detailed_results_file, index=False)
        logging.info(f"Detailed results saved to {detailed_results_file}")

        summary_data = {
            'model': model_name_str,
            'language': lang_code,
            'pipeline': pipeline_type,
            'shot_type': shot_str,
            'accuracy': metrics.get('accuracy'),
            'macro_f1': metrics.get('macro_f1'),
            'positive_precision': metrics.get('positive_precision'),
            'positive_recall': metrics.get('positive_recall'),
            'negative_precision': metrics.get('negative_precision'),
            'negative_recall': metrics.get('negative_recall'),
            'neutral_precision': metrics.get('neutral_precision'),
            'neutral_recall': metrics.get('neutral_recall'),
            'comet_text_lrl_en': results_df['comet_text_lrl_en'].mean() if pd.notna(results_df['comet_text_lrl_en']).any() else np.nan,
            'comet_label_en_lrl': results_df['comet_label_en_lrl'].mean() if pd.notna(results_df['comet_label_en_lrl']).any() else np.nan,
            'samples_processed': len(results_df),
            'params': json.dumps(experiment_params_log) # Store parameters used
        }
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(summary_file, index=False, float_format='%.4f')
        logging.info(f"Summary saved to {summary_file}")
        
        return summary_data
    except Exception as e:
        logging.error(f"Error in run_single_experiment_config: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run Sentiment Analysis CoTR experiments.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-expanse-8b,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated model names")
    parser.add_argument("--langs", nargs='+', default=['sw', 'ha'], 
                        help="Languages to evaluate (e.g., sw ha te)")
    parser.add_argument("--samples", type=int, default=50, 
                        help="Number of samples to process per language.")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/sentiment/cotr", help="Base directory to save results and summaries.")
    parser.add_argument("--pipeline_types", nargs='+', default=['multi_prompt', 'single_prompt'], choices=['multi_prompt', 'single_prompt'], help="CoTR pipeline types to run.")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Shot settings to evaluate.")
    
    # General Generation Parameters (can be overridden by step-specific logic or specific CLI args)
    parser.add_argument("--temperature", type=float, default=None, help="Global temperature for generation.")
    parser.add_argument("--top_p", type=float, default=None, help="Global top-p for generation.")
    parser.add_argument("--top_k", type=int, default=None, help="Global top-k for generation.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Global repetition penalty.")
    
    # Step-specific max_tokens
    parser.add_argument("--max_sentiment_tokens", type=int, default=None, help="Max tokens for sentiment classification step.")
    parser.add_argument("--max_text_trans_tokens", type=int, default=None, help="Max tokens for LRL text to English translation.")
    parser.add_argument("--max_label_trans_tokens", type=int, default=None, help="Max tokens for English label to LRL translation.")
    parser.add_argument("--max_single_prompt_tokens", type=int, default=None, help="Max tokens for the entire single-prompt CoTR generation.")

    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for gated models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Setup ---
    if args.hf_token:
        login(token=args.hf_token)
    else:
        login(token=get_token())
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    models_list = [m.strip() for m in args.models.split(',')]
    langs_list = [l.strip() for l in args.langs]

    all_summaries = []

    for model_name_str in models_list:
        logging.info(f"\n===== Initializing Model: {model_name_str} =====")
        model_initialized = False
        try:
            # Ensure correct unpacking order: tokenizer, model
            tokenizer, model = initialize_model(model_name_str)
            model_initialized = True
            logging.info(f"Model {model_name_str} initialized successfully.")
            
            # Explicitly link tokenizer and model if required by some models/tokenizers
            # (May not be universally needed, but can prevent certain issues)
            # if hasattr(tokenizer, 'model') and tokenizer.model is None:
            # tokenizer.model = model 
            # if hasattr(model, 'tokenizer') and model.tokenizer is None:
            # model.tokenizer = tokenizer
            # if hasattr(tokenizer, 'config') and hasattr(model, 'config'):
            #     if tokenizer.config is None or tokenizer.config != model.config:
            #         # This was the HACK, let's avoid if direct passing is correct
            #         # logging.info("Attempting to align tokenizer.config with model.config")
            #         # tokenizer.config = model.config 
            #         pass # Avoid the hack for now, rely on correct passing order

        except Exception as e_init:
            logging.error(f"Failed to initialize model {model_name_str}: {e_init}. Skipping this model.")
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        for lang_code in langs_list:
            logging.info(f"--- Loading data for {lang_code} ---")
            # Assuming load_afrisenti_samples takes lang_code and num_samples
            # And returns a DataFrame with 'text' and 'label' columns
            current_samples_df = load_afrisenti_samples(lang_code=lang_code, num_samples=args.samples, split="test")
            if current_samples_df.empty:
                logging.warning(f"No samples loaded for {lang_code}. Skipping.")
                continue
            
            # Standardize label column to lowercase string if it's not already
            if 'label' in current_samples_df.columns:
                current_samples_df['label'] = current_samples_df['label'].astype(str).str.lower()
        else:
                logging.error(f"Dataset for {lang_code} is missing 'label' column. Skipping.")
                continue

            for pipeline_type in args.pipeline_types:
                for shot_setting in args.shot_settings:
                    use_few_shot = (shot_setting == 'few_shot')
                    logging.info(f"Running: Model={model_name_str}, Lang={lang_code}, Pipeline={pipeline_type}, Shot={shot_setting}")
                    
                    summary = run_single_experiment_config(
                        model_name_str, tokenizer, model, current_samples_df, 
                        lang_code, pipeline_type, use_few_shot, 
                        args.base_output_dir, args
                    )
                    if summary:
                        all_summaries.append(summary)
        
        # Clear memory for the next model if multiple are run
        if model_initialized:
            logging.info(f"===== Finished all experiments for model {model_name_str}. Unloading... =====")
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("GPU memory cache cleared.")

    # --- Aggregate and Save Overall Summary --- 
    if all_summaries:
        overall_summary_df = pd.DataFrame(all_summaries)
        overall_summary_filename = os.path.join(args.base_output_dir, "summaries", f'sentiment_cotr_ALL_experiments_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        try:
            overall_summary_df.to_csv(overall_summary_filename, index=False, float_format='%.4f')
            logging.info(f"Overall summary saved to: {overall_summary_filename}")
            print("\n=== Overall CoTR Sentiment Summary ===")
            print(overall_summary_df.to_string())
        except Exception as e_save:
            logging.error(f"Error saving overall summary to {overall_summary_filename}: {e_save}")

        # Optional: Plotting (ensure matplotlib and seaborn are installed)
        try:
            plots_dir = os.path.join(args.base_output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            if not overall_summary_df.empty and 'macro_f1' in overall_summary_df.columns:
                plt.figure(figsize=(12, 7))
                sns.barplot(data=overall_summary_df, x='language', y='macro_f1', hue='model', palette='viridis',
                            ci=None) # Add other categories like pipeline_type, shot_type to hue or col/row for more detailed plots
                plt.title('Macro F1 Score for CoTR Sentiment Analysis')
                plt.ylabel('Macro F1 Score')
                plt.xlabel('Language')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'cotr_sentiment_macro_f1_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
                plt.close()
                logging.info(f"Summary plot saved to {plots_dir}")
        except ImportError:
            logging.warning("matplotlib or seaborn not installed. Skipping plot generation.")
        except Exception as e_plot:
            logging.error(f"Error during plot generation: {e_plot}")
    else:
        logging.info("No successful experiments were completed. No overall summary generated.")

    logging.info("====== Sentiment CoTR Script Finished ======")

if __name__ == "__main__":
    # This is a placeholder for the data loader.
    # You need to ensure `src.utils.data_loaders.load_afrisenti.load_afrisenti_samples` is correctly implemented.
    # Example of what it might look like:
    # def load_afrisenti_samples(lang_code: str, num_samples: int, split: str, balanced: bool = False, samples_per_class: Optional[int] = None):
    #     # ... logic to load your specific sentiment dataset ...
    #     # Should return a pd.DataFrame with 'text' and 'label' columns
    #     # Example:
    #     # data = [{"text": "example text 1", "label": "positive"}, ...]
    #     # df = pd.DataFrame(data)
    #     # if num_samples: df = df.sample(min(num_samples, len(df)), random_state=seed) # random_state not directly supported
    #     # return df
    #     print(f"Placeholder: load_afrisenti_samples called for {lang_code} with {num_samples} samples from {split} split.")
    #     # Return a dummy DataFrame for testing if the real loader isn't ready
    #     dummy_data = []
    #     labels = ["positive", "negative", "neutral"]
    #     for i in range(num_samples if num_samples else 5):
    #         dummy_data.append({"text": f"This is {lang_code} sample text number {i+1}. It feels very {labels[i%3]}.", "label": labels[i%3]})
    #     return pd.DataFrame(dummy_data)
    # Make sure the real loader is in the path `src/utils/data_loaders/load_afrisenti.py`
    # and named `load_afrisenti_samples`
    
    # Example usage for main()
    # args = parse_args() # Assuming parse_args() is defined and provides necessary arguments
    # model_name_to_test = "CohereLabs/aya-expanse-8b"
    # lang_to_test = "sw"
    # num_samples_to_test = 10
    # data_split_to_test = "test"

    # logging.info(f"Running CoTR sentiment analysis for {model_name_to_test} on {lang_to_test}")
    
    # samples = load_afrisenti_samples(
    # lang_code=lang_to_test,
    # num_samples=num_samples_to_test,
    # split=data_split_to_test
    # )
    
    # if not samples.empty:
    # # ... rest of the example main logic ...
    # else:
    # logging.info(f"No samples loaded for {lang_to_test}. Skipping CoTR evaluation.")
    main() 