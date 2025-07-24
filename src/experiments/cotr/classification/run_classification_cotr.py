import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import re
import json

# Tokenizer-Parallelität deaktivieren
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Projekt-Root 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Funktionen importieren
from src.utils.data_loaders.load_masakhanews import load_masakhanews_samples
from src.experiments.cotr.classification.classification_cotr import (
    evaluate_classification_cotr_multi_prompt,
    evaluate_classification_cotr_single_prompt,
    initialize_model,
    CLASS_LABELS_ENGLISH
)
from huggingface_hub import login
from config import get_token
from sklearn.metrics import accuracy_score, f1_score

# Standard-Parameter für die Generierung
DEFAULT_GENERATION_PARAMS = {
    "lrl_to_eng_translation": {"temperature": 0.3, "top_p": 0.9, "max_new_tokens": 300, "repetition_penalty": 1.0, "top_k": 40, "do_sample": True},
    "eng_classification": {"temperature": 0.1, "top_p": 0.85, "max_new_tokens": 50, "repetition_penalty": 1.05, "top_k": 30, "do_sample": True},
    "eng_to_lrl_label_translation": {"temperature": 0.3, "top_p": 0.9, "max_new_tokens": 30, "repetition_penalty": 1.0, "top_k": 40, "do_sample": True},
    "single_prompt_chain": {"temperature": 0.1, "top_p": 0.85, "max_new_tokens": 400, "repetition_penalty": 1.05, "top_k": 30, "do_sample": True}
}

# Korrekte Labels für MasakhaNEWS definieren
POSSIBLE_LABELS_EN = ['business', 'entertainment', 'health', 'politics', 'religion', 'sports', 'technology']

def parse_cli_args():
    # Argumente für die Kommandozeile definieren
    parser = argparse.ArgumentParser(description="Run Text Classification CoTR experiments with MasakhaNEWS.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated model names.")
    parser.add_argument("--langs", type=str, default="en,ha,sw", help="Comma-separated MasakhaNEWS language codes (e.g., sw,am,ha,yo,pcm,ig,en,pt).")
    parser.add_argument("--num_samples", type=int, default=80, help="Number of samples per language. Default: 80")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split.")
    parser.add_argument("--pipeline_types", nargs='+', default=['multi_prompt', 'single_prompt'], choices=['multi_prompt', 'single_prompt'])
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'])
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/classification/cotr_masakhanews")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token for gated models.")
    parser.add_argument("--overwrite_results", action='store_true', help="Overwrite existing result files if they exist.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    parser.add_argument("--test_mode", action='store_true', help="Run in test mode (uses a very small subset of data and fewer iterations).")
    return parser.parse_args()

def run_classification_experiment(
    model_name_str, tokenizer, model, samples_df,
    lang_code, possible_labels_en, pipeline_type, 
    use_few_shot, base_results_path, generation_args,
    overwrite_results = False
):
    # Führt ein einzelnes Klassifizierungsexperiment durch
    if samples_df.empty:
        print(f"No samples for {lang_code}, skipping {model_name_str}.")
        return None

    shot_type_str = "fs" if use_few_shot else "zs"
    model_short_name = model_name_str.split('/')[-1]

    results_dir = os.path.join(base_results_path, pipeline_type, shot_type_str, lang_code, model_short_name)
    summaries_dir = os.path.join(base_results_path, "summaries", pipeline_type, shot_type_str)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"results_cotr_classification_{lang_code}.csv")
    summary_file = os.path.join(summaries_dir, f"summary_cotr_classification_{lang_code}_{model_short_name}.csv")

    results_df_current_run = pd.DataFrame()

    if os.path.exists(results_file) and not overwrite_results:
        print(f"Results file {results_file} already exists. Skipping computation, attempting to load for summary.")
        try:
            results_df_current_run = pd.read_csv(results_file)
        except Exception as e:
            print(f"Could not load existing results {results_file}: {e}. Will recompute.")
            results_df_current_run = pd.DataFrame()
        
        if os.path.exists(summary_file):
            try:
                existing_summary_df = pd.read_csv(summary_file)
                print(f"Loaded existing summary: {summary_file}")
                return existing_summary_df.to_dict('records')[0]
            except Exception as es:
                print(f"Could not load existing summary {summary_file}: {es}. Recomputing.")
                if results_df_current_run.empty:
                    pass

    if overwrite_results and os.path.exists(results_file):
        print(f"Overwrite_results is True. Recomputing for {results_file}")
        results_df_current_run = pd.DataFrame()

    if results_df_current_run.empty:
        print(f"Running CoTR Classification: {model_name_str} on {lang_code} (Pipeline: {pipeline_type}, Shot: {shot_type_str})")
        start_time = time.time()
        try:
            eval_func_params = {
                "model_name": model_name_str,
                "model": model,
                "tokenizer": tokenizer,
                "samples_df": samples_df,
                "lang_code": lang_code,
                "possible_labels_en": possible_labels_en,
                "use_few_shot": use_few_shot,
            }
            if pipeline_type == 'multi_prompt':
                eval_func_params.update({
                    "text_translation_params": generation_args.get("lrl_to_eng_translation", {}),
                    "classification_params": generation_args.get("eng_classification", {}),
                    "label_translation_params": generation_args.get("eng_to_lrl_label_translation", {})
                })
                results_df_current_run = evaluate_classification_cotr_multi_prompt(**eval_func_params)
            else: # single_prompt
                eval_func_params["generation_params"] = generation_args.get("single_prompt_chain", {})
                results_df_current_run = evaluate_classification_cotr_single_prompt(**eval_func_params)

            runtime = time.time() - start_time
            if not results_df_current_run.empty:
                results_df_current_run['runtime_seconds_total'] = runtime
                results_df_current_run['runtime_per_sample'] = runtime / len(results_df_current_run)
                print(f"Evaluation completed successfully.")
            else:
                print("Evaluation returned empty DataFrame.")
        except Exception as e:
            print(f"Error during CoTR classification for {lang_code}, {model_short_name}, {pipeline_type}, {shot_type_str}: {e}")

    # Metriken berechnen
    if not results_df_current_run.empty:
        if pipeline_type == 'multi_prompt':
            predicted_col = 'predicted_label_eng_model'
            ground_truth_col = 'ground_truth_label_eng'
        else:
            predicted_col = 'predicted_label_accuracy'
            ground_truth_col = 'label_lrl_ground_truth'
        
        if predicted_col in results_df_current_run.columns and ground_truth_col in results_df_current_run.columns:
            results_df_current_run['is_correct'] = (
                results_df_current_run[predicted_col].str.lower().str.strip() == 
                results_df_current_run[ground_truth_col].str.lower().str.strip()
            )
            
            try:
                y_true = results_df_current_run[ground_truth_col].str.lower().str.strip()
                y_pred = results_df_current_run[predicted_col].str.lower().str.strip()
                
                missing_gt_mask = y_true.isin(['[missing ground truth]', ''])
                error_pred_mask = (y_pred.str.contains(r'\[unknown label\]', case=False, na=False) | 
                                  y_pred.str.contains(r'\[classification error\]', case=False, na=False) |
                                  y_pred.isin(['']))
                valid_mask = ~(missing_gt_mask | error_pred_mask)
                y_true_valid = y_true[valid_mask]
                y_pred_valid = y_pred[valid_mask]
                
                if len(y_true_valid) > 0:
                    accuracy = accuracy_score(y_true_valid, y_pred_valid)
                    macro_f1 = f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
                    weighted_f1 = f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
                else:
                    accuracy = 0.0
                    macro_f1 = 0.0
                    weighted_f1 = 0.0
                    
            except Exception as e:
                print(f"Error calculating metrics with sklearn: {e}. Using basic accuracy only.")
                accuracy = results_df_current_run['is_correct'].mean()
                macro_f1 = 0.0
                weighted_f1 = 0.0
        else:
            print(f"Missing columns for accuracy calculation. Expected {predicted_col} and {ground_truth_col}")
            accuracy = 0.0
            macro_f1 = 0.0
            weighted_f1 = 0.0
            
        if 'runtime_seconds_total' in results_df_current_run.columns:
            results_df_current_run.to_csv(results_file, index=False)
            print(f"Results saved to {results_file}")
    else:
        accuracy = 0.0
        macro_f1 = 0.0
        weighted_f1 = 0.0

    avg_accuracy = accuracy

    summary_data = {
        'model': model_name_str.split('/')[-1], 'language': lang_code, 'pipeline': pipeline_type, 'shot_type': 'few-shot' if use_few_shot else 'zero-shot',
        'samples': len(results_df_current_run), 'accuracy': avg_accuracy, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
        'generation_params': generation_args
    }

    summary_df_to_save = pd.DataFrame([summary_data])
    summary_df_to_save.to_csv(summary_file, index=False, float_format='%.4f')
    print(f"Summary saved to {summary_file}")
    print(summary_df_to_save.to_string())
    return summary_data

def main():
    args = parse_cli_args()
    # HF Login
    if args.hf_token:
        token = args.hf_token
    else:
        token = get_token()

    if token:
        login(token=token)

    models_list = [m.strip() for m in args.models.split(',')]
    lang_list = [l.strip() for l in args.langs.split(',')]
    all_experiment_summaries = []
    
    possible_labels_en_for_exp = POSSIBLE_LABELS_EN
    print(f"Using English labels for classification: {possible_labels_en_for_exp}")

    overall_summary_base_dir = os.path.join(args.base_output_dir, "summaries_overall")
    os.makedirs(overall_summary_base_dir, exist_ok=True)

    print(f"All Classification CoTR experiment outputs will be saved under: {args.base_output_dir}")
    print(f"Individual summaries in: {args.base_output_dir}/summaries/[pipeline]/[shot]")
    print(f"Overall summary in: {overall_summary_base_dir}")

    sampling_seed = args.seed

    for model_name_str in models_list:
        print(f"\n{'='*20} Initializing Model: {model_name_str} {'='*20}")
        tokenizer_main, model_main = None, None
        try:
            tokenizer_main, model_main = initialize_model(model_name_str)
        except Exception as e:
            print(f"Failed to initialize model {model_name_str}: {e}. Skipping this model.")
            if model_main is not None: del model_main
            if tokenizer_main is not None: del tokenizer_main
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        for lang_code in lang_list:
            print(f"\n--- Processing Language: {lang_code} for model {model_name_str} ---")
            
            samples_df_for_run = load_masakhanews_samples(
                lang_code, 
                split=args.data_split, 
                num_samples=args.num_samples,
                seed=sampling_seed
            )

            if samples_df_for_run.empty:
                print(f"    No MasakhaNEWS samples found or loaded for {lang_code}, skipping.")
                continue
            else:
                print(f"    Successfully loaded {len(samples_df_for_run)} MasakhaNEWS samples for {lang_code}.")
            
            if 'label' not in samples_df_for_run.columns or not all(isinstance(x, str) for x in samples_df_for_run['label']):
                 print(f"Warning: 'label' column for {lang_code} is missing or not all strings. Downstream metrics might fail.")
            else:
                samples_df_for_run['label'] = samples_df_for_run['label'].astype(str).str.lower().str.strip()

            if args.test_mode:
                print("    Running in TEST MODE with first 5 samples only.")
                samples_df_for_run = samples_df_for_run.head(5)
            
            if samples_df_for_run.empty:
                print(f"    No samples for {lang_code} after test_mode reduction. Skipping.")
                continue

            for pipeline_type in args.pipeline_types:
                for shot_setting_val_str in args.shot_settings:
                    use_few_shot_bool = (shot_setting_val_str == 'few_shot')
                    
                    generation_args_for_run_structured = DEFAULT_GENERATION_PARAMS
                    
                    print(f"\n  Running config: Model={model_name_str}, Lang={lang_code}, Pipeline={pipeline_type}, Shot={shot_setting_val_str}")
                    print(f"    Effective Gen Params: {json.dumps(generation_args_for_run_structured, indent=2)}")

                    summary = run_classification_experiment(
                        model_name_str, tokenizer_main, model_main, samples_df_for_run,
                        lang_code, possible_labels_en_for_exp, pipeline_type,
                        use_few_shot_bool, args.base_output_dir, generation_args_for_run_structured,
                        overwrite_results=args.overwrite_results
                    )
                    if summary:
                        all_experiment_summaries.append(summary)
        
        print(f"Finished all experiments for model {model_name_str}. Unloading...")
        del model_main; del tokenizer_main
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        if not overall_summary_df.empty:
            from datetime import datetime
            summary_filename_overall = f"cotr_classification_ALL_experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            overall_summary_path = os.path.join(overall_summary_base_dir, summary_filename_overall)
            overall_summary_df.to_csv(overall_summary_path, index=False, float_format='%.4f')
            print(f"\nOverall summary of Classification CoTR experiments saved to: {overall_summary_path}")
            print(overall_summary_df.to_string())

        else:
            print("Overall summary DataFrame is empty. No plots generated.")
    else:
        print("No summaries collected. Skipping overall summary and plot generation for Classification CoTR.")

    print("\nAll Classification CoTR experiments completed!")

if __name__ == "__main__":
    main() 