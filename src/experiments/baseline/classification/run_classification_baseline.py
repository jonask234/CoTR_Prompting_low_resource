import sys
import os
import argparse
import time
import pandas as pd
import torch
import csv
from huggingface_hub import login

# Fügt das Projektverzeichnis
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Deaktiviert die Tokenizer-Parallelität
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Importiert die Baseline-Klassifizierungsfunktionen
from src.experiments.baseline.classification.classification_baseline import (
    initialize_model,
    evaluate_classification_baseline,
    POSSIBLE_LABELS_EN
)

# Importiert MasakhaNEWS
from src.utils.data_loaders.load_masakhanews import load_masakhanews_samples 

# Importiert die Metrikberechnung
from evaluation.classification_metrics import calculate_classification_metrics

# Importiert Token
from config import get_token

# Parameter für die Generierung
GENERATION_PARAMETERS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "max_tokens": 20
}

def run_baseline_classification_experiment(
    model_name,
    tokenizer, 
    model, 
    samples_df,
    lang_code,
    possible_labels_en,
    use_few_shot, 
    base_results_path,
    generation_params
):
    # Führt ein einzelnes durch
    if samples_df.empty:
        print(f"No samples for {lang_code}, skipping baseline experiment.")
        return None

    model_short_name = model_name.split('/')[-1]
    shot_type_str = "fs" if use_few_shot else "zs"
    prompt_lang_str = "lrl" if lang_code != 'en' else "en"

    results_dir = os.path.join(base_results_path, prompt_lang_str, shot_type_str, lang_code, model_short_name)
    summaries_dir = os.path.join(base_results_path, "summaries", prompt_lang_str, shot_type_str)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"results_baseline_classification_{lang_code}.csv")
    summary_file = os.path.join(summaries_dir, f"summary_baseline_classification_{lang_code}_{model_short_name}.csv")

    print(f"Running Baseline Classification: {model_name} on {lang_code} (Prompt: {prompt_lang_str}, Shot: {shot_type_str})")
    start_time = time.time()
    try:
        results_df_computed = evaluate_classification_baseline(
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            samples_df=samples_df,
            lang_code=lang_code,
            possible_labels=possible_labels_en,
            use_few_shot=use_few_shot,
            **generation_params
        )
        runtime = time.time() - start_time
        if not results_df_computed.empty:
            results_df_computed['runtime_seconds_total'] = runtime
            results_df_computed['runtime_per_sample'] = runtime / len(results_df_computed)
            results_df_computed.to_csv(results_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
            print(f"Results saved to {results_file}")
        else:
            print("Baseline evaluation returned empty DataFrame.")
            return None
    except Exception as e:
        print(f"Error during baseline classification for {model_name}, {lang_code}, {prompt_lang_str}, {shot_type_str}: {e}")
        return None
    
    if results_df_computed.empty:
        print("No results to calculate metrics from for baseline.")
        return None

    # Metriken berechnen
    metrics = calculate_classification_metrics(results_df_computed)

    summary_data = {
        'model': model_short_name,
            'language': lang_code,
        'prompt_language': prompt_lang_str,
        'shot_type': shot_type_str,
        'accuracy': metrics.get('accuracy', 0.0),
        'macro_f1': metrics.get('macro_f1', 0.0),
        'samples_processed': len(results_df_computed),
        'runtime_total_s': results_df_computed['runtime_seconds_total'].iloc[0] if 'runtime_seconds_total' in results_df_computed.columns and not results_df_computed.empty else 0,
    }
    summary_data.update({k: v for k,v in generation_params.items()})
    for label in possible_labels_en:
        summary_data[f'{label}_precision'] = metrics.get(f'{label}_precision', 0.0)
        summary_data[f'{label}_recall'] = metrics.get(f'{label}_recall', 0.0)
        
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(summary_file, index=False, float_format='%.4f', na_rep='NaN') 
    print(f"Summary saved to {summary_file}")
    print(summary_df.to_string())
    return summary_data

def parse_args():
    parser = argparse.ArgumentParser(description="Run Baseline Classification Experiments with MasakhaNEWS.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated model names.")
    parser.add_argument("--langs", type=str, default="en,ha,sw", help="Comma-separated MasakhaNEWS language codes.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per language.")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split.")
    
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Prompting strategies to evaluate: zero-shot or few-shot.")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/classification/baseline_masakhanews", 
                        help="Base directory to save results and summaries.")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token for gated models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and reproducibility.")

    parser.add_argument("--temperature", type=float, default=0.1, help="Override temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Override top-p for generation.")
    parser.add_argument("--top_k", type=int, default=40, help="Override top-k for generation.")
    parser.add_argument("--max_tokens", type=int, default=20, help="Override max_tokens for classification label generation.")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Override repetition penalty.")
    parser.add_argument("--do_sample", type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help="Override do_sample (True/False).")

    return parser.parse_args()

def main():
    args = parse_args()

    token = get_token()
    login(token=token)
    
    model_list = [m.strip() for m in args.models.split(',')]
    lang_list = [l.strip() for l in args.langs.split(',')]

    os.makedirs(args.base_output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.base_output_dir, "summaries"), exist_ok=True)

    possible_labels_en = POSSIBLE_LABELS_EN
    print(f"Using English labels for classification: {possible_labels_en}")

    all_experiment_summaries = []
    sampling_seed = args.seed 

    for model_name in model_list:
        print(f"\n{'='*20} Initializing Model: {model_name} {'='*20}")
        tokenizer, model = None, None
        try:
            tokenizer, model = initialize_model(model_name)
        except Exception as e_init:
            print(f"Failed to initialize model {model_name}: {e_init}. Skipping.")
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        for lang_code in lang_list:
            print(f"\n--- Processing Language: {lang_code} for model {model_name} (Baseline) ---")
            
            loaded_df = load_masakhanews_samples(
                lang_code=lang_code,
                split=args.data_split,
                num_samples=args.num_samples,
                seed=args.seed
            )
            
            if loaded_df.empty:
                print(f"Skipping {lang_code} for {model_name} due to missing data.")
                continue

            for shot_setting in args.shot_settings:
                use_few_shot = (shot_setting == 'few_shot')
                
                generation_params = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "max_tokens": args.max_tokens,
                    "repetition_penalty": args.repetition_penalty,
                    "do_sample": args.do_sample
                }

                print(f"\n  Running config: Lang={lang_code}, Shot={shot_setting}")

                summary_data = run_baseline_classification_experiment(
                    model_name, tokenizer, model, loaded_df, lang_code, 
                    possible_labels_en,
                    use_few_shot,
                    args.base_output_dir, generation_params
                )
                if summary_data:
                    all_experiment_summaries.append(summary_data)
        
        print(f"Finished all baseline experiments for model {model_name}. Unloading...")
        del model
        del tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        
        summaries_dir = os.path.join(args.base_output_dir, "summaries")
        os.makedirs(summaries_dir, exist_ok=True)

        overall_summary_filename = os.path.join(summaries_dir, "baseline_classification_ALL_experiments_summary.csv")
        
        overall_summary_df.to_csv(overall_summary_filename, index=False, float_format='%.4f', na_rep='NaN')
        
        print(f"Overall summary saved to {overall_summary_filename}")
    else:
        print("No classification baseline experiments were successfully summarized.")

    print("\n====== Classification Baseline Script Finished ======")

if __name__ == "__main__":
    main() 