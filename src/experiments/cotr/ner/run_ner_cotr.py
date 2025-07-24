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
import itertools
import logging
from typing import Any, Dict, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path: 
    sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.experiments.cotr.ner.ner_cotr import (
    initialize_model,
    evaluate_ner_cotr,
    evaluate_ner_cotr_single_prompt,
    calculate_ner_metrics_for_sample as calculate_ner_metrics
)
from src.experiments.baseline.ner.ner_baseline import load_masakhaner_samples
from huggingface_hub import login
from config import get_token
from src.experiments.cotr.language_information import get_language_information


STANDARD_PARAMETERS = {
    "sw": {  # Swahili
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 200,
        "repetition_penalty": 1.1,
        "trans_temp": 0.3,
        "trans_top_p": 0.9,
        "trans_top_k": 40,
        "max_trans_tokens": 256,
        "trans_repetition_penalty": 1.0,
        "num_beams": 1
    },
    "ha": {  # Hausa
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 200,
        "repetition_penalty": 1.1,
        "trans_temp": 0.3,
        "trans_top_p": 0.9,
        "trans_top_k": 40,
        "max_trans_tokens": 256,
        "trans_repetition_penalty": 1.0,
        "num_beams": 1
    }
}

MODEL_PARAMETERS = {
    "qwen": { 
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 30,
        "max_tokens": 200,
        "repetition_penalty": 1.2,
        "trans_temp": 0.2,
        "trans_top_p": 0.8,
        "trans_top_k": 30,
        "max_trans_tokens": 256,
        "trans_repetition_penalty": 1.2,
        "num_beams": 4 
    },
    "aya": {  
        "temperature": 0.2,
        "top_p": 0.85,
        "top_k": 35,
        "max_tokens": 200,
        "repetition_penalty": 1.2,
        "trans_temp": 0.2,
        "trans_top_p": 0.85,
        "trans_top_k": 35,
        "max_trans_tokens": 256,
        "trans_repetition_penalty": 1.1,
        "num_beams": 3  
    }
}

LANGUAGE_PARAMETERS = {
    "sw": {  
        "temperature": 0.25,
        "top_p": 0.85,
        "top_k": 35,
        "max_tokens": 180,
        "repetition_penalty": 1.15,
        "trans_temp": 0.25,
        "trans_top_p": 0.85,
        "trans_top_k": 35,
        "max_trans_tokens": 256,
        "trans_repetition_penalty": 1.05
    },
    "ha": {  
        "temperature": 0.25,
        "top_p": 0.85,
        "top_k": 35,
        "max_tokens": 180,
        "repetition_penalty": 1.15,
        "trans_temp": 0.25,
        "trans_top_p": 0.85,
        "trans_top_k": 35,
        "max_trans_tokens": 256,
        "trans_repetition_penalty": 1.05
    }
}

def run_experiment(
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    base_results_path: str,
    pipeline_type: str,
    use_few_shot: bool,
    ner_params: Dict[str, Any],
    trans_params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    
    if lang_code in LANGUAGE_PARAMETERS:
        lang_params = LANGUAGE_PARAMETERS[lang_code]
        ner_params['temperature'] = lang_params.get("temperature", ner_params['temperature'])
        ner_params['top_p'] = lang_params.get("top_p", ner_params['top_p'])
        ner_params['top_k'] = lang_params.get("top_k", ner_params['top_k'])
        ner_params['max_new_tokens'] = lang_params.get("max_tokens", ner_params.get('max_new_tokens', 200))
        ner_params['repetition_penalty'] = lang_params.get("repetition_penalty", ner_params['repetition_penalty'])
        trans_params['temperature'] = lang_params.get("trans_temp", trans_params['temperature'])
        trans_params['top_p'] = lang_params.get("trans_top_p", trans_params['top_p'])
        trans_params['top_k'] = lang_params.get("trans_top_k", trans_params['top_k'])
        trans_params['max_new_tokens'] = lang_params.get("max_trans_tokens", trans_params['max_new_tokens'])
        trans_params['repetition_penalty'] = lang_params.get("trans_repetition_penalty", trans_params['repetition_penalty'])
        print(f"Using language-specific parameters for {lang_code}")

    model_lower = model_name.lower()
    model_key = None
    
    if "qwen" in model_lower:
        model_key = "qwen"
    elif "aya" in model_lower:
        model_key = "aya"
        
    if model_key and model_key in MODEL_PARAMETERS:
        model_params = MODEL_PARAMETERS[model_key]
        ner_params['temperature'] = model_params.get("temperature", ner_params['temperature'])
        ner_params['top_p'] = model_params.get("top_p", ner_params['top_p'])
        ner_params['top_k'] = model_params.get("top_k", ner_params['top_k'])
        ner_params['max_new_tokens'] = model_params.get("max_tokens", ner_params.get("max_new_tokens", 200))
        ner_params['repetition_penalty'] = model_params.get("repetition_penalty", ner_params['repetition_penalty'])
        ner_params['num_beams'] = model_params.get("num_beams", ner_params.get("num_beams", 1))
        
        if pipeline_type == "multi_prompt":
            trans_params['temperature'] = model_params.get("trans_temp", trans_params['temperature'])
            trans_params['top_p'] = model_params.get("trans_top_p", trans_params['top_p'])
            trans_params['top_k'] = model_params.get("trans_top_k", trans_params['top_k'])
            trans_params['max_new_tokens'] = model_params.get("max_trans_tokens", trans_params.get("max_new_tokens", 256))
            trans_params['repetition_penalty'] = model_params.get("trans_repetition_penalty", trans_params['repetition_penalty'])
            trans_params['num_beams'] = model_params.get("num_beams", trans_params.get("num_beams", 1))
            
        print(f"Applied {model_key} model-specific parameters")

   
    ner_temp_final = ner_params['temperature']
    ner_top_p_final = ner_params['top_p']
    ner_top_k_final = ner_params['top_k']
    ner_rep_penalty_final = ner_params['repetition_penalty']
    trans_temp_final = trans_params['temperature']
    trans_top_p_final = trans_params['top_p']
    trans_top_k_final = trans_params['top_k']
    trans_rep_penalty_final = trans_params['repetition_penalty']
    
    if "qwen" in model_name.lower() and pipeline_type == "single_prompt":
        ner_params['num_beams'] = max(4, ner_params.get('num_beams', 1))
        ner_params['do_sample'] = False
        print(f"Forcing beam search (num_beams={ner_params['num_beams']}, do_sample=False) for Qwen model in single prompt mode")
    
    results_dir = os.path.join(base_results_path, "results")
    summaries_dir = os.path.join(base_results_path, "summaries")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    model_short = model_name.split('/')[-1]
    shot_type_str = "fs" if use_few_shot else "zs"
    pipeline_short = "mp" if pipeline_type == "multi_prompt" else "sp"
    results_file = os.path.join(results_dir, f"results_cotr_{pipeline_short}_{shot_type_str}_ner_{lang_code}_{model_short}.csv")
    summary_file = os.path.join(summaries_dir, f"summary_cotr_{pipeline_short}_{shot_type_str}_ner_{lang_code}_{model_short}.csv")

    if os.path.exists(results_file):
        print(f"Results file exists, skipping: {results_file}")
        if not os.path.exists(summary_file):
            try:
                print(f"Attempting to load existing results from {results_file} to generate summary...")
                results_df = pd.read_csv(results_file)
            except Exception as e:
                print(f"Could not load existing results to regenerate summary: {e}")
                return None 
        else:
            try:
                existing_summary_df = pd.read_csv(summary_file)
                print(f"Loaded existing summary from {summary_file}")
                return existing_summary_df.to_dict('records')[0] # Return first row as dict
            except Exception as e:
                print(f"Could not load existing summary file {summary_file}: {e}")
                return None
    else:
        print(f"Running experiment: {model_name} on {lang_code} ({pipeline_type}, {shot_type_str})")
        try:
        if pipeline_type == 'multi_prompt':
            results_df = evaluate_ner_cotr(
                    model_name=model_name,
                    tokenizer=tokenizer,
                    model=model,
                    samples_df=samples_df,
                    lang_code=lang_code,
                    use_few_shot=use_few_shot,
                    translation_params=trans_params,
                ner_params=ner_params,
            )
            else: # single_prompt
            results_df = evaluate_ner_cotr_single_prompt(
                    model_name=model_name,
                    tokenizer=tokenizer,
                    model=model,
                    samples_df=samples_df,
                    lang_code=lang_code,
                    use_few_shot=use_few_shot,
                temperature=ner_params['temperature'],
                top_p=ner_params['top_p'],
                top_k=ner_params['top_k'],
                    max_tokens=ner_params['max_new_tokens'],
                repetition_penalty=ner_params['repetition_penalty'],
                    num_beams=ner_params.get('num_beams', 1) 
            )

        if results_df.empty:
                print("ERROR: Evaluation returned empty DataFrame.")
            return None

        results_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")

        except Exception as e:
            logging.error(f"Error during NER CoTR evaluation for {lang_code}, {model_short}, {pipeline_type}, {shot_type_str}: {e}", exc_info=True)
            return None
        
    print(f"Calculating metrics for {results_file}...")
    precisions, recalls, f1s = [], [], []
    successful_samples = 0

    
    needs_conversion = False
    first_gold = results_df['ground_truth_entities'].dropna().iloc[0] if not results_df['ground_truth_entities'].dropna().empty else None
    if first_gold is not None and isinstance(first_gold, str):
        needs_conversion = True
        print("Converting entity strings from CSV back to lists/tuples...")
        import ast
def safe_literal_eval(val):
            if pd.isna(val): return []
            try:
                evaluated = ast.literal_eval(val)
                return evaluated if isinstance(evaluated, list) else []
            except (ValueError, SyntaxError, TypeError):
                print(f"Warning: Could not evaluate entity string: {val[:50]}...")
                return []
            
    for idx, row in results_df.iterrows():
        try:
            gold = safe_literal_eval(row['ground_truth_entities']) if needs_conversion else row['ground_truth_entities']
            pred = safe_literal_eval(row['predicted_entities']) if needs_conversion else row['predicted_entities']

            if not isinstance(gold, list): gold = []
            if not isinstance(pred, list): pred = []
            
            metrics = calculate_ner_metrics(gold, pred) 
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1s.append(metrics.get('f1', 0.0)) 
            successful_samples += 1

        except Exception as e_metric:
            print(f"Error calculating metrics for row {idx}: {e_metric}")
            precisions.append(0.0); recalls.append(0.0); f1s.append(0.0)
    
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    avg_f1 = np.mean(f1s) if f1s else 0.0

    summary_data = {
        'model': model_short,
        'language': lang_code,
        'pipeline': pipeline_type,
        'shot_type': shot_type_str,
        'precision': float(avg_precision),
        'recall': float(avg_recall),
        'f1': float(avg_f1),
        'num_samples': len(samples_df),
        'num_successful': successful_samples,
        'ner_temp': ner_params['temperature'],
        'ner_top_p': ner_params['top_p'],
        'ner_top_k': ner_params['top_k'],
        'ner_rep_penalty': ner_params['repetition_penalty'],
        'trans_temp': trans_params['temperature'] if pipeline_type == 'multi_prompt' else None,
        'trans_top_p': trans_params['top_p'] if pipeline_type == 'multi_prompt' else None,
        'trans_top_k': trans_params['top_k'] if pipeline_type == 'multi_prompt' else None,
        'trans_rep_penalty': trans_params['repetition_penalty'] if pipeline_type == 'multi_prompt' else None,
        'max_tokens': ner_params['max_new_tokens'],
        'max_trans_tokens': trans_params.get('max_new_tokens') if pipeline_type == 'multi_prompt' else None,
        'num_beams': ner_params.get('num_beams', 1)
    }

    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(summary_file, index=False, float_format='%.4f')
    print(f"Summary metrics saved to {summary_file}")
    print(summary_df.to_string())
        
    return summary_data 

def run_grid_search(args, models, lang_codes, base_results_path, masakhaner_samples, pipeline_type_for_grid, use_few_shot_for_grid):
    
    param_grid = {
        "temperature": [0.2, 0.3, 0.4],
        "top_p": [0.8, 0.9, 1.0],
        "top_k": [40, 60],
        "max_tokens": [150, 200, 250],
        "num_beams": [1, 2]
    }
    
    if args.compact_grid:
        param_grid = {
            "temperature": [0.3],
            "top_p": [0.9],
            "top_k": [40],
            "max_tokens": [200],
            "num_beams": [1]
        }
    
    param_combinations = list(itertools.product(
        param_grid["temperature"],
        param_grid["top_p"],
        param_grid["top_k"],
        param_grid["max_tokens"],
        param_grid["num_beams"]
    ))
    
    print(f"Running grid search with {len(param_combinations)} parameter combinations")
    
    best_params = {}
    
    for lang_code in lang_codes:
        best_params[lang_code] = {
            "model": None,
            "params": None,
            "f1": -1  
        }

    for model_name in models:
        print(f"\n====== Starting experiments for model: {model_name} ======")
        model_initialized = False
        tokenizer, model = None, None
        try:
            print(f"  Attempting to initialize {model_name}...")
            tokenizer, model = initialize_model(model_name)
            model_initialized = True
            print(f"Model {model_name} initialized successfully.")
        except Exception as e:
            print(f"ERROR: Failed to initialize model {model_name}: {e}. Skipping this model for grid search.")
            if 'model' in locals() and model is not None: del model
            if 'tokenizer' in locals() and tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        for lang_code in lang_codes:
            if lang_code in masakhaner_samples and not masakhaner_samples[lang_code].empty:
                print(f"\n--- Running grid search for {model_name} on {lang_code} (Pipeline: {pipeline_type_for_grid}, Shot: {'Few-shot' if use_few_shot_for_grid else 'Zero-shot'}) ---")
                
                for temp, top_p, top_k, max_tokens, num_beams in tqdm(param_combinations, desc="Parameter combinations"):
                    result_df = run_experiment(
                        model_name=model_name,
                        tokenizer=tokenizer,
                        model=model,
                        samples_df=masakhaner_samples[lang_code],
                        lang_code=lang_code,
                        base_results_path=base_results_path,
                    pipeline_type=pipeline_type_for_grid,
                    use_few_shot=use_few_shot_for_grid,
                        ner_params={
                            "temperature": temp,
                            "top_p": top_p,
                            "top_k": top_k,
                            "max_new_tokens": max_tokens,
                            "repetition_penalty": 1.1,
                            "num_beams": num_beams
                        },
                        trans_params={
                            "temperature": temp,
                            "top_p": top_p,
                            "top_k": top_k,
                            "max_new_tokens": max_tokens,
                            "repetition_penalty": 1.1,
                            "num_beams": num_beams
                        },
                    )
                    
                    if result_df is not None and not result_df.empty:
                        avg_f1 = result_df["f1"].mean()
                        
                        if avg_f1 > best_params[lang_code]["f1"]:
                            best_params[lang_code] = {
                                "model": model_name,
                                "params": {
                                    "temperature": temp,
                                    "top_p": top_p,
                                    "top_k": top_k,
                                    "max_tokens": max_tokens,
                                    "num_beams": num_beams
                                },
                                "f1": avg_f1
                            }
                            
                            print(f"\nNew best parameters for {lang_code}: F1={avg_f1:.4f}")
                            print(f"  Model: {model_name}")
                            print(f"  Parameters: temp={temp}, top_p={top_p}, top_k={top_k}, max_tokens={max_tokens}, num_beams={num_beams}")
        
        print(f"Finished grid search for model {model_name}. Unloading...")
        del model
        del tokenizer
        if torch.cuda.is_available():
        torch.cuda.empty_cache()
            print("GPU memory cache cleared.")
        else:
            print(f"Model {model_name} was not initialized, no cleanup needed.")
        
    best_params_path = os.path.join(base_results_path, "best_params")
    os.makedirs(best_params_path, exist_ok=True)
    
    with open(os.path.join(best_params_path, "best_params_cotr.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    
    print("\nBest parameters saved to best_params_cotr.json")
    
    print("\n--- Best parameters summary ---")
    for lang_code, params in best_params.items():
        print(f"\n{lang_code}:")
        print(f"  Model: {params['model']}")
        print(f"  F1: {params['f1']:.4f}")
        print(f"  Parameters: {params['params']}")

def main():
    parser = argparse.ArgumentParser(description="Run NER CoTR experiments")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct", help="Model to use, comma-separated if multiple models")
    parser.add_argument("--languages", type=str, default="sw,ha", help="Language code(s), comma-separated if multiple")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per language")
    parser.add_argument("--pipeline_types", nargs='+', default=['multi_prompt', 'single_prompt'], choices=['multi_prompt', 'single_prompt'], help="CoTR pipeline types to run")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Shot settings to evaluate (zero_shot, few_shot)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for generation (overrides standard if set)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p for generation (overrides standard if set)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k for generation (overrides standard if set)")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens for generation (overrides standard if set)")
    parser.add_argument("--max_translation_tokens", type=int, default=200, help="Max tokens for translation steps (used in multi_prompt)")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search (if > 1, do_sample is set to False)")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty (overrides standard if set)")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/ner/cotr", help="Base directory for all outputs.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (e.g., test, dev)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    args = parser.parse_args()
    
   
    current_temp = args.temperature
    current_top_p = args.top_p
    current_top_k = args.top_k
    current_max_tokens = args.max_tokens
    current_repetition_penalty = args.repetition_penalty
    current_num_beams = args.num_beams
    
    token = get_token()
    login(token=token)
    
    if "," in args.models:
        models = args.models.split(",")
    else:
        models = [args.models]
    
    if "," in args.languages:
        lang_codes = args.languages.split(",")
    else:
        lang_codes = [args.languages]

    base_results_path = args.base_output_dir
    os.makedirs(base_results_path, exist_ok=True)
    os.makedirs(os.path.join(base_results_path, 'summaries'), exist_ok=True)
    
    print("\n--- Loading MasakhaNER Data ---")
    masakhaner_samples = {}
    for lang_code in lang_codes:
        print(f"Loading data for {lang_code}...")
        samples_df_full = load_masakhaner_samples(
            lang_code=lang_code,
            split=args.split,
            num_samples=None,
            seed=args.seed
        )
        if samples_df_full.empty:
            print(f"WARNING: No samples found for {lang_code} in split '{args.split}'. Skipping this language.")
            masakhaner_samples[lang_code] = pd.DataFrame()
        else:
            actual_loaded = len(samples_df_full)
            if args.num_samples is not None and actual_loaded > args.num_samples:
                masakhaner_samples[lang_code] = samples_df_full.sample(n=args.num_samples, random_state=args.seed)
            else:
                masakhaner_samples[lang_code] = samples_df_full
            print(f"Using {len(masakhaner_samples[lang_code])} samples for {lang_code}.")

            required_cols = ['text', 'entities'] 
            if not all(col in masakhaner_samples[lang_code].columns for col in required_cols):
                print(f"ERROR: Loaded data for {lang_code} is missing required columns ({required_cols}). Skipping this language.")
                masakhaner_samples[lang_code] = pd.DataFrame() 

    all_experiment_summaries = []

    for model_name_str in models:
        print(f"\n====== Starting experiments for model: {model_name_str} ======")
        model_initialized = False
        tokenizer, model = None, None
        try:
            print(f"  Attempting to initialize {model_name_str}...")
            tokenizer, model = initialize_model(model_name_str)
            model_initialized = True
            print(f"Model {model_name_str} initialized successfully.")
        except Exception as e_init:
            print(f"ERROR: Failed to initialize model {model_name_str}: {e_init}. Skipping this model.")
            continue #

        model_language_results = [] 

        for lang_code in lang_codes:
            print(f"\n--- Processing Language: {lang_code} for model {model_name_str} ---")
            if lang_code not in masakhaner_samples or masakhaner_samples[lang_code].empty:
                print(f"  Skipping {lang_code} due to missing or invalid data.")
                continue

            current_samples_df = masakhaner_samples[lang_code]

           
            default_lang_params = STANDARD_PARAMETERS.get(lang_code, STANDARD_PARAMETERS.get('sw', {})) # Default to sw if lang unknown
            lang_specific_params = LANGUAGE_PARAMETERS.get(lang_code, default_lang_params) # Get lang specific, or default

            effective_temp = current_temp if current_temp is not None else lang_specific_params.get("temperature", default_lang_params.get("temperature"))
            effective_top_p = current_top_p if current_top_p is not None else lang_specific_params.get("top_p", default_lang_params.get("top_p"))
            effective_top_k = current_top_k if current_top_k is not None else lang_specific_params.get("top_k", default_lang_params.get("top_k"))
            effective_max_tokens = current_max_tokens if current_max_tokens is not None else lang_specific_params.get("max_tokens", default_lang_params.get("max_tokens"))
            effective_rep_penalty = current_repetition_penalty if current_repetition_penalty is not None else lang_specific_params.get("repetition_penalty", default_lang_params.get("repetition_penalty"))

            effective_trans_temp = args.temperature if args.temperature is not None else lang_specific_params.get("trans_temp", default_lang_params.get("trans_temp"))
            effective_trans_top_p = args.top_p if args.top_p is not None else lang_specific_params.get("trans_top_p", default_lang_params.get("trans_top_p"))
            effective_trans_top_k = args.top_k if args.top_k is not None else lang_specific_params.get("trans_top_k", default_lang_params.get("trans_top_k"))
            effective_max_trans_tokens = args.max_translation_tokens # Always use arg for trans max tokens
            effective_trans_rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else lang_specific_params.get("trans_repetition_penalty", default_lang_params.get("trans_repetition_penalty"))

            print(f"  Base Params: Temp={effective_temp:.2f}, TopP={effective_top_p:.2f}, TopK={effective_top_k}, MaxTok={effective_max_tokens}, RepPen={effective_rep_penalty:.2f}")
            print(f"  Trans Params: Temp={effective_trans_temp:.2f}, TopP={effective_trans_top_p:.2f}, TopK={effective_trans_top_k}, MaxTok={effective_max_trans_tokens}, RepPen={effective_trans_rep_penalty:.2f}")

            for pipeline_type_to_run in args.pipeline_types:
                for shot_setting_str in args.shot_settings:
                    use_few_shot_to_run = (shot_setting_str == 'few_shot')
                    print(f"\n    Attempting to run: Model='{model_name_str}', Lang='{lang_code}', Pipeline='{pipeline_type_to_run}', Shot='{shot_setting_str}'")
                    print(f"      Effective NER Params: temp={effective_temp:.2f}, top_p={effective_top_p:.2f}, top_k={effective_top_k}, max_tokens={effective_max_tokens}, rep_penalty={effective_rep_penalty:.2f}")
                    if pipeline_type_to_run == 'multi_prompt':
                        print(f"      Effective Translation Params: temp={effective_trans_temp:.2f}, top_p={effective_trans_top_p:.2f}, top_k={effective_trans_top_k}, max_tokens={effective_max_trans_tokens}, rep_penalty={effective_trans_rep_penalty:.2f}")
                    
                    
                    ner_params_dict = {
                        "temperature": effective_temp,
                        "top_p": effective_top_p,
                        "top_k": effective_top_k,
                        "max_new_tokens": effective_max_tokens,
                        "repetition_penalty": effective_rep_penalty,
                        "num_beams": current_num_beams
                    }
                    
                    trans_params_dict = {
                        "temperature": effective_trans_temp,
                        "top_p": effective_trans_top_p,
                        "top_k": effective_trans_top_k,
                        "max_new_tokens": effective_max_trans_tokens,
                        "repetition_penalty": effective_trans_rep_penalty,
                        "num_beams": current_num_beams
                    }

                    summary_result = run_experiment(
                        model_name=model_name_str,
                        tokenizer=tokenizer,
                        model=model,
                        samples_df=current_samples_df,
                        lang_code=lang_code,
                        base_results_path=base_results_path,
                        pipeline_type=pipeline_type_to_run,
                        use_few_shot=use_few_shot_to_run,
                        ner_params=ner_params_dict,
                        trans_params=trans_params_dict,
                    )

                    if summary_result is not None:
                        print(f"    SUCCESS: Experiment completed for {lang_code}, {pipeline_type_to_run}, {shot_setting_str}.")
                        model_language_results.append(summary_result)
                        all_experiment_summaries.append(summary_result)
                    else:
                        print(f"    FAILURE/SKIP: Experiment returned None for {lang_code}, {pipeline_type_to_run}, {shot_setting_str}.")
            
        if model_initialized:
            print(f"\n====== Finished experiments for model {model_name_str}. Unloading... ======")
            del model
        del tokenizer
            model, tokenizer = None, None 
            if torch.cuda.is_available():
        torch.cuda.empty_cache()
                print("GPU memory cache cleared.")
            else:
                print(f"Model {model_name_str} was not initialized, no cleanup needed.")

    if all_experiment_summaries:
        print("\n--- Aggregating Overall Summary --- ")
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        overall_summary_path = os.path.join(base_results_path, 'summaries', f'ner_cotr_ALL_experiments_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        try:
            overall_summary_df.to_csv(overall_summary_path, index=False, float_format='%.4f')
            print(f"\n=== Overall CoTR NER Summary Saved ===")
            print(f"Path: {overall_summary_path}")
            print(overall_summary_df.to_string())
        except Exception as e_save:
            print(f"ERROR saving overall summary to {overall_summary_path}: {e_save}")

    else:
        print("\nNo successful experiments were completed. No overall summary generated.")

    print("\n====== NER CoTR Script Finished ======")

if __name__ == "__main__":
    main() 