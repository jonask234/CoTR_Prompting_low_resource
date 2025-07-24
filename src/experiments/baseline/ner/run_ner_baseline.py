#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import torch
import time
from tqdm import tqdm
import numpy as np
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import ast
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Fügt das Projekt-Stammverzeichnis zum Python-Pfad hinzu
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path: # Verhindert Duplikate
    sys.path.insert(0, project_root)

# Importiert notwendige Funktionen
from src.utils.data_loaders.load_masakhaner import load_masakhaner_samples
from src.experiments.baseline.ner.ner_baseline import (
    initialize_model,
    evaluate_ner_baseline,
    calculate_ner_metrics,
    create_dummy_ner_data
)
from config import get_token
from huggingface_hub import login

# Deaktiviert die Parallelitätswarnung für Tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Vereinheitlichte Generierungsparameter ---
# Standardwerte für Kern-Generierungsparameter
UNIFIED_GENERATION_PARAMETERS_CORE = {
    "temperature": 0.2,  # Etwas niedriger für deterministischere NER
    "top_p": 0.85,
    "top_k": 35,
    "repetition_penalty": 1.1,
    "do_sample": True # Standardmäßig Sampling, Temp wird es steuern
}

# Standardmäßige aufgabenspezifische max_tokens für Baseline NER (für die generierte Entitätenliste)
MAX_TOKENS_BASELINE_NER_OUTPUT = 150 

# --- Modellspezifische Überschreibungen ---
# Diese können UNIFIED_GENERATION_PARAMETERS_CORE oder MAX_TOKENS_BASELINE_NER_OUTPUT überschreiben
MODEL_SPECIFIC_OVERRIDES_BASELINE = {
    "CohereLabs/aya-expanse-8b": {
        "temperature": 0.225, # Strenger für Aya (Beispiel)
        "repetition_penalty": 1.15,
        "max_tokens": 180, 
        "top_p": 0.85,
        "top_k": 35,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "temperature": 0.15,
        "top_p": 0.80,
        "max_tokens": 160,
    }
}

logger = logging.getLogger(__name__)

# Modellspezifische Anpassungen (können erweitert werden)
# Dies sind Basiseinstellungen, die durch CLI-Argumente überschrieben werden können.
MODEL_SPECIFIC_ADJUSTMENTS = {
    "CohereLabs/aya-expanse-8b": {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 30,
        "repetition_penalty": 1.1,
        "max_tokens": 150 
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "temperature": 0.1,
        "top_p": 0.85,
        "top_k": 35,
        "repetition_penalty": 1.05,
        "max_tokens": 160
    }
    # Füge bei Bedarf weitere Modelle hinzu
}

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run Baseline NER Experiments with MasakhaNER.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated model names.")
    parser.add_argument("--langs", type=str, default="swa,hau", help="Comma-separated MasakhaNER language codes (e.g., swa, hau for Swahili, Hausa from MasakhaNER). Ensure these match dataset keys.")
    parser.add_argument("--num_samples", type=int, default=80, help="Number of samples per language. Default: 80")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split.")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/ner/baseline_masakhaner", help="Base directory to save results and summaries.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and reproducibility.")
    parser.add_argument(
        "--prompt_in_lrl", 
        action=argparse.BooleanOptionalAction, # Ermöglicht --prompt_in_lrl und --no-prompt_in_lrl
        default=True, # Setzt den Standardwert auf True
        help="If set, prompt instructions will be in LRL (few-shot examples remain in English). Default: True (LRL instructions)."
    )
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], choices=['zero_shot', 'few_shot'], help="Prompting strategies.")
    parser.add_argument(
        "--compare_shots", 
        action="store_true", 
        help="Run both few-shot (English examples) and zero-shot evaluations. Overrides --few_shot."
    )
    parser.add_argument(
        "--few_shot", 
        action="store_true", 
        help="Enable few-shot prompting (English examples). If --compare_shots is also set, this is ignored as both are run."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/work/bbd6522/results/ner/baseline", 
        help="Base output directory for results, summaries, and plots."
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default=None, 
        help="HuggingFace API token (optional, reads from config if not provided)."
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="If set, overwrite existing result files instead of skipping experiments."
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run in test mode: uses a very small number of samples (e.g., 5) for quick testing."
    )
    # Argumente für Generierungsparameter
    parser.add_argument("--temperature", type=float, default=None, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling top_p.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Global override for max_new_tokens for generation, affecting the NER output length.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Global override for repetition_penalty.")
    parser.add_argument(
        "--do_sample", 
        type=lambda x: (str(x).lower() == 'true'), 
        default=None,  # Standardmäßig None, damit es bei Nichtfestlegung aus der Temperatur abgeleitet werden kann
        help="Global override for do_sample (True/False). If not set, derived from temperature."
    )
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")

    args = parser.parse_args()
    return args

# --- Standardisierte Parameter (abgestimmt mit CoTR) ---
STANDARD_PARAMETERS = {
    "temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_tokens": 200, "repetition_penalty": 1.1
}

# Sprachspezifische Parameter (abgestimmt mit CoTR)
LANGUAGE_PARAMETERS = {
    "sw": { # Swahili (Beispiel - bei Bedarf anpassen)
        "temperature": 0.25, "top_p": 0.85, "top_k": 35, "max_tokens": 180, "repetition_penalty": 1.15
    },
    "ha": { # Hausa (Beispiel - bei Bedarf anpassen)
        "temperature": 0.25, "top_p": 0.85, "top_k": 35, "max_tokens": 180, "repetition_penalty": 1.15
    }
    # Füge weitere Sprachen hinzu, wenn eine Feinabstimmung erfolgt
}

# --- Modellspezifische Anpassungen (Funktion) ---
def apply_model_specific_adjustments(params: Dict, model_name: str) -> Dict:
    """Wendet modellspezifische Anpassungen auf Parameter an."""
    adjusted_params = params.copy()
    if "aya" in model_name.lower():
        adjusted_params["temperature"] = max(0.1, adjusted_params["temperature"] * 0.9)
        print(f"  Applied Aya adjustments: Temp={adjusted_params['temperature']:.2f}")
    elif "qwen" in model_name.lower():
        adjusted_params["top_p"] = max(0.7, adjusted_params["top_p"] * 0.9)
        adjusted_params["top_k"] = 35
        print(f"  Applied Qwen adjustments: TopP={adjusted_params['top_p']:.2f}, TopK={adjusted_params['top_k']}")
    return adjusted_params

# --- Umstrukturierte Experiment-Runner-Funktion ---
def run_experiment(
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    use_few_shot: bool,
    base_results_path: str,
    effective_params: Dict[str, Any],
    overwrite_results: bool,
    prompt_in_lrl_cli: bool
) -> Optional[Dict[str, Any]]: # Gibt Zusammenfassungs-Dict oder None zurück
    """Führt ein einzelnes NER-Experiment für ein bestimmtes Modell, eine Sprache und eine Konfiguration aus."""
    experiment_start_time = time.time() # Initialisiert am Anfang der Funktion

    if samples_df.empty:
        logger.warning(f"No samples for {lang_code}, skipping {model_name}.")
        return None

    model_short_name = model_name.split('/')[-1]

    # Vereinfacht: Baseline verwendet jetzt immer EN-instruct Prompts
    shot_setting_str = "few_shot" if use_few_shot else "zero_shot"
    # Bestimmt den Prompt-Sprachstring basierend auf dem übergebenen Flag
    if prompt_in_lrl_cli:
        prompt_lang_str = f"{lang_code}-instruct"
    else:
        prompt_lang_str = "EN-instruct" 
    logger.info(f"Prompt language for this run: {prompt_lang_str}") # Log hinzugefügt
    
    results_dir_for_run = os.path.join(base_results_path, "results")
    summaries_dir_for_run = os.path.join(base_results_path, "summaries")
    os.makedirs(results_dir_for_run, exist_ok=True)
    os.makedirs(summaries_dir_for_run, exist_ok=True)

    # Definiert Dateipfade (konsistente Benennung)
    results_file_for_run = os.path.join(results_dir_for_run, f"results_baseline_{shot_setting_str[0]}s_ner_{lang_code}_{model_short_name}.csv")
    summary_file_for_run = os.path.join(summaries_dir_for_run, f"summary_baseline_{shot_setting_str[0]}s_ner_{lang_code}_{model_short_name}.csv")

    # --- Lade-/Ausführungslogik ---
    results_df = pd.DataFrame() # Initialisiert als leer
    run_evaluation_and_save = False

    if overwrite_results:
        logging.info(f"Overwrite_results is True. Will run experiment for {model_name}, {lang_code}, {shot_setting_str}.")
        run_evaluation_and_save = True
    elif not os.path.exists(results_file_for_run):
        logging.info(f"Results file does not exist: {results_file_for_run}. Will run experiment for {model_name}, {lang_code}, {shot_setting_str}.")
        run_evaluation_and_save = True
    else: # Datei existiert und overwrite_results ist False
        logging.info(f"Results file exists: {results_file_for_run} and overwrite_results is False. Attempting to load.")
        try:
            loaded_df = pd.read_csv(results_file_for_run)
            logging.info(f"Successfully loaded existing results from {results_file_for_run}.")
            # Prüft auf die Zielspalte 'ground_truth_entities'
            if 'ground_truth_entities' in loaded_df.columns:
                results_df = loaded_df
            elif 'entities' in loaded_df.columns: # Fallback 1: 'entities'
                logging.info("Found 'entities' column in existing CSV; renaming to 'ground_truth_entities'.")
                results_df = loaded_df.rename(columns={'entities': 'ground_truth_entities'})
            elif 'gold_entities' in loaded_df.columns: # Fallback 2: 'gold_entities'
                logging.info("Found 'gold_entities' column in existing CSV; renaming to 'ground_truth_entities'.")
                results_df = loaded_df.rename(columns={'gold_entities': 'ground_truth_entities'})
            else:
                logging.warning(
                    f"'ground_truth_entities' (nor fallbacks 'entities', 'gold_entities') column not found in {results_file_for_run}. "
                    f"Columns present: {loaded_df.columns.tolist()}. Will re-run experiment for this configuration as data seems incomplete."
                )
                run_evaluation_and_save = True # Mark for re-run due to missing critical column
        except Exception as e:
            logging.error(f"Could not load or properly process existing results {results_file_for_run}: {e}. Will re-run experiment.")
            run_evaluation_and_save = True # Mark for re-run
    
    if run_evaluation_and_save:
        logging.info(f"Proceeding to run evaluation for {model_name}, {lang_code}, {shot_setting_str}.")
        # Run the core evaluation if results_df is not loaded
        if results_df.empty:
            logger.info(f"Running NER Baseline: Model={model_name}, Lang={lang_code}, Shot Type={shot_setting_str}, Prompt Lang={prompt_lang_str}")
            try:
                results_df = evaluate_ner_baseline(
                    tokenizer=tokenizer,
                    model=model,
                    model_name=model_name, # Pass model_name for potential internal use/logging
                    samples_df=samples_df.copy(), # Pass a copy of the original samples_df
                    lang_code=lang_code,
                    use_few_shot=use_few_shot,
                    prompt_in_lrl=prompt_in_lrl_cli,
                    temperature=effective_params["temperature"],
                    top_p=effective_params["top_p"],
                    top_k=effective_params["top_k"],
                    max_tokens=effective_params["max_tokens"],
                    repetition_penalty=effective_params["repetition_penalty"],
                    do_sample=effective_params["do_sample"] # Pass do_sample flag
                )

                if results_df is None or results_df.empty:
                    logging.error("Baseline evaluation returned None or empty DataFrame. No results to save.")
                    # results_df remains empty or as loaded if partial load was successful before deciding to re-run
                else:
                    results_df.to_csv(results_file_for_run, index=False)
                    logging.info(f"Results saved to {results_file_for_run}")

            except Exception as e: 
                logging.error(f"Error during NER Baseline evaluation for {lang_code}, {model_short_name}, {shot_setting_str}: {e}", exc_info=True)
                # results_df remains as it was (empty or loaded) - essentially the run failed
                # No specific return None here, will be caught by the check below.
            
    # --- Calculate Metrics ---
    if results_df.empty: # Check if results_df is empty after load/run attempts
        logging.error(f"No results DataFrame available for {model_name}, {lang_code}, {shot_setting_str} after attempting load/run. Skipping summary.")
        return None

    logging.info(f"Calculating metrics from results for {model_name}, {lang_code}, {shot_setting_str}...")
    precisions, recalls, f1s = [], [], []
    successful_samples = 0

    # Convert string representation of lists/tuples back if loaded from CSV
    needs_conversion = False
    # Check the type of the first non-null element
    first_gold = results_df['ground_truth_entities'].dropna().iloc[0] if not results_df['ground_truth_entities'].dropna().empty else None
    if first_gold is not None and isinstance(first_gold, str):
        needs_conversion = True
        print("Converting entity strings from CSV back to lists/tuples...")
        def safe_literal_eval(val):
            # Behandelt potenzielle Fehler während der Auswertung eleganter
            if pd.isna(val): return []
            try:
                evaluated = ast.literal_eval(val)
                # Stellt sicher, dass es nach der Auswertung eine Liste ist
                return evaluated if isinstance(evaluated, list) else []
            except (ValueError, SyntaxError, TypeError):
                print(f"Warning: Could not evaluate entity string: {str(val)[:50]}...")
                return [] # Gibt leere Liste zurück, wenn ein Fehler auftritt

    # Prüft auf die Spalte 'predicted_entities' und ihren Typ für die Konvertierung
    needs_pred_conversion = False
    if 'predicted_entities' in results_df.columns:
        first_pred = results_df['predicted_entities'].dropna().iloc[0] if not results_df['predicted_entities'].dropna().empty else None
        if first_pred is not None and isinstance(first_pred, str):
            needs_pred_conversion = True
            if not needs_conversion: # Nur diesen Text drucken, wenn er nicht bereits für gold_entities gedruckt wurde
                print("Converting predicted entity strings from CSV back to lists/tuples...")
    else:
        # Fallback für 'predicted_entities' falls fehlend
        if 'predictions' in results_df.columns:
            logging.info("Found 'predictions' column in existing CSV; renaming to 'predicted_entities'.")
            results_df.rename(columns={'predictions': 'predicted_entities'}, inplace=True)
            # Prüft erneut auf Konvertierung nach Umbenennung
            first_pred = results_df['predicted_entities'].dropna().iloc[0] if not results_df['predicted_entities'].dropna().empty else None
            if first_pred is not None and isinstance(first_pred, str):
                needs_pred_conversion = True
        else:
            logging.warning(
                f"'predicted_entities' (and fallback 'predictions') column not found in results_df. "
                f"Columns present: {results_df.columns.tolist()}. Metrics for predictions will be 0."
            )
            results_df['predicted_entities'] = [[] for _ in range(len(results_df))] # Füllt mit leeren Listen

    for idx, row in results_df.iterrows():
        try:
            # Wendet Konvertierung an, falls erforderlich
            gold = safe_literal_eval(row['ground_truth_entities']) if needs_conversion else row['ground_truth_entities']
            
            # Behandelt die Konvertierung von 'predicted_entities' oder fehlende Spalte
            if 'predicted_entities' in row:
                pred = safe_literal_eval(row['predicted_entities']) if needs_pred_conversion else row['predicted_entities']
            else: # Sollte durch die Spaltenerstellung oben behandelt werden, aber als Schutzmaßnahme
                pred = []

            # Stellt sicher, dass die Datentypen Listen sind, bevor sie an Metriken übergeben werden
            if not isinstance(gold, list): gold = []
            if not isinstance(pred, list): pred = []

            # Verwendet die Metriken-Funktion aus ner_baseline.py
            metrics = calculate_ner_metrics(gold, pred)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            # Stellt sicher, dass der Schlüssel mit dem übereinstimmt, was calculate_ner_metrics zurückgibt ('f1_score')
            f1s.append(metrics.get('f1_score', 0.0)) # Verwendet .get() für Sicherheit
            successful_samples += 1
        except Exception as e_metric:
            print(f"Error calculating metrics for row {idx}: {e_metric}")
            # Fügt Standardwerte an, um abstürzen zu vermeiden
            precisions.append(0.0); recalls.append(0.0); f1s.append(0.0)

    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    avg_f1 = np.mean(f1s) if f1s else 0.0
    total_runtime = time.time() - experiment_start_time # Berechnet Gesamtlaufzeit

    summary_data = {
        "model_name": model_short_name, 
        "language": lang_code,
        "prompt_lang": prompt_lang_str, # Dies ist jetzt dynamisch
        "shot_setting": shot_setting_str,
        "num_samples_processed": len(results_df),
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1,
        "runtime_seconds": total_runtime,
        **effective_params # Loggt alle effektiven Generierungsparameter, die für diesen Lauf verwendet wurden
    }

    summary_df = pd.DataFrame([summary_data])
    try:
        summary_path = os.path.join(summaries_dir_for_run, f"summary_baseline_{shot_setting_str[0]}s_ner_{lang_code}_{model_short_name}.csv")
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        logging.info(f"Summary metrics saved to {summary_path}")
    except Exception as e_save_sum:
        print(f"ERROR saving summary file {summary_path}: {e_save_sum}")

    return summary_data # Gibt das Dictionary zurück, um es für die Aggregation zu verwenden

# --- Hauptfunktion ---
def main():
    """Main function to orchestrate NER baseline experiments."""
    args = parse_cli_args()

    # Konfiguriert das Logging-Level
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Setup seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # HF Login
    token = args.hf_token if args.hf_token else get_token()
    if token:
        try:
            login(token=token)
            logging.info("Successfully logged into Hugging Face Hub.")
        except Exception as e:
            logging.error(f"Failed to login to Hugging Face Hub: {e}. Gated models might be inaccessible.")
    else:
        logging.warning("No Hugging Face token provided. Gated models might be inaccessible.")

    model_names_list = [m.strip() for m in args.models.split(',')]
    lang_codes_list = [lang.strip() for lang in args.langs.split(',')]

    all_experiment_summaries = []
    overall_summary_dir = os.path.join(args.base_output_dir, "summaries_overall") 
    os.makedirs(overall_summary_dir, exist_ok=True)
    overall_plots_dir = os.path.join(args.base_output_dir, "plots_overall")
    os.makedirs(overall_plots_dir, exist_ok=True)

    logging.info(f"NER Baseline Outputs Target: {args.base_output_dir}")
    logging.info(f"Overall Summaries in: {overall_summary_dir}")
    logging.info(f"Overall Plots in: {overall_plots_dir}")

    num_samples_to_load = args.num_samples
    current_seed = args.seed

    for model_name_str in model_names_list:
        logging.info(f"\n===== Initializing Model: {model_name_str} =====")
        
        # Initialisiert das Modell einmal pro Schleife
        model = None
        tokenizer = None
        del model
        del tokenizer
        gc.collect() # Vorschlägt Garbage Collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"Cleared CUDA cache and called gc.collect() before initializing {model_name_str}")
            
        current_model_tokenizer, current_model_object = None, None # Initialisiert für diese Iteration
        try:
            current_model_tokenizer, current_model_object = initialize_model(model_name_str)
        except Exception as e_init:
            logging.error(f"Failed to initialize model {model_name_str}: {e_init}. Skipping this model.", exc_info=True)
            if current_model_object is not None: del current_model_object
            if current_model_tokenizer is not None: del current_model_tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect() # Stellt sicher, dass Garbage Collection nach del und Cache-Leerung aufgerufen wird
            continue # Springt zum nächsten Modell

        # Defensive check: Ensure model and tokenizer are loaded before proceeding
        if current_model_tokenizer is None or current_model_object is None:
            logging.critical(f"Model or tokenizer is None for {model_name_str} even after initialize_model call did not raise an exception. This should not happen. Skipping model.")
            # Führt die Bereinigung ähnlich dem Ausnahmefall durch
            if current_model_object is not None: del current_model_object
            if current_model_tokenizer is not None: del current_model_tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
            continue # Springt zum nächsten Modell
            
        # Wenn wir hier sind, hat initialize_model erfolgreich abgeschlossen, und model/tokenizer sind nicht None.
        effective_gen_params = get_effective_generation_params(model_name_str, args)

        for lang_code in lang_codes_list:
            logger.info(f"\n--- Processing Language: {lang_code} for model {model_name_str} ---")
            
            current_samples_df = load_masakhaner_samples(
                lang_code=lang_code, 
                split=args.data_split, 
                num_samples=args.num_samples, 
                seed=args.seed
            )

            if current_samples_df.empty:
                logger.warning(f"No MasakhaNER samples for {lang_code} (split {args.data_split}). Skipping config.")
                continue
            else:
                logger.info(f"Loaded {len(current_samples_df)} samples for {lang_code} (split {args.data_split}).")

            # Feste Prompt-Sprachkonfiguration: Nur EN-instruct für Baseline-Vereinfachung
            prompt_lang_code_fixed = "EN"
            use_lrl_prompt_bool_fixed = False
            prompt_lang_str_fixed = "EN-instruct" # Dies wird durch use_lrl_prompt_bool_fixed bestimmt

            # Iteriert durch die angegebenen Shot-Einstellungen aus dem CLI
            for shot_setting_str in args.shot_settings: # z.B. ["zero_shot", "few_shot"]
                use_few_shot_bool = (shot_setting_str == "few_shot")
                
                # Bestimmt den tatsächlichen Prompt-Sprachstring für Protokollierung/Zusammenfassung basierend auf args.prompt_in_lrl
                actual_prompt_lang_for_log = f"{lang_code}-instruct" if args.prompt_in_lrl else "EN-instruct"
                
                exp_config_tuple = (lang_code, model_name_str, shot_setting_str, prompt_lang_str_fixed)
                logger.info(f"\n--- Running NER Baseline: Model={model_name_str}, Lang={lang_code}, Shot={shot_setting_str}, PromptLang={actual_prompt_lang_for_log} ---")
                logger.info(f"  Effective Generation Params: {effective_gen_params}")

                experiment_details_for_run = {
                    "model_name": model_name_str,
                    "tokenizer": current_model_tokenizer,
                    "model": current_model_object,
                    "samples_df": current_samples_df,
                    "lang_code": lang_code,
                    "use_few_shot": use_few_shot_bool,
                    "base_results_path": args.base_output_dir,
                    "effective_params": effective_gen_params,
                    "overwrite_results": args.overwrite_results,
                    "prompt_in_lrl_cli": args.prompt_in_lrl
                }
                
                # Ruft run_experiment auf
                summary_data = run_experiment(**experiment_details_for_run)
                
                if summary_data:
                    all_experiment_summaries.append(summary_data)
            # End of shot_setting_str loop
        # End of lang_code loop
        
        # Korrekt eingerückte Modell-Bereinigungsblock:
        logging.info(f"Finished all experiments for model {model_name_str}. Model and tokenizer unloaded, cache cleared.")
        if current_model_object is not None:
            del current_model_object
            current_model_object = None # Stellt sicher, dass es None ist, nachdem del aufgerufen wurde
        if current_model_tokenizer is not None:
            del current_model_tokenizer
            current_model_tokenizer = None # Stellt sicher, dass es None ist, nachdem del aufgerufen wurde
        
        # Vorschlägt aggressiveres Garbage Collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"CUDA cache cleared after processing model {model_name_str}")

    if all_experiment_summaries:
        overall_summary_df = pd.DataFrame(all_experiment_summaries)
        if not overall_summary_df.empty:
            # Stellt sicher, dass numerische Spalten numerisch sind für das Plotten und Anzeigen
            numeric_cols_for_summary = ['num_samples_processed', 'avg_runtime_seconds_per_sample', 'precision', 'recall', 'f1_score',
                                 'temperature', 'top_p', 'top_k', 'max_tokens', 'repetition_penalty']
            for nc in numeric_cols_for_summary:
                if nc in overall_summary_df.columns:
                    overall_summary_df[nc] = pd.to_numeric(overall_summary_df[nc], errors='coerce')
            
            summary_filename_overall = f"ner_baseline_ALL_experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            overall_summary_path_csv = os.path.join(overall_summary_dir, summary_filename_overall)
            overall_summary_df.to_csv(overall_summary_path_csv, index=False, float_format='%.4f')
            logging.info(f"\nOverall summary of NER Baseline experiments saved to: {overall_summary_path_csv}")
            print("\nOverall NER Baseline Summary:")
            try:
                print(overall_summary_df.to_string(index=False))
            except Exception as e_print_df:
                logging.error(f"Could not print overall_summary_df to string: {e_print_df}")
                print(overall_summary_df.head())

            # Plotting (example for F1 score)
            if 'f1_score' in overall_summary_df.columns:
                # Erstellt 'experiment_key' für eine einzigartige x-Achse-Kennung
                overall_summary_df['experiment_key'] = overall_summary_df['model_name'] + "_" + \
                                                       overall_summary_df['language'] + "_" + \
                                                       overall_summary_df['shot_setting']
                
                plt.figure(figsize=(15, 8))
                sns.barplot(data=overall_summary_df, x='experiment_key', y='f1_score', hue='language')
                plt.xticks(rotation=45, ha='right')
                plt.title('NER Baseline F1 Scores by Experiment Configuration')
                plt.ylabel('F1 Score')
                plt.xlabel('Experiment Configuration')
                plt.tight_layout()
                plot_path = os.path.join(overall_plots_dir, "ner_baseline_f1_scores.png")
                try:
                    plt.savefig(plot_path)
                    logging.info(f"F1 score plot saved to {plot_path}")
                except Exception as e_plot:
                    logging.error(f"Failed to save F1 plot: {e_plot}")
                finally:
                    plt.close() # Stellt sicher, dass das Diagramm geschlossen wird
            else:
                logging.warning("'f1_score' column not found in overall summary. Skipping plot.")
        else:
            logging.info("Overall summary DataFrame is empty. No CSV or plots generated.")
    else:
        logging.info("No summaries were collected from any experiment. No overall summary or plots.")

    logging.info("\nAll NER Baseline experiments completed.")

def get_effective_generation_params(model_name_str: str, cli_args: argparse.Namespace) -> Dict[str, Any]:
    """Holt und kombiniert Generierungsparameter aus verschiedenen Quellen."""
    
    # Beginnt mit den Standardparametern des Systems
    params = UNIFIED_GENERATION_PARAMETERS_CORE.copy()
    params["max_tokens"] = MAX_TOKENS_BASELINE_NER_OUTPUT

    # Wendet modellspezifische Überschreibungen an
    model_overrides = MODEL_SPECIFIC_OVERRIDES_BASELINE.get(model_name_str, {})
    params.update(model_overrides)

    # Wendet sprachspezifische Überschreibungen an
    lang_overrides = LANGUAGE_PARAMETERS.get(cli_args.langs, {}) # Angenommen, langs ist hier ein einzelner Code
    params.update(lang_overrides)

    # Wendet globale CLI-Überschreibungen an
    if cli_args.temperature is not None: params["temperature"] = cli_args.temperature
    if cli_args.top_p is not None: params["top_p"] = cli_args.top_p
    if cli_args.top_k is not None: params["top_k"] = cli_args.top_k
    if cli_args.max_tokens is not None: params["max_tokens"] = cli_args.max_tokens
    if cli_args.repetition_penalty is not None: params["repetition_penalty"] = cli_args.repetition_penalty
    
    # Leitet do_sample ab, wenn es nicht explizit festgelegt ist
    if cli_args.do_sample is not None:
        params["do_sample"] = cli_args.do_sample
    else:
        # Wenn die Temperatur > 0 ist, ist Sampling wahrscheinlich beabsichtigt
        params["do_sample"] = params.get("temperature", 0) > 0.0

    return params

def plot_ner_metrics(summary_df: pd.DataFrame, plots_dir: str):
    """Erstellt und speichert Diagramme, die NER-Metriken zusammenfassen."""
    if summary_df.empty:
        logging.warning("Summary DataFrame is empty. Skipping plotting.")
        return

    # Verbessert die Lesbarkeit von Modellnamen
    summary_df['model_short'] = summary_df['model'].apply(lambda x: x.split('/')[-1])
    
    # Diagramm 1: F1-Score nach Modell und Sprache
    plt.figure(figsize=(12, 7))
    sns.barplot(data=summary_df, x='model_short', y='avg_f1_score', hue='language', dodge=True)
    plt.title('Average F1-Score by Model and Language')
    plt.ylabel('Average F1-Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Language')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'f1_by_model_language.png'))
    plt.close()

    # Diagramm 2: F1-Score nach Shot-Einstellung
    if 'shot_setting' in summary_df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=summary_df, x='shot_setting', y='avg_f1_score')
        plt.title('F1-Score Distribution by Shot Setting')
        plt.ylabel('Average F1-Score')
        plt.xlabel('Shot Setting')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'f1_by_shot_setting.png'))
        plt.close()

    # Diagramm 3: Laufzeit nach Modell
    if 'avg_runtime_seconds' in summary_df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=summary_df, x='model_short', y='avg_runtime_seconds', dodge=False)
        plt.title('Average Runtime per Sample by Model')
        plt.ylabel('Average Runtime (seconds)')
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'runtime_by_model.png'))
        plt.close()
        
    logging.info(f"Plots saved to {plots_dir}")

# --- Einstiegspunkt ---
if __name__ == "__main__":
    main() 