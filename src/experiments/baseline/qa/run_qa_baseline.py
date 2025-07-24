# thesis_project/run_qa_baseline.py

import sys
import os
import argparse
import torch
import random
import numpy as np
import pandas as pd
from huggingface_hub import login

# Tokenizer-Parallelität deaktivieren, um Warnungen zu vermeiden
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Projekt-Root zum Python-Pfad hinzufügen
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Notwendige Funktionen importieren
from src.utils.data_loaders.load_tydiqa import load_tydiqa_samples
from src.experiments.baseline.qa.qa_baseline import evaluate_qa_baseline, initialize_model
from config import get_token

# Globale Parameter für die Texterzeugung
# Ein einfacher Weg, um die gleichen Einstellungen überall zu verwenden
GENERATION_PARAMETERS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 50,
    "repetition_penalty": 1.2,
    "do_sample": True
}

def run_single_experiment(
    model_name, 
    tokenizer,
    model,
    samples_df, 
    lang_code, 
    base_output_dir,
    use_few_shot,
    prompt_in_lrl,
    overwrite_results
):
    # Führt ein einzelnes Experiment für eine bestimmte Konfiguration aus

    if samples_df.empty:
        print(f"Warning: No samples for {lang_code}. Skipping experiment for {model_name}.")
        return None

    shot_desc = "few_shot" if use_few_shot else "zero_shot"
    model_name_safe = model_name.replace("/", "_")
    
    # Dateinamen für Ergebnisse und Zusammenfassungen erstellen
    results_filename = f"results_baseline_{shot_desc}_qa_{lang_code}_{model_name_safe}.csv"
    results_dir = os.path.join(base_output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_filepath = os.path.join(results_dir, results_filename)

    # Prüfen, ob die Ergebnisse bereits existieren und nicht überschrieben werden sollen
    if not overwrite_results and os.path.exists(results_filepath):
        print(f"Results file exists: {results_filepath}. Loading existing results.")
        try:
            results_df = pd.read_csv(results_filepath)
        except Exception as e:
            print(f"Could not load existing results: {e}. Rerunning experiment.")
            results_df = None
    else:
        results_df = None

    # Führt die Auswertung durch, wenn keine Ergebnisse geladen wurden
    if results_df is None:
        print(f"\nProcessing {lang_code} with {model_name} (Baseline, {shot_desc})...")
        results_df = evaluate_qa_baseline(
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            samples_df=samples_df.copy(),
            lang_code=lang_code,
            use_few_shot=use_few_shot,
            prompt_in_lrl=prompt_in_lrl,
            # Übergibt die globalen Generierungsparameter
            temperature=GENERATION_PARAMETERS["temperature"],
            top_p=GENERATION_PARAMETERS["top_p"],
            top_k=GENERATION_PARAMETERS["top_k"],
            max_tokens=GENERATION_PARAMETERS["max_tokens"],
            repetition_penalty=GENERATION_PARAMETERS["repetition_penalty"],
            do_sample=GENERATION_PARAMETERS["do_sample"]
        )

    if results_df is None or results_df.empty:
        print(f"Warning: No results generated for {lang_code} with {model_name} ({shot_desc}).")
        return None

    # Ergebnisse in eine CSV-Datei speichern
    results_df.to_csv(results_filepath, index=False, float_format='%.4f')
    print(f"Results saved to {results_filepath}")

    # Eine Zusammenfassung der Metriken erstellen
    avg_f1 = results_df["f1_score"].mean()
    summary = {
        'model': model_name,
        'language': lang_code,
        'shot_type': shot_desc,
        'prompt_in_lrl': prompt_in_lrl,
        'f1_score': avg_f1,
        'num_samples': len(results_df)
    }
    return summary

def main():
    # Hauptfunktion zum Ausführen der QA-Baseline-Experimente
    parser = argparse.ArgumentParser(description="Run Baseline QA Experiments with TyDiQA.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct", help="Comma-separated model names.")
    parser.add_argument("--langs", type=str, default="en,sw,fi", help="Comma-separated TyDiQA language codes.")
    parser.add_argument("--num_samples", type=int, default=80, help="Number of samples per language.")
    parser.add_argument("--data_split", type=str, default="validation", help="Dataset split (validation or train).")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/qa/baseline_tydiqa", help="Base directory to save results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], help="Prompting strategies: 'zero_shot' or 'few_shot'.")
    parser.add_argument("--prompt_in_lrl", action="store_true", help="Use LRL for prompt instructions.")
    parser.add_argument("--overwrite_results", action="store_true", help="Overwrite existing result files.")
    
    args = parser.parse_args()
    
    # Zufalls-Seeds setzen, damit die Ergebnisse gleich bleiben
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Listen der zu testenden Modelle und Sprachen
    model_names = [m.strip() for m in args.models.split(',')]
    lang_codes = [l.strip() for l in args.langs.split(',')]
    
    # Hugging Face Login
    try:
        token = get_token()
        if token:
            login(token=token)
            print("Successfully logged into Hugging Face.")
    except Exception as e:
        print(f"Hugging Face login failed: {e}. Attempting to continue...")

    # Verzeichnisse für Ergebnisse und Zusammenfassungen erstellen
    summaries_dir = os.path.join(args.base_output_dir, "summaries")
    os.makedirs(summaries_dir, exist_ok=True)
    
    # Lädt die Samples für jede Sprache
    tydiqa_samples = {}
    for lang_code in lang_codes:
        print(f"Loading samples for {lang_code}...")
        samples_df = load_tydiqa_samples(
            lang_code=lang_code, 
            num_samples=args.num_samples,
            split=args.data_split,
            seed=args.seed
        )
        
        if not samples_df.empty:
            print(f"  Loaded {len(samples_df)} samples for {lang_code}.")
            tydiqa_samples[lang_code] = samples_df
        else:
            print(f"  No samples loaded for {lang_code}.")

    all_summaries = []

    # Iteriert durch jedes Modell
    for model_name in model_names:
        print(f"\n===== Initializing model: {model_name} =====")
        tokenizer, model = None, None
        try:
            tokenizer, model = initialize_model(model_name)
        except Exception as e:
            print(f"Failed to initialize model {model_name}: {e}. Skipping this model.")
            continue

        # Iteriert durch jede Sprache und Shot-Einstellung
        for lang_code in lang_codes:
            if lang_code not in tydiqa_samples:
                continue

            for shot_setting in args.shot_settings:
                use_few_shot = (shot_setting == 'few_shot')
                
                summary = run_single_experiment(
                    model_name=model_name,
                    tokenizer=tokenizer,
                    model=model,
                    samples_df=tydiqa_samples[lang_code],
                    lang_code=lang_code,
                    base_output_dir=args.base_output_dir,
                    use_few_shot=use_few_shot,
                    prompt_in_lrl=args.prompt_in_lrl,
                    overwrite_results=args.overwrite_results
                )
                if summary:
                    all_summaries.append(summary)
        
        # Gibt Speicher frei, nachdem alle Experimente für ein Modell abgeschlossen sind
        print(f"All experiments for {model_name} complete. Unloading model.")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Speichert eine Gesamtübersicht aller Experimente
    if all_summaries:
        overall_summary_df = pd.DataFrame(all_summaries)
        summary_filename = os.path.join(summaries_dir, "baseline_qa_ALL_experiments_summary.csv")
        overall_summary_df.to_csv(summary_filename, index=False, float_format='%.4f')
        print(f"\nOverall summary saved to: {summary_filename}")
        print("\nOverall Summary:")
        print(overall_summary_df.to_string())
    else:
        print("No summaries to save.")

if __name__ == "__main__":
    main()