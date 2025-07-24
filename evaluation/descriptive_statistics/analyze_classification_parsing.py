#!/usr/bin/env python3
"""
Dieses Skript analysiert die Erfolgsraten beim Parsen von Textklassifizierungs-Ergebnissen.
Es zählt, wie oft ein gültiges Label vorhergesagt wurde.
"""

import os
import pandas as pd
import glob

def analyze_classification_parsing(base_results_dir="results"):
    """Analysiert die Erfolgsraten auf Sample-Ebene für Klassifizierungsaufgaben."""
    
    # Dies sind die einzigen Labels, die als gültig angesehen werden.
    valid_labels = ["politics", "sports", "health", "business", "religion", "entertainment", "technology"]
    
    # Ein Dictionary, um die Ergebnisse zu speichern.
    results = {
        'baseline': {'total_samples': 0, 'valid_samples': 0, 'files_processed': 0},
        'cotr_mp': {'total_samples': 0, 'valid_samples': 0, 'files_processed': 0},
        'cotr_sp': {'total_samples': 0, 'valid_samples': 0, 'files_processed': 0}
    }
    
    detailed_results = []
    
    # Verarbeite die Baseline-Dateien.
    baseline_pattern = f"{base_results_dir}/classification_new/baseline/**/*.csv"
    baseline_files = glob.glob(baseline_pattern, recursive=True)
    
    print("BASELINE CLASSIFICATION FILES:")
    for file_path in baseline_files:
        try:
            df = pd.read_csv(file_path)
            if 'final_predicted_label' in df.columns:
                total_samples = len(df)
                valid_samples = df['final_predicted_label'].isin(valid_labels).sum()
                
                # Extrahiere Details aus dem Dateipfad.
                model = "unknown"
                language = "unknown"
                shot_type = "unknown"
                
                if "aya-23-8B" in file_path:
                    model = "aya-23-8B"
                elif "Qwen2.5-7B-Instruct" in file_path:
                    model = "Qwen2.5-7B-Instruct"
                
                if "/en/" in file_path:
                    language = "en"
                elif "/ha/" in file_path or "_ha" in file_path:
                    language = "ha"
                elif "/sw/" in file_path or "_sw" in file_path:
                    language = "sw"
                
                if "/fs/" in file_path:
                    shot_type = "fs"
                elif "/zs/" in file_path:
                    shot_type = "zs"
                
                results['baseline']['total_samples'] += total_samples
                results['baseline']['valid_samples'] += valid_samples
                results['baseline']['files_processed'] += 1
                
                detailed_results.append({
                    'approach': 'baseline',
                    'model': model,
                    'language': language,
                    'shot_type': shot_type,
                    'file': os.path.basename(file_path),
                    'total_samples': total_samples,
                    'valid_samples': valid_samples,
                    'success_rate': valid_samples / total_samples if total_samples > 0 else 0
                })
                
                print(f"  {file_path}: {valid_samples}/{total_samples} ({round(100*valid_samples/total_samples, 1)}%)")
            else:
                print(f"  {file_path}: No 'final_predicted_label' column found")
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    # Verarbeite die CoTR Multi-Prompt Dateien.
    cotr_mp_pattern = f"{base_results_dir}/classification_new/cotr/multi_prompt/**/*.csv"
    cotr_mp_files = glob.glob(cotr_mp_pattern, recursive=True)
    
    print("\nCOTR MULTI-PROMPT CLASSIFICATION FILES:")
    for file_path in cotr_mp_files:
        try:
            df = pd.read_csv(file_path)
            if 'predicted_label_eng_model' in df.columns:
                total_samples = len(df)
                valid_samples = df['predicted_label_eng_model'].isin(valid_labels).sum()
                
                # Extrahiere Details.
                model = "unknown"
                language = "unknown" 
                shot_type = "unknown"
                
                if "aya-23-8B" in file_path:
                    model = "aya-23-8B"
                elif "Qwen2.5-7B-Instruct" in file_path:
                    model = "Qwen2.5-7B-Instruct"
                
                if "/en/" in file_path or "_en" in file_path:
                    language = "en"
                elif "/ha/" in file_path or "_ha" in file_path:
                    language = "ha"
                elif "/sw/" in file_path or "_sw" in file_path:
                    language = "sw"
                
                if "/fs/" in file_path:
                    shot_type = "fs"
                elif "/zs/" in file_path:
                    shot_type = "zs"
                
                results['cotr_mp']['total_samples'] += total_samples
                results['cotr_mp']['valid_samples'] += valid_samples
                results['cotr_mp']['files_processed'] += 1
                
                detailed_results.append({
                    'approach': 'cotr_mp',
                    'model': model,
                    'language': language,
                    'shot_type': shot_type,
                    'file': os.path.basename(file_path),
                    'total_samples': total_samples,
                    'valid_samples': valid_samples,
                    'success_rate': valid_samples / total_samples if total_samples > 0 else 0
                })
                
                print(f"  {file_path}: {valid_samples}/{total_samples} ({round(100*valid_samples/total_samples, 1)}%)")
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    # Verarbeite die CoTR Single-Prompt Dateien.
    cotr_sp_pattern = f"{base_results_dir}/classification_new/cotr/single_prompt/**/*.csv"
    cotr_sp_files = glob.glob(cotr_sp_pattern, recursive=True)
    
    print("\nCOTR SINGLE-PROMPT CLASSIFICATION FILES:")
    for file_path in cotr_sp_files:
        try:
            df = pd.read_csv(file_path)
            # Prüfe, welche Spalte für das Label verwendet wird.
            label_col = None
            if 'label_lrl_predicted_final' in df.columns:
                label_col = 'label_lrl_predicted_final'
            elif 'predicted_label_lrl_model' in df.columns:
                label_col = 'predicted_label_lrl_model'
            elif 'lrl_label_model_final' in df.columns:
                label_col = 'lrl_label_model_final'
            elif 'predicted_label_eng_model' in df.columns:
                label_col = 'predicted_label_eng_model'
            elif 'label_en_predicted_intermediate' in df.columns:
                label_col = 'label_en_predicted_intermediate'
            
            if label_col:
                total_samples = len(df)
                # Zähle gültige Samples und schließe ungültige aus.
                valid_mask = (
                    df[label_col].isin(valid_labels) &
                    df[label_col].notna() &
                    (df[label_col] != "[Unknown Label]") &
                    (df[label_col] != "N/A") &
                    (df[label_col] != "")
                )
                valid_samples = valid_mask.sum()
                
                # Extrahiere Details.
                model = "unknown"
                language = "unknown"
                shot_type = "unknown"
                
                if "aya-23-8B" in file_path:
                    model = "aya-23-8B"
                elif "Qwen2.5-7B-Instruct" in file_path:
                    model = "Qwen2.5-7B-Instruct"
                
                if "/en/" in file_path or "_en" in file_path:
                    language = "en"
                elif "/ha/" in file_path or "_ha" in file_path:
                    language = "ha"
                elif "/sw/" in file_path or "_sw" in file_path:
                    language = "sw"
                
                if "/fs/" in file_path:
                    shot_type = "fs"
                elif "/zs/" in file_path:
                    shot_type = "zs"
                
                results['cotr_sp']['total_samples'] += total_samples
                results['cotr_sp']['valid_samples'] += valid_samples
                results['cotr_sp']['files_processed'] += 1
                
                detailed_results.append({
                    'approach': 'cotr_sp',
                    'model': model,
                    'language': language,
                    'shot_type': shot_type,
                    'file': os.path.basename(file_path),
                    'total_samples': total_samples,
                    'valid_samples': valid_samples,
                    'success_rate': valid_samples / total_samples if total_samples > 0 else 0
                })
                
                print(f"  {file_path}: {valid_samples}/{total_samples} ({round(100*valid_samples/total_samples, 1)}%) using column '{label_col}'")
            else:
                print(f"  {file_path}: No valid label column found")
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    # Gib eine Zusammenfassung aus.
    print("\n" + "="*80)
    print("CLASSIFICATION SAMPLE-LEVEL PARSING SUMMARY")
    print("="*80)
    
    for approach, data in results.items():
        if data['total_samples'] > 0:
            success_rate = data['valid_samples'] / data['total_samples']
            print(f"\n{approach.upper()}:")
            print(f"  Files processed: {data['files_processed']}")
            print(f"  Total samples: {data['total_samples']}")
            print(f"  Valid samples: {data['valid_samples']}")
            print(f"  Success rate: {round(success_rate * 100, 1)}%")
    
    # Detaillierte Aufschlüsselung nach Modell und Sprache.
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN")
    print("="*80)
    
    # Gruppiere die Ergebnisse, um sie leichter anzuzeigen.
    grouped = {}
    
    for result in detailed_results:
        approach = result['approach']
        model = result['model']
        language = result['language']

        if approach not in grouped:
            grouped[approach] = {}
        if model not in grouped[approach]:
            grouped[approach][model] = {}
        if language not in grouped[approach][model]:
            grouped[approach][model][language] = {'total': 0, 'valid': 0}
            
        grouped[approach][model][language]['total'] += result['total_samples']
        grouped[approach][model][language]['valid'] += result['valid_samples']
    
    for approach in ['baseline', 'cotr_mp', 'cotr_sp']:
        if approach in grouped:
            print(f"\n{approach.upper()}:")
            for model in grouped[approach]:
                print(f"  {model}:")
                for language in grouped[approach][model]:
                    data = grouped[approach][model][language]
                    if data['total'] > 0:
                        rate = data['valid'] / data['total']
                        print(f"    {language}: {data['valid']}/{data['total']} ({round(rate * 100, 1)}%)")

if __name__ == "__main__":
    analyze_classification_parsing() 