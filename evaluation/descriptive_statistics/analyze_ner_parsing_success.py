#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse der Erfolgsrate beim Parsen von NER-Ergebnissen.
Dieses Skript untersucht NER-Ergebnisdateien, um die Erfolgsrate beim Parsen zu berechnen.
Es prüft, ob die Rohausgaben gültige Entitäts-Label (PER, ORG, LOC, DATE) enthalten.
"""

import pandas as pd
import numpy as np
import re
import json
import os
import glob
import ast

def load_comprehensive_metrics():
    """Laedt umfassende Metriken, um F1-Scores zu erhalten."""
    metrics_file = "/home/bbd6522/code/CoTR_Prompting_low_resource/results/analysis/comprehensive_metrics.json"
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # Ein Wörterbuch zum Nachschlagen von F1-Scores erstellen.
    f1_lookup = {}
    
    if 'ner' in data:
        for result in data['ner']:
            approach = result['approach']
            language = result['language']
            model = result['model']
            shot_type = result['shot_type']
            f1_score = result.get('f1_score', 0.0)
            
            # Erstellt verschiedene Schlüssel, um Variationen in den Namen zu behandeln.
            keys = [
                f"{approach}_{language}_{model}_{shot_type}",
                f"{approach}_{language}_{model.replace('/', '_').replace('-', '_')}_{shot_type}",
                f"{approach}_{language}_{model.replace('/', '_').replace('-', '_').replace('.', '_')}_{shot_type}"
            ]
            
            for key in keys:
                f1_lookup[key] = f1_score
    
    return f1_lookup

def create_lookup_key(approach, language, model, shot_type):
    """Erstellt mehrere mögliche Nachschlage-Schlüssel, um Variationen zu behandeln."""
    # Sprachcode-Variationen behandeln. 'swa' wird zu 'sw'.
    lang_map = {
        'swa': 'sw',
        'hau': 'ha',
        'sw': 'sw',
        'ha': 'ha'
    }
    
    # Holt den standardisierten Sprachcode.
    std_lang = lang_map.get(language, language)
    
    # Modellnamen-Variationen behandeln.
    model_variants = [
        model,
        model.replace('/', '_').replace('-', '_'),
        model.replace('/', '_').replace('-', '_').replace('.', '_')
    ]
    
    keys = []
    for model_variant in model_variants:
        for lang_variant in [language, std_lang]:
            key = f"{approach}_{lang_variant}_{model_variant}_{shot_type}"
            keys.append(key)
    
    return keys

def is_valid_ner_output(raw_output):
    """
    Prüft, ob die Rohausgabe gültige NER-Entitäts-Label im erwarteten Format enthält.
    Gültige Labels: PER, ORG, LOC, DATE
    Erwartetes Format: [LABEL: entity_text] oder ähnliche Variationen.
    """
    if not raw_output or pd.isna(raw_output):
        return False
    
    # Gültige Entitätstypen.
    valid_labels = ['PER', 'ORG', 'LOC', 'DATE']
    
    # Ein Muster, um das Format [LABEL: text] zu finden.
    pattern = r'\[(?:' + '|'.join(valid_labels) + r'):\s*[^\]]+\]'
    
    # Prüft, ob die Ausgabe mindestens ein gültiges Label enthält.
    matches = re.findall(pattern, raw_output, re.IGNORECASE)
    
    return len(matches) > 0

def extract_valid_entities_from_raw(raw_output):
    """Extrahiert gültige Entitäts-Label aus der Rohausgabe."""
    if not raw_output or pd.isna(raw_output):
        return []
    
    valid_labels = ['PER', 'ORG', 'LOC', 'DATE']
    pattern = r'\[(' + '|'.join(valid_labels) + r'):\s*[^\]]+\]'
    
    matches = re.findall(pattern, raw_output, re.IGNORECASE)
    return matches

def count_predicted_entities(predicted_entities_str):
    """Zählt die Anzahl der vorhergesagten Entitäten aus der Spalte 'predicted_entities'."""
    if not predicted_entities_str or pd.isna(predicted_entities_str):
        return 0
    
    try:
        # Wandelt den String sicher in eine Python-Liste um.
        entities = ast.literal_eval(predicted_entities_str)
        if isinstance(entities, list):
            return len(entities)
        return 0
    except (ValueError, SyntaxError):
        return 0

def analyze_ner_file(file_path, f1_lookup):
    """Analysiert eine einzelne NER-Ergebnisdatei."""
    try:
        df = pd.read_csv(file_path)
        
        # Extrahiert Metadaten aus dem Dateinamen.
        filename = os.path.basename(file_path)
        parts = filename.replace('.csv', '').split('_')
        
        # Analysiert den Dateinamen, z.B. results_baseline_fs_ner_swa_aya-23-8B.csv
        approach = 'baseline'
        if 'cotr' in filename:
            if 'mp' in filename:
                approach = 'cotr_mp'
            elif 'sp' in filename:
                approach = 'cotr_sp'
            else:
                approach = 'cotr'
        
        # Extrahiert die Sprache (sollte nach 'ner' kommen).
        language = 'unknown'
        try:
            ner_idx = parts.index('ner')
            if ner_idx + 1 < len(parts):
                language = parts[ner_idx + 1]
        except ValueError:
            pass
        
        # Extrahiert den Modellnamen (alles nach der Sprache).
        model = 'unknown'
        try:
            ner_idx = parts.index('ner')
            if ner_idx + 2 < len(parts):
                model = '_'.join(parts[ner_idx + 2:])
        except ValueError:
            pass
        
        # Extrahiert den "shot type".
        shot_type = 'unknown'
        if 'fs' in parts:
            shot_type = 'fewshot'
        elif 'zs' in parts:
            shot_type = 'zeroshot'
        
        # Analysiert den Parsing-Erfolg.
        total_samples = len(df)
        valid_samples = 0
        total_entities_extracted = 0
        
        if 'raw_output' in df.columns:
            for _, row in df.iterrows():
                if is_valid_ner_output(row['raw_output']):
                    valid_samples += 1
                
                # Zählt die Entitäten.
                if 'predicted_entities' in row:
                    total_entities_extracted += count_predicted_entities(str(row['predicted_entities']))
        
        parsing_success_rate = (valid_samples / total_samples) * 100 if total_samples > 0 else 0
        
        # Holt den F1-Score, indem mehrere mögliche Schlüssel probiert werden.
        f1_score = 0.0
        lookup_keys = create_lookup_key(approach, language, model, shot_type)
        matched_key = None
        
        for key in lookup_keys:
            if key in f1_lookup:
                f1_score = f1_lookup[key]
                matched_key = key
                break
        
        return {
            'file_path': file_path,
            'approach': approach,
            'language': language,
            'model': model,
            'shot_type': shot_type,
            'total_samples': total_samples,
            'valid_samples': valid_samples,
            'parsing_success_rate': parsing_success_rate,
            'f1_score': f1_score,
            'total_entities_extracted': total_entities_extracted,
            'lookup_key': matched_key if matched_key else lookup_keys[0]
        }
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def main():
    """Hauptfunktion zur Analyse aller NER-Ergebnisdateien."""
    
    # Laedt die F1-Scores.
    f1_lookup = load_comprehensive_metrics()
    print(f"Loaded {len(f1_lookup)} F1 scores from comprehensive metrics")
    
    # Findet alle NER-Ergebnisdateien.
    base_dir = "/home/bbd6522/code/CoTR_Prompting_low_resource/results"
    ner_files = glob.glob(os.path.join(base_dir, "**", "*ner*.csv"), recursive=True)
    
    print(f"Found {len(ner_files)} NER result files")
    
    all_results = []
    
    for file_path in ner_files:
        result = analyze_ner_file(file_path, f1_lookup)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("No valid results found!")
        return
    
    # Erstellt einen DataFrame für die einfache Analyse.
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("NER PARSING SUCCESS RATE ANALYSIS")
    print("="*80)
    
    # Gesamtstatistik.
    total_samples = results_df['total_samples'].sum()
    total_valid = results_df['valid_samples'].sum()
    overall_parsing_success = (total_valid / total_samples) * 100 if total_samples > 0 else 0
    overall_f1 = results_df['f1_score'].mean()
    
    print(f"\nOVERALL STATISTICS")
    print(f"  Total samples: {total_samples}")
    print(f"  Valid samples: {total_valid}")
    print(f"  Overall parsing success rate: {round(overall_parsing_success, 1)}%")
    print(f"  Overall mean F1 score: {round(overall_f1, 4)}")
    
    # Nach Ansatz gruppiert.
    print(f"\nBY APPROACH")
    approach_stats = results_df.groupby('approach').agg({
        'parsing_success_rate': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    })
    
    for approach, stats in approach_stats.iterrows():
        print(f"  {approach}: {round(stats['parsing_success_rate'], 1)}% parsing success, {round(stats['f1_score'], 4)} F1, {int(stats['total_samples'])} samples")
    
    # Nach Modell gruppiert.
    print(f"\nBY MODEL")
    model_stats = results_df.groupby('model').agg({
        'parsing_success_rate': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    })
    
    for model, stats in model_stats.iterrows():
        print(f"  {model}: {round(stats['parsing_success_rate'], 1)}% parsing success, {round(stats['f1_score'], 4)} F1, {int(stats['total_samples'])} samples")
    
    # Nach Sprache gruppiert.
    print(f"\nBY LANGUAGE")
    lang_stats = results_df.groupby('language').agg({
        'parsing_success_rate': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    })
    
    for language, stats in lang_stats.iterrows():
        print(f"  {language}: {round(stats['parsing_success_rate'], 1)}% parsing success, {round(stats['f1_score'], 4)} F1, {int(stats['total_samples'])} samples")
    
    # Nach "shot type" gruppiert.
    print(f"\nBY SHOT TYPE")
    shot_stats = results_df.groupby('shot_type').agg({
        'parsing_success_rate': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    })
    
    for shot_type, stats in shot_stats.iterrows():
        print(f"  {shot_type}: {round(stats['parsing_success_rate'], 1)}% parsing success, {round(stats['f1_score'], 4)} F1, {int(stats['total_samples'])} samples")
    
    # Speichert die detaillierten Ergebnisse.
    output_file = "/home/bbd6522/code/CoTR_Prompting_low_resource/ner_parsing_success_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Erstellt eine Zusammenfassung für das Kapitel.
    print(f"\nSUMMARY FOR CHAPTER UPDATE")
    print(f"Overall parsing success rate: {round(overall_parsing_success, 1)}%")
    print(f"Overall mean F1 score: {round(overall_f1, 4)}")
    
    # Baseline vs. CoTR Vergleich.
    baseline_mask = results_df['approach'] == 'baseline'
    cotr_mask = results_df['approach'].isin(['cotr_mp', 'cotr_sp'])
    
    if baseline_mask.any():
        baseline_parsing = results_df[baseline_mask]['parsing_success_rate'].mean()
        baseline_f1 = results_df[baseline_mask]['f1_score'].mean()
        print(f"Baseline: {round(baseline_parsing, 1)}% parsing success, {round(baseline_f1, 4)} F1")
    
    if cotr_mask.any():
        cotr_parsing = results_df[cotr_mask]['parsing_success_rate'].mean()
        cotr_f1 = results_df[cotr_mask]['f1_score'].mean()
        print(f"CoTR: {round(cotr_parsing, 1)}% parsing success, {round(cotr_f1, 4)} F1")
    
    # CoTR-Pipeline-Vergleich.
    cotr_mp_mask = results_df['approach'] == 'cotr_mp'
    cotr_sp_mask = results_df['approach'] == 'cotr_sp'
    
    if cotr_mp_mask.any():
        mp_parsing = results_df[cotr_mp_mask]['parsing_success_rate'].mean()
        mp_f1 = results_df[cotr_mp_mask]['f1_score'].mean()
        print(f"CoTR Multi-Prompt: {round(mp_parsing, 1)}% parsing success, {round(mp_f1, 4)} F1")
    
    if cotr_sp_mask.any():
        sp_parsing = results_df[cotr_sp_mask]['parsing_success_rate'].mean()
        sp_f1 = results_df[cotr_sp_mask]['f1_score'].mean()
        print(f"CoTR Single-Prompt: {round(sp_parsing, 1)}% parsing success, {round(sp_f1, 4)} F1")

if __name__ == "__main__":
    main() 