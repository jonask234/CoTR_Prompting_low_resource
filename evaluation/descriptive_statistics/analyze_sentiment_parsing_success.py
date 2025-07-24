#!/usr/bin/env python3
"""
Analyse der Erfolgsrate des Sentiment-Analyse-Parsings.

Dieses Skript analysiert die Erfolgsraten fuer Sentiment-Analyse-Experimente,
indem es ueberprueft, wie oft Modelle gueltige Sentiment-Labels (positive, negative, neutral)
im Vergleich zu un Gueltigen/nicht parsebaren Ausgaben produzieren.
"""

import os
import pandas as pd
import numpy as np
import re
import json

# Gueltige Sentiment-Labels (Gross- und Kleinschreibung wird nicht beachtet)
VALID_SENTIMENT_LABELS = ['positive', 'negative', 'neutral']

def is_valid_sentiment_label(predicted_label):
    """Prueft, ob ein vorhergesagtes Label ein gueltiges Sentiment-Label ist."""
    if pd.isna(predicted_label):
        return False
    
    label_clean = str(predicted_label).strip().lower()
    return label_clean in VALID_SENTIMENT_LABELS

def analyze_sentiment_file_parsing(file_path):
    """Analysiert den Parsing-Erfolg fuer eine einzelne Sentiment-Ergebnisdatei."""
    try:
        df = pd.read_csv(file_path)
        
        if 'predicted_label' not in df.columns:
            return None
        
        total_samples = len(df)
        
        # Zaehlt die gueltigen Labels mit einer einfachen Schleife
        valid_parses = 0
        for label in df['predicted_label']:
            if is_valid_sentiment_label(label):
                valid_parses += 1
        
        parsing_success_rate = valid_parses / total_samples if total_samples > 0 else 0
        parsing_failure_rate = 1 - parsing_success_rate
        
        return {
            'file_path': file_path,
            'total_samples': total_samples,
            'valid_parses': valid_parses,
            'invalid_parses': total_samples - valid_parses,
            'parsing_success_rate': parsing_success_rate,
            'parsing_failure_rate': parsing_failure_rate
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_config_from_path(file_path):
    """Extrahiert Konfigurationsinformationen aus dem Dateipfad."""
    # Initialisiert mit Standardwerten
    config = {
        'approach': 'unknown',
        'language': 'unknown', 
        'model': 'unknown',
        'shot_type': 'unknown',
        'pipeline': 'unknown'
    }
    
    filename = os.path.basename(file_path)
    
    # Bestimmt den Ansatz
    if 'baseline' in file_path:
        config['approach'] = 'baseline'
    elif 'multi_prompt' in file_path:
        config['approach'] = 'cotr_mp'
    elif 'single_prompt' in file_path:
        config['approach'] = 'cotr_sp'
    
    # Extrahiert die Sprache
    for lang in ['ha', 'sw', 'pt']:
        if f'_{lang}_' in filename or f'/{lang}/' in file_path:
            config['language'] = lang
            break
    
    # Extrahiert das Modell
    if 'Qwen2.5-7B-Instruct' in file_path:
        config['model'] = 'Qwen2.5-7B-Instruct'
    elif 'aya-23-8B' in file_path:
        config['model'] = 'aya-23-8B'
    
    # Extrahiert den "shot type"
    if 'few_shot' in filename or '_fs_' in filename:
        config['shot_type'] = 'fewshot'
    elif 'zero_shot' in filename or '_zs_' in filename:
        config['shot_type'] = 'zeroshot'
    
    # Pipeline fuer CoTR
    if config['approach'] == 'cotr_mp':
        config['pipeline'] = 'multi_prompt'
    elif config['approach'] == 'cotr_sp':
        config['pipeline'] = 'single_prompt'
    else:
        config['pipeline'] = 'baseline'
    
    return config

def analyze_sentiment_parsing_success():
    """Hauptfunktion zur Analyse der Erfolgsraten des Sentiment-Parsings."""
    
    print("=== SENTIMENT PARSING SUCCESS RATE ANALYSIS ===\n")
    
    # Basispfad fuer Sentiment-Ergebnisse
    sentiment_base_path = '/home/bbd6522/code/CoTR_Prompting_low_resource/results/sentiment_new'
    
    if not os.path.exists(sentiment_base_path):
        print(f"Sentiment results path not found: {sentiment_base_path}")
        return None
    
    # Findet alle Ergebnisdateien
    result_files = []
    for root, dirs, files in os.walk(sentiment_base_path):
        for file in files:
            if file.endswith('.csv') and 'results_' in file:
                result_files.append(os.path.join(root, file))
    
    print(f"Found {len(result_files)} sentiment result files")
    
    if not result_files:
        print("No sentiment result files found")
        return None
    
    # Analysiert jede Datei
    parsing_results = []
    total_samples = 0
    total_valid_parses = 0
    
    for file_path in result_files:
        file_analysis = analyze_sentiment_file_parsing(file_path)
        
        if file_analysis:
            config = extract_config_from_path(file_path)
            file_analysis.update(config)
            parsing_results.append(file_analysis)
            
            total_samples += file_analysis['total_samples']
            total_valid_parses += file_analysis['valid_parses']
            
            success_rate_percent = round(file_analysis['parsing_success_rate'] * 100, 1)
            print(f"{config['approach']} | {config['model']} | {config['language']} | {config['shot_type']}: "
                  f"{success_rate_percent}% success "
                  f"({file_analysis['valid_parses']}/{file_analysis['total_samples']})")
    
    if not parsing_results:
        print("No valid parsing results found")
        return None
    
    # Erstellt einen DataFrame fuer die Analyse
    parsing_df = pd.DataFrame(parsing_results)
    
    # Gesamtstatistik
    overall_success_rate = total_valid_parses / total_samples if total_samples > 0 else 0
    
    print(f"\nOVERALL PARSING STATISTICS:")
    print(f"   Total samples: {total_samples}")
    print(f"   Valid parses: {total_valid_parses}")
    print(f"   Success rate: {round(overall_success_rate * 100, 1)}%")
    print(f"   Failure rate: {round((1 - overall_success_rate) * 100, 1)}%")
    
    # Analyse nach Ansatz
    print(f"\nPARSING SUCCESS BY APPROACH:")
    approach_stats = parsing_df.groupby('approach').agg({
        'total_samples': 'sum',
        'valid_parses': 'sum'
    })
    
    for approach, stats in approach_stats.iterrows():
        success_rate = stats['valid_parses'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
        print(f"   {approach}:")
        print(f"      Samples: {int(stats['total_samples'])}")
        print(f"      Success rate: {round(success_rate * 100, 1)}%")
        print(f"      Failure rate: {round((1 - success_rate) * 100, 1)}%")
    
    # Analyse nach Modell
    print(f"\nPARSING SUCCESS BY MODEL:")
    model_stats = parsing_df.groupby('model').agg(parsing_success_rate=('parsing_success_rate', 'mean'))
    
    for model, stats in model_stats.iterrows():
        print(f"   {model}: {round(stats['parsing_success_rate'] * 100, 1)}% success, {round((1 - stats['parsing_success_rate']) * 100, 1)}% failure")
    
    # Analyse nach Sprache
    print(f"\nPARSING SUCCESS BY LANGUAGE:")
    lang_stats = parsing_df.groupby('language').agg(parsing_success_rate=('parsing_success_rate', 'mean'))
    
    for language, stats in lang_stats.iterrows():
        print(f"   {language}: {round(stats['parsing_success_rate'] * 100, 1)}% success, {round((1 - stats['parsing_success_rate']) * 100, 1)}% failure")
    
    # Findet die besten und schlechtesten Konfigurationen
    best_config = parsing_df.loc[parsing_df['parsing_success_rate'].idxmax()]
    worst_config = parsing_df.loc[parsing_df['parsing_success_rate'].idxmin()]
    
    print(f"\nBEST PARSING CONFIGURATION:")
    print(f"   {best_config['approach']} | {best_config['model']} | {best_config['language']} | {best_config['shot_type']}")
    print(f"   Success: {round(best_config['parsing_success_rate'] * 100, 1)}% ({best_config['valid_parses']}/{best_config['total_samples']})")
    
    print(f"\nWORST PARSING CONFIGURATION:")
    print(f"   {worst_config['approach']} | {worst_config['model']} | {worst_config['language']} | {worst_config['shot_type']}")
    print(f"   Success: {round(worst_config['parsing_success_rate'] * 100, 1)}% ({worst_config['valid_parses']}/{worst_config['total_samples']})")
    
    # CoTR-Pipeline-Vergleich
    cotr_data = parsing_df[parsing_df['approach'].isin(['cotr_mp', 'cotr_sp'])]
    if len(cotr_data) > 0:
        print(f"\nCOTR PIPELINE COMPARISON:")
        pipeline_stats = cotr_data.groupby('approach').agg(parsing_success_rate=('parsing_success_rate', 'mean'))
        
        for pipeline, stats in pipeline_stats.iterrows():
            pipeline_name = "Multi-prompt" if pipeline == "cotr_mp" else "Single-prompt"
            print(f"   {pipeline_name}: {round(stats['parsing_success_rate'] * 100, 1)}% success, {round((1 - stats['parsing_success_rate']) * 100, 1)}% failure")
    
    return parsing_df

def save_parsing_analysis(parsing_df, output_file='sentiment_parsing_analysis.csv'):
    """Speichert die Ergebnisse der Parsing-Analyse in einer CSV-Datei."""
    if parsing_df is not None:
        parsing_df.to_csv(output_file, index=False)
        print(f"\nParsing analysis saved to: {output_file}")
        return output_file
    return None

def generate_parsing_summary(parsing_df):
    """Erstellt einen Zusammenfassungsbericht der Parsing-Analyse."""
    if parsing_df is None:
        return "No parsing data available"
    
    summary = []
    summary.append("=== SENTIMENT PARSING SUCCESS SUMMARY ===\n")
    
    # Gesamtstatistik
    total_configs = len(parsing_df)
    avg_success = parsing_df['parsing_success_rate'].mean()
    
    summary.append(f"Total configurations analyzed: {total_configs}")
    summary.append(f"Average parsing success rate: {round(avg_success * 100, 1)}%")
    summary.append(f"Average parsing failure rate: {round((1 - avg_success) * 100, 1)}%")
    
    # Vergleich nach Ansatz
    summary.append(f"\nParsing success by approach:")
    approach_stats = parsing_df.groupby('approach')['parsing_success_rate'].mean().sort_values(ascending=False)
    for approach, success_rate in approach_stats.items():
        summary.append(f"  {approach}: {round(success_rate * 100, 1)}%")
    
    # Vergleich nach Modell
    summary.append(f"\nParsing success by model:")
    model_stats = parsing_df.groupby('model')['parsing_success_rate'].mean().sort_values(ascending=False)
    for model, success_rate in model_stats.items():
        summary.append(f"  {model}: {round(success_rate * 100, 1)}%")
    
    return "\n".join(summary)

if __name__ == "__main__":
    # Fuehrt die Parsing-Analyse aus
    parsing_df = analyze_sentiment_parsing_success()
    
    if parsing_df is not None:
        # Speichert die Ergebnisse
        output_file = save_parsing_analysis(parsing_df)
        
        # Erstellt und druckt die Zusammenfassung
        summary = generate_parsing_summary(parsing_df)
        print(f"\n{summary}")
        
        # Speichert die Zusammenfassung in einer Datei
        with open('sentiment_parsing_summary.txt', 'w') as f:
            f.write(summary)
        print(f"\nSummary saved to: sentiment_parsing_summary.txt")
    
    print(f"\nSentiment parsing analysis complete!") 