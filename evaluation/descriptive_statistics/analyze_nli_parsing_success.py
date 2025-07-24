import pandas as pd
import json
import os
import re
import numpy as np

def analyze_nli_parsing_success(comprehensive_metrics_path):
    """
    Analysiert die Erfolgsraten des NLI-Parsings durch Untersuchung der rohen Modellausgaben.
    """
    
    # Laedt die umfassenden Metriken
    with open(comprehensive_metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    nli_configs = metrics_data.get('nli', [])
    
    # Initialisiert Variablen, um die Ergebnisse zu verfolgen
    parsing_results = []
    total_samples = 0
    total_valid_labels = 0
    
    # Gueltige NLI-Labels (als Text oder Zahlen)
    valid_labels = {'0', '1', '2', 'entailment', 'neutral', 'contradiction'}
    
    # Verarbeitet jede NLI-Konfiguration
    for config in nli_configs:
        file_path = config['file_path']
        approach = config['approach']
        language = config['language']
        model = config['model']
        shot_type = config['shot_type']
        
        # Versucht, die Ergebnisdatei zu lesen
        full_path = os.path.join("/home/bbd6522/code/CoTR_Prompting_low_resource/results", file_path)
        
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue
            
        try:
            df = pd.read_csv(full_path)
            
            # Zaehlt die gueltigen Parsings fuer diese Konfiguration
            valid_count = 0
            total_count = len(df)
            
            # Sucht nach der Spalte mit den vorhergesagten Labels
            pred_col = None
            for col in df.columns:
                if 'predicted' in col.lower() or 'label' in col.lower():
                    pred_col = col
                    break
            
            if pred_col is None:
                print(f"No predicted label column found in {full_path}")
                continue
                
            # Zaehlt die gueltigen Labels
            for _, row in df.iterrows():
                pred_label = str(row[pred_col]).strip().lower()
                
                # Prueft, ob es ein gueltiges NLI-Label ist
                is_valid = False
                if pred_label in valid_labels:
                    is_valid = True
                elif pred_label in ['0', '1', '2']: # Einfacher Check fuer Zahlen
                    is_valid = True
                else:
                    # Prueft, ob das Label einen der Begriffe enthaelt
                    for label_part in ['entailment', 'neutral', 'contradiction']:
                        if label_part in pred_label:
                            is_valid = True
                            break
                
                if is_valid:
                    valid_count += 1
            
            parsing_success_rate = valid_count / total_count if total_count > 0 else 0
            
            parsing_results.append({
                'file_path': file_path,
                'approach': approach,
                'language': language,
                'model': model,
                'shot_type': shot_type,
                'total_samples': total_count,
                'valid_labels': valid_count,
                'parsing_success_rate': parsing_success_rate,
                'accuracy': config.get('accuracy', 0),
                'f1_score': config.get('f1_score', 0)
            })
            
            total_samples += total_count
            total_valid_labels += valid_count
            
        except Exception as e:
            print(f"Error processing {full_path}: {e}")
            continue
    
    # Berechnet die Gesamtstatistik
    overall_parsing_success = total_valid_labels / total_samples if total_samples > 0 else 0
    
    # Erstellt Zusammenfassungen nach verschiedenen Dimensionen
    results_df = pd.DataFrame(parsing_results)
    
    # Zusammenfassung nach Ansatz
    approach_summary = results_df.groupby('approach').agg({
        'parsing_success_rate': 'mean',
        'accuracy': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    })
    
    # Zusammenfassung nach Sprache
    language_summary = results_df.groupby('language').agg({
        'parsing_success_rate': 'mean',
        'accuracy': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    })
    
    # Zusammenfassung nach Modell
    model_summary = results_df.groupby('model').agg({
        'parsing_success_rate': 'mean',
        'accuracy': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    })
    
    # Zusammenfassung nach "shot type"
    shot_summary = results_df.groupby('shot_type').agg({
        'parsing_success_rate': 'mean',
        'accuracy': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    })
    
    return {
        'overall_parsing_success': overall_parsing_success,
        'total_samples': total_samples,
        'total_valid_labels': total_valid_labels,
        'detailed_results': parsing_results,
        'approach_summary': approach_summary,
        'language_summary': language_summary,
        'model_summary': model_summary,
        'shot_summary': shot_summary,
        'results_df': results_df
    }

def create_nli_comprehensive_analysis(comprehensive_metrics_path):
    """
    Erstellt eine umfassende NLI-Analyse, die den Parsing-Erfolg mit Leistungsmetriken kombiniert.
    """
    
    # Laedt die umfassenden Metriken
    with open(comprehensive_metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    nli_configs = metrics_data.get('nli', [])
    
    # Extrahiert Baseline- und CoTR-Ergebnisse
    baseline_results = []
    cotr_results = []
    
    for config in nli_configs:
        entry = {
            'approach': config['approach'],
            'language': config['language'],
            'model': config['model'],
            'shot_type': config['shot_type'],
            'accuracy': config.get('accuracy', 0),
            'f1_score': config.get('f1_score', 0),
            'sample_count': config.get('sample_count', 0),
            'file_path': config['file_path']
        }
        
        if config['approach'] == 'baseline':
            baseline_results.append(entry)
        else:
            cotr_results.append(entry)
    
    # Erstellt DataFrames
    baseline_df = pd.DataFrame(baseline_results)
    cotr_df = pd.DataFrame(cotr_results)
    
    # Berechnet Zusammenfassungen
    baseline_summary = {
        'total_configs': len(baseline_df),
        'avg_accuracy': baseline_df['accuracy'].mean(),
        'avg_f1': baseline_df['f1_score'].mean(),
        'best_accuracy': baseline_df['accuracy'].max(),
        'worst_accuracy': baseline_df['accuracy'].min(),
        'best_f1': baseline_df['f1_score'].max(),
        'worst_f1': baseline_df['f1_score'].min()
    }
    
    cotr_summary = {
        'total_configs': len(cotr_df),
        'avg_accuracy': cotr_df['accuracy'].mean(),
        'avg_f1': cotr_df['f1_score'].mean(),
        'best_accuracy': cotr_df['accuracy'].max(),
        'worst_accuracy': cotr_df['accuracy'].min(),
        'best_f1': cotr_df['f1_score'].max(),
        'worst_f1': cotr_df['f1_score'].min()
    }
    
    # Sprachspezifische Analyse
    languages = ['en', 'fr', 'sw', 'ur']
    language_analysis = {}
    
    for lang in languages:
        baseline_lang = baseline_df[baseline_df['language'] == lang]
        cotr_lang = cotr_df[cotr_df['language'] == lang]
        
        language_analysis[lang] = {
            'baseline_accuracy': baseline_lang['accuracy'].mean() if len(baseline_lang) > 0 else 0,
            'cotr_accuracy': cotr_lang['accuracy'].mean() if len(cotr_lang) > 0 else 0,
            'baseline_f1': baseline_lang['f1_score'].mean() if len(baseline_lang) > 0 else 0,
            'cotr_f1': cotr_lang['f1_score'].mean() if len(cotr_lang) > 0 else 0,
            'baseline_configs': len(baseline_lang),
            'cotr_configs': len(cotr_lang)
        }
    
    # Modellspezifische Analyse
    models = ['aya-23-8B', 'Qwen2.5-7B-Instruct']
    model_analysis = {}
    
    for model in models:
        baseline_model = baseline_df[baseline_df['model'] == model]
        cotr_model = cotr_df[cotr_df['model'] == model]
        
        model_analysis[model] = {
            'baseline_accuracy': baseline_model['accuracy'].mean() if len(baseline_model) > 0 else 0,
            'cotr_accuracy': cotr_model['accuracy'].mean() if len(cotr_model) > 0 else 0,
            'baseline_f1': baseline_model['f1_score'].mean() if len(baseline_model) > 0 else 0,
            'cotr_f1': cotr_model['f1_score'].mean() if len(cotr_model) > 0 else 0,
            'baseline_configs': len(baseline_model),
            'cotr_configs': len(cotr_model)
        }
    
    # "Shot type" Analyse
    shot_types = ['zeroshot', 'fewshot']
    shot_analysis = {}
    
    for shot in shot_types:
        baseline_shot = baseline_df[baseline_df['shot_type'] == shot]
        cotr_shot = cotr_df[cotr_df['shot_type'] == shot]
        
        shot_analysis[shot] = {
            'baseline_accuracy': baseline_shot['accuracy'].mean() if len(baseline_shot) > 0 else 0,
            'cotr_accuracy': cotr_shot['accuracy'].mean() if len(cotr_shot) > 0 else 0,
            'baseline_f1': baseline_shot['f1_score'].mean() if len(baseline_shot) > 0 else 0,
            'cotr_f1': cotr_shot['f1_score'].mean() if len(cotr_shot) > 0 else 0,
            'baseline_configs': len(baseline_shot),
            'cotr_configs': len(cotr_shot)
        }
    
    # Pipeline-Analyse für CoTR
    cotr_single = cotr_df[cotr_df['approach'] == 'cotr_single']
    cotr_multi = cotr_df[cotr_df['approach'] == 'cotr_multi']
    
    pipeline_analysis = {
        'single_prompt': {
            'accuracy': cotr_single['accuracy'].mean() if len(cotr_single) > 0 else 0,
            'f1_score': cotr_single['f1_score'].mean() if len(cotr_single) > 0 else 0,
            'configs': len(cotr_single)
        },
        'multi_prompt': {
            'accuracy': cotr_multi['accuracy'].mean() if len(cotr_multi) > 0 else 0,
            'f1_score': cotr_multi['f1_score'].mean() if len(cotr_multi) > 0 else 0,
            'configs': len(cotr_multi)
        }
    }
    
    return {
        'baseline_summary': baseline_summary,
        'cotr_summary': cotr_summary,
        'language_analysis': language_analysis,
        'model_analysis': model_analysis,
        'shot_analysis': shot_analysis,
        'pipeline_analysis': pipeline_analysis,
        'baseline_df': baseline_df,
        'cotr_df': cotr_df,
        'all_configs': nli_configs
    }

def main():
    """Hauptfunktion zum Ausfuehren der Analyse."""
    
    # Pfad zu den umfassenden Metriken
    comprehensive_metrics_path = "/home/bbd6522/code/CoTR_Prompting_low_resource/results/analysis/comprehensive_metrics.json"
    
    print("Analyzing NLI parsing success rates...")
    parsing_analysis = analyze_nli_parsing_success(comprehensive_metrics_path)
    
    print("\nCreating comprehensive NLI analysis...")
    comprehensive_analysis = create_nli_comprehensive_analysis(comprehensive_metrics_path)
    
    # Gibt die wichtigsten Ergebnisse aus
    print("\n" + "="*60)
    print("NLI COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nOVERALL PARSING SUCCESS: {round(parsing_analysis['overall_parsing_success'] * 100, 1)}%")
    print(f"TOTAL SAMPLES ANALYZED: {parsing_analysis['total_samples']}")
    print(f"TOTAL VALID LABELS: {parsing_analysis['total_valid_labels']}")
    
    print(f"\nBASELINE PERFORMANCE:")
    baseline_summary = comprehensive_analysis['baseline_summary']
    print(f"   Average Accuracy: {round(baseline_summary['avg_accuracy'] * 100, 1)}%")
    print(f"   Average F1: {round(baseline_summary['avg_f1'], 3)}")
    print(f"   Best Accuracy: {round(baseline_summary['best_accuracy'] * 100, 1)}%")
    print(f"   Worst Accuracy: {round(baseline_summary['worst_accuracy'] * 100, 1)}%")
    print(f"   Configurations: {baseline_summary['total_configs']}")
    
    print(f"\nCOTR PERFORMANCE:")
    cotr_summary = comprehensive_analysis['cotr_summary']
    print(f"   Average Accuracy: {round(cotr_summary['avg_accuracy'] * 100, 1)}%")
    print(f"   Average F1: {round(cotr_summary['avg_f1'], 3)}")
    print(f"   Best Accuracy: {round(cotr_summary['best_accuracy'] * 100, 1)}%")
    print(f"   Worst Accuracy: {round(cotr_summary['worst_accuracy'] * 100, 1)}%")
    print(f"   Configurations: {cotr_summary['total_configs']}")
    
    print(f"\nLANGUAGE PERFORMANCE:")
    for lang, stats in comprehensive_analysis['language_analysis'].items():
        print(f"   {lang.upper()}: Baseline {round(stats['baseline_accuracy'] * 100, 1)}% | CoTR {round(stats['cotr_accuracy'] * 100, 1)}%")
    
    print(f"\nMODEL PERFORMANCE:")
    for model, stats in comprehensive_analysis['model_analysis'].items():
        model_short = model.split('-')[0] if '-' in model else model
        print(f"   {model_short}: Baseline {round(stats['baseline_accuracy'] * 100, 1)}% | CoTR {round(stats['cotr_accuracy'] * 100, 1)}%")
    
    print(f"\nSHOT TYPE PERFORMANCE:")
    for shot, stats in comprehensive_analysis['shot_analysis'].items():
        print(f"   {shot.title()}: Baseline {round(stats['baseline_accuracy'] * 100, 1)}% | CoTR {round(stats['cotr_accuracy'] * 100, 1)}%")
    
    print(f"\nPIPELINE PERFORMANCE:")
    pipeline_stats = comprehensive_analysis['pipeline_analysis']
    print(f"   Single-prompt: {round(pipeline_stats['single_prompt']['accuracy'] * 100, 1)}%")
    print(f"   Multi-prompt: {round(pipeline_stats['multi_prompt']['accuracy'] * 100, 1)}%")
    
    # Speichert die detaillierten Ergebnisse
    print("\nSaving detailed results...")
    
    # Speichert die Analyse des Parsing-Erfolgs
    results_df = parsing_analysis['results_df']
    results_df.to_csv('nli_parsing_success_analysis.csv', index=False)
    
    # Speichert die umfassende Analyse
    with open('nli_comprehensive_analysis.json', 'w') as f:
        # Konvertiert DataFrames in Dictionaries fuer die JSON-Speicherung
        analysis_for_json = {
            'baseline_summary': comprehensive_analysis['baseline_summary'],
            'cotr_summary': comprehensive_analysis['cotr_summary'],
            'language_analysis': comprehensive_analysis['language_analysis'],
            'model_analysis': comprehensive_analysis['model_analysis'],
            'shot_analysis': comprehensive_analysis['shot_analysis'],
            'pipeline_analysis': comprehensive_analysis['pipeline_analysis'],
            'parsing_analysis': {
                'overall_parsing_success': parsing_analysis['overall_parsing_success'],
                'total_samples': parsing_analysis['total_samples'],
                'total_valid_labels': parsing_analysis['total_valid_labels']
            }
        }
        json.dump(analysis_for_json, f, indent=2)
    
    print("Analysis complete! Results saved to:")
    print("   - nli_parsing_success_analysis.csv")
    print("   - nli_comprehensive_analysis.json")
    
    return parsing_analysis, comprehensive_analysis

if __name__ == "__main__":
    parsing_analysis, comprehensive_analysis = main() 