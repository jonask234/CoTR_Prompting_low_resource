#!/usr/bin/env python3
"""
Simplified Statistical Analysis for CoTR Research Questions
==========================================================

This analysis focuses on three core hypotheses:
H1: Does CoTR significantly improve performance vs baseline? (paired t-tests)
H2: Relationship between translation quality and task performance (Pearson correlation)
H3: Task-specific effectiveness of CoTR (comparative analysis across tasks)

Fixes language code normalization issues (hau/ha, swa/sw mapping).
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_rel, pearsonr
import os

# Konfigurationskonstanten
ALPHA = 0.05
COMPREHENSIVE_METRICS_FILE = "/home/bbd6522/code/CoTR_Prompting_low_resource/results/analysis/comprehensive_metrics.json"

def normalize_language_code(lang_code):
    # Normalisiert Sprachcodes für eine korrekte Zuordnung
    normalization_map = {
        'hau': 'ha',  # Hausa: MasakhaNER → ISO  
        'swa': 'sw',  # Swahili: MasakhaNER → ISO
        'ha': 'ha',   # Already ISO
        'sw': 'sw',   # Already ISO
        'en': 'en',   # English
        'fi': 'fi',   # Finnish
        'fr': 'fr',   # French
        'ur': 'ur',   # Urdu
        'pt': 'pt',   # Portuguese
    }
    return normalization_map.get(lang_code.lower(), lang_code.lower())

def load_and_prepare_data():
    # Lädt und bereitet die Daten mit korrekter Normalisierung der Sprachcodes vor
    
    print("LOADING DATA FOR SIMPLIFIED STATISTICAL ANALYSIS")
    print("=" * 70)
    
    with open(COMPREHENSIVE_METRICS_FILE, 'r') as f:
        data = json.load(f)
    
    all_rows = []
    
    for task, configs in data.items():
        for config in configs:
            # Normalisiert Ansatznamen
            approach = config['approach']
            if approach in ['cotr_sp', 'cotr_single']:
                approach = 'cotr'
                pipeline = 'single_prompt'
            elif approach in ['cotr_mp', 'cotr_multi']:
                approach = 'cotr'
                pipeline = 'multi_prompt'
            else:
                approach = 'baseline'
                pipeline = 'baseline'
            
            # Normalisiert Schusstypen
            shot_type = config['shot_type']
            if shot_type in ['zeroshot', 'zero-shot', 'zs']:
                shot_type = 'zero_shot'
            elif shot_type in ['fewshot', 'few-shot', 'fs']:
                shot_type = 'few_shot'
            
            # KRITISCHE FIX: Normalisiert Sprachcodes
            language = normalize_language_code(config['language'])
            
            # Erstellt Zeile
            row = {
                'task': task,
                'approach': approach,
                'pipeline': pipeline,
                'model': config['model'],
                'language': language,  # Jetzt normalisiert
                'shot_type': shot_type,
                'sample_count': config.get('sample_count', 0)
            }
            
            # Fügt verfügbare Metriken hinzu
            if 'accuracy' in config:
                row['accuracy'] = config['accuracy']
            if 'f1_score' in config:
                row['f1_score'] = config['f1_score']
                
            all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    
    print(f" Loaded {len(df)} configurations across {df['task'].nunique()} tasks")
    print(f"   Baseline: {len(df[df['approach'] == 'baseline'])}")
    print(f"   CoTR: {len(df[df['approach'] == 'cotr'])}")
    print(f"   Languages: {sorted(df['language'].unique())}")
    
    return df

def create_matched_pairs(df):
    # Erstellt passende Paare für gepaarte t-Tests
    
    print(f"\nCREATING MATCHED PAIRS WITH NORMALIZED LANGUAGE CODES")
    
    # Trennt Baseline und CoTR
    baseline_df = df[df['approach'] == 'baseline'].copy()
    cotr_df = df[df['approach'] == 'cotr'].copy()
    
    matched_pairs = []
    
    for _, cotr_row in cotr_df.iterrows():
        # Findet passende Baseline: gleiches Modell, Sprache, Schusstyp, Aufgabe
        matching_baseline = baseline_df[
            (baseline_df['model'] == cotr_row['model']) &
            (baseline_df['language'] == cotr_row['language']) &  # Jetzt korrekt normalisiert
            (baseline_df['shot_type'] == cotr_row['shot_type']) &
            (baseline_df['task'] == cotr_row['task'])
        ]
        
        if len(matching_baseline) == 1:
            baseline_row = matching_baseline.iloc[0]
            
            pair = {
                'task': cotr_row['task'],
                'model': cotr_row['model'],
                'language': cotr_row['language'],
                'shot_type': cotr_row['shot_type'],
                'pipeline': cotr_row['pipeline']
            }
            
            # Fügt Metriken hinzu
            if 'accuracy' in baseline_row and 'accuracy' in cotr_row and pd.notna(baseline_row['accuracy']) and pd.notna(cotr_row['accuracy']):
                pair['baseline_accuracy'] = baseline_row['accuracy']
                pair['cotr_accuracy'] = cotr_row['accuracy']
                pair['diff_accuracy'] = cotr_row['accuracy'] - baseline_row['accuracy']
            
            if 'f1_score' in baseline_row and 'f1_score' in cotr_row and pd.notna(baseline_row['f1_score']) and pd.notna(cotr_row['f1_score']):
                pair['baseline_f1'] = baseline_row['f1_score']
                pair['cotr_f1'] = cotr_row['f1_score']
                pair['diff_f1'] = cotr_row['f1_score'] - baseline_row['f1_score']
                
            matched_pairs.append(pair)
    
    pairs_df = pd.DataFrame(matched_pairs)
    
    print(f" Created {len(pairs_df)} matched pairs")
    print(f"   Tasks: {sorted(pairs_df['task'].unique())}")
    print(f"   Task counts: {pairs_df['task'].value_counts().to_dict()}")
    
    return pairs_df

def analyze_h1_cotr_vs_baseline(pairs_df):
    """
    H1: Verbessert CoTR die Leistung im Vergleich zur Baseline signifikant?
    Verwendet gepaarte t-Tests für passende Paare, wie beschrieben.
    """
    
    print(f"\nH1: CoTR vs Baseline Performance (Paired T-Tests)")
    print("=" * 60)
    
    results = {'hypothesis': 'H1_CoTR_vs_Baseline'}
    
    # Gesamtanalyse über alle Aufgaben hinweg
    for metric in ['accuracy', 'f1']:
        diff_col = f'diff_{metric}'
        baseline_col = f'baseline_{metric}'
        cotr_col = f'cotr_{metric}'
        
        if diff_col in pairs_df.columns:
            valid_pairs = pairs_df.dropna(subset=[diff_col])
            
            if len(valid_pairs) >= 3:
                baseline_scores = valid_pairs[baseline_col].values
                cotr_scores = valid_pairs[cotr_col].values
                differences = valid_pairs[diff_col].values
                
                # Gepaarter t-Test
                t_stat, p_value = ttest_rel(cotr_scores, baseline_scores)
                
                # Grundlegende Statistiken
                mean_baseline = np.mean(baseline_scores)
                mean_cotr = np.mean(cotr_scores)
                mean_improvement = np.mean(differences)
                
                result = {
                    'n_pairs': len(valid_pairs),
                    'mean_baseline': mean_baseline,
                    'mean_cotr': mean_cotr,
                    'mean_improvement': mean_improvement,
                    'relative_improvement_pct': (mean_improvement / mean_baseline * 100) if mean_baseline != 0 else 0,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < ALPHA,
                    'effect_direction': 'positive' if mean_improvement > 0 else 'negative'
                }
                
                results[f'overall_{metric}'] = result
                
                print(f"\n{metric.upper()} - Overall Results:")
                print(f"  • Pairs analyzed: {len(valid_pairs)}")
                print(f"  • Baseline mean: {mean_baseline:.4f}")
                print(f"  • CoTR mean: {mean_cotr:.4f}")
                print(f"  • Mean improvement: {mean_improvement:+.4f} ({result['relative_improvement_pct']:+.1f}%)")
                print(f"  • t-statistic: {t_stat:.4f}")
                print(f"  • p-value: {p_value:.4f}")
                print(f"  • Significant: {'YES' if result['significant'] else 'NO'}")
    
    return results

def analyze_h2_translation_quality(pairs_df):
    """
    H2: Beziehung zwischen Übersetzungsqualität und Aufgabenleistung
    Verwendet Pearson-Korrelationsanalyse, wie beschrieben.
    Hinweis: Dies erfordert BLEU-Scores, die in den aktuellen Daten möglicherweise nicht verfügbar sind.
    """
    
    print(f"\nH2: Translation Quality Correlation Analysis")
    print("=" * 60)
    
    results = {'hypothesis': 'H2_Translation_Quality'}
    
    # Hinweis: BLEU-Scores müssten zu den Daten hinzugefügt werden
    print("Translation quality analysis requires BLEU scores from translation steps.")
    print("    This analysis would correlate BLEU scores with task performance metrics.")
    print("    Implementation placeholder - requires separate translation evaluation.")
    
    results['status'] = 'requires_bleu_scores'
    results['note'] = 'Translation quality metrics (BLEU) need to be computed separately'
    
    return results

def analyze_h3_task_effectiveness(pairs_df):
    """
    H3: Aufgabenspezifische Wirksamkeit von CoTR
    Vergleichende Analyse der H1-Ergebnisse über verschiedene Aufgaben hinweg.
    """
    
    print(f"\nH3: Task-Specific Effectiveness Analysis")
    print("=" * 60)
    
    results = {'hypothesis': 'H3_Task_Effectiveness'}
    
    # Pro-Aufgaben-Analyse
    task_results = {}
    
    for task in pairs_df['task'].unique():
        task_data = pairs_df[pairs_df['task'] == task]
        task_results[task] = {}
        
        print(f"\n{task.upper()} Task:")
        
        for metric in ['accuracy', 'f1']:
            diff_col = f'diff_{metric}'
            baseline_col = f'baseline_{metric}'
            cotr_col = f'cotr_{metric}'
            
            if diff_col in task_data.columns:
                valid_pairs = task_data.dropna(subset=[diff_col])
                
                if len(valid_pairs) >= 3:
                    baseline_scores = valid_pairs[baseline_col].values
                    cotr_scores = valid_pairs[cotr_col].values
                    differences = valid_pairs[diff_col].values
                    
                    # Paired t-test per task
                    t_stat, p_value = ttest_rel(cotr_scores, baseline_scores)
                    
                    # Grundlegende Statistiken
                    mean_baseline = np.mean(baseline_scores)
                    mean_cotr = np.mean(cotr_scores)
                    mean_improvement = np.mean(differences)
                    
                    task_result = {
                        'n_pairs': len(valid_pairs),
                        'mean_improvement': mean_improvement,
                        'relative_improvement_pct': (mean_improvement / mean_baseline * 100) if mean_baseline != 0 else 0,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < ALPHA
                    }
                    
                    task_results[task][metric] = task_result
                    
                    print(f"  • {metric}: {mean_improvement:+.4f} ({task_result['relative_improvement_pct']:+.1f}%), p={p_value:.4f} {'YES' if task_result['significant'] else 'NO'}")
    
    results['task_specific'] = task_results
    
    # Zusammenfassung der aufgabenspezifischen Wirksamkeit
    print(f"\nTASK EFFECTIVENESS SUMMARY:")
    
    for task in task_results:
        if 'f1' in task_results[task]:
            result = task_results[task]['f1']
            effectiveness = "High" if result['relative_improvement_pct'] > 10 else "Medium" if result['relative_improvement_pct'] > 0 else "Low/Negative"
            print(f"  • {task.upper()}: {effectiveness} effectiveness ({result['relative_improvement_pct']:+.1f}% F1)")
    
    return results

def analyze_pipeline_effects(pairs_df):
    """
    Zusätzliche Analyse: Wirksamkeit der Single-Prompt- vs. Multi-Prompt-Pipeline
    """
    
    print(f"\nPIPELINE ANALYSIS: Single vs Multi-prompt")
    print("=" * 60)
    
    results = {'analysis': 'Pipeline_Effects'}
    
    # Trennt nach Pipeline-Typ
    single_prompt = pairs_df[pairs_df['pipeline'] == 'single_prompt']
    multi_prompt = pairs_df[pairs_df['pipeline'] == 'multi_prompt']
    
    for pipeline_type, data in [('single_prompt', single_prompt), ('multi_prompt', multi_prompt)]:
        if len(data) > 0:
            results[pipeline_type] = {}
            print(f"\n{pipeline_type.replace('_', '-').upper()} Pipeline:")
            
            for metric in ['accuracy', 'f1']:
                diff_col = f'diff_{metric}'
                baseline_col = f'baseline_{metric}'
                cotr_col = f'cotr_{metric}'
                
                if diff_col in data.columns:
                    valid_pairs = data.dropna(subset=[diff_col])
                    
                    if len(valid_pairs) >= 3:
                        baseline_scores = valid_pairs[baseline_col].values
                        cotr_scores = valid_pairs[cotr_col].values
                        differences = valid_pairs[diff_col].values
                        
                        # Paired t-test
                        t_stat, p_value = ttest_rel(cotr_scores, baseline_scores)
                        
                        mean_improvement = np.mean(differences)
                        mean_baseline = np.mean(baseline_scores)
                        
                        result = {
                            'n_pairs': len(valid_pairs),
                            'mean_improvement': mean_improvement,
                            'relative_improvement_pct': (mean_improvement / mean_baseline * 100) if mean_baseline != 0 else 0,
                            'p_value': p_value,
                            'significant': p_value < ALPHA
                        }
                        
                        results[pipeline_type][metric] = result
                        
                        print(f"  • {metric}: {mean_improvement:+.4f} ({result['relative_improvement_pct']:+.1f}%), p={p_value:.4f} {'YES' if result['significant'] else 'NO'}")
    
    return results

def create_summary_report(h1_results, h2_results, h3_results, pipeline_results, pairs_df):
    # Erstellt eine umfassende Zusammenfassung der statistischen Analyse
    
    summary = {
        'analysis_type': 'Simplified Statistical Analysis',
        'total_matched_pairs': len(pairs_df),
        'tasks_analyzed': sorted(pairs_df['task'].unique()),
        'languages_analyzed': sorted(pairs_df['language'].unique()),
        'models_analyzed': sorted(pairs_df['model'].unique()),
        'significance_level': ALPHA,
        'hypotheses': {
            'H1': h1_results,
            'H2': h2_results, 
            'H3': h3_results
        },
        'additional_analyses': {
            'pipeline_effects': pipeline_results
        }
    }
    
    return summary

def save_results(summary, pairs_df):
    # Speichert die Analyseergebnisse und das Dataset der passenden Paare
    
    output_dir = "evaluation/statistical_analysis/simplified_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Speichert das Dataset der passenden Paare
    pairs_file = os.path.join(output_dir, "matched_pairs_corrected.csv")
    pairs_df.to_csv(pairs_file, index=False)
    print(f"\nMatched pairs saved to: {pairs_file}")
    
    # Speichert die statistischen Ergebnisse
    results_file = os.path.join(output_dir, "simplified_statistical_results.json")
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Results saved to: {results_file}")
    
    # Speichert den Zusammenfassungsbericht
    summary_file = os.path.join(output_dir, "statistical_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("SIMPLIFIED STATISTICAL ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total matched pairs: {summary['total_matched_pairs']}\n")
        f.write(f"Tasks: {', '.join(summary['tasks_analyzed'])}\n")
        f.write(f"Languages: {', '.join(summary['languages_analyzed'])}\n")
        f.write(f"Models: {', '.join(summary['models_analyzed'])}\n\n")
        
        # H1 Summary
        f.write("H1: CoTR vs Baseline\n")
        f.write("-" * 20 + "\n")
        h1 = summary['hypotheses']['H1']
        if 'overall_f1' in h1:
            result = h1['overall_f1']
            f.write(f"F1 Score: {result['mean_improvement']:+.4f} ({result['relative_improvement_pct']:+.1f}%), ")
            f.write(f"p={result['p_value']:.4f} {'Significant' if result['significant'] else 'Not significant'}\n")
        if 'overall_accuracy' in h1:
            result = h1['overall_accuracy']
            f.write(f"Accuracy: {result['mean_improvement']:+.4f} ({result['relative_improvement_pct']:+.1f}%), ")
            f.write(f"p={result['p_value']:.4f} {'Significant' if result['significant'] else 'Not significant'}\n")
        
        f.write("\nH3: Task-Specific Results\n")
        f.write("-" * 25 + "\n")
        h3 = summary['hypotheses']['H3']
        if 'task_specific' in h3:
            for task, task_data in h3['task_specific'].items():
                if 'f1' in task_data:
                    result = task_data['f1']
                    f.write(f"{task.upper()}: {result['relative_improvement_pct']:+.1f}% F1, ")
                    f.write(f"p={result['p_value']:.4f} {'accurate' if result['significant'] else 'not accurate'}\n")
    
    print(f"Summary saved to: {summary_file}")

def main():
    # Hauptanalysefunktion
    
    print("SIMPLIFIED STATISTICAL ANALYSIS FOR CoTR")
    print("=" * 80)
    print("Analyzing three core hypotheses:")
    print("H1: CoTR vs Baseline effectiveness (paired t-tests)")
    print("H2: Translation quality correlation (Pearson correlation)")  
    print("H3: Task-specific effectiveness (comparative analysis)")
    print("=" * 80)
    
    # Lädt und bereitet Daten mit Normalisierung der Sprachcodes vor
    df = load_and_prepare_data()
    
    # Erstellt passende Paare
    pairs_df = create_matched_pairs(df)
    
    if len(pairs_df) == 0:
        print(" No matched pairs found. Cannot proceed with analysis.")
        return
    
    # Führt Hypothesentests durch
    h1_results = analyze_h1_cotr_vs_baseline(pairs_df)
    h2_results = analyze_h2_translation_quality(pairs_df)
    h3_results = analyze_h3_task_effectiveness(pairs_df)
    
    # Zusätzliche Analyse
    pipeline_results = analyze_pipeline_effects(pairs_df)
    
    # Erstellt Zusammenfassung
    summary = create_summary_report(h1_results, h2_results, h3_results, pipeline_results, pairs_df)
    
    # Speichert Ergebnisse
    save_results(summary, pairs_df)
    
    print(f"\nANALYSIS COMPLETE")
    print(f"    {len(pairs_df)} matched pairs analyzed")
    print(f"    {len(pairs_df['task'].unique())} tasks covered")
    print(f"    {len(pairs_df['language'].unique())} languages covered")

if __name__ == "__main__":
    main() 