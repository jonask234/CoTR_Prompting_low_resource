#!/usr/bin/env python3
"""
Runtime Analysis Script for Baseline vs CoTR Comparison
=======================================================

This is the script that generated the data for runtime_analysis_comprehensive_report.md

The script analyzes runtime performance across all tasks (classification, NER, NLI, QA, sentiment)
comparing baseline approaches with CoTR (Chain-of-Thought Reasoning) approaches.

Usage:
    python runtime_analysis_script.py

Outputs:
    - runtime_analysis_results/ directory with:
      - runtime_data.csv (raw data)
      - efficiency_metrics.csv (overhead calculations)
      - Various PNG plots
      - runtime_analysis_summary.txt
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Definiert die Ergebnisverzeichnisse
RESULT_DIRS = {
    'classification': '/home/bbd6522/code/CoTR_Prompting_low_resource/results/classification_new',
    'ner': '/home/bbd6522/code/CoTR_Prompting_low_resource/results/ner_new', 
    'nli': '/home/bbd6522/code/CoTR_Prompting_low_resource/results/nli_new',
    'qa': '/home/bbd6522/code/CoTR_Prompting_low_resource/results/qa_new',
    'sentiment': '/home/bbd6522/code/CoTR_Prompting_low_resource/results/sentiment_new'
}

# Spaltenzuordnungen für Laufzeit für verschiedene Aufgaben
RUNTIME_COLUMNS = {
    'runtime_seconds': 'runtime_seconds',
    'runtime_sec': 'runtime_seconds', 
    'runtime_per_sample': 'runtime_per_sample',
    'duration': 'runtime_seconds'
}

def normalize_model_name(model_name):
    # Normalisiert Modellnamen für einen konsistenten Vergleich
    if 'aya' in model_name.lower():
        return 'Aya-23-8B'
    elif 'qwen' in model_name.lower():
        return 'Qwen2.5-7B-Instruct'
    else:
        return model_name.split('/')[-1] if '/' in model_name else model_name

def parse_file_info(file_path):
    # Extrahiert Experimentinformationen aus dem Dateipfad und -namen
    path_parts = file_path.split('/')
    filename = os.path.basename(file_path)
    
    info = {
        'task': None,
        'approach': None,
        'pipeline': None,
        'shot_type': None,
        'language': None,
        'model': None
    }
    
    # Extrahiert die Aufgabe
    for task in RESULT_DIRS.keys():
        if task in file_path:
            info['task'] = task
            break
    
    # Extrahiert den Ansatz (Baseline oder CoTR)
    if '/baseline/' in file_path:
        info['approach'] = 'baseline'
    elif '/cotr/' in file_path:
        info['approach'] = 'cotr'
    
    # Extrahiert den Pipeline-Typ für CoTR
    if 'mp_' in filename or 'multi_prompt' in filename:
        info['pipeline'] = 'multi_prompt'
    elif 'sp_' in filename or 'single_prompt' in filename:
        info['pipeline'] = 'single_prompt'
    
    # Extrahiert den Schusstyp
    if '/fs/' in file_path or '_fs_' in filename or 'few_shot' in filename:
        info['shot_type'] = 'few_shot'
    elif '/zs/' in file_path or '_zs_' in filename or 'zero_shot' in filename:
        info['shot_type'] = 'zero_shot'
    
    # Extrahiert die Sprache aus dem Pfad (sucht nach Sprachcodes)
    lang_codes = ['en', 'sw', 'fi', 'ha', 'pt', 'ur', 'fr', 'te']
    for part in path_parts:
        if part in lang_codes:
            info['language'] = part
            break
    
    # Extrahiert den Modellnamen aus dem Pfad
    model_indicators = ['aya', 'qwen', 'Aya', 'Qwen']
    for part in path_parts:
        for indicator in model_indicators:
            if indicator.lower() in part.lower():
                info['model'] = normalize_model_name(part)
                break
        if info['model']:
            break
    
    return info

def load_runtime_data(result_dirs):
    # Lädt Laufzeitdaten aus allen Ergebnisdateien
    all_data = []
    
    for task, base_dir in result_dirs.items():
        print(f"Processing {task} results from {base_dir}")
        
        if not os.path.exists(base_dir):
            print(f"Directory {base_dir} does not exist, skipping {task}")
            continue
        
        # Findet alle CSV-Dateien rekursiv
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".csv"):
                    csv_file = os.path.join(root, file)
                    try:
                        # Parst Dateiinformationen
                        file_info = parse_file_info(str(csv_file))
                        file_info['file_path'] = str(csv_file)
                        
                        # Lädt die CSV-Datei
                        df = pd.read_csv(csv_file)
                        
                        if df.empty:
                            continue
                        
                        # Findet die Laufzeitspalte
                        runtime_col = None
                        for col in df.columns:
                            col_lower = col.lower()
                            if any(rt_col in col_lower for rt_col in RUNTIME_COLUMNS.keys()):
                                runtime_col = col
                                break
                        
                        if runtime_col is None:
                            print(f"No runtime column found in {csv_file}")
                            continue
                        
                        # Extrahiert Laufzeitstatistiken
                        valid_runtimes = df[runtime_col].dropna()
                        if len(valid_runtimes) == 0:
                            continue
                        
                        runtime_stats = {
                            'mean_runtime': valid_runtimes.mean(),
                            'median_runtime': valid_runtimes.median(),
                            'std_runtime': valid_runtimes.std(),
                            'min_runtime': valid_runtimes.min(),
                            'max_runtime': valid_runtimes.max(),
                            'total_runtime': valid_runtimes.sum(),
                            'sample_count': len(valid_runtimes)
                        }
                        
                        # Kombiniert Dateiinformationen mit Laufzeitstatistiken
                        record = {**file_info, **runtime_stats}
                        all_data.append(record)
                        
                    except Exception as e:
                        print(f"Error processing {csv_file}: {e}")
                        continue
    
    return pd.DataFrame(all_data)

def calculate_efficiency_metrics(df):
    # Berechnet Effizienzmetriken, die CoTR mit der Baseline vergleichen
    efficiency_data = []
    
    # Gruppiert nach Aufgabe, Modell, Sprache, Schusstyp
    grouping_cols = ['task', 'model', 'language', 'shot_type']
    
    for group_key, group_df in df.groupby(grouping_cols):
        if len(group_key) != 4:
            continue
            
        task, model, language, shot_type = group_key
        
        # Holt Baseline- und CoTR-Daten
        baseline_data = group_df[group_df['approach'] == 'baseline']
        cotr_data = group_df[group_df['approach'] == 'cotr']
        
        if baseline_data.empty or cotr_data.empty:
            continue
        
        baseline_runtime = baseline_data['mean_runtime'].iloc[0]
        
        # Vergleicht mit verschiedenen CoTR-Pipelines
        for _, cotr_row in cotr_data.iterrows():
            cotr_runtime = cotr_row['mean_runtime']
            overhead_ratio = cotr_runtime / baseline_runtime
            overhead_absolute = cotr_runtime - baseline_runtime
            
            efficiency_record = {
                'task': task,
                'model': model,
                'language': language,
                'shot_type': shot_type,
                'pipeline': cotr_row['pipeline'],
                'baseline_runtime': baseline_runtime,
                'cotr_runtime': cotr_runtime,
                'overhead_ratio': overhead_ratio,
                'overhead_absolute': overhead_absolute,
                'overhead_percentage': (overhead_ratio - 1) * 100
            }
            efficiency_data.append(efficiency_record)
    
    return pd.DataFrame(efficiency_data)

def create_runtime_plots(df, efficiency_df, output_dir):
    # Erstellt umfassende Laufzeitanalyse-Plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Setzt den Stil
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Laufzeitvergleich nach Aufgabe und Ansatz
    plt.figure(figsize=(15, 8))
    plot_df = df[df['approach'].isin(['baseline', 'cotr'])].copy()
    sns.boxplot(data=plot_df, x='task', y='mean_runtime', hue='approach')
    plt.title('Runtime Comparison: Baseline vs CoTR by Task', fontsize=14)
    plt.ylabel('Mean Runtime (seconds)', fontsize=12)
    plt.xlabel('Task', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Approach')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'runtime_by_task.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. CoTR-Pipeline-Vergleich
    plt.figure(figsize=(15, 8))
    cotr_df = df[df['approach'] == 'cotr'].copy()
    if not cotr_df.empty:
        sns.boxplot(data=cotr_df, x='task', y='mean_runtime', hue='pipeline')
        plt.title('CoTR Pipeline Comparison: Single vs Multi-Prompt', fontsize=14)
        plt.ylabel('Mean Runtime (seconds)', fontsize=12)
        plt.xlabel('Task', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Pipeline Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cotr_pipeline_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Overhead-Analyse
    if not efficiency_df.empty:
        plt.figure(figsize=(15, 8))
        sns.barplot(data=efficiency_df, x='task', y='overhead_percentage', hue='pipeline')
        plt.title('CoTR Runtime Overhead by Task and Pipeline', fontsize=14)
        plt.ylabel('Runtime Overhead (%)', fontsize=12)
        plt.xlabel('Task', fontsize=12)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.legend(title='Pipeline Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overhead_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Modellvergleich
    plt.figure(figsize=(15, 8))
    model_df = df.dropna(subset=['model'])
    if not model_df.empty:
        sns.boxplot(data=model_df, x='model', y='mean_runtime', hue='approach')
        plt.title('Runtime by Model and Approach', fontsize=14)
        plt.ylabel('Mean Runtime (seconds)', fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Approach')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'runtime_by_model.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Sprachspezifische Analyse
    plt.figure(figsize=(15, 8))
    lang_df = df.dropna(subset=['language'])
    if not lang_df.empty:
        sns.boxplot(data=lang_df, x='language', y='mean_runtime', hue='approach')
        plt.title('Runtime by Language and Approach', fontsize=14)
        plt.ylabel('Mean Runtime (seconds)', fontsize=12)
        plt.xlabel('Language', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Approach')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'runtime_by_language.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_runtime_summary(df, efficiency_df):
    # Erstellt eine umfassende Zusammenfassung der Laufzeitanalyse
    summary = []
    summary.append("=" * 80)
    summary.append("RUNTIME COMPARISON ANALYSIS: BASELINE vs CoTR")
    summary.append("=" * 80)
    summary.append("")
    
    # Gesamtstatistiken
    baseline_df = df[df['approach'] == 'baseline']
    cotr_df = df[df['approach'] == 'cotr']
    
    if not baseline_df.empty and not cotr_df.empty:
        baseline_avg = baseline_df['mean_runtime'].mean()
        cotr_avg = cotr_df['mean_runtime'].mean()
        overall_overhead = ((cotr_avg / baseline_avg) - 1) * 100
        
        summary.append("OVERALL PERFORMANCE:")
        summary.append(f"  • Baseline Average Runtime: {baseline_avg:.2f} seconds")
        summary.append(f"  • CoTR Average Runtime: {cotr_avg:.2f} seconds")
        summary.append(f"  • Overall CoTR Overhead: {overall_overhead:.1f}%")
        summary.append("")
    
    # Aufgabenspezifische Analyse
    summary.append("TASK-SPECIFIC ANALYSIS:")
    for task in df['task'].unique():
        if pd.isna(task):
            continue
            
        task_df = df[df['task'] == task]
        task_baseline = task_df[task_df['approach'] == 'baseline']['mean_runtime'].mean()
        task_cotr = task_df[task_df['approach'] == 'cotr']['mean_runtime'].mean()
        
        if not pd.isna(task_baseline) and not pd.isna(task_cotr):
            task_overhead = ((task_cotr / task_baseline) - 1) * 100
            summary.append(f"  • {task.upper()}:")
            summary.append(f"    - Baseline: {task_baseline:.2f}s, CoTR: {task_cotr:.2f}s")
            summary.append(f"    - Overhead: {task_overhead:.1f}%")
    summary.append("")
    
    # Pipeline-Vergleich
    if not efficiency_df.empty:
        summary.append("PIPELINE COMPARISON:")
        pipeline_stats = efficiency_df.groupby('pipeline')['overhead_percentage'].agg(['mean', 'std']).round(1)
        for pipeline, stats in pipeline_stats.iterrows():
            summary.append(f"  • {pipeline}: {stats['mean']:.1f}% ± {stats['std']:.1f}% overhead")
        summary.append("")
    
    # Modellvergleich
    summary.append("MODEL COMPARISON:")
    for model in df['model'].unique():
        if pd.isna(model):
            continue
            
        model_df = df[df['model'] == model]
        model_baseline = model_df[model_df['approach'] == 'baseline']['mean_runtime'].mean()
        model_cotr = model_df[model_df['approach'] == 'cotr']['mean_runtime'].mean()
        
        if not pd.isna(model_baseline) and not pd.isna(model_cotr):
            model_overhead = ((model_cotr / model_baseline) - 1) * 100
            summary.append(f"  • {model}:")
            summary.append(f"    - Baseline: {model_baseline:.2f}s, CoTR: {model_cotr:.2f}s")
            summary.append(f"    - Overhead: {model_overhead:.1f}%")
    summary.append("")
    
    # Effizienzeinblicke
    if not efficiency_df.empty:
        summary.append("KEY INSIGHTS:")
        
        # Am besten abschneidende Konfigurationen
        best_overhead = efficiency_df.loc[efficiency_df['overhead_percentage'].idxmin()]
        worst_overhead = efficiency_df.loc[efficiency_df['overhead_percentage'].idxmax()]
        
        summary.append(f"  • Most Efficient CoTR: {best_overhead['task']}/{best_overhead['pipeline']} ")
        summary.append(f"    ({best_overhead['overhead_percentage']:.1f}% overhead)")
        summary.append(f"  • Least Efficient CoTR: {worst_overhead['task']}/{worst_overhead['pipeline']} ")
        summary.append(f"    ({worst_overhead['overhead_percentage']:.1f}% overhead)")
        
        # Pipeline-Präferenzen
        avg_overhead_by_pipeline = efficiency_df.groupby('pipeline')['overhead_percentage'].mean()
        if 'single_prompt' in avg_overhead_by_pipeline.index and 'multi_prompt' in avg_overhead_by_pipeline.index:
            sp_overhead = avg_overhead_by_pipeline['single_prompt']
            mp_overhead = avg_overhead_by_pipeline['multi_prompt']
            if sp_overhead < mp_overhead:
                summary.append(f"  • Single-prompt is more efficient ({sp_overhead:.1f}% vs {mp_overhead:.1f}% overhead)")
            else:
                summary.append(f"  • Multi-prompt is more efficient ({mp_overhead:.1f}% vs {sp_overhead:.1f}% overhead)")
    
    summary.append("")
    summary.append("=" * 80)
    
    return "\n".join(summary)

def main():
    # Hauptfunktion zur Ausführung der Laufzeitvergleichsanalyse
    output_dir = "runtime_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting runtime comparison analysis...")
    
    # Daten laden
    print("Loading runtime data from all result files...")
    runtime_df = load_runtime_data(RESULT_DIRS)
    
    if runtime_df.empty:
        print("No runtime data found!")
        return
    
    print(f"Loaded {len(runtime_df)} runtime records")
    
    # Effizienzmetriken berechnen
    print("Calculating efficiency metrics...")
    efficiency_df = calculate_efficiency_metrics(runtime_df)
    
    # Daten speichern
    runtime_df.to_csv(os.path.join(output_dir, "runtime_data.csv"), index=False)
    if not efficiency_df.empty:
        efficiency_df.to_csv(os.path.join(output_dir, "efficiency_metrics.csv"), index=False)
    
    # Plots erstellen
    print("Creating runtime analysis plots...")
    create_runtime_plots(runtime_df, efficiency_df, output_dir)
    
    # Zusammenfassung erstellen
    summary = generate_runtime_summary(runtime_df, efficiency_df)
    
    # Zusammenfassung speichern
    with open(os.path.join(output_dir, "runtime_analysis_summary.txt"), "w") as f:
        f.write(summary)
    
    # Zusammenfassung drucken
    print(summary)
    
    print(f"Runtime analysis complete! Results saved to {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  • {output_dir}/runtime_data.csv - Raw runtime statistics")
    print(f"  • {output_dir}/efficiency_metrics.csv - Overhead calculations")
    print(f"  • {output_dir}/runtime_analysis_summary.txt - Text summary")
    print(f"  • {output_dir}/*.png - Visualization plots")
    print(f"\nTo create the comprehensive report, this data was manually")
    print(f"analyzed and written to runtime_analysis_comprehensive_report.md")

if __name__ == "__main__":
    main() 