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
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define result directories
RESULT_DIRS = {
    'classification': '/home/bbd6522/code/CoTR_Prompting_low_resource/results/classification_new',
    'ner': '/home/bbd6522/code/CoTR_Prompting_low_resource/results/ner_new', 
    'nli': '/home/bbd6522/code/CoTR_Prompting_low_resource/results/nli_new',
    'qa': '/home/bbd6522/code/CoTR_Prompting_low_resource/results/qa_new',
    'sentiment': '/home/bbd6522/code/CoTR_Prompting_low_resource/results/sentiment_new'
}

# Runtime column mappings for different tasks
RUNTIME_COLUMNS = {
    'runtime_seconds': 'runtime_seconds',
    'runtime_sec': 'runtime_seconds', 
    'runtime_per_sample': 'runtime_per_sample',
    'duration': 'runtime_seconds'
}

def normalize_model_name(model_name: str) -> str:
    """Normalize model names for consistent comparison"""
    if 'aya' in model_name.lower():
        return 'Aya-23-8B'
    elif 'qwen' in model_name.lower():
        return 'Qwen2.5-7B-Instruct'
    else:
        return model_name.split('/')[-1] if '/' in model_name else model_name

def parse_file_info(file_path: str) -> Dict[str, str]:
    """Extract experiment information from file path and name"""
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
    
    # Extract task
    for task in RESULT_DIRS.keys():
        if task in file_path:
            info['task'] = task
            break
    
    # Extract approach (baseline or cotr)
    if '/baseline/' in file_path:
        info['approach'] = 'baseline'
    elif '/cotr/' in file_path:
        info['approach'] = 'cotr'
    
    # Extract pipeline type for CoTR
    if 'mp_' in filename or 'multi_prompt' in filename:
        info['pipeline'] = 'multi_prompt'
    elif 'sp_' in filename or 'single_prompt' in filename:
        info['pipeline'] = 'single_prompt'
    
    # Extract shot type
    if '/fs/' in file_path or '_fs_' in filename or 'few_shot' in filename:
        info['shot_type'] = 'few_shot'
    elif '/zs/' in file_path or '_zs_' in filename or 'zero_shot' in filename:
        info['shot_type'] = 'zero_shot'
    
    # Extract language from path (look for language codes)
    lang_codes = ['en', 'sw', 'fi', 'ha', 'pt', 'ur', 'fr', 'te']
    for part in path_parts:
        if part in lang_codes:
            info['language'] = part
            break
    
    # Extract model name from path
    model_indicators = ['aya', 'qwen', 'Aya', 'Qwen']
    for part in path_parts:
        for indicator in model_indicators:
            if indicator.lower() in part.lower():
                info['model'] = normalize_model_name(part)
                break
        if info['model']:
            break
    
    return info

def load_runtime_data(result_dirs: Dict[str, str]) -> pd.DataFrame:
    """Load runtime data from all result files"""
    all_data = []
    
    for task, base_dir in result_dirs.items():
        logger.info(f"Processing {task} results from {base_dir}")
        
        if not os.path.exists(base_dir):
            logger.warning(f"Directory {base_dir} does not exist, skipping {task}")
            continue
        
        # Find all CSV files recursively
        for csv_file in Path(base_dir).rglob("*.csv"):
            try:
                # Parse file information
                file_info = parse_file_info(str(csv_file))
                file_info['file_path'] = str(csv_file)
                
                # Load the CSV file
                df = pd.read_csv(csv_file)
                
                if df.empty:
                    continue
                
                # Find runtime column
                runtime_col = None
                for col in df.columns:
                    col_lower = col.lower()
                    if any(rt_col in col_lower for rt_col in RUNTIME_COLUMNS.keys()):
                        runtime_col = col
                        break
                
                if runtime_col is None:
                    logger.warning(f"No runtime column found in {csv_file}")
                    continue
                
                # Extract runtime statistics
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
                
                # Combine file info with runtime stats
                record = {**file_info, **runtime_stats}
                all_data.append(record)
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
                continue
    
    return pd.DataFrame(all_data)

def calculate_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate efficiency metrics comparing CoTR to baseline"""
    efficiency_data = []
    
    # Group by task, model, language, shot_type
    grouping_cols = ['task', 'model', 'language', 'shot_type']
    
    for group_key, group_df in df.groupby(grouping_cols):
        if len(group_key) != 4:
            continue
            
        task, model, language, shot_type = group_key
        
        # Get baseline and CoTR data
        baseline_data = group_df[group_df['approach'] == 'baseline']
        cotr_data = group_df[group_df['approach'] == 'cotr']
        
        if baseline_data.empty or cotr_data.empty:
            continue
        
        baseline_runtime = baseline_data['mean_runtime'].iloc[0]
        
        # Compare with different CoTR pipelines
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

def create_runtime_plots(df: pd.DataFrame, efficiency_df: pd.DataFrame, output_dir: str):
    """Create comprehensive runtime analysis plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Runtime comparison by task and approach
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
    
    # 2. CoTR pipeline comparison
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
    
    # 3. Overhead analysis
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
    
    # 4. Model comparison
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
    
    # 5. Language-specific analysis
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

def generate_runtime_summary(df: pd.DataFrame, efficiency_df: pd.DataFrame) -> str:
    """Generate a comprehensive runtime analysis summary"""
    summary = []
    summary.append("=" * 80)
    summary.append("RUNTIME COMPARISON ANALYSIS: BASELINE vs CoTR")
    summary.append("=" * 80)
    summary.append("")
    
    # Overall statistics
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
    
    # Task-specific analysis
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
    
    # Pipeline comparison
    if not efficiency_df.empty:
        summary.append("PIPELINE COMPARISON:")
        pipeline_stats = efficiency_df.groupby('pipeline')['overhead_percentage'].agg(['mean', 'std']).round(1)
        for pipeline, stats in pipeline_stats.iterrows():
            summary.append(f"  • {pipeline}: {stats['mean']:.1f}% ± {stats['std']:.1f}% overhead")
        summary.append("")
    
    # Model comparison
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
    
    # Efficiency insights
    if not efficiency_df.empty:
        summary.append("KEY INSIGHTS:")
        
        # Best performing configurations
        best_overhead = efficiency_df.loc[efficiency_df['overhead_percentage'].idxmin()]
        worst_overhead = efficiency_df.loc[efficiency_df['overhead_percentage'].idxmax()]
        
        summary.append(f"  • Most Efficient CoTR: {best_overhead['task']}/{best_overhead['pipeline']} ")
        summary.append(f"    ({best_overhead['overhead_percentage']:.1f}% overhead)")
        summary.append(f"  • Least Efficient CoTR: {worst_overhead['task']}/{worst_overhead['pipeline']} ")
        summary.append(f"    ({worst_overhead['overhead_percentage']:.1f}% overhead)")
        
        # Pipeline preferences
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
    """Main function to run the runtime comparison analysis"""
    output_dir = "runtime_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting runtime comparison analysis...")
    
    # Load data
    logger.info("Loading runtime data from all result files...")
    runtime_df = load_runtime_data(RESULT_DIRS)
    
    if runtime_df.empty:
        logger.error("No runtime data found!")
        return
    
    logger.info(f"Loaded {len(runtime_df)} runtime records")
    
    # Calculate efficiency metrics
    logger.info("Calculating efficiency metrics...")
    efficiency_df = calculate_efficiency_metrics(runtime_df)
    
    # Save data
    runtime_df.to_csv(os.path.join(output_dir, "runtime_data.csv"), index=False)
    if not efficiency_df.empty:
        efficiency_df.to_csv(os.path.join(output_dir, "efficiency_metrics.csv"), index=False)
    
    # Generate plots
    logger.info("Creating runtime analysis plots...")
    create_runtime_plots(runtime_df, efficiency_df, output_dir)
    
    # Generate summary
    summary = generate_runtime_summary(runtime_df, efficiency_df)
    
    # Save summary
    with open(os.path.join(output_dir, "runtime_analysis_summary.txt"), "w") as f:
        f.write(summary)
    
    # Print summary
    print(summary)
    
    logger.info(f"Runtime analysis complete! Results saved to {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  • {output_dir}/runtime_data.csv - Raw runtime statistics")
    print(f"  • {output_dir}/efficiency_metrics.csv - Overhead calculations")
    print(f"  • {output_dir}/runtime_analysis_summary.txt - Text summary")
    print(f"  • {output_dir}/*.png - Visualization plots")
    print(f"\nTo create the comprehensive report, this data was manually")
    print(f"analyzed and written to runtime_analysis_comprehensive_report.md")

if __name__ == "__main__":
    main() 