import pandas as pd
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_nli_parsing_success(comprehensive_metrics_path: str) -> Dict[str, any]:
    """
    Analyze NLI parsing success rates by examining raw model outputs.
    
    Args:
        comprehensive_metrics_path: Path to comprehensive metrics JSON file
        
    Returns:
        Dictionary with parsing success analysis
    """
    
    # Load comprehensive metrics
    with open(comprehensive_metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    nli_configs = metrics_data.get('nli', [])
    
    # Initialize tracking variables
    parsing_results = []
    total_samples = 0
    total_valid_labels = 0
    
    # Valid NLI labels (as strings and integers)
    valid_labels = {'0', '1', '2', 'entailment', 'neutral', 'contradiction'}
    
    # Process each NLI configuration
    for config in nli_configs:
        file_path = config['file_path']
        approach = config['approach']
        language = config['language']
        model = config['model']
        shot_type = config['shot_type']
        
        # Try to read the actual results file
        full_path = os.path.join("/home/bbd6522/code/CoTR_Prompting_low_resource/results", file_path)
        
        if not os.path.exists(full_path):
            logger.warning(f"File not found: {full_path}")
            continue
            
        try:
            df = pd.read_csv(full_path)
            
            # Count valid parsing for this configuration
            valid_count = 0
            total_count = len(df)
            
            # Check for predicted labels column
            pred_col = None
            for col in df.columns:
                if 'predicted' in col.lower() or 'label' in col.lower():
                    pred_col = col
                    break
            
            if pred_col is None:
                logger.warning(f"No predicted label column found in {full_path}")
                continue
                
            # Count valid labels
            for _, row in df.iterrows():
                pred_label = str(row[pred_col]).strip().lower()
                
                # Check if it's a valid NLI label
                if (pred_label in valid_labels or 
                    re.match(r'^[0-2]$', pred_label) or
                    any(label in pred_label for label in ['entailment', 'neutral', 'contradiction'])):
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
            logger.error(f"Error processing {full_path}: {e}")
            continue
    
    # Calculate overall statistics
    overall_parsing_success = total_valid_labels / total_samples if total_samples > 0 else 0
    
    # Create summary by different dimensions
    results_df = pd.DataFrame(parsing_results)
    
    # Summary by approach
    approach_summary = results_df.groupby('approach').agg({
        'parsing_success_rate': 'mean',
        'accuracy': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    }).round(4)
    
    # Summary by language
    language_summary = results_df.groupby('language').agg({
        'parsing_success_rate': 'mean',
        'accuracy': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    }).round(4)
    
    # Summary by model
    model_summary = results_df.groupby('model').agg({
        'parsing_success_rate': 'mean',
        'accuracy': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    }).round(4)
    
    # Summary by shot type
    shot_summary = results_df.groupby('shot_type').agg({
        'parsing_success_rate': 'mean',
        'accuracy': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    }).round(4)
    
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

def create_nli_comprehensive_analysis(comprehensive_metrics_path: str) -> Dict[str, any]:
    """
    Create comprehensive NLI analysis combining parsing success with performance metrics.
    
    Args:
        comprehensive_metrics_path: Path to comprehensive metrics JSON file
        
    Returns:
        Dictionary with comprehensive analysis
    """
    
    # Load comprehensive metrics
    with open(comprehensive_metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    nli_configs = metrics_data.get('nli', [])
    
    # Extract baseline and CoTR results
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
    
    # Create DataFrames
    baseline_df = pd.DataFrame(baseline_results)
    cotr_df = pd.DataFrame(cotr_results)
    
    # Calculate summaries
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
    
    # Language-specific analysis
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
    
    # Model-specific analysis
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
    
    # Shot type analysis
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
    
    # Pipeline analysis for CoTR
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
    """Main function to run the analysis."""
    
    # Path to comprehensive metrics
    comprehensive_metrics_path = "/home/bbd6522/code/CoTR_Prompting_low_resource/results/analysis/comprehensive_metrics.json"
    
    print("🔍 Analyzing NLI parsing success rates...")
    parsing_analysis = analyze_nli_parsing_success(comprehensive_metrics_path)
    
    print("\n📊 Creating comprehensive NLI analysis...")
    comprehensive_analysis = create_nli_comprehensive_analysis(comprehensive_metrics_path)
    
    # Print key findings
    print("\n" + "="*60)
    print("📋 NLI COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📊 OVERALL PARSING SUCCESS: {parsing_analysis['overall_parsing_success']:.1%}")
    print(f"📊 TOTAL SAMPLES ANALYZED: {parsing_analysis['total_samples']}")
    print(f"📊 TOTAL VALID LABELS: {parsing_analysis['total_valid_labels']}")
    
    print(f"\n🎯 BASELINE PERFORMANCE:")
    baseline_summary = comprehensive_analysis['baseline_summary']
    print(f"   Average Accuracy: {baseline_summary['avg_accuracy']:.1%}")
    print(f"   Average F1: {baseline_summary['avg_f1']:.3f}")
    print(f"   Best Accuracy: {baseline_summary['best_accuracy']:.1%}")
    print(f"   Worst Accuracy: {baseline_summary['worst_accuracy']:.1%}")
    print(f"   Configurations: {baseline_summary['total_configs']}")
    
    print(f"\n🔄 COTR PERFORMANCE:")
    cotr_summary = comprehensive_analysis['cotr_summary']
    print(f"   Average Accuracy: {cotr_summary['avg_accuracy']:.1%}")
    print(f"   Average F1: {cotr_summary['avg_f1']:.3f}")
    print(f"   Best Accuracy: {cotr_summary['best_accuracy']:.1%}")
    print(f"   Worst Accuracy: {cotr_summary['worst_accuracy']:.1%}")
    print(f"   Configurations: {cotr_summary['total_configs']}")
    
    print(f"\n🌍 LANGUAGE PERFORMANCE:")
    for lang, stats in comprehensive_analysis['language_analysis'].items():
        print(f"   {lang.upper()}: Baseline {stats['baseline_accuracy']:.1%} | CoTR {stats['cotr_accuracy']:.1%}")
    
    print(f"\n🤖 MODEL PERFORMANCE:")
    for model, stats in comprehensive_analysis['model_analysis'].items():
        model_short = model.split('-')[0] if '-' in model else model
        print(f"   {model_short}: Baseline {stats['baseline_accuracy']:.1%} | CoTR {stats['cotr_accuracy']:.1%}")
    
    print(f"\n🎯 SHOT TYPE PERFORMANCE:")
    for shot, stats in comprehensive_analysis['shot_analysis'].items():
        print(f"   {shot.title()}: Baseline {stats['baseline_accuracy']:.1%} | CoTR {stats['cotr_accuracy']:.1%}")
    
    print(f"\n🔄 PIPELINE PERFORMANCE:")
    pipeline_stats = comprehensive_analysis['pipeline_analysis']
    print(f"   Single-prompt: {pipeline_stats['single_prompt']['accuracy']:.1%}")
    print(f"   Multi-prompt: {pipeline_stats['multi_prompt']['accuracy']:.1%}")
    
    # Save detailed results
    print("\n💾 Saving detailed results...")
    
    # Save parsing success analysis
    results_df = parsing_analysis['results_df']
    results_df.to_csv('nli_parsing_success_analysis.csv', index=False)
    
    # Save comprehensive analysis
    with open('nli_comprehensive_analysis.json', 'w') as f:
        # Convert DataFrames to dictionaries for JSON serialization
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
    
    print("✅ Analysis complete! Results saved to:")
    print("   - nli_parsing_success_analysis.csv")
    print("   - nli_comprehensive_analysis.json")
    
    return parsing_analysis, comprehensive_analysis

if __name__ == "__main__":
    parsing_analysis, comprehensive_analysis = main() 