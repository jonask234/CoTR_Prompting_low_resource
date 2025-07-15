#!/usr/bin/env python3
"""
Analyze NER Parsing Success Rate
This script examines NER result files to calculate parsing success rate
by checking if raw outputs contain valid entity labels (PER, ORG, LOC, DATE)
"""

import pandas as pd
import numpy as np
import re
import json
import os
import glob
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import ast

def load_comprehensive_metrics():
    """Load comprehensive metrics to get F1 scores"""
    metrics_file = "/home/bbd6522/code/CoTR_Prompting_low_resource/results/analysis/comprehensive_metrics.json"
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # Create a lookup dictionary for F1 scores
    f1_lookup = {}
    
    if 'ner' in data:
        for result in data['ner']:
            approach = result['approach']
            language = result['language']
            model = result['model']
            shot_type = result['shot_type']
            f1_score = result.get('f1_score', 0.0)
            
            # Create multiple possible keys to handle variations
            keys = [
                f"{approach}_{language}_{model}_{shot_type}",
                f"{approach}_{language}_{model.replace('/', '_').replace('-', '_')}_{shot_type}",
                f"{approach}_{language}_{model.replace('/', '_').replace('-', '_').replace('.', '_')}_{shot_type}"
            ]
            
            for key in keys:
                f1_lookup[key] = f1_score
    
    return f1_lookup

def create_lookup_key(approach: str, language: str, model: str, shot_type: str) -> List[str]:
    """Create multiple possible lookup keys to handle variations"""
    # Handle language code variations
    lang_map = {
        'swa': 'sw',
        'hau': 'ha',
        'sw': 'sw',
        'ha': 'ha'
    }
    
    # Get the standardized language code
    std_lang = lang_map.get(language, language)
    
    # Handle model name variations
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

def is_valid_ner_output(raw_output: str) -> bool:
    """
    Check if raw output contains valid NER entity labels in expected format
    Valid labels: PER, ORG, LOC, DATE
    Expected format: [LABEL: entity_text] or similar variations
    """
    if not raw_output or pd.isna(raw_output):
        return False
    
    # Valid entity types
    valid_labels = ['PER', 'ORG', 'LOC', 'DATE']
    
    # Pattern to match [LABEL: text] format
    pattern = r'\[(?:' + '|'.join(valid_labels) + r'):\s*[^\]]+\]'
    
    # Check if output contains at least one valid entity label
    matches = re.findall(pattern, raw_output, re.IGNORECASE)
    
    return len(matches) > 0

def extract_valid_entities_from_raw(raw_output: str) -> List[str]:
    """Extract valid entity labels from raw output"""
    if not raw_output or pd.isna(raw_output):
        return []
    
    valid_labels = ['PER', 'ORG', 'LOC', 'DATE']
    pattern = r'\[(' + '|'.join(valid_labels) + r'):\s*[^\]]+\]'
    
    matches = re.findall(pattern, raw_output, re.IGNORECASE)
    return matches

def count_predicted_entities(predicted_entities_str: str) -> int:
    """Count number of predicted entities from the predicted_entities column"""
    if not predicted_entities_str or pd.isna(predicted_entities_str):
        return 0
    
    try:
        # Parse the string as a Python literal (list of dicts)
        entities = ast.literal_eval(predicted_entities_str)
        if isinstance(entities, list):
            return len(entities)
        return 0
    except (ValueError, SyntaxError):
        return 0

def analyze_ner_file(file_path: str, f1_lookup: Dict[str, float]) -> Dict[str, Any]:
    """Analyze a single NER result file"""
    try:
        df = pd.read_csv(file_path)
        
        # Extract metadata from filename
        filename = os.path.basename(file_path)
        parts = filename.replace('.csv', '').split('_')
        
        # Parse filename: results_baseline_fs_ner_swa_aya-23-8B.csv
        approach = 'baseline'
        if 'cotr' in filename:
            if 'mp' in filename:
                approach = 'cotr_mp'
            elif 'sp' in filename:
                approach = 'cotr_sp'
            else:
                approach = 'cotr'
        
        # Extract language (should be after 'ner')
        language = 'unknown'
        try:
            ner_idx = parts.index('ner')
            if ner_idx + 1 < len(parts):
                language = parts[ner_idx + 1]
        except ValueError:
            pass
        
        # Extract model name (everything after language)
        model = 'unknown'
        try:
            ner_idx = parts.index('ner')
            if ner_idx + 2 < len(parts):
                model = '_'.join(parts[ner_idx + 2:])
        except ValueError:
            pass
        
        # Extract shot type
        shot_type = 'unknown'
        if 'fs' in parts:
            shot_type = 'fewshot'
        elif 'zs' in parts:
            shot_type = 'zeroshot'
        
        # Analyze parsing success
        total_samples = len(df)
        valid_samples = 0
        total_entities_extracted = 0
        
        if 'raw_output' in df.columns:
            for _, row in df.iterrows():
                if is_valid_ner_output(row['raw_output']):
                    valid_samples += 1
                
                # Count entities
                if 'predicted_entities' in row:
                    total_entities_extracted += count_predicted_entities(str(row['predicted_entities']))
        
        parsing_success_rate = (valid_samples / total_samples) * 100 if total_samples > 0 else 0
        
        # Get F1 score from comprehensive metrics using multiple possible keys
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
    """Main function to analyze all NER result files"""
    
    # Load comprehensive metrics for F1 scores
    f1_lookup = load_comprehensive_metrics()
    print(f"Loaded {len(f1_lookup)} F1 scores from comprehensive metrics")
    
    # Find all NER result files
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
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("NER PARSING SUCCESS RATE ANALYSIS")
    print("="*80)
    
    # Overall statistics
    total_samples = results_df['total_samples'].sum()
    total_valid = results_df['valid_samples'].sum()
    overall_parsing_success = (total_valid / total_samples) * 100 if total_samples > 0 else 0
    overall_f1 = results_df['f1_score'].mean()
    
    print(f"\n📊 **OVERALL STATISTICS**")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Valid samples: {total_valid:,}")
    print(f"  Overall parsing success rate: {overall_parsing_success:.1f}%")
    print(f"  Overall mean F1 score: {overall_f1:.4f}")
    
    # By approach
    print(f"\n📊 **BY APPROACH**")
    approach_stats = results_df.groupby('approach').agg({
        'parsing_success_rate': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    }).round(4)
    
    for approach, stats in approach_stats.iterrows():
        print(f"  {approach}: {stats['parsing_success_rate']:.1f}% parsing success, {stats['f1_score']:.4f} F1, {stats['total_samples']} samples")
    
    # By model
    print(f"\n📊 **BY MODEL**")
    model_stats = results_df.groupby('model').agg({
        'parsing_success_rate': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    }).round(4)
    
    for model, stats in model_stats.iterrows():
        print(f"  {model}: {stats['parsing_success_rate']:.1f}% parsing success, {stats['f1_score']:.4f} F1, {stats['total_samples']} samples")
    
    # By language
    print(f"\n📊 **BY LANGUAGE**")
    lang_stats = results_df.groupby('language').agg({
        'parsing_success_rate': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    }).round(4)
    
    for language, stats in lang_stats.iterrows():
        print(f"  {language}: {stats['parsing_success_rate']:.1f}% parsing success, {stats['f1_score']:.4f} F1, {stats['total_samples']} samples")
    
    # By shot type
    print(f"\n📊 **BY SHOT TYPE**")
    shot_stats = results_df.groupby('shot_type').agg({
        'parsing_success_rate': 'mean',
        'f1_score': 'mean',
        'total_samples': 'sum'
    }).round(4)
    
    for shot_type, stats in shot_stats.iterrows():
        print(f"  {shot_type}: {stats['parsing_success_rate']:.1f}% parsing success, {stats['f1_score']:.4f} F1, {stats['total_samples']} samples")
    
    # Save detailed results
    output_file = "/home/bbd6522/code/CoTR_Prompting_low_resource/ner_parsing_success_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Create summary for chapter
    print(f"\n📝 **SUMMARY FOR CHAPTER UPDATE**")
    print(f"Overall parsing success rate: {overall_parsing_success:.1f}%")
    print(f"Overall mean F1 score: {overall_f1:.4f}")
    
    # Baseline vs CoTR comparison
    baseline_mask = results_df['approach'] == 'baseline'
    cotr_mask = results_df['approach'].isin(['cotr_mp', 'cotr_sp'])
    
    if baseline_mask.any():
        baseline_parsing = results_df[baseline_mask]['parsing_success_rate'].mean()
        baseline_f1 = results_df[baseline_mask]['f1_score'].mean()
        print(f"Baseline: {baseline_parsing:.1f}% parsing success, {baseline_f1:.4f} F1")
    
    if cotr_mask.any():
        cotr_parsing = results_df[cotr_mask]['parsing_success_rate'].mean()
        cotr_f1 = results_df[cotr_mask]['f1_score'].mean()
        print(f"CoTR: {cotr_parsing:.1f}% parsing success, {cotr_f1:.4f} F1")
    
    # CoTR pipeline comparison
    cotr_mp_mask = results_df['approach'] == 'cotr_mp'
    cotr_sp_mask = results_df['approach'] == 'cotr_sp'
    
    if cotr_mp_mask.any():
        mp_parsing = results_df[cotr_mp_mask]['parsing_success_rate'].mean()
        mp_f1 = results_df[cotr_mp_mask]['f1_score'].mean()
        print(f"CoTR Multi-Prompt: {mp_parsing:.1f}% parsing success, {mp_f1:.4f} F1")
    
    if cotr_sp_mask.any():
        sp_parsing = results_df[cotr_sp_mask]['parsing_success_rate'].mean()
        sp_f1 = results_df[cotr_sp_mask]['f1_score'].mean()
        print(f"CoTR Single-Prompt: {sp_parsing:.1f}% parsing success, {sp_f1:.4f} F1")

if __name__ == "__main__":
    main() 