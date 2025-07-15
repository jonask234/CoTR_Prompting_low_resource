#!/usr/bin/env python3
"""
Sentiment Analysis Parsing Success Rate Analysis

This script analyzes the parsing success rates for sentiment analysis experiments
by checking how often models produce valid sentiment labels (positive, negative, neutral)
versus invalid/unparseable outputs.
"""

import os
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import json

# Valid sentiment labels (case-insensitive)
VALID_SENTIMENT_LABELS = ['positive', 'negative', 'neutral']

def is_valid_sentiment_label(predicted_label):
    """Check if a predicted label is a valid sentiment label"""
    if pd.isna(predicted_label):
        return False
    
    label_clean = str(predicted_label).strip().lower()
    return label_clean in VALID_SENTIMENT_LABELS

def analyze_sentiment_file_parsing(file_path):
    """Analyze parsing success for a single sentiment result file"""
    try:
        df = pd.read_csv(file_path)
        
        if 'predicted_label' not in df.columns:
            return None
        
        total_samples = len(df)
        valid_parses = sum(is_valid_sentiment_label(label) for label in df['predicted_label'])
        
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
    """Extract configuration information from file path"""
    # Initialize with defaults
    config = {
        'approach': 'unknown',
        'language': 'unknown', 
        'model': 'unknown',
        'shot_type': 'unknown',
        'pipeline': 'unknown'
    }
    
    filename = os.path.basename(file_path)
    path_parts = file_path.split('/')
    
    # Determine approach
    if 'baseline' in file_path:
        config['approach'] = 'baseline'
    elif 'multi_prompt' in file_path:
        config['approach'] = 'cotr_mp'
    elif 'single_prompt' in file_path:
        config['approach'] = 'cotr_sp'
    
    # Extract language
    for lang in ['ha', 'sw', 'pt']:
        if f'_{lang}_' in filename or f'/{lang}/' in file_path:
            config['language'] = lang
            break
    
    # Extract model
    if 'Qwen2.5-7B-Instruct' in filename or 'Qwen2.5-7B-Instruct' in file_path:
        config['model'] = 'Qwen2.5-7B-Instruct'
    elif 'aya-23-8B' in filename or 'aya-23-8B' in file_path:
        config['model'] = 'aya-23-8B'
    
    # Extract shot type
    if 'few_shot' in filename or '_fs_' in filename:
        config['shot_type'] = 'fewshot'
    elif 'zero_shot' in filename or '_zs_' in filename:
        config['shot_type'] = 'zeroshot'
    
    # Pipeline for CoTR
    if config['approach'] == 'cotr_mp':
        config['pipeline'] = 'multi_prompt'
    elif config['approach'] == 'cotr_sp':
        config['pipeline'] = 'single_prompt'
    else:
        config['pipeline'] = 'baseline'
    
    return config

def analyze_sentiment_parsing_success():
    """Main function to analyze sentiment parsing success rates"""
    
    print("=== SENTIMENT PARSING SUCCESS RATE ANALYSIS ===\n")
    
    # Base path for sentiment results
    sentiment_base_path = '/home/bbd6522/code/CoTR_Prompting_low_resource/results/sentiment_new'
    
    if not os.path.exists(sentiment_base_path):
        print(f"❌ Sentiment results path not found: {sentiment_base_path}")
        return None
    
    # Find all result files
    result_files = []
    for root, dirs, files in os.walk(sentiment_base_path):
        for file in files:
            if file.endswith('.csv') and 'results_' in file:
                result_files.append(os.path.join(root, file))
    
    print(f"📁 Found {len(result_files)} sentiment result files")
    
    if not result_files:
        print("❌ No sentiment result files found")
        return None
    
    # Analyze each file
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
            
            print(f"✅ {config['approach']} | {config['model']} | {config['language']} | {config['shot_type']}: "
                  f"{file_analysis['parsing_success_rate']:.1%} success "
                  f"({file_analysis['valid_parses']}/{file_analysis['total_samples']})")
    
    if not parsing_results:
        print("❌ No valid parsing results found")
        return None
    
    # Create DataFrame for analysis
    parsing_df = pd.DataFrame(parsing_results)
    
    # Overall statistics
    overall_success_rate = total_valid_parses / total_samples if total_samples > 0 else 0
    overall_failure_rate = 1 - overall_success_rate
    
    print(f"\n📊 OVERALL PARSING STATISTICS:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Valid parses: {total_valid_parses:,}")
    print(f"   Success rate: {overall_success_rate:.1%}")
    print(f"   Failure rate: {overall_failure_rate:.1%}")
    
    # Analysis by approach
    print(f"\n📈 PARSING SUCCESS BY APPROACH:")
    approach_stats = parsing_df.groupby('approach').agg({
        'total_samples': 'sum',
        'valid_parses': 'sum',
        'parsing_success_rate': 'mean',
        'parsing_failure_rate': 'mean'
    }).round(4)
    
    for approach, stats in approach_stats.iterrows():
        print(f"   {approach}:")
        print(f"      Samples: {stats['total_samples']:,}")
        print(f"      Success rate: {stats['parsing_success_rate']:.1%}")
        print(f"      Failure rate: {stats['parsing_failure_rate']:.1%}")
    
    # Analysis by model
    print(f"\n🤖 PARSING SUCCESS BY MODEL:")
    model_stats = parsing_df.groupby('model').agg({
        'parsing_success_rate': 'mean',
        'parsing_failure_rate': 'mean'
    }).round(4)
    
    for model, stats in model_stats.iterrows():
        print(f"   {model}: {stats['parsing_success_rate']:.1%} success, {stats['parsing_failure_rate']:.1%} failure")
    
    # Analysis by language
    print(f"\n🌍 PARSING SUCCESS BY LANGUAGE:")
    lang_stats = parsing_df.groupby('language').agg({
        'parsing_success_rate': 'mean',
        'parsing_failure_rate': 'mean'
    }).round(4)
    
    for language, stats in lang_stats.iterrows():
        print(f"   {language}: {stats['parsing_success_rate']:.1%} success, {stats['parsing_failure_rate']:.1%} failure")
    
    # Find best and worst configurations
    best_config = parsing_df.loc[parsing_df['parsing_success_rate'].idxmax()]
    worst_config = parsing_df.loc[parsing_df['parsing_success_rate'].idxmin()]
    
    print(f"\n🏆 BEST PARSING CONFIGURATION:")
    print(f"   {best_config['approach']} | {best_config['model']} | {best_config['language']} | {best_config['shot_type']}")
    print(f"   Success: {best_config['parsing_success_rate']:.1%} ({best_config['valid_parses']}/{best_config['total_samples']})")
    
    print(f"\n💥 WORST PARSING CONFIGURATION:")
    print(f"   {worst_config['approach']} | {worst_config['model']} | {worst_config['language']} | {worst_config['shot_type']}")
    print(f"   Success: {worst_config['parsing_success_rate']:.1%} ({worst_config['valid_parses']}/{worst_config['total_samples']})")
    
    # CoTR pipeline comparison
    cotr_data = parsing_df[parsing_df['approach'].isin(['cotr_mp', 'cotr_sp'])]
    if len(cotr_data) > 0:
        print(f"\n🔄 COTR PIPELINE COMPARISON:")
        pipeline_stats = cotr_data.groupby('approach').agg({
            'parsing_success_rate': 'mean',
            'parsing_failure_rate': 'mean'
        }).round(4)
        
        for pipeline, stats in pipeline_stats.iterrows():
            pipeline_name = "Multi-prompt" if pipeline == "cotr_mp" else "Single-prompt"
            print(f"   {pipeline_name}: {stats['parsing_success_rate']:.1%} success, {stats['parsing_failure_rate']:.1%} failure")
    
    return parsing_df

def save_parsing_analysis(parsing_df, output_file='sentiment_parsing_analysis.csv'):
    """Save parsing analysis results to CSV"""
    if parsing_df is not None:
        parsing_df.to_csv(output_file, index=False)
        print(f"\n💾 Parsing analysis saved to: {output_file}")
        return output_file
    return None

def generate_parsing_summary(parsing_df):
    """Generate a summary report of parsing analysis"""
    if parsing_df is None:
        return "No parsing data available"
    
    summary = []
    summary.append("=== SENTIMENT PARSING SUCCESS SUMMARY ===\n")
    
    # Overall statistics
    total_configs = len(parsing_df)
    avg_success = parsing_df['parsing_success_rate'].mean()
    avg_failure = parsing_df['parsing_failure_rate'].mean()
    
    summary.append(f"Total configurations analyzed: {total_configs}")
    summary.append(f"Average parsing success rate: {avg_success:.1%}")
    summary.append(f"Average parsing failure rate: {avg_failure:.1%}")
    
    # Approach comparison
    summary.append(f"\nParsing success by approach:")
    approach_stats = parsing_df.groupby('approach')['parsing_success_rate'].mean().sort_values(ascending=False)
    for approach, success_rate in approach_stats.items():
        summary.append(f"  {approach}: {success_rate:.1%}")
    
    # Model comparison
    summary.append(f"\nParsing success by model:")
    model_stats = parsing_df.groupby('model')['parsing_success_rate'].mean().sort_values(ascending=False)
    for model, success_rate in model_stats.items():
        summary.append(f"  {model}: {success_rate:.1%}")
    
    return "\n".join(summary)

if __name__ == "__main__":
    # Run parsing analysis
    parsing_df = analyze_sentiment_parsing_success()
    
    if parsing_df is not None:
        # Save results
        output_file = save_parsing_analysis(parsing_df)
        
        # Generate and print summary
        summary = generate_parsing_summary(parsing_df)
        print(f"\n{summary}")
        
        # Save summary to file
        with open('sentiment_parsing_summary.txt', 'w') as f:
            f.write(summary)
        print(f"\n💾 Summary saved to: sentiment_parsing_summary.txt")
    
    print(f"\n✅ Sentiment parsing analysis complete!") 