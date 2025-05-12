# src/experiments/simple_baseline/ner/ner_simple_baselines_theoretical.py

import sys
import os
import argparse
import pandas as pd
import numpy as np
import random
from collections import Counter
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

# Import utility functions
from src.utils.data_loaders.load_masakhaner import load_masakhaner_samples

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run NER simple theoretical baselines")
    parser.add_argument("--langs", nargs='+', default=['sw', 'ha'],
                        help="Languages to test (default: sw, ha)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to use per language (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="/work/bbd6522/results/ner/theoretical_baselines",
                        help="Base output directory for results, plots, and summaries")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use (train, dev, test)")
    return parser.parse_args()

def analyze_dataset_structure(samples_df, lang_code, output_dir):
    """
    Analyzes and reports on the structure of the dataset.
    
    Args:
        samples_df: DataFrame containing NER samples
        lang_code: Language code
        output_dir: Output directory for saving analysis
    
    Returns:
        DataFrame with dataset statistics
    """
    print(f"\n--- Analyzing dataset structure for {lang_code} ---")
    
    # Basic statistics
    num_samples = len(samples_df)
    total_tokens = sum(len(tokens) for tokens in samples_df['tokens'])
    total_entities = sum(len(entities) for entities in samples_df['entities'])
    
    # Collect entity type statistics
    entity_types = []
    entity_lengths = []
    tokens_per_sample = []
    entities_per_sample = []
    
    for _, row in samples_df.iterrows():
        entities = row['entities']
        tokens = row['tokens']
        
        tokens_per_sample.append(len(tokens))
        entities_per_sample.append(len(entities))
        
        for entity in entities:
            entity_types.append(entity['entity_type'])
            # Calculate entity length in tokens
            entity_length = entity['end'] - entity['start'] + 1
            entity_lengths.append(entity_length)
    
    # Calculate statistics
    type_counts = Counter(entity_types)
    
    # Create statistics dictionary
    stats = {
        'language': lang_code,
        'num_samples': num_samples,
        'total_tokens': total_tokens,
        'total_entities': total_entities,
        'avg_tokens_per_sample': np.mean(tokens_per_sample),
        'avg_entities_per_sample': np.mean(entities_per_sample),
        'avg_entity_length': np.mean(entity_lengths) if entity_lengths else 0,
        'entity_type_distribution': dict(type_counts),
    }
    
    # Print summary
    print(f"Number of samples: {num_samples}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total entities: {total_entities}")
    print(f"Average tokens per sample: {stats['avg_tokens_per_sample']:.2f}")
    print(f"Average entities per sample: {stats['avg_entities_per_sample']:.2f}")
    print(f"Average entity length: {stats['avg_entity_length']:.2f}")
    print(f"Entity type distribution: {dict(type_counts)}")
    
    # Save analysis
    stats_df = pd.DataFrame([stats])
    stats_file = os.path.join(output_dir, 'analysis', f'{lang_code}_dataset_analysis.csv')
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    stats_df.to_csv(stats_file, index=False)
    
    # Plot entity type distribution
    plt.figure(figsize=(10, 6))
    plt.bar(type_counts.keys(), type_counts.values())
    plt.title(f'Entity Type Distribution - {lang_code}')
    plt.xlabel('Entity Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'analysis', f'{lang_code}_entity_type_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Plot entity length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(entity_lengths, bins=range(1, max(entity_lengths) + 2), alpha=0.7)
    plt.title(f'Entity Length Distribution - {lang_code}')
    plt.xlabel('Entity Length (tokens)')
    plt.ylabel('Count')
    plt.xticks(range(1, max(entity_lengths) + 1))
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'analysis', f'{lang_code}_entity_length_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    
    return stats_df

def single_label_baseline(samples_df, target_label=None):
    """
    Implements a single-label baseline for NER.
    Assigns one specific entity type to all gold entity spans.
    
    Args:
        samples_df: DataFrame containing NER samples
        target_label: The entity type to predict for all entities (None = automatically use most common)
        
    Returns:
        DataFrame with predictions and metrics
    """
    # If no target label is provided, find the most common entity type
    if target_label is None:
        all_entity_types = []
        for _, row in samples_df.iterrows():
            for entity in row['entities']:
                all_entity_types.append(entity['entity_type'])
        
        if all_entity_types:
            type_counts = Counter(all_entity_types)
            target_label = type_counts.most_common(1)[0][0]
        else:
            target_label = "PER"  # Default if no entities
    
    print(f"Single-label baseline predicting: {target_label} for all entity spans")
    
    results = []
    for _, row in samples_df.iterrows():
        gold_entities = row['entities']
        predicted_entities = []
        if gold_entities:
            for entity in gold_entities:
                # Keep gold span, predict target_label for all
                predicted_entities.append({
                    'start': entity['start'],
                    'end': entity['end'],
                    'entity_type': target_label
                })
        results.append(predicted_entities)
    
    results_df = samples_df.copy()
    results_df['predicted_entities'] = results
    results_df['approach'] = f'single_label_{target_label}'
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    return results_df, metrics

def majority_type_baseline(samples_df):
    """
    Implements a majority entity type baseline for NER.
    Determines the most frequent entity type in the training data (or falls back)
    and predicts that type for all identified entities.
    For simplicity here, we'll just predict 'O' (Outside) for all tokens implicitly,
    meaning zero predicted entities, which is a common simple baseline.

    Args:
        samples_df: DataFrame containing NER samples with 'tokens' and 'entities'

    Returns:
        DataFrame with predictions (empty list) and metrics
    """
    majority_prediction = [] # Predict no entities
    print(f"Majority baseline predicting: No entities (equivalent to predicting 'O' for all tokens)")

    results_df = samples_df.copy()
    results_df['predicted_entities'] = [majority_prediction for _ in range(len(results_df))]
    results_df['approach'] = 'majority_type'

    # Calculate metrics
    metrics = calculate_metrics(results_df)
    return results_df, metrics

def random_type_baseline(samples_df, seed=42):
    """
    Implements a random type baseline for NER.
    For each *gold* entity, randomly assign one of the possible types.

    Args:
        samples_df: DataFrame containing NER samples with 'tokens' and 'entities'
        seed: Random seed for reproducibility

    Returns:
        DataFrame with predictions and metrics
    """
    np.random.seed(seed)
    possible_types = ["PER", "LOC", "ORG", "DATE"]
    print(f"Random baseline assigning types from: {possible_types}")

    results = []
    for _, row in samples_df.iterrows():
        gold_entities = row['entities']
        predicted_entities = []
        if gold_entities:
            for entity in gold_entities:
                # Keep gold span, predict random type
                predicted_entities.append({
                    'start': entity['start'],
                    'end': entity['end'],
                    'entity_type': np.random.choice(possible_types)
                })
        results.append(predicted_entities)

    results_df = samples_df.copy()
    results_df['predicted_entities'] = results
    results_df['approach'] = 'random_type'

    # Calculate metrics
    metrics = calculate_metrics(results_df)
    return results_df, metrics

def calculate_metrics(results_df):
    """
    Calculate evaluation metrics (P/R/F1) for NER predictions.

    Args:
        results_df: DataFrame with predicted_entities and entities (gold)

    Returns:
        Dictionary with aggregated metrics
    """
    if results_df.empty or 'predicted_entities' not in results_df.columns or 'entities' not in results_df.columns:
        print("ERROR: Invalid results DataFrame for metric calculation.")
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

    # Implement our own metrics calculation instead of using NEREvaluator
    # This avoids the 'NEREvaluator' object has no attribute 'calculate_metrics' error
    
    total_tp = 0  # true positives
    total_fp = 0  # false positives
    total_fn = 0  # false negatives
    
    for _, row in results_df.iterrows():
        gold_entities = row['entities']
        pred_entities = row['predicted_entities']
        
        # Create sets of (start, end, entity_type) tuples for matching
        gold_set = {(e['start'], e['end'], e['entity_type']) for e in gold_entities}
        pred_set = {(e['start'], e['end'], e['entity_type']) for e in pred_entities}
        
        # Calculate exact matches (true positives)
        tp = len(gold_set.intersection(pred_set))
        
        # False positives: predicted but not in gold
        fp = len(pred_set) - tp
        
        # False negatives: in gold but not predicted
        fn = len(gold_set) - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate precision, recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    overall_metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }
    
    return overall_metrics

def plot_baseline_comparison(summary_df, lang_code, output_dir):
    """Plot comparison between simple baselines."""
    plt.figure(figsize=(10, 6))

    approaches = summary_df['approach'].tolist()
    f1_scores = summary_df['f1'].tolist()
    precisions = summary_df['precision'].tolist()
    recalls = summary_df['recall'].tolist()

    x = np.arange(len(approaches))
    width = 0.25

    plt.bar(x - width, f1_scores, width, label='F1 Score', color='skyblue')
    plt.bar(x, precisions, width, label='Precision', color='lightgreen')
    plt.bar(x + width, recalls, width, label='Recall', color='lightcoral')

    plt.xlabel('Approach')
    plt.ylabel('Score')
    plt.title(f'NER Simple Baseline Comparison - {lang_code}')
    plt.xticks(x, approaches, rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add score labels
    for i, val in enumerate(f1_scores):
        plt.text(i - width, val + 0.02, f'{val:.3f}', ha='center', fontsize=9)
    for i, val in enumerate(precisions):
        plt.text(i, val + 0.02, f'{val:.3f}', ha='center', fontsize=9)
    for i, val in enumerate(recalls):
        plt.text(i + width, val + 0.02, f'{val:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'plots', lang_code, f'ner_simple_theoretical_baselines_comparison_{lang_code}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

def run_simple_ner_baselines(samples_df, lang_code, output_dir, seed=42):
    """
    Run all simple theoretical baselines on the given NER dataset.
    """
    plots_dir = os.path.join(output_dir, 'plots', lang_code)
    results_dir = os.path.join(output_dir, 'results', lang_code)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    summary_metrics = []

    print(f"\n--- Running simple NER theoretical baselines for {lang_code} ---")

    # Ensure required columns exist
    if 'tokens' not in samples_df.columns or 'entities' not in samples_df.columns:
         print(f"ERROR: Loaded data for {lang_code} is missing 'tokens' or 'entities'. Skipping.")
         return pd.DataFrame(), pd.DataFrame()

    # First analyze dataset structure
    analyze_dataset_structure(samples_df, lang_code, output_dir)

    # 1. Majority Type Baseline (Predict 'O')
    print("Running majority type baseline...")
    majority_results, majority_metrics = majority_type_baseline(samples_df)
    all_results.append(majority_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'majority_type',
        **majority_metrics
    })

    # 2. Random Type Baseline
    print("Running random type baseline...")
    random_results, random_metrics = random_type_baseline(samples_df, seed)
    all_results.append(random_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'random_type',
        **random_metrics
    })
    
    # 3. Single-label baselines for each entity type
    # Get all unique entity types in the dataset
    all_entity_types = set()
    for _, row in samples_df.iterrows():
        for entity in row['entities']:
            all_entity_types.add(entity['entity_type'])
    
    for entity_type in all_entity_types:
        print(f"Running single-label baseline for type: {entity_type}...")
        single_label_results, single_label_metrics = single_label_baseline(samples_df, entity_type)
        all_results.append(single_label_results)
        summary_metrics.append({
            'language': lang_code,
            'approach': f'single_label_{entity_type}',
            **single_label_metrics
    })

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(os.path.join(results_dir, f'ner_simple_theoretical_baselines_{lang_code}.csv'), index=False)

    summary_df = pd.DataFrame(summary_metrics)
    summary_df.to_csv(os.path.join(output_dir, 'summaries', f'ner_simple_theoretical_baselines_summary_{lang_code}.csv'), index=False)

    plot_baseline_comparison(summary_df, lang_code, output_dir)

    print(f"\nSummary of NER simple theoretical baselines for {lang_code}:")
    print(summary_df.to_string(index=False))

    return combined_results, summary_df

def create_theoretical_bounds_report(summaries, output_dir):
    """
    Create a comprehensive report on theoretical bounds for the NER task.
    
    Args:
        summaries: Dictionary mapping language codes to summary DataFrames
        output_dir: Output directory for saving the report
    """
    print("\n--- Creating Theoretical Bounds Report ---")
    
    report = []
    
    # For each language, compile min/max metrics
    for lang_code, summary_df in summaries.items():
        min_f1 = summary_df['f1'].min()
        max_f1 = summary_df['f1'].max()
        
        min_precision = summary_df['precision'].min()
        max_precision = summary_df['precision'].max()
        
        min_recall = summary_df['recall'].min()
        max_recall = summary_df['recall'].max()
        
        worst_approach = summary_df.loc[summary_df['f1'].idxmin(), 'approach']
        best_approach = summary_df.loc[summary_df['f1'].idxmax(), 'approach']
        
        report.append({
            'language': lang_code,
            'min_f1': min_f1,
            'max_f1': max_f1,
            'min_precision': min_precision,
            'max_precision': max_precision,
            'min_recall': min_recall,
            'max_recall': max_recall,
            'worst_approach': worst_approach,
            'best_approach': best_approach,
            'theoretical_range': max_f1 - min_f1
        })
    
    report_df = pd.DataFrame(report)
    report_path = os.path.join(output_dir, 'theoretical_bounds_report.csv')
    report_df.to_csv(report_path, index=False)
    
    print("\nTheoretical Bounds Report:")
    print(report_df.to_string(index=False))
    
    # Create a plot comparing theoretical bounds across languages
    if len(summaries) > 1:
        plt.figure(figsize=(12, 8))
        
        languages = report_df['language'].tolist()
        x = np.arange(len(languages))
        width = 0.35
        
        min_f1s = report_df['min_f1'].tolist()
        max_f1s = report_df['max_f1'].tolist()
        
        plt.bar(x - width/2, min_f1s, width, label='Min F1', color='lightcoral')
        plt.bar(x + width/2, max_f1s, width, label='Max F1', color='lightgreen')
        
        plt.xlabel('Language')
        plt.ylabel('F1 Score')
        plt.title('NER Theoretical Performance Bounds by Language')
        plt.xticks(x, languages)
        plt.ylim(0, 1.0)
        plt.legend()
        
        # Add labels on bars
        for i, (min_val, max_val) in enumerate(zip(min_f1s, max_f1s)):
            plt.text(i - width/2, min_val + 0.02, f'{min_val:.3f}', ha='center', fontsize=9)
            plt.text(i + width/2, max_val + 0.02, f'{max_val:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'theoretical_bounds_comparison.png')
        plt.savefig(plot_path)
        plt.close()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output
    summaries_dir = os.path.join(output_dir, 'summaries')
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    all_language_summaries = {}

    for lang_code in args.langs:
        print(f"\n=== Processing {lang_code} ===")
        # Load data
        samples_df_full = load_masakhaner_samples(lang_code, num_samples=None, split=args.split, seed=args.seed)

        if samples_df_full.empty:
            print(f"WARNING: No samples loaded for {lang_code}. Skipping.")
            continue

        # Sample data
        actual_loaded = len(samples_df_full)
        if args.samples is not None and actual_loaded > args.samples:
            print(f"Sampling {args.samples} from {actual_loaded} samples...")
            samples_df = samples_df_full.sample(n=args.samples, random_state=args.seed)
        else:
            samples_df = samples_df_full
            print(f"Using all {actual_loaded} samples loaded.")

        # Run baselines
        _, summary_df = run_simple_ner_baselines(samples_df, lang_code, output_dir, args.seed)
        all_language_summaries[lang_code] = summary_df
    
    # Create theoretical bounds report comparing across languages
    if all_language_summaries:
        create_theoretical_bounds_report(all_language_summaries, output_dir)
    
    print(f"\nAll results saved to: {output_dir}")

if __name__ == "__main__":
    main() 