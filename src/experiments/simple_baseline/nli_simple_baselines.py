import sys
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
import time

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Import utility functions
from src.utils.data_loaders.load_afrinli import load_afrinli_samples

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run NLI simple baselines")
    parser.add_argument("--langs", nargs='+', default=['en', 'sw', 'ur'], 
                        help="Languages to test (default: en, sw, ur)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples to use per language (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

def majority_baseline(samples_df):
    """
    Implements a majority class baseline.
    Always predicts the most common label in the dataset.
    
    Args:
        samples_df: DataFrame containing NLI samples
        
    Returns:
        DataFrame with predictions and metrics
    """
    # Count label distribution and find the majority class
    label_counts = samples_df['label'].value_counts()
    majority_label = label_counts.index[0]
    
    print(f"Majority class: {majority_label} ({label_counts[majority_label]} of {len(samples_df)} samples, {label_counts[majority_label]/len(samples_df):.2%})")
    
    # Create a copy of the input DataFrame
    results_df = samples_df.copy()
    
    # Add predictions using majority class
    results_df['predicted_label'] = majority_label
    results_df['approach'] = 'majority'
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    return results_df, metrics

def random_baseline(samples_df, seed=42):
    """
    Implements a random baseline.
    Randomly selects between the three labels with equal probability.
    
    Args:
        samples_df: DataFrame containing NLI samples
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with predictions and metrics
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Possible NLI labels
    nli_labels = ['entailment', 'neutral', 'contradiction']
    
    # Create a copy of the input DataFrame
    results_df = samples_df.copy()
    
    # Add random predictions
    results_df['predicted_label'] = np.random.choice(nli_labels, size=len(results_df))
    results_df['approach'] = 'random'
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    return results_df, metrics

def stratified_random_baseline(samples_df, seed=42):
    """
    Implements a stratified random baseline.
    Samples predictions based on the label distribution in the dataset.
    
    Args:
        samples_df: DataFrame containing NLI samples
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with predictions and metrics
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Count label distribution
    label_counts = samples_df['label'].value_counts(normalize=True)
    labels = label_counts.index.tolist()
    proportions = label_counts.values
    
    print(f"Label distribution: {dict(zip(labels, proportions))}")
    
    # Create a copy of the input DataFrame
    results_df = samples_df.copy()
    
    # Generate predictions based on label distribution
    results_df['predicted_label'] = np.random.choice(labels, size=len(results_df), p=proportions)
    results_df['approach'] = 'stratified'
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    return results_df, metrics

def lexical_overlap_baseline(samples_df):
    """
    Implements a lexical overlap baseline.
    Predicts 'entailment' if there's high word overlap between premise and hypothesis,
    'contradiction' if there's low overlap, and 'neutral' otherwise.
    
    Args:
        samples_df: DataFrame containing NLI samples
        
    Returns:
        DataFrame with predictions and metrics
    """
    # Create a copy of the input DataFrame
    results_df = samples_df.copy()
    
    # Calculate lexical overlap for each sample
    def get_overlap_ratio(premise, hypothesis):
        premise_words = set(premise.lower().split())
        hypothesis_words = set(hypothesis.lower().split())
        
        if len(hypothesis_words) == 0:
            return 0.0
            
        common_words = premise_words.intersection(hypothesis_words)
        return len(common_words) / len(hypothesis_words)
    
    results_df['overlap_ratio'] = results_df.apply(
        lambda row: get_overlap_ratio(row['premise'], row['hypothesis']), axis=1
    )
    
    # Set thresholds for class assignment
    # These thresholds can be tuned based on dataset characteristics
    high_threshold = 0.6
    low_threshold = 0.2
    
    # Assign labels based on overlap ratio
    def assign_label(ratio):
        if ratio >= high_threshold:
            return 'entailment'
        elif ratio <= low_threshold:
            return 'contradiction'
        else:
            return 'neutral'
    
    results_df['predicted_label'] = results_df['overlap_ratio'].apply(assign_label)
    results_df['approach'] = 'lexical_overlap'
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    return results_df, metrics

def length_ratio_baseline(samples_df):
    """
    Implements a length ratio baseline.
    Predicts based on the length ratio between premise and hypothesis.
    
    Args:
        samples_df: DataFrame containing NLI samples
        
    Returns:
        DataFrame with predictions and metrics
    """
    # Create a copy of the input DataFrame
    results_df = samples_df.copy()
    
    # Calculate length ratio (hypothesis / premise)
    def get_length_ratio(premise, hypothesis):
        if len(premise) == 0:
            return float('inf')
        return len(hypothesis) / len(premise)
    
    results_df['length_ratio'] = results_df.apply(
        lambda row: get_length_ratio(row['premise'], row['hypothesis']), axis=1
    )
    
    # Set thresholds for class assignment
    # Very short hypothesis compared to premise often indicates contradiction
    # Very long hypothesis compared to premise may indicate neutrality
    # Similar lengths might indicate entailment
    def assign_label(ratio):
        if ratio < 0.5:
            return 'contradiction'  # Hypothesis is much shorter
        elif ratio > 1.5:
            return 'neutral'  # Hypothesis is much longer
        else:
            return 'entailment'  # Similar length
    
    results_df['predicted_label'] = results_df['length_ratio'].apply(assign_label)
    results_df['approach'] = 'length_ratio'
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    return results_df, metrics

def calculate_metrics(results_df):
    """
    Calculate evaluation metrics for NLI predictions.
    
    Args:
        results_df: DataFrame with predicted and gold labels
        
    Returns:
        Dictionary with metrics
    """
    # Make sure we have valid data
    if results_df.empty or 'predicted_label' not in results_df.columns or 'label' not in results_df.columns:
        print("ERROR: Invalid results DataFrame")
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'class_metrics': {},
            'confusion_matrix': None
        }
    
    # Convert numeric labels to strings if needed
    eval_df = results_df.copy()
    
    # Ensure label types are consistent
    label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction',
                 '0': 'entailment', '1': 'neutral', '2': 'contradiction'}
    
    # Check if we need to convert numeric labels
    if pd.api.types.is_numeric_dtype(eval_df['label']):
        eval_df['label'] = eval_df['label'].map(lambda x: label_map[x])
    
    # Also ensure predicted_label is in the same format
    if pd.api.types.is_numeric_dtype(eval_df['predicted_label']):
        eval_df['predicted_label'] = eval_df['predicted_label'].map(lambda x: label_map[x])
    
    # Convert string numbers to string labels if needed
    if eval_df['label'].dtype == object:
        eval_df['label'] = eval_df['label'].apply(
            lambda x: label_map.get(x, x) if x in label_map else x
        )
    
    if eval_df['predicted_label'].dtype == object:
        eval_df['predicted_label'] = eval_df['predicted_label'].apply(
            lambda x: label_map.get(x, x) if x in label_map else x
        )
    
    # Calculate accuracy
    accuracy = accuracy_score(eval_df['label'], eval_df['predicted_label'])
    
    # Calculate macro F1 score
    macro_f1 = f1_score(
        eval_df['label'],
        eval_df['predicted_label'],
        average='macro',
        zero_division=0
    )
    
    # Get detailed metrics for each class
    class_report = classification_report(
        eval_df['label'],
        eval_df['predicted_label'],
        output_dict=True,
        zero_division=0
    )
    
    # Extract class metrics
    class_metrics = {}
    for label in ['entailment', 'neutral', 'contradiction']:
        if label in class_report:
            class_metrics[label] = {
                'precision': class_report[label]['precision'],
                'recall': class_report[label]['recall'],
                'f1': class_report[label]['f1-score']
            }
    
    # Calculate confusion matrix
    cm = confusion_matrix(eval_df['label'], eval_df['predicted_label'], labels=['entailment', 'neutral', 'contradiction'])
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'class_metrics': class_metrics,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, labels, title, output_path):
    """
    Plot and save a confusion matrix.
    
    Args:
        cm: Confusion matrix
        labels: Class labels
        title: Plot title
        output_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_dataset_distribution(samples_df, lang_code, output_path):
    """
    Plot and save the dataset label distribution.
    
    Args:
        samples_df: DataFrame containing NLI samples
        lang_code: Language code
        output_path: Path to save the plot
    """
    # Count label distribution
    label_counts = samples_df['label'].value_counts().sort_index()
    
    # Convert numeric labels to strings if needed
    if pd.api.types.is_numeric_dtype(label_counts.index):
        label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        label_counts.index = [label_map.get(i, i) for i in label_counts.index]
    
    plt.figure(figsize=(10, 6))
    ax = label_counts.plot(kind='bar', color='skyblue')
    plt.title(f'NLI Label Distribution for {lang_code}')
    plt.xlabel('Label')
    plt.ylabel('Count')
    
    # Add count and percentage labels on bars
    total = len(samples_df)
    for i, count in enumerate(label_counts):
        percentage = 100 * count / total
        ax.text(i, count + 0.1, f'{count}\n({percentage:.1f}%)', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_simple_baselines(samples_df, lang_code, output_dir, seed=42):
    """
    Run all simple baselines on the given dataset.
    
    Args:
        samples_df: DataFrame containing NLI samples
        lang_code: Language code
        output_dir: Directory to save results and plots
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with combined results
    """
    # Create output directories
    plots_dir = os.path.join(output_dir, 'plots', lang_code)
    results_dir = os.path.join(output_dir, 'results', lang_code)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot dataset distribution
    plot_dataset_distribution(
        samples_df,
        lang_code,
        os.path.join(plots_dir, f'nli_label_distribution_{lang_code}.png')
    )
    
    # Initialize combined results
    all_results = []
    summary_metrics = []
    
    # Run all baselines
    print(f"\n--- Running simple baselines for {lang_code} ---")
    
    # 1. Majority class baseline
    print("\nRunning majority class baseline...")
    majority_results, majority_metrics = majority_baseline(samples_df)
    all_results.append(majority_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'majority',
        'accuracy': majority_metrics['accuracy'],
        'macro_f1': majority_metrics['macro_f1']
    })
    
    # Plot confusion matrix
    plot_confusion_matrix(
        majority_metrics['confusion_matrix'],
        ['entailment', 'neutral', 'contradiction'],
        f'Majority Baseline Confusion Matrix - {lang_code}',
        os.path.join(plots_dir, f'nli_majority_cm_{lang_code}.png')
    )
    
    # 2. Random baseline
    print("\nRunning random baseline...")
    random_results, random_metrics = random_baseline(samples_df, seed)
    all_results.append(random_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'random',
        'accuracy': random_metrics['accuracy'],
        'macro_f1': random_metrics['macro_f1']
    })
    
    # Plot confusion matrix
    plot_confusion_matrix(
        random_metrics['confusion_matrix'],
        ['entailment', 'neutral', 'contradiction'],
        f'Random Baseline Confusion Matrix - {lang_code}',
        os.path.join(plots_dir, f'nli_random_cm_{lang_code}.png')
    )
    
    # 3. Stratified random baseline
    print("\nRunning stratified random baseline...")
    stratified_results, stratified_metrics = stratified_random_baseline(samples_df, seed)
    all_results.append(stratified_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'stratified',
        'accuracy': stratified_metrics['accuracy'],
        'macro_f1': stratified_metrics['macro_f1']
    })
    
    # Plot confusion matrix
    plot_confusion_matrix(
        stratified_metrics['confusion_matrix'],
        ['entailment', 'neutral', 'contradiction'],
        f'Stratified Random Baseline Confusion Matrix - {lang_code}',
        os.path.join(plots_dir, f'nli_stratified_cm_{lang_code}.png')
    )
    
    # 4. Lexical overlap baseline
    print("\nRunning lexical overlap baseline...")
    overlap_results, overlap_metrics = lexical_overlap_baseline(samples_df)
    all_results.append(overlap_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'lexical_overlap',
        'accuracy': overlap_metrics['accuracy'],
        'macro_f1': overlap_metrics['macro_f1']
    })
    
    # Plot confusion matrix
    plot_confusion_matrix(
        overlap_metrics['confusion_matrix'],
        ['entailment', 'neutral', 'contradiction'],
        f'Lexical Overlap Baseline Confusion Matrix - {lang_code}',
        os.path.join(plots_dir, f'nli_lexical_overlap_cm_{lang_code}.png')
    )
    
    # 5. Length ratio baseline
    print("\nRunning length ratio baseline...")
    length_results, length_metrics = length_ratio_baseline(samples_df)
    all_results.append(length_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'length_ratio',
        'accuracy': length_metrics['accuracy'],
        'macro_f1': length_metrics['macro_f1']
    })
    
    # Plot confusion matrix
    plot_confusion_matrix(
        length_metrics['confusion_matrix'],
        ['entailment', 'neutral', 'contradiction'],
        f'Length Ratio Baseline Confusion Matrix - {lang_code}',
        os.path.join(plots_dir, f'nli_length_ratio_cm_{lang_code}.png')
    )
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    combined_results.to_csv(os.path.join(results_dir, f'nli_simple_baselines_{lang_code}.csv'), index=False)
    
    # Create and save summary metrics
    summary_df = pd.DataFrame(summary_metrics)
    summary_df.to_csv(os.path.join(output_dir, 'summaries', f'nli_simple_baselines_summary_{lang_code}.csv'), index=False)
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    
    approaches = summary_df['approach'].tolist()
    accuracies = summary_df['accuracy'].tolist()
    macro_f1s = summary_df['macro_f1'].tolist()
    
    x = np.arange(len(approaches))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
    plt.bar(x + width/2, macro_f1s, width, label='Macro F1', color='lightcoral')
    
    plt.xlabel('Approach')
    plt.ylabel('Score')
    plt.title(f'NLI Simple Baselines Performance Comparison - {lang_code}')
    plt.xticks(x, approaches)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add score labels on bars
    for i, acc in enumerate(accuracies):
        plt.text(i - width/2, acc + 0.02, f'{acc:.3f}', ha='center')
    
    for i, f1 in enumerate(macro_f1s):
        plt.text(i + width/2, f1 + 0.02, f'{f1:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'nli_baselines_comparison_{lang_code}.png'))
    plt.close()
    
    # Print summary
    print(f"\nSummary of NLI simple baselines for {lang_code}:")
    print(summary_df.to_string(index=False))
    
    return combined_results, summary_df

def main():
    """Main function to run all simple baselines."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directories
    output_dir = "/work/bbd6522/results/nli/simple_baselines"
    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    
    # Store all summary results for comparison
    all_summaries = []
    
    # Process each language
    for lang_code in args.langs:
        print(f"\n=== Processing {lang_code} ===")
        
        # Load samples
        full_samples_df = load_afrinli_samples(lang_code, num_samples=None, split='test')
        
        if full_samples_df.empty:
            print(f"WARNING: No samples loaded for {lang_code}. Skipping.")
            continue
        
        total_loaded = len(full_samples_df)
        print(f"Loaded {total_loaded} samples for {lang_code}.")
        
        # Sample if requested
        if args.samples is not None:
            num_to_sample = min(total_loaded, args.samples)
            print(f"Sampling {num_to_sample} samples...")
            samples_df = full_samples_df.sample(n=num_to_sample, random_state=args.seed)
        else:
            samples_df = full_samples_df
        
        # Run all simple baselines
        _, summary_df = run_simple_baselines(samples_df, lang_code, output_dir, args.seed)
        all_summaries.append(summary_df)
    
    # Combine all summaries for cross-language comparison
    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_summary.to_csv(os.path.join(summaries_dir, 'nli_simple_baselines_all_languages.csv'), index=False)
        
        # Create cross-language comparison plot
        plt.figure(figsize=(15, 8))
        
        # Prepare data for grouped bar chart
        approaches = combined_summary['approach'].unique()
        languages = combined_summary['language'].unique()
        
        x = np.arange(len(approaches))
        width = 0.8 / len(languages)
        
        for i, lang in enumerate(languages):
            lang_data = combined_summary[combined_summary['language'] == lang]
            lang_acc = [lang_data[lang_data['approach'] == app]['accuracy'].values[0] for app in approaches]
            
            plt.bar(x + (i - len(languages)/2 + 0.5) * width, lang_acc, width, label=lang)
        
        plt.xlabel('Approach')
        plt.ylabel('Accuracy')
        plt.title('NLI Simple Baselines Performance Across Languages')
        plt.xticks(x, approaches)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plots', 'nli_baselines_cross_language_comparison.png'))
        plt.close()
        
        print("\nCross-language comparison summary:")
        print(combined_summary.pivot_table(
            index='approach', columns='language', values=['accuracy', 'macro_f1']
        ))

if __name__ == "__main__":
    main() 