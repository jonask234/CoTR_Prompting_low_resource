import sys
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
import re
import time
import string

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Import utility functions
from src.utils.data_loaders.load_masakhaner import load_masakhaner_samples

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run NER simple baselines")
    parser.add_argument("--langs", nargs='+', default=['sw', 'yo'], 
                        help="Languages to test (default: sw, yo)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples to use per language (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

def format_ner_sample(text, entities=None):
    """
    Format a NER sample for display or evaluation
    
    Args:
        text: Text content with tokens
        entities: List of entity annotations (optional)
    
    Returns:
        Formatted string with highlighted entities
    """
    if entities is None or len(entities) == 0:
        return text
    
    # Convert text to tokens if it's a string
    tokens = text if isinstance(text, list) else text.split()
    
    result = []
    current_entity = None
    entity_start = -1
    
    for i, token in enumerate(tokens):
        # Check if token starts an entity
        entity_match = None
        for entity in entities:
            if entity['start'] == i:
                entity_match = entity
                break
        
        if entity_match:
            current_entity = entity_match
            entity_start = i
            result.append(f"[{current_entity['entity_type']}: {token}")
        elif current_entity and i == entity_start + (current_entity['end'] - current_entity['start']):
            # End of entity
            result.append(f"{token}]")
            current_entity = None
        else:
            result.append(token)
    
    return " ".join(result)

def majority_baseline(samples_df):
    """
    Majority baseline for NER.
    Always predicts the most common entity type for each entity span.
    
    Args:
        samples_df: DataFrame containing NER samples
        
    Returns:
        DataFrame with predictions and metrics
    """
    # Create a copy of the input DataFrame
    results_df = samples_df.copy()
    
    # Count entity types in the training data
    entity_counts = Counter()
    for _, row in samples_df.iterrows():
        for entity in row['entities']:
            entity_counts[entity['entity_type']] += 1
    
    # Find the most common entity type
    if entity_counts:
        majority_entity = entity_counts.most_common(1)[0][0]
    else:
        majority_entity = "O"  # Outside/non-entity as default
    
    print(f"Most common entity type: {majority_entity} (appeared {entity_counts[majority_entity]} times)")
    
    # Create a predicted entities column that simply assigns the majority entity type
    # to all detected entity spans (keeping the same spans as ground truth)
    def predict_majority_entities(row):
        # Copy ground truth entities but change all types to the majority type
        predicted_entities = []
        for entity in row['entities']:
            predicted_entity = entity.copy()
            predicted_entity['entity_type'] = majority_entity
            predicted_entities.append(predicted_entity)
        return predicted_entities
    
    results_df['predicted_entities'] = results_df.apply(predict_majority_entities, axis=1)
    results_df['approach'] = 'majority'
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    return results_df, metrics

def random_baseline(samples_df):
    """
    Random baseline for NER.
    Randomly assigns entity types from the observed set to each entity span.
    
    Args:
        samples_df: DataFrame containing NER samples
        
    Returns:
        DataFrame with predictions and metrics
    """
    # Create a copy of the input DataFrame
    results_df = samples_df.copy()
    
    # Collect all unique entity types from the dataset
    entity_types = set()
    for _, row in samples_df.iterrows():
        for entity in row['entities']:
            entity_types.add(entity['entity_type'])
    
    entity_types = list(entity_types)
    if not entity_types:
        entity_types = ["O"]  # Default if no entities
    
    # Function to assign random entity types
    def predict_random_entities(row):
        predicted_entities = []
        for entity in row['entities']:
            # Keep the same span but assign random entity type
            predicted_entity = entity.copy()
            predicted_entity['entity_type'] = random.choice(entity_types)
            predicted_entities.append(predicted_entity)
        return predicted_entities
    
    results_df['predicted_entities'] = results_df.apply(predict_random_entities, axis=1)
    results_df['approach'] = 'random'
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    return results_df, metrics

def length_based_baseline(samples_df):
    """
    Length-based baseline for NER.
    Assigns entity types based on the length of the entity span.
    Shorter spans tend to be certain entity types (like people) while
    longer spans might be organizations or locations.
    
    Args:
        samples_df: DataFrame containing NER samples
        
    Returns:
        DataFrame with predictions and metrics
    """
    # Create a copy of the input DataFrame
    results_df = samples_df.copy()
    
    # Collect entity type statistics by length
    entity_length_stats = {}
    for _, row in samples_df.iterrows():
        for entity in row['entities']:
            length = entity['end'] - entity['start']
            if length not in entity_length_stats:
                entity_length_stats[length] = Counter()
            entity_length_stats[length][entity['entity_type']] += 1
    
    # Map length to most common entity type
    length_to_entity = {}
    for length, counts in entity_length_stats.items():
        if counts:
            length_to_entity[length] = counts.most_common(1)[0][0]
    
    # Use a default entity type for unseen lengths
    all_entities = [entity for row in samples_df.iterrows() for entity in row[1]['entities']]
    if all_entities:
        default_entity = Counter([entity['entity_type'] for entity in all_entities]).most_common(1)[0][0]
    else:
        default_entity = "O"
    
    # Function to predict based on entity length
    def predict_length_based_entities(row):
        predicted_entities = []
        for entity in row['entities']:
            length = entity['end'] - entity['start']
            # Keep the same span but assign length-based entity type
            predicted_entity = entity.copy()
            predicted_entity['entity_type'] = length_to_entity.get(length, default_entity)
            predicted_entities.append(predicted_entity)
        return predicted_entities
    
    results_df['predicted_entities'] = results_df.apply(predict_length_based_entities, axis=1)
    results_df['approach'] = 'length_based'
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    return results_df, metrics

def position_based_baseline(samples_df):
    """
    Position-based baseline for NER.
    Assigns entity types based on the position in the text.
    Entities at the beginning, middle, or end might have different distributions.
    
    Args:
        samples_df: DataFrame containing NER samples
        
    Returns:
        DataFrame with predictions and metrics
    """
    # Create a copy of the input DataFrame
    results_df = samples_df.copy()
    
    # Function to categorize position (beginning, middle, end)
    def get_position_category(start, end, text_length):
        if start < text_length * 0.33:
            return "beginning"
        elif start < text_length * 0.66:
            return "middle"
        else:
            return "end"
    
    # Collect entity type statistics by position
    position_stats = {"beginning": Counter(), "middle": Counter(), "end": Counter()}
    for _, row in samples_df.iterrows():
        tokens = row['tokens'] if isinstance(row['tokens'], list) else row['tokens'].split()
        text_length = len(tokens)
        
        for entity in row['entities']:
            position = get_position_category(entity['start'], entity['end'], text_length)
            position_stats[position][entity['entity_type']] += 1
    
    # Map position to most common entity type
    position_to_entity = {}
    for position, counts in position_stats.items():
        if counts:
            position_to_entity[position] = counts.most_common(1)[0][0]
        else:
            position_to_entity[position] = "O"
    
    # Function to predict based on entity position
    def predict_position_based_entities(row):
        tokens = row['tokens'] if isinstance(row['tokens'], list) else row['tokens'].split()
        text_length = len(tokens)
        
        predicted_entities = []
        for entity in row['entities']:
            position = get_position_category(entity['start'], entity['end'], text_length)
            # Keep the same span but assign position-based entity type
            predicted_entity = entity.copy()
            predicted_entity['entity_type'] = position_to_entity[position]
            predicted_entities.append(predicted_entity)
        return predicted_entities
    
    results_df['predicted_entities'] = results_df.apply(predict_position_based_entities, axis=1)
    results_df['approach'] = 'position_based'
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    return results_df, metrics

def first_word_baseline(samples_df):
    """
    First-word baseline for NER.
    Assigns entity types based on the first word of the entity.
    Some entities often start with specific words (e.g., "Mr." for person).
    
    Args:
        samples_df: DataFrame containing NER samples
        
    Returns:
        DataFrame with predictions and metrics
    """
    # Create a copy of the input DataFrame
    results_df = samples_df.copy()
    
    # Collect entity type statistics by first word
    first_word_stats = {}
    for _, row in samples_df.iterrows():
        tokens = row['tokens'] if isinstance(row['tokens'], list) else row['tokens'].split()
        
        for entity in row['entities']:
            if entity['start'] < len(tokens):
                first_word = tokens[entity['start']].lower()
                if first_word not in first_word_stats:
                    first_word_stats[first_word] = Counter()
                first_word_stats[first_word][entity['entity_type']] += 1
    
    # Map first word to most common entity type
    first_word_to_entity = {}
    for word, counts in first_word_stats.items():
        if counts:
            first_word_to_entity[word] = counts.most_common(1)[0][0]
    
    # Use default entity type for unseen first words
    all_entities = [entity for row in samples_df.iterrows() for entity in row[1]['entities']]
    if all_entities:
        default_entity = Counter([entity['entity_type'] for entity in all_entities]).most_common(1)[0][0]
    else:
        default_entity = "O"
    
    # Function to predict based on first word
    def predict_first_word_entities(row):
        tokens = row['tokens'] if isinstance(row['tokens'], list) else row['tokens'].split()
        
        predicted_entities = []
        for entity in row['entities']:
            # Keep the same span but assign first-word-based entity type
            predicted_entity = entity.copy()
            
            if entity['start'] < len(tokens):
                first_word = tokens[entity['start']].lower()
                predicted_entity['entity_type'] = first_word_to_entity.get(first_word, default_entity)
            else:
                predicted_entity['entity_type'] = default_entity
                
            predicted_entities.append(predicted_entity)
        return predicted_entities
    
    results_df['predicted_entities'] = results_df.apply(predict_first_word_entities, axis=1)
    results_df['approach'] = 'first_word'
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    return results_df, metrics

def calculate_metrics(results_df):
    """
    Calculate evaluation metrics for NER predictions.
    
    Args:
        results_df: DataFrame with predicted and gold entity annotations
        
    Returns:
        Dictionary with metrics
    """
    # Initialize counters for overall metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Counters for entity-type specific metrics
    entity_metrics = {}
    
    for _, row in results_df.iterrows():
        gold_entities = row['entities']
        pred_entities = row['predicted_entities']
        
        # Create sets for exact matching (entity span + type)
        gold_set = {(e['start'], e['end'], e['entity_type']) for e in gold_entities}
        pred_set = {(e['start'], e['end'], e['entity_type']) for e in pred_entities}
        
        # Count matches and errors
        matches = gold_set.intersection(pred_set)
        true_positives += len(matches)
        false_positives += len(pred_set - gold_set)
        false_negatives += len(gold_set - pred_set)
        
        # Entity-type specific metrics
        for gold_entity in gold_entities:
            entity_type = gold_entity['entity_type']
            if entity_type not in entity_metrics:
                entity_metrics[entity_type] = {'tp': 0, 'fp': 0, 'fn': 0}
            
            gold_item = (gold_entity['start'], gold_entity['end'], entity_type)
            if gold_item in matches:
                entity_metrics[entity_type]['tp'] += 1
            else:
                entity_metrics[entity_type]['fn'] += 1
        
        for pred_entity in pred_entities:
            entity_type = pred_entity['entity_type']
            if entity_type not in entity_metrics:
                entity_metrics[entity_type] = {'tp': 0, 'fp': 0, 'fn': 0}
            
            pred_item = (pred_entity['start'], pred_entity['end'], entity_type)
            if pred_item not in matches and pred_item not in gold_set:
                entity_metrics[entity_type]['fp'] += 1
    
    # Calculate overall precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate per-entity metrics
    entity_scores = {}
    for entity_type, counts in entity_metrics.items():
        entity_precision = counts['tp'] / (counts['tp'] + counts['fp']) if (counts['tp'] + counts['fp']) > 0 else 0
        entity_recall = counts['tp'] / (counts['tp'] + counts['fn']) if (counts['tp'] + counts['fn']) > 0 else 0
        entity_f1 = 2 * (entity_precision * entity_recall) / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0
        
        entity_scores[entity_type] = {
            'precision': entity_precision,
            'recall': entity_recall,
            'f1': entity_f1,
            'support': counts['tp'] + counts['fn']
        }
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'entity_scores': entity_scores
    }

def plot_dataset_stats(samples_df, lang_code, output_path):
    """
    Plot statistics about the NER dataset.
    
    Args:
        samples_df: DataFrame containing NER samples
        lang_code: Language code
        output_path: Path to save the plot
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Entity type distribution
    entity_types = []
    for _, row in samples_df.iterrows():
        for entity in row['entities']:
            entity_types.append(entity['entity_type'])
    
    entity_counts = Counter(entity_types)
    
    axs[0, 0].bar(entity_counts.keys(), entity_counts.values(), color='skyblue')
    axs[0, 0].set_title(f'Entity Type Distribution - {lang_code}')
    axs[0, 0].set_xlabel('Entity Type')
    axs[0, 0].set_ylabel('Count')
    plt.setp(axs[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Entity length distribution
    entity_lengths = []
    for _, row in samples_df.iterrows():
        for entity in row['entities']:
            entity_lengths.append(entity['end'] - entity['start'])
    
    axs[0, 1].hist(entity_lengths, bins=range(1, max(entity_lengths) + 2), color='lightgreen')
    axs[0, 1].set_title(f'Entity Length Distribution - {lang_code}')
    axs[0, 1].set_xlabel('Number of Tokens')
    axs[0, 1].set_ylabel('Count')
    
    # 3. Sentence length distribution
    sentence_lengths = samples_df['tokens'].apply(lambda x: len(x) if isinstance(x, list) else len(x.split()))
    
    axs[1, 0].hist(sentence_lengths, bins=30, color='salmon')
    axs[1, 0].set_title(f'Sentence Length Distribution - {lang_code}')
    axs[1, 0].set_xlabel('Number of Tokens')
    axs[1, 0].set_ylabel('Count')
    
    # 4. Entities per sentence distribution
    entities_per_sentence = samples_df['entities'].apply(len)
    
    axs[1, 1].hist(entities_per_sentence, bins=range(0, max(entities_per_sentence) + 2), color='plum')
    axs[1, 1].set_title(f'Entities per Sentence - {lang_code}')
    axs[1, 1].set_xlabel('Number of Entities')
    axs[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_entity_distribution_by_position(samples_df, lang_code, output_path):
    """
    Plot entity type distribution by position in text.
    
    Args:
        samples_df: DataFrame containing NER samples
        lang_code: Language code
        output_path: Path to save the plot
    """
    # Collect entity positions and types
    entity_positions = []
    entity_types = []
    
    for _, row in samples_df.iterrows():
        tokens = row['tokens'] if isinstance(row['tokens'], list) else row['tokens'].split()
        text_length = len(tokens)
        
        for entity in row['entities']:
            position = entity['start'] / text_length if text_length > 0 else 0
            entity_positions.append(position)
            entity_types.append(entity['entity_type'])
    
    # Create DataFrame for plotting
    pos_df = pd.DataFrame({
        'Position': entity_positions,
        'EntityType': entity_types
    })
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Use violin plot to show distribution
    ax = sns.violinplot(x='EntityType', y='Position', data=pos_df, inner='quartile')
    
    plt.title(f'Entity Type Distribution by Position - {lang_code}')
    plt.xlabel('Entity Type')
    plt.ylabel('Relative Position in Text (0=start, 1=end)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_simple_baselines(samples_df, lang_code, output_dir, seed=42):
    """
    Run all simple baselines on the given NER dataset.
    
    Args:
        samples_df: DataFrame containing NER samples
        lang_code: Language code
        output_dir: Directory to save results and plots
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with combined results and summary DataFrame
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    plots_dir = os.path.join(output_dir, 'plots', lang_code)
    results_dir = os.path.join(output_dir, 'results', lang_code)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot dataset statistics
    plot_dataset_stats(
        samples_df,
        lang_code,
        os.path.join(plots_dir, f'ner_dataset_stats_{lang_code}.png')
    )
    
    plot_entity_distribution_by_position(
        samples_df,
        lang_code,
        os.path.join(plots_dir, f'ner_entity_position_{lang_code}.png')
    )
    
    # Initialize combined results
    all_results = []
    summary_metrics = []
    
    # Run all baselines
    print(f"\n--- Running simple baselines for {lang_code} ---")
    
    # 1. Majority baseline
    print("\nRunning majority baseline...")
    majority_results, majority_metrics = majority_baseline(samples_df)
    all_results.append(majority_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'majority',
        'precision': majority_metrics['precision'],
        'recall': majority_metrics['recall'],
        'f1': majority_metrics['f1']
    })
    
    # 2. Random baseline
    print("\nRunning random baseline...")
    random_results, random_metrics = random_baseline(samples_df)
    all_results.append(random_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'random',
        'precision': random_metrics['precision'],
        'recall': random_metrics['recall'],
        'f1': random_metrics['f1']
    })
    
    # 3. Length-based baseline
    print("\nRunning length-based baseline...")
    length_results, length_metrics = length_based_baseline(samples_df)
    all_results.append(length_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'length_based',
        'precision': length_metrics['precision'],
        'recall': length_metrics['recall'],
        'f1': length_metrics['f1']
    })
    
    # 4. Position-based baseline
    print("\nRunning position-based baseline...")
    position_results, position_metrics = position_based_baseline(samples_df)
    all_results.append(position_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'position_based',
        'precision': position_metrics['precision'],
        'recall': position_metrics['recall'],
        'f1': position_metrics['f1']
    })
    
    # 5. First-word baseline
    print("\nRunning first-word baseline...")
    first_word_results, first_word_metrics = first_word_baseline(samples_df)
    all_results.append(first_word_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'first_word',
        'precision': first_word_metrics['precision'],
        'recall': first_word_metrics['recall'],
        'f1': first_word_metrics['f1']
    })
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    combined_results.to_csv(os.path.join(results_dir, f'ner_simple_baselines_{lang_code}.csv'), index=False)
    
    # Create and save summary metrics
    summary_df = pd.DataFrame(summary_metrics)
    summary_df.to_csv(os.path.join(output_dir, 'summaries', f'ner_simple_baselines_summary_{lang_code}.csv'), index=False)
    
    # Plot precision, recall, F1 comparison
    plt.figure(figsize=(12, 6))
    
    approaches = summary_df['approach'].tolist()
    precisions = summary_df['precision'].tolist()
    recalls = summary_df['recall'].tolist()
    f1_scores = summary_df['f1'].tolist()
    
    x = np.arange(len(approaches))
    width = 0.25
    
    plt.bar(x - width, precisions, width, label='Precision', color='skyblue')
    plt.bar(x, recalls, width, label='Recall', color='lightcoral')
    plt.bar(x + width, f1_scores, width, label='F1', color='lightgreen')
    
    plt.xlabel('Approach')
    plt.ylabel('Score')
    plt.title(f'NER Simple Baselines Performance Comparison - {lang_code}')
    plt.xticks(x, [a.replace('_', ' ').title() for a in approaches])
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add score labels on bars
    for i, score in enumerate(zip(precisions, recalls, f1_scores)):
        plt.text(i - width, score[0] + 0.02, f'{score[0]:.2f}', ha='center')
        plt.text(i, score[1] + 0.02, f'{score[1]:.2f}', ha='center')
        plt.text(i + width, score[2] + 0.02, f'{score[2]:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'ner_baselines_comparison_{lang_code}.png'))
    plt.close()
    
    # Plot per-entity F1 scores for best approach
    best_approach_idx = f1_scores.index(max(f1_scores))
    best_approach = approaches[best_approach_idx]
    best_metrics = [majority_metrics, random_metrics, length_metrics, position_metrics, first_word_metrics][best_approach_idx]
    
    plt.figure(figsize=(12, 6))
    
    entity_types = list(best_metrics['entity_scores'].keys())
    entity_f1s = [best_metrics['entity_scores'][et]['f1'] for et in entity_types]
    entity_support = [best_metrics['entity_scores'][et]['support'] for et in entity_types]
    
    # Sort by F1 score
    sorted_indices = np.argsort(entity_f1s)[::-1]
    sorted_types = [entity_types[i] for i in sorted_indices]
    sorted_f1s = [entity_f1s[i] for i in sorted_indices]
    sorted_support = [entity_support[i] for i in sorted_indices]
    
    plt.bar(sorted_types, sorted_f1s, color='lightgreen')
    
    plt.xlabel('Entity Type')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score by Entity Type for {best_approach.title()} Approach - {lang_code}')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Add support count as text
    for i, (f1, support) in enumerate(zip(sorted_f1s, sorted_support)):
        plt.text(i, f1 + 0.02, f'n={support}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'ner_entity_f1_{lang_code}.png'))
    plt.close()
    
    # Print summary
    print(f"\nSummary of NER simple baselines for {lang_code}:")
    print(summary_df.to_string(index=False))
    
    return combined_results, summary_df

def main():
    """Main function to run all simple baselines."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directories
    output_dir = "/work/bbd6522/results/ner/simple_baselines"
    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    
    # Store all summary results for comparison
    all_summaries = []
    
    # Process each language
    for lang_code in args.langs:
        print(f"\n=== Processing {lang_code} ===")
        
        # Load samples
        full_samples_df = load_masakhaner_samples(lang_code, num_samples=None, split='test')
        
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
        combined_summary.to_csv(os.path.join(summaries_dir, 'ner_simple_baselines_all_languages.csv'), index=False)
        
        # Create cross-language comparison plot
        plt.figure(figsize=(15, 8))
        
        # Prepare data for grouped bar chart
        approaches = combined_summary['approach'].unique()
        languages = combined_summary['language'].unique()
        
        x = np.arange(len(approaches))
        width = 0.8 / len(languages)
        
        for i, lang in enumerate(languages):
            lang_data = combined_summary[combined_summary['language'] == lang]
            lang_f1 = [lang_data[lang_data['approach'] == app]['f1'].values[0] for app in approaches]
            
            plt.bar(x + (i - len(languages)/2 + 0.5) * width, lang_f1, width, label=lang)
        
        plt.xlabel('Approach')
        plt.ylabel('F1 Score')
        plt.title('NER Simple Baselines Performance Across Languages')
        plt.xticks(x, [a.replace('_', ' ').title() for a in approaches])
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plots', 'ner_baselines_cross_language_comparison.png'))
        plt.close()
        
        print("\nCross-language comparison summary:")
        print(combined_summary.pivot_table(
            index='approach', columns='language', values=['precision', 'recall', 'f1']
        ))

if __name__ == "__main__":
    main() 