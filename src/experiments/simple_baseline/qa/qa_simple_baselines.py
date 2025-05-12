# src/experiments/simple_baseline/qa/qa_simple_baselines.py

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
from src.utils.data_loaders.load_tydiqa import load_tydiqa_samples
from evaluation.baseline.qa_metrics_baseline import calculate_qa_f1 # Use existing F1 calculator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run QA simple theoretical baselines")
    parser.add_argument("--langs", nargs='+', default=['en', 'sw', 'te'],
                        help="Languages to test (default: en, sw, te)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to use per language (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="/work/bbd6522/results/qa/simple_baselines",
                        help="Base output directory for results, plots, and summaries")
    return parser.parse_args()

def majority_baseline(samples_df):
    """
    Implements a majority class baseline for QA.
    Predicts 'Yes' for all questions (common for yes/no, fallback for others).

    Args:
        samples_df: DataFrame containing QA samples

    Returns:
        DataFrame with predictions and metrics
    """
    majority_label = "Yes" # Simple majority assumption for QA
    print(f"Majority baseline predicting: '{majority_label}'")

    results_df = samples_df.copy()
    results_df['predicted_answer'] = majority_label
    results_df['approach'] = 'majority'

    # Calculate metrics
    metrics = calculate_metrics(results_df)
    return results_df, metrics

def random_baseline(samples_df, seed=42):
    """
    Implements a random baseline for QA.
    Randomly selects between 'Yes', 'No'.

    Args:
        samples_df: DataFrame containing QA samples
        seed: Random seed for reproducibility

    Returns:
        DataFrame with predictions and metrics
    """
    np.random.seed(seed)
    possible_answers = ["Yes", "No"]
    print(f"Random baseline predicting from: {possible_answers}")

    results_df = samples_df.copy()
    results_df['predicted_answer'] = np.random.choice(possible_answers, size=len(results_df))
    results_df['approach'] = 'random'

    # Calculate metrics
    metrics = calculate_metrics(results_df)
    return results_df, metrics

def calculate_metrics(results_df):
    """
    Calculate evaluation metrics (F1) for QA predictions.

    Args:
        results_df: DataFrame with predicted_answer and ground_truth

    Returns:
        Dictionary with metrics
    """
    if results_df.empty or 'predicted_answer' not in results_df.columns or 'ground_truth' not in results_df.columns:
        print("ERROR: Invalid results DataFrame for metric calculation.")
        return {'f1_score': 0.0}

    # Ensure ground_truth is present (handle potential list format from loader)
    if isinstance(results_df['ground_truth'].iloc[0], list):
         results_df['ground_truth_str'] = results_df['ground_truth'].apply(lambda x: x[0] if x else "")
    else:
         results_df['ground_truth_str'] = results_df['ground_truth']

    # Calculate F1 score using the existing helper function
    results_df["f1_score"] = results_df.apply(lambda row: calculate_qa_f1(row, pred_col='predicted_answer', gold_col='ground_truth_str'), axis=1)

    avg_f1 = results_df["f1_score"].mean()

    return {
        'f1_score': avg_f1
    }

def plot_baseline_comparison(summary_df, lang_code, output_dir):
    """Plot comparison between simple baselines."""
    plt.figure(figsize=(8, 5))
    approaches = summary_df['approach'].tolist()
    f1_scores = summary_df['f1_score'].tolist()

    plt.bar(approaches, f1_scores, color=['skyblue', 'lightcoral'])

    plt.xlabel('Approach')
    plt.ylabel('Average F1 Score')
    plt.title(f'QA Simple Baseline Comparison - {lang_code}')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, score in enumerate(f1_scores):
        plt.text(i, score + 0.02, f'{score:.3f}', ha='center')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'plots', lang_code, f'qa_simple_baselines_comparison_{lang_code}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

def run_simple_qa_baselines(samples_df, lang_code, output_dir, seed=42):
    """
    Run all simple baselines on the given QA dataset.
    """
    plots_dir = os.path.join(output_dir, 'plots', lang_code)
    results_dir = os.path.join(output_dir, 'results', lang_code)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    summary_metrics = []

    print(f"\n--- Running simple QA baselines for {lang_code} ---")

    # --- Refined Ground Truth Extraction ---
    # Prioritize the expected TyDiQA format: answers dictionary -> text list -> first answer
    if 'answers' in samples_df.columns:
        try:
            # Attempt to extract the first text answer from the dictionary structure
            samples_df['ground_truth'] = samples_df['answers'].apply(
                lambda ans_dict: ans_dict.get('text', ["."])[0] if isinstance(ans_dict, dict) and ans_dict.get('text') else "."
            )
            # Handle cases where extraction might result in None or empty string, default to "."
            samples_df['ground_truth'] = samples_df['ground_truth'].fillna(".").replace("", ".")
            print("Extracted ground truth from 'answers' column.")
        except Exception as e:
            print(f"Warning: Failed to extract ground truth from 'answers' column structure: {e}. Checking for direct 'ground_truth' column.")
            if 'ground_truth' not in samples_df.columns:
                print("Error: Neither 'answers' structure nor 'ground_truth' column found.")
                return pd.DataFrame(), pd.DataFrame()
            else:
                 # Use existing ground_truth column if present (fallback)
                 print("Using existing 'ground_truth' column as fallback.")
                 # Ensure it's not a list
                 if isinstance(samples_df['ground_truth'].iloc[0], list):
                     samples_df['ground_truth'] = samples_df['ground_truth'].apply(lambda x: x[0] if x else ".")

    elif 'ground_truth' not in samples_df.columns:
         print("Error: Could not find ground truth answers. Missing 'answers' and 'ground_truth' columns.")
         return pd.DataFrame(), pd.DataFrame()
    else:
        # Use existing ground_truth column if 'answers' isn't present
        print("Using existing 'ground_truth' column.")
        if isinstance(samples_df['ground_truth'].iloc[0], list):
             samples_df['ground_truth'] = samples_df['ground_truth'].apply(lambda x: x[0] if x else ".")

    # Ensure ground_truth column has valid strings after processing
    samples_df['ground_truth'] = samples_df['ground_truth'].fillna(".").astype(str)
    # ----------------------------------------

    # 1. Majority Baseline
    print("Running majority baseline...")
    majority_results, majority_metrics = majority_baseline(samples_df)
    all_results.append(majority_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'majority',
        'f1_score': majority_metrics['f1_score']
    })

    # 2. Random Baseline
    print("Running random baseline...")
    random_results, random_metrics = random_baseline(samples_df, seed)
    all_results.append(random_results)
    summary_metrics.append({
        'language': lang_code,
        'approach': 'random',
        'f1_score': random_metrics['f1_score']
    })

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(os.path.join(results_dir, f'qa_simple_baselines_{lang_code}.csv'), index=False)

    summary_df = pd.DataFrame(summary_metrics)
    summary_df.to_csv(os.path.join(output_dir, 'summaries', f'qa_simple_baselines_summary_{lang_code}.csv'), index=False)

    plot_baseline_comparison(summary_df, lang_code, output_dir)

    print(f"\nSummary of QA simple baselines for {lang_code}:")
    print(summary_df.to_string(index=False))

    return combined_results, summary_df

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output
    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)

    all_summaries = []

    for lang_code in args.langs:
        print(f"\n=== Processing {lang_code} ===")
        # Load data - Using validation split and sampling
        full_samples_df = load_tydiqa_samples(lang_code, num_samples=None, split='validation')

        if full_samples_df.empty:
            print(f"WARNING: No samples loaded for {lang_code}. Skipping.")
            continue

        total_loaded = len(full_samples_df)
        num_to_sample = min(total_loaded, args.samples)
        print(f"Loaded {total_loaded} validation samples. Sampling {num_to_sample}...")
        samples_df = full_samples_df.sample(n=num_to_sample, random_state=args.seed)

        # Run baselines
        _, summary_df = run_simple_qa_baselines(samples_df, lang_code, output_dir, args.seed)
        if not summary_df.empty:
            all_summaries.append(summary_df)

    # Combine all summaries
    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_summary.to_csv(os.path.join(summaries_dir, 'qa_simple_baselines_all_languages.csv'), index=False)

        # Plot cross-language comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x='language', y='f1_score', hue='approach', data=combined_summary, ci=None)
        plt.title('QA Simple Baselines F1 Score Across Languages')
        plt.ylabel('Average F1 Score')
        plt.xlabel('Language')
        plt.ylim(0, 1.0)
        plt.legend(title='Approach')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plots', 'qa_simple_baselines_cross_language.png'))
        plt.close()

        print("\nCross-language comparison summary:")
        print(combined_summary.pivot_table(index='approach', columns='language', values='f1_score'))
    else:
        print("\nNo simple baseline results generated.")

if __name__ == "__main__":
    main() 