import pandas as pd
import argparse
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the language family utility
try:
    from src.utils.language_families import get_major_language_family
except ImportError as e:
    print(f"Error importing language families utility: {e}")
    print("Please ensure src/utils/language_families.py exists.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_family_analysis(summary_file_path: str, output_dir: str):
    """
    Performs language family-based analysis on an experiment summary file.

    Args:
        summary_file_path (str): Path to the overall experiment summary CSV.
        output_dir (str): Directory to save the analysis results and plots.
    """
    if not os.path.exists(summary_file_path):
        logging.error(f"Summary file not found at: {summary_file_path}")
        return

    logging.info(f"Loading summary data from: {summary_file_path}")
    df = pd.read_csv(summary_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Add Language Family Column ---
    if 'language' not in df.columns:
        logging.error("Summary file must contain a 'language' column with language codes.")
        return
    df['language_family'] = df['language'].apply(get_major_language_family)
    
    # Exclude languages with 'Unknown' or 'Multilingual' families for aggregation
    analysis_df = df[~df['language_family'].isin(['Unknown', 'Multilingual'])].copy()
    logging.info(f"Filtered out 'Unknown' and 'Multilingual' families. Analyzing {len(analysis_df)} results.")

    # --- 2. Normalize Columns for Aggregation ---
    # Create a unified 'method' column (e.g., 'Baseline', 'CoTR-Multi', 'CoTR-Single')
    if 'pipeline' in analysis_df.columns: # CoTR results
        analysis_df['method'] = 'CoTR-' + analysis_df['pipeline'].str.replace('_prompt', '')
    elif 'prompt_language' in analysis_df.columns: # Baseline results
        analysis_df['method'] = 'Baseline-' + analysis_df['prompt_language'].str.replace('-instruct', '')
    else:
        logging.warning("Could not determine a detailed method from 'pipeline' or 'prompt_language'. Using 'model' column if available, else 'Unknown'.")
        if 'model' in analysis_df.columns:
            analysis_df['method'] = analysis_df['model']
        else:
            analysis_df['method'] = 'Unknown'

    if 'shot_type' in analysis_df.columns:
        analysis_df['method'] = analysis_df['method'] + '-' + analysis_df['shot_type'].astype(str)

    # --- 3. Perform Aggregation ---
    # Define metrics to average
    metrics_to_agg = ['accuracy', 'macro_f1', 'weighted_f1', 'avg_comet_lrl_text_to_en', 'avg_comet_en_label_to_lrl', 'f1']
    # Filter for metrics that actually exist in the DataFrame
    existing_metrics = [m for m in metrics_to_agg if m in analysis_df.columns]

    if not existing_metrics:
        logging.error("No standard metric columns (e.g., 'accuracy', 'macro_f1', 'f1') found in the summary file.")
        return

    # Define the core grouping columns, only including those that exist in the dataframe
    base_groups = ['language_family', 'model', 'method']
    aggregation_groups = [col for col in base_groups if col in analysis_df.columns]
    
    # Ensure language_family is always the primary group if it exists
    if 'language_family' not in aggregation_groups and 'language_family' in analysis_df.columns:
        aggregation_groups.insert(0, 'language_family')
    
    if not aggregation_groups:
        logging.error("Could not establish any columns for grouping the data (e.g., 'language_family', 'model').")
        return

    family_summary = analysis_df.groupby(aggregation_groups)[existing_metrics].mean().reset_index()

    # Save the aggregated summary
    aggregated_summary_path = os.path.join(output_dir, "summary_by_language_family.csv")
    family_summary.to_csv(aggregated_summary_path, index=False, float_format='%.4f')
    logging.info(f"Aggregated summary saved to: {aggregated_summary_path}")
    print("\n--- Aggregated Performance by Language Family ---")
    print(family_summary.to_string())

    # --- 4. Generate Plots ---
    plot_performance_by_family(family_summary, output_dir, 'macro_f1', 'Macro F1-Score')
    plot_performance_by_family(family_summary, output_dir, 'accuracy', 'Accuracy')
    if 'avg_comet_lrl_text_to_en' in family_summary.columns:
        plot_performance_by_family(family_summary, output_dir, 'avg_comet_lrl_text_to_en', 'COMET Score (LRL Text -> EN)')

def plot_performance_by_family(summary_df: pd.DataFrame, output_dir: str, metric_col: str, metric_name: str):
    """Generates and saves a bar plot of performance by language family."""
    if summary_df.empty or metric_col not in summary_df.columns:
        logging.warning(f"Cannot plot '{metric_name}'; column '{metric_col}' not in summary or summary is empty.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.barplot(
        data=summary_df,
        x='language_family',
        y=metric_col,
        hue='method',
        ax=ax,
        palette='viridis'
    )

    ax.set_title(f'Mean {metric_name} by Language Family and Method', fontsize=16, pad=20)
    ax.set_xlabel('Language Family', fontsize=12)
    ax.set_ylabel(f'Mean {metric_name}', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9, padding=3)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    plot_filename = os.path.join(output_dir, f"plot_family_{metric_col}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)
    logging.info(f"Saved plot to: {plot_filename}")

def main():
    parser = argparse.ArgumentParser(description="Run Language Family analysis on experiment summary files.")
    parser.add_argument(
        "summary_file",
        type=str,
        help="Path to the overall experiment summary CSV file (e.g., from .../summaries_overall/)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/analysis/family_analysis_results",
        help="Directory to save the analysis results and plots."
    )
    args = parser.parse_args()

    perform_family_analysis(args.summary_file, args.output_dir)

if __name__ == "__main__":
    main() 