import pandas as pd
from scipy.stats import pearsonr
import os

def analyze_h2_task_level_correlation(bleu_df, perf_df, alpha=0.05):
    """
    Performs a task-level correlation analysis to evaluate Hypothesis H2,
    aggregating all model, language, and shot-type variations.
    """
    print("--- Evaluation of Hypothesis H2 (Task-Level Correlation Analysis) ---")
    print("H2: Translation quality (BLEU) is positively correlated with task performance (F1).\n")
    print("Methodology: Correlating the AVERAGE BLEU score per task with the AVERAGE F1 score per task.\n")

    # --- Data Preparation and Aggregation ---
    # 1. Aggregate Performance per Task
    # Use the CoTR F1 score column, renaming for clarity if needed
    if 'cotr_f1' not in perf_df.columns:
        perf_df.rename(columns={'F1 Score (CoTR)': 'cotr_f1'}, inplace=True) # Adjust if name is different
    
    # Filter for relevant tasks and drop rows with missing F1 scores
    perf_df = perf_df.dropna(subset=['cotr_f1'])
    # Filter out classification as it's often handled differently or has no BLEU scores
    # perf_df = perf_df[perf_df['task'] != 'classification'] # This line was incorrect and has been removed.

    task_perf_avg = perf_df.groupby('task')['cotr_f1'].mean().reset_index()
    task_perf_avg.rename(columns={'cotr_f1': 'average_f1'}, inplace=True)

    # 2. Aggregate Translation Quality per Task
    bleu_df = bleu_df.dropna(subset=['overall_bleu'])
    task_bleu_avg = bleu_df.groupby('task')['overall_bleu'].mean().reset_index()
    task_bleu_avg.rename(columns={'overall_bleu': 'average_bleu'}, inplace=True)

    # 3. Merge aggregated data
    merged_df = pd.merge(task_perf_avg, task_bleu_avg, on='task', how='inner')

    print("--- Aggregated Data per Task ---")
    print(merged_df.to_string(index=False))
    print("-" * 35 + "\n")

    # --- Correlation Analysis ---
    if len(merged_df) < 3:
        print("Not enough task data points to perform a meaningful correlation.")
        return

    print("--- Correlation Result ---")
    corr, p_value = pearsonr(merged_df['average_bleu'], merged_df['average_f1'])
    
    print(f"  - Number of Data Points (Tasks): {len(merged_df)}")
    print(f"  - Pearson Correlation Coefficient: {corr:.4f}")
    print(f"  - p-value: {p_value:.4f}")

    if p_value < alpha:
        print("  - Result: The correlation is statistically significant.")
        if corr > 0:
            print("  - Interpretation: Tasks with higher average translation quality are associated with higher average F1 scores.\n")
        else:
            print("  - Interpretation: Tasks with higher average translation quality are associated with lower average F1 scores.\n")
    else:
        print("  - Result: The correlation is NOT statistically significant.")
        print("  - Interpretation: With this few data points, we cannot conclude there is a statistically significant linear relationship between average translation quality and average F1 score at the task level.\n")

def main():
    """
    Main function to load data and run the correlation analysis.
    """
    bleu_csv_path = os.path.join('evaluation', 'translation', 'comprehensive_assessment_fixed', 'comprehensive_translation_assessment_fixed.csv')
    perf_csv_path = os.path.join('evaluation', 'statistical_analysis', 'simplified_results', 'matched_pairs_corrected.csv')

    try:
        bleu_df = pd.read_csv(bleu_csv_path)
        # The performance CSV might have a different name for the F1 column, so let's check
        perf_df = pd.read_csv(perf_csv_path)

        print("--- Data Loading Confirmation ---")
        print(f"Successfully read {len(bleu_df)} rows from comprehensive_translation_assessment_fixed.csv.")
        print(f"Successfully read {len(perf_df)} rows from matched_pairs_corrected.csv.")
        print("All of these rows will be used to calculate the task-level averages.\n")


        if 'cotr_f1' not in perf_df.columns and 'F1 Score (CoTR)' in perf_df.columns:
             perf_df.rename(columns={'F1 Score (CoTR)': 'cotr_f1'}, inplace=True)

    except FileNotFoundError as e:
        print(f"Error: A required results file was not found.")
        print(e)
        return
    except KeyError as e:
        print(f"A required column was not found: {e}")
        return

    analyze_h2_task_level_correlation(bleu_df, perf_df)


if __name__ == "__main__":
    main() 