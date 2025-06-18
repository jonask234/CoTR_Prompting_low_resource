import pandas as pd
import argparse
from scipy.stats import ttest_rel
from pathlib import Path
from scipy.stats import pearsonr

def run_paired_t_test(file1_path: Path, file2_path: Path, metric_col: str = 'accuracy'):
    """
    Performs a paired t-test on the results of two experiments.

    Args:
        file1_path (Path): Path to the first results CSV file.
        file2_path (Path): Path to the second results CSV file.
        metric_col (str): The metric to use for the test. 'accuracy' or 'f1'.
                          'accuracy' is based on correctness of prediction.
    """
    print(f"Comparing results from:")
    print(f"  - File 1: {file1_path.name}")
    print(f"  - File 2: {file2_path.name}")
    print("-" * 30)

    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the files. Details: {e}")
        return

    # --- Data Preparation and Sanity Checks ---
    # Use 'id' column for sentiment/classification, 'premise_lrl' for NLI
    id_col = 'id'
    if 'premise_lrl' in df1.columns and 'premise_lrl' in df2.columns:
        id_col = 'premise_lrl' # NLI uses premise as a unique identifier
        # A more robust ID would be to combine premise and hypothesis
        df1['id'] = df1['premise'] + df1['hypothesis']
        df2['id'] = df2['premise'] + df2['hypothesis']
    elif 'original_text' in df1.columns and 'original_text' in df2.columns:
        id_col = 'original_text' # NER uses original_text
    elif 'id' not in df1.columns or 'id' not in df2.columns:
        print(f"Error: Cannot find a common ID column ('id', 'premise_lrl', 'original_text') to merge results.")
        # Create a fallback ID based on index if all else fails
        df1['id'] = df1.index
        df2['id'] = df2.index
        id_col = 'id'
        print("Warning: Using row index as a fallback ID. This is only reliable if the input data order was identical.")

    # Merge dataframes to ensure pairing
    merged_df = pd.merge(df1, df2, on=id_col, suffixes=('_1', '_2'))

    if len(merged_df) == 0:
        print("Error: The two result files have no samples in common based on the ID column. Cannot perform a paired test.")
        return
        
    print(f"Found {len(merged_df)} common samples between the two files.")

    # --- Score Calculation ---
    scores1, scores2 = None, None
    if metric_col == 'accuracy':
        print("Using metric: Accuracy (1 for correct, 0 for incorrect)")
        # Determine ground truth and prediction columns for each file
        gt_col_1 = 'ground_truth_label' if 'ground_truth_label' in merged_df else 'ground_truth_eng_label_str'
        pred_col_1 = 'predicted_label' if 'predicted_label' in merged_df else 'final_predicted_label_for_accuracy'
        
        gt_col_2 = 'ground_truth_label_2' if 'ground_truth_label_2' in merged_df else 'ground_truth_eng_label_str_2'
        pred_col_2 = 'predicted_label_2' if 'predicted_label_2' in merged_df else 'final_predicted_label_for_accuracy_2'

        # Check if columns exist
        if not all(c in merged_df.columns for c in [f"{gt_col_1}_1", f"{pred_col_1}_1", f"{gt_col_2}", f"{pred_col_2}"]):
            print("Error: Could not find required ground_truth/prediction columns in the merged dataframe.")
            print(f"Required: {gt_col_1}_1, {pred_col_1}_1, {gt_col_2}, {pred_col_2}")
            print(f"Available: {merged_df.columns.tolist()}")
            return

        scores1 = (merged_df[f'{gt_col_1}_1'].str.lower() == merged_df[f'{pred_col_1}_1'].str.lower()).astype(int)
        scores2 = (merged_df[f'{gt_col_2}'].str.lower() == merged_df[f'{pred_col_2}'].str.lower()).astype(int)
    else:
        print(f"Metric '{metric_col}' is not implemented yet. Please use 'accuracy'.")
        return

    # --- Perform t-test ---
    if scores1.equals(scores2):
        print("\nThe performance scores for both models are identical for all common samples.")
        t_stat, p_value = 0.0, 1.0
    else:
        t_stat, p_value = ttest_rel(scores1, scores2)

    print("\n--- T-Test Results ---")
    print(f"Mean Score Model 1: {scores1.mean():.4f}")
    print(f"Mean Score Model 2: {scores2.mean():.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    # --- Interpretation ---
    alpha = 0.05
    print("\n--- Interpretation (at alpha = 0.05) ---")
    if p_value < alpha:
        print("Result: The difference is STATISTICALLY SIGNIFICANT.")
        print("Conclusion: You can reject the null hypothesis. The observed difference between the two models is unlikely to be due to random chance.")
        if scores1.mean() > scores2.mean():
            print("Model 1 performed significantly better than Model 2.")
        else:
            print("Model 2 performed significantly better than Model 1.")
    else:
        print("Result: The difference is NOT statistically significant.")
        print("Conclusion: You cannot reject the null hypothesis. The observed difference in performance could be due to random chance.")

def run_pearson_correlation(file_path: Path, comet_col: str, metric_col: str = 'accuracy'):
    """
    Performs a Pearson correlation analysis on a single result file.

    Args:
        file_path (Path): Path to the results CSV file from a CoTR experiment.
        comet_col (str): The name of the COMET score column to use for correlation.
        metric_col (str): The performance metric to use ('accuracy').
    """
    print(f"Running Pearson Correlation Analysis on:")
    print(f"  - File: {file_path.name}")
    print(f"  - Correlating: '{comet_col}' with per-sample accuracy.")
    print("-" * 30)

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find the file. Details: {e}")
        return

    # --- Data Preparation ---
    if comet_col not in df.columns:
        print(f"Error: COMET score column '{comet_col}' not found in the file.")
        print(f"Available columns: {df.columns.tolist()}")
        return
        
    # Drop rows where COMET score is NaN, as correlation cannot be computed
    df.dropna(subset=[comet_col], inplace=True)
    
    if len(df) < 3:
        print(f"Error: Not enough data points ({len(df)}) to perform a meaningful correlation analysis after dropping NaNs.")
        return

    # --- Score Calculation ---
    performance_scores = None
    if metric_col == 'accuracy':
        # Determine ground truth and prediction columns
        gt_col = 'ground_truth_label' if 'ground_truth_label' in df else 'ground_truth_eng_label_str'
        pred_col = 'predicted_label' if 'predicted_label' in df else 'final_predicted_label' if 'final_predicted_label' in df else 'predicted_label_for_accuracy' if 'predicted_label_for_accuracy' in df else 'intermediate_en_label'
        
        if not all(c in df.columns for c in [gt_col, pred_col]):
            print("Error: Could not find required ground_truth/prediction columns for accuracy calculation.")
            return
            
        performance_scores = (df[gt_col].astype(str).str.lower() == df[pred_col].astype(str).str.lower()).astype(int)
    else:
        print(f"Metric '{metric_col}' is not implemented yet for correlation. Please use 'accuracy'.")
        return

    comet_scores = df[comet_col]

    # --- Perform Pearson Correlation ---
    r_value, p_value = pearsonr(comet_scores, performance_scores)

    print("\n--- Pearson Correlation Results ---")
    print(f"Correlation Coefficient (r): {r_value:.4f}")
    print(f"p-value: {p_value:.4f}")

    # --- Interpretation ---
    alpha = 0.05
    print("\n--- Interpretation (at alpha = 0.05) ---")

    # Interpret the strength of the correlation
    strength = "no"
    if abs(r_value) >= 0.7:
        strength = "very strong"
    elif abs(r_value) >= 0.5:
        strength = "strong"
    elif abs(r_value) >= 0.3:
        strength = "moderate"
    elif abs(r_value) >= 0.1:
        strength = "weak"
    
    direction = "positive" if r_value > 0 else "negative" if r_value < 0 else ""

    if p_value < alpha:
        print(f"Result: The correlation is STATISTICALLY SIGNIFICANT.")
        print(f"Conclusion: There is a significant {strength} {direction} linear relationship between {comet_col} and task performance.")
    else:
        print("Result: The correlation is NOT statistically significant.")
        print("Conclusion: There is not enough evidence to conclude a significant linear relationship exists between these variables.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform statistical tests on experiment result files.")
    subparsers = parser.add_subparsers(dest="test_type", required=True, help="The type of statistical test to run.")

    # --- T-Test Subparser ---
    parser_ttest = subparsers.add_parser("ttest", help="Perform a paired t-test between two result files.")
    parser_ttest.add_argument("file1", type=Path, help="Path to the first detailed results CSV file.")
    parser_ttest.add_argument("file2", type=Path, help="Path to the second detailed results CSV file.")

    # --- Correlation Subparser ---
    parser_corr = subparsers.add_parser("correlation", help="Perform a Pearson correlation analysis on one result file.")
    parser_corr.add_argument("file", type=Path, help="Path to a detailed CoTR results CSV file.")
    parser_corr.add_argument("--comet_col", type=str, default="comet_lrl_text_to_en", help="Name of the COMET score column to use.")

    args = parser.parse_args()
    
    if args.test_type == "ttest":
        run_paired_t_test(args.file1, args.file2)
        print("\nExample t-test command:")
        print("python src/analysis/statistical_tests.py ttest /path/to/results_model_A.csv /path/to/results_model_B.csv")
    elif args.test_type == "correlation":
        run_pearson_correlation(args.file, args.comet_col)
        print("\nExample correlation command:")
        print("python src/analysis/statistical_tests.py correlation /path/to/cotr_results.csv --comet_col comet_lrl_text_to_en") 