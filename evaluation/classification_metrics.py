import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

def calculate_classification_metrics(results_df: pd.DataFrame, possible_labels_en: list = None) -> dict:
    """
    Calculates classification metrics from a results DataFrame.

    Args:
        results_df (pd.DataFrame): DataFrame containing 'ground_truth_label' and 
                                   a column for predicted labels (e.g., 'final_predicted_label' or 'predicted_label').
        possible_labels_en (list, optional): A list of all possible English label names. 
                                             Used for per-class metrics and ensuring report consistency.
                                             If None, labels will be inferred from the data.

    Returns:
        dict: A dictionary containing various classification metrics.
    """
    # Determine the column name for predictions. Common names are 'final_predicted_label' or 'predicted_label'.
    # Prioritize 'final_predicted_label' if it exists.
    if 'final_predicted_label' in results_df.columns:
        pred_col = 'final_predicted_label'
    elif 'predicted_label' in results_df.columns:
        pred_col = 'predicted_label'
    else:
        raise ValueError("DataFrame must contain either 'final_predicted_label' or 'predicted_label' column.")

    if 'ground_truth_label' not in results_df.columns:
        raise ValueError("DataFrame must contain 'ground_truth_label' column.")

    y_true = results_df['ground_truth_label'].astype(str)
    y_pred = results_df[pred_col].astype(str)

    # Handle cases where labels might not be present in y_pred if possible_labels_en is provided
    labels_to_use = possible_labels_en
    if labels_to_use is None:
        # Infer labels from the union of true and predicted, maintaining an order
        labels_to_use = sorted(list(set(y_true) | set(y_pred)))
    
    if not labels_to_use: # If no labels are found at all
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'samples': len(y_true)
        }

    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate precision, recall, F1 for macro and weighted averages
    # zero_division=0 means if a class has no predictions or no true labels, its score is 0 for that metric
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0, labels=labels_to_use if possible_labels_en else None
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0, labels=labels_to_use if possible_labels_en else None
    )

    metrics = {
        'accuracy': accuracy,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'weighted_precision': precision_weighted,
        'weighted_recall': recall_weighted,
        'weighted_f1': f1_weighted,
        'samples': len(y_true)
    }

    # Per-class metrics
    # classification_report can also provide this, but direct use of prfs gives more control
    # Ensure labels_to_use includes all labels present in y_true or y_pred to avoid warnings
    unique_labels_in_data = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
    report_labels = possible_labels_en if possible_labels_en else unique_labels_in_data

    if report_labels:
        try:
            report = classification_report(y_true, y_pred, labels=report_labels, output_dict=True, zero_division=0)
            for label in report_labels:
                if label in report: # Check if label is in the report (it might not be if it had no support)
                    metrics[f'{label}_precision'] = report[label]['precision']
                    metrics[f'{label}_recall'] = report[label]['recall']
                    metrics[f'{label}_f1-score'] = report[label]['f1-score']
                    metrics[f'{label}_support'] = report[label]['support']
                else: # If a label from possible_labels_en was not in the report (e.g., no true/pred instances)
                    metrics[f'{label}_precision'] = 0.0
                    metrics[f'{label}_recall'] = 0.0
                    metrics[f'{label}_f1-score'] = 0.0
                    metrics[f'{label}_support'] = 0
        except Exception as e:
            print(f"Warning: Could not generate detailed classification report: {e}")
            # Fallback if classification_report fails for some edge cases
            if possible_labels_en:
                for label in possible_labels_en:
                    metrics[f'{label}_precision'] = 0.0
                    metrics[f'{label}_recall'] = 0.0
                    metrics[f'{label}_f1-score'] = 0.0
                    metrics[f'{label}_support'] = 0


    return metrics 