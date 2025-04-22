from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
from typing import Dict, Any

def calculate_classification_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate classification metrics (Accuracy, Macro F1-score, 
    and per-class precision and recall).

    Args:
        results_df: DataFrame with 'ground_truth_label' and 'predicted_label' columns.

    Returns:
        Dictionary containing the calculated metrics.
    """
    y_true = results_df['ground_truth_label']
    y_pred = results_df['predicted_label']

    # Get the unique labels actually present in the ground truth for this run
    valid_labels = sorted(list(y_true.unique())) 
    print(f"Ground truth labels found: {valid_labels}")
    print(f"Predicted labels found: {set(y_pred.unique())}")

    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=valid_labels, zero_division=0)
    
    print("\nClassification Report:")
    report_dict = None
    try:
        # Generate a detailed report per class, output as dict
        # Ensure we use the valid_labels found in y_true for the report generation
        report_dict = classification_report(y_true, y_pred, labels=valid_labels, zero_division=0, output_dict=True)
        # Print the string version for logging
        print(classification_report(y_true, y_pred, labels=valid_labels, zero_division=0)) 
    except Exception as e:
        print(f"Could not generate classification report: {e}")

    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
    }

    # Extract per-class metrics from the report dictionary if available
    if report_dict:
        # Iterate through the labels actually found in the ground truth
        for label in valid_labels:
            if label in report_dict: # Check if the label exists as a key in the report
                metrics[f'{label}_precision'] = report_dict[label].get('precision', 0.0)
                metrics[f'{label}_recall'] = report_dict[label].get('recall', 0.0) # Recall = Accuracy within class
            else:
                # This case should ideally not happen if valid_labels are derived from y_true
                metrics[f'{label}_precision'] = 0.0
                metrics[f'{label}_recall'] = 0.0
    else:
        # Ensure keys exist even if report failed, using valid_labels from ground truth
        for label in valid_labels:
            metrics[f'{label}_precision'] = float('nan')
            metrics[f'{label}_recall'] = float('nan')
            
    return metrics 