from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd
from typing import Dict, Any

# Define the expected labels in order, if needed for specific reports, 
# or rely on labels present in the data.
# Ensure this matches EXPECTED_LABELS in the evaluation scripts if possible.
LABELS_ORDER = ["positive", "negative", "neutral"]

def calculate_sentiment_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate sentiment classification metrics (Accuracy, Macro F1-score, 
    and per-class precision and recall).

    Args:
        results_df: DataFrame with 'ground_truth_label' and 'predicted_label' columns.

    Returns:
        Dictionary containing the calculated metrics.
    """
    y_true = results_df['ground_truth_label']
    y_pred = results_df['predicted_label']

    # Handle cases where prediction might be invalid (e.g., '[Unknown]')
    # Option 1: Treat unknowns as incorrect (simplest)
    # Option 2: Filter out unknowns before metric calculation (might bias results)
    # Sticking with Option 1 for now.
    valid_labels = sorted(list(set(y_true.unique()))) # Use sorted list of labels present
    print(f"Ground truth labels found: {valid_labels}")
    print(f"Predicted labels found: {set(y_pred.unique())}")

    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate Macro F1-score: average F1 per class, unweighted.
    # Handles label imbalance better than micro-average for classification.
    # Ensure labels parameter includes all possible ground truth labels for proper averaging
    # Use zero_division=0 to avoid warnings if a class has no predictions.
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=valid_labels, zero_division=0)
    
    print("\nClassification Report:")
    report_dict = None
    try:
        # Generate a detailed report per class, output as dict
        report = classification_report(y_true, y_pred, labels=valid_labels, zero_division=0, output_dict=True)
        # Print the string version for logging
        print(classification_report(y_true, y_pred, labels=valid_labels, zero_division=0)) 
        report_dict = report
    except Exception as e:
        print(f"Could not generate classification report: {e}")

    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        # Add other metrics if needed (e.g., weighted F1, precision/recall per class)
    }

    # Extract per-class metrics from the report dictionary if available
    if report_dict:
        for label in LABELS_ORDER: # Use predefined order to ensure keys exist
            if label in report_dict: # Check if the label was actually in the report
                metrics[f'{label}_precision'] = report_dict[label].get('precision', 0.0)
                metrics[f'{label}_recall'] = report_dict[label].get('recall', 0.0) # Recall = Accuracy within class
                # metrics[f'{label}_f1'] = report_dict[label].get('f1-score', 0.0) # Optionally add per-class F1 too
            else:
                # Handle cases where a label might be missing in results (e.g., only 2 classes predicted)
                metrics[f'{label}_precision'] = 0.0
                metrics[f'{label}_recall'] = 0.0
                # metrics[f'{label}_f1'] = 0.0
    else:
        # Ensure keys exist even if report failed
        for label in LABELS_ORDER:
            metrics[f'{label}_precision'] = float('nan')
            metrics[f'{label}_recall'] = float('nan')
            # metrics[f'{label}_f1'] = float('nan')
            
    return metrics

# Helper function to apply calculation to a DataFrame row (if needed, but usually done on full DF)
# def calculate_metrics_row(row: Dict[str, Any]) -> Dict[str, float]:
#     # This approach is less common for classification metrics, which are calculated dataset-wide
#     # It might be useful if calculating something like simple match accuracy per row
#     accuracy = 1.0 if row['ground_truth_label'] == row['predicted_label'] else 0.0
#     return {'accuracy': accuracy} 