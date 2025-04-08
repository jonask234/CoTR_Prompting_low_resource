from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
from typing import Dict, Any

# Define the expected labels in order, if needed for specific reports, 
# or rely on labels present in the data.
# Ensure this matches EXPECTED_LABELS in the evaluation scripts if possible.
LABELS_ORDER = ["positive", "negative", "neutral"]

def calculate_sentiment_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate sentiment classification metrics (Accuracy, Macro F1-score).

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
    valid_labels = set(y_true.unique()) # Get actual labels present in ground truth
    print(f"Ground truth labels found: {valid_labels}")
    print(f"Predicted labels found: {set(y_pred.unique())}")

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate Macro F1-score: average F1 per class, unweighted.
    # Handles label imbalance better than micro-average for classification.
    # Ensure labels parameter includes all possible ground truth labels for proper averaging
    # Use zero_division=0 to avoid warnings if a class has no predictions.
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=list(valid_labels), zero_division=0)
    
    print("\nClassification Report:")
    try:
        # Generate a detailed report per class
        report = classification_report(y_true, y_pred, labels=list(valid_labels), zero_division=0)
        print(report)
    except Exception as e:
        print(f"Could not generate classification report: {e}")

    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        # Add other metrics if needed (e.g., weighted F1, precision/recall per class)
    }

    return metrics

# Helper function to apply calculation to a DataFrame row (if needed, but usually done on full DF)
# def calculate_metrics_row(row: Dict[str, Any]) -> Dict[str, float]:
#     # This approach is less common for classification metrics, which are calculated dataset-wide
#     # It might be useful if calculating something like simple match accuracy per row
#     accuracy = 1.0 if row['ground_truth_label'] == row['predicted_label'] else 0.0
#     return {'accuracy': accuracy} 