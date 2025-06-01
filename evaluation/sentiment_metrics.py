# Placeholder for sentiment metrics

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import pandas as pd
import numpy as np

EXPECTED_SENTIMENT_LABELS = ["positive", "negative", "neutral"]

def calculate_sentiment_metrics(results_df: pd.DataFrame, labels: list = None) -> dict:
    """
    Calculate sentiment analysis metrics including accuracy, F1 (macro, weighted),
    precision (macro, weighted), recall (macro, weighted), and per-class metrics.

    Args:
        results_df (pd.DataFrame): DataFrame with 'ground_truth_label' and 'predicted_label' columns.
        labels (list, optional): The set of labels to include for F1, precision, recall. 
                                 Defaults to EXPECTED_SENTIMENT_LABELS.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    if labels is None:
        labels = EXPECTED_SENTIMENT_LABELS

    if results_df.empty:
        print("WARNING: Results DataFrame is empty. Returning zeroed metrics.")
        metrics_dict = {
            'accuracy': 0.0, 'macro_f1': 0.0, 'macro_precision': 0.0, 'macro_recall': 0.0,
            'weighted_f1': 0.0, 'weighted_precision': 0.0, 'weighted_recall': 0.0
        }
        for label in labels:
            metrics_dict[f'{label}_precision'] = 0.0
            metrics_dict[f'{label}_recall'] = 0.0
            metrics_dict[f'{label}_f1-score'] = 0.0
            metrics_dict[f'{label}_support'] = 0
        return metrics_dict

    y_true = results_df['ground_truth_label'].astype(str).fillna("unknown").str.lower()
    y_pred = results_df['predicted_label'].astype(str).fillna("unknown").str.lower()

    # Ensure labels for metrics are present in the data or are the default expected ones
    active_labels = sorted(list(set(y_true) | set(y_pred) & set(labels)))
    if not active_labels:
        print(f"WARN: No common labels found between data and provided labels list: {labels}. Using default set for reporting if any overlap.")
        active_labels = [l for l in labels if l in y_true or l in y_pred] 
        if not active_labels: # If still no overlap, use all unique labels from data, can lead to issues if unexpected labels appear
            active_labels = sorted(list(set(y_true) | set(y_pred)))
            if not active_labels: # If no labels at all, return zeroed metrics
                 print("ERROR: No labels found in y_true or y_pred. Returning zeroed metrics.")
                 return calculate_sentiment_metrics(pd.DataFrame(), labels=labels) # Recurse with empty to get zeroed structure
            print(f"WARN: Defaulting to all unique labels found in data for reporting: {active_labels}")

    accuracy = accuracy_score(y_true, y_pred)
    
    macro_f1 = f1_score(y_true, y_pred, labels=active_labels, average='macro', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, labels=active_labels, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=active_labels, average='macro', zero_division=0)
    
    weighted_f1 = f1_score(y_true, y_pred, labels=active_labels, average='weighted', zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, labels=active_labels, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, labels=active_labels, average='weighted', zero_division=0)
    
    report_dict = classification_report(y_true, y_pred, labels=active_labels, output_dict=True, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'weighted_f1': weighted_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
    }
    
    for label_name in labels: # Report on all expected labels passed in the `labels` arg
        class_metrics = report_dict.get(label_name, {})
        metrics[f'{label_name}_precision'] = class_metrics.get('precision', 0.0)
        metrics[f'{label_name}_recall'] = class_metrics.get('recall', 0.0)
        metrics[f'{label_name}_f1-score'] = class_metrics.get('f1-score', 0.0)
        metrics[f'{label_name}_support'] = class_metrics.get('support', 0)
        
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=active_labels, zero_division=0))
    
    return metrics 