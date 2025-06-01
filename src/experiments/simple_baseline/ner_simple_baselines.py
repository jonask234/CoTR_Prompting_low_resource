import argparse
import pandas as pd
import os
import sys
import random
from typing import List, Dict, Any
from collections import defaultdict # Added for per-type metrics

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Corrected indentation

from src.utils.data_loaders.load_masakhaner import load_masakhaner_samples

# Entity types for MasakhaNER - ensure this matches the dataset
MASAKHANER_ENTITY_TYPES = ["PER", "ORG", "LOC", "DATE"]

def calculate_ner_metrics(ground_truth_entities_list: List[List[Dict]], 
                          predicted_entities_list: List[List[Dict]],
                          all_entity_types: List[str]) -> Dict[str, Any]:
    """
    Calculates overall and per-type Precision, Recall, F1 for NER.
    Assumes exact match for (entity_text_lowercase_stripped, entity_type).
    Args:
        ground_truth_entities_list: List of lists of GT entity dicts.
        predicted_entities_list: List of lists of predicted entity dicts.
        all_entity_types: List of all possible entity type strings.

    Returns:
        A dictionary containing overall and per-type P, R, F1 scores.
    """
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    
    per_type_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for gt_sample_entities, pred_sample_entities in zip(ground_truth_entities_list, predicted_entities_list):
        # Convert GT and Pred to sets of (text_lower_stripped, type) for easy comparison
        gt_set = set()
        for e in gt_sample_entities:
            if isinstance(e, dict) and "entity" in e and "type" in e:
                gt_set.add((str(e["entity"]).lower().strip(), str(e["type"])))
        
        pred_set = set()
        for e in pred_sample_entities:
            if isinstance(e, dict) and "entity" in e and "type" in e:
                pred_set.add((str(e["entity"]).lower().strip(), str(e["type"])))

        # Overall TP, FP, FN
        sample_tp = len(gt_set.intersection(pred_set))
        sample_fp = len(pred_set) - sample_tp
        sample_fn = len(gt_set) - sample_tp
        
        overall_tp += sample_tp
        overall_fp += sample_fp
        overall_fn += sample_fn

        # Per-type TP, FP, FN
        for entity_type in all_entity_types:
            gt_type_set = {e for e in gt_set if e[1] == entity_type}
            pred_type_set = {e for e in pred_set if e[1] == entity_type}
            
            type_tp = len(gt_type_set.intersection(pred_type_set))
            type_fp = len(pred_type_set) - type_tp
            type_fn = len(gt_type_set) - type_tp
            
            per_type_stats[entity_type]['tp'] += type_tp
            per_type_stats[entity_type]['fp'] += type_fp
            per_type_stats[entity_type]['fn'] += type_fn
            
    metrics_results = {}

    # Calculate overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    if overall_tp == 0 and overall_fp == 0 and overall_fn == 0: # Special case: no GT entities and no predictions
        overall_precision = 1.0
        overall_recall = 1.0
        overall_f1 = 1.0
        
    metrics_results['overall_precision'] = overall_precision
    metrics_results['overall_recall'] = overall_recall
    metrics_results['overall_f1'] = overall_f1
    metrics_results['overall_tp'] = overall_tp
    metrics_results['overall_fp'] = overall_fp
    metrics_results['overall_fn'] = overall_fn

    # Calculate per-type metrics
    for entity_type in all_entity_types:
        tp = per_type_stats[entity_type]['tp']
        fp = per_type_stats[entity_type]['fp']
        fn = per_type_stats[entity_type]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if tp == 0 and fp == 0 and fn == 0:
             precision = 1.0
             recall = 1.0
             f1 = 1.0

        metrics_results[f'{entity_type}_precision'] = precision
        metrics_results[f'{entity_type}_recall'] = recall
        metrics_results[f'{entity_type}_f1'] = f1
        metrics_results[f'{entity_type}_tp'] = tp
        metrics_results[f'{entity_type}_fp'] = fp
        metrics_results[f'{entity_type}_fn'] = fn
        
    return metrics_results

def main():
    parser = argparse.ArgumentParser(description="Run NER Simple Baselines (Predict None, Fixed Type for GT Spans).")
    parser.add_argument("--langs", nargs='+', default=['sw', 'ha', 'yo', 'pcm'], help="Languages to evaluate (MasakhaNER defaults).")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument("--sample_percentage", type=float, default=10.0,
                        help="Percentage of samples to use from the specified split (default: 10.0 for 10%%).")
    parser.add_argument("--output_dir", type=str, default="/work/bbd6522/results/ner/simple_baselines_fixed", 
                        help="Directory to save evaluation results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    random.seed(args.seed) # General random seed
    os.makedirs(args.output_dir, exist_ok=True)
    all_run_summaries = []

    for lang_code in args.langs:
        print(f"\n--- Processing NER Fixed Prediction Baselines for Language: {lang_code} ---")

        try:
            # Directly load the sampled percentage using the loader's capabilities
            print(f"Loading {args.sample_percentage}%% of '{args.split}' MasakhaNER samples for {lang_code} with seed {args.seed}...")
            samples_df = load_masakhaner_samples(
                lang_code=lang_code, 
                split=args.split, 
                sample_percentage=args.sample_percentage, # Pass sample_percentage
                seed=args.seed # Pass seed
            )
            
            if samples_df.empty:
                print(f"No MasakhaNER samples loaded for {lang_code} on split '{args.split}' with {args.sample_percentage}%% sampling. Skipping.")
                continue

            # The manual sampling block below is no longer needed as the loader handles it.
            # num_to_sample = int(len(all_samples_df) * (args.sample_percentage / 100.0))
            # if num_to_sample == 0 and len(all_samples_df) > 0:
            #     num_to_sample = 1
            # 
            # if num_to_sample == 0:
            #      print(f"Calculated 0 samples to select for {lang_code} with {args.sample_percentage}%%. Skipping.")
            #      continue
            #
            # print(f"Total MasakhaNER samples for {lang_code} ({args.split}): {len(all_samples_df)}. Sampling {num_to_sample} ({args.sample_percentage}%%).")
            # samples_df = all_samples_df.sample(n=num_to_sample, random_state=args.seed)
            
            if 'tokens' not in samples_df.columns or 'entities' not in samples_df.columns:
                print(f"Required columns ('tokens', 'entities') not found for {lang_code} after loading. Skipping.")
                continue
            print(f"Successfully loaded {len(samples_df)} MasakhaNER samples for {lang_code}.")

        except Exception as e:
            print(f"Error loading or sampling MasakhaNER data for {lang_code}: {e}")
            continue
        
        # Prepare ground truth in the format: List[List[Dict[str, str]]] -> [{'entity': 'text', 'type': 'TYPE'}]
        ground_truth_entities_list = []
        for _, row in samples_df.iterrows():
            formatted_gt_sample = []
            if isinstance(row['entities'], list) and isinstance(row['tokens'], list):
                for entity_dict in row['entities']:
                    if isinstance(entity_dict, dict) and all(k in entity_dict for k in ['start', 'end', 'entity_type']):
                        try:
                            start_idx = int(entity_dict['start'])
                            end_idx = int(entity_dict['end'])
                            if 0 <= start_idx < end_idx <= len(row['tokens']):
                                entity_text = " ".join(row['tokens'][start_idx:end_idx])
                                formatted_gt_sample.append({"entity": entity_text, "type": entity_dict['entity_type']})
                        except (ValueError, TypeError):
                            print(f"Warning: Skipping malformed entity_dict entry: {entity_dict} in lang {lang_code}")
                            pass # Skip malformed entity dicts
            ground_truth_entities_list.append(formatted_gt_sample)

        # Baseline 1: Predict No Entities
        print(f"  Evaluating 'Predict No Entities' baseline for {lang_code}...")
        predict_none_entities_list = [[] for _ in range(len(samples_df))]
        metrics_none = calculate_ner_metrics(ground_truth_entities_list, predict_none_entities_list, MASAKHANER_ENTITY_TYPES)
        
        summary_none = {"language": lang_code, "baseline_strategy": "predict_none", "num_samples": len(samples_df)}
        summary_none.update(metrics_none)
        all_run_summaries.append(summary_none)
        print(f"    Overall F1 for 'Predict No Entities': {metrics_none['overall_f1']:.4f}")

        # Baseline 2: Fixed Entity Type Assignment for Ground Truth Spans
        for fixed_type in MASAKHANER_ENTITY_TYPES:
            print(f"  Evaluating 'Fixed Type ({fixed_type}) for GT Spans' baseline for {lang_code}...")
            predict_fixed_type_list = []
            for gt_sample_entities in ground_truth_entities_list:
                current_sample_preds = []
                for gt_entity in gt_sample_entities:
                    current_sample_preds.append({"entity": gt_entity["entity"], "type": fixed_type})
                predict_fixed_type_list.append(current_sample_preds)
            
            metrics_fixed_type = calculate_ner_metrics(ground_truth_entities_list, predict_fixed_type_list, MASAKHANER_ENTITY_TYPES)
            summary_fixed = {"language": lang_code, "baseline_strategy": f"fixed_type_gt_spans_{fixed_type}", "num_samples": len(samples_df)}
            summary_fixed.update(metrics_fixed_type)
            all_run_summaries.append(summary_fixed)
            print(f"    Overall F1 for 'Fixed Type ({fixed_type}) for GT Spans': {metrics_fixed_type['overall_f1']:.4f}")

    # Save overall summary
    summary_df = pd.DataFrame(all_run_summaries)
    if not summary_df.empty:
        summary_file_path = os.path.join(args.output_dir, "ner_fixed_prediction_summary.csv")
        summary_df.to_csv(summary_file_path, index=False, float_format='%.4f')
        print(f"\nOverall summary for NER fixed prediction baselines saved to {summary_file_path}")
        print("\nSummary Table:")
        # Display relevant columns
        cols_to_show = ['language', 'baseline_strategy', 'num_samples', 'overall_f1', 'overall_precision', 'overall_recall'] 
        for etype in MASAKHANER_ENTITY_TYPES:
            cols_to_show.extend([f'{etype}_f1', f'{etype}_precision', f'{etype}_recall'])
        # Filter columns that actually exist in the dataframe before trying to print
        existing_cols_to_show = [col for col in cols_to_show if col in summary_df.columns]
        print(summary_df[existing_cols_to_show].to_string())
    else: # CORRECTED: else aligned with if
        print("\nNo NER summary data generated.")

if __name__ == "__main__":
    main() 