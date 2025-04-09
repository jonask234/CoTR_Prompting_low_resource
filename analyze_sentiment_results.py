# Point to the sentiment results directory
BASE_RESULTS_DIR = "/work/bbd6522/results/sentiment" 
# Output directory for plots - Create a new one for sentiment
PLOTS_OUTPUT_DIR = "/work/bbd6522/results/vizualizations_sentiment" 
# Ensure kaleido is available for PDF export
# pio.renderers.default = "vscode" # Not needed for file export 

import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
import plotly.io as pio
from sklearn.metrics import f1_score # Import f1_score

def parse_sentiment_filename(filepath):
    """Extracts metadata from sentiment result filename."""
    filename = os.path.basename(filepath)
    parts = filename.replace('.csv', '').split('_')
    # Expected format: {pipeline}_sentiment_{dataset}_{lang}_{model}.csv
    # Example: baseline_sentiment_afrisenti_ha_Qwen2-7B.csv 
    # Parts: ['baseline', 'sentiment', 'afrisenti', 'ha', 'Qwen2-7B']
    if len(parts) < 5 or parts[1] != 'sentiment':
        print(f"WARN: Skipping file with unexpected format: {filename}")
        return None
    
    pipeline = parts[0] # baseline or cotr
    dataset = parts[2]  # afrisenti
    lang_code = parts[3] # sw or ha
    model_name = '_'.join(parts[4:]) 
    
    if pipeline not in ['baseline', 'cotr']:
         print(f"WARN: Unexpected pipeline '{pipeline}' in filename: {filename}. Skipping.")
         return None
         
    return {'pipeline': pipeline, 'dataset': dataset, 'language': lang_code, 'model': model_name}

def load_and_combine_sentiment_results(base_dir):
    """Loads sentiment result CSVs, adds metadata, and combines them."""
    all_data = []
    search_pattern = os.path.join(base_dir, '**', '*.csv') 
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print(f"ERROR: No CSV files found in {base_dir} or its subdirectories.")
        return pd.DataFrame()

    print(f"Found {len(csv_files)} CSV files. Loading...")
    
    for fpath in csv_files:
        # Skip summary files
        if 'summary' in os.path.basename(fpath):
            continue
            
        metadata = parse_sentiment_filename(fpath)
        if metadata is None:
            continue
            
        try:
            df = pd.read_csv(fpath)
            df['pipeline'] = metadata['pipeline']
            df['dataset'] = metadata['dataset']
            df['language'] = metadata['language']
            df['model'] = metadata['model']
            all_data.append(df)
        except pd.errors.EmptyDataError:
            print(f"WARN: File is empty, skipping: {fpath}")
        except Exception as e:
            print(f"ERROR loading file {fpath}: {e}")
            
    if not all_data:
        print("ERROR: No valid data loaded from CSV files.")
        return pd.DataFrame()
        
    print("Combining sentiment data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined Sentiment DataFrame shape: {combined_df.shape}")
    return combined_df

# --- Main Analysis Logic ---

# 1. Load and Combine Data
results_df = load_and_combine_sentiment_results(BASE_RESULTS_DIR)

if not results_df.empty:
    # 2. Aggregate Metrics - Sentiment specific
    print("\n--- Aggregating Sentiment Metrics ---")
    
    if 'ground_truth_label' not in results_df.columns or 'predicted_label' not in results_df.columns:
        print("ERROR: Cannot calculate metrics. Missing 'ground_truth_label' or 'predicted_label' column.")
    else:
        # Group by experiment configuration
        grouped = results_df.groupby(['pipeline', 'language', 'model', 'dataset'])
        
        aggregated_metrics = []
        for name, group in grouped:
            pipeline, language, model, dataset = name
            y_true = group['ground_truth_label']
            y_pred = group['predicted_label']
            
            # Calculate metrics for this group
            valid_labels = set(y_true.unique()) | set(y_pred.unique()) - {'[Unknown]'} # Include all seen labels except Unknown
            accuracy = np.mean(y_true == y_pred) # Simple accuracy
            macro_f1 = f1_score(y_true, y_pred, average='macro', labels=list(valid_labels), zero_division=0)
            
            # Calculate avg translation quality if CoTR
            avg_translation_quality = np.nan
            if pipeline == 'cotr' and 'translation_quality_overlap' in group.columns:
                avg_translation_quality = group['translation_quality_overlap'].mean()

            aggregated_metrics.append({
                'pipeline': pipeline,
                'language': language,
                'model': model,
                'dataset': dataset,
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'avg_translation_quality': avg_translation_quality
            })
            
        aggregated_results = pd.DataFrame(aggregated_metrics)

        print("\n--- Average Sentiment Scores --- ")
        print(aggregated_results.to_string())

        # Create the output directory for plots if it doesn't exist
        os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
        print(f"\nSaving plots to: {PLOTS_OUTPUT_DIR}")

        # 3. Visualize Results with Plotly
        print("\n--- Generating Sentiment Visualizations ---")

        # Plot 1: Average Accuracy Comparison
        try:
            fig_acc = px.bar(
                aggregated_results,
                x='language',
                y='accuracy',
                color='model',
                barmode='group',
                facet_col='pipeline', 
                title='Average Accuracy Comparison (Sentiment Analysis)',
                labels={'accuracy': 'Average Accuracy', 'language': 'Language', 'model': 'Model', 'pipeline': 'Pipeline'}
            )
            fig_acc.update_layout(yaxis_range=[0, 1]) # Accuracy is 0-1
            acc_filename_pdf = "sentiment_accuracy_comparison.pdf"
            acc_save_path = os.path.join(PLOTS_OUTPUT_DIR, acc_filename_pdf)
            try:
                fig_acc.write_image(acc_save_path)
                print(f"Saved Accuracy Comparison plot to: {acc_save_path}")
            except ValueError as ve:
                print(f"ERROR saving Accuracy plot as PDF: {ve}")
                print("Ensure kaleido is installed correctly ('pip install kaleido')")
        except Exception as e:
            print(f"Could not generate or save Accuracy plot: {e}")

        # Plot 2: Average Macro F1 Score Comparison
        try:
            fig_f1 = px.bar(
                aggregated_results,
                x='language',
                y='macro_f1',
                color='model',
                barmode='group',
                facet_col='pipeline', 
                title='Average Macro F1 Score Comparison (Sentiment Analysis)',
                labels={'macro_f1': 'Average Macro F1 Score', 'language': 'Language', 'model': 'Model', 'pipeline': 'Pipeline'}
            )
            fig_f1.update_layout(yaxis_range=[0, 1]) # F1 is 0-1
            f1_filename_pdf = "sentiment_macro_f1_comparison.pdf"
            f1_save_path = os.path.join(PLOTS_OUTPUT_DIR, f1_filename_pdf)
            try:
                fig_f1.write_image(f1_save_path)
                print(f"Saved Macro F1 Score Comparison plot to: {f1_save_path}")
            except ValueError as ve:
                print(f"ERROR saving Macro F1 plot as PDF: {ve}")
                print("Ensure kaleido is installed correctly ('pip install kaleido')")
        except Exception as e:
            print(f"Could not generate or save Macro F1 plot: {e}")

        # Plot 3: Average Translation Quality (CoTR only)
        cotr_results = aggregated_results[aggregated_results['pipeline'] == 'cotr'].copy()
        if not cotr_results.empty and cotr_results['avg_translation_quality'].notna().any():
            try:
                fig_trans = px.bar(
                    cotr_results,
                    x='language',
                    y='avg_translation_quality',
                    color='model',
                    barmode='group',
                    title='Average Translation Quality (Sentiment CoTR - Token Overlap)',
                    labels={'avg_translation_quality': 'Avg. Translation Quality (Overlap)', 'language': 'Language', 'model': 'Model'}
                )
                fig_trans.update_layout(yaxis_range=[0, max(cotr_results['avg_translation_quality'].max() * 1.1, 0.1)]) 
                trans_filename_pdf = "sentiment_translation_quality_comparison.pdf"
                trans_save_path = os.path.join(PLOTS_OUTPUT_DIR, trans_filename_pdf)
                try:
                    fig_trans.write_image(trans_save_path)
                    print(f"Saved Translation Quality plot to: {trans_save_path}")
                except ValueError as ve:
                    print(f"ERROR saving Translation Quality plot as PDF: {ve}")
                    print("Ensure kaleido is installed correctly ('pip install kaleido')")
            except Exception as e:
                print(f"Could not generate or save Translation Quality plot: {e}")
        else:
            print("Skipping translation quality plot: No valid CoTR data points found.")
else:
    print("Analysis aborted due to issues loading data.")

print("\n--- Sentiment Analysis Complete ---")