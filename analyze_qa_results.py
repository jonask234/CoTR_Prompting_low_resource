import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
import plotly.io as pio

# --- Configuration ---
BASE_RESULTS_DIR = "/work/bbd6522/results"
# Define the output directory for plots
PLOTS_OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "vizualizations_qa") 
# Set default renderer for environments like VSCode/servers
pio.renderers.default = "vscode" 
# --- End Configuration ---

def parse_filename(filepath):
    """Extracts metadata (pipeline, dataset, lang, model) from filename."""
    filename = os.path.basename(filepath)
    parts = filename.replace('.csv', '').split('_')
    # Expected format: {pipeline}_qa_{dataset}_{lang}_{model}.csv
    # Example: baseline_qa_tydiqa_sw_Qwen2-7B.csv -> parts = ['baseline', 'qa', 'tydiqa', 'sw', 'Qwen2-7B']
    if len(parts) < 5:
        print(f"WARN: Could not parse filename: {filename}. Skipping.")
        return None
    
    pipeline = parts[0] # baseline or cotr
    dataset = parts[2]  # tydiqa
    lang_code = parts[3] # sw or id
    # Model name might contain underscores, join the rest
    model_name = '_'.join(parts[4:]) 
    
    # Basic validation
    if pipeline not in ['baseline', 'cotr'] or dataset != 'tydiqa':
         print(f"WARN: Unexpected parts in filename: {filename}. Skipping.")
         return None
         
    return {'pipeline': pipeline, 'language': lang_code, 'model': model_name}

def load_and_combine_results(base_dir):
    """Loads all result CSVs, adds metadata, and combines them."""
    all_data = []
    # Search for CSV files in baseline and cotr subdirectories
    search_pattern = os.path.join(base_dir, '**', '*.csv') 
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print(f"ERROR: No CSV files found in {base_dir} or its subdirectories.")
        return pd.DataFrame()

    print(f"Found {len(csv_files)} CSV files. Loading...")
    
    for fpath in csv_files:
        # Skip summary files for row-level analysis if they exist in same dirs
        if 'summary' in os.path.basename(fpath):
            continue
            
        metadata = parse_filename(fpath)
        if metadata is None:
            continue
            
        try:
            df = pd.read_csv(fpath)
            # Add metadata columns
            df['pipeline'] = metadata['pipeline']
            df['language'] = metadata['language']
            df['model'] = metadata['model']
            df['filepath'] = fpath # Keep track of origin file
            all_data.append(df)
            # print(f"  Loaded {os.path.basename(fpath)}")
        except pd.errors.EmptyDataError:
            print(f"WARN: File is empty, skipping: {fpath}")
        except Exception as e:
            print(f"ERROR loading file {fpath}: {e}")
            
    if not all_data:
        print("ERROR: No valid data loaded from CSV files.")
        return pd.DataFrame()
        
    print("Combining data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")
    # print("Combined DataFrame columns:", combined_df.columns.tolist()) # Debugging columns
    return combined_df

# --- Main Analysis Logic ---

# 1. Load and Combine Data
results_df = load_and_combine_results(BASE_RESULTS_DIR)

if not results_df.empty:
    # 2. Aggregate Metrics
    print("\n--- Aggregating Metrics ---")
    
    # Define aggregations - use nanmean to handle potential NaNs safely
    agg_funcs = {
        'f1_score': np.nanmean,
        # Only aggregate translation metrics if they exist (primarily in CoTR)
    }
    if 'average_translation_quality' in results_df.columns:
         agg_funcs['average_translation_quality'] = np.nanmean
    if 'question_translation_quality' in results_df.columns:
         agg_funcs['question_translation_quality'] = np.nanmean
    if 'answer_translation_quality' in results_df.columns:
         agg_funcs['answer_translation_quality'] = np.nanmean
    if 'comet_source_to_en' in results_df.columns:
         agg_funcs['comet_source_to_en'] = np.nanmean
    if 'comet_en_to_source' in results_df.columns:
         agg_funcs['comet_en_to_source'] = np.nanmean
         
    aggregated_results = results_df.groupby(['pipeline', 'language', 'model']).agg(
       agg_funcs
    ).reset_index()

    print("\n--- Average Scores ---")
    print(aggregated_results.to_string()) # Print full table

    # Create the output directory for plots if it doesn't exist
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    print(f"\nSaving plots to: {PLOTS_OUTPUT_DIR}")

    # 3. Visualize Results with Plotly
    print("\n--- Generating Visualizations ---")

    # Plot 1: Average F1 Score Comparison
    try:
        fig_f1 = px.bar(
            aggregated_results,
            x='language',
            y='f1_score',
            color='model',
            barmode='group',
            facet_col='pipeline', # Separate columns for 'baseline' and 'cotr'
            title='Average F1 Score Comparison (TyDi QA)',
            labels={'f1_score': 'Average F1 Score', 'language': 'Language', 'model': 'Model', 'pipeline': 'Pipeline'}
        )
        fig_f1.update_layout(yaxis_range=[0, max(aggregated_results['f1_score'].max() * 1.1, 0.1)]) # Adjust y-axis
        # Save as PDF instead of HTML
        f1_filename_pdf = "f1_score_comparison.pdf"
        f1_save_path = os.path.join(PLOTS_OUTPUT_DIR, f1_filename_pdf)
        try:
            fig_f1.write_image(f1_save_path)
            print(f"Saved F1 Score Comparison plot to: {f1_save_path}")
        except ValueError as ve:
            print(f"ERROR saving F1 plot as PDF: {ve}")
        # fig_f1.write_html(f1_filename) # Keep HTML save commented out or remove
        # print(f"Saved F1 Score Comparison plot to: {f1_filename}")
    except Exception as e:
        print(f"Could not generate or save F1 plot: {e}")

    # Plot 2: Average Translation Quality (CoTR only)
    if 'average_translation_quality' in aggregated_results.columns:
        cotr_results = aggregated_results[aggregated_results['pipeline'] == 'cotr'].copy()
        if not cotr_results.empty:
            # Check if the column actually has non-NaN values
            if cotr_results['average_translation_quality'].notna().any():
                 try:
                     fig_trans = px.bar(
                         cotr_results,
                         x='language',
                         y='average_translation_quality',
                         color='model',
                         barmode='group',
                         title='Average Translation Quality (CoTR - Normalized 0-1)',
                         labels={'average_translation_quality': 'Avg. Translation Quality', 'language': 'Language', 'model': 'Model'}
                     )
                     fig_trans.update_layout(yaxis_range=[0, max(cotr_results['average_translation_quality'].max() * 1.1, 0.1)]) # Adjust y-axis
                     # Save as PDF instead of HTML
                     trans_filename_pdf = "translation_quality_comparison.pdf"
                     trans_save_path = os.path.join(PLOTS_OUTPUT_DIR, trans_filename_pdf)
                     try:
                         fig_trans.write_image(trans_save_path)
                         print(f"Saved Translation Quality plot to: {trans_save_path}")
                     except ValueError as ve:
                         print(f"ERROR saving Translation Quality plot as PDF: {ve}")
                        # fig_trans.write_html(trans_filename)
                        # print(f"Saved Translation Quality plot to: {trans_filename}")
                 except Exception as e:
                     print(f"Could not generate or save Translation Quality plot: {e}")
            else:
                 print("Skipping translation quality plot: No valid data points found.")
        else:
            print("Skipping translation quality plot: No CoTR results found.")

    # Plot 3: Raw COMET Scores (CoTR only, if available)
    if 'comet_source_to_en' in aggregated_results.columns and 'comet_en_to_source' in aggregated_results.columns:
         cotr_results_comet = aggregated_results[aggregated_results['pipeline'] == 'cotr'].copy()
         if not cotr_results_comet.empty:
             # Melt dataframe for easier plotting of both COMET scores
             comet_melted = pd.melt(cotr_results_comet, 
                                    id_vars=['language', 'model'], 
                                    value_vars=['comet_source_to_en', 'comet_en_to_source'],
                                    var_name='comet_direction', 
                                    value_name='comet_score')
                                    
             # Only plot if there are valid scores
             if comet_melted['comet_score'].notna().any():
                 try:
                     fig_comet = px.bar(
                         comet_melted,
                         x='language',
                         y='comet_score',
                         color='model',
                         barmode='group',
                         facet_col='comet_direction',
                         title='Average Raw COMET Scores (CoTR)',
                         labels={'comet_score': 'Avg. COMET Score', 'language': 'Language', 'model': 'Model', 'comet_direction': 'Translation Direction'}
                     )
                     # Save as PDF instead of HTML
                     comet_filename_pdf = "comet_score_comparison.pdf"
                     comet_save_path = os.path.join(PLOTS_OUTPUT_DIR, comet_filename_pdf)
                     try:
                         fig_comet.write_image(comet_save_path)
                         print(f"Saved Raw COMET Scores plot to: {comet_save_path}")
                     except ValueError as ve:
                         print(f"ERROR saving COMET score plot as PDF: {ve}")
                         print("Ensure kaleido is installed correctly ('pip install kaleido')")
                    # fig_comet.write_html(comet_filename)
                    # print(f"Saved Raw COMET Scores plot to: {comet_filename}")
                 except Exception as e:
                     print(f"Could not generate or save COMET score plot: {e}")
             else:
                 print("Skipping COMET score plot: No valid data points found.")

else:
    print("Analysis aborted due to issues loading data.")

print("\n--- Analysis Complete ---") 