import pandas as pd
from datasets import load_dataset, concatenate_datasets
from typing import Optional

# --- Mapping from common codes to AfriQA config names ---
AFRIQA_CONFIG_MAP = {
    'sw': 'swa', # Swahili
    'ha': 'hau', # Hausa
    # Add mappings for other languages if needed, ensure they match the available configs:
    # 'bem': 'bem', 'fon': 'fon', 'ibo': 'ibo', 'kin': 'kin', 
    # 'twi': 'twi', 'wol': 'wol', 'yor': 'yor', 'zul': 'zul'
}
# -----------------------------------------------------

def load_afriqa_samples(
    lang_code: str, 
    num_samples: Optional[int] = None, 
    split: str = 'validation' # AfriQA primarily uses train/validation
) -> pd.DataFrame:
    """
    Loads question answering samples from the AfriQA dataset for a specific language.

    Args:
        lang_code: Language code (e.g., 'sw', 'ha', 'en').
        num_samples: Number of samples to load. Loads all if None.
        split: Dataset split to use ('train' or 'validation'). 
               Defaults to 'validation' as it's commonly used for evaluation.

    Returns:
        DataFrame with 'id', 'title', 'context', 'question', 'answers' columns.
        'answers' column contains a dictionary {'text': [answer_string], 'answer_start': [start_index]}.
    """
    dataset_name = "masakhane/afriqa"
    
    # --- Map lang_code to config_name and handle unavailable langs ---
    if lang_code == 'en':
        print(f"WARNING: English ('en') is not available in the {dataset_name} dataset. Returning empty DataFrame.")
        return pd.DataFrame()
        
    config_name = AFRIQA_CONFIG_MAP.get(lang_code)
    if not config_name:
        print(f"ERROR: Language code '{lang_code}' is not supported or mapped for {dataset_name}. Supported: {list(AFRIQA_CONFIG_MAP.keys())}. Returning empty DataFrame.")
        # You could also attempt to list available configs directly from HF if needed
        return pd.DataFrame()
    # ---------------------------------------------------------------

    print(f"Loading {dataset_name} dataset for language config: {config_name}, split: {split}...")
    
    try:
        # Load the specific language configuration and split
        dataset = load_dataset(dataset_name, name=config_name, split=split, cache_dir="/work/bbd6522/cache_dir")
        print(f"  Raw columns loaded: {dataset.column_names}") # Debug print

        # --- Check for essential columns based on observed error ---
        # Context is absolutely essential for QA
        if 'context' not in dataset.column_names:
             print(f"FATAL ERROR: 'context' column is missing from the loaded dataset for {config_name}. Cannot perform QA.")
             return pd.DataFrame()
             
        required_data_cols = ['question', 'answers'] # Check columns reported in error
        if not all(col in dataset.column_names for col in required_data_cols):
             print(f"ERROR: Dataset missing expected columns (question, answers) in {dataset_name} for {config_name}.")
             print(f"Found columns: {dataset.column_names}")
             return pd.DataFrame() 
        # ---------------------------------------------------------

        total_available = len(dataset)
        print(f"  Total samples available in split: {total_available}")

        if num_samples is not None:
            if num_samples > total_available:
                print(f"  Warning: Requested {num_samples} samples, but only {total_available} are available. Loading all.")
                num_samples = total_available
            elif num_samples <= 0:
                 print(f"  Warning: Requested {num_samples} samples. Loading all available samples instead.")
                 num_samples = total_available

            # Shuffle and select samples if num_samples is specified and less than total
            if num_samples < total_available:
                 print(f"  Selecting {num_samples} samples...")
                 dataset = dataset.shuffle(seed=42).select(range(num_samples))
            else:
                 print(f"  Loading all {total_available} samples.")
        else:
             print(f"  Loading all {total_available} samples.")

        # Convert to Pandas DataFrame
        samples_df = dataset.to_pandas()
        
        # --- Add 'id' if missing --- 
        if 'id' not in samples_df.columns:
             print("  Generating 'id' column as it was missing.")
             samples_df['id'] = [f"{config_name}_{split}_{i}" for i in range(len(samples_df))] # Generate unique IDs
        # -------------------------
        
        # --- Update required columns check for DataFrame ---
        # We need 'id', 'context', 'question', and 'answers' for processing
        required_df_cols = ['id', 'context', 'question', 'answers']
        if not all(col in samples_df.columns for col in required_df_cols):
            print(f"ERROR: DataFrame is missing required columns after conversion/ID generation. Columns found: {samples_df.columns.tolist()}")
            return pd.DataFrame()
        # ---------------------------------------------------
            
        # --- Extract ground truth from 'answers' column --- 
        # Try to handle different possible structures in 'answers'
        def get_answer_from_answers_col(ans_data):
            if isinstance(ans_data, dict) and 'text' in ans_data and isinstance(ans_data['text'], list) and ans_data['text']:
                return ans_data['text'][0] # SQuAD-like format
            elif isinstance(ans_data, list) and ans_data: 
                return ans_data[0] # List format (like original answer_text)
            elif isinstance(ans_data, str):
                 return ans_data # Simple string format
            return None 

        samples_df['ground_truth'] = samples_df['answers'].apply(get_answer_from_answers_col)
        # ---------------------------------------------------
        
        # Drop rows where ground_truth could not be extracted
        samples_df.dropna(subset=['ground_truth'], inplace=True)
        
        if samples_df.empty:
             print(f"WARNING: No valid samples remaining after extracting ground truth answers for {lang_code}.")
        else:
            print(f"Successfully loaded and processed {len(samples_df)} samples for {lang_code}.")

        # --- Return standard columns plus the original answer columns for reference if needed ---
        final_cols = ['id', 'context', 'question', 'answers', 'ground_truth']
        if 'title' in samples_df.columns:
            final_cols.insert(1, 'title') 
        # Include other relevant columns from the dataset if needed for analysis
        other_cols_to_keep = ['lang', 'split', 'translated_question', 'translated_answer', 'translation_type']
        for col in other_cols_to_keep:
             if col in samples_df.columns and col not in final_cols:
                 final_cols.append(col)
            
        return samples_df[[col for col in final_cols if col in samples_df.columns]]

    except Exception as e:
        print(f"ERROR loading or processing dataset {dataset_name} for language {config_name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# Example usage:
if __name__ == '__main__':
    sw_samples = load_afriqa_samples('sw', num_samples=5, split='validation')
    print("\nSwahili Samples Head:")
    print(sw_samples.head())
    
    ha_samples = load_afriqa_samples('ha', num_samples=5, split='validation')
    print("\nHausa Samples Head:")
    print(ha_samples.head())

    en_samples = load_afriqa_samples('en', num_samples=5, split='validation')
    print("\nEnglish Samples Head:")
    print(en_samples.head()) 