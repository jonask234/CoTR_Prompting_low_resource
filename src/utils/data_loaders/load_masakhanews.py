import pandas as pd
import os
from typing import Optional, List
# Remove Hugging Face datasets imports
# from datasets import load_dataset, get_dataset_split_names, Dataset
from tqdm import tqdm
import random

# Define expected columns and the label column
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'
# Define assumed source column names from TSV
SOURCE_TEXT_COLUMN = 'text'
SOURCE_LABEL_COLUMN = 'category'

# Base path for local datasets
LOCAL_DATASET_BASE_PATH = "/work/bbd6522/datasets/masakhane_news"

def _load_samples_from_split(lang_code: str, split: str, num_samples_to_load: Optional[int]) -> List[dict]:
    """Helper function to load samples from a specific local TSV split."""
    samples = []
    file_path = os.path.join(LOCAL_DATASET_BASE_PATH, lang_code, f"{split}.tsv")
    
    print(f"Attempting to load {lang_code} samples from local file: {file_path}...")

    if not os.path.exists(file_path):
        print(f"ERROR: Local file not found: {file_path}")
        return []

    try:
        # Load the TSV file using pandas
        df = pd.read_csv(file_path, sep='\t', on_bad_lines='warn')
        print(f"  Successfully loaded {len(df)} rows from {file_path}.")
        
        # Check if expected source columns exist
        if SOURCE_TEXT_COLUMN not in df.columns or SOURCE_LABEL_COLUMN not in df.columns:
            print(f"ERROR: Expected columns '{SOURCE_TEXT_COLUMN}' or '{SOURCE_LABEL_COLUMN}' not found in {file_path}.")
            print(f"  Available columns: {df.columns.tolist()}")
            return []
        
        # Rename columns to expected format ('text', 'label')
        df = df.rename(columns={SOURCE_TEXT_COLUMN: TEXT_COLUMN, SOURCE_LABEL_COLUMN: LABEL_COLUMN})
        
        # Add an ID column if it doesn't exist (using index)
        if 'id' not in df.columns:
             df['id'] = [f'{lang_code}_{split}_{i}' for i in df.index]
        else:
             # Ensure ID is string
             df['id'] = df['id'].astype(str)

        # Sample if needed BEFORE converting to dicts (more efficient)
        dataset_size = len(df)
        df_to_process = df
        effective_total = dataset_size

        if num_samples_to_load is not None and num_samples_to_load < dataset_size:
            print(f"  Applying RANDOM sampling for {lang_code}: Selecting {num_samples_to_load} samples...")
            df_to_process = df.sample(n=num_samples_to_load, random_state=42)
            effective_total = num_samples_to_load
            print(f"  Selected {len(df_to_process)} samples after sampling.")
        elif num_samples_to_load is not None:
            print(f"  Requested samples ({num_samples_to_load}) >= dataset size ({dataset_size}). Processing all.")
        else:
            print("  Processing all samples in the split.")

        # Convert DataFrame rows to list of dictionaries (expected format by rest of the code)
        # Ensure NaN labels become None (string) for consistency with filtering later
        df_to_process[LABEL_COLUMN] = df_to_process[LABEL_COLUMN].fillna('None').astype(str)
        df_to_process[TEXT_COLUMN] = df_to_process[TEXT_COLUMN].fillna('').astype(str)
        
        samples = df_to_process[[TEXT_COLUMN, LABEL_COLUMN, 'id']].to_dict('records')

    except Exception as e:
        print(f"Error loading or processing local file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

    print(f"  Successfully prepared {len(samples)} samples (before final filtering). ")
    return samples

def load_masakhanews_samples(lang_code: str, num_samples: Optional[int] = None, split: str = "train") -> pd.DataFrame:
    """
    Load MasakhaNEWS samples for a specific language FROM LOCAL TSV FILES.

    Args:
        lang_code: Language code ('swa' for Swahili, 'hau' for Hausa).
        num_samples: Target number of samples to load (None for all available).
                     Samples are randomly selected if num_samples < dataset size.
        split: Dataset split to use ('train', 'dev', 'test').

    Returns:
        DataFrame containing the samples ('text', 'label', 'id').
    """
    print(f"--- Loading MasakhaNEWS from LOCAL TSV --- Lang: {lang_code}, Split: {split}")

    # Define valid splits for local files
    valid_splits = ['train', 'dev', 'test']
    if split not in valid_splits:
        print(f"WARN: Invalid split '{split}' requested. Defaulting to 'train'. Valid splits: {valid_splits}")
        split = 'train'

    # Load samples (potentially including rows with bad labels initially)
    all_samples_raw = _load_samples_from_split(lang_code, split, num_samples)
    
    print(f"Loaded {len(all_samples_raw)} raw {lang_code} samples initially from the local '{split}.tsv' file.")

    if not all_samples_raw:
        print(f"WARNING: No {lang_code} samples loaded from the local '{split}.tsv' file!")
        return pd.DataFrame({'text': [], 'label': [], 'id': []})
        
    # Convert to DataFrame
    df = pd.DataFrame(all_samples_raw)
    
    # Ensure required columns exist after conversion, even if empty
    for col in [TEXT_COLUMN, LABEL_COLUMN, 'id']:
        if col not in df.columns:
            df[col] = None # Add missing column filled with None
    
    df = df[[TEXT_COLUMN, LABEL_COLUMN, 'id']] # Reorder/select required columns

    # --- Diagnostic: Print unique labels found BEFORE filtering --- 
    if not df.empty:
        # Convert label column to string to handle potential mixed types before unique()
        unique_labels = df[LABEL_COLUMN].astype(str).unique()
        print(f"DEBUG: Unique values found in 'label' column for {lang_code} ({split}): {unique_labels}")
    else:
        print(f"DEBUG: DataFrame empty for {lang_code} ({split}) before label filtering.")
    # --- End Diagnostic --- 
    
    # --- Post-filtering of labels (Keep this logic) --- 
    # Ensure label column is string type before checking
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)
    # Remove rows with None, empty, whitespace-only, or the literal string 'None' labels
    initial_count = len(df)
    # Make filtering robust: check for actual None, empty string, or the string 'None' (case-insensitive)
    df = df[df[LABEL_COLUMN].notna() & (df[LABEL_COLUMN].str.strip() != '') & (df[LABEL_COLUMN].str.lower() != 'none')]
    filtered_count = len(df)
    print(f"Filtered samples: Removed {initial_count - filtered_count} rows due to invalid/empty/'None' labels.")
    # --------------------------------
    
    if df.empty:
        print(f"WARNING: No {lang_code} samples remaining after filtering for valid labels!")
        return pd.DataFrame({'text': [], 'label': [], 'id': []})

    return df

# Example usage (optional, for testing) - Update paths if needed
if __name__ == '__main__':
    # Test Hausa (load 100 random samples from train)
    hausa_samples = load_masakhanews_samples('hau', 100, split='train')
    print("\nRandom Hausa Samples (n=100 from local train.tsv):")
    if not hausa_samples.empty:
        print(hausa_samples.head())
        print(hausa_samples['label'].value_counts())

    # Test Swahili (load all samples from test)
    swahili_samples_test = load_masakhanews_samples('swa', None, split='test')
    print("\nSwahili Samples (all from local test.tsv):")
    if not swahili_samples_test.empty:
        print(swahili_samples_test.head())
        print(f"Total loaded: {len(swahili_samples_test)}")
        print(swahili_samples_test['label'].value_counts()) 