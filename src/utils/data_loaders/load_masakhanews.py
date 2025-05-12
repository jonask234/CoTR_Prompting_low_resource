import pandas as pd
import os
from typing import Optional, List
from datasets import load_dataset # Import Hugging Face datasets library
from tqdm import tqdm
import random

# Define expected columns and the label column
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'
# Define assumed source column names from TSV and HF Hub
SOURCE_TEXT_COLUMN_TSV = 'text'
SOURCE_LABEL_COLUMN_TSV = 'category'
SOURCE_TEXT_COLUMN_HF = 'text' # Assuming 'text' for HF dataset
SOURCE_LABEL_COLUMN_HF = 'label' # Assuming 'label' for HF dataset (needs verification)

# Base path for local datasets
LOCAL_DATASET_BASE_PATH = "/work/bbd6522/datasets/masakhane_news"
HF_DATASET_NAME = "masakhane/masakhanews"

def _load_samples_from_local_split(lang_code: str, split: str, local_file_path: str, num_samples_to_load: Optional[int]) -> List[dict]:
    """Helper to load and parse samples from a local TSV file."""
    samples = []
    try:
        print(f"DEBUG: Attempting to load {lang_code} samples from local file: {local_file_path}...")
        directory_path = os.path.dirname(local_file_path)
        
        # --- New Detailed Path Debugging ---
        print(f"DEBUG: --- Detailed Path Traversal --- ")
        path_components = os.path.abspath(directory_path).split(os.sep)
        current_path_check = os.sep if os.path.abspath(directory_path).startswith(os.sep) else ""
        
        for component_idx, component in enumerate(path_components):
            if component_idx == 0 and component == "": # Handle root for absolute paths
                if os.sep == '/': # Linux/Mac
                    current_path_part_to_check = os.sep
                else: # Windows might have C:
                    current_path_part_to_check = path_components[0] + os.sep
                    if len(path_components) > 1: component = path_components[1] # Next component for windows
                    else: component = "" # No further components
            else:
                current_path_part_to_check = os.path.join(current_path_check, component)

            if not current_path_part_to_check.strip(): # Skip empty components that might arise from split
                if component_idx > 0: current_path_check = current_path_part_to_check # update current_path_check if it was root
                continue

            exists = os.path.exists(current_path_part_to_check)
            is_dir = os.path.isdir(current_path_part_to_check)
            can_read = os.access(current_path_part_to_check, os.R_OK)
            can_execute = os.access(current_path_part_to_check, os.X_OK) # Execute is needed to list dir contents
            print(f"DEBUG: Checking: {current_path_part_to_check} | Exists: {exists} | IsDir: {is_dir} | Readable: {can_read} | Executable: {can_execute}")
            current_path_check = current_path_part_to_check # Update for next iteration
            if not exists or (component_idx < len(path_components) -1 and not is_dir) or not can_execute:
                print(f"DEBUG: Path traversal issue detected at: {current_path_part_to_check}")
                # break # Optional: break if an issue is found early
        print(f"DEBUG: --- End Detailed Path Traversal --- ")
        # --- End New Detailed Path Debugging ---

        print(f"DEBUG: Final check for directory {directory_path} | Exists: {os.path.exists(directory_path)} | IsDir: {os.path.isdir(directory_path)}")
        print(f"DEBUG: Final check for file {local_file_path} | Exists: {os.path.exists(local_file_path)} | IsFile: {os.path.isfile(local_file_path)}")

        if os.path.exists(local_file_path) and os.path.isfile(local_file_path):
            df = pd.read_csv(local_file_path, sep='\t', on_bad_lines='warn')
            print(f"  Successfully loaded {len(df)} rows from {local_file_path}.")
            
            if SOURCE_TEXT_COLUMN_TSV not in df.columns or SOURCE_LABEL_COLUMN_TSV not in df.columns:
                print(f"ERROR: Expected columns '{SOURCE_TEXT_COLUMN_TSV}' or '{SOURCE_LABEL_COLUMN_TSV}' not found in {local_file_path}.")
                return []
            
            df = df.rename(columns={SOURCE_TEXT_COLUMN_TSV: TEXT_COLUMN, SOURCE_LABEL_COLUMN_TSV: LABEL_COLUMN})
            
            if 'id' not in df.columns:
                df['id'] = [f'{lang_code}_{split}_{i}' for i in df.index]
            else:
                df['id'] = df['id'].astype(str)

            # --- Map numerical labels to string labels ---
            label_map = {
                0: 'health', 1: 'religion', 2: 'politics', 3: 'sports',
                4: 'local', 5: 'business', 6: 'entertainment'
            }
            # Convert label column to numeric first to handle potential strings like "0", "1"
            df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors='coerce')
            df = df.dropna(subset=[LABEL_COLUMN]) # Drop rows where conversion failed
            df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int).map(label_map)
            # Drop rows where mapping might have failed (e.g., unexpected numerical label)
            df = df.dropna(subset=[LABEL_COLUMN])
            # --- End label mapping ---

            dataset_size = len(df)
            df_to_process = df
            if num_samples_to_load is not None and num_samples_to_load < dataset_size:
                df_to_process = df.sample(n=num_samples_to_load, random_state=42)
            
            df_to_process[LABEL_COLUMN] = df_to_process[LABEL_COLUMN].fillna('None').astype(str)
            df_to_process[TEXT_COLUMN] = df_to_process[TEXT_COLUMN].fillna('').astype(str)
            samples = df_to_process[[TEXT_COLUMN, LABEL_COLUMN, 'id']].to_dict('records')

    except Exception as e:
        print(f"Error loading or processing local file {local_file_path}: {e}")
        return []

    print(f"  Successfully prepared {len(samples)} samples from local TSV (before final filtering).")
    return samples

def _load_samples_from_hf_hub(lang_code: str, split: str, num_samples_to_load: Optional[int]) -> List[dict]:
    """Load samples from Hugging Face Hub, handling potential errors."""
    from datasets import load_dataset # Moved import here

    HF_DATASET_NAME = "masakhane/masakhanews"
    # Map our 2-letter codes to Hugging Face's 3-letter config names
    hf_lang_code_map = {
        "en": "eng",
        "sw": "swa",
        "ha": "hau",
        "yo": "yor",
        "pcm": "pcm", # Pidgin
        "ibo": "ibo", # Igbo
        "amh": "amh", # Amharic
        "fra": "fra", # French
        "lin": "lin", # Lingala
        "lug": "lug", # Luganda
        "orm": "orm", # Oromo
        "run": "run", # Rundi
        "sna": "sna", # Shona
        "som": "som", # Somali
        "tir": "tir", # Tigrinya
        "xho": "xho"  # Xhosa
        # Add other mappings if needed
    }
    
    hf_config_name = hf_lang_code_map.get(lang_code, lang_code) # Fallback to original if not in map

    # Map our split names to Hugging Face split names if necessary
    # For MasakhaNEWS, 'train', 'validation', 'test' are standard.
    hf_split_name_map = {
        "train": "train",
        "dev": "validation", # Assuming 'dev' corresponds to 'validation'
        "validation": "validation",
        "test": "test"
    }
    hf_split_name = hf_split_name_map.get(split, split) # Fallback to original split name

    samples_list = []
    print(f"Attempting to load {lang_code} (HF config: {hf_config_name}) samples from Hugging Face Hub: {HF_DATASET_NAME}, split: {hf_split_name}...") # Updated print

    try:
        # Use the mapped hf_config_name
        dataset = load_dataset(HF_DATASET_NAME, name=hf_config_name, split=hf_split_name, trust_remote_code=True)
        
        if dataset is None:
            print(f"Warning: load_dataset returned None for {lang_code} (HF config: {hf_config_name}), split {hf_split_name}.")
            return []

        # Convert to pandas DataFrame for easier manipulation
        df = dataset.to_pandas()

        # Verify and rename columns
        if SOURCE_TEXT_COLUMN_HF not in df.columns or SOURCE_LABEL_COLUMN_HF not in df.columns:
            print(f"ERROR: Expected columns '{SOURCE_TEXT_COLUMN_HF}' or '{SOURCE_LABEL_COLUMN_HF}' not found in HF dataset for {lang_code}.")
            print(f"  Available columns: {df.columns.tolist()}")
            return []
        
        df = df.rename(columns={SOURCE_TEXT_COLUMN_HF: TEXT_COLUMN, SOURCE_LABEL_COLUMN_HF: LABEL_COLUMN})
        
        if 'id' not in df.columns:
             df['id'] = [f'{lang_code}_{split}_{i}' for i in df.index] # Use original split name for ID
        else:
             df['id'] = df['id'].astype(str)

        # --- Map numerical labels to string labels ---
        label_map = {
            0: 'health', 1: 'religion', 2: 'politics', 3: 'sports',
            4: 'local', 5: 'business', 6: 'entertainment'
        }
        # Convert label column to numeric first to handle potential strings like "0", "1"
        df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors='coerce')
        df = df.dropna(subset=[LABEL_COLUMN]) # Drop rows where conversion failed
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int).map(label_map)
        # Drop rows where mapping might have failed (e.g., unexpected numerical label)
        df = df.dropna(subset=[LABEL_COLUMN])
        # --- End label mapping ---

        dataset_size = len(df)
        df_to_process = df
        if num_samples_to_load is not None and num_samples_to_load < dataset_size:
            # HF datasets are often pre-shuffled, but for consistency with local, sample if needed.
            # Can also use dataset.select(range(num_samples_to_load)) if dataset is already shuffled.
            # Using pandas sample for now for direct replacement of local logic.
            df_to_process = df.sample(n=num_samples_to_load, random_state=42)
            print(f"  Sampled {len(df_to_process)} from {dataset_size} available samples for {lang_code} from HF Hub.")
        
        df_to_process[LABEL_COLUMN] = df_to_process[LABEL_COLUMN].fillna('None').astype(str)
        df_to_process[TEXT_COLUMN] = df_to_process[TEXT_COLUMN].fillna('').astype(str)
        samples_list = df_to_process[[TEXT_COLUMN, LABEL_COLUMN, 'id']].to_dict('records')

    except ValueError as ve: # Catch ValueError specifically for BuilderConfig issues
        print(f"Error loading or processing dataset from Hugging Face Hub for {lang_code} (HF config: {hf_config_name}), split {hf_split_name}: {ve}")
        if "BuilderConfig" in str(ve) and "not found" in str(ve):
            print(f"  This often means the language configuration '{hf_config_name}' is incorrect for {HF_DATASET_NAME}.")
            print(f"  Please check available configurations on Hugging Face Hub for {HF_DATASET_NAME}.")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred while loading from Hugging Face Hub for {lang_code} (HF config: {hf_config_name}), split {hf_split_name}: {e}")
        import traceback
        traceback.print_exc()

    num_loaded_hf = len(samples_list)
    print(f"Loaded {num_loaded_hf} raw {lang_code} samples initially from HUGGING FACE HUB ('{split}' split).")

    if not samples_list:
        print(f"WARNING: No {lang_code} samples loaded from any source for split '{split}'!")
        return []

    return samples_list

def load_masakhanews_samples(lang_code: str, num_samples: Optional[int] = None, split: str = "train") -> pd.DataFrame:
    """
    Load MasakhaNEWS samples for a given language and split.
    Tries local loading first, then falls back to Hugging Face Hub.
    """
    print(f"--- Loading MasakhaNEWS --- Lang: {lang_code}, Split: {split}, Target Samples: {'All' if num_samples is None else num_samples}")

    # Define LANG_CODE_MAP for local paths (keeps 'en' as 'en' for local)
    LOCAL_LANG_CODE_MAP = {
        "en": "en",
        "sw": "sw",
        "ha": "ha",
        "yo": "yo",
        "pcm": "pcm",
        "ibo": "ibo",
        "amh": "amh",
        "fra": "fra"
        # Add more if your local structure uses 2-letter codes
    }
    local_lang_dir_name = LOCAL_LANG_CODE_MAP.get(lang_code, lang_code)

    # Construct the full local file path to pass to _load_samples_from_local_split
    local_file_path = os.path.join(LOCAL_DATASET_BASE_PATH, local_lang_dir_name, f"{split}.tsv")

    samples_list = _load_samples_from_local_split(local_lang_dir_name, split, local_file_path, num_samples)

    if not samples_list:
        print(f"Local loading failed or returned no samples for {lang_code}, split {split}. Attempting Hugging Face Hub...")
        # For HF Hub, lang_code is already mapped inside _load_samples_from_hf_hub
        samples_list = _load_samples_from_hf_hub(lang_code, split, num_samples)
    
    if not samples_list:
        print(f"WARNING: No {lang_code} samples loaded from any source for split '{split}'!")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(samples_list)
    
    # Select specific number of samples if requested
    if num_samples is not None and len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
        print(f"  Sampled down to {len(df)} examples for {lang_code} ({split} split).")
    
    print(f"  Successfully loaded {len(df)} samples for {lang_code} ({split} split).")
    return df

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Test Swahili (local might fail, then should try HF Hub)
    swahili_samples_train = load_masakhanews_samples('sw', 5, split='train')
    print("\nSwahili Train Samples (n=5):")
    if not swahili_samples_train.empty:
        print(swahili_samples_train.head())
        print(swahili_samples_train['label'].value_counts())

    # Test Hausa (local might fail, then should try HF Hub)
    hausa_samples_test = load_masakhanews_samples('ha', 10, split='test')
    print("\nHausa Test Samples (n=10):")
    if not hausa_samples_test.empty:
        print(hausa_samples_test.head())
        print(hausa_samples_test['label'].value_counts())

    # Test English (local might fail, then should try HF Hub)
    english_samples_dev = load_masakhanews_samples('en', None, split='dev') # Load all from dev
    print("\nEnglish Dev Samples (all):")
    if not english_samples_dev.empty:
        print(english_samples_dev.head())
        print(f"Total loaded: {len(english_samples_dev)}")
        print(english_samples_dev['label'].value_counts()) 