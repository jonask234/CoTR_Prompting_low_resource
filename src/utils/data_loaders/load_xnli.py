import os
import pandas as pd
import random
from datasets import load_dataset, get_dataset_split_names
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# Standard XNLI label mapping
XNLI_LABEL_MAP: Dict[int, str] = {
    0: 'entailment',
    1: 'neutral',
    2: 'contradiction'
}

# XNLI languages available in facebook/xnli
# From documentation: ar, bg, de, el, en, es, fr, hi, ru, sw, th, tr, ur, vi, zh
XNLI_SUPPORTED_LANGUAGES = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

# --- Available languages in XNLI and their specific Hugging Face dataset identifiers ---
# (usually it's 'xnli', 'language_code')
XNLI_LANG_CONFIG_MAP = {
    "en": ("xnli", "en"),
    "fr": ("xnli", "fr"),
    "es": ("xnli", "es"),
    "de": ("xnli", "de"),
    "el": ("xnli", "el"), # Greek
    "bg": ("xnli", "bg"), # Bulgarian
    "ru": ("xnli", "ru"),
    "tr": ("xnli", "tr"),
    "ar": ("xnli", "ar"),
    "vi": ("xnli", "vi"),
    "th": ("xnli", "th"),
    "zh": ("xnli", "zh"), # Chinese
    "hi": ("xnli", "hi"),
    "sw": ("xnli", "sw"), # Swahili
    "ur": ("xnli", "ur"),
    # Removed "yo" and "ha" as they are not direct valid configs for "xnli"
    # To support them, one would need to use "all_languages" and filter,
    # or find a different NLI dataset that has them as direct configs.
}

def load_xnli_samples(
    lang_code: str,
    num_samples: Optional[int] = None, # Changed from sample_percentage
    split: str = 'test', # Default to 'test' split for evaluation
    seed: int = 42 # Added seed parameter
) -> pd.DataFrame:
    """
    Load NLI samples for a given language from the XNLI dataset.

    Args:
        lang_code (str): The language code (e.g., 'en', 'sw').
        num_samples (Optional[int]): The number of samples to return. If None, all samples from the split are returned.
        split (str): The dataset split to load ('train', 'validation', 'test'). Default is 'test'.
        seed (int): Random seed for shuffling if num_samples is specified.

    Returns:
        pd.DataFrame: A DataFrame containing the samples, with columns 'premise', 'hypothesis', 'label' (string), and 'original_label_int'.
                      Returns an empty DataFrame if the language is not supported or data loading fails.
    """
    logger.info(f"Requesting XNLI samples for language: {lang_code}, split: {split}, num_samples: {num_samples}")

    if lang_code not in XNLI_LANG_CONFIG_MAP:
        logger.warning(
            f"Language code '{lang_code}' is not supported by the current XNLI_LANG_CONFIG_MAP "
            f"or does not have a direct configuration in the 'xnli' dataset. "
            f"Supported direct configs mapped: {list(XNLI_LANG_CONFIG_MAP.keys())}. "
            f"Please check Hugging Face for available 'xnli' dataset configurations. "
            f"Skipping loading for '{lang_code}'."
        )
        return pd.DataFrame() # Return empty DataFrame if lang_code is not supported

    dataset_name, config_name = XNLI_LANG_CONFIG_MAP[lang_code]
    
    try:
        logging.info(f"Loading XNLI dataset: '{dataset_name}', config: '{config_name}', split: '{split}' for language: {lang_code}")
        # Load the specified split
        dataset = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
        
        # Convert to pandas DataFrame for easier manipulation
        all_samples_df = pd.DataFrame(dataset)
        
        if all_samples_df.empty:
            logging.warning(f"No samples found for XNLI lang '{lang_code}', split '{split}'.")
            return pd.DataFrame()

        # Rename columns for consistency if necessary (assuming standard XNLI structure)
        # XNLI typically has 'premise', 'hypothesis', 'label'
        # We need to ensure these columns exist.
        required_cols = {'premise', 'hypothesis', 'label'}
        if not required_cols.issubset(all_samples_df.columns):
            # Attempt common renames if columns are different (e.g. from multilingual versions)
            # This part might need adjustment based on actual column names in the dataset if they vary.
            logging.warning(f"Standard columns {required_cols} not all found in XNLI data for {lang_code}. Columns found: {all_samples_df.columns}. Attempting to map if possible or proceed if direct.")
            # For now, assume columns are correctly named. If issues arise, specific mapping may be needed here.

        # Map integer labels to string labels
        if 'label' in all_samples_df.columns and pd.api.types.is_numeric_dtype(all_samples_df['label']):
            all_samples_df['original_label_int'] = all_samples_df['label']
            all_samples_df['label'] = all_samples_df['label'].map(XNLI_LABEL_MAP)
        else:
            # If 'label' is already string or missing, handle appropriately
            logging.warning(f"Label column in XNLI for {lang_code} is not numeric or missing. Cannot map to string labels using XNLI_LABEL_MAP. Current labels: {all_samples_df['label'].head() if 'label' in all_samples_df.columns else 'N/A'}")
            # Add 'original_label_int' as -1 if label wasn't numeric
            all_samples_df['original_label_int'] = -1 
        
        # Select relevant columns (premise, hypothesis, label, original_label_int)
        # Ensure all expected columns are present, fill with default if not
        final_columns = ['premise', 'hypothesis', 'label', 'original_label_int']
        for col in final_columns:
            if col not in all_samples_df.columns:
                if col == 'label': # If actual label string column is missing
                    all_samples_df[col] = "unknown" # Default value
                elif col == 'original_label_int':
                    all_samples_df[col] = -1
                else: # premise or hypothesis
                    all_samples_df[col] = "" # Default empty string
                logging.warning(f"Column '{col}' was missing from XNLI data for {lang_code}. Added with default values.")
        
        # Filter to only existing columns to avoid KeyErrors if some are truly missing despite checks
        existing_final_columns = [col for col in final_columns if col in all_samples_df.columns]
        all_samples_df = all_samples_df[existing_final_columns]


        # Shuffle samples and take the specified number if num_samples is provided
        if num_samples is not None:
            if num_samples > 0:
                # Shuffle all loaded samples first
                all_samples_df = all_samples_df.sample(frac=1, random_state=seed).reset_index(drop=True)
                
                if num_samples >= len(all_samples_df):
                    logging.info(f"Requested {num_samples} samples for {lang_code} ({split}), but only {len(all_samples_df)} are available. Using all available samples.")
                    final_samples_df = all_samples_df
                else:
                    final_samples_df = all_samples_df.head(num_samples)
                    logging.info(f"Selected {len(final_samples_df)} samples for {lang_code} ({split}) after requesting {num_samples} with seed {seed}.")
            else: # num_samples is 0 or negative
                logging.warning(f"Requested {num_samples} samples for XNLI {lang_code} ({split}). Returning empty DataFrame.")
                return pd.DataFrame()
        else: # num_samples is None, return all loaded and processed samples
            final_samples_df = all_samples_df
            logging.info(f"Returning all {len(final_samples_df)} loaded samples for XNLI {lang_code} ({split}) as num_samples was None.")

        logging.info(f"Successfully loaded and processed {len(final_samples_df)} XNLI samples for language '{lang_code}', split '{split}'.")
        return final_samples_df

    except FileNotFoundError:
        logging.error(f"ERROR loading XNLI for lang '{lang_code}', split '{split}': FileNotFoundError", exc_info=True)
        return pd.DataFrame()

def get_xnli_stats():
    """Prints statistics or information about the XNLI dataset structure if needed."""
    # This function can be expanded to show available languages, splits, etc.
    # For now, it's a placeholder or can be used for manual inspection.
    try:
        # Example: Get info for English configuration
        info = load_dataset("facebook/xnli", name="en", trust_remote_code=True)
        print("Dataset info for facebook/xnli (config 'en'):")
        print(info)

        print("\nSupported languages by facebook/xnli (configs):")
        # The 'facebook/xnli' dataset uses language codes as configuration names.
        # You can list them or refer to Hugging Face dataset card.
        # This is a hardcoded list for now based on common XNLI languages.
        print(XNLI_SUPPORTED_LANGUAGES)

    except Exception as e:
        print(f"Error fetching XNLI dataset info: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("--- Testing XNLI Loader (facebook/xnli) ---")

    # Test English
    en_samples = load_xnli_samples('en', num_samples=100, split='test') # 100 samples of test
    if not en_samples.empty:
        print(f"\nEnglish XNLI 'test' samples (100 - {len(en_samples)}):")
        print(en_samples.head())
        print(en_samples['label'].value_counts())
    else:
        print("\nFailed to load English XNLI samples.")

    # Test Swahili
    sw_samples = load_xnli_samples('sw', num_samples=100, split='train') # 100 samples of train
    if not sw_samples.empty:
        print(f"\nSwahili XNLI 'train' samples (100 - {len(sw_samples)}):")
        print(sw_samples.head())
        print(sw_samples['label'].value_counts())
    else:
        print("\nFailed to load Swahili XNLI samples.")

    # Test Urdu
    ur_samples = load_xnli_samples('ur', num_samples=50, split='validation') # 50 samples of validation
    if not ur_samples.empty:
        print(f"\nUrdu XNLI 'validation' samples (50 - {len(ur_samples)}):")
        print(ur_samples.head())
        print(ur_samples['label'].value_counts())
    else:
        print("\nFailed to load Urdu XNLI samples.")
    
    # Test getting stats
    # get_xnli_stats() 