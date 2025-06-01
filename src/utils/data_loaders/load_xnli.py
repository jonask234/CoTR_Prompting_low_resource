import os
import pandas as pd
import random
from datasets import load_dataset, get_dataset_split_names
from typing import Optional, List, Dict
import logging

# Standard XNLI label mapping
XNLI_LABEL_MAP: Dict[int, str] = {
    0: 'entailment',
    1: 'neutral',
    2: 'contradiction'
}

# XNLI languages available in facebook/xnli
# From documentation: ar, bg, de, el, en, es, fr, hi, ru, sw, th, tr, ur, vi, zh
XNLI_SUPPORTED_LANGUAGES = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

def load_xnli_samples(
    lang_code: str,
    sample_percentage: float = 10.0, # Default to 10%
    split: str = 'test' # Default to 'test' split for evaluation
) -> pd.DataFrame:
    """
    Load XNLI samples for a specific language from 'facebook/xnli'.

    Args:
        lang_code: Language code (e.g., 'en', 'sw', 'ur').
        sample_percentage: Percentage of samples to load from the split.
        split: Dataset split to use ('train', 'validation', 'test').

    Returns:
        DataFrame containing 'premise', 'hypothesis', 'label' (string), 'language',
        and 'original_label_int'. Returns empty DataFrame on failure.
    """
    dataset_name = "facebook/xnli"

    if lang_code not in XNLI_SUPPORTED_LANGUAGES:
        logging.error(f"Language code '{lang_code}' is not supported by the facebook/xnli dataset. Supported: {XNLI_SUPPORTED_LANGUAGES}")
        return pd.DataFrame()

    available_splits = []
    try:
        # For XNLI, each language is a configuration.
        # We check splits for a common language like 'en' as an example.
        available_splits = get_dataset_split_names(dataset_name, config_name='en')
        logging.info(f"Available splits for {dataset_name} (e.g., 'en' config): {available_splits}")
        if split not in available_splits:
            logging.warning(f"Requested split '{split}' not in typical XNLI splits {available_splits}. Attempting to load anyway.")
    except Exception as e:
        logging.warning(f"Could not dynamically verify splits for {dataset_name}: {e}. Proceeding with requested split '{split}'.")

    logging.info(f"Attempting to load '{lang_code}' samples from {dataset_name}, config '{lang_code}', split '{split}'...")

    try:
        # For facebook/xnli, the language code is the configuration name.
        dataset = load_dataset(dataset_name, name=lang_code, split=split, trust_remote_code=True)
        logging.info(f"Successfully loaded dataset for {lang_code}, split {split}. Full size: {len(dataset)}")

        num_total_samples = len(dataset)
        if num_total_samples == 0:
            logging.warning(f"No samples found in {dataset_name} for lang '{lang_code}', split '{split}'.")
            return pd.DataFrame()

        num_to_sample = max(1, int(num_total_samples * (sample_percentage / 100.0)))
        if num_to_sample > num_total_samples : # Should not happen if percentage is <=100
             num_to_sample = num_total_samples
        
        logging.info(f"Sampling {sample_percentage}% ({num_to_sample} examples) from {num_total_samples} total for {lang_code}, split '{split}'.")
        
        # Shuffle before selecting to ensure randomness if num_to_sample < num_total_samples
        if num_to_sample < num_total_samples:
            dataset = dataset.shuffle(seed=42).select(range(num_to_sample))
        elif num_to_sample == num_total_samples: # If 100% or more requested, use all but still shuffle
            dataset = dataset.shuffle(seed=42)
        
        all_samples_list = []
        for example in dataset:
            premise = example.get('premise', '')
            hypothesis = example.get('hypothesis', '')
            label_int = example.get('label', -1) # Labels are integer encoded

            if not premise or not hypothesis or label_int == -1:
                logging.warning(f"Skipping sample due to missing premise, hypothesis, or label: {example}")
                continue

            all_samples_list.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'original_label_int': label_int, # Keep original integer label
                'label': XNLI_LABEL_MAP.get(label_int, 'unknown'), # Mapped string label
                'language': lang_code
            })
        
        if not all_samples_list:
            logging.warning(f"No valid samples processed for {lang_code}, split '{split}' after filtering.")
            return pd.DataFrame()

        samples_df = pd.DataFrame(all_samples_list)
        logging.info(f"Successfully loaded and processed {len(samples_df)} samples for {lang_code}, split '{split}'.")
        logging.info(f"Label distribution for {lang_code}, {split}: \n{samples_df['label'].value_counts(normalize=True)}")
        return samples_df

    except Exception as e:
        logging.error(f"ERROR loading XNLI for lang '{lang_code}', split '{split}': {e}", exc_info=True)
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
    en_samples = load_xnli_samples('en', sample_percentage=1.0, split='test') # 1% of test
    if not en_samples.empty:
        print(f"\nEnglish XNLI 'test' samples (1% - {len(en_samples)}):")
        print(en_samples.head())
        print(en_samples['label'].value_counts())
    else:
        print("\nFailed to load English XNLI samples.")

    # Test Swahili
    sw_samples = load_xnli_samples('sw', sample_percentage=10.0, split='train') # 10% of train
    if not sw_samples.empty:
        print(f"\nSwahili XNLI 'train' samples (10% - {len(sw_samples)}):")
        print(sw_samples.head())
        print(sw_samples['label'].value_counts())
    else:
        print("\nFailed to load Swahili XNLI samples.")

    # Test Urdu
    ur_samples = load_xnli_samples('ur', sample_percentage=5.0, split='validation') # 5% of validation
    if not ur_samples.empty:
        print(f"\nUrdu XNLI 'validation' samples (5% - {len(ur_samples)}):")
        print(ur_samples.head())
        print(ur_samples['label'].value_counts())
    else:
        print("\nFailed to load Urdu XNLI samples.")
    
    # Test getting stats
    # get_xnli_stats() 