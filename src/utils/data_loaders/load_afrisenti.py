import pandas as pd
import os
from typing import Optional, List
from datasets import load_dataset, get_dataset_split_names
from tqdm import tqdm
import random

# Mapping from simple language codes to AfriSenti dataset configuration names
# Check the dataset viewer on Hugging Face for exact config names
# Example: https://huggingface.co/datasets/masakhane/afrisenti_twitter_corpus
LANG_CODE_MAP = {
    'am': 'amh',  # Amharic
    'dz': 'dzo',  # Dzongkha
    'ha': 'hau',  # Hausa
    'ig': 'ibo',  # Igbo
    'kr': 'kin',  # Kinyarwanda
    'ma': 'mor',  # Moroccan Arabic/Darija
    'ny': 'pcm',  # Nigerian Pidgin
    'pt': 'por',  # Portuguese
    'sw': 'swa',  # Swahili
    'ti': 'tir',  # Tigrinya
    'yo': 'yor',  # Yoruba
    # Add other potential mappings if needed
}

# Remove integer-to-string mapping, as dataset seems to provide strings
# LABEL_MAP = {
#     0: 'negative',
#     1: 'neutral',
#     2: 'positive'
# }

# Define expected string labels for validation
EXPECTED_STRING_LABELS = {'positive', 'negative', 'neutral'}

def _load_samples_from_split(dataset_name: str, lang_config: str, split: str, num_samples_to_load: Optional[int]) -> List[dict]:
    """Helper function to load samples from a specific split of the AfriSenti dataset."""
    samples = []
    print(f"Attempting to load {lang_config} samples from {dataset_name} ({split} split)...")

    try:
        # Load the specific language configuration and split
        # Load WITHOUT streaming first
        print(f"  Loading dataset {dataset_name} config '{lang_config}' split '{split}'...")
        dataset = load_dataset(dataset_name, name=lang_config, split=split) 
        print("  Dataset loaded.")
        # --- Debugging Removed ---
        # print(f"  Dataset object loaded: {dataset}")
        # print(f"  Dataset features: {dataset.features}")
        # try:
        #     first_example = next(iter(dataset))
        #     print(f"  First raw example: {first_example}")
        # except StopIteration:
        #     print("  WARN: Dataset appears empty after loading!")
        # --- End Debugging ---
        
        dataset_size = len(dataset)
        print(f"  Full split size: {dataset_size}")

        # --- Revert Sampling Logic to Iterate All (or up to num_samples) ---
        # Directly iterate over the dataset loaded in memory
        # Slice the dataset if num_samples is specified
        if num_samples_to_load is not None and num_samples_to_load < dataset_size:
            # Note: This loads the full dataset then slices, might be memory intensive for huge datasets
            # Consider dataset.select(range(num_samples_to_load)) if memory is an issue before full iteration.
            dataset_to_iterate = dataset.select(range(num_samples_to_load))
            print(f"  Selected first {num_samples_to_load} samples for processing.")
            effective_total = num_samples_to_load
        else:
            dataset_to_iterate = dataset
            effective_total = dataset_size
            if num_samples_to_load is not None:
                print(f"  Requested samples ({num_samples_to_load}) >= dataset size ({dataset_size}). Processing all.")
        # -----------------------------------------------------------------

        count = 0
        processed_records = 0
        # Iterate directly over the potentially sliced dataset
        for example in tqdm(dataset_to_iterate, total=effective_total, desc=f"Processing {lang_config} ({split})"):
            processed_records += 1
            example_id = example.get('ID', f'{lang_config}_{split}_{processed_records}') 
            # Use 'tweet' field for text content
            text = example.get('tweet', '') 
            # Get the label directly - assuming it's a string like 'positive'
            label_name = example.get('label', None) 
            # Ensure it's a string and lowercase for consistent checking
            if isinstance(label_name, str):
                label_name = label_name.lower()
            else: # If it's not a string (e.g., None or unexpected type), set to None for validation failure
                label_name = None 

            # --- Debugging Removed ---
            # print(f"  DEBUG FINAL CHECK: record={example_id}, label_name='{repr(label_name)}', is_in_expected={label_name in EXPECTED_STRING_LABELS}") 
            # --- End Debugging ---

            # Basic validation: Check text exists (after stripping) and label is valid
            if not text or not text.strip() or label_name not in EXPECTED_STRING_LABELS:
                # print(f"  Skipping sample {example_id}: Missing/empty text or invalid label ('{label_name}').") # Reduce noise
                continue

            # Create and append the sample
            sample = {
                'text': text,
                'label': label_name, # Store the validated string label
                # 'label_id': label_id, # No longer needed if label is string
                'id': example_id
            }
            samples.append(sample)
            count += 1
            # Stop once we have enough valid samples
            if num_samples_to_load is not None and count >= num_samples_to_load:
                print(f"\n  Collected requested {num_samples_to_load} samples after processing {processed_records} records.")
                break

    except Exception as e:
        print(f"Error loading or processing {dataset_name} ({lang_config}, {split} split): {e}")
        return []

    print(f"  Successfully extracted {len(samples)} valid samples.")
    return samples

def load_afrisenti_samples(lang_code: str, num_samples: Optional[int] = None, split: str = "train") -> pd.DataFrame:
    """
    Load AfriSenti samples for a specific language.

    Args:
        lang_code: Language code (e.g., 'ha' for Hausa, 'sw' for Swahili).
        num_samples: Target number of samples to load (None for all available).
                     Samples are randomly selected if num_samples < dataset size.
        split: Dataset split to use ('train', 'validation', 'test'). 'train' is often the largest.

    Returns:
        DataFrame containing the samples ('text', 'label', 'id').
    """
    # Use the correct dataset identifier including the organization
    dataset_name = "masakhane/afrisenti"

    afrisenti_lang_config = LANG_CODE_MAP.get(lang_code)
    if not afrisenti_lang_config:
        print(f"ERROR: Unsupported or unmapped language code '{lang_code}' for AfriSenti loader.")
        return pd.DataFrame({'text': [], 'label': [], 'id': []})

    # Check available splits (optional but good practice)
    try:
        available_splits = get_dataset_split_names(dataset_name, afrisenti_lang_config)
        print(f"Available splits for {dataset_name}/{afrisenti_lang_config}: {available_splits}")
        if split not in available_splits:
            original_split = split
            if 'train' in available_splits:
                split = 'train'
            elif available_splits:
                split = available_splits[0]
            else:
                print(f"ERROR: No splits found for {dataset_name}/{afrisenti_lang_config}.")
                return pd.DataFrame({'text': [], 'label': [], 'id': []})
            print(f"WARN: Requested split '{original_split}' not found for {afrisenti_lang_config}. Using '{split}' split instead.")

    except Exception as e:
        print(f"WARN: Could not verify splits for {dataset_name}/{afrisenti_lang_config}: {e}. Attempting to load '{split}' split.")

    # Load samples
    all_samples = _load_samples_from_split(dataset_name, afrisenti_lang_config, split, num_samples)

    print(f"Loaded {len(all_samples)} {afrisenti_lang_config} samples in total from the '{split}' split.")

    # Return DataFrame
    if not all_samples:
        print(f"WARNING: No {afrisenti_lang_config} samples found or loaded from the '{split}' split!")
        # Adjust columns if label_id was removed
        return pd.DataFrame({'text': [], 'label': [], 'id': []})

    # Adjust columns if label_id was removed
    return pd.DataFrame(all_samples)[['text', 'label', 'id']]

# Example usage (optional, for testing)
if __name__ == '__main__':
    hausa_samples = load_afrisenti_samples('ha', 100) # Load 100 Hausa samples
    print("Hausa Samples:")
    if not hausa_samples.empty:
        print(hausa_samples.head())
        print(hausa_samples['label'].value_counts())

    swahili_samples = load_afrisenti_samples('sw', 50, split='train') # Load 50 Swahili samples
    print("Swahili Samples:")
    if not swahili_samples.empty:
        print(swahili_samples.head())
        print(swahili_samples['label'].value_counts()) 