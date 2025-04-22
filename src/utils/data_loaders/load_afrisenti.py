import pandas as pd
import os
from typing import Optional, List
from datasets import load_dataset, get_dataset_split_names, Dataset
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
        print(f"  Loading dataset {dataset_name} config '{lang_config}' split '{split}'...")
        dataset = load_dataset(dataset_name, name=lang_config, split=split)
        print("  Dataset loaded.")

        dataset_size = len(dataset)
        print(f"  Full split size: {dataset_size}")

        # --- Sampling Logic --- 
        dataset_to_iterate = None
        effective_total = 0

        if num_samples_to_load is not None and num_samples_to_load < dataset_size:
            # Apply random sampling universally when sampling is needed
            print(f"  Applying RANDOM sampling for {lang_config}: Shuffling and selecting {num_samples_to_load} samples...")
            dataset_to_iterate = dataset.shuffle(seed=42).select(range(num_samples_to_load))
            print(f"  Selected {len(dataset_to_iterate)} samples after shuffling.")
            effective_total = num_samples_to_load
        elif num_samples_to_load is not None:
            print(f"  Requested samples ({num_samples_to_load}) >= dataset size ({dataset_size}). Processing all.")
            dataset_to_iterate = dataset # Use the whole dataset
            effective_total = dataset_size
        else:
            print("  Processing all samples in the split.")
            # Even when processing all samples, shuffle to ensure randomness
            dataset_to_iterate = dataset.shuffle(seed=42)
            effective_total = dataset_size
        # --------------------------------

        count = 0
        processed_records = 0
        # Iterate directly over the selected dataset portion
        for example in tqdm(dataset_to_iterate, total=effective_total, desc=f"Processing {lang_config} ({split}) sample"):
            processed_records += 1
            example_id = example.get('ID', f'{lang_config}_{split}_{processed_records}')
            text = example.get('tweet', '')
            label_name = example.get('label', None)
            if isinstance(label_name, str):
                label_name = label_name.lower()
            else:
                label_name = None

            # Basic validation: Check text exists and label is valid
            if not text or not text.strip() or label_name not in EXPECTED_STRING_LABELS:
                continue

            # Create and append the sample
            sample = {
                'text': text,
                'label': label_name,
                'id': example_id
            }
            samples.append(sample)
            count += 1
            # Stop once we have enough valid samples (redundant with select but safe)
            if num_samples_to_load is not None and count >= num_samples_to_load:
                break

    except Exception as e:
        print(f"Error loading or processing {dataset_name} ({lang_config}, {split} split): {e}")
        return []

    print(f"  Successfully extracted {len(samples)} valid samples.")
    return samples

def load_afrisenti_samples(
    lang_code: str, 
    num_samples: Optional[int] = None, 
    split: str = "train", 
    balanced: bool = False,
    samples_per_class: Optional[int] = None
) -> pd.DataFrame:
    """
    Load AfriSenti samples for a specific language.

    Args:
        lang_code: Language code (e.g., 'ha' for Hausa, 'sw' for Swahili).
        num_samples: Target number of samples to load (None for all available).
                     If balanced=False: samples are randomly selected if num_samples < dataset size.
                     If balanced=True: interpreted as samples per class, if also None then uses
                     the size of the smallest class.
        split: Dataset split to use ('train', 'validation', 'test'). 'train' is often the largest.
        balanced: If True, creates a balanced dataset with equal samples from each class.
        samples_per_class: Number of samples to select per class when balanced=True.
                          If None, uses the minimum class size or num_samples (if specified).

    Returns:
        DataFrame containing the samples ('text', 'label', 'id'), shuffled to ensure randomness.
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

    # If balanced sampling is not requested, use the standard loading logic
    if not balanced:
        # Load samples using standard approach
        all_samples = _load_samples_from_split(dataset_name, afrisenti_lang_config, split, num_samples)
        
        print(f"Loaded {len(all_samples)} {afrisenti_lang_config} samples in total from the '{split}' split.")

        # Return DataFrame
        if not all_samples:
            print(f"WARNING: No {afrisenti_lang_config} samples found or loaded from the '{split}' split!")
            return pd.DataFrame({'text': [], 'label': [], 'id': []})

        # Create DataFrame and shuffle it to ensure random order
        df = pd.DataFrame(all_samples)[['text', 'label', 'id']]
        
        # Always shuffle regardless of whether num_samples was provided
        print(f"Shuffling {len(df)} samples to ensure randomness...")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Print label distribution to verify data is properly loaded and shuffled
        print(f"Final label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
    
    else:
        # --- Balanced Sampling Logic ---
        print(f"Balanced class sampling requested for {afrisenti_lang_config}...")
        
        # Load all samples for the language (we'll filter by class later)
        all_samples = _load_samples_from_split(dataset_name, afrisenti_lang_config, split, None)
        
        if not all_samples:
            print(f"WARNING: No {afrisenti_lang_config} samples found or loaded from the '{split}' split!")
            return pd.DataFrame({'text': [], 'label': [], 'id': []})
        
        # Convert to DataFrame for easier class-based filtering
        df_all = pd.DataFrame(all_samples)
        
        # Check label distribution
        label_counts = df_all['label'].value_counts()
        print(f"Original label distribution: {label_counts.to_dict()}")
        
        # Determine samples per class
        min_class_count = label_counts.min()
        
        if samples_per_class is not None:
            # User-specified samples per class
            n_per_class = min(samples_per_class, min_class_count)
        elif num_samples is not None:
            # Interpret num_samples as per-class count
            n_per_class = min(num_samples, min_class_count)
        else:
            # Default to minimum class count for fully balanced dataset
            n_per_class = min_class_count
        
        print(f"Using {n_per_class} samples per class for balanced dataset")
        
        # Create balanced DataFrame by sampling equally from each class
        balanced_samples = []
        
        for label in EXPECTED_STRING_LABELS:
            class_samples = df_all[df_all['label'] == label]
            
            if len(class_samples) > 0:
                # Sample at most n_per_class samples from this class
                if len(class_samples) > n_per_class:
                    sampled = class_samples.sample(n=n_per_class, random_state=42)
                else:
                    sampled = class_samples  # Take all if we have fewer than needed
                
                balanced_samples.append(sampled)
                print(f"  Selected {len(sampled)} samples for class '{label}'")
            else:
                print(f"  WARNING: No samples found for class '{label}'")
        
        # Combine all balanced class samples
        df_balanced = pd.concat(balanced_samples, ignore_index=True)
        
        # Shuffle to ensure classes are mixed
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Print final counts to verify balance
        final_distribution = df_balanced['label'].value_counts().to_dict()
        print(f"Final balanced distribution: {final_distribution}")
        total_samples = len(df_balanced)
        print(f"Total samples in balanced dataset: {total_samples}")
        
        return df_balanced[['text', 'label', 'id']]

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Test standard loading
    hausa_samples = load_afrisenti_samples('ha', 100)
    print("\nRandom Hausa Samples (n=100):")
    if not hausa_samples.empty:
        print(hausa_samples.head())
        print(hausa_samples['label'].value_counts())

    # Test balanced loading
    swahili_balanced = load_afrisenti_samples('sw', balanced=True)
    print("\nBalanced Swahili Samples:")
    if not swahili_balanced.empty:
        print(swahili_balanced.head())
        print(swahili_balanced['label'].value_counts())
    
    # Test balanced loading with specific samples per class
    swahili_balanced_100 = load_afrisenti_samples('sw', samples_per_class=100, balanced=True)
    print("\nBalanced Swahili Samples (100 per class):")
    if not swahili_balanced_100.empty:
        print(swahili_balanced_100.head())
        print(swahili_balanced_100['label'].value_counts())

    # Test loading more than available (should load all)
    # Assuming Swahili train split is smaller than 3000
    swahili_samples_all = load_afrisenti_samples('sw', 3000, split='train') 
    print("\nSwahili Samples (requested 3000 from train):")
    if not swahili_samples_all.empty:
        print(swahili_samples_all.head())
        print(f"Total loaded: {len(swahili_samples_all)}")
        print(swahili_samples_all['label'].value_counts())
    
    # Test loading all samples (num_samples=None)
    amh_samples_all = load_afrisenti_samples('am', None, split='train') 
    print("\nAmharic Samples (all from train):")
    if not amh_samples_all.empty:
        print(amh_samples_all.head())
        print(f"Total loaded: {len(amh_samples_all)}")
        print(amh_samples_all['label'].value_counts()) 