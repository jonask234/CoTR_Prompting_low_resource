import pandas as pd
from datasets import load_dataset, concatenate_datasets, DatasetDict, get_dataset_split_names, Dataset
from typing import List, Optional, Dict # Keep if Dict is used, otherwise can remove
import os
import random # Keep if used
from tqdm import tqdm # Keep if used
import logging  # <<< ADD THIS LINE

# Setup logger for this module
logger = logging.getLogger(__name__)  # <<< ADD THIS LINE

# Mapping from simple language codes to AfriSenti dataset configuration names
# This map helps to resolve any discrepancies if your input lang_codes (e.g., "sw")
# differ from the exact names used in the Hugging Face dataset configurations for AfriSenti.
AFRISENTI_LANG_MAP: Dict[str, str] = {
    "am": "amharic",        # Amharic
    "ar": "arabic",         # Arabic (MSA or a specific dialect used in dataset)
    "dz": "algerian_arabic", # Algerian Arabic (Darija)
    "en": "english",        # English
    "fr": "french",         # French
    
    "ha": "hau",            # Hausa (Standard short code 'ha' maps to HF config 'hau')
    "hau": "hau",           # Hausa (Direct HF config 'hau' maps to itself)
    # "hau": "hausa",       # Old incorrect mapping for 'hau'
    
    "id": "indonesian",     # Indonesian (Often in African multilingual datasets due to colonial links or data availability)
    "ig": "igbo",           # Igbo
    "kr": "kimbundu",       # Kimbundu (Angola) - Note: Double check exact HF config name if different
    "ma": "moroccan_arabic", # Moroccan Arabic (Darija)
    "multi": "multilingual",# For a combined multilingual track if available
    "pcm": "pidgin",        # Nigerian Pidgin
    
    "pt": "por",            # Portuguese (Standard short code 'pt' maps to HF config 'por')
    "por": "por",           # Portuguese (Direct HF config 'por' maps to itself)
    
    "sw": "swa",            # Swahili (Standard short code 'sw' maps to HF config 'swa')
    "swa": "swa",           # Swahili (Direct HF config 'swa' maps to itself)

    "ts": "tsonga",         # Tsonga
    "twi": "twi",           # Twi (Akan)
    "yo": "yoruba",         # Yoruba
    "zh": "chinese",        # Chinese (Sometimes included in broad multilingual sets)
    
    # Additional mappings from dataset's available configs, if different from above keys
    "amh": "amh",           # Amharic (config name)
    "arq": "arq",           # Algerian Arabic (config name, though 'dz' is more common short code)
    "ary": "ary",           # Moroccan Arabic (config name, though 'ma' is more common short code)
    "ibo": "ibo",           # Igbo (config name, same as 'ig')
    "kin": "kin",           # Kinyarwanda
    "orm": "orm",           # Oromo
    "tir": "tir",           # Tigrinya
    "tso": "tso",           # Xitsonga (config name, same as 'ts')
    # Ensure 'pcm', 'twi', 'yor' also map if their config names are identical and used as keys
}
# List of all unique configuration names used in AfriSenti on Hugging Face
# This should be updated if the dataset adds/changes config names.
# You can get this list from `datasets.get_dataset_config_names("afrisenti")`
AFRISENTI_CONFIG_NAMES = [
    'amharic', 'arabic', 'algerian_arabic', 'english', 'french', 'hausa', 
    'igbo', 'kimbundu', 'moroccan_arabic', 'pidgin', 'portuguese', 
    'swahili', 'tsonga', 'twi', 'yoruba'
    # 'multilingual' and 'indonesian', 'chinese' might not be direct configs but tracks or related datasets.
    # Verify with `get_dataset_config_names("afrisenti")`
]


# Local cache directory for Hugging Face datasets
# Ensure this path is writable and has enough space.
# LOCAL_HF_DATASETS_CACHE = os.path.expanduser("~/.cache/huggingface/datasets")
LOCAL_HF_DATASETS_CACHE = "/work/bbd6522/hf_datasets_cache" # Example custom cache path
os.makedirs(LOCAL_HF_DATASETS_CACHE, exist_ok=True)

# Define expected string labels for validation
EXPECTED_STRING_LABELS = {'positive', 'negative', 'neutral'}

def _load_samples_from_split(dataset_name: str, lang_config: str, split: str, num_samples_to_load: Optional[int]) -> List[dict]:
    """Helper function to load samples from a specific split of the AfriSenti dataset."""
    samples = []
    print(f"Attempting to load {lang_config} samples from {dataset_name} ({split} split) via Hugging Face Hub...")

    try:
        # Load the specific language configuration and split
        print(f"  Loading dataset {dataset_name} config '{lang_config}' split '{split}' from Hugging Face Hub...")
        dataset = load_dataset(dataset_name, name=lang_config, split=split, trust_remote_code=True)
        print("  Dataset loaded from Hugging Face Hub.")

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
            label_raw = example.get('label', None)
            label_name = None

            if isinstance(label_raw, str):
                label_lower = label_raw.lower()
                # Check for new style labels like "0positive", "1neutral", "2negative"
                if label_lower.startswith(("0", "1", "2")) and len(label_lower) > 1:
                    # Attempt to extract the part after the digit
                    potential_label = label_lower[1:]
                    if potential_label in EXPECTED_STRING_LABELS:
                        label_name = potential_label
                    # Fallback for just "positive", "negative", "neutral" if extraction fails or not prefixed
                    elif label_lower in EXPECTED_STRING_LABELS: 
                        label_name = label_lower
                elif label_lower in EXPECTED_STRING_LABELS:
                    label_name = label_lower
            elif isinstance(label_raw, int): # Handle integer labels if they exist in some configs
                # This part might need adjustment if integer mapping is complex
                # For now, assuming a simple map if 0,1,2 directly correspond to neg, neu, pos or similar
                # This part of the original LAF-MACRO was commented out, reactivating with caution:
                temp_label_map = {0: 'positive', 1: 'neutral', 2: 'negative'} # Or whatever the correct int mapping is
                if label_raw in temp_label_map:
                    label_name = temp_label_map[label_raw]

            # Basic validation: Check text exists and label is valid
            if not text or not text.strip() or label_name not in EXPECTED_STRING_LABELS:
                # print(f"Skipping invalid sample: Text: '{text[:50]}...', Raw Label: '{label_raw}', Parsed Label: '{label_name}'")
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

def create_dummy_sentiment_data(lang_code: str, num_samples: int = 3) -> pd.DataFrame:
    logger.warning(f"Creating dummy sentiment data for {lang_code} ({num_samples} samples).")
    data = {
        'id': [f'dummy_{i}' for i in range(num_samples)],
        'text': [f"This is a dummy {lang_code} sentence {i}." for i in range(num_samples)],
        'label': ['positive', 'negative', 'neutral'][:num_samples] if num_samples <= 3 else ['neutral'] * num_samples
    }
    return pd.DataFrame(data)

def load_afrisenti_samples(
    lang_code: str, 
    num_samples: Optional[int] = None,
    split: str = 'test', 
    dataset_name: str = "shmuhammad/AfriSenti-twitter-sentiment",
    seed: int = 42
) -> pd.DataFrame:
    """
    Load samples for a given language from the AfriSenti dataset.

    Args:
        lang_code (str): Language code (e.g., 'am', 'ha', 'sw').
        num_samples (Optional[int]): Number of samples to load. If None, loads all available for the split.
        split (str): Dataset split ('train', 'dev', 'test').
        dataset_name (str): Name of the dataset on Hugging Face.
        seed (int): Random seed for shuffling if num_samples is specified.

    Returns:
        pd.DataFrame: DataFrame with 'id', 'text' (tweet), and 'label' columns.
                      Returns empty DataFrame on failure.
    """
    if lang_code not in AFRISENTI_LANG_MAP:
        logger.error(f"Language code '{lang_code}' not found in AFRISENTI_LANG_MAP. Available keys: {list(AFRISENTI_LANG_MAP.keys())}")
        logger.info("Falling back to dummy data for this language.")
        return create_dummy_sentiment_data(lang_code, num_samples if num_samples is not None else 3)

    hf_config_name = AFRISENTI_LANG_MAP[lang_code]
    
    hf_split = split
    if split == 'dev':
        hf_split = 'validation' 
        logger.info(f"Mapping requested split 'dev' to 'validation' for AfriSenti config '{hf_config_name}'.")
    elif split not in ['train', 'validation', 'test']:
        logger.error(f"Invalid split '{split}' for AfriSenti. Must be one of 'train', 'dev', 'test'. Using 'test' as default.")
        hf_split = 'test'

    logger.info(f"Loading AfriSenti samples for language: {lang_code} (HF config: {hf_config_name}), split: {hf_split} from {dataset_name}")

    try:
        # Attempt to load the dataset
        dataset = load_dataset(dataset_name, name=hf_config_name, split=hf_split, trust_remote_code=True)
        logger.info(f"Successfully loaded {len(dataset)} samples from {dataset_name}/{hf_config_name} (split: {hf_split}).")
        
        # Convert to DataFrame
        samples_df = dataset.to_pandas()
        
        # Standardize column names
        if 'tweet' in samples_df.columns and 'text' not in samples_df.columns:
            samples_df.rename(columns={'tweet': 'text'}, inplace=True)
        
        if 'text' not in samples_df.columns or 'label' not in samples_df.columns:
            logger.error(f"Dataset {dataset_name}/{hf_config_name} for {lang_code} is missing 'text' or 'label' column. Columns: {samples_df.columns.tolist()}")
            raise ValueError("Missing required columns 'text' or 'label'")

        # Ensure 'id' column exists or create it
        if 'id' not in samples_df.columns:
            samples_df['id'] = [f"{hf_config_name}_{hf_split}_{i}" for i in range(len(samples_df))]

        # Select subset of samples if num_samples is specified
        if num_samples is not None and 0 < num_samples < len(samples_df):
            samples_df = samples_df.sample(n=num_samples, random_state=seed, replace=False).reset_index(drop=True)
            logger.info(f"Sampled down to {len(samples_df)} for {lang_code} ({hf_config_name}).")
        elif num_samples is not None and num_samples >= len(samples_df):
            logger.info(f"Requested {num_samples} samples, but only {len(samples_df)} available for {lang_code} ({hf_config_name}). Using all available.")
        elif num_samples is None:
            logger.info(f"Using all {len(samples_df)} available samples for {lang_code} ({hf_config_name}).")
        
        # Ensure labels are strings and handle potential float/int types from raw data
        if 'label' in samples_df.columns:
            samples_df['label'] = samples_df['label'].astype(str)
            # Map numeric string labels to descriptive string labels
            # AfriSenti typically uses: 0 -> negative, 1 -> neutral, 2 -> positive
            label_mapping = {
                "0": "negative", 
                "1": "neutral", 
                "2": "positive",
                # Ensure existing string labels are preserved (case-insensitively)
                "negative": "negative",
                "neutral": "neutral",
                "positive": "positive"
            }
            # Apply mapping, handling potential case issues for string labels already in correct format
            samples_df['label'] = samples_df['label'].str.lower().map(label_mapping).fillna(samples_df['label'])
            
            # Validate that all labels are now one of the expected strings
            if not samples_df['label'].isin(list(EXPECTED_STRING_LABELS)).all():
                unknown_labels = samples_df[~samples_df['label'].isin(list(EXPECTED_STRING_LABELS))]['label'].unique()
                logger.warning(f"Labels in {lang_code} after mapping still contain unexpected values: {unknown_labels}. These might cause issues in metrics.")
                # Optionally, map these unknown to a default like 'neutral' or drop them
                # For now, we log a warning.

        logger.info(f"Prepared DataFrame for {lang_code} with columns: {samples_df.columns.tolist()}")
        return samples_df[['id', 'text', 'label']] # Return only essential columns

    except ValueError as ve: # Catch specific ValueError from missing builder config or columns
        logger.error(f"ValueError loading or processing AfriSenti for language {lang_code} (HF: {hf_config_name}), split {hf_split}: {ve}")
        logger.info(f"Falling back to dummy data for {lang_code} due to ValueError.")
        return create_dummy_sentiment_data(lang_code, num_samples if num_samples is not None else 3)
    except Exception as e:
        logger.error(f"Generic error loading or processing AfriSenti for language {lang_code} (HF: {hf_config_name}), split {hf_split}: {e}", exc_info=True)
        logger.info(f"Falling back to dummy data for {lang_code} due to generic error.")
        return create_dummy_sentiment_data(lang_code, num_samples if num_samples is not None else 3)

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Test standard loading
    hausa_samples = load_afrisenti_samples('ha', 100)
    print("\nRandom Hausa Samples (n=100) from Hugging Face Hub:")
    if not hausa_samples.empty:
        print(hausa_samples.head())
        print(hausa_samples['label'].value_counts())

    # Test balanced loading
    swahili_balanced = load_afrisenti_samples('sw', balanced=True)
    print("\nBalanced Swahili Samples from Hugging Face Hub:")
    if not swahili_balanced.empty:
        print(swahili_balanced.head())
        print(swahili_balanced['label'].value_counts())
    
    # Test balanced loading with specific samples per class
    swahili_balanced_100 = load_afrisenti_samples('sw', samples_per_class=100, balanced=True)
    print("\nBalanced Swahili Samples (100 per class) from Hugging Face Hub:")
    if not swahili_balanced_100.empty:
        print(swahili_balanced_100.head())
        print(swahili_balanced_100['label'].value_counts())

    # Test loading more than available (should load all)
    # Assuming Swahili train split is smaller than 3000
    swahili_samples_all = load_afrisenti_samples('sw', 3000, split='train') 
    print("\nSwahili Samples (requested 3000 from train) from Hugging Face Hub:")
    if not swahili_samples_all.empty:
        print(swahili_samples_all.head())
        print(f"Total loaded: {len(swahili_samples_all)}")
        print(swahili_samples_all['label'].value_counts())
    
    # Test loading all samples (num_samples=None)
    amh_samples_all = load_afrisenti_samples('am', None, split='train') 
    print("\nAmharic Samples (all from train) from Hugging Face Hub:")
    if not amh_samples_all.empty:
        print(amh_samples_all.head())
        print(f"Total loaded: {len(amh_samples_all)}")
        print(amh_samples_all['label'].value_counts()) 