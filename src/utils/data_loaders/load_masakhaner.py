import os
import pandas as pd
import json
import random
from datasets import load_dataset, concatenate_datasets
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the specific languages to be used from MasakhaNER
ALLOWED_LANGS = ['ha', 'sw'] # Hausa and Swahili

def convert_ner_tags_to_entities(tokens: List[str], ner_tags: List[int], tag_names: List[str]) -> List[Dict[str, str]]:
    """
    Converts token-level NER tags to a list of entity dictionaries.
    Example tag_names: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', ...]
    """
    entities = []
    current_entity_tokens = []
    current_entity_type = None

    for token, tag_idx in zip(tokens, ner_tags):
        tag_name = tag_names[tag_idx]

        if tag_name.startswith('B-'):
            # If there's a previous entity, save it
            if current_entity_tokens:
                entities.append({"text": " ".join(current_entity_tokens), "type": current_entity_type})
            current_entity_tokens = [token]
            current_entity_type = tag_name[2:]
        elif tag_name.startswith('I-') and current_entity_type == tag_name[2:]:
            current_entity_tokens.append(token)
        else: # O tag or inconsistent I- tag
            if current_entity_tokens:
                entities.append({"text": " ".join(current_entity_tokens), "type": current_entity_type})
            current_entity_tokens = []
            current_entity_type = None
    
    # Add any remaining entity
    if current_entity_tokens:
        entities.append({"text": " ".join(current_entity_tokens), "type": current_entity_type})
    
    return entities

def load_masakhaner_samples(
    lang_code: str, 
    sample_percentage: float = 10.0, 
    split: str = 'test', 
    seed: int = 42
) -> pd.DataFrame:
    """
    Load MasakhaNER samples for a specific language from Hugging Face Hub.
    Includes a mapping for common language codes to HF dataset config names.
    Only uses the 'test' split and samples 10% of the data by default.

    Args:
        lang_code: Language code, must be 'ha' or 'sw'.
        sample_percentage: Percentage of samples to load (default is 10.0 for 10%).
        split: Dataset split to use (fixed to 'test').
        seed: Random seed for sampling.

    Returns:
        DataFrame containing loaded samples with 'id', 'tokens', 'ner_tags', and 'entities' columns,
        or empty DataFrame if loading/processing fails or lang_code is not supported.
    """
    dataset_name = "masakhane/masakhaner"
    
    # Map common lang codes to HF dataset config names
    # Add to this map as needed
    lang_config_map = {
        'ha': 'hau', # Hausa
        'sw': 'swa', # Swahili
        'en': 'eng', # English (MasakhaNER doesn't have English, but for consistency if used elsewhere)
        'yo': 'yor', # Yoruba
        'pcm': 'pcm', # Nigerian Pidgin
        'amh': 'amh', # Amharic
        # The following are direct matches, but good to list them
        'ibo': 'ibo',
        'kin': 'kin',
        'lug': 'lug',
        'luo': 'luo',
        'wol': 'wol'
    }

    hf_config_name = lang_config_map.get(lang_code.lower(), lang_code.lower())
    # If lang_code was, e.g., 'HAU', it becomes 'hau'. If it was 'ha', it becomes 'hau'.
    # If it was 'amh', it becomes 'amh'.

    logger.info(f"Attempting to load '{lang_code}' (resolved to HF config: '{hf_config_name}') samples from {dataset_name}, split '{split}'...")

    if lang_code not in ALLOWED_LANGS:
        logger.error(f"Unsupported language code '{lang_code}'. Only 'ha' and 'sw' are allowed for MasakhaNER.")
        return pd.DataFrame()

    if split != 'test':
        logger.warning(f"MasakhaNER loader is fixed to 'test' split. Requested split '{split}' ignored.")
        split = 'test' # Force test split

    all_samples_list = []
    try:
        dataset = load_dataset(dataset_name, name=hf_config_name, split=split, trust_remote_code=True)
        logger.info(f"Successfully loaded dataset for {lang_code} (HF config: {hf_config_name}), split {split}.")
        
        # Get the tag names (feature information)
        # For MasakhaNER, 'ner_tags' is a Sequence feature with associated ClassLabel
        tag_feature = dataset.features['ner_tags']
        if hasattr(tag_feature, 'feature') and hasattr(tag_feature.feature, 'names'):
            tag_names = tag_feature.feature.names
        else:
            logger.error(f"Could not retrieve NER tag names for {lang_code}. Aborting.")
            return pd.DataFrame()

        for i, example in enumerate(dataset):
            tokens = example.get('tokens', [])
            ner_tags_indices = example.get('ner_tags', []) # These are indices

            if not tokens or not isinstance(ner_tags_indices, list): # check if ner_tags_indices is a list
                logger.warning(f"Skipping sample {i} for {lang_code} due to missing tokens or invalid ner_tags format.")
                continue

            # Convert tag indices to entity list
            entities = convert_ner_tags_to_entities(tokens, ner_tags_indices, tag_names)
            
            all_samples_list.append({
                'id': f"{lang_code}_{split}_{i}",
                'tokens': tokens,
                'text': " ".join(tokens), # Add full text field
                'ner_tags_indices': ner_tags_indices, # Keep original indices if needed
                'tag_names': tag_names, # Store tag names for reference
                'entities': entities, # Store the processed list of entity dicts
                'language': lang_code
            })

        if not all_samples_list:
            logger.warning(f"No samples processed for language '{lang_code}', split '{split}'.")
            return pd.DataFrame()
                
        samples_df = pd.DataFrame(all_samples_list)
    
    except ValueError as ve:
        # Check if the error is about the config name
        if "BuilderConfig" in str(ve) and "not found" in str(ve):
            logger.error(f"Failed to load MasakhaNER for language {lang_code} (HF config: '{hf_config_name}'), split '{split}'. Configuration not found.")
            logger.error(f"Original ValueError: {ve}")
            logger.warning(f"Please ensure '{lang_code}' (mapped to '{hf_config_name}') is a valid configuration for {dataset_name}. Available configs usually listed in the error or on the HF dataset page.")
        else:
            logger.error(f"ValueError while loading/processing MasakhaNER for {lang_code} (HF config: '{hf_config_name}'), split '{split}': {ve}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading/processing MasakhaNER for {lang_code} (HF config: '{hf_config_name}'), split '{split}': {e}", exc_info=True)
        return pd.DataFrame()

    # Sampling
    if sample_percentage is not None and 0 < sample_percentage <= 100:
        num_total_samples = len(samples_df)
        num_to_sample = max(1, int(num_total_samples * (sample_percentage / 100.0)))
        
        if num_to_sample < num_total_samples:
            logger.info(f"Sampling {sample_percentage}% ({num_to_sample} examples) from {num_total_samples} total for {lang_code} (split: {split}, seed: {seed}).")
            samples_df = samples_df.sample(n=num_to_sample, random_state=seed).reset_index(drop=True)
        else:
            logger.info(f"Sample percentage ({sample_percentage}%) results in selecting all {num_total_samples} samples for {lang_code}. Using all samples.")
    elif sample_percentage is not None: # Handles cases where sample_percentage is not None but not in (0, 100]
        logger.warning(f"Invalid sample_percentage value ({sample_percentage}). Using all {len(samples_df)} samples for {lang_code}.")

    logger.info(f"Loaded and processed {len(samples_df)} samples for {lang_code}, split '{split}'.")
    return samples_df

# Example usage (for testing this script directly)
if __name__ == '__main__':
    logger.info("--- Testing MasakhaNER Loader ---")

    # Test Hausa
    ha_samples = load_masakhaner_samples('ha', sample_percentage=10.0)
    if not ha_samples.empty:
        logger.info(f"Hausa test samples (10%): {len(ha_samples)}")
        logger.info(ha_samples.head())
        if 'entities' in ha_samples.columns and len(ha_samples.iloc[0]['entities']) > 0:
            logger.info(f"Entities of first HA sample: {ha_samples.iloc[0]['entities']}")
        else:
            logger.info("First HA sample has no entities or 'entities' column is missing.")
    else:
        logger.info("Failed to load Hausa samples or no samples returned.")

    # Test Swahili
    sw_samples = load_masakhaner_samples('sw', sample_percentage=5.0) # Test with a different percentage
    if not sw_samples.empty:
        logger.info(f"Swahili test samples (5%): {len(sw_samples)}")
        logger.info(sw_samples.head())
        if 'entities' in sw_samples.columns and not sw_samples.iloc[0]['entities'].empty:
             logger.info(f"Entities of first SW sample: {sw_samples.iloc[0]['entities']}")
    else:
        logger.info("Failed to load Swahili samples or no samples returned.")

    # Test an unsupported language
    xx_samples = load_masakhaner_samples('xx')
    if xx_samples.empty:
        logger.info("Correctly returned empty DataFrame for unsupported language 'xx'.")
    else:
        logger.error("Error: Loaded samples for an unsupported language 'xx'.")
        
    # Test with invalid split (should default to 'test' and log a warning)
    ha_samples_train_split = load_masakhaner_samples('ha', split='train')
    if not ha_samples_train_split.empty:
        logger.info(f"Hausa samples with 'train' split request (should use test): {len(ha_samples_train_split)}")
    else:
        logger.info("Failed to load Hausa samples when requesting 'train' split.") 