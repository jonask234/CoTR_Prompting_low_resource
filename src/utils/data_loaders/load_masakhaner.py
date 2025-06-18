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
MASAKHANER_LANGUAGES = ["swa", "amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "wol", "yor"]

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
    num_samples: Optional[int] = None, # Changed from sample_percentage, added seed
    split: str = 'test', 
    seed: int = 42
) -> pd.DataFrame:
    """
    Load NER samples for a given language from the MasakhaNER dataset.

    Args:
        lang_code (str): The MasakhaNER language code (e.g., 'swa', 'hau').
        num_samples (Optional[int]): The number of samples to return. If None, all samples from the split are returned.
        split (str): The dataset split to load ('train', 'validation', 'test'). Default is 'test'.
        seed (int): Random seed for shuffling if num_samples is specified.

    Returns:
        pd.DataFrame: A DataFrame containing the samples, with columns 'id', 'tokens', 'ner_tags' (original integer tags),
                      'tag_names' (list of tag names like ['O', 'B-PER', ...]), and 'entities' (list of dicts).
                      Returns an empty DataFrame if the language is not supported or data loading fails.
    """
    dataset_name = "masakhaner"
    
    if lang_code not in MASAKHANER_LANGUAGES:
        logger.error(f"Language code '{lang_code}' is not a valid MasakhaNER language. Supported: {MASAKHANER_LANGUAGES}")
        return pd.DataFrame()

    logger.info(f"Attempting to load MasakhaNER for language: {lang_code}, split: {split}")
    
    try:
        # MasakhaNER uses the language code as the configuration name.
        dataset = load_dataset(dataset_name, name=lang_code, split=split, trust_remote_code=True)
        logger.info(f"Successfully loaded MasakhaNER for {lang_code}, split {split}. Full size: {len(dataset)}")

        all_samples_list = []
        if not hasattr(dataset, 'features') or 'ner_tags' not in dataset.features:
            logger.error(f"Dataset for MasakhaNER {lang_code} does not have 'ner_tags' features. Cannot process.")
            return pd.DataFrame()
            
        tag_names = dataset.features['ner_tags'].feature.names
        
        for i, example in enumerate(dataset):
            tokens = example.get('tokens', [])
            ner_tags_int = example.get('ner_tags', [])
            sample_id = example.get('id', str(i))

            if not tokens or not ner_tags_int:
                logger.warning(f"Skipping sample {sample_id} due to missing tokens or ner_tags.")
                continue
            
            entities = convert_ner_tags_to_entities(tokens, ner_tags_int, tag_names)
            
            all_samples_list.append({
                'id': sample_id,
                'tokens': tokens,
                'ner_tags': ner_tags_int, # Store original integer tags
                'tag_names': tag_names,    # Store the list of tag names for context
                'entities': entities       # Store the processed entity list
            })
        
        if not all_samples_list:
            logger.warning(f"No valid samples processed for MasakhaNER {lang_code}, split '{split}'.")
            return pd.DataFrame()

        all_samples_df = pd.DataFrame(all_samples_list)

        # Create 'text' column from 'tokens'
        if 'tokens' in all_samples_df.columns:
            all_samples_df['text'] = all_samples_df['tokens'].apply(lambda t: " ".join(t) if isinstance(t, list) else "")
            # Check for empty text strings after creation
            empty_texts_count = all_samples_df[all_samples_df['text'] == ''].shape[0]
            if empty_texts_count > 0:
                logger.warning(f"{empty_texts_count}/{len(all_samples_df)} samples have empty 'text' after token joining for {lang_code} ({split}).")
        else:
            logger.error(f"'tokens' column not found for MasakhaNER {lang_code} ({split}). Cannot create 'text' column. This will likely cause issues downstream.")
            all_samples_df['text'] = "" # Add empty text column to prevent KeyErrors, though processing might fail

        # Shuffle samples and take the specified number if num_samples is provided
        if num_samples is not None:
            if num_samples > 0:
                all_samples_df = all_samples_df.sample(frac=1, random_state=seed).reset_index(drop=True)
                if num_samples >= len(all_samples_df):
                    logger.info(f"Requested {num_samples} samples for MasakhaNER {lang_code} ({split}), but only {len(all_samples_df)} are available. Using all available samples.")
                    final_samples_df = all_samples_df
                else:
                    final_samples_df = all_samples_df.head(num_samples)
                    logger.info(f"Selected {len(final_samples_df)} samples for MasakhaNER {lang_code} ({split}) after requesting {num_samples} with seed {seed}.")
            else: # num_samples is 0 or negative
                logger.warning(f"Requested {num_samples} samples for MasakhaNER {lang_code} ({split}). Returning empty DataFrame.")
                return pd.DataFrame()
        else: # num_samples is None, return all loaded and processed samples
            final_samples_df = all_samples_df
            logging.info(f"Returning all {len(final_samples_df)} loaded samples for MasakhaNER {lang_code} ({split}) as num_samples was None.")

        # Log columns and head of the final DataFrame before returning
        logger.debug(f"Columns in final DataFrame for {lang_code} ({split}): {final_samples_df.columns.tolist()}")
        logger.debug(f"Data types of final DataFrame for {lang_code} ({split}):\n{final_samples_df.dtypes}")
        logger.debug(f"First 3 rows of final DataFrame for {lang_code} ({split}):\n{final_samples_df.head(3).to_string()}")

        logging.info(f"Successfully loaded and processed {len(final_samples_df)} MasakhaNER samples for language '{lang_code}' (HF config: {split}), split '{split}'.")
        return final_samples_df

    except Exception as e:
        logger.error(f"An unexpected error occurred while loading/processing MasakhaNER for {lang_code}, split '{split}': {e}", exc_info=True)
        return pd.DataFrame()

# Example usage (for testing this script directly)
if __name__ == '__main__':
    logger.info("--- Testing MasakhaNER Loader ---")

    # Test Hausa
    ha_samples = load_masakhaner_samples('ha', num_samples=10)
    if not ha_samples.empty:
        logger.info(f"Hausa test samples: {len(ha_samples)}")
        logger.info(ha_samples.head())
        if 'entities' in ha_samples.columns and len(ha_samples.iloc[0]['entities']) > 0:
            logger.info(f"Entities of first HA sample: {ha_samples.iloc[0]['entities']}")
        else:
            logger.info("First HA sample has no entities or 'entities' column is missing.")
    else:
        logger.info("Failed to load Hausa samples or no samples returned.")

    # Test Swahili
    sw_samples = load_masakhaner_samples('sw', num_samples=5) # Test with a different number
    if not sw_samples.empty:
        logger.info(f"Swahili test samples: {len(sw_samples)}")
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