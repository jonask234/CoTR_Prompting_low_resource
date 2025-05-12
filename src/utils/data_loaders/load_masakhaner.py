import os
import pandas as pd
import json
import random
from datasets import load_dataset

def load_masakhaner_samples(lang_code, num_samples=None, split='test', seed=42):
    """
    Load MasakhaNER samples for the specified language from HuggingFace.
    
    Args:
        lang_code: Language code (e.g., 'sw', 'ha', 'yo')
        num_samples: Number of samples to load (None = all)
        split: Data split ('train', 'validation', 'test')
        seed: Random seed for sampling
    
    Returns:
        DataFrame containing NER samples
    """
    # Map our codes to MasakhaNER dataset codes
    # Available languages: 'amh', 'hau', 'ibo', 'kin', 'lug', 'luo', 'pcm', 'swa', 'wol', 'yor'
    lang_map = {
        'en': None,  # English not available in MasakhaNER
        'sw': 'swa',  # Swahili
        'yo': 'yor',  # Yoruba
        'hau': 'hau', # Hausa (Corrected from 'ha')
        'ha': 'hau',  # Added 'ha' mapping to 'hau'
        'ibo': 'ibo', # Igbo
        'lug': 'lug', # Luganda
        'luo': 'luo', # Luo
        'wol': 'wol', # Wolof
        'amh': 'amh', # Amharic
        'kin': 'kin', # Kinyarwanda
        'pcm': 'pcm'  # Nigerian Pidgin
    }
    
    # Get dataset language code
    dataset_lang = lang_map.get(lang_code)
    
    # Check if language is supported
    if dataset_lang is None:
        if lang_code == 'en':
            print(f"English (en) is not available in MasakhaNER. Using CoNLL-2003 English dataset instead.")
            # Ensure seed is passed to CoNLL loader if it accepts it
            return load_conll2003_samples(split=split, num_samples=num_samples, seed=seed)
        else:
            print(f"Error: Language '{lang_code}' not supported in MasakhaNER or CoNLL-2003 fallback.")
            return pd.DataFrame()
    
    # Adjust split name for HuggingFace's naming convention
    hf_split = 'validation' if split == 'dev' else split
    
    # Load dataset from HuggingFace
    try:
        print(f"Loading MasakhaNER for {dataset_lang} (mapped from {lang_code}) from HuggingFace...")
        dataset = load_dataset("masakhane/masakhaner", dataset_lang, split=hf_split, trust_remote_code=True)
        print(f"Successfully loaded dataset with {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset from HuggingFace: {e}")
        return pd.DataFrame()
    
    # Convert to our format
    samples_list = []
    
    for item in dataset:
        # Get tokens and NER tags
        tokens = item['tokens']
        ner_tags = item['ner_tags']
        
        # Convert numeric NER tags to entity spans
        entities = []
        current_entity = None
        
        # Map tag IDs to entity types (0=O, odd=B-X, even=I-X)
        tag_map = {
            0: "O",
            1: "B-PER", 2: "I-PER",
            3: "B-ORG", 4: "I-ORG",
            5: "B-LOC", 6: "I-LOC",
            7: "B-DATE", 8: "I-DATE"
        }
        
        # Get string tags from numeric ids
        string_tags = [tag_map.get(tag, "O") for tag in ner_tags]
        
        # Extract entities from BIO tags
        for i, tag in enumerate(string_tags):
            if tag.startswith('B-'):
                # End previous entity if exists
                if current_entity is not None:
                    entities.append(current_entity)
                # Beginning of a new entity
                entity_type = tag[2:]  # Remove B- prefix
                current_entity = {
                    'start': i,
                    'end': i + 1,
                    'entity_type': entity_type
                }
            elif tag.startswith('I-') and current_entity is not None:
                # Inside of an entity - extend it if types match
                if tag[2:] == current_entity['entity_type']:
                    current_entity['end'] = i + 1
                else: # Type mismatch, end previous entity and start new one? Or treat as O?
                      # Standard BIO: End previous, this I-tag is invalid without B-tag
                      entities.append(current_entity)
                      current_entity = None
            else:
                # Outside (O) or beginning of a new entity type
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add the last entity if there is one
        if current_entity is not None:
            entities.append(current_entity)
        
        # Create sample
        sample = {
            'id': item.get('id', str(len(samples_list))),
            'tokens': tokens,
            'entities': entities
        }
        
        samples_list.append(sample)
    
    # Create DataFrame
    samples_df = pd.DataFrame(samples_list)
    
    # Subsample if requested
    if num_samples is not None and num_samples < len(samples_df):
        print(f"Sampling {num_samples} from {len(samples_df)} loaded samples using seed {seed}...")
        samples_df = samples_df.sample(n=num_samples, random_state=seed)
    
    print(f"Processed {len(samples_df)} samples for {lang_code} from {split} split.")
    
    return samples_df

def load_conll2003_samples(split='test', num_samples=None, seed=42):
    """
    Load CoNLL-2003 English NER dataset as fallback for English.
    
    Args:
        split: Data split ('train', 'validation', 'test')
        num_samples: Number of samples to load (None = all)
        seed: Random seed for sampling
    
    Returns:
        DataFrame containing NER samples
    """
    try:
        print(f"Loading CoNLL-2003 English dataset from HuggingFace...")
        # Make sure we're using the right split name
        hf_split = 'validation' if split == 'dev' else split
        dataset = load_dataset("conll2003", split=hf_split)
        print(f"Successfully loaded dataset with {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading CoNLL-2003 dataset: {e}")
        return pd.DataFrame()
    
    # Convert to our format
    samples_list = []
    
    for item in dataset:
        tokens = item['tokens']
        ner_tags = item['ner_tags']
        
        # Convert CoNLL tags to entity spans
        entities = []
        current_entity = None
        
        # Map CoNLL tag IDs to entity types
        tag_map = {
            0: "O",
            1: "B-PER", 2: "I-PER", 
            3: "B-ORG", 4: "I-ORG",
            5: "B-LOC", 6: "I-LOC",
            7: "B-MISC", 8: "I-MISC" # MISC will be mapped to DATE for consistency
        }
        
        # Map CoNLL string tags to our entity types
        entity_type_map = {
            "PER": "PER",
            "ORG": "ORG", 
            "LOC": "LOC",
            "MISC": "DATE"  # Map MISC to DATE for consistency with MasakhaNER
        }
        
        # Get string tags from numeric ids
        string_tags = [tag_map.get(tag, "O") for tag in ner_tags]
        
        # Extract entities from BIO tags
        for i, tag in enumerate(string_tags):
            if tag.startswith('B-'):
                # Beginning of a new entity
                original_type = tag[2:]
                entity_type = entity_type_map.get(original_type, original_type)
                current_entity = {
                    'start': i,
                    'end': i + 1,
                    'entity_type': entity_type
                }
            elif tag.startswith('I-') and current_entity is not None:
                # Inside of an entity - extend it if types match
                current_type = current_entity['entity_type']
                tag_type = entity_type_map.get(tag[2:], tag[2:])
                if tag_type == current_type:
                    current_entity['end'] = i + 1
            else:
                # Outside (O) or beginning of a new entity type
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add the last entity if there is one
        if current_entity is not None:
            entities.append(current_entity)
        
        # Create sample
        sample = {
            'id': item.get('id', str(len(samples_list))),
            'tokens': tokens,
            'entities': entities
        }
        
        samples_list.append(sample)
    
    # Create DataFrame
    samples_df = pd.DataFrame(samples_list)
    
    # Subsample if requested
    if num_samples is not None and num_samples < len(samples_df):
        print(f"Sampling {num_samples} from {len(samples_df)} loaded CoNLL samples using seed {seed}...")
        samples_df = samples_df.sample(n=num_samples, random_state=seed)
    
    print(f"Processed {len(samples_df)} samples for English (en) from {split} split.")
    
    return samples_df

if __name__ == "__main__":
    # Test the loader
    for lang in ['en', 'sw', 'yo']:
        samples = load_masakhaner_samples(lang, num_samples=5, split='test')
        print(f"\nExample for {lang}:")
        if not samples.empty:
            sample = samples.iloc[0]
            print(f"Tokens: {' '.join(sample['tokens'])}")
            print(f"Entities: {sample['entities']}") 