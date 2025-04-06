import pandas as pd
import os
from typing import Optional, List
from datasets import load_dataset, get_dataset_split_names
from tqdm import tqdm

# Mapping from simple language codes to AfriQA dataset configuration names if needed
# For afriqa, the config name seems to directly match the language code (e.g., 'ha', 'yo')
LANG_CODE_MAP = {
    'ha': 'ha',
    'yo': 'yo',
    'sw': 'sw', # Swahili might also be available
    # Add other languages if needed, check dataset viewer on Hugging Face Hub
}

def _load_samples_from_split(dataset_name: str, lang_config: str, split: str, num_samples_to_load: Optional[int]) -> List[dict]:
    """Helper function to load samples from a specific split of the AfriQA dataset."""
    samples = []
    count = 0
    print(f"Attempting to load {lang_config} samples from {dataset_name} ({split} split)...")

    try:
        # Load the specific language configuration and split
        # Using streaming=True to avoid downloading the entire dataset variant upfront if only taking a few samples
        dataset = load_dataset(dataset_name, lang_config, split=split, streaming=True)

        processed_count = 0
        for example in tqdm(dataset, desc=f"Processing {lang_config} ({split})"):
            processed_count += 1
            example_id = example.get('id', f'split_{split}_idx_{processed_count}') # Use index if ID missing
            question = example.get('question', '')
            context = example.get('context', '')
            answers = example.get('answers', {'text': [], 'answer_start': []})

            # Basic validation: Skip if essential fields are missing
            if not question or not context or not answers or not answers.get('text'):
                print(f"  Skipping sample {example_id}: Missing essential fields (question, context, or answer text).")
                continue

            # Ensure answers format is consistent (list of strings)
            if not isinstance(answers['text'], list):
                 answers['text'] = [str(answers['text'])] # Convert single answer to list
                 # If answer_start was also single, convert it too (assuming it corresponds)
                 if 'answer_start' in answers and not isinstance(answers['answer_start'], list):
                     answers['answer_start'] = [int(answers.get('answer_start', -1))]


            # Create and append the sample
            sample = {
                'context': context,
                'question': question,
                'answers': answers, # Keep the original answer format from dataset
                'id': example_id
            }
            samples.append(sample)
            count += 1

            # Check if requested number of samples is reached
            if num_samples_to_load is not None and count >= num_samples_to_load:
                print(f"  Reached requested number of samples ({num_samples_to_load}).")
                break

    except Exception as e:
        print(f"Error loading or processing {dataset_name} ({lang_config}, {split} split): {e}")
        # Depending on the error, might want to return empty list or raise exception
        return []

    print(f"  Successfully extracted {len(samples)} valid samples from {processed_count} processed records.")
    return samples

def load_afriqa_samples(lang_code: str, num_samples: Optional[int] = None, split: str = "train") -> pd.DataFrame:
    """
    Load AfriQA samples for a specific language.
    Tries the specified split first. AfriQA might primarily have a 'train' split for some languages.
    Check Hugging Face dataset viewer for available splits per language.

    Args:
        lang_code: Language code (e.g., 'ha' for Hausa, 'yo' for Yoruba)
        num_samples: Target number of samples to load (None for all available in the specified split).
        split: Preferred dataset split to use ('train', 'validation', 'test'). 'train' is often the main one.

    Returns:
        DataFrame containing the samples
    """
    dataset_name = "facebook/afriqa"

    afriqa_lang_config = LANG_CODE_MAP.get(lang_code)
    if not afriqa_lang_config:
        print(f"ERROR: Unsupported or unmapped language code '{lang_code}' for AfriQA loader.")
        return pd.DataFrame({'context': [], 'question': [], 'answers': [], 'id': []})

    # Check available splits for the language configuration
    try:
        available_splits = get_dataset_split_names(dataset_name, afriqa_lang_config)
        print(f"Available splits for {dataset_name}/{afriqa_lang_config}: {available_splits}")
        if split not in available_splits:
            # Fallback logic: if preferred split doesn't exist, try 'train' or the first available split
            if 'train' in available_splits:
                print(f"WARN: Requested split '{split}' not found. Using 'train' split instead.")
                split = 'train'
            elif available_splits:
                print(f"WARN: Requested split '{split}' not found. Using first available split '{available_splits[0]}' instead.")
                split = available_splits[0]
            else:
                 print(f"ERROR: No splits found for {dataset_name}/{afriqa_lang_config}.")
                 return pd.DataFrame({'context': [], 'question': [], 'answers': [], 'id': []})

    except Exception as e:
        print(f"Could not verify splits for {dataset_name}/{afriqa_lang_config}: {e}. Attempting to load '{split}' split anyway.")
        # Proceed with the requested split, hoping it exists

    # Load samples from the determined split
    all_samples = _load_samples_from_split(dataset_name, afriqa_lang_config, split, num_samples)

    print(f"Loaded {len(all_samples)} {afriqa_lang_config} samples in total from the '{split}' split.")

    # Return DataFrame
    if not all_samples:
        print(f"WARNING: No {afriqa_lang_config} samples found or loaded from the '{split}' split!")
        return pd.DataFrame({'context': [], 'question': [], 'answers': [], 'id': []})

    # Ensure we don't return more samples than requested if num_samples was specified
    if num_samples is not None and len(all_samples) > num_samples:
        all_samples = all_samples[:num_samples]
        print(f"  Truncated to requested {num_samples} samples.")

    return pd.DataFrame(all_samples)

# Example usage (optional, for testing)
if __name__ == '__main__':
    hausa_samples = load_afriqa_samples('ha', 5)
    print("\nHausa Samples:")
    print(hausa_samples.head())

    yoruba_samples = load_afriqa_samples('yo', 5, split='train') # Example specifying split
    print("\nYoruba Samples:")
    print(yoruba_samples.head()) 