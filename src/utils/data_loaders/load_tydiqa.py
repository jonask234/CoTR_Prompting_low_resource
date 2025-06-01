import pandas as pd
from datasets import load_dataset
from typing import Optional, List
import os
import logging

# Language map for khalidalt/tydiqa-goldp dataset configurations
# This maps your 2-letter codes to the HF dataset's expected config names.
# Verify these on the HF dataset page if issues arise.
TYDIQA_LANG_CONFIG_MAP = {
    "goldp": {
        "dataset_name": "khalidalt/tydiqa-goldp",
        "split_map": {"train": "train", "validation": "validation", "dev": "validation", "test": "validation"}, # GoldP has train and validation
        "default_split": "validation" # Changed from "train" to "validation"
    },
    "primary": { # Minimal task (TyDiQA Primary)
        "dataset_name": "tydiqa", # Main HF dataset
    },
    'en': 'english',
    'sw': 'swahili',
    'te': 'telugu',
    'ar': 'arabic',
    'bn': 'bengali',
    'fi': 'finnish',
    'id': 'indonesian',
    'ja': 'japanese', # Added japanese as it's in TyDiQA GoldP
    'ko': 'korean',
    'ru': 'russian',
    # 'th': 'thai' # Thai might not be in khalidalt/tydiqa-goldp, verify
}

# Function to create fallback samples (can be kept for robustness or testing)
def create_fallback_samples(lang_code: str) -> list:
    """Creates a few fallback samples if loading from HF fails."""
    print(f"WARNING: Using fallback samples for language: {lang_code}")
    if lang_code == 'en':
        return [
            {'id': 'fallback_en_1', 'context': 'The sky is blue.', 'question': 'What color is the sky?', 'answers': {'text': ['blue'], 'answer_start': [11]}, 'language': 'en', 'ground_truth': 'blue'},
            {'id': 'fallback_en_2', 'context': 'The capital of France is Paris.', 'question': 'What is the capital of France?', 'answers': {'text': ['Paris'], 'answer_start': [26]}, 'language': 'en', 'ground_truth': 'Paris'}
        ]
    elif lang_code == 'sw':
        return [
            {'id': 'fallback_sw_1', 'context': 'Jua ni la manjano.', 'question': 'Jua lina rangi gani?', 'answers': {'text': ['manjano'], 'answer_start': [10]}, 'language': 'sw', 'ground_truth': 'manjano'}
        ]
    elif lang_code == 'te':
        return [
            {'id': 'fallback_te_1', 'context': 'ఆకాశం నీలం రంగులో ఉంది.', 'question': 'ఆకాశం ఏ రంగులో ఉంది?', 'answers': {'text': ['నీలం'], 'answer_start': [8]}, 'language': 'te', 'ground_truth': 'నీలం'}
        ]
    return []

def load_tydiqa_samples(
    lang_code: str,
    sample_percentage: Optional[float] = None,
    split: str = 'validation'  # Default split for TyDiQA GoldP is often validation/dev
) -> pd.DataFrame:
    """
    Load TyDiQA samples for a specific language from Hugging Face Hub (khalidalt/tydiqa-goldp).
    
    Args:
        lang_code: Language code (e.g., 'sw' for Swahili, 'en' for English, 'te' for Telugu).
        sample_percentage: Percentage of samples to load (e.g., 10 for 10%), or None for all.
        split: Dataset split to use. Maps 'dev' to 'validation'.
               TyDiQA-GoldP typically has 'train', 'validation' (sometimes as 'dev').
        
    Returns:
        DataFrame containing loaded samples, or empty DataFrame if loading fails.
    """
    dataset_name = "khalidalt/tydiqa-goldp"
    hf_lang_config = TYDIQA_LANG_CONFIG_MAP.get(lang_code)

    if not hf_lang_config:
        print(f"ERROR: Language code '{lang_code}' not mapped for {dataset_name}. Cannot load.")
        return pd.DataFrame()
    
    # Map 'dev' split to 'validation' as khalidalt/tydiqa-goldp uses 'validation'
    hf_split = 'validation' if split == 'dev' else split
    # Ensure split is one of the typical HF dataset splits
    if hf_split not in ['train', 'validation', 'test']:
        print(f"Warning: Invalid split '{hf_split}' specified. Defaulting to 'validation'.")
        hf_split = 'validation'

    print(f"Attempting to load '{lang_code}' (config: '{hf_lang_config}') samples from {dataset_name}, split '{hf_split}'...")

    all_samples_list = []
    try:
        # Load the dataset with the specific language configuration
        # trust_remote_code=True might be needed for some datasets
        dataset = load_dataset(dataset_name, name=hf_lang_config, split=hf_split, trust_remote_code=True)
        print(f"Successfully loaded dataset for {lang_code} ({hf_lang_config}), split {hf_split} from Hugging Face Hub.")

        # Process dataset into a list of dicts
        for example in dataset:
            # The structure of khalidalt/tydiqa-goldp provides:
            # 'id', 'title', 'context', 'question', 'answers' (dict with 'text' list and 'answer_start' list)
            sample_id = example.get('id', f"{lang_code}_{hf_split}_{len(all_samples_list)}")
            context = example.get('passage_text', '')
            question = example.get('question_text', '')
            answers_dict = example.get('answers', {'text': [], 'answer_start': []})
            
            # Ensure answers_dict['text'] is a list, even if empty or None
            answer_texts = answers_dict.get('text', [])
            if not isinstance(answer_texts, list):
                answer_texts = [] # Default to empty list if not a list

            # Ensure answer_starts_list matches length of answer_texts if possible, or provide sensible defaults
            answer_starts_list = answers_dict.get('answer_start', [])
            if not isinstance(answer_starts_list, list) or len(answer_starts_list) != len(answer_texts):
                 answer_starts_list = [-1] * len(answer_texts) # Default answer_start if problematic


            all_samples_list.append({
                'id': sample_id,
                'context': context,
                'question': question,
                'answers': {'text': answer_texts, 'answer_start': answer_starts_list},
                'language': lang_code # Store the original 2-letter lang_code
            })

    except Exception as e:
        logging.error(f"Failed to load TyDiQA GoldP for language {lang_code}, split '{split}': {e}", exc_info=True)
        # Fallback to returning an empty DataFrame in case of any error during loading or processing
        return pd.DataFrame()

    if not all_samples_list:
        print(f"No samples found for language '{lang_code}' (config: {hf_lang_config}), split '{hf_split}'.")
        return pd.DataFrame()
                
    samples_df = pd.DataFrame(all_samples_list)
    
    # Sample if requested
    if sample_percentage is not None and 0 < sample_percentage <= 100:
        num_total_samples = len(samples_df)
        num_to_sample = max(1, int(num_total_samples * (sample_percentage / 100.0)))
        if num_to_sample < num_total_samples:
            print(f"Sampling {sample_percentage}% ({num_to_sample} examples) from {num_total_samples} total for {lang_code}.")
            samples_df = samples_df.sample(n=num_to_sample, random_state=42).reset_index(drop=True)
    elif sample_percentage is not None: # Handles cases where sample_percentage is not None but not in (0, 100]
        print(f"Invalid sample_percentage value ({sample_percentage}). Using all {len(samples_df)} samples for {lang_code}.")
        # No change to samples_df, all samples are used if sample_percentage is invalid but not None.
    
    # Extract the first answer text as ground_truth
    # Ensure 'answers' column exists and its 'text' entry is a list
    samples_df['ground_truth'] = samples_df['answers'].apply(
        lambda ans: ans['text'][0] if isinstance(ans, dict) and 'text' in ans and isinstance(ans['text'], list) and ans['text'] else None
    )
    
    # Drop rows where ground_truth could not be extracted (e.g. no answers)
    samples_df.dropna(subset=['ground_truth'], inplace=True)

    print(f"Loaded and processed {len(samples_df)} samples for {lang_code} ({hf_lang_config}), split '{hf_split}'.")
    return samples_df
    
# Example usage (for testing this script directly)
if __name__ == '__main__':
    print("--- Testing TyDiQA GoldP Loader ---")

    # Test English
    en_samples = load_tydiqa_samples('en', sample_percentage=1, split='validation')
    if not en_samples.empty:
        print(f"\\nEnglish validation samples (1%): {len(en_samples)}")
        print(en_samples.head())
        print(f"Context of first EN sample: {en_samples.iloc[0]['context'][:100]}...")
        print(f"Question of first EN sample: {en_samples.iloc[0]['question']}")
        print(f"Ground truth of first EN sample: {en_samples.iloc[0]['ground_truth']}")
    else:
        print("\\nFailed to load English samples.")

    # Test Swahili
    sw_samples = load_tydiqa_samples('sw', sample_percentage=5, split='train') # Try train split for sw
    if not sw_samples.empty:
        print(f"\\nSwahili train samples (5%): {len(sw_samples)}")
        print(sw_samples.head())
    else:
        print("\\nFailed to load Swahili samples.")

    # Test Telugu
    te_samples = load_tydiqa_samples('te', sample_percentage=2, split='validation')
    if not te_samples.empty:
        print(f"\\nTelugu validation samples (2%): {len(te_samples)}")
        print(te_samples.head())
    else:
        print("\\nFailed to load Telugu samples.")
    
    # Test a language not in our TYDIQA_LANG_CONFIG_MAP to see fallback/error
    # xx_samples = load_tydiqa_samples('xx', sample_percentage=1)
    # if xx_samples.empty:
    #     print("\\nCorrectly returned empty for unsupported lang 'xx'.")

    # Test loading all samples for a language
    # fi_samples_all = load_tydiqa_samples('fi', split='train')
    # if not fi_samples_all.empty:
    #     print(f"\\nFinnish train samples (all): {len(fi_samples_all)}")
    #     print(fi_samples_all.head(2))
    # else:
    #     print("\\nFailed to load all Finnish train samples.")

# Remove load_tydiqa_local and its helper extract_qa_pairs as they are no longer used.
# The rest of the file can be cleaned up if those functions are the only ones being removed.
# For this edit, I will focus on removing load_tydiqa_local specifically. 