import pandas as pd
from datasets import load_dataset
from typing import Optional, List
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Language map for khalidalt/tydiqa-goldp dataset configurations
# This maps your 2-letter codes to the HF dataset's expected config names.
# Verify these on the HF dataset page if issues arise.
TYDIQA_LANG_CONFIG_MAP = {
    'en': 'english',
    'sw': 'swahili',
    'ar': 'arabic',
    'bn': 'bengali',
    'fi': 'finnish',
    'id': 'indonesian',
    'ja': 'japanese',
    'ko': 'korean',
    'ru': 'russian',
    'th': 'thai',
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
    elif lang_code == 'fi':
        return [
            {'id': 'fallback_fi_1', 'context': 'Taivas on sininen.', 'question': 'Minkä värinen taivas on?', 'answers': {'text': ['sininen'], 'answer_start': [9]}, 'language': 'fi', 'ground_truth': 'sininen'}
        ]
    return []

def load_tydiqa_samples(
    lang_code: str,
    num_samples: Optional[int] = None,
    split: str = 'validation',
    seed: int = 42
) -> pd.DataFrame:
    """
    Loads samples from the TyDiQA-GoldP dataset on Hugging Face.
    This version directly loads the language-specific configuration.
    """
    hf_dataset_name = "khalidalt/tydiqa-goldp"
    hf_lang_config = TYDIQA_LANG_CONFIG_MAP.get(lang_code)

    if not hf_lang_config:
        logger.error(f"Language code '{lang_code}' not found in TYDIQA_LANG_CONFIG_MAP. Aborting.")
        return pd.DataFrame()

    # For TyDiQA GoldP, the 'validation' split is standard for testing.
    # The dataset might not have a 'test' split for language-specific configs.
    if split == 'test':
        split = 'validation'
        logger.info(f"Defaulting to 'validation' split for TyDiQA GoldP language '{lang_code}'.")

    logger.info(f"Loading TyDiQA GoldP for language: {lang_code} (HF config: {hf_lang_config}), split: {split}")

    try:
        # Load the dataset using the language-specific configuration name.
        dataset = load_dataset(hf_dataset_name, name=hf_lang_config, split=split, trust_remote_code=True)
        logger.info(f"Successfully loaded TyDiQA GoldP dataset for '{hf_lang_config}' (split {split}). Full size: {len(dataset)}.")

    except Exception as e:
        logger.error(f"Failed to load TyDiQA for {lang_code} (config: {hf_lang_config}). Error: {e}. Using fallback.", exc_info=True)
        return pd.DataFrame(create_fallback_samples(lang_code))

    all_samples = []
    for i, example in enumerate(dataset):
        # Be more robust to key naming variations ('question' vs 'question_text', etc.)
        question = example.get('question_text') or example.get('question', '')
        context = example.get('passage_text') or example.get('context', '')
        
        # Handle different answer/annotation structures by checking multiple possibilities
        annotations_list = example.get('annotations', [])
        answers_list = example.get('answers', []) # For original TyDiQA structure
        
        answer_texts = []
        
        # 1. Prioritize 'annotations' structure (list of dicts with 'answer_text')
        if isinstance(annotations_list, list) and annotations_list:
            for annotation in annotations_list:
                if isinstance(annotation, dict):
                    answer_text = annotation.get('answer_text')
                    if answer_text and isinstance(answer_text, str):
                        answer_texts.append(answer_text)

        # 2. Fallback to 'answers' structure (dict with a 'text' list)
        elif isinstance(answers_list, dict) and 'text' in answers_list:
            if isinstance(answers_list['text'], list):
                 answer_texts.extend([str(t) for t in answers_list['text'] if t])

        # We need at least one valid answer for a sample to be useful
        if not question or not context or not answer_texts:
            logger.warning(f"Skipping sample {i} for {lang_code} due to missing question, context, or answer text.")
            continue
        
        # Take the first valid answer as the ground truth for simplicity in baseline/CoT F1.
        ground_truth = answer_texts[0]

        all_samples.append({
            'id': example.get('example_id', f'{lang_code}-{split}-{i}'),
            'question': question,
            'context': context,
            'answers': ground_truth, # Storing the single ground truth string
            'all_answers': answer_texts, # Storing all possible answers
            'language': lang_code
        })

    if not all_samples:
        logger.warning(f"No valid samples could be processed for {lang_code}, split {split}. Using fallback.")
        return pd.DataFrame(create_fallback_samples(lang_code))

    all_samples_df = pd.DataFrame(all_samples)

    # Shuffle samples and take the specified number if num_samples is provided
    if num_samples is not None and num_samples > 0:
        if num_samples < len(all_samples_df):
            all_samples_df = all_samples_df.sample(n=num_samples, random_state=seed).reset_index(drop=True)
            logger.info(f"Selected {len(all_samples_df)} samples for {lang_code} after requesting {num_samples}.")
        else:
            logger.info(f"Requested {num_samples} samples, but only {len(all_samples_df)} are available. Using all.")
    
    logger.info(f"Successfully processed {len(all_samples_df)} samples for TyDiQA language '{lang_code}', split '{split}'.")
    return all_samples_df

# Example usage (for testing this script directly)
if __name__ == '__main__':
    print("--- Testing TyDiQA GoldP Loader ---")

    # Test English
    en_samples = load_tydiqa_samples('en', num_samples=1, split='validation')
    if not en_samples.empty:
        print(f"\\nEnglish validation samples (1%): {len(en_samples)}")
        print(en_samples.head())
        print(f"Context of first EN sample: {en_samples.iloc[0]['context'][:100]}...")
        print(f"Question of first EN sample: {en_samples.iloc[0]['question']}")
        print(f"Ground truth of first EN sample: {en_samples.iloc[0]['ground_truth']}")
    else:
        print("\\nFailed to load English samples.")

    # Test Swahili
    sw_samples = load_tydiqa_samples('sw', num_samples=5, split='train') # Try train split for sw
    if not sw_samples.empty:
        print(f"\\nSwahili train samples (5%): {len(sw_samples)}")
        print(sw_samples.head())
    else:
        print("\\nFailed to load Swahili samples.")

    # Test Finnish
    fi_samples = load_tydiqa_samples('fi', num_samples=2, split='validation')
    if not fi_samples.empty:
        print(f"\\nFinnish validation samples (2%): {len(fi_samples)}")
        print(fi_samples.head())
    else:
        print("\\nFailed to load Finnish samples.")
    
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