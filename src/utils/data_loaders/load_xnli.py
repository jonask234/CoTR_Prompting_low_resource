import os
import pandas as pd
import random
from datasets import load_dataset

def load_xnli_samples(lang_code, num_samples=None, split='test'):
    """
    Load samples from the XNLI dataset for a specific language.
    Supports English (en), Swahili (sw), and Urdu (ur).
    
    Args:
        lang_code: Language code (e.g., 'en', 'sw', 'ur')
        num_samples: Number of samples to retrieve (None for all)
        split: Dataset split to use ('train', 'test', 'validation')
    
    Returns:
        DataFrame with columns: premise, hypothesis, label, language, original_lang
    """
    try:
        # XNLI supports: 'ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'
        # We're focusing on 'en', 'sw', and 'ur' for our experiments
        supported_langs = ['en', 'sw', 'ur']
        xnli_langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
        
        # Check if language is directly supported
        if lang_code not in xnli_langs:
            print(f"Warning: Language {lang_code} not available in XNLI. Available languages: {', '.join(xnli_langs)}")
            print(f"Using English as fallback.")
            dataset_lang = 'en'
            using_fallback = True
        else:
            dataset_lang = lang_code
            using_fallback = False
        
        if using_fallback:
            print(f"WARNING: Using {dataset_lang} data as fallback for {lang_code}!")
        
        # Load dataset from Hugging Face
        dataset = load_dataset("xnli", dataset_lang)
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(dataset[split])
        
        # Randomly sample if num_samples is specified
        if num_samples is not None and num_samples < len(df):
            df = df.sample(n=num_samples, random_state=42)
        
        # Add language columns
        df['language'] = lang_code  # Requested language code
        df['original_lang'] = dataset_lang  # Actual language of the text
        df['is_fallback'] = using_fallback  # Flag to indicate if this is a fallback
        
        print(f"Loaded {len(df)} samples for {lang_code} from XNLI {split} split.")
        if using_fallback:
            print(f"NOTE: These samples are actually in {dataset_lang}, not {lang_code}!")
        return df
    
    except Exception as e:
        print(f"Error loading XNLI dataset for {lang_code}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def get_xnli_stats():
    """
    Get statistics about available languages in XNLI.
    
    Returns:
        Dictionary with language statistics
    """
    stats = {}
    
    try:
        # List available languages in XNLI
        languages = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
        
        for lang in languages:
            try:
                dataset = load_dataset("xnli", lang)
                stats[lang] = {
                    'train': len(dataset['train']),
                    'validation': len(dataset['validation']),
                    'test': len(dataset['test']),
                    'total': len(dataset['train']) + len(dataset['validation']) + len(dataset['test'])
                }
            except Exception as e:
                print(f"Could not load statistics for {lang}: {e}")
                stats[lang] = {'error': str(e)}
        
    except Exception as e:
        print(f"Error loading XNLI statistics: {e}")
    
    return stats 