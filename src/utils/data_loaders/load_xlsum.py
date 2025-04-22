from datasets import load_dataset
import pandas as pd
import random
import numpy as np

# Mapping of ISO language codes to XL-Sum dataset names
XLSUM_LANG_MAP = {
    "sw": "swahili",
    "te": "telugu",
    "en": "english",
    "am": "amharic",
    "ha": "hausa"
}

def load_xlsum_samples(lang_code, num_samples=None, split="validation", random_state=42):
    """
    Load samples from the XL-Sum dataset for a specific language.
    
    Args:
        lang_code: ISO language code (e.g., 'sw', 'te', 'en')
        num_samples: Number of samples to load (None for all)
        split: Dataset split to use ('train', 'validation', 'test')
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame containing samples with 'text' (article) and 'summary' columns
    """
    # Check if language is supported
    if lang_code not in XLSUM_LANG_MAP:
        raise ValueError(f"Language code {lang_code} not supported. "
                        f"Available languages: {list(XLSUM_LANG_MAP.keys())}")
    
    # Get dataset name for this language
    dataset_name = XLSUM_LANG_MAP[lang_code]
    
    try:
        # Load dataset
        dataset = load_dataset("csebuetnlp/xlsum", dataset_name)
        
        # Check if split exists
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(dataset.keys())}")
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset[split])
        
        # Sample if needed
        if num_samples is not None:
            if num_samples == "10%":
                # Calculate 10% of the dataset
                num_to_sample = max(1, int(len(df) * 0.1))
                print(f"Sampling {num_to_sample} examples (10% of {len(df)}) for {dataset_name}")
            else:
                num_to_sample = min(num_samples, len(df))
                
            # Sample randomly
            df = df.sample(n=num_to_sample, random_state=random_state)
        
        # Select only relevant columns
        if 'id' in df.columns:
            return df[['id', 'text', 'summary']]
        else:
            return df[['text', 'summary']]
            
    except Exception as e:
        print(f"Error loading XL-Sum dataset for {dataset_name}: {e}")
        return pd.DataFrame()

def get_xlsum_stats():
    """Get statistics about the XL-Sum dataset for all supported languages."""
    stats = {}
    for code, name in XLSUM_LANG_MAP.items():
        try:
            dataset = load_dataset("csebuetnlp/xlsum", name)
            stats[code] = {
                "name": name,
                "train": len(dataset.get("train", [])),
                "validation": len(dataset.get("validation", [])),
                "test": len(dataset.get("test", []))
            }
        except Exception as e:
            stats[code] = {"error": str(e)}
    
    return stats

# Example usage
if __name__ == "__main__":
    print("XL-Sum Dataset Statistics:")
    stats = get_xlsum_stats()
    for lang_code, lang_stats in stats.items():
        print(f"{lang_code} ({lang_stats.get('name', 'unknown')}): "
              f"Train: {lang_stats.get('train', 'N/A')}, "
              f"Validation: {lang_stats.get('validation', 'N/A')}, "
              f"Test: {lang_stats.get('test', 'N/A')}")
    
    # Test loading samples
    print("\nLoading 5 Swahili samples:")
    sw_samples = load_xlsum_samples("sw", 5)
    for i, row in sw_samples.iterrows():
        print(f"Sample {i+1}:")
        print(f"Article: {row['text'][:100]}...")
        print(f"Summary: {row['summary']}\n") 