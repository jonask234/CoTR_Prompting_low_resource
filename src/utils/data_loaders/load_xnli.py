# -*- coding: utf-8 -*-
import pandas as pd
from datasets import load_dataset

# XNLI-Label-Mapping
XNLI_LABEL_MAP = {
    0: 'entailment',
    1: 'neutral',
    2: 'contradiction'
}

# Supported Languages
XNLI_SUPPORTED_LANGUAGES = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

# Language Configurations
XNLI_LANG_CONFIG_MAP = {
    "en": ("xnli", "en"),
    "fr": ("xnli", "fr"),
    "es": ("xnli", "es"),
    "de": ("xnli", "de"),
    "el": ("xnli", "el"),
    "bg": ("xnli", "bg"),
    "ru": ("xnli", "ru"),
    "tr": ("xnli", "tr"),
    "ar": ("xnli", "ar"),
    "vi": ("xnli", "vi"),
    "th": ("xnli", "th"),
    "zh": ("xnli", "zh"),
    "hi": ("xnli", "hi"),
    "sw": ("xnli", "sw"),
    "ur": ("xnli", "ur"),
}

def load_xnli_samples(
    lang_code,
    num_samples = None,
    split = 'test',
    seed = 42
):
    # Lädt NLI-Samples 
    if lang_code not in XNLI_LANG_CONFIG_MAP:
        print(f"Language code '{lang_code}' is not supported.")
        return pd.DataFrame()

    dataset_name, config_name = XNLI_LANG_CONFIG_MAP[lang_code]
    
    try:
        dataset = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
        
        all_samples_df = pd.DataFrame(dataset)
        
        if all_samples_df.empty:
            return pd.DataFrame()

        # Labels von ganzen Zahlen in Strings umwandeln
        if 'label' in all_samples_df.columns and pd.api.types.is_numeric_dtype(all_samples_df['label']):
            all_samples_df['original_label_int'] = all_samples_df['label']
            all_samples_df['label'] = all_samples_df['label'].map(XNLI_LABEL_MAP)
        else:
            all_samples_df['original_label_int'] = -1 
        
        # Relevante Spalten auswählen
        final_columns = ['premise', 'hypothesis', 'label', 'original_label_int']
        for col in final_columns:
            if col not in all_samples_df.columns:
                if col == 'label':
                    all_samples_df[col] = "unknown"
                elif col == 'original_label_int':
                    all_samples_df[col] = -1
                else:
                    all_samples_df[col] = ""
        
        existing_final_columns = [col for col in final_columns if col in all_samples_df.columns]
        all_samples_df = all_samples_df[existing_final_columns]

        # Samples auswählen
        if num_samples is not None:
            if num_samples > 0:
                all_samples_df = all_samples_df.sample(frac=1, random_state=seed).reset_index(drop=True)
                
                if num_samples < len(all_samples_df):
                    final_samples_df = all_samples_df.head(num_samples)
                else:
                    final_samples_df = all_samples_df
            else:
                return pd.DataFrame()
        else:
            final_samples_df = all_samples_df

        return final_samples_df

    except Exception as e:
        print(f"Error loading XNLI for language '{lang_code}': {e}")
        return pd.DataFrame()

# Beispielverwendung zum Testen
if __name__ == '__main__':
    print("--- Testing XNLI Loader ---")

    en_samples = load_xnli_samples('en', num_samples=100, split='test')
    if not en_samples.empty:
        print(f"\nEnglish XNLI 'test' samples (100 - {len(en_samples)}):")
        print(en_samples.head())
        print(en_samples['label'].value_counts())
    else:
        print("\nFailed to load English XNLI samples.")

    sw_samples = load_xnli_samples('sw', num_samples=100, split='train')
    if not sw_samples.empty:
        print(f"\nSwahili XNLI 'train' samples (100 - {len(sw_samples)}):")
        print(sw_samples.head())
        print(sw_samples['label'].value_counts())
    else:
        print("\nFailed to load Swahili XNLI samples.")

    ur_samples = load_xnli_samples('ur', num_samples=50, split='validation')
    if not ur_samples.empty:
        print(f"\nUrdu XNLI 'validation' samples (50 - {len(ur_samples)}):")
        print(ur_samples.head())
        print(ur_samples['label'].value_counts())
    else:
        print("\nFailed to load Urdu XNLI samples.") 