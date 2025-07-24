# -*- coding: utf-8 -*-
import pandas as pd
from datasets import load_dataset

# Spaltennamen
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'

# Sprachcodes
MASAKHANEWS_LANG_MAP = {
    "am": "amh",
    "en": "eng",
    "ha": "hau",
    "ig": "ibo",
    "pcm": "pcm",
    "pt": "por",
    "sw": "swa",
    "yo": "yor"
}

# Labels
MASAKHANEWS_LABEL_MAP = {
    0: "business",
    1: "entertainment", 
    2: "health",
    3: "politics",
    4: "religion",
    5: "sports",
    6: "technology"
}

def load_masakhanews_samples(
    lang_code, 
    num_samples = None,
    split = 'test', 
    dataset_repo = "masakhane/masakhanews",
    seed = 42
):
    # Lädt Samples für eine bestimmte Sprache 
    if lang_code not in MASAKHANEWS_LANG_MAP:
        print(f"Language code '{lang_code}' not found.")
        return pd.DataFrame()

    hf_config_name = MASAKHANEWS_LANG_MAP[lang_code]
    if split not in ['train', 'validation', 'test']:
        print(f"Invalid split '{split}'.")
        return pd.DataFrame()

    try:
        dataset = load_dataset(dataset_repo, name=hf_config_name, split=split, trust_remote_code=True)
        
        all_samples_list = []
        for i, example in enumerate(dataset):
            text = example.get('text')
            label_int = example.get('label')
            url = example.get('url', f"{lang_code}_{split}_{i}")

            if text is None or label_int is None:
                continue
            
            label_str = MASAKHANEWS_LABEL_MAP.get(label_int)
            if label_str is None:
                continue

            all_samples_list.append({
                'id': url,
                'text': text,
                'label': label_str,
                'language': lang_code
            })
        
        if not all_samples_list:
            return pd.DataFrame()
            
        samples_df = pd.DataFrame(all_samples_list)
        
        # auswählen
        if num_samples is not None:
            if num_samples > 0:
                samples_df = samples_df.sample(frac=1, random_state=seed).reset_index(drop=True)
                if num_samples < len(samples_df):
                    samples_df = samples_df.head(num_samples)
            else:
                return pd.DataFrame()
            
        return samples_df

    except Exception as e:
        print(f"Error loading MasakhaNEWS for language {lang_code}: {e}")
        return pd.DataFrame()

# Beispielverwendung zum Testen
if __name__ == '__main__':
    swahili_samples_train = load_masakhanews_samples('sw', 5, split='train')
    print("\nSwahili Train Samples (n=5):")
    if not swahili_samples_train.empty:
        print(swahili_samples_train.head())
        print(swahili_samples_train['label'].value_counts())

    hausa_samples_test = load_masakhanews_samples('ha', 10, split='test')
    print("\nHausa Test Samples (n=10):")
    if not hausa_samples_test.empty:
        print(hausa_samples_test.head())
        print(hausa_samples_test['label'].value_counts())

    english_samples_dev = load_masakhanews_samples('en', None, split='validation')
    print("\nEnglish Dev Samples (all):")
    if not english_samples_dev.empty:
        print(english_samples_dev.head())
        print(f"Total loaded: {len(english_samples_dev)}")
        print(english_samples_dev['label'].value_counts()) 