import pandas as pd
from datasets import load_dataset
import os

# Sprachcodes 
AFRISENTI_LANG_MAP = {
    "am": "amharic",
    "ar": "arabic",
    "dz": "algerian_arabic",
    "en": "english",
    "fr": "french",
    "ha": "hau",
    "hau": "hau",
    "id": "indonesian",
    "ig": "igbo",
    "kr": "kimbundu",
    "ma": "moroccan_arabic",
    "multi": "multilingual",
    "pcm": "pidgin",
    "pt": "por",
    "por": "por",
    "sw": "swa",
    "swa": "swa",
    "ts": "tsonga",
    "twi": "twi",
    "yo": "yoruba",
    "zh": "chinese",
    "amh": "amh",
    "arq": "arq",
    "ary": "ary",
    "ibo": "ibo",
    "kin": "kin",
    "orm": "orm",
    "tir": "tir",
    "tso": "tso",
}

# Labels
EXPECTED_STRING_LABELS = {'positive', 'negative', 'neutral'}

def load_afrisenti_samples(
    lang_code, 
    num_samples = None,
    split = 'test', 
    dataset_name = "shmuhammad/AfriSenti-twitter-sentiment",
    seed = 42
):
    # Lädt Samples für eine Sprache
    if lang_code not in AFRISENTI_LANG_MAP:
        print(f"Language code '{lang_code}' not found.")
        return pd.DataFrame()

    hf_config_name = AFRISENTI_LANG_MAP[lang_code]
    
    hf_split = split
    if split == 'dev':
        hf_split = 'validation' 
    elif split not in ['train', 'validation', 'test']:
        print(f"Invalid split '{split}'. Using 'test' as default.")
        hf_split = 'test'

    try:
        # Versucht, den Datensatz zu laden.
        dataset = load_dataset(dataset_name, name=hf_config_name, split=hf_split, trust_remote_code=True)
        
        # In DataFrame umwandeln
        samples_df = dataset.to_pandas()
        
        # Spaltennamen standardisieren
        if 'tweet' in samples_df.columns and 'text' not in samples_df.columns:
            samples_df.rename(columns={'tweet': 'text'}, inplace=True)
        
        if 'text' not in samples_df.columns or 'label' not in samples_df.columns:
            raise ValueError("Missing columns 'text' or 'label'")

        # 'id'-Spalte erstellen, falls nicht vorhanden
        if 'id' not in samples_df.columns:
            samples_df['id'] = [f"{hf_config_name}_{hf_split}_{i}" for i in range(len(samples_df))]

        # Samples auswählen, falls num_samples angegeben ist
        if num_samples is not None and 0 < num_samples < len(samples_df):
            samples_df = samples_df.sample(n=num_samples, random_state=seed, replace=False).reset_index(drop=True)
        
        # Labels in Strings umwandeln
        if 'label' in samples_df.columns:
            samples_df['label'] = samples_df['label'].astype(str)
            # Numerische Labels auf String-Labels abbilden
            label_mapping = {
                "0": "negative", 
                "1": "neutral", 
                "2": "positive",
                "negative": "negative",
                "neutral": "neutral",
                "positive": "positive"
            }
            samples_df['label'] = samples_df['label'].str.lower().map(label_mapping).fillna(samples_df['label'])
            
        return samples_df[['id', 'text', 'label']]

    except Exception as e:
        print(f"Error loading AfriSenti for language {lang_code}: {e}")
        return pd.DataFrame()

# Beispielverwendung zum Testen
if __name__ == '__main__':
    hausa_samples = load_afrisenti_samples('ha', 100)
    print("\nHausa Samples (n=100):")
    if not hausa_samples.empty:
        print(hausa_samples.head())
        print(hausa_samples['label'].value_counts())

    swahili_samples = load_afrisenti_samples('sw', 100)
    print("\nSwahili Samples (n=100):")
    if not swahili_samples.empty:
        print(swahili_samples.head())
        print(swahili_samples['label'].value_counts())
    
    amh_samples_all = load_afrisenti_samples('am', None, split='train') 
    print("\nAmharic Samples (all from train):")
    if not amh_samples_all.empty:
        print(amh_samples_all.head())
        print(f"Total loaded: {len(amh_samples_all)}")
        print(amh_samples_all['label'].value_counts()) 