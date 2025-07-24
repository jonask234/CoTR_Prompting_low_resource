# -*- coding: utf-8 -*-
import pandas as pd
from datasets import load_dataset
import os

# Erlaubte Sprachen
ALLOWED_LANGS = ['ha', 'sw']
MASAKHANER_LANGUAGES = ["swa", "amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "wol", "yor"]

def convert_ner_tags_to_entities(tokens, ner_tags, tag_names):
    # Wandelt NER-Tags um
    entities = []
    current_entity_tokens = []
    current_entity_type = None

    for token, tag_idx in zip(tokens, ner_tags):
        tag_name = tag_names[tag_idx]

        if tag_name.startswith('B-'):
            if current_entity_tokens:
                entities.append({"text": " ".join(current_entity_tokens), "type": current_entity_type})
            current_entity_tokens = [token]
            current_entity_type = tag_name[2:]
        elif tag_name.startswith('I-') and current_entity_type == tag_name[2:]:
            current_entity_tokens.append(token)
        else:
            if current_entity_tokens:
                entities.append({"text": " ".join(current_entity_tokens), "type": current_entity_type})
            current_entity_tokens = []
            current_entity_type = None
    
    if current_entity_tokens:
        entities.append({"text": " ".join(current_entity_tokens), "type": current_entity_type})
    
    return entities

def load_masakhaner_samples(
    lang_code, 
    num_samples = None,
    split = 'test', 
    seed = 42
):
    # Lädt NER-Samples für eine bestimmte Sprache 
    dataset_name = "masakhaner"
    
    if lang_code not in MASAKHANER_LANGUAGES:
        print(f"Language code '{lang_code}' is not a valid MasakhaNER language.")
        return pd.DataFrame()

    try:
        # MasakhaNER verwendet den Sprachcode 
        dataset = load_dataset(dataset_name, name=lang_code, split=split, trust_remote_code=True)
        
        all_samples_list = []
        if not hasattr(dataset, 'features') or 'ner_tags' not in dataset.features:
            print("Dataset does not have 'ner_tags' features.")
            return pd.DataFrame()
            
        tag_names = dataset.features['ner_tags'].feature.names
        
        for i, example in enumerate(dataset):
            tokens = example.get('tokens', [])
            ner_tags_int = example.get('ner_tags', [])
            sample_id = example.get('id', str(i))

            if not tokens or not ner_tags_int:
                continue
            
            entities = convert_ner_tags_to_entities(tokens, ner_tags_int, tag_names)
            
            all_samples_list.append({
                'id': sample_id,
                'tokens': tokens,
                'ner_tags': ner_tags_int,
                'tag_names': tag_names,
                'entities': entities
            })
        
        if not all_samples_list:
            return pd.DataFrame()

        all_samples_df = pd.DataFrame(all_samples_list)

        # 'text'-Spalte aus 'tokens' erstellen
        if 'tokens' in all_samples_df.columns:
            all_samples_df['text'] = all_samples_df['tokens'].apply(lambda t: " ".join(t) if isinstance(t, list) else "")
        else:
            all_samples_df['text'] = ""

        # Samples mischen und auswählen
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
        print(f"An error occurred: {e}")
        return pd.DataFrame()

# Beispielverwendung zum Testen
if __name__ == '__main__':
    ha_samples = load_masakhaner_samples('hau', num_samples=10)
    if not ha_samples.empty:
        print(f"Hausa test samples: {len(ha_samples)}")
        print(ha_samples.head())
        if 'entities' in ha_samples.columns and len(ha_samples.iloc[0]['entities']) > 0:
            print(f"Entities of first HA sample: {ha_samples.iloc[0]['entities']}")

    sw_samples = load_masakhaner_samples('swa', num_samples=5)
    if not sw_samples.empty:
        print(f"Swahili test samples: {len(sw_samples)}")
        print(sw_samples.head())
        if 'entities' in sw_samples.columns and len(sw_samples.iloc[0]['entities']) > 0:
             print(f"Entities of first SW sample: {sw_samples.iloc[0]['entities']}")

    xx_samples = load_masakhaner_samples('xx')
    if xx_samples.empty:
        print("Correctly returned empty DataFrame for unsupported language 'xx'.")
        
    ha_samples_train_split = load_masakhaner_samples('hau', split='train')
    if not ha_samples_train_split.empty:
        print(f"Hausa samples with 'train' split request: {len(ha_samples_train_split)}") 