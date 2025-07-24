# -*- coding: utf-8 -*-
import pandas as pd
from datasets import load_dataset

# Sprachcodes für TyDiQA
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

def load_tydiqa_samples(
    lang_code,
    num_samples = None,
    split = 'validation',
    seed = 42
):
    # Lädt Samples 
    hf_dataset_name = "khalidalt/tydiqa-goldp"
    hf_lang_config = TYDIQA_LANG_CONFIG_MAP.get(lang_code)

    if not hf_lang_config:
        print(f"Language code '{lang_code}' not found. Aborting.")
        return pd.DataFrame()

    if split == 'test':
        split = 'validation'

    try:
        dataset = load_dataset(hf_dataset_name, name=hf_lang_config, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading TyDiQA for {lang_code}. Error: {e}.")
        return pd.DataFrame()

    all_samples = []
    for i, example in enumerate(dataset):
        question = example.get('question_text') or example.get('question', '')
        context = example.get('passage_text') or example.get('context', '')
        annotations_list = example.get('annotations', [])
        answers_list = example.get('answers', [])
        
        answer_texts = []
        
        if isinstance(annotations_list, list) and annotations_list:
            for annotation in annotations_list:
                if isinstance(annotation, dict):
                    answer_text = annotation.get('answer_text')
                    if answer_text and isinstance(answer_text, str):
                        answer_texts.append(answer_text)
        elif isinstance(answers_list, dict) and 'text' in answers_list:
            if isinstance(answers_list['text'], list):
                 answer_texts.extend([str(t) for t in answers_list['text'] if t])

        if not question or not context or not answer_texts:
            continue
        
        ground_truth = answer_texts[0]

        all_samples.append({
            'id': example.get('example_id', f'{lang_code}-{split}-{i}'),
            'question': question,
            'context': context,
            'answers': ground_truth,
            'all_answers': answer_texts,
            'language': lang_code
        })

    if not all_samples:
        return pd.DataFrame()

    all_samples_df = pd.DataFrame(all_samples)

    if num_samples is not None and num_samples > 0:
        if num_samples < len(all_samples_df):
            all_samples_df = all_samples_df.sample(n=num_samples, random_state=seed).reset_index(drop=True)
    
    return all_samples_df

# Beispielverwendung zum Testen
if __name__ == '__main__':
    print("--- Testing TyDiQA GoldP Loader ---")

    en_samples = load_tydiqa_samples('en', num_samples=1, split='validation')
    if not en_samples.empty:
        print(f"\\nEnglish validation samples: {len(en_samples)}")
        print(en_samples.head())
    else:
        print("\\nFailed to load English samples.")

    sw_samples = load_tydiqa_samples('sw', num_samples=5, split='train')
    if not sw_samples.empty:
        print(f"\\nSwahili train samples: {len(sw_samples)}")
        print(sw_samples.head())
    else:
        print("\\nFailed to load Swahili samples.")

    fi_samples = load_tydiqa_samples('fi', num_samples=2, split='validation')
    if not fi_samples.empty:
        print(f"\\nFinnish validation samples: {len(fi_samples)}")
        print(fi_samples.head())
    else:
        print("\\nFailed to load Finnish samples.") 