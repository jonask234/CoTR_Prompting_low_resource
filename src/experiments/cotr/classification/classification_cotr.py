import torch
import pandas as pd
from tqdm import tqdm
import os
import sys

# Add project root to path to find model_initialization
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from model_initialization import initialize_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, List

# Define expected labels (can be dynamically determined from data later)
# EXPECTED_LABELS = ["positive", "negative", "neutral"]

def generate_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """Generate a prompt for translation with structured format."""
    # Special handling for English to English "translation" (no-op)
    if source_lang == 'en' and target_lang == 'en':
        return text  # Simply return the original text for English->English

    # Use full language names for better prompting
    lang_names = {
        "en": "English",
        "swa": "Swahili",
        "hau": "Hausa"
    }
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)
    
    return f"""Text: '{text}'

Translate this {source_name} text to {target_name}.
Preserve the exact meaning without adding or removing information.
Provide only the direct translation without explanations.

Translation:"""

def generate_classification_prompt(text: str, labels: List[str]) -> str:
    """Generate a prompt for classification with structured format."""
    label_string = ", ".join(labels)
    prompt = f"""Text: '{text}'

Classify the above text into one of these categories: {label_string}.
Respond with only the category name in English.

Category:"""
    return prompt

def translate_text(
    model: Any,
    tokenizer: Any,
    text: str,
    source_lang: str,
    target_lang: str,
    max_input_length: int = 1024, # Allow longer inputs for news
    max_new_tokens: int = 512   # Allow longer translations if needed
) -> str:
    """Translate text from source language to target language using the model."""
    # Special handling for English to English "translation" (no-op)
    if source_lang == 'en' and target_lang == 'en':
        return text  # No translation needed

    prompt = generate_translation_prompt(text, source_lang, target_lang)
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, # Enable sampling for more natural output
            temperature=0.7, # Less deterministic  
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Basic cleanup
    translation = response.strip()
    
    # Handle potential empty output
    if not translation:
        return "[Translation failed]"
    
    return translation

def get_label_translation(label: str, target_lang: str) -> str:
    """
    Get a direct mapping of English labels to target language.
    This is more reliable than generating translations on the fly.
    """
    # Hard-coded translations for common news categories
    translations = {
        "hau": {  # Hausa
            "politics": "siyasa",
            "sports": "wasanni",
            "entertainment": "nishadantarwa",
            "technology": "fasaha",
            "health": "lafiya",
            "business": "kasuwanci",
            "religion": "addini"
        },
        "swa": {  # Swahili
            "politics": "siasa",
            "sports": "michezo",
            "entertainment": "burudani",
            "technology": "teknolojia",
            "health": "afya",
            "business": "biashara",
            "religion": "dini"
        }
    }
    
    # Check if we have a mapping for this language and label
    if target_lang in translations and label.lower() in translations[target_lang]:
        return translations[target_lang][label.lower()]
    
    # If no mapping exists, return the English label with a note
    return f"{label}"  # Keep the English label

def process_classification_english(
    model: Any,
    tokenizer: Any,
    text_en: str,
    possible_labels: List[str], # Pass possible labels
    max_input_length: int = 1024,
    max_new_tokens: int = 10 # Labels are short
) -> str:
    """Process English text for classification."""
    prompt = generate_classification_prompt(text_en, possible_labels)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Post-process the output to extract the label
    predicted_label_raw = response.strip().lower()
    predicted_label_lines = predicted_label_raw.split('\n')
    predicted_label = predicted_label_lines[0].strip()

    # Extract label
    final_label = "[Unknown]"
    for label in possible_labels:
        # Check for exact match or if prediction starts with the label
        if label.lower() == predicted_label or predicted_label.startswith(label.lower()):
            final_label = label
            break
            
    # Fallback: Check if the raw output contained the label name clearly
    if final_label == "[Unknown]":
         for label in possible_labels:
             if label.lower() in predicted_label_raw:
                 final_label = label
                 break
                 
    return final_label

def evaluate_classification_cotr(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str,
    use_lrl_ground_truth: bool = False
) -> pd.DataFrame:
    """
    Evaluate text classification using Chain of Translation Prompting (CoTR).
    Steps:
    1. Translate LRL text to English
    2. Perform classification in English
    3. Translate result back to original language
    
    Args:
        model_name: Model to use
        samples_df: DataFrame with samples
        lang_code: Language code
        use_lrl_ground_truth: If True, evaluation is LRL-to-LRL (ground truth in LRL)
    """
    try:
        model, tokenizer = initialize_model(model_name)
    except Exception as e:
        print(f"ERROR: Failed to initialize model {model_name}: {e}")
        return pd.DataFrame()

    results = []
    
    # Determine possible labels from the dataset itself
    possible_labels = sorted(list(samples_df['label'].unique()))
    print(f"Using labels found in dataset for prompts: {possible_labels}")
    
    model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
    safe_max_input_length = min(model_max_len, 1024)
    print(f"Using max_input_length: {safe_max_input_length}")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} CoTR classification ({model_name})"):
        try:
            original_text = row['text']
            ground_truth_label = row['label']

            # Step 1: Translate text to English
            text_en = translate_text(model, tokenizer, original_text, lang_code, "en", max_input_length=safe_max_input_length)

            # Step 2: Process classification in English
            predicted_label_en = process_classification_english(
                model, tokenizer, text_en, 
                possible_labels=possible_labels, 
                max_input_length=safe_max_input_length
            )
            
            # Step 3: Use a mapping-based approach for label translation
            if predicted_label_en != "[Unknown]":
                # Use direct mapping instead of translation generation
                predicted_label_lrl = get_label_translation(predicted_label_en, lang_code)
            else:
                predicted_label_lrl = "[Unknown]"

            # Determine which predicted label to use for evaluation
            # For LRL ground truth, use the back-translated LRL label
            # For English ground truth, use the English prediction
            predicted_label = predicted_label_lrl if use_lrl_ground_truth else predicted_label_en

            # Store results
            result = {
                'id': row.get('id', idx),
                'original_text': original_text,         # Original text in LRL
                'text_en': text_en,                     # Translated text in English
                'ground_truth_label': ground_truth_label,
                'predicted_label_en': predicted_label_en,  # Prediction in English
                'predicted_label_lrl': predicted_label_lrl,  # Prediction translated back to LRL
                'predicted_label': predicted_label,     # Label to use for evaluation (changes based on use_lrl_ground_truth)
                'language': lang_code,
                'lrl_evaluation': use_lrl_ground_truth  # Flag for analysis
            }
            results.append(result)

        except Exception as e:
            print(f"ERROR processing sample {row.get('id', idx)}: {e}")
            continue

    if not results:
        print(f"WARNING: No results successfully processed for {lang_code} with {model_name} using CoTR.")
        return pd.DataFrame()

    return pd.DataFrame(results) 