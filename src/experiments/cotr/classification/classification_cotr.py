import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os
from typing import Any, List

# Define expected labels (can be dynamically determined from data later)
# EXPECTED_LABELS = ["positive", "negative", "neutral"]

def initialize_model(model_name: str) -> tuple:
    """Initialize the model and tokenizer, specifying cache directory."""
    print(f"Loading model {model_name}...")
    cache_path = "/work/bbd6522/cache_dir"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_path,
        trust_remote_code=True # Needed for some models like Qwen
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_path,
        trust_remote_code=True # Needed for some models like Qwen
    )
    return model, tokenizer

def generate_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """Generate a prompt for translation."""
    # Simple prompt, assumes model understands language codes
    # Add language-specific prompts if needed (like for Hausa)
    if source_lang == 'ha' and target_lang == 'en':
        return f"Translate Hausa to English:\n\nHausa: {text}\n\nEnglish:"
    elif source_lang == 'sw' and target_lang == 'en':
         return f"Translate Swahili to English:\n\nSwahili: {text}\n\nEnglish:"
    else: # Generic fallback
        return f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}\n\n{target_lang} translation:"

def generate_classification_prompt(text: str, labels: List[str]) -> str:
    """Generate a zero-shot prompt for English text classification."""
    label_string = ", ".join(labels)
    prompt = f"""Classify the following English text into one of these categories: {label_string}.
Respond with only the category name.

Text: {text}

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
    prompt = generate_translation_prompt(text, source_lang, target_lang)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7, # Allow variation for translation
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    translation = response.strip()
    if not translation:
        translation = "[No translation generated]"
    return translation

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
        if label == predicted_label or predicted_label.startswith(label):
            final_label = label
            break
            
    # Fallback: Check if the raw output contained the label name clearly
    if final_label == "[Unknown]":
         for label in possible_labels:
             if label in predicted_label_raw:
                 final_label = label
                 break
                 
    return final_label

def evaluate_classification_cotr(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str,
) -> pd.DataFrame:
    """
    Evaluate text classification using Chain of Translation Prompting (CoTR).
    Steps:
    1. Translate LRL text to English
    2. Perform classification in English
    3. Translate result back to original language (optional)
    Assumes samples_df has 'text' and 'label' columns.
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
            
            # Step 3: Translate the classification result back to the original language
            # Only translate back if we have a valid prediction (not [Unknown])
            if predicted_label_en != "[Unknown]":
                # Create a prompt to translate the predicted label
                label_translation_prompt = f"Translate the word '{predicted_label_en}' from English to {lang_code}."
                
                # Use the model to translate the label
                inputs = tokenizer(label_translation_prompt, return_tensors="pt", truncation=True, max_length=safe_max_input_length)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,  # Short output expected
                        temperature=0.3,    # Less variation for label translation
                        do_sample=False,    # Deterministic for consistency
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                predicted_label_lrl = response.strip()
                
                # Clean up the translated label - extract just the translated word
                # This handles cases where model might output extra text
                words = predicted_label_lrl.split()
                if words:
                    predicted_label_lrl = words[0].strip(',.?!;:"\'')
                else:
                    predicted_label_lrl = predicted_label_en  # Fallback to English if translation failed
            else:
                predicted_label_lrl = "[Unknown]"  # Keep unknown as is

            # Store results
            result = {
                'id': row.get('id', idx),
                'original_text': original_text,         # Original text in LRL
                'text_en': text_en,                     # Translated text in English
                'ground_truth_label': ground_truth_label,
                'predicted_label_en': predicted_label_en,  # Prediction in English
                'predicted_label_lrl': predicted_label_lrl,  # Prediction translated back to LRL
                'predicted_label': predicted_label_en,  # Keep English as the main prediction for evaluation
                'language': lang_code
            }
            results.append(result)

        except Exception as e:
            print(f"ERROR processing sample {row.get('id', idx)}: {e}")
            continue

    if not results:
        print(f"WARNING: No results successfully processed for {lang_code} with {model_name} using CoTR.")
        return pd.DataFrame()

    return pd.DataFrame(results) 