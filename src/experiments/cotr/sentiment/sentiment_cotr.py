import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os
from typing import Any

# Define expected labels (should match baseline)
EXPECTED_LABELS = ["positive", "negative", "neutral"]

def initialize_model(model_name: str) -> tuple:
    """Initialize the model and tokenizer, specifying cache directory."""
    print(f"Loading model {model_name}...")
    cache_path = "/work/bbd6522/cache_dir"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_path,
        trust_remote_code=True # Qwen might need this
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_path,
        trust_remote_code=True # Qwen might need this
    )
    return model, tokenizer

def generate_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """Generate a prompt for translation with structured format."""
    # Simple prompt, assumes model understands language codes
    # More direct prompt specifically for Hausa -> English case
    if source_lang == 'ha' and target_lang == 'en':
        return f"""Text: '{text}'

Translate this Hausa text to English.
Preserve the exact meaning without adding or removing information.
Provide only the direct translation without explanations.

Translation:"""
    else: # Improved structured format for other language pairs
        # Get full language names for better prompting
        lang_names = {
            "en": "English",
            "ha": "Hausa",
            "sw": "Swahili"
        }
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        
        return f"""Text: '{text}'

Translate this {source_name} text to {target_name}.
Preserve the exact meaning without adding or removing information.
Provide only the direct translation without explanations.

Translation:"""

def generate_sentiment_prompt(text: str) -> str:
    """Generate a zero-shot prompt for English sentiment classification with structured format."""
    prompt = f"""Text: '{text}'

Analyze the sentiment of the text above.
Respond with only one of these labels: positive, negative, or neutral.

Sentiment:"""
    return prompt

def translate_text(
    model: Any,
    tokenizer: Any,
    text: str,
    source_lang: str,
    target_lang: str,
    max_input_length: int = 1024, # Adjust as needed
    max_new_tokens: int = 256   # Allow longer translations if needed
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

def process_sentiment_english(
    model: Any,
    tokenizer: Any,
    text_en: str,
    max_input_length: int = 1024,
    max_new_tokens: int = 5 # Labels are short
) -> str:
    """Process English text for sentiment classification."""
    prompt = generate_sentiment_prompt(text_en)
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
    predicted_label = response.strip().lower()

    # Extract label
    final_label = "[Unknown]"
    for label in EXPECTED_LABELS:
        if label in predicted_label:
            final_label = label
            break
    return final_label

def evaluate_sentiment_cotr(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str,
) -> pd.DataFrame:
    """
    Evaluate sentiment analysis using Chain of Translation Prompting (CoTR).
    Steps:
    1. Translate LRL text to English
    2. Perform sentiment classification in English
    3. Translate result back to original language (added for complete pipeline)
    """
    try:
        model, tokenizer = initialize_model(model_name)
    except Exception as e:
        print(f"ERROR: Failed to initialize model {model_name}: {e}")
        return pd.DataFrame()

    results = []
    model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
    safe_max_input_length = min(model_max_len, 1024)
    print(f"Using max_input_length: {safe_max_input_length}")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} CoTR sentiment ({model_name})"):
        try:
            original_text = row['text']
            ground_truth_label = row['label']

            # Step 1: Translate text to English
            text_en = translate_text(model, tokenizer, original_text, lang_code, "en", max_input_length=safe_max_input_length)

            # Step 2: Process sentiment in English
            predicted_label_en = process_sentiment_english(model, tokenizer, text_en, max_input_length=safe_max_input_length)
            
            # Step 3: Translate the sentiment result back to the original language
            # Only translate back if we have a valid prediction (not [Unknown])
            if predicted_label_en != "[Unknown]":
                # Create a prompt to translate the predicted label with structured format
                label_translation_prompt = f"""Text: '{predicted_label_en}'

Translate this English sentiment label to {lang_code}.
Provide only the direct translation without explanations.

Translation:"""
                
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
                'original_text': original_text,
                'text_en': text_en,
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