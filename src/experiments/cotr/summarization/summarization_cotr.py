from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any, List
import pandas as pd
import time
from tqdm import tqdm
import re

# Define language names dictionary at module level (reused from qa_cotr.py)
lang_names = {
    "sw": "Swahili",
    "te": "Telugu",
    "en": "English",
    "am": "Amharic",
    "ha": "Hausa"
}

def initialize_model(model_name: str) -> tuple:
    """Initialize the model and tokenizer."""
    print(f"Loading model {model_name}...")
    cache_path = "/work/bbd6522/cache_dir"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_path
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_path
    )
    return model, tokenizer

def generate_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """Generate a prompt for translation with structured format."""
    # Special handling for English to English "translation" (no-op)
    if source_lang == 'en' and target_lang == 'en':
        return text
    
    # Get language names
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)
    
    # Enhanced specialized prompts for LRLs → English
    if target_lang == 'en':
        if source_lang in ['sw', 'te', 'am']:
            return f"""Text: '{text}'

Translate this {source_name} text to English.
Preserve the exact meaning without adding or removing information.
Provide only the direct translation without explanations.

Translation:"""
    
    # Enhanced specialized prompts for English → LRLs
    if source_lang == 'en':
        if target_lang in ['sw', 'te', 'am']:
            return f"""Text: '{text}'

Translate this English text to {target_name}.
Preserve the exact meaning without adding or removing information.
Provide only the direct translation without explanations.

Translation:"""
    
    # Default/original prompt for other language pairs - more structured
    return f"""Text: '{text}'

Translate this {source_name} text to {target_name}.
Preserve the exact meaning and important details.
Provide only the direct translation without explanations.

Translation:"""

def generate_summarization_prompt(text: str) -> str:
    """Generate a prompt for summarization in English with structured format."""
    # Truncate long articles to 8000 chars to avoid token limits
    if len(text) > 8000:
        text = text[:8000] + "..."
        
    return f"""Text: '{text}'

Summarize the above text in 2-3 sentences. Capture the main points only.
Provide your summary in a direct, concise format without additional explanation.

Summary:"""

def clean_translation_response(response: str, target_lang: str, source_lang: str) -> str:
    """Clean up translation response to extract the actual translation."""
    translation = response.strip()
    
    # Remove potential prompt remnants
    target_name = lang_names.get(target_lang, target_lang)
    translation_prompt_start = f"{target_name} translation:"
    translation = re.sub(f"^{re.escape(translation_prompt_start)}", '', translation, flags=re.IGNORECASE).strip()
    
    # Remove artifacts
    translation = re.sub(r'^answer:', '', translation, flags=re.IGNORECASE).strip()
    translation = re.sub(r'^summary:', '', translation, flags=re.IGNORECASE).strip()
    
    # Remove empty square/round brackets
    translation = re.sub(r'\[\]', '', translation).strip()
    translation = re.sub(r'\(\)', '', translation).strip()
    
    # Handle potential empty answers
    if not translation:
        translation = "[No translation generated]"
    
    return translation

def clean_summary_response(response: str) -> str:
    """Clean up summary response to extract the actual summary."""
    summary = response.strip()
    
    # Remove "Summary:" prefix if present
    if summary.lower().startswith("summary:"):
        summary = summary[8:].strip()
    
    # Remove bullet points if present
    summary = re.sub(r'^\s*[\-\*•]\s+', '', summary)
    
    # Remove numbered points if present
    summary = re.sub(r'^\s*\d+[\.\)]\s+', '', summary)
    
    # Remove quotes if they wrap the entire output
    if (summary.startswith('"') and summary.endswith('"')) or \
       (summary.startswith("'") and summary.endswith("'")):
        summary = summary[1:-1].strip()
    
    # Handle potential empty answers
    if not summary:
        summary = "[No summary generated]"
    
    return summary

def translate_text(
    model: Any,
    tokenizer: Any,
    text: str,
    source_lang: str,
    target_lang: str,
    max_input_length: int = 4096,
    max_new_tokens: int = 512
) -> str:
    """Translate text from source language to target language using the model."""
    # Special handling for English to English "translation" (no-op)
    if source_lang == 'en' and target_lang == 'en':
        return text
    
    # Generate translation prompt
    prompt = generate_translation_prompt(text, source_lang, target_lang)
    
    # Get the device of the model
    device = next(model.parameters()).device
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Clean up the response
    translation = clean_translation_response(response, target_lang, source_lang)
    
    return translation

def generate_summary(
    model: Any,
    tokenizer: Any,
    text: str,
    max_input_length: int = 4096,
    max_new_tokens: int = 100
) -> str:
    """Generate a summary for the given text in English."""
    # Generate summarization prompt
    prompt = generate_summarization_prompt(text)
    
    # Get the device of the model
    device = next(model.parameters()).device
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Determine if this is the Aya model for specialized processing
    is_aya_model = "aya" in model.config._name_or_path.lower()
    
    # Generate summary
    with torch.no_grad():
        if is_aya_model:
            # Aya tends to respond better with beam search for structured outputs
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            # For other models, use sampling with temperature
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
    
    # Decode only the newly generated tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Clean up the response
    summary = clean_summary_response(response)
    
    return summary

def evaluate_summarization_cotr(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str
) -> pd.DataFrame:
    """
    Evaluate summarization using Chain of Translation Reasoning (CoTR) approach.
    
    Steps:
    1. Translate article from source language to English
    2. Generate summary in English
    3. Translate summary back to the source language
    
    Args:
        model_name: Name of the model to use
        samples_df: DataFrame containing the samples (text and summary columns)
        lang_code: Language code
        
    Returns:
        DataFrame with results including all translations and summaries
    """
    try:
        model, tokenizer = initialize_model(model_name)
    except Exception as e:
        print(f"ERROR: Failed to initialize model {model_name}: {e}")
        return pd.DataFrame()
        
    results = []
    
    # Determine safe input length
    model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
    safe_max_input_length = min(model_max_len, 8192) 
    print(f"Using max_input_length: {safe_max_input_length}")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), 
                         desc=f"Processing {lang_code} samples ({model_name} CoTR)"):
        try:
            original_article = row['text']
            reference_summary = row['summary']
            
            # Truncate very long articles
            if len(original_article) > 8000:
                original_article = original_article[:8000]
            
            # Step 1: Translate article to English (skip if already English)
            if lang_code == 'en':
                article_en = original_article
            else:
                article_en = translate_text(
                    model, tokenizer, original_article, lang_code, "en", 
                    max_input_length=safe_max_input_length
                )
            
            # Step 2: Generate summary in English
            summary_en = generate_summary(
                model, tokenizer, article_en, 
                max_input_length=safe_max_input_length,
                max_new_tokens=100
            )
            
            # Step 3: Translate summary back to original language (skip if already English)
            if lang_code == 'en':
                summary_lrl = summary_en
            else:
                summary_lrl = translate_text(
                    model, tokenizer, summary_en, "en", lang_code, 
                    max_input_length=safe_max_input_length
                )
            
            # Store results
            result = {
                'article': original_article[:500] + "..." if len(original_article) > 500 else original_article,
                'article_en': article_en[:500] + "..." if len(article_en) > 500 else article_en,
                'reference_summary': reference_summary,
                'summary_en': summary_en,
                'predicted_summary': summary_lrl,
                'language': lang_code
            }
            
            # Add the row ID if available
            if "id" in row:
                result["id"] = row["id"]
                
            results.append(result)

        except Exception as e:
            print(f"ERROR processing sample {idx}: {e}")
            continue

    if not results:
        print(f"WARNING: No results were successfully processed for {lang_code} with {model_name}.")
        return pd.DataFrame()
        
    return pd.DataFrame(results) 