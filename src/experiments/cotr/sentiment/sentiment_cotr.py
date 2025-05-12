import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os
import re
from typing import Any, Dict, List, Tuple

# Define expected labels (should match baseline)
EXPECTED_LABELS = ["positive", "negative", "neutral"]
LANG_NAMES = {
    "en": "English",
    "sw": "Swahili",
    "ha": "Hausa",
    "te": "Telugu" 
    # Add other languages if your dataset includes them
}

def initialize_model(model_name: str) -> tuple:
    """Initialize the model and tokenizer."""
    print(f"Loading model {model_name}...")
    cache_path = "/work/bbd6522/cache_dir"  # Define cache path

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_path
    )
    
    # Robustly set pad_token and model.config.pad_token_id
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")
        else:
            # If no eos_token, add a new pad token
            print("No pad_token or eos_token found. Adding a new [PAD] token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Important: Resize model embeddings if a new token is added
            model.resize_token_embeddings(len(tokenizer))
            print("Resized model token embeddings for the new [PAD] token.")

    # Ensure model.config.pad_token_id is also aligned with the tokenizer
    if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Aligned model.config.pad_token_id with tokenizer.pad_token_id ({tokenizer.pad_token_id})")
            
    return tokenizer, model

def generate_translation_prompt(text: str, source_lang: str, target_lang: str, is_label: bool = False) -> str:
    """Generate a prompt for translation with structured format."""
    source_name = LANG_NAMES.get(source_lang, source_lang)
    target_name = LANG_NAMES.get(target_lang, target_lang)

    if is_label:
        # Specific prompt for translating a sentiment label
        return f"""Translate the following English sentiment label to {target_name}: '{text}'
Provide ONLY the single translated word for the sentiment label in {target_name}.
{target_name} Translation:"""
    else:
        # General text translation prompt
        return f"""Translate the following {source_name} text to {target_name}.
Preserve the original meaning and sentiment.
Provide only the direct translation without any explanations or introductory phrases.
Original Text ({source_name}): '{text}'
{target_name} Translation:"""

def generate_sentiment_prompt_english(text_en: str, use_few_shot: bool = True) -> str:
    """Generate a prompt for English sentiment classification."""
    # Instructions are always in English for this step
    base_instruction = "Analyze the sentiment of the English text below.\\nRespond with only one of these labels: positive, negative, or neutral."
    
    prompt_parts = [f"Text: '{text_en}'\\n", base_instruction]

    if use_few_shot:
        examples_en = """
Examples:
Text: 'This movie was fantastic, I loved it!'
Sentiment: positive

Text: 'I am not happy with the service provided.'
Sentiment: negative

Text: 'The meeting is scheduled for 3 PM.'
Sentiment: neutral

Text: 'I'm really happy with the service! 😊' 
Sentiment: positive

Text: 'I was very disappointed with this product.'
Sentiment: negative

Text: 'This experience has been terrible and I lost money. 😢'
Sentiment: negative"""
        prompt_parts.append("\\n" + examples_en)
            
    prompt_parts.append("\\nSentiment:")
    return "\\n".join(prompt_parts)

def extract_sentiment_label_cotr(output_text: str) -> str:
    """Extracts sentiment label from model output, prioritizing exact matches."""
    text_lower = output_text.lower().strip()
    
    # Remove common prefixes models might add
    common_prefixes = [
        "sentiment:", "the sentiment is", "this text is", "i think the sentiment is", 
        "hisia:", "ra'ayi:", "jibu:", "answer:", "label:"
        "translation:", "tafsiri:", "fassara:" # For translated labels
    ]
    for prefix in common_prefixes:
        if text_lower.startswith(prefix):
            text_lower = text_lower[len(prefix):].strip()

    # Remove surrounding quotes or brackets
    text_lower = text_lower.strip('\'\"()[]')

    # Check for exact labels first (case-insensitive)
    for label in EXPECTED_LABELS:
        if text_lower == label:
            return label
    
    # Check if the cleaned text starts with one of the expected labels
    for label in EXPECTED_LABELS:
        if text_lower.startswith(label): # e.g., "positive."
            return label
        
    # Fallback for very short responses containing the label or keywords
    # This helps if the model just outputs the label or a close variant
    if len(text_lower.split()) < 3: # Increased from 2 to 3
        if "positive" in text_lower or "nzuri" in text_lower or "da kyau" in text_lower : # Added LRL keywords
            return "positive"
        if "negative" in text_lower or "mbaya" in text_lower or "mara kyau" in text_lower: # Added LRL keywords
            return "negative"
        if "neutral" in text_lower or "kati" in text_lower or "tsaka-tsaki" in text_lower: # Added LRL keywords
            return "neutral"
                
    # Default if no clear label found
    return "neutral" # Default to neutral as it's often safer

def translate_text(
    model: Any,
    tokenizer: Any,
    text: str,
    source_lang: str,
    target_lang: str,
    is_label: bool = False,
    max_input_length: int = 1024,
    max_new_tokens: int = 50, 
    temperature: float = 0.5,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.05,
    do_sample: bool = True,
    pad_token_id: int = None
) -> str:
    """Translate text or a label from source language to target language."""
    if not text or text.strip() == "":
        return "[Empty input to translate]"

    prompt = generate_translation_prompt(text, source_lang, target_lang, is_label=is_label)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # HACK: Temporarily attach model.config to tokenizer for problematic models
    original_tokenizer_config = getattr(tokenizer, 'config', None)
    tokenizer.config = model.config

    try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=pad_token_id
            )
    finally:
        # Restore original tokenizer config or remove the temporary one
        if original_tokenizer_config is not None:
            tokenizer.config = original_tokenizer_config
        elif hasattr(tokenizer, 'config'): # if it was added and wasn't there before
            del tokenizer.config

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    # Use extract_sentiment_label_cotr for labels, otherwise just strip.
    translation = extract_sentiment_label_cotr(response) if is_label and target_lang != "en" else response.strip()

    if not translation and is_label: 
        translation = text 
    elif not translation:
        translation = "[No translation generated]"
    return translation

def process_sentiment_english(
    model: Any,
    tokenizer: Any,
    text_en: str,
    use_few_shot: bool,
    generation_params: Dict = None
) -> str:
    """Process English text for sentiment classification."""
    prompt = generate_sentiment_prompt_english(text_en, use_few_shot=use_few_shot)
    
    default_gen_params = {
        "temperature": 0.2, "top_p": 0.9, "top_k": 30, 
        "max_new_tokens": 10, # Labels are short
        "repetition_penalty": 1.1, "do_sample": True
    }
    current_gen_params = {**default_gen_params, **(generation_params or {})}

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=current_gen_params.get("max_input_length", 2048))
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # HACK: Temporarily attach model.config to tokenizer for problematic models
    original_tokenizer_config = getattr(tokenizer, 'config', None)
    tokenizer.config = model.config

    try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
                max_new_tokens=current_gen_params["max_new_tokens"],
                temperature=current_gen_params["temperature"],
                top_p=current_gen_params["top_p"],
                top_k=current_gen_params["top_k"],
                repetition_penalty=current_gen_params["repetition_penalty"],
                do_sample=current_gen_params["do_sample"],
                pad_token_id=current_gen_params.get("pad_token_id", tokenizer.pad_token_id)
            )
    finally:
        # Restore original tokenizer config or remove the temporary one
        if original_tokenizer_config is not None:
            tokenizer.config = original_tokenizer_config
        elif hasattr(tokenizer, 'config'): # if it was added and wasn't there before
            del tokenizer.config

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return extract_sentiment_label_cotr(response)

def evaluate_sentiment_cotr_multi_prompt(
    model: Any, # Pass initialized model
    tokenizer: Any, # Pass initialized tokenizer
    samples_df: pd.DataFrame,
    lang_code: str,
    use_few_shot: bool,
    text_translation_params: Dict,
    sentiment_classification_params: Dict,
    label_translation_params: Dict
) -> pd.DataFrame:
    """Evaluate sentiment using Multi-Prompt CoTR."""
    results = []
    
    # Determine safe_max_input_length once
    model_max_len = getattr(model.config, 'max_position_embeddings', 2048) # Default if not found
    safe_max_input_length = min(model_max_len, 4096) # Cap at a reasonable value

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Multi-Prompt CoTR {lang_code}"):
            original_text = row['text']
            ground_truth_label = row['label']

        # 1. Translate LRL text to English
        # Using minimal, known-safe parameters for debugging
        text_en = translate_text(
            model, tokenizer, original_text, lang_code, "en",
            is_label=False,
            max_input_length=safe_max_input_length,
            max_new_tokens=200, # Fixed value for debugging
            temperature=0.7,    # Fixed value for debugging
            do_sample=True,     # Fixed value for debugging
            pad_token_id=tokenizer.pad_token_id # Explicitly set pad_token_id
            # Removed top_p, top_k, repetition_penalty for now
        )

        # 2. Classify sentiment of English text
        # Using minimal, known-safe parameters for debugging
        predicted_label_en = process_sentiment_english(
            model, tokenizer, text_en, use_few_shot,
            generation_params={ # Minimal set for debugging
                "max_new_tokens": 10,
                "temperature": 0.2,
                "do_sample": False, # Often safer for classification
                "pad_token_id": tokenizer.pad_token_id # Explicitly set pad_token_id
            }
        )
            
        # 3. Translate English sentiment label back to LRL
        predicted_label_lrl = predicted_label_en 
        if predicted_label_en in EXPECTED_LABELS:
            # Using minimal, known-safe parameters for debugging
            predicted_label_lrl = translate_text(
                model, tokenizer, predicted_label_en, "en", lang_code,
                is_label=True,
                max_input_length=safe_max_input_length, 
                max_new_tokens=10,   # Fixed value for debugging
                temperature=0.7,     # Fixed value for debugging
                do_sample=True,      # Fixed value for debugging
                pad_token_id=tokenizer.pad_token_id # Explicitly set pad_token_id
                # Removed top_p, top_k, repetition_penalty for now
            )
        else: 
            predicted_label_lrl = predicted_label_en if predicted_label_en != "neutral" else "[Unknown]"

        results.append({
            'id': row.get('id', idx),
            'original_text': original_text,
            'text_en': text_en,
            'ground_truth_label': ground_truth_label,
            'predicted_label_en': predicted_label_en,
            'predicted_label_lrl': predicted_label_lrl,
            'final_predicted_label': predicted_label_en,
            'language': lang_code,
            'pipeline': 'multi_prompt',
            'shot_type': 'few-shot' if use_few_shot else 'zero-shot'
        })
    return pd.DataFrame(results)

def generate_single_prompt_sentiment_cotr(lrl_text: str, lang_code: str, use_few_shot: bool = True) -> str:
    """Generate a single prompt for the entire CoTR Sentiment pipeline."""
    lrl_name = LANG_NAMES.get(lang_code, lang_code)
    
    instructions = f"""Text ({lrl_name}): '{lrl_text}'

Your task is to determine the sentiment of this {lrl_name} text.
Follow these steps carefully:
1. Translate the {lrl_name} text into English.
2. Analyze the sentiment of the translated English text (positive, negative, or neutral).
3. Translate the English sentiment label back into {lrl_name}.
4. Provide ONLY the final {lrl_name} sentiment label as your response. Do not include any intermediate steps.

Example of final output format if the text was Swahili and sentiment was positive:
Jibu: chanya""" # Model should output the LRL label.

    examples = ""
    if use_few_shot:
                if lang_code == 'sw':
            examples = f"""
Examples:
Text (Swahili): 'Kitabu hiki ni kizuri sana!'
English Translation: This book is very good!
English Sentiment: positive
Swahili Sentiment: chanya
Final Answer (Swahili): chanya

Text (Swahili): 'Nimechukizwa na huduma hii.'
English Translation: I am disgusted with this service.
English Sentiment: negative
Swahili Sentiment: hasi
Final Answer (Swahili): hasi"""
        elif lang_code == 'ha':
            examples = f"""
Examples:
Text (Hausa): 'Wannan fim din ya yi kyau kwarai!'
English Translation: This film was very good!
English Sentiment: positive
Hausa Sentiment: tabbatacce
Final Answer (Hausa): tabbatacce

Text (Hausa): 'Ban gamsu da wannan hidimar ba.'
English Translation: I am not satisfied with this service.
English Sentiment: negative
Hausa Sentiment: korau
Final Answer (Hausa): korau"""
            # Add other LRL examples here

    if use_few_shot and examples:
        prompt = f"{instructions}\\n\\n{examples}\\n\\nFinal Answer ({lrl_name}):"
    else:
        prompt = f"{instructions}\\n\\nFinal Answer ({lrl_name}):"
    return prompt

def extract_lrl_sentiment_from_single_prompt(response_text: str, lang_code: str) -> str:
    """Extracts the final LRL sentiment label from the single prompt CoTR response."""
    # Attempt to parse structured output first if present
    # Example: "Final Answer (Swahili): chanya"
    lrl_name = LANG_NAMES.get(lang_code, lang_code)
    match = re.search(rf"Final Answer\s*\({re.escape(lrl_name)}\):\s*(\w+)", response_text, re.IGNORECASE)
    if match:
        return match.group(1).lower().strip()

    # Fallback: try to extract based on common LRL sentiment words if direct parsing fails
    # This is a simplified fallback; a more robust one might map known LRL words to standard labels.
    response_lower = response_text.lower().strip()
    
    # Keywords for Swahili
    if lang_code == 'sw':
        if 'chanya' in response_lower: return "positive" # Map to English for consistency if needed by metrics
        if 'hasi' in response_lower: return "negative"
        if 'kati' in response_lower or 'sio upande wowote' in response_lower : return "neutral"
    # Keywords for Hausa
                elif lang_code == 'ha':
        if 'tabbatacce' in response_lower: return "positive"
        if 'korau' in response_lower: return "negative"
        if 'tsaka-tsaki' in response_lower: return "neutral"
    
    # If no specific LRL keyword match, try English labels as a last resort from the full response
    return extract_sentiment_label_cotr(response_text)


def evaluate_sentiment_cotr_single_prompt(
    model: Any, 
    tokenizer: Any, 
    samples_df: pd.DataFrame,
    lang_code: str,
    use_few_shot: bool,
    generation_params: Dict
) -> pd.DataFrame:
    """Evaluate sentiment using Single-Prompt CoTR."""
    results = []
    default_gen_params = {
        "temperature": 0.3, "top_p": 0.9, "top_k": 40, 
        "max_new_tokens": 100, # Needs to be enough for all steps if model includes them
        "repetition_penalty": 1.1, "do_sample": True
    }
    current_gen_params = {**default_gen_params, **(generation_params or {})}

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Single-Prompt CoTR {lang_code}"):
        original_text = row['text']
        ground_truth_label = row['label']
        
        prompt = generate_single_prompt_sentiment_cotr(original_text, lang_code, use_few_shot)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=current_gen_params.get("max_input_length", 1024))
        
        # Ensure all parts of inputs are tensors and on the correct device
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, list): # If tokenizer somehow returned a list of ints for input_ids
                v = torch.tensor(v, dtype=torch.long)
            if hasattr(v, "to"): # Check if it's a tensor or tensor-like
                processed_inputs[k] = v.to(model.device)
            else:
                # This case should ideally not happen with return_tensors="pt"
                # but as a safeguard or for debugging:
                print(f"Warning: Input item '{k}' is not a tensor. Type: {type(v)}. Value: {v}")
                # Depending on the item, you might need specific handling or raise an error.
                # For now, we'll try to pass it as is, but this might be the source of issues.
                processed_inputs[k] = v 
                
                with torch.no_grad():
                    outputs = model.generate(
                **processed_inputs, # Use processed_inputs
                max_new_tokens=current_gen_params["max_new_tokens"],
                temperature=current_gen_params["temperature"],
                top_p=current_gen_params["top_p"],
                top_k=current_gen_params["top_k"],
                repetition_penalty=current_gen_params["repetition_penalty"],
                do_sample=current_gen_params["do_sample"],
                pad_token_id=tokenizer.pad_token_id # Explicitly set pad_token_id
                    )
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Extract the final LRL sentiment label
        # This part is tricky and relies on the model following the "Final Answer (LRL): label" format
        # or a robust `extract_lrl_sentiment_from_single_prompt`
        predicted_label_lrl = extract_lrl_sentiment_from_single_prompt(response_text, lang_code)
        
        # For consistent evaluation, we might want to map this LRL prediction back to a standard English label
        # This depends on how `calculate_sentiment_metrics` expects labels.
        # Assuming for now metrics function handles LRL or we map it. Let's output the LRL one for now.
        # We also need to decide what the 'final_predicted_label' for metrics will be.
        # If the single prompt is supposed to output LRL, then predicted_label_lrl is the one.
        # However, most metric functions expect consistent (e.g., English) labels.
        # For now, let's assume metrics will handle it or we'll standardize later.
        
        # Attempt to extract intermediate English sentiment if possible (for debugging/analysis)
        # This is hard without very structured output from the model.
        extracted_en_sentiment = "N/A" # Placeholder
        # Example crude extraction (highly dependent on model output format for single prompt)
        en_sent_match = re.search(r"English Sentiment:\s*(\w+)", response_text, re.IGNORECASE)
        if en_sent_match:
            extracted_en_sentiment = en_sent_match.group(1).lower().strip()


        results.append({
                'id': row.get('id', idx),
                'original_text': original_text,
                'ground_truth_label': ground_truth_label,
            'predicted_label_lrl': predicted_label_lrl, # The LRL label the model (hopefully) outputted
            'predicted_label_en_intermediate': extracted_en_sentiment, # If we can extract it
            'final_predicted_label': predicted_label_lrl, # This is what we'd compare against GT (after mapping GT to LRL or this to EN)
            'raw_response_single_prompt': response_text,
            'language': lang_code,
            'pipeline': 'single_prompt',
            'shot_type': 'few-shot' if use_few_shot else 'zero-shot'
        })
    return pd.DataFrame(results) 