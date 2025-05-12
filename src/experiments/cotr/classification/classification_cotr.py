import torch
import pandas as pd
from tqdm import tqdm
import os
import sys
import re
from typing import Any, Dict, List, Tuple

# Add project root to path to find model_initialization
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from model_initialization import initialize_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, List
# Import COMET utilities
from src.evaluation.cotr.translation_metrics import calculate_comet_score, COMET_AVAILABLE

# Define expected labels (can be dynamically determined from data later)
# EXPECTED_LABELS = ["positive", "negative", "neutral"]

# Define language names globally for access by multiple functions
lang_names = {
    "en": "English",
    "swa": "Swahili",
    "hau": "Hausa"
    # Add other potential languages if needed
}

# Define English labels - these should match what the model is trained/prompted to output
# For MasakhaNEWS, these are the actual categories:
CLASS_LABELS_ENGLISH = ['health', 'religion', 'politics', 'sports', 'local', 'business', 'entertainment']
# OLD: CLASS_LABELS_ENGLISH = ["topic_a", "topic_b", "topic_c", "other"]

# Placeholder for LRL translations of these English labels
CLASS_LABELS_LRL = {
    "sw": {"topic_a": "mada_a", "topic_b": "mada_b", "topic_c": "mada_c", "other": "nyingine"},
    "ha": {"topic_a": "jigo_a", "topic_b": "jigo_b", "topic_c": "jigo_c", "other": "daban"},
    "te": {"topic_a": "విషయం_a", "topic_b": "విషయం_b", "topic_c": "విషయం_c", "other": "ఇతర"}
    # Add other languages and their label translations
}

LANG_NAMES = {
    "en": "English",
    "sw": "Swahili",
    "ha": "Hausa",
    "te": "Telugu"
}

def generate_translation_prompt(text: str, source_lang: str, target_lang: str, is_label: bool = False) -> str:
    """Generate a prompt for translation (text or label)."""
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)
    if is_label:
        return f"Translate the following English category label to {target_name}: '{text}'\nProvide ONLY the single translated word for the category label in {target_name}.\n{target_name} Translation:"
    else:
        return f"Translate the following {source_name} text to {target_name}. Preserve the original meaning precisely.\nProvide only the direct translation.\nOriginal Text ({source_name}): '{text}'\n{target_name} Translation:"

def generate_classification_prompt_english(text_en: str, possible_labels: List[str], use_few_shot: bool = True) -> str:
    """Generate a prompt for English classification, asking for an English label."""
    labels_str = ", ".join([f"'{label}'" for label in possible_labels])
    instruction = f"Classify the following English text into one of these categories: {labels_str}.\nRespond with only the English category label."
    examples_en = ""
    if use_few_shot:
        examples_en = f"""
Examples:
Text: 'This article discusses recent political events in Africa.'
Category: {possible_labels[0] if len(possible_labels)>0 else 'topic_a'}

Text: 'The local sports team won their match yesterday evening.'
Category: {possible_labels[1] if len(possible_labels)>1 else 'topic_b'}
"""
    prompt = f"Text: '{text_en}'\n\n{instruction}"
    if use_few_shot:
        prompt += f"\n\n{examples_en}"
    prompt += "\n\nCategory:" # Expect English label
    return prompt

def extract_classification_label_cotr(output_text: str, expected_labels_en: List[str]) -> str:
    """Extracts English classification label from model output."""
    text_lower = output_text.lower().strip()
    common_prefixes = ["category:", "the category is", "label:", "classification:", "kategoria:", "rukuni:", "వర్గం:"]
    for prefix in common_prefixes:
        if text_lower.startswith(prefix):
            text_lower = text_lower[len(prefix):].strip()
    text_lower = text_lower.strip('\'\".()')
    for label in expected_labels_en:
        if label.lower() == text_lower:
            return label
    for label in expected_labels_en:
        if label.lower() in text_lower:
            return label
    return "other" if "other" in expected_labels_en else expected_labels_en[0]

def translate_text(
    model: Any, tokenizer: Any, text: str, source_lang: str, target_lang: str, 
    is_label: bool = False, generation_params: Dict = None
) -> str:
    """Translate text or a label using the model."""
    if not text or not text.strip(): return "[Empty input to translate]"
    prompt = generate_translation_prompt(text, source_lang, target_lang, is_label=is_label)
    default_params = {"temperature": 0.4, "top_p": 0.9, "top_k": 40, "max_new_tokens": 75 if is_label else 512, "repetition_penalty": 1.0, "do_sample": True}
    gen_params = {**default_params, **(generation_params or {})}
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=gen_params.get("max_input_length", 1024))
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_params, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    # For labels, use specific extractor to get the LRL label word
    if is_label:
        # This simple extraction might need improvement based on how model translates single words
        translated_word = response.strip().split()[0] if response.strip() else text # Fallback to original EN text if translation fails
        return translated_word.lower().strip('\'\".()')
    return response.strip() or "[No translation generated]"

def process_classification_english(
    model: Any, tokenizer: Any, text_en: str, possible_labels_en: List[str], 
    use_few_shot: bool, generation_params: Dict = None
) -> str:
    """Classify English text, expecting an English label."""
    prompt = generate_classification_prompt_english(text_en, possible_labels_en, use_few_shot)
    gen_params = generation_params if generation_params is not None else {}

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=gen_params.get("max_input_length", 1024))
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    model_gen_params = gen_params.copy()
    model_gen_params.pop("max_input_length", None)

    with torch.no_grad():
        outputs = model.generate(**inputs, **model_gen_params, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return extract_classification_label_cotr(response, possible_labels_en)

def evaluate_classification_cotr_multi_prompt(
    model: Any, tokenizer: Any, samples_df: pd.DataFrame, lang_code: str, 
    possible_labels_en: List[str], use_few_shot: bool, 
    text_translation_params: Dict, classification_params: Dict, label_translation_params: Dict
) -> pd.DataFrame:
    results = []
    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Multi-Prompt CoTR {lang_code}"):
        original_text = str(row['text'])
        ground_truth_label = str(row['label']).lower().strip()
        
        # 1. Translate LRL text to English
        text_en = translate_text(model, tokenizer, original_text, lang_code, "en", False, text_translation_params)
        
        comet_score_text_lrl_en = None
        if COMET_AVAILABLE and original_text and text_en:
            try:
                score = calculate_comet_score(
                    sources=[original_text],
                    predictions=[text_en]
                )
                if isinstance(score, list) and score: comet_score_text_lrl_en = score[0]
                elif isinstance(score, float): comet_score_text_lrl_en = score
            except Exception as e:
                print(f"Error calculating COMET for LRL text to EN (sample {row.get('id', idx)}): {e}")

        # 2. Classify sentiment of English text
        predicted_label_en = process_classification_english(model, tokenizer, text_en, possible_labels_en, use_few_shot, classification_params)
        
        # 3. Translate English sentiment label back to LRL
        predicted_label_lrl = predicted_label_en # Default
        comet_score_label_en_lrl = None
        if predicted_label_en in possible_labels_en: # Only translate valid EN predictions
            predicted_label_lrl = translate_text(model, tokenizer, predicted_label_en, "en", lang_code, True, label_translation_params)
            if COMET_AVAILABLE and predicted_label_en and predicted_label_lrl:
                try:
                    score = calculate_comet_score(
                        sources=[predicted_label_en],
                        predictions=[predicted_label_lrl]
                    )
                    if isinstance(score, list) and score: comet_score_label_en_lrl = score[0]
                    elif isinstance(score, float): comet_score_label_en_lrl = score
                except Exception as e:
                    print(f"Error calculating COMET for EN label to LRL (sample {row.get('id', idx)}): {e}")

        results.append({
            'id': row.get('id', idx), 'original_text': original_text, 'text_en': text_en,
            'ground_truth_label': ground_truth_label, 'predicted_label_en': predicted_label_en,
            'predicted_label_lrl': predicted_label_lrl,
            'comet_score_text_lrl_en': comet_score_text_lrl_en, # Added COMET score
            'comet_score_label_en_lrl': comet_score_label_en_lrl, # Added COMET score
            'final_predicted_label': predicted_label_en, # For metrics against English GT
            'language': lang_code, 'pipeline': 'multi_prompt', 'shot_type': 'few-shot' if use_few_shot else 'zero-shot'
        })
    return pd.DataFrame(results)

def generate_single_prompt_classification_cotr(lrl_text: str, lang_code: str, possible_labels_en: List[str], use_few_shot: bool = True) -> str:
    lrl_name = lang_names.get(lang_code, lang_code)
    en_labels_str = ", ".join([f"'{l}'" for l in possible_labels_en])
    instructions = f"""Text ({lrl_name}): '{lrl_text}'

Your task is to classify this {lrl_name} text.
Follow these steps:
1. Translate the {lrl_name} text to English.
2. Classify the English text into one of these English categories: {en_labels_str}.
3. Translate the chosen English category label back into {lrl_name}.
4. Provide ONLY the final {lrl_name} category label.

Example final output format if text was Swahili and category was '{possible_labels_en[0]}':
Kategoria: {CLASS_LABELS_LRL.get(lang_code, {}).get(possible_labels_en[0], possible_labels_en[0])}"""
    examples = ""
    if use_few_shot:
        lrl_labels_map = CLASS_LABELS_LRL.get(lang_code, {})
        if lang_code == 'sw':
            examples = f"""
Examples:
Text (Swahili): 'Nakala hii inajadili matukio ya hivi karibuni ya kisiasa barani Afrika.'
English Translation: This article discusses recent political events in Africa.
English Category: {possible_labels_en[0]}
Swahili Category: {lrl_labels_map.get(possible_labels_en[0], possible_labels_en[0])}
Final Answer (Swahili): {lrl_labels_map.get(possible_labels_en[0], possible_labels_en[0])}
"""
        elif lang_code == 'ha':
             examples = f"""
Examples:
Text (Hausa): 'Wannan makala tana tattauna abubuwan da suka faru na siyasa a Afirka kwanan nan.'
English Translation: This article discusses recent political events in Africa.
English Category: {possible_labels_en[0]}
Hausa Category: {lrl_labels_map.get(possible_labels_en[0], possible_labels_en[0])}
Final Answer (Hausa): {lrl_labels_map.get(possible_labels_en[0], possible_labels_en[0])}
"""
    prompt = f"{instructions}"
    if use_few_shot and examples: prompt += f"\n\n{examples}"
    prompt += f"\n\nFinal Answer ({lrl_name}):"
    return prompt

def extract_lrl_classification_from_single_prompt(response_text: str, lang_code: str, possible_labels_en: List[str]) -> str:
    lrl_name = lang_names.get(lang_code, lang_code)
    match = re.search(rf"Final Answer\s*\({re.escape(lrl_name)}\):\s*([\w\s]+)", response_text, re.IGNORECASE)
    if match:
        lrl_label_pred = match.group(1).lower().strip()
        # Attempt to map LRL label back to a standard English one for metrics
        lrl_to_en_map = {v.lower(): k for k, v in CLASS_LABELS_LRL.get(lang_code, {}).items() if k in possible_labels_en}
        if lrl_label_pred in lrl_to_en_map:
            return lrl_to_en_map[lrl_label_pred]
        # If LRL label is not in map but is one of the EN labels (model directly outputted EN)
        if lrl_label_pred in [l.lower() for l in possible_labels_en]:
             return lrl_label_pred
        return "other" if "other" in possible_labels_en else possible_labels_en[0] # Fallback
    # Fallback if regex fails, try general extraction on the whole response
    return extract_classification_label_cotr(response_text, possible_labels_en)

def evaluate_classification_cotr_single_prompt(
    model: Any, tokenizer: Any, samples_df: pd.DataFrame, lang_code: str, 
    possible_labels_en: List[str], use_few_shot: bool, generation_params: Dict
) -> pd.DataFrame:
    results = []
    gen_params = generation_params if generation_params is not None else {}

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Single-Prompt CoTR {lang_code}"):
        original_text = str(row['text'])
        ground_truth_label = str(row['label']).lower().strip()
        prompt = generate_single_prompt_classification_cotr(original_text, lang_code, possible_labels_en, use_few_shot)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=gen_params.get("max_input_length", 1024))
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        model_gen_params = gen_params.copy()
        model_gen_params.pop("max_input_length", None)

        with torch.no_grad():
            outputs = model.generate(**inputs, **model_gen_params, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        predicted_label_mapped_en = extract_lrl_classification_from_single_prompt(response_text, lang_code, possible_labels_en)
        # Try to get the raw LRL output if possible for analysis, before mapping
        raw_lrl_pred = "N/A"
        lrl_name_match = lang_names.get(lang_code, lang_code)
        match_raw = re.search(rf"Final Answer\s*\({re.escape(lrl_name_match)}\):\s*([\w\s]+)", response_text, re.IGNORECASE)
        if match_raw: raw_lrl_pred = match_raw.group(1).strip()

        results.append({
            'id': row.get('id', idx), 'original_text': original_text, 'ground_truth_label': ground_truth_label,
            'predicted_label_lrl_raw': raw_lrl_pred, # The model's direct LRL output
            'final_predicted_label': predicted_label_mapped_en, # Mapped to EN for metrics
            'raw_response_single_prompt': response_text,
            'language': lang_code, 'pipeline': 'single_prompt', 'shot_type': 'few-shot' if use_few_shot else 'zero-shot'
        })
    return pd.DataFrame(results)

def initialize_model(model_name: str) -> Tuple[Any, Any]:
    """Initialize the model and tokenizer."""
    print(f"Initializing model: {model_name}")
    # Define cache path or retrieve from config
    cache_path = "/work/bbd6522/cache_dir" # Make sure this is correct
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto", # Automatically distribute model across available GPUs
        trust_remote_code=True,
        cache_dir=cache_path
    )
    print(f"Successfully loaded {model_name}")
    return tokenizer, model # CHANGED: return tokenizer, then model 