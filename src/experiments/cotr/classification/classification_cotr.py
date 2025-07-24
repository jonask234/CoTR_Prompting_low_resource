import torch
import pandas as pd
from tqdm import tqdm
import os
import sys
import re
import time

# Projekt-Root zum Pfad hinzufügen
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Modell und Tokenizer importieren
from transformers import AutoTokenizer, AutoModelForCausalLM

# Sprachnamen definieren
LANG_NAMES = {
    "en": "English",
    "sw": "Swahili",
    "ha": "Hausa"
}

# Englische Labels für die Kategorien
CLASS_LABELS_ENGLISH = ['business', 'entertainment', 'health', 'politics', 'religion', 'sports', 'technology']

# Übersetzungen der englischen Labels
CLASS_LABELS_LRL = {
    "sw": {
        "health": "afya", "religion": "dini", "politics": "siasa",
        "sports": "michezo", "business": "biashara",
        "entertainment": "burudani", "technology": "teknolojia"
    },
    "ha": {
        "health": "lafiya", "religion": "addini", "politics": "siyasa",
        "sports": "wasanni", "business": "kasuwanci",
        "entertainment": "nishadi", "technology": "fasaha"
    }
}

def get_language_name(lang_code):
    # Gibt den vollen Sprachnamen zurück
    return LANG_NAMES.get(lang_code.lower(), lang_code.capitalize())

def generate_translation_prompt(text, source_lang, target_lang, is_label = False):
    # Erstellt einen Prompt für die Übersetzung
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)
    if is_label:
        return f"Translate the following English category label to {target_name}: '{text}'\nProvide ONLY the single translated word for the category label in {target_name}.\n{target_name} Translation:"
    else:
        return f"Translate the following {source_name} text to {target_name}. Preserve the original meaning precisely.\nProvide only the direct translation.\nOriginal Text ({source_name}): '{text}'\n{target_name} Translation:"

def generate_classification_prompt_english(text_en, possible_labels, use_few_shot = True):
    # Erstellt einen Prompt für die Klassifizierung auf Englisch
    labels_str = ", ".join([f"'{l}'" for l in possible_labels])
    
    instruction_segment = f"Classify the following English text into one of these categories: {labels_str}.\\nRespond with only the English category label."
    
    prompt_parts = []
    prompt_parts.append(f"Text: '{text_en}'")
    prompt_parts.append("\\n") 
    prompt_parts.append(instruction_segment)
    
    examples_segment = ""
    if use_few_shot:
        example_lines = ["\\n\\nExamples:"]
        # Beispiele für alle 7 Kategorien
        example_lines.append(f"\\nText: 'The new healthcare bill was debated in parliament today, focusing on hospital funding and patient care.'\\nCategory: health")
        example_lines.append(f"\\nText: 'Local elections are scheduled for next month, with several candidates vying for the mayoral position.'\\nCategory: politics")  
        example_lines.append(f"\\nText: 'The home team secured a stunning victory in the final minutes of the match.'\\nCategory: sports")
        example_lines.append(f"\\nText: 'The stock market reached an all-time high as investors showed confidence in the new economic policies.'\\nCategory: business")
        example_lines.append(f"\\nText: 'The latest smartphone features advanced AI capabilities and improved battery life for consumers.'\\nCategory: technology")
        example_lines.append(f"\\nText: 'The blockbuster movie premiered last night with celebrities walking the red carpet.'\\nCategory: entertainment")
        example_lines.append(f"\\nText: 'Religious leaders gathered for an interfaith dialogue to promote peace and understanding.'\\nCategory: religion")
        
        examples_segment = "".join(example_lines)
        prompt_parts.append(examples_segment)
            
    prompt_parts.append("\\n\\nCategory:") 
    return "".join(prompt_parts)

def extract_classification_label_cotr(output_text, expected_labels_en):
    # Extrahiert das Label aus der Modell-Ausgabe
    default_fallback_label = "[Unknown Label]"
    if not output_text or not isinstance(output_text, str):
        return default_fallback_label

    text_lower = output_text.lower().strip()
    
    common_prefixes = ["category:", "the category is", "label:", "classification:", "kategoria:", "rukuni:", "వర్గం:", "the correct category is", "this text falls under", "i would classify this as"]
    for prefix in common_prefixes:
        if text_lower.startswith(prefix):
            text_lower = text_lower[len(prefix):].strip()
    text_lower = text_lower.strip('\'".().[]{}!?;' )

    # Exakte Übereinstimmung zuerst
    for label in expected_labels_en:
        if label.lower() == text_lower:
            return label

    # Teilstring-Prüfung
    for label in expected_labels_en:
        if re.search(r'\b' + re.escape(label.lower()) + r'\b', text_lower):
            return label
    
    return default_fallback_label

def translate_text(
    model, tokenizer, text, source_lang, target_lang,
    is_label = False, generation_params = None
):
    # Übersetzt einen Text oder ein Label
    start_time = time.time()
    if not text or not text.strip():
        return "[Empty input to translate]", "[Empty input to translate]", time.time() - start_time

    prompt = generate_translation_prompt(text, source_lang, target_lang, is_label=is_label)
    
    default_gen_params = {
        "temperature": 0.3, "top_p": 0.9, "top_k": 40,
        "max_new_tokens": 60 if is_label else 300,
        "repetition_penalty": 1.0, "do_sample": True,
        "max_input_length": 2048
    }
    effective_gen_params = {**default_gen_params, **(generation_params if generation_params is not None else {})}
    effective_gen_params["do_sample"] = effective_gen_params.get("temperature", 0) > 0.01

    if 'max_tokens' in effective_gen_params and 'max_new_tokens' not in effective_gen_params:
        effective_gen_params['max_new_tokens'] = effective_gen_params.pop('max_tokens')
    elif 'max_tokens' in effective_gen_params and 'max_new_tokens' in effective_gen_params:
        del effective_gen_params['max_tokens']
        
    max_input_tok_len = effective_gen_params.pop("max_input_length", 2048)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tok_len)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    raw_model_output_str = "[No translation generated]"
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **effective_gen_params, pad_token_id=tokenizer.eos_token_id)
        raw_model_output_str = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during model.generate in translate_text ({source_lang}->{target_lang}): {e}")
        raw_model_output_str = f"[Generation Error: {e}]"
        return raw_model_output_str, raw_model_output_str, time.time() - start_time

    cleaned_response = raw_model_output_str.strip()
    if is_label:
        parts = cleaned_response.split()
        translated_word = parts[0] if parts else text
        final_translation = translated_word.lower().strip('\'\".().,;' )
    else:
        translation_header = f"{get_language_name(target_lang)} Translation:"
        if cleaned_response.lower().startswith(translation_header.lower()):
            final_translation = cleaned_response[len(translation_header):].strip()
        else:
            final_translation = cleaned_response
        if not final_translation: final_translation = "[No translation content]"
            
    duration = time.time() - start_time
    return final_translation, raw_model_output_str, duration

def process_classification_english(
    model, tokenizer, text_en, possible_labels_en,
    use_few_shot, generation_params = None
):
    # Klassifiziert englischen Text
    prompt = generate_classification_prompt_english(text_en, possible_labels_en, use_few_shot)
    default_gen_params = {
        "temperature": 0.2, "top_p": 0.9, "top_k": 30,
        "max_new_tokens": 20, 
        "repetition_penalty": 1.05, "do_sample": True,
        "max_input_length": 2048
    }
    effective_gen_params = {**default_gen_params, **(generation_params if generation_params is not None else {})}
    effective_gen_params["do_sample"] = effective_gen_params.get("temperature", 0) > 0.01

    if 'max_tokens' in effective_gen_params and 'max_new_tokens' not in effective_gen_params:
        effective_gen_params['max_new_tokens'] = effective_gen_params.pop('max_tokens')
    elif 'max_tokens' in effective_gen_params and 'max_new_tokens' in effective_gen_params:
        del effective_gen_params['max_tokens']

    max_input_tok_len = effective_gen_params.pop("max_input_length", 2048)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tok_len)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    response_str = "[Classification Error]"
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **effective_gen_params, pad_token_id=tokenizer.eos_token_id)
        response_str = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during model.generate in process_classification_english: {e}")
        return "[Classification Error]"

    return extract_classification_label_cotr(response_str, possible_labels_en)

def evaluate_classification_cotr_multi_prompt(
    model, tokenizer, samples_df, lang_code,
    possible_labels_en, use_few_shot,
    text_translation_params,
    classification_params,
    label_translation_params,
    model_name = None
):
    # Führt die CoTR-Klassifizierung mit mehreren Prompts aus
    results = []
    active_possible_labels_en = possible_labels_en if possible_labels_en else CLASS_LABELS_ENGLISH

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Multi-Prompt CoTR {lang_code} ({model_name})"):
        original_text = str(row['text'])
        ground_truth_label = str(row['label']).lower().strip() if 'label' in row and pd.notna(row['label']) else "[Missing Ground Truth]"

        text_en, raw_text_translation, time_text_trans = "[Translation Error]", None, 0.0
        if lang_code.lower() == "en":
            text_en = original_text
        elif original_text and original_text.strip() and original_text != "[Empty input to translate]":
            text_en, raw_text_translation, time_text_trans = translate_text(
                model, tokenizer, original_text, lang_code, "en", False, text_translation_params
            )

        predicted_label_en = process_classification_english(
            model, tokenizer, text_en, active_possible_labels_en, use_few_shot, classification_params
        )

        predicted_label_lrl, raw_label_translation, time_label_trans = predicted_label_en, None, 0.0
        if lang_code.lower() != "en" and predicted_label_en in active_possible_labels_en and "[Classification Error]" not in predicted_label_en:
            predicted_label_lrl, raw_label_translation, time_label_trans = translate_text(
                model, tokenizer, predicted_label_en, "en", lang_code, True, label_translation_params
            )
        
        results.append({
            'id': row.get('id', idx), 'original_text': original_text, 'text_en_model': text_en,
            'ground_truth_label_eng': ground_truth_label,
            'predicted_label_eng_model': predicted_label_en,
            'predicted_label_lrl_model': predicted_label_lrl,
            'raw_text_translation_output': raw_text_translation,
            'raw_label_translation_output': raw_label_translation,
            'time_text_translation_sec': time_text_trans,
            'time_label_translation_sec': time_label_trans,
            'language': lang_code, 'pipeline_type': 'multi_prompt', 
            'shot_setting': 'few-shot' if use_few_shot else 'zero-shot'
        })
    return pd.DataFrame(results)

def generate_single_prompt_classification_cotr(lrl_text, lang_code, possible_labels_en, use_few_shot = True):
    # Erstellt einen einzelnen CoTR-Prompt
    original_lrl_name = get_language_name(lang_code)
    english_labels_str = ", ".join(f"'{label}'" for label in possible_labels_en)

    prompt = f"""You are an advanced AI assistant. You will be given a text in {original_lrl_name}.
Your task is to perform text classification by following these steps precisely and outputting each step as specified:

1.  **Translate to English**: Translate the original {original_lrl_name} text accurately into English.
    Output this as:
    Translated English Text: [Your English Translation of the text]

2.  **Classify English Text**: From the translated English text (from Step 1), classify it into ONE of the following English categories: {english_labels_str}.
    Your response for this step should be ONLY the English category name from the list.
    Output this as:
    English Category: [Chosen English Category]

3.  **Translate Category to {original_lrl_name}**: Take the English category chosen in Step 2 and translate it accurately into {original_lrl_name}.
    Output this as:
    {original_lrl_name} Category: [Translated {original_lrl_name} Category]

Ensure each step's output is clearly labeled EXACTLY as shown above.
The final "{original_lrl_name} Category" is the ultimate result for classification.

Original {original_lrl_name} Text:
'{lrl_text}'
"""
    if use_few_shot:
        # Beispiele für alle 7 Kategorien
        examples = [
            {
                "text": "The new healthcare bill was debated in parliament today, focusing on hospital funding and patient care.",
                "category_en": "health"
            },
            {
                "text": "Local elections are scheduled for next month, with several candidates vying for the mayoral position.",
                "category_en": "politics"
            },
            {
                "text": "The home team secured a stunning victory in the final minutes of the match.",
                "category_en": "sports"
            },
            {
                "text": "The stock market reached an all-time high as investors showed confidence in the new economic policies.",
                "category_en": "business"
            },
            {
                "text": "The latest smartphone features advanced AI capabilities and improved battery life for consumers.",
                "category_en": "technology"
            },
            {
                "text": "The blockbuster movie premiered last night with celebrities walking the red carpet.",
                "category_en": "entertainment"
            },
            {
                "text": "Religious leaders gathered for an interfaith dialogue to promote peace and understanding.",
                "category_en": "religion"
            }
        ]
        
        prompt += f"""\nExamples:

"""
        for i, example in enumerate(examples, 1):
            example_text = example["text"]
            ex_cat_en = example["category_en"]
            ex_cat_lrl = CLASS_LABELS_LRL.get(lang_code, {}).get(ex_cat_en.lower(), ex_cat_en)
            
            prompt += f"""Original {original_lrl_name} Text:
'{example_text}'
Translated English Text: {example_text}
English Category: {ex_cat_en}
{original_lrl_name} Category: {ex_cat_lrl}

"""
    prompt += "\nNow, generate all the above steps for the Original Text provided at the beginning:"
    return prompt

def extract_lrl_classification_from_single_prompt(response_text, lang_code, possible_labels_en):
    # Extrahiert das Label aus dem einzelnen Prompt
    lrl_name = get_language_name(lang_code)
    fallback_label = "[Unknown Label]"

    match_lrl = re.search(rf"{re.escape(lrl_name)}\s*Category\s*:\s*(.+?)(?:\n\s*(?:\S|$)|$)", response_text, re.IGNORECASE | re.DOTALL)
    if match_lrl:
        lrl_label_pred_raw = match_lrl.group(1).strip().lower()
        lrl_to_en_map = {v.lower(): k for k, v_list in CLASS_LABELS_LRL.get(lang_code, {}).items() for v in (v_list if isinstance(v_list, list) else [v_list]) if k.lower() in [lbl.lower() for lbl in possible_labels_en]}
        # Direkte Zuordnung
        for en_label_orig_case, lrl_translation_val in CLASS_LABELS_LRL.get(lang_code, {}).items():
            if en_label_orig_case.lower() not in [pl.lower() for pl in possible_labels_en]:
                continue
            lrl_options = [lrl_translation_val] if isinstance(lrl_translation_val, str) else lrl_translation_val
            if lrl_label_pred_raw in [opt.lower() for opt in lrl_options]:
                return en_label_orig_case

        if lrl_label_pred_raw in [l.lower() for l in possible_labels_en]:
            for en_l in possible_labels_en: 
                if en_l.lower() == lrl_label_pred_raw: return en_l
        print(f"SP Extract: LRL '{lrl_label_pred_raw}' not mapped. Checking English step.")

    match_eng = re.search(rf"English\s*Category\s*:\s*(.+?)(?:\n\s*{re.escape(lrl_name)}\s*Category\s*:|$)", response_text, re.IGNORECASE | re.DOTALL)
    if match_eng:
        eng_label_pred = match_eng.group(1).strip().lower()
        if eng_label_pred in [l.lower() for l in possible_labels_en]:
            for en_l in possible_labels_en:
                if en_l.lower() == eng_label_pred: return en_l
        return fallback_label

    print(f"SP Extract: Failed to extract LRL/English label from: '{response_text[:150]}...'. Fallback: '{fallback_label}'.")
    return fallback_label

def evaluate_classification_cotr_single_prompt(
    model, tokenizer, samples_df, lang_code,
    possible_labels_en, use_few_shot, generation_params,
    model_name = None
):
    # Führt die CoTR-Klassifizierung mit einem einzelnen Prompt aus
    results = []
    active_possible_labels_en = possible_labels_en if possible_labels_en else CLASS_LABELS_ENGLISH

    default_gen_params = {
        "temperature": 0.2, "top_p": 0.9, "top_k": 30,
        "max_new_tokens": 350, 
        "repetition_penalty": 1.05, "do_sample": True,
        "max_input_length": 2048
    }
    effective_gen_params = {**default_gen_params, **(generation_params if generation_params is not None else {})}
    effective_gen_params["do_sample"] = effective_gen_params.get("temperature", 0) > 0.01

    if 'max_tokens' in effective_gen_params and 'max_new_tokens' not in effective_gen_params:
        effective_gen_params['max_new_tokens'] = effective_gen_params.pop('max_tokens')
    elif 'max_tokens' in effective_gen_params and 'max_new_tokens' in effective_gen_params:
        del effective_gen_params['max_tokens']
        
    max_input_tok_len = effective_gen_params.pop("max_input_length", 2048)

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Single-Prompt CoTR {lang_code} ({model_name})"):
        original_text = str(row['text'])
        ground_truth_label = str(row['label']).lower().strip() if 'label' in row and pd.notna(row['label']) else "[Missing Ground Truth]"

        prompt = generate_single_prompt_classification_cotr(original_text, lang_code, active_possible_labels_en, use_few_shot)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tok_len)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        response_text, raw_lrl_pred_from_response = "[Generation Error]", "N/A"
        time_generation = 0.0
        start_gen_time = time.time()
        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, **effective_gen_params, pad_token_id=tokenizer.eos_token_id)
            response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            time_generation = time.time() - start_gen_time
            lrl_name_match = get_language_name(lang_code)
            match_raw = re.search(rf"{re.escape(lrl_name_match)}\s*Category\s*:\s*(.+?)(?:\n\s*(?:\S|$)|$)", response_text, re.IGNORECASE | re.DOTALL)
            if match_raw: raw_lrl_pred_from_response = match_raw.group(1).strip()
        except Exception as e_gen:
            print(f"Error in SP generation (sample {row.get('id', idx)}): {e_gen}")
            time_generation = time.time() - start_gen_time

        predicted_label_mapped_en = extract_lrl_classification_from_single_prompt(response_text, lang_code, active_possible_labels_en)
        intermediate_en_text, intermediate_en_label = None, None
        match_en_text = re.search(r"Translated English Text:\s*(.*?)(?:\nEnglish Category:|$)", response_text, re.DOTALL | re.IGNORECASE)
        if match_en_text: intermediate_en_text = match_en_text.group(1).strip()
        match_en_label = re.search(rf"English Category:\s*(.*?)(?:\n{re.escape(get_language_name(lang_code))} Category:|$)", response_text, re.DOTALL | re.IGNORECASE)
        if match_en_label: intermediate_en_label = match_en_label.group(1).strip().lower()

        current_result = {
            'id': row.get('id', idx), 'text_lrl': original_text, 'text_en_translated': intermediate_en_text, 'label_lrl_ground_truth': ground_truth_label, 
            'label_en_predicted_intermediate': intermediate_en_label, 'label_lrl_predicted_final': raw_lrl_pred_from_response,
            'predicted_label_accuracy': predicted_label_mapped_en,
            'raw_text_translation_output': None, 'raw_classification_output': response_text,
            'raw_label_translation_output': None,
            'error_message': None, 'runtime_sec': time_generation
        }
        results.append(current_result)
    return pd.DataFrame(results)

def initialize_model(model_name):
    # Initialisiert das Modell und den Tokenizer
    print(f"Initializing model: {model_name}")
    cache_path = "/work/bbd6522/cache_dir"
    
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

    # Setzt das Padding-Token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Successfully loaded {model_name}.")
    return tokenizer, model 