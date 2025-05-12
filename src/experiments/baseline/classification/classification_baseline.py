import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os
import re
import time
import traceback
from typing import Tuple, Dict, List, Any, Optional

# Define English labels as the canonical ones
POSSIBLE_LABELS_EN = ['health', 'religion', 'politics', 'sports', 'local', 'business', 'entertainment']

def initialize_model(model_name: str) -> Tuple[Any, Any]:
    """Initialize a model and tokenizer, specifying cache directory."""
    print(f"Initializing {model_name}...")
    cache_path = "/work/bbd6522/cache_dir"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

def generate_classification_prompt(
    text: str,
    possible_labels: List[str], # These will always be English for this function
    lang_code: str = "en", # lang_code of the text itself
    model_name: str = "",
    use_few_shot: bool = True
) -> str:
    """
    Generate a prompt for text classification with English instructions and English candidate labels.
    The model is expected to output one of the provided English labels.
    """
    system_message = "You are an expert text classifier. Your task is to accurately categorize the provided text."
    instruction = (
        f"Carefully read the text below. Based on its content, classify it into ONE of the following categories: "
        f"{', '.join(possible_labels)}. "
        f"Your entire response must be ONLY the name of the chosen category. Do not add any other words or explanations."
    )
    few_shot_examples_en = ""
    if use_few_shot:
        few_shot_examples_en = f"""
Examples:
Text: 'The new healthcare bill was debated in parliament today, focusing on hospital funding and patient care.'
Category: health

Text: 'Local elections are scheduled for next month, with several candidates vying for the mayoral position.'
Category: politics

Text: 'The home team secured a stunning victory in the final minutes of the match.'
Category: sports

Text: 'A new community center opened downtown, offering various activities for residents.'
Category: local

Text: 'The company announced record profits for the third quarter, driven by strong sales in its new product line.'
Category: business

Text: 'The movie premiere was a star-studded event, with critics praising the lead actor\'s performance.'
Category: entertainment

Text: 'Thousands of pilgrims gathered for the annual religious festival, participating in traditional ceremonies and prayers.'
Category: religion
"""
    prompt = f"{system_message}\\n\\n{instruction}\\n\\n"
    if use_few_shot:
        prompt += f"{few_shot_examples_en}\\n\\n"
    prompt += f"Text to classify: '{text}'\\nCategory:"
    return prompt

def generate_lrl_instruct_classification_prompt(
    text: str, # Text in LRL
    lang_code: str, # e.g., 'sw', 'ha'
    possible_labels_en: List[str], # Model is still asked to output English labels
    model_name: str = "",
    use_few_shot: bool = True
) -> str: # Returns only the prompt string
    """
    Generate a prompt for text classification with LRL instructions.
    The model is still expected to output one of the provided ENGLISH labels.
    """
    system_message_lrl = ""
    instruction_lrl = ""
    few_shot_examples_lrl = ""
    
    # The candidate labels listed in the LRL prompt are the English ones.
    en_candidate_labels_str = ", ".join([f"'{label}'" for label in possible_labels_en])

    if lang_code == 'sw':
        system_message_lrl = "Wewe ni mtaalamu wa kuainisha maandishi. Kazi yako ni kuweka maandishi yaliyotolewa katika kategoria sahihi."
        instruction_lrl = (
            f"Soma kwa makini maandishi yaliyo hapa chini. Kulingana na yaliyomo, ainisha katika KATEGORIA MOJA kati ya hizi za Kiingereza: "
            f"{en_candidate_labels_str}. "
            f"Jibu lako lote lazima liwe TU jina la kategoria uliyochagua kwa Kiingereza. Usiongeze maneno mengine au maelezo."
        )
        if use_few_shot:
            # Few-shot examples show LRL text mapping to an ENGLISH label
            few_shot_examples_lrl = f"""
Mifano:
Maandishi: 'Muswada mpya wa afya ulijadiliwa bungeni leo, ukilenga ufadhili wa hospitali na huduma kwa wagonjwa.'
Kategoria: {possible_labels_en[0] if possible_labels_en else 'health'} 

Maandishi: 'Uchaguzi wa mitaa umepangwa kufanyika mwezi ujao, huku wagombea kadhaa wakigombea nafasi ya umeya.'
Kategoria: {possible_labels_en[2] if len(possible_labels_en) > 2 else 'politics'}

Maandishi: 'Timu ya nyumbani ilipata ushindi wa kushangaza katika dakika za mwisho za mechi.'
Kategoria: {possible_labels_en[3] if len(possible_labels_en) > 3 else 'sports'}
"""
    elif lang_code == 'ha':
        system_message_lrl = "Kai kwararren mai rarraba rubutu ne. Aikinka shine ka sanya rubutun da aka bayar cikin rukunin da ya dace."
        instruction_lrl = (
            f"Karanta rubutun da ke kasa a hankali. Dangane da abin da ke ciki, sanya shi cikin RUKUNI DAYA daga cikin wadannan na Turanci: "
            f"{en_candidate_labels_str}. "
            f"Dukkan amsarka dole ta zama KAWAI sunan rukunin da ka zaba da Turanci. Kada ka kara wasu kalmomi ko bayanai."
        )
        if use_few_shot:
            few_shot_examples_lrl = f"""
Misalai:
Rubutu: 'An yi muhawara kan sabon kudirin kiwon lafiya a majalisa a yau, inda aka mai da hankali kan samar da kudade ga asibitoci da kula da marasa lafiya.'
Rukuni: {possible_labels_en[0] if possible_labels_en else 'health'}

Rubutu: 'An shirya gudanar da zabukan kananan hukumomi a wata mai zuwa, inda 'yan takara da dama ke neman kujerar shugaban karamar hukuma.'
Rukuni: {possible_labels_en[2] if len(possible_labels_en) > 2 else 'politics'}

Rubutu: 'Kungiyar gida ta samu nasara mai ban mamaki a cikin mintuna na karshe na wasan.'
Rukuni: {possible_labels_en[3] if len(possible_labels_en) > 3 else 'sports'}
"""
    else: # Fallback for other LRLs if specific prompts aren't added
        print(f"Warning: LRL instructions not defined for '{lang_code}'. Using English instructions as fallback.")
        return generate_classification_prompt(text, POSSIBLE_LABELS_EN, lang_code)

    prompt = f"{system_message_lrl}\\n\\n{instruction_lrl}\\n\\n"
    if use_few_shot:
        prompt += f"{few_shot_examples_lrl}\\n\\n"
    # Adjust the final part of the prompt based on the LRL for clarity if desired, but still expect English label
    final_prompt_instruction = "Kategoria:" if lang_code == 'sw' else "Rukuni:" if lang_code == 'ha' else "Category:"
    prompt += f"Maandishi ya kuainishwa: '{text}'\\n{final_prompt_instruction}" 
    
    return prompt

def extract_classification_label(
    output_text: str,
    expected_en_labels: List[str] # Always expects English labels from the model output
) -> str:
    """
    Extracts the English classification label from the model's output text.
    This version assumes the model was instructed (even in LRL) to output an English label.
    """
    cleaned_output = output_text.strip().lower()
    
    best_match_en = ""
    highest_similarity = -1
    
    prefixes_to_strip = ["category:", "label:", "the category is", "classification:", "kategoria:", "rukuni:"]
    temp_output = cleaned_output
    for prefix in prefixes_to_strip:
        if temp_output.startswith(prefix):
            temp_output = temp_output[len(prefix):].strip()
    
    if temp_output in [l.lower() for l in expected_en_labels]:
        return temp_output

    for en_label in expected_en_labels:
        if en_label.lower() in cleaned_output:
            similarity = len(en_label)
            if cleaned_output == en_label.lower() or similarity > highest_similarity:
                if cleaned_output == en_label.lower():
                    best_match_en = en_label
                    break
                if similarity > highest_similarity:
                    best_match_en = en_label
                    highest_similarity = similarity
    
    if best_match_en:
        return best_match_en.lower()
    else:
        # If no specific label is found, return a placeholder or a default
        # This indicates the model did not produce a clearly identifiable label
        # print(f"WARN: Could not reliably extract label. Output: '{output_text}'. Defaulting to random choice.")
        # return random.choice(expected_en_labels) # OLD: random fallback
        print(f"WARN: Could not reliably extract label from '{output_text[:100]}...'. Returning '[Unknown Label]'.")
        return "[Unknown Label]" # NEW: Specific unknown label string

def process_classification_baseline(
    model: Any,
    tokenizer: Any,
    text: str,
    lang_code: str,
    model_name: str,
    generation_params: Dict,
    use_few_shot: bool,
    prompt_in_lrl: bool = False
) -> Tuple[str, float, str]: # Returns predicted_EN_label, runtime, raw_model_output
    start_time = time.time()
    raw_model_output = ""
    
    if prompt_in_lrl and lang_code != 'en':
        prompt = generate_lrl_instruct_classification_prompt(
            text, lang_code, POSSIBLE_LABELS_EN, model_name, use_few_shot
        )
    else:
        prompt = generate_classification_prompt(
            text, POSSIBLE_LABELS_EN, lang_code, model_name, use_few_shot
        )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=generation_params.get("max_input_length", 2048))
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    gen_params_to_use = {
        "max_new_tokens": generation_params.get("max_tokens", 10),
        "temperature": generation_params.get("temperature", 0.2),
        "top_p": generation_params.get("top_p", 0.9),
        "top_k": generation_params.get("top_k", 40),
        "repetition_penalty": generation_params.get("repetition_penalty", 1.1),
        "do_sample": True if generation_params.get("temperature", 0.2) > 0.01 else False,
        "pad_token_id": tokenizer.eos_token_id
    }

    try:
        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, **gen_params_to_use)
        
        raw_model_output = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        predicted_en_label = extract_classification_label(raw_model_output, POSSIBLE_LABELS_EN)

    except Exception as e:
        print(f"Error during model generation or label extraction for text: '{text[:50]}...'")
        print(traceback.format_exc())
        predicted_en_label = "other" # Default on error
        raw_model_output = f"[ERROR: {str(e)}]"

    runtime = time.time() - start_time
    return predicted_en_label, runtime, raw_model_output

def evaluate_classification_baseline(
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    possible_labels_en: List[str], # Canonical English labels for metrics
    prompt_in_lrl: bool,
    use_few_shot: bool,
    generation_params: Dict
) -> pd.DataFrame:
    results = []
    shot_description = "few-shot" if use_few_shot else "zero-shot"
    prompt_lang_description = "LRL-instruct" if prompt_in_lrl and lang_code != 'en' else "EN-instruct"

    print(f"Evaluating {model_name} on {lang_code} ({prompt_lang_description}, {shot_description})...")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} Classification Baseline"):
        text = row['text']
        ground_truth_label = row['label'] 

        predicted_en_label, runtime, raw_output = process_classification_baseline(
            model, tokenizer, text, lang_code, 
            model_name,
            generation_params=generation_params,
            use_few_shot=use_few_shot,
            prompt_in_lrl=prompt_in_lrl
        )

        results.append({
                'id': row.get('id', idx),
            'text': text,
                'ground_truth_label': ground_truth_label,
            'final_predicted_label': predicted_en_label,
            'raw_model_output': raw_output,
                'language': lang_code,
            'runtime_seconds': runtime,
            'prompt_language': prompt_lang_description,
            'shot_type': shot_description,
            'temperature': generation_params.get("temperature"),
            'top_p': generation_params.get("top_p"),
            'top_k': generation_params.get("top_k"),
            'max_tokens': generation_params.get("max_tokens"),
            'repetition_penalty': generation_params.get("repetition_penalty")
        })

    results_df = pd.DataFrame(results)
    return results_df
