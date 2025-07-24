import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
import re
import time
import sys

# Labels
POSSIBLE_LABELS_EN = ['business', 'entertainment', 'health', 'politics', 'religion', 'sports', 'technology']

# Übersetzungen für die Labels (Google Translate)
CLASS_LABELS_LRL = {
    "sw": {
        "health": "afya",
        "religion": "dini",
        "politics": "siasa",
        "sports": "michezo",
        "business": "biashara",
        "entertainment": "burudani",
        "technology": "teknolojia"
    },
    "ha": {
        "health": "lafiya",
        "religion": "addini",
        "politics": "siyasa",
        "sports": "wasanni",
        "business": "kasuwanci",
        "entertainment": "nishadi",
        "technology": "fasaha"
    },
    "fr": {
        "health": "santé",
        "religion": "religion",
        "politics": "politique",
        "sports": "sport",
        "business": "affaires",
        "entertainment": "divertissement",
        "technology": "technologie"
    }
}

# Projektverzeichnis 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def initialize_model(model_name):
    # Initialisiert Modell und einen Tokenizer
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
    text,
    possible_labels,
    lang_code = "en",
    model_name = "",
    use_few_shot = True
):
    # Prompt für die Textklassifizierung
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
Category: {possible_labels[0] if possible_labels and 'health' in possible_labels else (possible_labels[0] if possible_labels else 'health')} 

Text: 'Local elections are scheduled for next month, with several candidates vying for the mayoral position.'
Category: {possible_labels[2] if len(possible_labels) > 2 and 'politics' in possible_labels else (possible_labels[2] if len(possible_labels) > 2 else 'politics')}

Text: 'The home team secured a stunning victory in the final minutes of the match.'
Category: {possible_labels[3] if len(possible_labels) > 3 and 'sports' in possible_labels else (possible_labels[3] if len(possible_labels) > 3 else 'sports')}
"""
    prompt = f"{system_message}\n\n{instruction}\n\n"
    if use_few_shot:
        prompt += f"{few_shot_examples_en}\n\n"
    prompt += f"Text to classify: '{text}'\nCategory:"
    return prompt

def generate_lrl_instruct_classification_prompt(
    text,
    lang_code,
    possible_labels_en,
    model_name = "",
    use_few_shot = True
):
    # Prompt mit LRL-Anweisungen
    system_message_lrl = ""
    instruction_lrl = ""
    few_shot_examples_section = ""
    
    en_candidate_labels_str = ", ".join([f"'{label}'" for label in possible_labels_en])

    # Englische Few-Shot-Beispiele
    english_example_texts_and_labels = [
        {'text': 'The new healthcare bill was debated in parliament today, focusing on hospital funding and patient care.', 
         'label_key': 'health'},
        {'text': 'Local elections are scheduled for next month, with several candidates vying for the mayoral position.', 
         'label_key': 'politics'},
        {'text': 'The home team secured a stunning victory in the final minutes of the match.', 
         'label_key': 'sports'}
    ]

    instruction_block_current_sample = ""
    full_prompt_parts = []

    if lang_code == "sw":
        instruction_block = """Nakala: '<TEXT>'
Kategoria zinazowezekana (kwa Kiingereza): <POSSIBLE_LABELS>
Maagizo: Tambua kategoria ya nakala hii. Jibu lako lazima liwe MOJA TU ya kategoria za Kiingereza zilizotajwa.
Kategoria:"""
        instruction_block_current_sample = instruction_block.replace("<TEXT>", text.replace("'", "\'")).replace("<POSSIBLE_LABELS>", ", ".join([f"'{l}'" for l in possible_labels_en]))
        
        full_prompt_parts.append(instruction_block_current_sample)

        if use_few_shot:
            full_prompt_parts.append("\nMifano (Nakala za Kiingereza, Majibu ya Kiingereza):")
            for ex_en in english_example_texts_and_labels:
                target_en_label = ex_en['label_key'] 
                for pl_en in possible_labels_en:
                    if pl_en.lower() == ex_en['label_key'].lower():
                        target_en_label = pl_en
                        break
                full_prompt_parts.append(f"Nakala (Kiingereza): '{ex_en['text']}'\nKategoria (Kiingereza): {target_en_label}\n")
        
        full_prompt_parts.append("\nKategoria:")
        return "\n".join(full_prompt_parts)

    elif lang_code == "ha":
        instruction_block = """Rubutu: '<TEXT>'
Rukunonin da za su yiwu (da Turanci): <POSSIBLE_LABELS>
Umarni: Gano rukunin wannan rubutu. Amsarka dole ta kasance DAYA KAWAI daga cikin rukunonin Turanci da aka ambata.
Rukuni:"""
        instruction_block_current_sample = instruction_block.replace("<TEXT>", text.replace("'", "\\'")).replace("<POSSIBLE_LABELS>", ", ".join([f"'{l}'" for l in possible_labels_en]))

        full_prompt_parts.append(instruction_block_current_sample)

        if use_few_shot:
            full_prompt_parts.append("\nMisalai (Rubutun Turanci, Amsoshin Turanci):")
            for ex_en in english_example_texts_and_labels:
                target_en_label = ex_en['label_key'] 
                for pl_en in possible_labels_en:
                    if pl_en.lower() == ex_en['label_key'].lower():
                        target_en_label = pl_en
                        break
                full_prompt_parts.append(f"Rubutu (Turanci): '{ex_en['text']}'\nRukuni (Turanci): {target_en_label}\n")
        
        full_prompt_parts.append("\nRukuni:")
        return "\n".join(full_prompt_parts)

    elif lang_code == "fr":
        instruction_block = """Texte: '<TEXT>'
Catégories possibles (en anglais): <POSSIBLE_LABELS>
Instructions: Identifiez la catégorie de ce texte. Votre réponse doit être UNIQUEMENT l'une des catégories anglaises mentionnées.
Catégorie:"""
        instruction_block_current_sample = instruction_block.replace("<TEXT>", text.replace("'", "\\'")).replace("<POSSIBLE_LABELS>", ", ".join([f"'{l}'" for l in possible_labels_en]))

        full_prompt_parts.append(instruction_block_current_sample)

        if use_few_shot:
            full_prompt_parts.append("\nExemples (Textes en anglais, Réponses en anglais):")
            for ex_en in english_example_texts_and_labels:
                target_en_label = ex_en['label_key'] 
                for pl_en in possible_labels_en:
                    if pl_en.lower() == ex_en['label_key'].lower():
                        target_en_label = pl_en
                        break
                full_prompt_parts.append(f"Texte (Anglais): '{ex_en['text']}'\nCatégorie (Anglais): {target_en_label}\n")
        
        full_prompt_parts.append("\nCatégorie:")
        return "\n".join(full_prompt_parts)

    else: # Fallback für andere Sprachen
        # Fallback auf englische Anweisungen
        return generate_classification_prompt(text, possible_labels_en, lang_code='en', model_name=model_name, use_few_shot=use_few_shot)

def extract_classification_label(
    output_text,
    expected_en_labels
):
    # Extrahiert aus der Modellausgabe
    cleaned_output = output_text.strip().lower()
    
    best_match_en = ""
    highest_similarity = -1
    
    prefixes_to_strip = ["category:", "label:", "the category is", "classification:", "kategoria:", "rukuni:"]
    temp_output = cleaned_output
    for prefix in prefixes_to_strip:
        if temp_output.startswith(prefix):
            temp_output = temp_output[len(prefix):].strip()
    
    # Prüft auf direkte Übereinstimmung
    if temp_output in [l.lower() for l in expected_en_labels]:
        for en_label in expected_en_labels:
            if en_label.lower() == temp_output:
                return en_label

    # Prüft auf Übereinstimmung in der bereinigten Ausgabe
    for en_label in expected_en_labels:
        if en_label.lower() in cleaned_output:
            similarity = len(en_label)
            # Priorisiert exakte Übereinstimmung
            if cleaned_output == en_label.lower():
                best_match_en = en_label
                break
            if similarity > highest_similarity:
                best_match_en = en_label
                highest_similarity = similarity
    
    if best_match_en:
        return best_match_en
    else:
        # Fallback
        return "[Unknown Label]"

def process_classification_baseline(
    model,
    tokenizer,
    text,
    possible_labels, 
    lang_code,
    use_few_shot,
    # Parameter für die Generierung
    temperature, 
    top_p, 
    top_k, 
    max_tokens,
    repetition_penalty, 
    do_sample
):
    # Verarbeitet einen Text für die Klassifizierung
    start_time = time.time()
    raw_model_output = ""
    
    # Bestimmt den Prompt basierend auf dem Sprachcode
    if lang_code != 'en':
        prompt = generate_lrl_instruct_classification_prompt(
            text, lang_code, POSSIBLE_LABELS_EN, model_name="", use_few_shot=use_few_shot
        )
    else: # lang_code == 'en'
        prompt = generate_classification_prompt(
            text, POSSIBLE_LABELS_EN, lang_code, model_name="", use_few_shot=use_few_shot
        )

    # Bestimmt die maximale Länge für die Tokenisierung
    input_tokenize_max_length = getattr(tokenizer, 'model_max_length', 2048)
    if input_tokenize_max_length > 4096:
        input_tokenize_max_length = 4096
        
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_tokenize_max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Bestimmt, ob Sampling verwendet werden soll
    actual_do_sample = do_sample 

    with torch.no_grad():
        output_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=actual_do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
        
        raw_model_output = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        predicted_label = extract_classification_label(raw_model_output, possible_labels)

    runtime = time.time() - start_time
    return predicted_label, runtime, raw_model_output

def evaluate_classification_baseline(
    model_name,
    tokenizer,
    model,
    samples_df,
    lang_code,
    possible_labels,
    use_few_shot,
    # Parameter für die Generierung
    temperature, 
    top_p, 
    top_k, 
    max_tokens,
    repetition_penalty, 
    do_sample
):
    results = []
    shot_description = "few-shot" if use_few_shot else "zero-shot"
    prompt_lang_description = "LRL-instruct" if lang_code != 'en' else "EN-instruct"

    print(f"Evaluating {model_name} on {lang_code} ({prompt_lang_description}, {shot_description})...")

    for idx, row in samples_df.iterrows():
        text = row['text']
        ground_truth_label = row['label'] 

        predicted_label, runtime, raw_output = process_classification_baseline(
            model,
            tokenizer,
            text,
            possible_labels,
            lang_code=lang_code,
            use_few_shot=use_few_shot,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens, 
            repetition_penalty=repetition_penalty,
            do_sample=do_sample
        )

        results.append({
                'id': row.get('id', idx),
            'text': text,
                'ground_truth_label': ground_truth_label,
            'final_predicted_label': predicted_label,
            'raw_model_output': raw_output,
                'language': lang_code,
            'runtime_seconds': runtime,
            'prompt_language': prompt_lang_description,
            'shot_type': shot_description,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'max_tokens': max_tokens,
            'repetition_penalty': repetition_penalty
        })

    results_df = pd.DataFrame(results)
    return results_df
