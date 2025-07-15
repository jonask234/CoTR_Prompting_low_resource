import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os
import re
import time
import traceback
from typing import Tuple, Dict, List, Any, Optional
import logging
import sys

# Define English labels as the canonical ones
POSSIBLE_LABELS_EN = ['business', 'entertainment', 'health', 'politics', 'religion', 'sports', 'technology']

# Define LRL translations for MasakhaNEWS labels (for LRL instruction headers)
CLASS_LABELS_LRL = {
    "sw": { # Swahili
        "health": "afya",
        "religion": "dini",
        "politics": "siasa",
        "sports": "michezo",
        "business": "biashara",
        "entertainment": "burudani",
        "technology": "teknolojia"
    },
    "ha": { # Hausa
        "health": "lafiya",
        "religion": "addini",
        "politics": "siyasa",
        "sports": "wasanni",
        "business": "kasuwanci",
        "entertainment": "nishadi",
        "technology": "fasaha"
    },
    "fr": { # Adding French for completeness, assuming MasakhaNEWS has French
        "health": "santé",
        "religion": "religion",
        "politics": "politique",
        "sports": "sport",
        "business": "affaires",
        "entertainment": "divertissement",
        "technology": "technologie"
    }
}

# Initialize logging
logger = logging.getLogger(__name__)

# Add project root to Python path if not already (helps with module resolution)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    (Adopted from user's old script)
    """
    system_message = "You are an expert text classifier. Your task is to accurately categorize the provided text."
    instruction = (
        f"Carefully read the text below. Based on its content, classify it into ONE of the following categories: "
        f"{', '.join(possible_labels)}. "
        f"Your entire response must be ONLY the name of the chosen category. Do not add any other words or explanations."
    )
    few_shot_examples_en = ""
    if use_few_shot:
        # Standardized to 3 examples for consistency across all tasks
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
    text: str, # Text in LRL
    lang_code: str, # e.g., 'sw', 'ha'
    possible_labels_en: List[str], # Model is still asked to output English labels
    model_name: str = "",
    use_few_shot: bool = True
) -> str: # Returns only the prompt string
    """
    Generate a prompt for text classification with LRL instructions.
    The model is still expected to output one of the provided ENGLISH labels.
    Uses English few-shot example texts.
    (Adopted from user's old script, modified for English few-shot examples)
    """
    system_message_lrl = ""
    instruction_lrl = ""
    few_shot_examples_section = "" # Changed variable name for clarity
    
    en_candidate_labels_str = ", ".join([f"'{label}'" for label in possible_labels_en])

    # English few-shot examples (texts and categories)
    # These are used regardless of the LRL instruction language
    english_example_texts_and_labels = [
        {'text': 'The new healthcare bill was debated in parliament today, focusing on hospital funding and patient care.', 
         'label_key': 'health'},
        {'text': 'Local elections are scheduled for next month, with several candidates vying for the mayoral position.', 
         'label_key': 'politics'},
        {'text': 'The home team secured a stunning victory in the final minutes of the match.', 
         'label_key': 'sports'}
    ]

    # All few-shot examples are now consistently in English across all configurations

    instruction_block_current_sample = "" # Will be populated by lang-specific block
    full_prompt_parts = [] # Initialize here

    logging.info(f"Inside generate_lrl_instruct_classification_prompt: Received lang_code='{lang_code}', use_few_shot={use_few_shot}")

    if lang_code == "sw":
        logging.info(f"Generating Swahili-specific classification prompt for text: '{text[:50]}...'")
        instruction_block = """Nakala: '<TEXT>'
Kategoria zinazowezekana (kwa Kiingereza): <POSSIBLE_LABELS>
Maagizo: Tambua kategoria ya nakala hii. Jibu lako lazima liwe MOJA TU ya kategoria za Kiingereza zilizotajwa.
Kategoria:"""
        instruction_block_current_sample = instruction_block.replace("<TEXT>", text.replace("'", "\'")).replace("<POSSIBLE_LABELS>", ", ".join([f"'{l}'" for l in possible_labels_en]))
        
        full_prompt_parts.append(instruction_block_current_sample)

        if use_few_shot:
            full_prompt_parts.append("\nMifano (Nakala za Kiingereza, Majibu ya Kiingereza):") # Clarify examples are English text and English labels
            for ex_en in english_example_texts_and_labels:
                # Find the English label from possible_labels_en that corresponds to ex_en['label_key']
                # This assumes possible_labels_en contains the actual label string, e.g., 'health', 'politics'
                target_en_label = ex_en['label_key'] # Default to key if not found, though ideally it should match
                for pl_en in possible_labels_en:
                    if pl_en.lower() == ex_en['label_key'].lower():
                        target_en_label = pl_en
                        break
                full_prompt_parts.append(f"Nakala (Kiingereza): '{ex_en['text']}'\nKategoria (Kiingereza): {target_en_label}\n")
        
        full_prompt_parts.append("\nKategoria:") # Final prompt for the model to complete
        return "\n".join(full_prompt_parts)

    elif lang_code == "ha":
        logging.info(f"Generating Hausa-specific classification prompt for text: '{text[:50]}...'")
        instruction_block = """Rubutu: '<TEXT>'
Rukunonin da za su yiwu (da Turanci): <POSSIBLE_LABELS>
Umarni: Gano rukunin wannan rubutu. Amsarka dole ta kasance DAYA KAWAI daga cikin rukunonin Turanci da aka ambata.
Rukuni:"""
        instruction_block_current_sample = instruction_block.replace("<TEXT>", text.replace("'", "\\'")).replace("<POSSIBLE_LABELS>", ", ".join([f"'{l}'" for l in possible_labels_en]))

        full_prompt_parts.append(instruction_block_current_sample)

        if use_few_shot:
            full_prompt_parts.append("\nMisalai (Rubutun Turanci, Amsoshin Turanci):") # Clarify examples are English text and English labels
            for ex_en in english_example_texts_and_labels:
                target_en_label = ex_en['label_key'] 
                for pl_en in possible_labels_en:
                    if pl_en.lower() == ex_en['label_key'].lower():
                        target_en_label = pl_en
                        break
                full_prompt_parts.append(f"Rubutu (Turanci): '{ex_en['text']}'\nRukuni (Turanci): {target_en_label}\n")
        
        full_prompt_parts.append("\nRukuni:") # Final prompt for the model to complete
        return "\n".join(full_prompt_parts)

    elif lang_code == "fr": # Added French
        logging.info(f"Generating French-specific classification prompt for text: '{text[:50]}...'")
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

    else: # Fallback for other LRLs not explicitly defined
        logging.warning(f"LRL instructions for '{lang_code}' not specifically defined in generate_lrl_instruct_classification_prompt. Falling back to English instructions via generate_classification_prompt.")
        # This fallback correctly uses English instructions by calling the main English-based prompter
        return generate_classification_prompt(text, possible_labels_en, lang_code='en', model_name=model_name, use_few_shot=use_few_shot)

def extract_classification_label(
    output_text: str,
    expected_en_labels: List[str] # Always expects English labels from the model output
) -> str:
    """
    Extracts the English classification label from the model's output text.
    This version assumes the model was instructed (even in LRL) to output an English label.
    (Adopted from user's old script)
    """
    cleaned_output = output_text.strip().lower()
    
    best_match_en = ""
    highest_similarity = -1
    
    prefixes_to_strip = ["category:", "label:", "the category is", "classification:", "kategoria:", "rukuni:"]
    temp_output = cleaned_output
    for prefix in prefixes_to_strip:
        if temp_output.startswith(prefix):
            temp_output = temp_output[len(prefix):].strip()
    
    # After stripping prefixes, check for direct match of the remaining string
    if temp_output in [l.lower() for l in expected_en_labels]:
        # Find the original cased label
        for en_label in expected_en_labels:
            if en_label.lower() == temp_output:
                return en_label # Return original casing

    # If direct match failed after stripping, check for labels within the cleaned_output (original logic from old script)
    for en_label in expected_en_labels:
        if en_label.lower() in cleaned_output: # Check in potentially less stripped version
            similarity = len(en_label)
            # Prioritize exact match found anywhere in the cleaned_output if it's one of the labels
            if cleaned_output == en_label.lower():
                best_match_en = en_label
                break
            if similarity > highest_similarity:
                best_match_en = en_label
                highest_similarity = similarity
    
    if best_match_en:
        return best_match_en # Return original casing
    else:
        # print(f"WARN: Could not reliably extract label. Output: '{output_text}'. Defaulting to random choice.")
        # return random.choice(expected_en_labels) # OLD: random fallback
        # print(f"WARN: Could not reliably extract label from '{output_text[:100]}...'. Returning '[Unknown Label]'.")
        return "[Unknown Label]" # NEW: Specific unknown label string

def process_classification_baseline(
    model: Any,
    tokenizer: Any,
    text: str,
    possible_labels: List[str], 
    lang_code: str,
    use_few_shot: bool,
    # Unified generation parameters (expected from runner script)
    temperature: float, 
    top_p: float, 
    top_k: int, 
    max_tokens: int,
    repetition_penalty: float, 
    do_sample: bool
) -> Tuple[str, float, str]:
    """
    Process a text for classification using the baseline approach.
    Accepts and uses unified generation parameters.
    """
    start_time = time.time()
    raw_model_output = ""
    
    # Determine prompt based on lang_code to meet new requirements:
    # - LRL instructions for LRL text, asking for English labels.
    # - English instructions for English text.
    if lang_code != 'en':
        prompt = generate_lrl_instruct_classification_prompt(
            text, lang_code, POSSIBLE_LABELS_EN, model_name="", use_few_shot=use_few_shot
        )
    else: # lang_code == 'en'
        prompt = generate_classification_prompt(
            text, POSSIBLE_LABELS_EN, lang_code, model_name="", use_few_shot=use_few_shot
        )

    # Determine appropriate max_length for tokenization
    # Use tokenizer.model_max_length if available and reasonable, else a default.
    # max_tokens is for the *output* generation, not input tokenization length.
    input_tokenize_max_length = getattr(tokenizer, 'model_max_length', 2048)
    if input_tokenize_max_length > 4096: # Cap at 4096 for safety if model_max_length is excessively large
        input_tokenize_max_length = 4096
        
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_tokenize_max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Determine if sampling should be used based on temperature and explicit do_sample
    # The `do_sample` parameter passed to this function should be the definitive source of truth.
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
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    possible_labels: List[str],
    use_few_shot: bool,
    # Unified generation parameters (expected from runner script)
    temperature: float, 
    top_p: float, 
    top_k: int, 
    max_tokens: int,
    repetition_penalty: float, 
    do_sample: bool
) -> pd.DataFrame:
    """
    Evaluate classification baseline approach on a dataset.
    Accepts unified generation parameters.
    """
    results = []
    shot_description = "few-shot" if use_few_shot else "zero-shot"
    prompt_lang_description = "LRL-instruct" if lang_code != 'en' else "EN-instruct"

    print(f"Evaluating {model_name} on {lang_code} ({prompt_lang_description}, {shot_description})...")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} Classification Baseline"):
        text = row['text']
        ground_truth_label = row['label'] 

        predicted_label, runtime, raw_output = process_classification_baseline(
            model,
            tokenizer,
            text,
            possible_labels,
            lang_code=lang_code,
            use_few_shot=use_few_shot,
            # Pass unified parameters directly
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens, # For the classification label output
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
