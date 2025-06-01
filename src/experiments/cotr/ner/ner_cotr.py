import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import json
import time
from typing import List, Dict, Any, Tuple, Optional
import sys
import argparse
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from evaluation module
from evaluation.cotr.translation_metrics import calculate_comet_score, COMET_AVAILABLE

# Global logger
logger = logging.getLogger(__name__)

# --- Constants ---
LANG_NAMES = {
    "en": "English",
    "sw": "Swahili",
    "ha": "Hausa",
    # Add other languages as needed for prompts if they differ from codes
}

# Standardized English entity types for the intermediate English NER step
ENGLISH_ENTITY_TYPES = sorted([
    "PERSON", "ORGANIZATION", "LOCATION", "DATE", 
    "PRODUCT", "EVENT", "MISC" 
])
ENGLISH_ENTITY_TYPES_STR = ", ".join(ENGLISH_ENTITY_TYPES)


# --- Utility Functions ---
def get_language_name(lang_code: str) -> str:
    """Get full language name from language code."""
    return LANG_NAMES.get(lang_code.lower(), lang_code.capitalize())

def _sanitize_for_prompt(text: str) -> str:
    """Basic sanitization for text included in prompts."""
    return str(text).replace("'''", "'''").replace('"""', '\"\"\"').replace('`', '\`')

# --- Model Initialization (Adapted from ner_cotr_old.py and classification_cotr.py) ---
def initialize_model(model_name: str, cache_path: Optional[str] = "/work/bbd6522/cache_dir") -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Initialize a model and tokenizer."""
    logger.info(f"Initializing model: {model_name} for NER CoTR...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path
    )

    try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto", # Leverage multiple GPUs if available
            trust_remote_code=True,
            cache_dir=cache_path,
            low_cpu_mem_usage=True 
        )
    except Exception as e_load_auto_fp16:
        logger.warning(f"Failed to load {model_name} with device_map='auto' and float16. Error: {e_load_auto_fp16}. Attempting fallback.")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_path
                # Not specifying dtype or device_map initially for broader compatibility in fallback
            )
    if torch.cuda.is_available():
                logger.info(f"Fallback: Manually moving {model_name} to CUDA and attempting float16 conversion.")
                try:
                    model = model.to("cuda").half()
                except Exception as e_cuda_half:
                    logger.error(f"Fallback: Failed to move to CUDA and convert to float16: {e_cuda_half}. Using full precision on CUDA if possible.")
                    model = model.to("cuda")
        except Exception as e_load_fallback_critical:
            logger.critical(f"CRITICAL: Fallback loading for {model_name} also failed: {e_load_fallback_critical}. Cannot proceed with this model.")
                 raise

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer.pad_token to eos_token ('{tokenizer.eos_token}') for {model_name}")
        else:
            new_pad_token = '[PAD]' # Default pad token
            tokenizer.add_special_tokens({'pad_token': new_pad_token})
            model.resize_token_embeddings(len(tokenizer)) # Important
            logger.warning(f"Added new pad_token '{new_pad_token}' and resized model embeddings for {model_name}.")

    if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.info(f"Aligned model.config.pad_token_id with tokenizer.pad_token_id ({tokenizer.pad_token_id}) for {model_name}")
    
    logger.info(f"Successfully loaded {model_name}. Tokenizer pad: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id}). Model pad_token_id: {model.config.pad_token_id}")
    logger.info(f"Model {model_name} is on device: {next(model.parameters()).device}")
    return tokenizer, model

# --- Prompt Generation Functions (English-instructed) ---
def generate_translation_prompt(text: str, source_lang: str, target_lang: str, is_entity_list: bool = False) -> str:
    """Generates an English-instructed prompt for translation (text or entity list)."""
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)
    safe_text = _sanitize_for_prompt(text)

    if is_entity_list: # Translating a JSON list of entities
        # Prompt from ner_cotr.py (current) is good for this, focuses on JSON structure
        # Ensuring instructions are clear about source/target language of entity TEXT.
        # The 'type' must always be preserved as English.
        if source_lang == 'en': # English entity texts to LRL entity texts
            return f"""Input English Named Entities (JSON list of dictionaries):
{safe_text}

Instructions:
1.  The input is a JSON list of dictionaries, representing English named entities with 'text' and 'type' keys.
2.  Translate ONLY the 'text' field of each English entity into {target_name}.
3.  The 'type' of each entity (e.g., {ENGLISH_ENTITY_TYPES_STR}) MUST remain unchanged.
4.  Format your output STRICTLY as a JSON list of dictionaries, identical in structure to the input.
5.  If an entity's text (e.g., a person's name that doesn't typically change, a specific brand, a numeric date) does not have a direct or common translation in {target_name}, return the original English entity text for that item.
6.  Output ONLY the translated JSON list. Do not add explanations.

{target_name} Named Entities (JSON list of dictionaries):"""
        else: # LRL entity texts to English entity texts
            return f"""Input {source_name} Named Entities (JSON list of dictionaries):
{safe_text}

Instructions:
1.  The input is a JSON list of dictionaries, representing {source_name} named entities.
2.  Translate ONLY the 'text' field of each {source_name} entity into English.
3.  The 'type' of each entity MUST remain unchanged (it should be an English type like {ENGLISH_ENTITY_TYPES_STR}).
4.  Format your output STRICTLY as a JSON list of dictionaries.
5.  Output ONLY the translated JSON list.

English Named Entities (JSON list of dictionaries):"""
    else: # Translating a block of text
        if target_lang == 'en': # LRL to English text
            # Adapted from ner_cotr_old.py generate_translation_prompt and ner_cotr.py's specific Sw/Ha prompts
            # Adding English few-shot examples
            example_lrl_sw = "Shirika la Umoja wa Mataifa lina makao yake makuu New York."
            example_en_sw = "The United Nations organization is headquartered in New York."
            example_lrl_ha = "Shugaba Bola Tinubu ya gana da gwamnoni a Abuja."
            example_en_ha = "President Bola Tinubu met with governors in Abuja."
            
            few_shot_text_trans = ""
            if source_lang == 'sw':
                few_shot_text_trans = f"Example Swahili Text: '{example_lrl_sw}'\nExample English Translation: '{example_en_sw}'"
            elif source_lang == 'ha':
                few_shot_text_trans = f"Example Hausa Text: '{example_lrl_ha}'\nExample English Translation: '{example_en_ha}'"

            return f"""Original Text ({source_name}):
'{safe_text}'

Instructions:
Translate the {source_name} text above to fluent and accurate English.
CRITICAL: Preserve all named entities (people, organizations, locations, dates) exactly as they appear or with their standard English equivalents.
Provide ONLY the English translation. Do not add any introductory text, labels, or explanations.

{few_shot_text_trans}

English Translation:"""
        else: # English to LRL text (less common for NER CoTR input, but needed for completeness)
            # Adapted from ner_cotr.py current
            examples = ""
            if target_lang == 'sw':
                examples = f"Example English Text: 'Microsoft announced earnings.'\nExample Swahili Translation: 'Microsoft ilitangaza mapato.'"
            elif target_lang == 'ha':
                examples = f"Example English Text: 'He visited Kano on Monday.'\nExample Hausa Translation: 'Ya ziyarci Kano ranar Litinin.'"
            return f"""Original Text (English):
'{safe_text}'

Instructions:
Translate the English text above to accurate {target_name}.
CRITICAL: Preserve named entities. Translate them appropriately into {target_name} if common translations exist; otherwise, retain their English form if they are proper nouns.
Provide ONLY the translated {target_name} text.

{examples}

{target_name} Translation:"""

def generate_ner_prompt_english(text_en: str, use_few_shot: bool = True) -> str:
    """Generates an English prompt for NER on English text, expecting JSON output. (Adapted from ner_cotr_old.py)"""
    safe_text_en = _sanitize_for_prompt(text_en)
    instructions = f"""Your task is to perform Named Entity Recognition (NER) on the English text provided.
Identify all named entities. For each entity, provide its exact text and its type.
The ONLY allowed entity types are: {ENGLISH_ENTITY_TYPES_STR}.
Format your output as a JSON list of dictionaries, where each dictionary has 'text' and 'type' keys.
Example: [{{"text": "Entity1", "type": "TYPE1"}}, {{"text": "Entity2", "type": "TYPE2"}}]
If no entities are found, return an empty list [].
Provide ONLY the JSON list. Do not add explanations.

Text:
'{safe_text_en}'
"""
    few_shot_examples = """
Examples:

Text:
'On January 1st, 2023, Alice from Google visited Berlin to attend the EuroConf about AI.'
Extracted Entities:
[{"text": "January 1st, 2023", "type": "DATE"}, {"text": "Alice", "type": "PERSON"}, {"text": "Google", "type": "ORGANIZATION"}, {"text": "Berlin", "type": "LOCATION"}, {"text": "EuroConf", "type": "EVENT"}, {"text": "AI", "type": "MISC"}]

Text:
'The new product, ZetaBeam, was launched by Innovatech last Monday.'
Extracted Entities:
[{"text": "ZetaBeam", "type": "PRODUCT"}, {"text": "Innovatech", "type": "ORGANIZATION"}, {"text": "last Monday", "type": "DATE"}]

Text:
'This sentence has no relevant entities.'
Extracted Entities:
[]
"""
    prompt = instructions
    if use_few_shot:
        prompt += few_shot_examples
    prompt += "\nExtracted Entities:"
    return prompt

def generate_single_prompt_ner_cotr(lrl_text: str, lang_code: str, use_few_shot: bool = True) -> str:
    """ English-instructed single prompt for full NER CoTR chain. (Adapted from ner_cotr_old.py) """
    original_lrl_name = get_language_name(lang_code)
    safe_lrl_text = _sanitize_for_prompt(lrl_text)

    # Main instructions - entirely in English
    prompt = f"""You are an AI assistant. Given a text in {original_lrl_name}, perform Named Entity Recognition by following these steps precisely:

1.  **Translate to English**: Translate the original {original_lrl_name} text accurately into English.
    Output this as:
    Translated English Text: [Your English Translation]

2.  **Extract English Entities**: From the translated English text (Step 1), identify all named entities.
    Use ONLY these entity types: {ENGLISH_ENTITY_TYPES_STR}.
    Output this as a JSON list of dictionaries (e.g., [{{"text": "EntityText", "type": "ENTITY_TYPE"}}]):
    Extracted English Entities: [Your JSON list of English entities]
    If no entities, output: Extracted English Entities: []

3.  **Translate Entities to {original_lrl_name}**: Translate the 'text' of each English entity from Step 2 into {original_lrl_name}.
    The 'type' of each entity MUST remain the same as its English counterpart from Step 2.
    If an entity's text (like a name or brand) has no common {original_lrl_name} translation, use the original English text.
    Output this as a JSON list of dictionaries:
    Translated {original_lrl_name} Entities: [Your JSON list of {original_lrl_name} entities with original English types]
    If Step 2 was [], output: Translated {original_lrl_name} Entities: []

Ensure each step's output is clearly labeled as shown. The final list of "Translated {original_lrl_name} Entities" is your ultimate result.

Original {original_lrl_name} Text:
'{safe_lrl_text}'
"""
    if use_few_shot:
        # English few-shot example (illustrative LRL text is also English for clarity of process)
        example_lrl_text_en = "Dr. Eve Smith from NeuroCorp presented on 'AlphaGo' in London on May 10, 2024."
        example_en_translation = example_lrl_text_en # Since example LRL text is English
        example_en_entities_json = json.dumps([
            {"text": "Eve Smith", "type": "PERSON"}, 
            {"text": "NeuroCorp", "type": "ORGANIZATION"},
            {"text": "AlphaGo", "type": "PRODUCT"}, # Or MISC depending on strictness
            {"text": "London", "type": "LOCATION"},
            {"text": "May 10, 2024", "type": "DATE"}
        ])
        # For the example, LRL entities are also shown in English for clarity, but with LRL name in text part
        example_lrl_entities_placeholder_json = json.dumps([
            {"text": f"Eve Smith ({original_lrl_name})", "type": "PERSON"},
            {"text": f"NeuroCorp ({original_lrl_name})", "type": "ORGANIZATION"},
            {"text": f"AlphaGo ({original_lrl_name})", "type": "PRODUCT"},
            {"text": f"London ({original_lrl_name})", "type": "LOCATION"},
            {"text": f"May 10, 2024 ({original_lrl_name})", "type": "DATE"}
        ])

        prompt += f"""
--- Example ---
Original {original_lrl_name} Text (Illustrative):
'{example_lrl_text_en}'

Translated English Text: {example_en_translation}
Extracted English Entities: {example_en_entities_json}
Translated {original_lrl_name} Entities: {example_lrl_entities_placeholder_json}
--- End Example ---
"""
    prompt += f"\nNow, generate all the above steps for the Original Text provided at the beginning:"
    return prompt


# --- Core Processing Functions ---
def clean_translation_response(response: str, target_lang: str, source_lang: str, is_entity_list: bool = False) -> str:
    """Clean the translation response. (Adapted from ner_cotr_old.py and ner_cotr.py)"""
    cleaned = response.strip()
    target_name = get_language_name(target_lang)

    if is_entity_list:
        # For JSON lists, be less aggressive with prefix removal, rely on JSON parsing
        # But remove common conversational model outputs if they appear before the list
        json_list_start = cleaned.find('[')
        json_list_end = cleaned.rfind(']')
        if json_list_start != -1 and json_list_end != -1 and json_list_end > json_list_start:
            cleaned = cleaned[json_list_start : json_list_end+1]
        else: # If no clear JSON list, just basic strip. Parsing will handle errors.
            logger.debug(f"No clear JSON list found in entity translation response for {target_lang}: {cleaned[:100]}...")
        return cleaned # Return potentially unparsed string, let parse_json_entity_list handle it
    else: # Text translation
        prefixes_to_remove = [
            f"{target_name} Translation:", f"Translation to {target_name}:",
            f"The {target_name} translation is:", "English Translation:",
            "Translated text:", f"Here is the {target_name} translation:",
            f"Original Text ({get_language_name(source_lang)}):" # If model repeats input
        ]
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        # Remove leading/trailing quotes that models sometimes add
        cleaned = cleaned.strip('"\'`) \t\n')
        return cleaned

def translate_text(
    model: Any, tokenizer: Any, text_to_translate: str,
    source_lang: str, target_lang: str,
    generation_params: Dict, # Pass all gen params (temp, top_p, max_new_tokens, etc.)
    is_entity_list: bool = False
) -> Tuple[str, float]: # Returns (translated_text, duration)
    """Translate text or entity list. (Adapted from ner_cotr_old.py and ner_cotr.py)"""
    start_time = time.time()
    if not text_to_translate or not text_to_translate.strip():
        logger.warning(f"translate_text received empty input for {source_lang}->{target_lang}.")
        return "", time.time() - start_time
    if source_lang == target_lang and not is_entity_list: # Text doesn't need translation
         return text_to_translate, time.time() - start_time
    # If source==target but is_entity_list, it might be a format conversion (e.g. from model's raw to JSON)
    # For CoTR, entity list translation always involves a language change or explicit processing.

    prompt = generate_translation_prompt(text_to_translate, source_lang, target_lang, is_entity_list)
    
    # Generation params are passed directly
    effective_temp = generation_params.get('temperature', 0.3)
    effective_do_sample = generation_params.get('do_sample', effective_temp > 1e-5)
    
    # Max input length for tokenizer, not for generation.
    max_input_length = generation_params.get("max_input_length", 2048) # Default if not in params

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    raw_model_output = "[Translation Error: Generation Failed]"
    try:
        with torch.no_grad(): 
            gen_kwargs_for_model = {
                "max_new_tokens": generation_params.get('max_new_tokens', 200 if is_entity_list else 512),
                "temperature": effective_temp if effective_do_sample else None, 
                "top_p": generation_params.get('top_p', 0.9) if effective_do_sample else None,
                "top_k": generation_params.get('top_k', 50) if effective_do_sample else None,
                "repetition_penalty": generation_params.get('repetition_penalty', 1.0),
                "do_sample": effective_do_sample,
                "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            }
            outputs = model.generate(inputs["input_ids"], **gen_kwargs_for_model)
        
        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        raw_model_output = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error during model.generate in translate_text ({source_lang}->{target_lang}): {e}", exc_info=True)
        raw_model_output = f"[Translation Error: {e}]"
        # Fall through to cleaning, error marker will be cleaned or remain.

    cleaned_translation = clean_translation_response(raw_model_output, target_lang, source_lang, is_entity_list)
    duration = time.time() - start_time
    logger.debug(f"Translate {source_lang}->{target_lang} ('{text_to_translate[:30]}...'): Raw='{raw_model_output[:50]}...', Cleaned='{cleaned_translation[:50]}...', Time={duration:.2f}s")
    return cleaned_translation, duration

def parse_json_entity_list(json_string: str, context: str = "") -> List[Dict[str, str]]:
    """Safely parses a JSON string expected to be a list of entity dictionaries. (Adapted from ner_cotr.py)"""
    entities = []
    if not json_string or not json_string.strip():
        logger.debug(f"Empty JSON string provided for parsing in context: {context}")
        return []

    try:
        # Attempt to find a valid JSON list within the string
        match = re.search(r'\[(.*?)\]', json_string, re.DOTALL)
        potential_json = json_string # Default to full string
        if match:
            potential_json = match.group(0) # Use the bracketed part
        
                parsed_list = json.loads(potential_json)
                if isinstance(parsed_list, list):
                    for item in parsed_list:
                        if isinstance(item, dict) and 'text' in item and 'type' in item:
                    entity_text = str(item['text']).strip()
                            entity_type_upper = str(item['type']).upper().strip()
                    
                    if not entity_text: # Skip entities with empty text
                        logger.warning(f"Empty entity text found in {context} from JSON: {item}. Skipping.")
                        continue

                    # Validate entity type against standard list, default to MISC if not recognized
                    if entity_type_upper not in ENGLISH_ENTITY_TYPES:
                        logger.warning(f"Unknown entity type '{item['type']}' (normalized to '{entity_type_upper}') found in {context}. Original item: {item}. Defaulting to 'MISC'.")
                        entity_type_upper = 'MISC'
                    entities.append({'text': entity_text, 'type': entity_type_upper})
                            else:
                    logger.warning(f"Invalid item format in JSON entity list for {context}: {item}. Expected dict with 'text' and 'type'.")
                        else:
            logger.warning(f"Parsed JSON for {context} is not a list: {type(parsed_list)}. String: {potential_json[:100]}...")
    except json.JSONDecodeError as e_json:
        logger.warning(f"JSONDecodeError parsing for {context}: '{e_json}'. String: {json_string[:150]}...")
    except Exception as e_outer:
        logger.error(f"Error parsing entity list string for {context} '{json_string[:150]}...': {e_outer}", exc_info=True)

    if not entities and json_string.strip() and json_string.strip() != "[]":
        logger.warning(f"Failed to parse any valid entities from non-empty/non-'[]' string for {context}: {json_string[:100]}...")
    return entities

def process_ner_english(
    model: Any, tokenizer: Any, text_en: str, use_few_shot: bool,
    generation_params: Dict # Pass all gen params (temp, top_p, max_new_tokens, etc.)
) -> Tuple[List[Dict[str, str]], float, str]: # Returns (entities, duration, raw_output)
    """Performs NER on English text, expects JSON output. (Adapted from ner_cotr_old.py)"""
    start_time = time.time()
    if not text_en or not text_en.strip():
        return [], time.time() - start_time, ""

    prompt = generate_ner_prompt_english(text_en, use_few_shot)
    
    effective_temp = generation_params.get('temperature', 0.2) # Lower temp for structured JSON
    effective_do_sample = generation_params.get('do_sample', effective_temp > 1e-5)
    max_input_length = generation_params.get("max_input_length", 2048)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    raw_model_output_ner = "[NER Error: Generation Failed]"
    try:
        with torch.no_grad():
            gen_kwargs_for_model = {
                "max_new_tokens": generation_params.get('max_new_tokens', 300), # Max tokens for entity list
                "temperature": effective_temp if effective_do_sample else None,
                "top_p": generation_params.get('top_p', 0.85) if effective_do_sample else None,
                "top_k": generation_params.get('top_k', 40) if effective_do_sample else None,
                "repetition_penalty": generation_params.get('repetition_penalty', 1.1),
                "do_sample": effective_do_sample,
                "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            }
            outputs = model.generate(inputs["input_ids"], **gen_kwargs_for_model)
        
        input_len = inputs["input_ids"].shape[1]
        raw_model_output_ner = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    except Exception as e:
        logger.error(f"Error during model.generate in process_ner_english: {e}", exc_info=True)
        raw_model_output_ner = f"[NER Error: {e}]"

    entities = parse_json_entity_list(raw_model_output_ner, context=f"English NER for '{text_en[:30]}...'")
    duration = time.time() - start_time
    logger.debug(f"English NER for '{text_en[:30]}...': Parsed={entities}, Raw='{raw_model_output_ner[:50]}...', Time={duration:.2f}s")
    return entities, duration, raw_model_output_ner

def translate_entities_to_lrl(
    model: Any, tokenizer: Any,
    english_entities: List[Dict[str, str]], # [{'text': '...', 'type': '...'}]
    target_lrl_code: str,
    generation_params: Dict # Pass all relevant gen params
) -> Tuple[List[Dict[str, str]], float, str]: # (lrl_entities, duration, raw_translation_output)
    """Translates 'text' of English entities to LRL, keeps English 'type'. (Adapted from ner_cotr_old.py)"""
    start_time = time.time()
    if not english_entities:
        return [], time.time() - start_time, ""
    if target_lrl_code == 'en': # No translation needed if target is English
        return english_entities, time.time() - start_time, json.dumps(english_entities)

    entities_json_str = json.dumps(english_entities)
    
    # Max new tokens for entity list translation should be generous
    # It's passed within generation_params as 'max_new_tokens' specific to this translation call
    trans_gen_params = generation_params.copy() # Ensure we don't modify the original dict if it's reused
    if 'max_new_tokens' not in trans_gen_params: # Default if not specified for this step
        trans_gen_params['max_new_tokens'] = max(200, len(english_entities) * 20) # Heuristic

    translated_lrl_entities_str, trans_duration = translate_text(
        model, tokenizer, entities_json_str,
        source_lang='en', target_lang=target_lrl_code,
        generation_params=trans_gen_params, # Pass specific params for this step
        is_entity_list=True
    )

    final_lrl_entities_parsed = []
    if "[Translation Error:" not in translated_lrl_entities_str:
        parsed_from_translation = parse_json_entity_list(translated_lrl_entities_str, context=f"Entity list translation to {target_lrl_code}")
        
        # Critical: Ensure types are from original English entities, and counts match
        if len(parsed_from_translation) == len(english_entities):
            for i, lrl_ent_dict in enumerate(parsed_from_translation):
                final_lrl_entities_parsed.append({
                    'text': lrl_ent_dict.get('text', english_entities[i]['text']), # Fallback text
                    'type': english_entities[i]['type'] # MUST use original English type
                })
    else:
            logger.warning(f"Mismatch in entity count after EN->{target_lrl_code} translation. Original: {len(english_entities)}, Parsed from translation: {len(parsed_from_translation)}. Raw translated string: '{translated_lrl_entities_str[:100]}...'. Falling back to original English entities for this sample's LRL output.")
            # Fallback: return original English entities but mark them to indicate translation failure or use placeholders
            final_lrl_entities_parsed = [{'text': f"[TransFail] {e['text']}", 'type': e['type']} for e in english_entities]
    else: # Translation itself failed with an error string
        logger.error(f"Translate_text returned an error for EN->{target_lrl_code} entity list: {translated_lrl_entities_str}")
        final_lrl_entities_parsed = [{'text': f"[TransError] {e['text']}", 'type': e['type']} for e in english_entities]

    full_duration = time.time() - start_time # This includes the translate_text duration
    return final_lrl_entities_parsed, full_duration, translated_lrl_entities_str


def extract_parts_from_single_prompt(
    response_text: str, lang_code: str
) -> Tuple[Optional[str], Optional[List[Dict[str, str]]], Optional[List[Dict[str, str]]]]:
    """ Extracts parts from single-prompt CoTR NER output. (Adapted from ner_cotr_old.py version) """
    lrl_name = get_language_name(lang_code)
    logger.debug(f"SP Extract: Raw response for {lang_code} (first 300 chars):\n{response_text[:300]}")

    translated_en_text, extracted_eng_entities, final_lrl_entities = None, None, None

    # Regex to capture content after labels, robust to newlines and minor variations.
    # We assume English labels for sections as per English-instructed prompts.
    def get_section_content(label: str, text: str, is_json_list: bool = False) -> Optional[str]:
        # Pattern: Optional "Step X:", optional "##", the label, optional separator, then content.
        # Stops at the next likely section header or end of text.
        header_pattern = rf"(?:(?:Step\s*\d+\s*[:\\-–—]?\s*)?(?:##\s*)?{re.escape(label)}\s*[:\\-–—]?\s*)"
        
        # For JSON, specifically try to grab content within '[' and ']'
        if is_json_list:
            content_pattern = r"(\[.*?\])" # Non-greedy capture within brackets
            full_pattern = rf"{header_pattern}{content_pattern}"
            match = re.search(full_pattern, text, re.IGNORECASE | re.DOTALL)
            if match: return match.group(1).strip()
            # Fallback if no explicit JSON list found after header, try to get text until next header
            # This might happen if model fails to produce JSON list but gives text.
            
        # For general text or fallback for JSON if brackets not found right after header
        # Stop before next potential step label like "Step", "Extracted", "Translated", or end of string
        # The negative lookahead (?!...) ensures we don't cross into the next structured section.
        content_pattern_general = r"(.*?)(?=\s*(?:Step\s*\d|Extracted\s+English\s+Entities|Translated\s+\w+\s+Entities)|$)"
        full_pattern_general = rf"{header_pattern}{content_pattern_general}"
        match_general = re.search(full_pattern_general, text, re.IGNORECASE | re.DOTALL)
        if match_general:
            return match_general.group(1).strip().strip("'\"") # Clean outer quotes
        return None

    # 1. Translated English Text
    en_text_content = get_section_content("Translated English Text", response_text)
    if en_text_content:
        translated_en_text = en_text_content
        logger.debug(f"SP Extract - Found EN Text: {translated_en_text[:100]}...")
    else:
        logger.warning(f"SP Extract - Could not find 'Translated English Text' for {lang_code}.")

    # 2. Extracted English Entities
    eng_entities_json_str = get_section_content("Extracted English Entities", response_text, is_json_list=True)
    if eng_entities_json_str:
        logger.debug(f"SP Extract - Found EN Entities JSON string: {eng_entities_json_str[:100]}...")
        extracted_eng_entities = parse_json_entity_list(eng_entities_json_str, context=f"SP English Entities for {lang_code}")
    else: # Fallback if no JSON list directly found, try to get text and parse later
        eng_entities_text_content = get_section_content("Extracted English Entities", response_text, is_json_list=False)
        if eng_entities_text_content:
            logger.debug(f"SP Extract - Found EN Entities (non-JSON) text: {eng_entities_text_content[:100]}...")
            extracted_eng_entities = parse_json_entity_list(eng_entities_text_content, context=f"SP English Entities (fallback) for {lang_code}")
    else:
            logger.warning(f"SP Extract - Could not find 'Extracted English Entities' for {lang_code}.")
            
    # 3. Translated LRL Entities
    lrl_entities_label = f"Translated {lrl_name} Entities"
    lrl_entities_json_str = get_section_content(lrl_entities_label, response_text, is_json_list=True)
    if lrl_entities_json_str:
        logger.debug(f"SP Extract - Found LRL Entities JSON string: {lrl_entities_json_str[:100]}...")
        final_lrl_entities = parse_json_entity_list(lrl_entities_json_str, context=f"SP {lrl_name} Entities for {lang_code}")
    else: # Fallback for LRL entities
        lrl_entities_text_content = get_section_content(lrl_entities_label, response_text, is_json_list=False)
        if lrl_entities_text_content:
            logger.debug(f"SP Extract - Found LRL Entities (non-JSON) text: {lrl_entities_text_content[:100]}...")
            final_lrl_entities = parse_json_entity_list(lrl_entities_text_content, context=f"SP {lrl_name} Entities (fallback) for {lang_code}")
    else:
            logger.warning(f"SP Extract - Could not find '{lrl_entities_label}' for {lang_code}.")

    return translated_en_text, extracted_eng_entities, final_lrl_entities


# --- Main Evaluation Functions ---
def evaluate_ner_cotr_multi_prompt( # Renamed from evaluate_ner_cotr
    model: Any, tokenizer: Any, samples_df: pd.DataFrame, lang_code: str, use_few_shot: bool,
    # NER step specific gen params
    ner_generation_params: Dict,
    # Translation step specific gen params (used for both LRL->EN text and EN->LRL entities)
    translation_generation_params: Dict,
    model_name: Optional[str] = None # For logging
) -> pd.DataFrame:
    """ Multi-Prompt NER CoTR. (Adapted from ner_cotr_old.py and ner_cotr.py) """
    results_list = []
    logger.info(f"Starting Multi-Prompt NER CoTR for {lang_code} with {model_name or 'UnknownModel'} ({'few-shot' if use_few_shot else 'zero-shot'}).")
    # Log generation parameters being used
    logger.debug(f"  NER Gen Params: {ner_generation_params}")
    logger.debug(f"  Translation Gen Params: {translation_generation_params}")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Multi-Prompt NER {lang_code}"):
        original_lrl_text = str(row['text'])
        # Ground truth entities are expected to be lists of dicts: [{'text': '...', 'type': '...'}]
        # The types in ground truth are LRL types from MasakhaNER (e.g., "PER", "ORG", "LOC", "DATE").
        # These will be compared against our LRL predictions (which will have English types).
        ground_truth_lrl_entities_raw = row.get('entities', []) 
        
        # Ensure GT entities are in the expected list-of-dicts format for consistent processing
        if isinstance(ground_truth_lrl_entities_raw, str):
            try:
                ground_truth_lrl_entities = json.loads(ground_truth_lrl_entities_raw)
                if not isinstance(ground_truth_lrl_entities, list): ground_truth_lrl_entities = []
            except json.JSONDecodeError:
                ground_truth_lrl_entities = [] # If GT is a bad string, treat as no entities
        elif isinstance(ground_truth_lrl_entities_raw, list):
            ground_truth_lrl_entities = ground_truth_lrl_entities_raw
        else:
            ground_truth_lrl_entities = []

        # 1. Translate LRL text to English
        translated_english_text, time_text_trans, raw_text_trans_output = "", 0.0, ""
        if lang_code != 'en':
            translated_english_text, time_text_trans = translate_text(
            model, tokenizer, original_lrl_text, lang_code, "en",
                generation_params=translation_generation_params, # Use dedicated translation params
            is_entity_list=False
        )
            raw_text_trans_output = translated_english_text # translate_text now returns cleaned
        else: # Input is already English
            translated_english_text = original_lrl_text
        
        comet_lrl_text_to_en = None
        if COMET_AVAILABLE and lang_code != 'en' and original_lrl_text and translated_english_text and "[Translation Error:" not in translated_english_text:
            try:
                score_data = calculate_comet_score(sources=[original_lrl_text], predictions=[translated_english_text])
                comet_lrl_text_to_en = score_data['mean_score'] if isinstance(score_data, dict) else (score_data if isinstance(score_data, (float, int)) else None)
            except Exception as e_comet_text:
                logger.warning(f"COMET LRL Text->EN Error (MP, sample {row.get('id', idx)}): {e_comet_text}")
        
        # 2. Perform NER on translated English text (expects English types)
        extracted_english_entities, time_en_ner, raw_en_ner_output = process_ner_english(
            model, tokenizer, translated_english_text, use_few_shot,
            generation_params=ner_generation_params # Use dedicated NER params
        )
            
        # 3. Translate extracted English entities' text back to LRL, keeping English types
        predicted_lrl_entities_final, time_ent_trans, raw_lrl_ent_trans_output = [], 0.0, "[]"
        comet_en_entity_to_lrl = None

        if extracted_english_entities and lang_code != 'en':
            predicted_lrl_entities_final, time_ent_trans, raw_lrl_ent_trans_output = translate_entities_to_lrl(
                model, tokenizer, extracted_english_entities, lang_code,
                generation_params=translation_generation_params # Use dedicated translation params
            )
            if COMET_AVAILABLE and extracted_english_entities and predicted_lrl_entities_final:
                # For COMET: sources are EN entity texts, predictions are LRL entity texts
                # References could be LRL GT entity texts if aligned, or self-reference for quality check
                # Here, we check quality of EN entity text -> LRL entity text translation
                # We need to be careful about comparing lists of entities.
                # Let's average COMET scores for individual entity text translations if possible.
                # This requires aligning english_entities with predicted_lrl_entities_final.
                # Assuming they are aligned by translate_entities_to_lrl:
                comet_scores_ents = []
                if len(extracted_english_entities) == len(predicted_lrl_entities_final):
                    for i in range(len(extracted_english_entities)):
                        en_txt = extracted_english_entities[i]['text']
                        lrl_txt = predicted_lrl_entities_final[i]['text']
                        if en_txt and lrl_txt and "[TransFail]" not in lrl_txt and "[TransError]" not in lrl_txt:
                            try:
                                score_data = calculate_comet_score(sources=[en_txt], predictions=[lrl_txt])
                                if isinstance(score_data, dict) and 'mean_score' in score_data: comet_scores_ents.append(score_data['mean_score'])
                                elif isinstance(score_data, (float, int)): comet_scores_ents.append(score_data)
                            except Exception as e_comet_ent:
                                logger.warning(f"COMET EN Ent->LRL Ent Error (MP, sample {row.get('id', idx)}, entity '{en_txt}'): {e_comet_ent}")
                if comet_scores_ents: comet_en_entity_to_lrl = np.mean(comet_scores_ents)
        elif lang_code == 'en': # If original lang is English, EN entities are the final LRL entities
            predicted_lrl_entities_final = extracted_english_entities
            raw_lrl_ent_trans_output = json.dumps(predicted_lrl_entities_final) # Store as JSON
            # No LRL entity translation COMET score if lang_code is 'en'
        
        results_list.append({
            'id': row.get('id', idx),
            'original_lrl_text': original_lrl_text,
            'ground_truth_lrl_entities': json.dumps(ground_truth_lrl_entities), # Store LRL GT
            'intermediate_english_text': translated_english_text,
            'intermediate_english_entities': json.dumps(extracted_english_entities),
            'final_predicted_lrl_entities': json.dumps(predicted_lrl_entities_final), # This is compared to GT
            'comet_lrl_text_to_en': comet_lrl_text_to_en,
            'comet_en_entity_text_to_lrl': comet_en_entity_to_lrl,
            'time_lrl_text_to_en_translation_sec': time_text_trans,
            'time_english_ner_sec': time_en_ner,
            'time_en_entity_to_lrl_translation_sec': time_ent_trans,
            'raw_lrl_to_en_text_translation_output': raw_text_trans_output if lang_code != 'en' else "N/A",
            'raw_english_ner_output': raw_en_ner_output,
            'raw_en_to_lrl_entity_translation_output': raw_lrl_ent_trans_output if lang_code != 'en' else "N/A",
            'language': lang_code, 'pipeline_type': 'multi_prompt', 
            'shot_setting': 'few-shot' if use_few_shot else 'zero-shot'
        })
    return pd.DataFrame(results_list)

def evaluate_ner_cotr_single_prompt(
    model: Any, tokenizer: Any, samples_df: pd.DataFrame, lang_code: str, use_few_shot: bool,
    # Generation parameters for the entire single prompt chain
    chain_generation_params: Dict,
    model_name: Optional[str] = None # For logging
) -> pd.DataFrame:
    """ Single-Prompt NER CoTR. (Adapted from ner_cotr_old.py and ner_cotr.py) """
    results_list = []
    logger.info(f"Starting Single-Prompt NER CoTR for {lang_code} with {model_name or 'UnknownModel'} ({'few-shot' if use_few_shot else 'zero-shot'}).")
    logger.debug(f"  Single Chain Gen Params: {chain_generation_params}")

    effective_temp = chain_generation_params.get('temperature', 0.2)
    effective_do_sample = chain_generation_params.get('do_sample', effective_temp > 1e-5)
    max_input_length = chain_generation_params.get("max_input_length", 2048)

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Single-Prompt NER {lang_code}"):
        original_lrl_text = str(row['text'])
        ground_truth_lrl_entities_raw = row.get('entities', [])
        if isinstance(ground_truth_lrl_entities_raw, str):
            try: ground_truth_lrl_entities = json.loads(ground_truth_lrl_entities_raw)
            except: ground_truth_lrl_entities = []
        elif isinstance(ground_truth_lrl_entities_raw, list):
            ground_truth_lrl_entities = ground_truth_lrl_entities_raw
        else: ground_truth_lrl_entities = []


        prompt = generate_single_prompt_ner_cotr(original_lrl_text, lang_code, use_few_shot)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        raw_model_output = "[SP NER Error: Generation Failed]"
        time_generation = 0.0
        start_gen_time = time.time()
        try:
            with torch.no_grad():
                gen_kwargs_for_model = {
                    "max_new_tokens": chain_generation_params.get('max_new_tokens', 512), # For the whole chain
                    "temperature": effective_temp if effective_do_sample else None,
                    "top_p": chain_generation_params.get('top_p', 0.85) if effective_do_sample else None,
                    "top_k": chain_generation_params.get('top_k', 40) if effective_do_sample else None,
                    "repetition_penalty": chain_generation_params.get('repetition_penalty', 1.1),
                    "do_sample": effective_do_sample,
                    "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                }
                outputs = model.generate(inputs["input_ids"], **gen_kwargs_for_model)
            
            input_len = inputs["input_ids"].shape[1]
            raw_model_output = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            time_generation = time.time() - start_gen_time
        except Exception as e_gen:
            logger.error(f"Error in SP generation (sample {row.get('id', idx)}): {e_gen}", exc_info=True)
            raw_model_output = f"[SP NER Error: {e_gen}]"
            time_generation = time.time() - start_gen_time

        # Extract all parts from the structured response
        inter_en_text, inter_en_entities, final_lrl_entities_pred = extract_parts_from_single_prompt(
            raw_model_output, lang_code
        )
        
        comet_lrl_text_to_en_sp, comet_en_entity_to_lrl_sp = None, None
        if COMET_AVAILABLE:
            if lang_code != 'en' and original_lrl_text and inter_en_text and "[Extraction Failed]" not in inter_en_text:
                try:
                    score_data = calculate_comet_score(sources=[original_lrl_text], predictions=[inter_en_text])
                    comet_lrl_text_to_en_sp = score_data['mean_score'] if isinstance(score_data, dict) else (score_data if isinstance(score_data, (float,int)) else None)
                except Exception as e_comet_text_sp: logger.warning(f"COMET LRL Text->EN Error (SP, sample {row.get('id', idx)}): {e_comet_text_sp}")

            if lang_code != 'en' and inter_en_entities and final_lrl_entities_pred:
                comet_scores_ents_sp = []
                if len(inter_en_entities) == len(final_lrl_entities_pred):
                    for i in range(len(inter_en_entities)):
                        en_txt = inter_en_entities[i]['text']
                        lrl_txt = final_lrl_entities_pred[i]['text']
                        if en_txt and lrl_txt and "[TransFail]" not in lrl_txt and "[TransError]" not in lrl_txt:
                            try:
                                score_data = calculate_comet_score(sources=[en_txt], predictions=[lrl_txt])
                                if isinstance(score_data, dict) and 'mean_score' in score_data: comet_scores_ents_sp.append(score_data['mean_score'])
                                elif isinstance(score_data, (float,int)): comet_scores_ents_sp.append(score_data)
                            except Exception as e_comet_ent_sp: logger.warning(f"COMET EN Ent->LRL Ent Error (SP, sample {row.get('id', idx)}, entity '{en_txt}'): {e_comet_ent_sp}")
                if comet_scores_ents_sp: comet_en_entity_to_lrl_sp = np.mean(comet_scores_ents_sp)

        results_list.append({
            'id': row.get('id', idx),
            'original_lrl_text': original_lrl_text,
            'ground_truth_lrl_entities': json.dumps(ground_truth_lrl_entities),
            'intermediate_english_text_model': inter_en_text if inter_en_text else "[Extraction Failed]",
            'intermediate_english_entities_model': json.dumps(inter_en_entities) if inter_en_entities else "[]",
            'final_predicted_lrl_entities': json.dumps(final_lrl_entities_pred) if final_lrl_entities_pred else "[]",
            'comet_lrl_text_to_en_sp': comet_lrl_text_to_en_sp,
            'comet_en_entity_text_to_lrl_sp': comet_en_entity_to_lrl_sp,
            'raw_model_output_single_prompt': raw_model_output,
            'time_generation_sec': time_generation,
            'language': lang_code, 'pipeline_type': 'single_prompt',
            'shot_setting': 'few-shot' if use_few_shot else 'zero-shot'
        })
    return pd.DataFrame(results_list)


# --- Metrics Calculation (Example - can be expanded or moved to run script) ---
def calculate_ner_metrics_for_sample(
    ground_truth_entities: List[Dict[str, str]], # LRL ground truth entities
    predicted_entities: List[Dict[str, str]]    # LRL predicted entities (with English types)
) -> Dict[str, float]:
    """Calculates P, R, F1 for a single sample. Types are matched case-insensitively (original GT types vs. EN types from prediction). Texts are exact match after strip."""
    
    # Normalize GT: text, TYPE_FROM_GT (e.g. "PER", "ORG" from MasakhaNER)
    gt_set = set()
    if isinstance(ground_truth_entities, list):
        for ent in ground_truth_entities:
            if isinstance(ent, dict) and 'text' in ent and 'type' in ent:
                # For GT, we use its original type string, but uppercase it for consistency.
                # The comparison will be between this GT type and the English type from prediction.
                # This means metrics are strict: predicted type must match GT's original type name.
                gt_set.add((str(ent['text']).strip(), str(ent['type']).upper()))
            # Add elif for tuple if GT can be in that format

    # Normalize Predictions: text, ENGLISH_TYPE (e.g. "PERSON", "ORGANIZATION")
    pred_set = set()
    if isinstance(predicted_entities, list):
        for ent in predicted_entities:
            if isinstance(ent, dict) and 'text' in ent and 'type' in ent:
                # Prediction types are already English and standardized (e.g., "PERSON")
                pred_set.add((str(ent['text']).strip(), str(ent['type']).upper()))
            # Add elif for tuple

    true_positives = len(gt_set.intersection(pred_set))
    
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = true_positives / len(gt_set) if len(gt_set) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Placeholder main for direct testing of this script (usually run via run_ner_cotr.py)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    logger.info("NER CoTR script direct execution: Testing prompts and parsing.")

    # Example: Test Single Prompt Generation & Extraction
    sample_sw_text = "Rais Samia Suluhu Hassan wa Tanzania alitembelea Nairobi Januari 2023."
    sp_prompt_sw = generate_single_prompt_ner_cotr(sample_sw_text, 'sw', use_few_shot=True)
    logger.debug(f"Generated Single Prompt (Swahili, Few-Shot):\n{sp_prompt_sw}")

    # Mock model response for single prompt
    mock_sp_response_sw = f"""
Translated English Text: President Samia Suluhu Hassan of Tanzania visited Nairobi in January 2023.
Extracted English Entities: [{{"text": "Samia Suluhu Hassan", "type": "PERSON"}}, {{"text": "Tanzania", "type": "LOCATION"}}, {{"text": "Nairobi", "type": "LOCATION"}}, {{"text": "January 2023", "type": "DATE"}}]
Translated Swahili Entities: [{{"text": "Samia Suluhu Hassan", "type": "PERSON"}}, {{"text": "Tanzania", "type": "LOCATION"}}, {{"text": "Nairobi", "type": "LOCATION"}}, {{"text": "Januari 2023", "type": "DATE"}}]
"""
    en_text, en_ents, lrl_ents = extract_parts_from_single_prompt(mock_sp_response_sw, 'sw')
    logger.info(f"SP Extracted EN Text: {en_text}")
    logger.info(f"SP Extracted EN Entities: {en_ents}")
    logger.info(f"SP Extracted LRL Entities: {lrl_ents}")

    # Example: Test Translation Prompt for Entity List (EN to LRL)
    en_entity_list_json = json.dumps([{"text": "Google", "type": "ORGANIZATION"}, {"text": "New York", "type": "LOCATION"}])
    trans_prompt_en_to_ha_list = generate_translation_prompt(en_entity_list_json, 'en', 'ha', is_entity_list=True)
    logger.debug(f"Generated EN->HA Entity List Translation Prompt:\n{trans_prompt_en_to_ha_list}")
    
    logger.info("NER CoTR script direct execution test complete.")

def main():
    parser = argparse.ArgumentParser(description="Run NER CoTR (Translate-Process-Translate) experiments using local models")
    parser.add_argument("--models", nargs='+', default=["Qwen/Qwen2.5-7B-Instruct"],
                        help="List of model names (Hugging Face ID)")
    parser.add_argument("--langs", nargs='+', default=['sw', 'ha'],
                        help="Languages to test (e.g., sw ha yo)")
    parser.add_argument("--pipeline_types", nargs='+', default=['multi_prompt', 'single_prompt'],
                        help="CoTR pipeline types to run (e.g., multi_prompt single_prompt)")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'],
                        help="Shot settings to evaluate (zero_shot, few_shot)")
    parser.add_argument("--samples", type=int, default=20,
                        help="Max samples per language")
    parser.add_argument("--output", type=str, default="/work/bbd6522/results/ner/cotr_local_translate_pipeline",
                        help="Output directory")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    # Add generation params (keep relevant ones for translation/NER)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_ner_new_tokens", type=int, default=200, help="Max new tokens for English NER step")
    parser.add_argument("--max_translate_new_tokens", type=int, default=512, help="Max new tokens for translation steps")
    # Keep other relevant generation params if needed by the underlying functions
    # parser.add_argument("--temperature", type=float, default=0.1) # These might be set within functions
    # parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'results'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'metrics'), exist_ok=True) # Keep metrics dir
    os.makedirs(os.path.join(args.output, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'summaries'), exist_ok=True)

    # --- Run Experiments --- #
    all_overall_results = [] # To store summaries from all model runs

    for model_name_str in args.models:
        print(f"\n{'='*20} Evaluating Model: {model_name_str} {'='*20}")
        try:
            tokenizer, model = initialize_model(model_name_str)
        except Exception as e:
            print(f"FATAL: Failed to initialize model {model_name_str}. Skipping this model. Error: {e}")
            continue # Skip to the next model

        model_language_results = [] # Results for the current model across all its configs

        for lang_code in args.langs:
            print(f"\n======= Processing Language: {lang_code} for model {model_name_str} ======")
            # Load data once per language
            samples_df_full = load_masakhaner_samples(lang_code, num_samples=None, split=args.split, seed=args.seed)
            if samples_df_full.empty:
                print(f"WARNING: No samples for {lang_code}, skipping.")
                continue # Correctly placed inside if

            actual_loaded = len(samples_df_full)
            if args.samples is not None and actual_loaded > args.samples:
                current_samples_df = samples_df_full.sample(n=args.samples, random_state=args.seed)
            else:
                current_samples_df = samples_df_full
            print(f"Using {len(current_samples_df)} samples for {lang_code}.")

            if 'tokens' not in current_samples_df.columns or 'entities' not in current_samples_df.columns:
                print(f"ERROR: Loaded data for {lang_code} is missing 'tokens' or 'entities'. Skipping.")
                continue # Correctly placed inside if

            for pipeline_type in args.pipeline_types:
                for shot_setting in args.shot_settings:
                    use_few_shot = (shot_setting == 'few_shot')
                    print(f"\n--- Running: {lang_code}, {pipeline_type}, {shot_setting} ({model_name_str}) ---")
                    experiment_start_time = time.time()

                    # Call the refactored evaluate_ner_cotr function
                    try: # Add try block for evaluation robustness
                        lang_results_df = evaluate_ner_cotr(
                            model_name=model_name_str, # Pass model_name for logging/internal use
                            tokenizer=tokenizer,
                            model=model,
                            samples_df=current_samples_df,
                            lang_code=lang_code,
                            temperature=args.temperature if hasattr(args, 'temperature') else 0.3, # Provide defaults if not in args
                            top_p=args.top_p if hasattr(args, 'top_p') else 0.9,
                            top_k=args.top_k if hasattr(args, 'top_k') else 50,
                            max_tokens=args.max_ner_new_tokens, # Ensure this arg exists or use a default
                            repetition_penalty=args.repetition_penalty if hasattr(args, 'repetition_penalty') else 1.2, # Ensure this arg exists or use a default
                            pipeline_type=pipeline_type,
                            use_few_shot=use_few_shot,
                            max_translation_tokens=args.max_translate_new_tokens, # Ensure this arg exists or use a default
                            num_beams=args.num_beams if hasattr(args, 'num_beams') else 1
                        )
                    except Exception as eval_e:
                        print(f"ERROR during evaluation call for {lang_code}, {pipeline_type}, {shot_setting}: {eval_e}")
                        lang_results_df = pd.DataFrame() # Create empty DF on error

                    total_runtime = time.time() - experiment_start_time

                    if lang_results_df.empty:
                        print(f"WARNING: No results for {lang_code}, {pipeline_type}, {shot_setting} with {model_name_str}. Skipping save.")
                        continue # Correctly placed inside if, skipping to next shot_setting

                    # Calculate average metrics
                    avg_f1 = lang_results_df["f1"].mean()
                    avg_precision = lang_results_df["precision"].mean()
                    avg_recall = lang_results_df["recall"].mean()

                    # Save detailed results for this specific configuration
                    model_path_name = model_name_str.replace("/", "_")

                    # Create specific output subdirectories
                    output_subdir = os.path.join(args.output, pipeline_type, shot_setting, lang_code, model_path_name)
                    os.makedirs(output_subdir, exist_ok=True)
                    results_filename = f"cotr_results.csv"
                    results_path = os.path.join(output_subdir, results_filename)
                    lang_results_df.to_csv(results_path, index=False)
                    print(f"Detailed results for {model_name_str} ({lang_code}, {pipeline_type}, {shot_setting}) saved to {results_path}")

                    # Store summary metrics for this configuration
                    summary_metrics = {
                        'model': model_name_str,
                        'language': lang_code,
                        'pipeline_type': pipeline_type,
                        'shot_setting': shot_setting,
                        'f1': avg_f1,
                        'precision': avg_precision,
                        'recall': avg_recall,
                        'runtime_seconds': total_runtime,
                        'samples_processed': len(lang_results_df)
                    }
                    model_language_results.append(summary_metrics)
                    all_overall_results.append(summary_metrics)
            # END of shot_setting loop
        # END of pipeline_type loop
    # END of lang_code loop

    # Clean up model after all its experiments are done
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Model {model_name_str} unloaded.")
    # END of model_name_str loop

    # --- Aggregate and Save Overall Summary --- # (DEDENTED to be after model_name_str loop)
    if all_overall_results:
        summary_df = pd.DataFrame(all_overall_results)
        # Overall summary file name can be simpler or include all models if desired
        overall_summary_path = os.path.join(args.output, 'summaries', f'ner_cotr_overall_summary.csv')
        os.makedirs(os.path.dirname(overall_summary_path), exist_ok=True)
        summary_df.to_csv(overall_summary_path, index=False)
        print(f"\n=== Overall CoTR NER Summary ===")
        print(summary_df.to_string())
        # Optional: Plotting similar to baseline script if needed
    else:
        print("\nNo CoTR NER experiments were successfully completed to summarize.")

    # Remove final cleanup, as it's done per model
    # print("Model unloaded.") # Already done in the loop

if __name__ == "__main__":
    main() 