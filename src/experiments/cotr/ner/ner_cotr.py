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
import ast

# Set up logging
logger = logging.getLogger(__name__)

# Add project root to Python path BEFORE other project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path: # Optional: prevent adding multiple times
    sys.path.insert(0, project_root)

# Import shared functionality from baseline
from src.experiments.baseline.ner.ner_baseline import (
    initialize_model, 
    extract_entities, 
    calculate_ner_metrics,
    load_masakhaner_samples,
    create_dummy_ner_data
)

from src.evaluation.cotr.translation_metrics import calculate_comet_score

# Remove the old block that added project root to Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
# sys.path.insert(0, project_root)

# Import utility functions
from src.utils.data_loaders.load_masakhaner import load_masakhaner_samples
# Remove OpenAI specific import
# from src.utils.llm_utils import call_llm_api

# Import COMET if available
from evaluation.cotr.translation_metrics import COMET_AVAILABLE 

# Baseline imports for alignment
from src.experiments.baseline.ner.ner_baseline import extract_entities as extract_entities_baseline # Import baseline extraction
from src.experiments.baseline.ner.ner_baseline import generate_ner_prompt as generate_ner_prompt_baseline # Import baseline EN prompt

# Define language names dictionary (ensure it includes all needed langs)
lang_names = {
    "en": "English",
    "sw": "Swahili",
    "ha": "Hausa",
    # Add other languages if needed by translation prompts
}

# Define NER entity types (consistent with baseline)
ENTITY_TYPES = ["PER", "ORG", "LOC", "DATE"]

def generate_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """
    Generate a prompt for translation with clear structure and entity preservation focus.
    
    Args:
        text: The text to translate
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        Formatted translation prompt
    """
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)
    
    # Enhanced prompts focusing on preserving NER-relevant info (LRL -> EN)
    if target_lang == 'en':
        example_lrl_text = ""
        example_en_text = ""
        if source_lang == 'sw':
            example_lrl_text = "Mfano: 'Rais Samia Suluhu Hassan alienda Dodoma jana.'"
            example_en_text = "Translation: 'President Samia Suluhu Hassan went to Dodoma yesterday.'"
            return f"""Translate this Swahili text to fluent English. 
Original Text (Swahili):
'{text}'

Instructions:
- Translate directly and accurately.
- CRITICAL: Preserve all named entities (people, organizations, locations, dates) exactly as they appear or with their standard English equivalents. For example, if a name is 'Juma Mussa', it should remain 'Juma Mussa' in English unless there is a very common English version.
- Provide ONLY the English translation, without any introductory text, labels, or explanations.

{example_lrl_text}
{example_en_text}

English Translation:"""
        elif source_lang == 'ha':
            example_lrl_text = "Misali: 'Gwamna Bala Mohammed ya ziyarci Abuja ranar Litinin.'"
            example_en_text = "Translation: 'Governor Bala Mohammed visited Abuja on Monday.'"
            return f"""Translate this Hausa text to accurate English.
Original Text (Hausa):
'{text}'

Instructions:
- Translate directly and accurately.
- CRITICAL: Ensure all named entities (persons, organizations, locations, dates) are maintained correctly in the translation. For example, if a name is 'Aisha Bello', it should remain 'Aisha Bello' in English.
- Provide ONLY the English translation, without any introductory text, labels, or explanations.

{example_lrl_text}
{example_en_text}

English Translation:"""
    
    # --- CRITICAL IMPROVEMENT: EN -> LRL Back-Translation for Entities --- 
    # Focus on translating English *entities* back accurately for NER evaluation
    if source_lang == 'en':
        # Check if the input looks like a list of entity tags. This helps customize the prompt slightly.
        # This regex looks for one or more lines each starting with [TYPE: entity text]
        is_entity_list_format = bool(re.fullmatch(r'(\[(?:PER|ORG|LOC|DATE): [^\n]+\][\n]?)+', text.strip()))

        instruction_intro = f"Translate the following English text/entities to {target_name}."
        if is_entity_list_format:
            instruction_detail = f"IMPORTANT: The input is a list of tagged entities. For each line like '[TYPE: English Entity Text]', translate ONLY the 'English Entity Text' part into {target_name}. Keep the '[TYPE: ]' tag structure identical in your output. Each translated entity should be on a new line."
        else:
            instruction_detail = f"If the input is a general text, translate it to {target_name}. If it looks like a tagged entity like '[TYPE: Entity Text]', translate only the 'Entity Text' part to {target_name} and keep the '[TYPE: ]' structure."

        examples = ""
        if target_lang == 'sw':
            examples = f"""Examples for translation to Swahili:
Input: [LOC: New York City]
Output: [LOC: Jiji la New York]

Input: [PER: President Joe Biden]
Output: [PER: Rais Joe Biden]

Input: [ORG: Google]
Output: [ORG: Google]

Input: [DATE: January 1st, 2023]
Output: [DATE: Januari 1, 2023]

Input: Microsoft announced earnings.
Output: Microsoft ilitangaza mapato."""
        elif target_lang == 'ha':
            examples = f"""Examples for translation to Hausa:
Input: [ORG: Dangote Group]
Output: [ORG: Dangote Group]

Input: [PER: Governor Abba Yusuf]
Output: [PER: Gwamna Abba Yusuf]

Input: [LOC: Washington D.C.]
Output: [LOC: Washington D.C.]

Input: [DATE: Next Monday]
Output: [DATE: Litinin mai zuwa]

Input: He visited Kano on Monday.
Output: Ya ziyarci Kano ranar Litinin."""
            
        return f"""{instruction_intro}
{instruction_detail}

{examples}

Input Text/Entities (English):
'{text}'

{target_name} Translation:"""
    
    # Default prompt for other pairs (e.g., LRL -> LRL if ever needed, or unhandled LRL -> EN)
    # This will likely be less effective than the specialized ones above.
    return f"""Original Text ({source_name}):
'{text}'

Instructions:
Translate this {source_name} text to {target_name} accurately.
Pay close attention to named entities.
Provide only the direct translation, without any introductory text.

{target_name} Translation:"""

def get_language_name(lang_code: str) -> str:
    """Convert language code to full language name."""
    language_map = {
        "en": "English",
        "sw": "Swahili",
        "ha": "Hausa"
    }
    return language_map.get(lang_code, lang_code)

def clean_translation_response(response: str, target_lang: str, source_lang: str) -> str:
    """
    Clean up a translation response from the model.
    
    Args:
        response: Raw response from the model
        target_lang: Target language code
        source_lang: Source language code
        
    Returns:
        Cleaned translation
    """
    # Check for empty response
    if not response or response.strip() == "":
        return ""
    
    # Normalize whitespace
    response = response.strip()
    
    # Remove common prefixes that models might add
    prefixes_to_remove = [
        "Translation:", "Translated text:", "Here's the translation:",
        "The translation is:", "In English:", "In Swahili:", "In Hausa:",
        "English translation:", "Swahili translation:", "Hausa translation:",
        "Translated:", "Answer:", "Response:"
    ]
    
    for prefix in prefixes_to_remove:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    
    # Language-specific cleaning
    if target_lang == "sw":
        # Swahili - remove explanations sometimes added after translations
        lines = response.split("\n")
        for i, line in enumerate(lines):
            # Look for transitions that might indicate a shift to explanation
            if i > 0 and any(marker in line.lower() for marker in ["note:", "explanation:", "literal:"]):
                response = "\n".join(lines[:i]).strip()
                break
    
    elif target_lang == "ha":
        # Hausa - similar to Swahili
        lines = response.split("\n")
        for i, line in enumerate(lines):
            if i > 0 and any(marker in line.lower() for marker in ["note:", "bayani:", "fassara:"]):
                response = "\n".join(lines[:i]).strip()
                break
    
    # Remove quotes that might be added around the translation
    response = response.strip('"\'')
    
    return response

def translate_text(
    model: Any,
    tokenizer: Any,
    text: str,
    source_lang: str,
    target_lang: str,
    generation_params: Dict[str, Any],
    max_input_length: int = 4096,
) -> str:
    """Translate text using a unified generation parameter dictionary."""
    # Special handling for same-language "translation" (no-op)
    if source_lang == target_lang:
        return text
        
    # Handle empty text case
    if not text or text.strip() == "":
        return ""

    # Generate translation prompt
    prompt = generate_translation_prompt(text, source_lang, target_lang)
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Use the passed-in generation parameters directly.
    gen_params = generation_params.copy()
    gen_params.setdefault("pad_token_id", tokenizer.eos_token_id)

    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_params
        )
    
    # Decode only the newly generated tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Clean up the response
    translation = clean_translation_response(response, target_lang, source_lang)
    
    return translation

def generate_ner_prompt_english(text: str, model_name: str = "") -> str:
    """
    Generates a prompt for English NER.
    """
    # Adding a one-shot example to guide the model better.
    example_text = "Microsoft, a company founded by Bill Gates, is based in Redmond."
    example_json = '''```json
{
  "entities": [
    {
      "entity_text": "Microsoft",
      "entity_type": "ORG"
    },
    {
      "entity_text": "Bill Gates",
      "entity_type": "PER"
    },
    {
      "entity_text": "Redmond",
      "entity_type": "LOC"
    }
  ]
}
```'''
    prompt = f"""You are an expert entity extraction system.
Extract all named entities (PER, ORG, LOC, DATE) from the text.
Provide your answer as a single JSON object with a key "entities" that contains a list of objects.
Do not provide any explanation or other text. Only the JSON object.

--- Example ---
Text: "{example_text}"

JSON:
{example_json}
---
Text: "{text}"

JSON:
"""
    return prompt

def process_ner_english(
    model: Any,
    tokenizer: Any,
    text: str,
    generation_params: Dict[str, Any],
    max_input_length: int = 4096,
    model_name: str = ""
) -> List[Dict[str, str]]:
    """
    Process a text for NER in English using a unified generation parameter dictionary.
    """
    try:
        # Generate the NER prompt in English
        prompt = generate_ner_prompt_english(text, model_name)
        
        # Tokenize with truncation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Use the passed-in generation parameters directly.
        gen_params = generation_params.copy()
        gen_params.setdefault("pad_token_id", tokenizer.eos_token_id)
        
        # Generate with optimized parameters
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **gen_params
            )
        
        # Decode only the newly generated tokens
        raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Use more robust regex to find either a markdown JSON block or a raw JSON object.
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```|(\{[\s\S]*\})", raw_output)

        if json_match:
            # json_match.group(1) will be the markdown block, group(2) will be the raw JSON.
            json_str = json_match.group(1) if json_match.group(1) else json_match.group(2)
            json_str = json_str.strip()
            try:
                # Use ast.literal_eval for safer and more robust parsing
                data = ast.literal_eval(json_str)
                if isinstance(data, dict) and "entities" in data:
                    entities = data.get("entities", [])
                    # Convert entity_text/entity_type to entity/type format for consistency
                    converted_entities = []
                    for e in entities:
                        if isinstance(e, dict):
                            converted_entity = {}
                            if "entity_text" in e:
                                converted_entity["entity"] = e["entity_text"]
                            elif "text" in e:
                                converted_entity["entity"] = e["text"]
                            
                            if "entity_type" in e:
                                converted_entity["type"] = e["entity_type"]
                            elif "type" in e:
                                converted_entity["type"] = e["type"]
                            
                            if "entity" in converted_entity and "type" in converted_entity:
                                converted_entities.append(converted_entity)
                    return converted_entities
            except (ValueError, SyntaxError):
                try:
                    # Fallback to json.loads for pure JSON
                    data = json.loads(json_str)
                    if isinstance(data, dict) and "entities" in data:
                        entities = data.get("entities", [])
                        # Convert entity_text/entity_type to entity/type format for consistency
                        converted_entities = []
                        for e in entities:
                            if isinstance(e, dict):
                                converted_entity = {}
                                if "entity_text" in e:
                                    converted_entity["entity"] = e["entity_text"]
                                elif "text" in e:
                                    converted_entity["entity"] = e["text"]
                                
                                if "entity_type" in e:
                                    converted_entity["type"] = e["entity_type"]
                                elif "type" in e:
                                    converted_entity["type"] = e["type"]
                                
                                if "entity" in converted_entity and "type" in converted_entity:
                                    converted_entities.append(converted_entity)
                        return converted_entities
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from English NER step: {e}. Raw string: '{json_str[:100]}...'")
                    return []

        logger.warning("No JSON block found in English NER response.")
        return []
    
    except Exception as e:
        import traceback
        print(f"Error in process_ner_english: {str(e)}")
        traceback.print_exc()
        return []

def translate_entities_to_original_language(
    model: Any,
    tokenizer: Any,
    entities: List[Dict[str, str]],
    source_lang: str, # This will be 'en'
    target_lang: str, # The LRL
    text_orig: str,
    generation_params: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Translates a list of English entities back to the original language in a single batch.
    """
    if not entities:
        return []

    target_lang_name = get_language_name(target_lang)
    
    # Create a JSON string of the English entities to be translated.
    entities_json_str = json.dumps({"entities": entities}, indent=2)

    prompt = f"""You are a translation expert. Your task is to translate the 'entity_text' values in the following JSON object from English to {target_lang_name}.
- Keep the JSON structure and entity types exactly the same.
- Use the original {target_lang_name} text as context to ensure the translations are accurate.
- Provide only the translated JSON object as your response.

Original Context ({target_lang_name}):
"{text_orig}"

Translate the 'entity_text' in this JSON from English to {target_lang_name}:
```json
{entities_json_str}
```

Translated JSON:
"""
    
    # Use the translate_text function for the API call
    # Note: translate_text is designed for single string translation, but we can adapt its use here.
    # The prompt contains the full instruction. We expect a JSON string back.
    
    # We pass 'repetition_penalty' and other params via generation_params
    translated_json_str = translate_text(
        model=model,
        tokenizer=tokenizer,
        text=prompt, # The full prompt is the "text" to process
        source_lang=source_lang, # Conceptually 'en'
        target_lang=target_lang, # Conceptually the LRL
        generation_params=generation_params
    )

    # Now, parse the returned JSON string
    translated_entities = []
    if translated_json_str and translated_json_str.strip():
        # Find the JSON block in the response
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```|(\{[\s\S]*\})", translated_json_str)
        if json_match:
            json_str = json_match.group(1) if json_match.group(1) else json_match.group(2)
            json_str = json_str.strip()
            try:
                data = json.loads(json_str)
                entities = data.get("entities", [])
                # Convert entity_text/entity_type to entity/type format for consistency
                translated_entities = []
                for e in entities:
                    if isinstance(e, dict):
                        converted_entity = {}
                        if "entity_text" in e:
                            converted_entity["entity"] = e["entity_text"]
                        elif "text" in e:
                            converted_entity["entity"] = e["text"]
                        elif "entity" in e:
                            converted_entity["entity"] = e["entity"]
                        
                        if "entity_type" in e:
                            converted_entity["type"] = e["entity_type"]
                        elif "type" in e:
                            converted_entity["type"] = e["type"]
                        
                        if "entity" in converted_entity and "type" in converted_entity:
                            translated_entities.append(converted_entity)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON from batched entity translation response: {json_str[:100]}...")
                # As a fallback, maybe the model just returned the list?
                try:
                    data = json.loads(f'{{"entities": {json_str}}}')
                    entities = data.get("entities", [])
                    # Convert entity_text/entity_type to entity/type format for consistency
                    translated_entities = []
                    for e in entities:
                        if isinstance(e, dict):
                            converted_entity = {}
                            if "entity_text" in e:
                                converted_entity["entity"] = e["entity_text"]
                            elif "text" in e:
                                converted_entity["entity"] = e["text"]
                            elif "entity" in e:
                                converted_entity["entity"] = e["entity"]
                            
                            if "entity_type" in e:
                                converted_entity["type"] = e["entity_type"]
                            elif "type" in e:
                                converted_entity["type"] = e["type"]
                            
                            if "entity" in converted_entity and "type" in converted_entity:
                                translated_entities.append(converted_entity)
                except json.JSONDecodeError as e:
                    logger.error(f"Ultimate fallback for entity translation parsing also failed: {e}")
                    return [] # Return empty if parsing fails completely
    
    return translated_entities

def evaluate_ner_cotr(
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    use_few_shot: bool,
    translation_params: Dict[str, Any],
    ner_params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Evaluate NER using a multi-prompt CoTR approach.
    """
    results = []
    
    for idx, row in tqdm(samples_df.iterrows(), desc=f"Processing {lang_code} samples ({model_name} CoTR Multi-Prompt)"):
        text = row['text']
        # Ensure ground truth entities are in the correct list-of-dicts format
        ground_truth_entities_raw = row.get('entities', [])
        if isinstance(ground_truth_entities_raw, str):
            try:
                ground_truth_entities = ast.literal_eval(ground_truth_entities_raw)
            except (ValueError, SyntaxError):
                ground_truth_entities = [] # Default to empty if parsing fails
        else:
            ground_truth_entities = ground_truth_entities_raw

        result_row = {
            "text": text,
            "ground_truth_entities": ground_truth_entities,
            "text_en": "",
            "entities_en": [],
            "predicted_entities": [],
            "error": None,
            "runtime_seconds": 0.0,
            "comet_score_text": None,
            "comet_score_entities": None,
        }
        start_time = time.time()

        try:
            # 1. Translate LRL text to English
            text_en = translate_text(
                model=model,
                tokenizer=tokenizer,
                text=text,
                source_lang=lang_code,
                target_lang="en",
                generation_params=translation_params,
            )
            result_row["text_en"] = text_en

            # 2. Process NER on English text
            entities_en = process_ner_english(
                model=model,
                tokenizer=tokenizer,
                text=text_en,
                generation_params=ner_params,
                model_name=model_name,
            )
            result_row["entities_en"] = entities_en

            # 3. Translate entities back to LRL
            predicted_entities = translate_entities_to_original_language(
                model=model,
                tokenizer=tokenizer,
                entities=entities_en,
                source_lang="en",
                target_lang=lang_code,
                text_orig=text,
                generation_params=translation_params
            )
            result_row["predicted_entities"] = predicted_entities
            
            # 4. (Optional) Calculate COMET scores if available and successful
            if COMET_AVAILABLE:
                # Score for text translation
                if text and text_en:
                    result_row["comet_score_text"] = calculate_comet_score([text], [text_en])
                
                # Score for entity back-translation (batched)
                if entities_en and predicted_entities and len(entities_en) == len(predicted_entities):
                    source_texts = [str(e.get('entity', e.get('entity_text', ''))) for e in entities_en]
                    pred_texts = [str(p.get('entity', p.get('entity_text', ''))) for p in predicted_entities]
                    if all(source_texts) and all(pred_texts):
                         result_row["comet_score_entities"] = calculate_comet_score(source_texts, pred_texts)

        except Exception as e:
            logger.error(f"ERROR processing sample {row.get('id', idx)} for {lang_code} (multi_prompt, {'few' if use_few_shot else 'zero'}-shot): {e}", exc_info=True)
            result_row["error"] = str(e)

        result_row["runtime_seconds"] = time.time() - start_time
        results.append(result_row)
        
    return pd.DataFrame(results)

def calculate_ner_metrics_for_sample(ground_truth, predictions):
    """
    Calculate precision, recall, and F1 score for a single sample.
    
    Args:
        ground_truth: List of ground truth entities
        predictions: List of predicted entities
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    # Convert to sets of (entity, type) tuples for comparison
    gt_set = {(e["entity"].lower(), e["type"]) for e in ground_truth}
    pred_set = {(e["entity"].lower(), e["type"]) for e in predictions}
    
    # Calculate true positives, false positives, false negatives
    true_positives = len(gt_set.intersection(pred_set))
    false_positives = len(pred_set) - true_positives
    false_negatives = len(gt_set) - true_positives
    
    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

# --- Experiment Runner --- #

# Comment out or remove the previous run_ner_cotr_experiment function
# def run_ner_cotr_experiment(...): ...

def generate_single_prompt_ner_cotr(text_lrl: str, lang_code: str, use_few_shot: bool = True) -> str:
    """Generates a single comprehensive prompt for NER CoTR, instructing the model to output a JSON list."""
    lrl_name = get_language_name(lang_code)
    safe_lrl_text = text_lrl.replace('`', "'").replace('"', "'") # More robust escaping
    
    # The prompt will be more direct and explicit about the expected output structure
    prompt_instruction = f"""You are an expert multilingual entity extraction system. Extract named entities (PER, ORG, LOC, DATE) from the given text in {lrl_name}.

OUTPUT FORMAT REQUIREMENTS:
1. Your response must be ONLY a single JSON object with this exact structure:
   {{
     "entities": [
       {{"entity_text": "extracted entity", "entity_type": "entity category"}}
     ]
   }}
2. Use ONLY "entity_text" and "entity_type" as the field names.
3. For entity_type, use only: PER (person), ORG (organization), LOC (location), or DATE (date/time).
4. Do not include markdown, explanations, or any text before or after the JSON.
5. If no entities are found, return an empty entities array: {{"entities": []}}
"""

    few_shot_example_str = ""
    if use_few_shot:
        example_lrl_text = "Rais wa zamani wa Marekani, Barack Obama, alizuru Kenya Jumanne."
        # This is the ONLY output the model should generate. No reasoning text.
        example_lrl_entities_json = """{
  "entities": [
    {"entity_text": "Barack Obama", "entity_type": "PER"},
    {"entity_text": "Marekani", "entity_type": "LOC"},
    {"entity_text": "Kenya", "entity_type": "LOC"},
    {"entity_text": "Jumanne", "entity_type": "DATE"}
  ]
}"""
        few_shot_example_str = f"""
--- Example ---
Text: "{example_lrl_text}"

JSON Response (exactly as shown, no additional text):
{example_lrl_entities_json}
--- End Example ---
"""

    final_prompt = f"""{prompt_instruction}
{few_shot_example_str}

--- Your Task ---
Text: "{safe_lrl_text}"

JSON Output (ONLY return this JSON, with NO other text):
"""
    return final_prompt

def extract_lrl_entities_from_single_prompt(response_text: str, lang_code: str) -> Tuple[str, List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Extracts entities from the single-prompt NER CoTR response.
    This version is updated for a direct-to-JSON prompt and has more robust parsing.
    """
    response_text = response_text.strip()
    
    # With the new prompt, the entire response should be the JSON object.
    # The concept of "intermediate english text" from the reasoning is no longer applicable here.
    intermediate_english_text = ""
    english_entities = []
    lrl_entities = []

    # Before anything else, try to strip everything after the first closing bracket of a complete JSON object
    # This handles cases where the model outputs JSON followed by commentary or repeats the JSON
    first_json_pattern = r'(\{[\s\S]*?\}\s*)(?:\s*The JSON Response:|\s*```|$)'
    first_json_match = re.search(first_json_pattern, response_text)
    if first_json_match:
        response_text = first_json_match.group(1).strip()

    # Attempt to find a JSON object, either in a markdown block or as the whole response.
    json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*\})", response_text, re.DOTALL)
    
    if not json_match:
        # If no JSON structure is found at all, log a warning and return empty.
        logger.warning("Could not find a JSON structure in the response. The model did not follow instructions.")
        return intermediate_english_text, english_entities, lrl_entities

    # Prioritize the first capture group (markdown block) if it exists, otherwise use the second (raw object).
    json_str = json_match.group(1) if json_match.group(1) else json_match.group(2)
    json_str = json_str.strip()

    # --- Robust JSON cleanup ---
    # 1. Remove trailing commas from lists and objects, which is a common model error.
    json_str = re.sub(r",\s*(\]|\})", r"\1", json_str)
    # 2. Ensure quotes are standard double quotes.
    json_str = json_str.replace("'", '"')
    # 3. Fix common key name inconsistencies - map "type" to "entity_type" and "value"/"text" to "entity_text"
    json_str = json_str.replace('"type":', '"entity_type":')
    json_str = json_str.replace('"text":', '"entity_text":')
    json_str = json_str.replace('"value":', '"entity_text":')

    # Define mapping of entity types to standard forms
    valid_entity_types = {"PER", "ORG", "LOC", "DATE"}
    entity_type_mapping = {
        "PERSON": "PER",
        "PEOPLE": "PER",
        "PERSONAL": "PER",
        "PERSONNEL": "PER",
        "ORGANIZATION": "ORG",
        "ORGANISATIONS": "ORG",
        "ORGANISATIONS": "ORG",
        "COMPANY": "ORG",
        "LOCATION": "LOC",
        "PLACE": "LOC",
        "PLACES": "LOC",
        "DATETIME": "DATE",
        "TIME": "DATE",
        "DATES": "DATE",
        "TIMES": "DATE",
        "EVENT": "DATE"  # Sometimes models miscategorize events as dates
    }

    try:
        data = json.loads(json_str)
        if isinstance(data, dict) and "entities" in data:
            # Basic validation of the entities list
            raw_entities = data.get("entities", [])
            if isinstance(raw_entities, list):
                # Filter for entities that are dicts with the required keys or keys that can be mapped
                valid_entities = []
                for e in raw_entities:
                    if not isinstance(e, dict):
                        continue
                        
                    # Create a normalized entity with standard field names
                    normalized_entity = {}
                    
                    # Handle various field name possibilities
                    if "entity_text" in e:
                        normalized_entity["entity"] = e["entity_text"]
                    elif "text" in e:
                        normalized_entity["entity"] = e["text"]
                    elif "value" in e:
                        normalized_entity["entity"] = e["value"]
                    else:
                        # Skip entities without required text field
                        continue
                        
                    if "entity_type" in e:
                        entity_type = e["entity_type"].upper()
                    elif "type" in e:
                        entity_type = e["type"].upper()
                    else:
                        # Skip entities without a type field
                        continue
                    
                    # Normalize entity type to standard values
                    if entity_type in valid_entity_types:
                        normalized_entity["type"] = entity_type
                    elif entity_type in entity_type_mapping:
                        normalized_entity["type"] = entity_type_mapping[entity_type]
                    else:
                        # Try to find a partial match
                        matched = False
                        for valid_type in valid_entity_types:
                            if valid_type in entity_type:
                                normalized_entity["type"] = valid_type
                                matched = True
                                break
                        if not matched:
                            # Skip entities without a valid type
                            continue
                        
                    # Skip empty entities
                    if not normalized_entity["entity"] or not normalized_entity["type"]:
                        continue
                        
                    valid_entities.append(normalized_entity)
                
                lrl_entities = valid_entities
                if len(lrl_entities) != len(raw_entities):
                    logger.warning("Some elements in the parsed 'entities' list were malformed.")
            else:
                logger.warning(f"Parsed 'entities' key is not a list. Found type: {type(raw_entities)}")
        else:
            logger.warning("Parsed JSON does not contain the 'entities' key.")
    except json.JSONDecodeError as e:
        logger.warning(f"Final JSON decoding attempt failed: {e}. Cleaned string was: '{json_str[:200]}...'")
        # Fallback parsing using regex to extract entity_text/entity_type pairs
        # This regex handles both "entity_text"/"entity_type" and "text"/"type" patterns
        fallback_patterns = [
            r'"entity_text"\s*:\s*"([^"]+)"\s*,\s*"entity_type"\s*:\s*"([^"]+)"',
            r'"text"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"',
            r'"value"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"',
            r'"type"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"([^"]+)"',
            r'"type"\s*:\s*"([^"]+)"\s*,\s*"value"\s*:\s*"([^"]+)"',
            r'"type"\s*:\s*"([^"]+)"\s*,\s*"entity_text"\s*:\s*"([^"]+)"',
        ]
        
        all_matches = []
        for pattern in fallback_patterns:
            matches = re.findall(pattern, json_str)
            for match in matches:
                if pattern.startswith(r'"entity_text"') or pattern.startswith(r'"text"') or pattern.startswith(r'"value"'):
                    # Pattern starts with text field
                    text = match[0]
                    entity_type = match[1].upper()
                else:
                    # Pattern starts with type field, so swap the order
                    text = match[1]
                    entity_type = match[0].upper()
                
                # Skip empty entities
                if not text or not entity_type:
                    continue
                
                # Normalize entity type to standard values
                if entity_type in valid_entity_types:
                    normalized_type = entity_type
                elif entity_type in entity_type_mapping:
                    normalized_type = entity_type_mapping[entity_type]
                else:
                    # Try to find a partial match
                    matched = False
                    for valid_type in valid_entity_types:
                        if valid_type in entity_type:
                            normalized_type = valid_type
                            matched = True
                            break
                    if not matched:
                        # Use PER as fallback for unknown types
                        normalized_type = "PER"
                
                all_matches.append((text, normalized_type))
        
        if all_matches:
            lrl_entities = [{"entity": text, "type": type_} for text, type_ in all_matches]
            logger.info(f"Used regex fallback to extract {len(lrl_entities)} entities from malformed JSON.")

    return intermediate_english_text, english_entities, lrl_entities

def evaluate_ner_cotr_single_prompt(
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    use_few_shot: bool = True,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 40,
    max_tokens: int = 512, # Needs to be large enough for the whole structured output
    repetition_penalty: float = 1.1, # Add missing param
    num_beams: int = 1
) -> pd.DataFrame:
    """
    Evaluate NER using a single CoTR prompt.
    """
    results = []
    model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
    safe_max_input_length = min(model_max_len, 8192) 

    # For more controlled and predictable outputs
    gen_temperature = min(0.2, temperature) if num_beams == 1 else temperature
    gen_top_p = min(0.85, top_p) if num_beams == 1 else top_p
    gen_top_k = min(30, top_k) if num_beams == 1 else top_k
    gen_repetition_penalty = max(1.1, repetition_penalty)
    
    # Apply model specific adjustments if needed (consistent with multi-prompt/baseline)
    if "aya" in model_name.lower():
        gen_temperature = max(0.1, gen_temperature * 0.9)
        # Use beam search for Aya to get more consistent outputs
        num_beams = max(2, num_beams)
    elif "qwen" in model_name.lower():
        gen_top_p = max(0.7, gen_top_p * 0.9)
        gen_top_k = 30
        # Use beam search for Qwen to get more consistent outputs
        num_beams = max(2, num_beams)

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} samples ({model_name} CoTR Single Prompt)"):
        text_lrl = row['text']
        
        # Define ground truth and runtime vars outside the try block
        ground_truth_entities = []
        start_time = time.time()
        
        # Safely evaluate ground truth entities
        raw_gt = row.get('entities', '[]')
        if isinstance(raw_gt, str):
            try:
                ground_truth_entities = ast.literal_eval(raw_gt)
            except (ValueError, SyntaxError):
                logger.warning(f"Could not parse ground truth entities for sample {idx}: {raw_gt}")
                ground_truth_entities = []
        elif isinstance(raw_gt, list):
            ground_truth_entities = raw_gt

        try:
            prompt = generate_single_prompt_ner_cotr(text_lrl, lang_code, use_few_shot)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=safe_max_input_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Set up generation arguments with focus on structured output
            generation_args = {
                "max_new_tokens": max_tokens,
                "repetition_penalty": gen_repetition_penalty,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            if num_beams > 1:
                # Use beam search for more structured outputs
                generation_args["num_beams"] = num_beams
                generation_args["do_sample"] = False
                # Add early stopping to avoid trailing text
                generation_args["early_stopping"] = True
            else:
                # Sample-based generation
                generation_args["temperature"] = gen_temperature
                generation_args["top_p"] = gen_top_p
                generation_args["top_k"] = gen_top_k
                generation_args["do_sample"] = True

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    **generation_args
                )
            
            response_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            
            # Extract parts from the single prompt response
            intermediate_en_text, intermediate_en_entities, predicted_entities = \
                extract_lrl_entities_from_single_prompt(response_text, lang_code)

            runtime = time.time() - start_time
            
            results.append({
                "text": text_lrl,
                "ground_truth_entities": ground_truth_entities,
                "predicted_entities": predicted_entities,
                "intermediate_en_text": intermediate_en_text,
                "intermediate_en_entities": intermediate_en_entities,
                "raw_model_response": response_text,
                "error": None,
                "runtime_seconds": runtime
            })

        except Exception as e:
            logger.error(f"ERROR processing sample {idx} (Single Prompt NER): {e}", exc_info=True)
            runtime = time.time() - start_time
            # Append error row
            results.append({
                "text": text_lrl,
                "ground_truth_entities": ground_truth_entities, # Now this is safely defined
                "predicted_entities": [],
                "intermediate_en_text": "",
                "intermediate_en_entities": [],
                "raw_model_response": "",
                "error": str(e),
                "runtime_seconds": runtime
            })

    results_df = pd.DataFrame(results)
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Run NER CoTR experiments.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use (e.g., 'CohereForAI/c4ai-command-r-plus').")
    # ... existing code ...

    args = parser.parse_args()
    # ... existing code ...

    # ... rest of the function ...

if __name__ == "__main__":
    main() 