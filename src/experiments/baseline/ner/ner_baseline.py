import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import json
from typing import Dict, List, Any, Tuple
import time
import logging

ENTITY_TYPES = ["PER", "ORG", "LOC", "DATE"]

def initialize_model(model_name: str) -> Tuple[Any, Any]:
    """
    Initialize a model and tokenizer.
    
    Args:
        model_name: Name of the model to initialize
    
    Returns:
        tokenizer, model
    """
    print(f"Initializing {model_name}...")
    cache_path = "/work/bbd6522/cache_dir" # Define cache path
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        cache_dir=cache_path
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        cache_dir=cache_path
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

def generate_ner_prompt(text: str, lang_code: str = "en", model_name: str = "", use_few_shot: bool = True) -> str:
    """
    Generate an NER prompt with English instructions.
    Focuses on the [TYPE: Entity Text] format for output.
    """
    processed_text = text.strip()
    entity_types_desc = "PER (Person), ORG (Organization), LOC (Location), DATE (Date)"
    no_entities_marker = "[NO_ENTITIES_FOUND]"

    instruction = (
        f"Your task is to identify named entities in the text below for the language '{lang_code}'. "
        f"Extract all entities corresponding to {entity_types_desc}. "
        f"For each entity, provide its type and text. "
        f"Present your ENTIRE output as a list of entities, each in the format [TYPE: Entity Text], separated by spaces or newlines. "
        f"Example of format: [PER: John Doe] [LOC: Paris] [DATE: July 2024]. "
        f"If no entities are found, output the exact phrase: {no_entities_marker}."
    )

    prompt = instruction
    
    if use_few_shot:
        prompt += "\n\nExamples (ensure your output strictly follows this format for the actual task):\n"
        # English examples using the new format
        examples_en = [
            {"text": "Angela Merkel visited Paris on September 1st, 2021 with a delegation from the European Union.", 
             "entities_str": "[PER: Angela Merkel] [LOC: Paris] [DATE: September 1st, 2021] [ORG: European Union]"},
            {"text": "The quick brown fox jumps over the lazy dog in New York.", 
             "entities_str": "[LOC: New York]"},
            {"text": "There are no entities here.",
             "entities_str": no_entities_marker}
        ]
        for ex in examples_en:
            prompt += f"\nText ({lang_code}): '{ex['text']}'\nEntities: {ex['entities_str']}"
    else: # Zero-shot
        prompt += f"\n\nNow, analyze the following text and provide the entities in the specified format ([TYPE: Entity Text] or {no_entities_marker}):"

    prompt += f"\n\nText ({lang_code}): '{processed_text}'\n\nEntities:" # Ensure 'Entities:' is the final cue
    return prompt

# This function needs to be defined or imported if used by generate_ner_prompt
# For now, assuming it's not critical or handled elsewhere.
# def preprocess_text(text: str, lang_code: str) -> str:
#     return text.strip()

def generate_lrl_instruct_ner_prompt(text: str, lang_code: str = "sw", model_name: str = "", use_few_shot: bool = True) -> str:
    """
    Generate an NER prompt with LRL instructions (Swahili, Hausa).
    Focuses on the [TYPE: Entity Text] format for output.
    CRITICAL CHANGE: Few-shot examples will now ALWAYS be in English, even if instructions are in LRL.
    """
    processed_text = text.strip()
    
    instruction = ""
    examples_content = ""
    no_entities_marker_lrl = ""
    no_entities_marker_en = "[NO_ENTITIES_FOUND]" # English marker for examples
    final_cue = ""
    example_header_lrl = ""

    entity_types_lrl_desc_map = { 
        "sw": "PER (Mtu), ORG (Shirika), LOC (Mahali), DATE (Tarehe)",
        "ha": "PER (Mutum), ORG (Kungiya), LOC (Wuri), DATE (Kwanan wata)",
        "en": "PER (Person), ORG (Organization), LOC (Location), DATE (Date)" # Fallback
    }
    entity_types_lrl = entity_types_lrl_desc_map.get(lang_code, entity_types_lrl_desc_map["en"])

    # Define UNIVERSAL English few-shot examples here for LRL prompts as well
    # These examples use the English 'no_entities_marker_en'
    english_few_shot_examples_for_lrl_prompt = [
        {"text": "Angela Merkel visited Paris on September 1st, 2021 with a delegation from the European Union.", 
         "entities_str": "[PER: Angela Merkel] [LOC: Paris] [DATE: September 1st, 2021] [ORG: European Union]"},
        {"text": "The quick brown fox jumps over the lazy dog in New York.", 
         "entities_str": "[LOC: New York]"},
        {"text": "There are no entities here.",
         "entities_str": no_entities_marker_en}
    ]

    if lang_code == 'sw':
        no_entities_marker_lrl = "[HAKUNA_VITU_VILIVYOPATIKANA]"
        final_cue = "Vitu:"
        example_header_lrl = "Mifano (kwa Kiingereza - hakikisha matokeo yako kwa kazi halisi yanafuata muundo huu):"
        instruction = (
            f"Kazi yako ni kutambua vitu vilivyotajwa (named entities) katika maandishi yaliyo hapa chini kwa lugha ya Kiswahili. "
            f"Vitu hivyo ni vya aina hizi: {entity_types_lrl}. "
            f"Kwa kila kitu, toa aina yake na maandishi yake. "
            f"Toa matokeo yako YOTE kama orodha ya vitu, kila kimoja kikiwa katika muundo huu: [AINA: Maandishi ya Kitu], vikitenganishwa na nafasi au mistari mipya. "
            f"Mfano wa muundo (kwa Kiswahili): [PER: Juma] [LOC: Nairobi] [DATE: Julai 2024]. " # LRL format example
            f"Ikiwa hakuna vitu vilivyopatikana, andika maneno haya hasa: {no_entities_marker_lrl}."
        )
        if use_few_shot:
            examples_content = example_header_lrl + "\n"
            for ex in english_few_shot_examples_for_lrl_prompt:
                # Note: The example text prompt is still framed as if it's LRL, but the content is English.
                examples_content += f"\nMaandishi (mfano wa Kiingereza): '{ex['text']}'\n"
                examples_content += f"Vitu (kwa Kiingereza): {ex['entities_str']}\n"

    elif lang_code == 'ha':
        no_entities_marker_lrl = "[BABU_SUNAYEN_DA_AKA_SAMU]"
        final_cue = "Sunaye:"
        example_header_lrl = "Misalai (da Turanci - tabbatar amsarka ta ainihin aiki ta bi wannan tsarin sosai):"
        instruction = (
            f"Aikin ka shine ka gano sunayen da aka ambata (named entities) a cikin rubutun da ke ƙasa da harshen Hausa. "
            f"Sunayen suna cikin waɗannan nau'ikan: {entity_types_lrl}. "
            f"Ga kowane suna, ka bayar da nau'in sa da rubutun sa. "
            f"Gabatar da DUKKAN amsarka a matsayin jerin sunaye, kowanne a cikin wannan tsarin: [NAU'I: Rubutun Suna], an raba su da sarari ko sabbin layuka. "
            f"Misali na tsari (da Hausa): [PER: Musa] [LOC: Kano] [DATE: Yuli 2024]. " # LRL format example
            f"Idan ba a sami sunaye ba, rubuta wannan jimlar daidai: {no_entities_marker_lrl}."
        )
        if use_few_shot:
            examples_content = example_header_lrl + "\n"
            for ex in english_few_shot_examples_for_lrl_prompt:
                examples_content += f"\nRubutu (misali na Turanci): '{ex['text']}'\n"
                examples_content += f"Sunaye (da Turanci): {ex['entities_str']}\n"
    else: 
        logging.warning(f"LRL instructions for '{lang_code}' in generate_lrl_instruct_ner_prompt are not defined. Falling back to English-instructed prompt via generate_ner_prompt.")
        return generate_ner_prompt(text, lang_code, model_name, use_few_shot)

    prompt = instruction
    if use_few_shot and examples_content:
        prompt += "\n\n" + examples_content
    else: 
        zero_shot_guide_lrl = {
            "sw": f"\n\nChanganua maandishi yafuatayo na utoe vitu kwa muundo uliobainishwa ([AINA: Maandishi ya Kitu] au {no_entities_marker_lrl}):",
            "ha": f"\n\nBincika rubutun da ke tafe kuma ka bayar da sunayen ta hanyar da aka kayyade ([NAU'I: Rubutun Suna] ko {no_entities_marker_lrl}):"
        }.get(lang_code, f"\n\nNow, analyze the following text and provide the entities in the specified format ([TYPE: Entity Text] or {no_entities_marker_en}):") 
        prompt += zero_shot_guide_lrl
    
    prompt += f"\n\nRubutu ({lang_code}): '{processed_text}'\n\n{final_cue}"
    return prompt

def extract_entities(text: str, lang_code: str = "en", verbose: bool = False):
    """
    Extract named entities from model output text with improved robustness.
    
    Args:
        text: Model-generated text with entity mentions
        lang_code: Language code
        verbose: Whether to print debug information
        
    Returns:
        List of entity dictionaries with text and type
    """
    extracted_entities = []
    # Regex to find entities in the format [TYPE: Entity Text]
    # It allows for spaces around the type, colon, and entity text.
    # It captures the TYPE and the Entity Text.
    entity_pattern = r"\[\s*([A-Z]+)\s*:\s*([^]]+?)\s*\]"

    # Language-specific "no entities found" markers
    no_entities_markers = {
        "en": "[NO_ENTITIES_FOUND]",
        "sw": "[HAKUNA_VITU_VILIVYOPATIKANA]",
        "ha": "[BABU_SUNAYEN_DA_AKA_SAMU]"
    }
    no_entities_marker = no_entities_markers.get(lang_code, "[NO_ENTITIES_FOUND]") # Default to English if lang_code is unknown

    # Check if the model output indicates no entities found
    if no_entities_marker in text:
        if verbose:
            logging.info(f"Marker '{no_entities_marker}' found. No entities extracted.")
        return []

    # More robust check for variations like "[NONE](NONE)" or similar, primarily for older prompt compatibility
    # This should be less necessary with the new specific markers but kept for safety.
    none_patterns = [
        r"\[NONE\]\(NONE\)", # Exact match for [NONE](NONE)
        r"\[HAKUNA\]\(HAKUNA\)", # Swahili [HAKUNA](HAKUNA)
        r"\[BABU\]\(BABU\)"    # Hausa [BABU](BABU)
    ]
    for none_pattern_str in none_patterns:
        if re.search(none_pattern_str, text, re.IGNORECASE):
            if verbose:
                logging.info(f"Found legacy no-entity marker via regex: {none_pattern_str}. No entities extracted.")
            return []

    matches = re.finditer(entity_pattern, text)
    for match in matches:
        entity_type = match.group(1)
        entity_text = match.group(2)
        entity_type_norm = normalize_entity_type(entity_type.strip().upper())
        entity_text_clean = entity_text.strip().strip('"\'')
        if len(entity_text_clean) > 1 and entity_type_norm in ["PER", "ORG", "LOC", "DATE"]:
            extracted_entities.append({"text": entity_text_clean, "type": entity_type_norm})
            if verbose: print(f"    Added: {entity_text_clean} [{entity_type_norm}]")
        elif verbose:
            print(f"    Skipped (pattern): Type='{entity_type_norm}', Text='{entity_text_clean}'")

    # Remove duplicates while preserving order (case-insensitive text match for same type)
    unique_entities = []
    seen = set()
    for entity in extracted_entities:
        # Create a unique key for each entity (text + type)
        key = (entity["text"].lower(), entity["type"])
        if key not in seen:
            # Additional check: Ensure type is standard
            if entity["type"] in ["PER", "ORG", "LOC", "DATE"]:
                seen.add(key)
                unique_entities.append(entity)
            elif verbose:
                 print(f"  Filtering out entity with non-standard type '{entity['type']}' during deduplication: {entity['text']}")
        elif verbose:
            print(f"  Removing duplicate: {entity['text']} [{entity['type']}]")

    if verbose: print(f"  Found {len(unique_entities)} unique entities after deduplication.")
    return unique_entities

def normalize_entity_type(entity_type):
    """
    Normalize entity type to standard NER categories.
    
    Args:
        entity_type: Raw entity type from model output
        
    Returns:
        Normalized entity type
    """
    entity_type = entity_type.upper().strip()
    
    # Map to standard types with more comprehensive matching
    if any(t in entity_type for t in ["PERSON", "PER", "INDIVIDUAL", "HUMAN", "NAME"]):
        return "PER"
    elif any(t in entity_type for t in ["ORG", "ORGANIZATION", "COMPANY", "INSTITUTION", "AGENCY", "GROUP"]):
        return "ORG"
    elif any(t in entity_type for t in ["LOC", "LOCATION", "PLACE", "AREA", "REGION", "COUNTRY", "CITY"]):
        return "LOC"
    elif any(t in entity_type for t in ["DATE", "TIME", "DATETIME", "PERIOD", "DAY", "MONTH", "YEAR"]):
        return "DATE"
    else:
        return entity_type  # Return as is if no match

def process_ner_baseline(
    tokenizer: Any, 
    model: Any, 
    text: str, 
    max_input_length: int = 4096,
    max_new_tokens: int = 200, # Max tokens for the NER output itself
    # Unified generation parameters (expected to be passed)
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    do_sample: bool = True, # Keep do_sample as it might be fixed based on temp
    lang_code: str = "en",
    model_name: str = "",
    prompt_in_lrl: bool = False,
    use_few_shot: bool = True,
    verbose: bool = False
) -> List[Dict[str, str]]:
    """
    Process a text for NER using the baseline approach (direct prompting).
    Now uses unified generation parameters passed from the evaluation function.
    Internal language/model-specific logic for these parameters is removed.
    """
    # --- Add Type Check ---
    print(f"DEBUG process_ner_baseline: Type of tokenizer arg: {type(tokenizer)}")
    print(f"DEBUG process_ner_baseline: Type of model arg: {type(model)}")
    # --- End Type Check ---
    try:
        # Generate prompt based on language and instruction preference
        if prompt_in_lrl:
            prompt = generate_lrl_instruct_ner_prompt(text, lang_code, model_name, use_few_shot)
        else:
            prompt = generate_ner_prompt(text, "en", model_name, use_few_shot)  # English instructions
        
        # Tokenize with truncation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # --- Use unified parameters directly --- 
        # The language/model-specific adjustments are assumed to have been done 
        # by the caller (evaluate_ner_baseline or the runner script) 
        # before passing the final values here.
        gen_temperature = temperature
        gen_top_p = top_p
        gen_top_k = top_k
        gen_max_tokens = max_new_tokens # Use the specific max_tokens for NER output
        gen_repetition_penalty = repetition_penalty
        # Decide on sampling based on final temperature
        gen_do_sample = do_sample if temperature > 0 else False 

        if verbose:
            print("--- NER Baseline Generation Params ---")
            print(f"  Temperature: {gen_temperature}")
            print(f"  Top P: {gen_top_p}")
            print(f"  Top K: {gen_top_k}")
            print(f"  Max New Tokens: {gen_max_tokens}")
            print(f"  Repetition Penalty: {gen_repetition_penalty}")
            print(f"  Do Sample: {gen_do_sample}")
            print("------------------------------------")

        # Generate answer using the final parameters
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=gen_max_tokens,
                do_sample=gen_do_sample,
                temperature=gen_temperature,
                top_p=gen_top_p,
                top_k=gen_top_k,
                repetition_penalty=gen_repetition_penalty,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # TEMPORARY DEBUG: Print raw model output for specific cases
        # Adjust the condition to target the failing case, e.g., Swahili zero-shot with Aya
        if lang_code == 'sw' and not use_few_shot and "aya" in model_name.lower():
            print(f"\nDEBUG SWahili ZERO-SHOT (Aya) RAW OUTPUT for text: {text[:100]}...")
            print(f"RAW MODEL OUTPUT: >>>{output_text}<<<")
            
        # DEBUG: Print model output for inspection (only for first few samples)
        static_sample_counter = getattr(process_ner_baseline, "sample_counter", 0)
        if verbose and static_sample_counter < 3:  # Print only first 3 samples to avoid flooding logs
            print(f"\n=== DEBUG: MODEL OUTPUT (Sample {static_sample_counter + 1}) ===")
            print(f"Text: {text[:50]}...")
            print(f"Raw model output: {output_text[:500]}...")
            print("=======================================")
            process_ner_baseline.sample_counter = static_sample_counter + 1
        
        # Extract entities
        entities = extract_entities(output_text, lang_code, verbose=verbose)
        
        # DEBUG: Print extracted entities
        if verbose and static_sample_counter <= 3:
            print(f"Extracted entities: {entities}")
        
        return entities
    except Exception as e:
        import traceback
        print(f"Error in process_ner_baseline: {str(e)}")
        traceback.print_exc()
        return []

# Add a static counter to the function
process_ner_baseline.sample_counter = 0

def evaluate_ner_baseline(
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    model_name: str,
    prompt_in_lrl: bool,
    use_few_shot: bool,
    # Unified generation parameters (expected from runner script)
    temperature: float, 
    top_p: float, 
    top_k: int, 
    max_tokens: int,
    repetition_penalty: float, 
    do_sample: bool = True
) -> pd.DataFrame:
    """
    Evaluate NER baseline approach on a dataset.
    Accepts unified generation parameters.
    """
    print(f"DEBUG evaluate_ner_baseline ENTRY: Type of tokenizer param: {type(tokenizer)}")
    print(f"DEBUG evaluate_ner_baseline ENTRY: Type of model param: {type(model)}")
    # --- End Type Check ---
    results = []
    # The input samples_df should already have 'text' and 'entities' (as ground_truth) correctly formatted by load_masakhaner_samples.
    # If 'tokens' is present, it can be used, but 'text' is primary for process_ner_baseline.

    shot_description = "few-shot" if use_few_shot else "zero-shot"
    prompt_lang_description = "LRL-instruct" if prompt_in_lrl else "EN-instruct"

    logging.info(f"Starting NER baseline evaluation for {model_name} on {lang_code} ({prompt_lang_description}, {shot_description}).")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing NER baseline ({lang_code}, {shot_description})"):
        text_to_process = row.get('text')
        # Ground truth entities should be in row['entities'] and already be a list of dicts
        # e.g., [{'entity': 'text', 'type': 'PER'}, ...]
        ground_truth_entities = row.get('entities', []) 

        if not text_to_process:
            logging.warning(f"Sample {idx} has no text. Skipping.")
            # Optionally, append a row with error or skip
            results.append({
                'id': row.get('id', idx),
                'text': text_to_process,
                'ground_truth_entities': ground_truth_entities,
                'predicted_entities': [], # No prediction possible
                'error_message': 'Missing text',
                # Add other relevant fields for consistency in the DataFrame
                'language': lang_code,
                'model_name': model_name,
                'shot_type': shot_description,
                'prompt_language': prompt_lang_description
            })
            continue

        start_time_sample = time.time()
        predicted_entities_list = process_ner_baseline(
            tokenizer=tokenizer,
            model=model,
            text=text_to_process,
            lang_code=lang_code, # lang_code of the text
            model_name=model_name,
            prompt_in_lrl=prompt_in_lrl, # Whether to use LRL instructions
            use_few_shot=use_few_shot,
            # Pass unified generation parameters
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_tokens, # max_tokens is for NER output length
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            verbose=False # Set to True for detailed extraction logs from process_ner_baseline
        )
        runtime_sample = time.time() - start_time_sample

        results.append({
            'id': row.get('id', idx),
            'text': text_to_process,
            'ground_truth_entities': ground_truth_entities, # This should be the pre-processed list of dicts
            'predicted_entities': predicted_entities_list,
            'runtime_seconds': runtime_sample,
            'language': lang_code,
            'model_name': model_name, 
            'shot_type': shot_description,
            'prompt_language': prompt_lang_description,
            # Optionally log the exact generation parameters used for this sample
            'temperature_used': temperature,
            'top_p_used': top_p,
            'top_k_used': top_k,
            'max_tokens_used': max_tokens,
            'repetition_penalty_used': repetition_penalty,
            'do_sample_used': do_sample
        })

    results_df = pd.DataFrame(results)
    return results_df

def calculate_ner_metrics(gold_entities: List[Dict[str, str]], predicted_entities: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for NER predictions.
    
    Args:
        gold_entities: List of ground truth entities
        predicted_entities: List of predicted entities
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    # Convert to sets of (entity, type) tuples for comparison
    gt_set = set()
    for e in gold_entities:
        if "entity" in e:
            entity_text = e["entity"].lower()
        elif "text" in e:
            entity_text = e["text"].lower()
        else:
            continue  # Skip entities with unrecognized format
            
        entity_type = e.get("type", e.get("entity_type", ""))
        if entity_type:  # Only add if entity_type is not empty
            gt_set.add((entity_text, entity_type))
    
    pred_set = set()
    for e in predicted_entities:
        if "entity" in e:
            entity_text = e["entity"].lower()
        elif "text" in e:
            entity_text = e["text"].lower()
        else:
            continue  # Skip entities with unrecognized format
            
        entity_type = e.get("type", e.get("entity_type", ""))
        if entity_type:  # Only add if entity_type is not empty
            pred_set.add((entity_text, entity_type))
        
        # Calculate true positives, false positives, false negatives
        true_positives = len(gt_set.intersection(pred_set))
        false_positives = len(pred_set) - true_positives
        false_negatives = len(gt_set) - true_positives
        
        # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def load_masakhaner_samples(lang_code: str, split: str = "test", num_samples: int = None, seed: int = 42) -> pd.DataFrame:
    """
    Load samples from MasakhaNER dataset.
    
    Args:
        lang_code: Language code (e.g., 'sw', 'ha', 'en')
        split: Dataset split ('train', 'dev', 'test')
        num_samples: Number of samples to load (None for all)
        seed: Random seed for sampling
        
    Returns:
        DataFrame with text and entities
    """
    # Import the HuggingFace loader
    from src.utils.data_loaders.load_masakhaner import load_masakhaner_samples as load_from_hf

    try:
        # Load data from HuggingFace
        samples_df = load_from_hf(lang_code, num_samples=num_samples, split=split, seed=seed)
        
        if samples_df.empty:
            print(f"No samples found for {lang_code} in MasakhaNER split '{split}'. Falling back to dummy data...")
            return create_dummy_ner_data(lang_code, num_samples or 5)
        
        # --- Create 'text' column OUTSIDE the if block --- 
        if 'tokens' in samples_df.columns:
            samples_df['text'] = samples_df['tokens'].apply(lambda tokens: ' '.join(tokens))
        else:
            print("ERROR: 'tokens' column missing. Cannot create 'text' column.")
            return pd.DataFrame() # Return empty if tokens are missing
            
        # --- Convert entities OUTSIDE the if block ---
        if 'entities' in samples_df.columns:
            # --- Define convert_entities helper function --- 
            def convert_entities(row):
                formatted_entities = []
                # Ensure entities column contains lists (handle potential loading issues)
                entities_list = row['entities']
                if not isinstance(entities_list, list):
                    print(f"Warning: 'entities' data is not a list for a row: {type(entities_list)}. Skipping entity conversion for this row.")
                    return [] 
                    
                for entity_info in entities_list:
                    # Add checks for expected keys in entity_info dict
                    if not all(k in entity_info for k in ('start', 'end', 'entity_type')):
                        print(f"Warning: Skipping malformed entity_info: {entity_info}")
                        continue
                    try:
                        start_idx = entity_info['start']
                        end_idx = entity_info['end']
                        entity_type = entity_info['entity_type']
                        # Ensure indices are within bounds
                        if start_idx >= 0 and end_idx <= len(row['tokens']):
                            entity_text = ' '.join(row['tokens'][start_idx:end_idx])
                            formatted_entities.append({
                                "entity": entity_text,
                                "type": entity_type
                            })
                        else:
                            print(f"Warning: Entity indices [{start_idx}:{end_idx}] out of bounds for tokens length {len(row['tokens'])}")
                    except Exception as e_conv:
                        print(f"Warning: Error converting entity: {entity_info}. Error: {e_conv}")
                return formatted_entities
            
            # Apply the conversion function
            samples_df['entities'] = samples_df.apply(convert_entities, axis=1)
        else:
            print("ERROR: 'entities' column missing. Cannot process entities.")
            return pd.DataFrame() # Return empty if entities are missing
        
        print(f"DEBUG: Columns in samples_df *before returning* from load_masakhaner_samples: {samples_df.columns.tolist()}") # ADD DEBUG PRINT
        print(f"  Loaded and prepared {len(samples_df)} samples for {lang_code}.") # Updated print message
        return samples_df
    
    except Exception as e:
        print(f"Error loading MasakhaNER data from HuggingFace: {e}")
        print("Falling back to dummy data...")
        return create_dummy_ner_data(lang_code, num_samples or 5)

def create_dummy_ner_data(lang_code: str, num_samples: int = 5) -> pd.DataFrame:
    """
    Create dummy NER data for testing.
    
    Args:
        lang_code: Language code
        num_samples: Number of samples to create
        
    Returns:
        DataFrame with text and entities
    """
    samples = []
    
    # English dummy data
    if lang_code == "en":
        texts = [
            "The President of the United States, Joe Biden, visited Berlin last Tuesday.",
            "Apple Inc. announced a new partnership with Microsoft Corporation in January 2023.",
            "Mount Kilimanjaro is located in Tanzania and is the highest mountain in Africa.",
            "The United Nations was founded on October 24, 1945, in San Francisco, California.",
            "Elon Musk is the CEO of Tesla and SpaceX, and he recently acquired Twitter."
        ]
        
        entities = [
            [
                {"entity": "Joe Biden", "type": "PER"},
                {"entity": "United States", "type": "LOC"},
                {"entity": "Berlin", "type": "LOC"},
                {"entity": "Tuesday", "type": "DATE"}
            ],
            [
                {"entity": "Apple Inc.", "type": "ORG"},
                {"entity": "Microsoft Corporation", "type": "ORG"},
                {"entity": "January 2023", "type": "DATE"}
            ],
            [
                {"entity": "Mount Kilimanjaro", "type": "LOC"},
                {"entity": "Tanzania", "type": "LOC"},
                {"entity": "Africa", "type": "LOC"}
            ],
            [
                {"entity": "United Nations", "type": "ORG"},
                {"entity": "October 24, 1945", "type": "DATE"},
                {"entity": "San Francisco", "type": "LOC"},
                {"entity": "California", "type": "LOC"}
            ],
            [
                {"entity": "Elon Musk", "type": "PER"},
                {"entity": "Tesla", "type": "ORG"},
                {"entity": "SpaceX", "type": "ORG"},
                {"entity": "Twitter", "type": "ORG"}
            ]
        ]
    
    # Swahili dummy data
    elif lang_code == "sw":
        texts = [
            "Rais wa Marekani, Joe Biden, alitembelea Berlin Jumanne iliyopita.",
            "Kampuni ya Apple Inc. ilitangaza ushirikiano mpya na Kampuni ya Microsoft mwezi Januari 2023.",
            "Mlima Kilimanjaro upo Tanzania na ni mlima mrefu zaidi Afrika.",
            "Umoja wa Mataifa ulianzishwa tarehe 24 Oktoba, 1945, huko San Francisco, California.",
            "Elon Musk ni Mkurugenzi Mtendaji wa Tesla na SpaceX, na hivi karibuni alinunua Twitter."
        ]
        
        entities = [
            [
                {"entity": "Joe Biden", "type": "PER"},
                {"entity": "Marekani", "type": "LOC"},
                {"entity": "Berlin", "type": "LOC"},
                {"entity": "Jumanne", "type": "DATE"}
            ],
            [
                {"entity": "Apple Inc.", "type": "ORG"},
                {"entity": "Kampuni ya Microsoft", "type": "ORG"},
                {"entity": "Januari 2023", "type": "DATE"}
            ],
            [
                {"entity": "Mlima Kilimanjaro", "type": "LOC"},
                {"entity": "Tanzania", "type": "LOC"},
                {"entity": "Afrika", "type": "LOC"}
            ],
            [
                {"entity": "Umoja wa Mataifa", "type": "ORG"},
                {"entity": "24 Oktoba, 1945", "type": "DATE"},
                {"entity": "San Francisco", "type": "LOC"},
                {"entity": "California", "type": "LOC"}
            ],
            [
                {"entity": "Elon Musk", "type": "PER"},
                {"entity": "Tesla", "type": "ORG"},
                {"entity": "SpaceX", "type": "ORG"},
                {"entity": "Twitter", "type": "ORG"}
            ]
        ]
    
    # Hausa dummy data
    elif lang_code == "ha":
        texts = [
            "Shugaban Amurka, Joe Biden, ya ziyarci Berlin ranar Talata da ta gabata.",
            "Kamfanin Apple Inc. ya sanar da sabon hadin gwiwa da Kamfanin Microsoft a watan Janairu 2023.",
            "Dutsen Kilimanjaro yana a Tanzania kuma shine dutsen da ya fi tsawo a Afrika.",
            "Majalisar Dinkin Duniya an kafa ta ne a ranar 24 ga Oktoba, 1945, a San Francisco, California.",
            "Elon Musk shine Babban Daraktan Tesla da SpaceX, kuma kwanannan ya sayo Twitter."
        ]
        
        entities = [
            [
                {"entity": "Joe Biden", "type": "PER"},
                {"entity": "Amurka", "type": "LOC"},
                {"entity": "Berlin", "type": "LOC"},
                {"entity": "Talata", "type": "DATE"}
            ],
            [
                {"entity": "Apple Inc.", "type": "ORG"},
                {"entity": "Kamfanin Microsoft", "type": "ORG"},
                {"entity": "Janairu 2023", "type": "DATE"}
            ],
            [
                {"entity": "Dutsen Kilimanjaro", "type": "LOC"},
                {"entity": "Tanzania", "type": "LOC"},
                {"entity": "Afrika", "type": "LOC"}
            ],
            [
                {"entity": "Majalisar Dinkin Duniya", "type": "ORG"},
                {"entity": "24 ga Oktoba, 1945", "type": "DATE"},
                {"entity": "San Francisco", "type": "LOC"},
                {"entity": "California", "type": "LOC"}
            ],
            [
                {"entity": "Elon Musk", "type": "PER"},
                {"entity": "Tesla", "type": "ORG"},
                {"entity": "SpaceX", "type": "ORG"},
                {"entity": "Twitter", "type": "ORG"}
            ]
        ]
    
    # Default to English if language not supported
    else:
        return create_dummy_ner_data("en", num_samples)
    
    # Create samples
    for i in range(min(num_samples, len(texts))):
        samples.append({
            "text": texts[i],
            "entities": entities[i]
        })
    
    return pd.DataFrame(samples) 