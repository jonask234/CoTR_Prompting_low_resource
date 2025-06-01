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

from src.experiments.cotr.cotr_utils import CoTRSystem
from src.experiments.cotr.language_information import get_language_information

# Import the new metrics function
from src.evaluation.cotr.translation_metrics import calculate_comet_score

# Remove the old block that added project root to Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
# sys.path.insert(0, project_root)

# Import utility functions
from src.utils.data_loaders.load_masakhaner import load_masakhaner_samples
# Remove OpenAI specific import
# from src.utils.llm_utils import call_llm_api

# Import COMET if available
from evaluation.cotr.qa_metrics_cotr import COMET_AVAILABLE # Use QA metrics path as it likely contains the check

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
    max_input_length: int = 4096,
    max_new_tokens: int = 512,
    model_name: str = ""
) -> str:
    """
    Translate text from source language to target language using the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: The text to translate
        source_lang: Source language code
        target_lang: Target language code
        max_input_length: Maximum input sequence length
        max_new_tokens: Maximum tokens for the translated output
        model_name: Model name for model-specific adjustments
        
    Returns:
        Translated text
    """
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
    
    # Enhanced parameter adjustment for different languages and models
    temperature = 0.7     # Default temperature
    top_p = 0.9           # Default top_p
    repetition_penalty = 1.2  # Default repetition penalty
    
    # Detect model types
    is_aya_model = "aya" in model_name.lower()
    is_qwen_model = "qwen" in model_name.lower()
    
    # Language-specific parameters
    if source_lang == 'ha' or target_lang == 'ha':
        # Hausa needs special handling
        temperature = 0.5      # Lower temperature for Hausa
        repetition_penalty = 1.4  # Higher repetition penalty
        
        # Model-specific adjustments for Hausa
        if is_aya_model:
            temperature = 0.4  # Even lower for Aya with Hausa
        elif is_qwen_model:
            temperature = 0.6  # Slightly higher for Qwen with Hausa
            repetition_penalty = 1.3
    
    elif source_lang == 'sw' or target_lang == 'sw':
        # Swahili adjustments
        temperature = 0.6  # Moderate temperature
        repetition_penalty = 1.3
        
        # Model-specific adjustments for Swahili
        if is_aya_model:
            temperature = 0.5  # Lower for Aya with Swahili
        elif is_qwen_model:
            temperature = 0.7  # Higher for Qwen with Swahili
            repetition_penalty = 1.2
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Clean up the response
    translation = clean_translation_response(response, target_lang, source_lang)
    
    return translation

def generate_ner_prompt_english(text: str, model_name: str = "") -> str:
    """
    Generate a prompt for NER in English with clear instructions.
    
    Args:
        text: The text to analyze for named entities
        model_name: Model name for potential model-specific adjustments
        
    Returns:
        Formatted prompt
    """
    # Base instructions for English
    base_instruction = """Identify all named entities in the text below. 
Tag each entity with one of these categories: PER (person), ORG (organization), LOC (location), DATE (date).
Format your response as a list of entities with their types: 
Entity 1 [TYPE], Entity 2 [TYPE], etc.
ONLY return the entities - do not repeat or explain the text.
"""

    # English examples
    examples = """Examples:

Text: The President of the United States, Joe Biden, visited Berlin last Tuesday.
Entities: Joe Biden [PER], United States [LOC], Berlin [LOC], Tuesday [DATE]

Text: Apple Inc. announced a new partnership with Microsoft Corporation in January 2023.
Entities: Apple Inc. [ORG], Microsoft Corporation [ORG], January 2023 [DATE]

Text: Mount Kilimanjaro is located in Tanzania and is the highest mountain in Africa.
Entities: Mount Kilimanjaro [LOC], Tanzania [LOC], Africa [LOC]
"""

    # Final prompt assembly
    prompt = f"{base_instruction}\n\n{examples}\n\nText: {text}\nEntities:"
    
    return prompt

def process_ner_english(
    model: Any,
    tokenizer: Any,
    text: str,
    max_input_length: int = 4096,
    max_new_tokens: int = 200,
    model_name: str = ""
) -> List[Dict[str, str]]:
    """
    Process a text for NER in English using the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: The text to analyze
        max_input_length: Maximum input sequence length
        max_new_tokens: Maximum tokens for the generated answer
        model_name: Model name for model-specific adjustments
        
    Returns:
        List of extracted entities
    """
    try:
        # Generate the NER prompt in English
        prompt = generate_ner_prompt_english(text, model_name)
        
        # Tokenize with truncation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Set up generation parameters
        is_aya_model = "aya" in model_name.lower()
        is_qwen_model = "qwen" in model_name.lower()
        
        # Default parameters
        temperature = 0.2
        top_p = 0.8
        repetition_penalty = 1.2
        
        # Model-specific adjustments
        if is_aya_model:
            temperature = 0.3
            repetition_penalty = 1.3
        elif is_qwen_model:
            temperature = 0.2
            top_p = 0.75
        
        # Generate with optimized parameters
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Extract entities from the output
        entities = extract_entities(output_text, lang_code="en")
        
        return entities
    
    except Exception as e:
        import traceback
        print(f"Error in process_ner_english: {str(e)}")
        traceback.print_exc()
        return []

def translate_entities_to_original_language(
    model: Any,
    tokenizer: Any,
    entities: List[Dict[str, str]],
    source_lang: str,
    text_orig: str,
    model_name: str = ""
) -> List[Dict[str, str]]:
    """
    Translate entities found in English back to the original language.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        entities: List of entities extracted from English text
        source_lang: Original language code
        text_orig: Original text (for context)
        model_name: Model name for model-specific adjustments
        
    Returns:
        List of entities with text translated to original language
    """
    # If we're already in English, no translation needed
    if source_lang == "en":
        return entities
    
    # If no entities found, return empty list
    if not entities:
        return []
    
    # For matching entities back to original text
    translated_entities = []
    
    # Create a lookup for potential translations
    entity_lookup = {}
    
    # First, try to use the original text to find exact matches
    text_lower = text_orig.lower()
    
    for entity in entities:
        # Handle different key formats: some entity dictionaries may use 'text', others 'entity'
        if "text" in entity:
            entity_name = entity["text"]
        elif "entity" in entity:
        entity_name = entity["entity"]
        else:
            # Skip entities with unrecognized format
            print(f"Warning: Entity has unexpected format: {entity}")
            continue
        
        # Similarly handle different type key formats
        if "type" in entity:
        entity_type = entity["type"]
        elif "entity_type" in entity:
            entity_type = entity["entity_type"]
        else:
            # Skip entities with unrecognized format
            print(f"Warning: Entity missing type information: {entity}")
            continue
        
        # Check if this entity's translation appears in the original text
        # Try to translate the entity name back to source language
        translated_entity = translate_text(
            model, tokenizer, entity_name, "en", source_lang, 
            model_name=model_name
        )
        
        # Check if the translated entity appears in the original text
        if translated_entity.lower() in text_lower:
            translated_entities.append({
                "entity": translated_entity,
                "type": entity_type
            })
            entity_lookup[entity_name.lower()] = translated_entity
            continue
        
        # If not found, try to find a potential match based on character positions
        # (This is less reliable but can help in some cases)
        found = False
        
        # For now, we'll just use the translation
        translated_entities.append({
            "entity": translated_entity,
            "type": entity_type
        })
        entity_lookup[entity_name.lower()] = translated_entity
    
    return translated_entities

def evaluate_ner_cotr_sample(
    model: Any,
    tokenizer: Any,
    text: str,
    ground_truth_entities: List[Dict[str, str]],
    lang_code: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Evaluate a single NER sample using the CoTR approach.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: The text to analyze
        ground_truth_entities: The ground truth entities
        lang_code: Language code
        model_name: Model name
        
    Returns:
        Dictionary with results and metrics
    """
    start_time = time.time()
    result = {
        "text": text,
        "ground_truth_entities": ground_truth_entities,
        "language": lang_code
    }
    
    try:
        # Step 1: Translate text to English
        text_en = translate_text(
            model, tokenizer, text, lang_code, "en", 
            model_name=model_name
        )
        
        # Step 2: Process NER in English
        entities_en = process_ner_english(
            model, tokenizer, text_en,
            model_name=model_name
        )
        
        # Step 3: Translate entities back to the source language
        predicted_entities = translate_entities_to_original_language(
            model, tokenizer, entities_en, lang_code, text,
            model_name=model_name
        )
        
        # Store results
        result["text_en"] = text_en
        result["entities_en"] = entities_en
        result["predicted_entities"] = predicted_entities
        
        # Calculate runtime
        runtime = time.time() - start_time
        result["runtime_seconds"] = runtime
    
    except Exception as e:
        print(f"Error processing sample: {str(e)}")
        result["error"] = str(e)
        result["predicted_entities"] = []
    
    return result

def evaluate_ner_cotr(
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    pipeline_type: str = "multi_prompt",
    use_few_shot: bool = True,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: float = 40,
    max_tokens: int = 200,
    max_translation_tokens: int = 200,
    num_beams: int = 1
) -> pd.DataFrame:
    """
    Evaluate NER using Chain of Translation Prompting (CoTR) approach.
    Now uses standardized parameters for consistent comparison with the baseline.
    
    Args:
        model_name: Name of the model to use
        tokenizer: The tokenizer
        model: The language model
        samples_df: DataFrame containing the samples
        lang_code: Language code
        pipeline_type: Type of CoTR pipeline ('multi_prompt' or 'single_prompt')
        use_few_shot: Whether to include few-shot examples
        temperature: Temperature for sampling
        top_p: Top-p for sampling
        top_k: Top-k for sampling
        max_tokens: Maximum tokens for NER generation
        max_translation_tokens: Maximum tokens for translation
        num_beams: Number of beams for beam search
        
    Returns:
        DataFrame with predictions and metrics
    """
    # try:
    #     tokenizer, model = initialize_model(model_name) # Remove model initialization from here
    # except Exception as e:
    #     print(f"ERROR: Failed to initialize model {model_name}: {e}")
    #     return pd.DataFrame()
    
    # Process samples
    results = []
    shot_type = "few-shot" if use_few_shot else "zero-shot"
    
    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} samples ({model_name}, {pipeline_type})"):
        text = row["text"]
        ground_truth_entities_original = row["entities"] # Original entities from DataFrame
        start_time_sample = time.time() # For per-sample runtime

        # Format ground truth entities for consistency in metrics, do this early
        formatted_ground_truth = []
        try:
            for entity in ground_truth_entities_original:
                if "start" in entity and "end" in entity and "tokens" in row: # Ensure tokens key exists
                    entity_text = " ".join(row["tokens"][entity["start"]:entity["end"]])
                    entity_type = entity["entity_type"]
                else:
                    entity_text = entity.get("text", entity.get("entity", ""))
                    entity_type = entity.get("type", "")
                formatted_ground_truth.append({"entity": entity_text, "type": entity_type})
        except Exception as e_gt_format:
            print(f"Error formatting ground truth for sample {idx}: {e_gt_format}")
            # Append an error row if ground truth formatting fails for some reason
            results.append({
                "text": text,
                "ground_truth_entities": ground_truth_entities_original, # Store original GT
                "predicted_entities": [],
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "language": lang_code,
                "pipeline_type": pipeline_type,
                "shot_type": shot_type,
                "intermediate_results": {"error": f"Ground truth formatting error: {e_gt_format}"},
                "runtime_seconds": time.time() - start_time_sample,
                "error_message": f"Ground truth formatting error: {e_gt_format}"
            })
            continue # Skip to next sample

        try:
        if pipeline_type == "multi_prompt":
            # Step 1: Translate from LRL to English
            text_en = translate_text(
                model, tokenizer, text, lang_code, "en", 
                max_new_tokens=max_translation_tokens,
                model_name=model_name
            )
            
            # Step 2: Process NER in English
            entities_en = process_ner_english(
                model, tokenizer, text_en,
                max_new_tokens=max_tokens,
                model_name=model_name
            )
            
            # Step 3: Translate entities back to original language
                predicted_entities_raw = translate_entities_to_original_language(
                model, tokenizer, entities_en, lang_code, text,
                model_name=model_name
            )
            
            # Store intermediate results for analysis
            intermediate_results = {
                "text_en": text_en,
                "entities_en": entities_en
            }
            else:  # single_prompt (logic was here previously, but it's better handled by its own function)
                   # This path should ideally call evaluate_ner_cotr_single_prompt or have its own logic
                   # For now, let's assume multi_prompt is the main path for this function
                   # and single_prompt is handled by evaluate_ner_cotr_single_prompt
                print(f"Warning: evaluate_ner_cotr called with pipeline_type='single_prompt'. This should be handled by evaluate_ner_cotr_single_prompt.")
                # Fallback to a basic single prompt flow if called incorrectly, or raise error
                # For now, returning empty or erroring might be safer if this path is unexpected.
                # Let's make it skip this sample with an error if pipeline_type is not multi_prompt
                raise ValueError("evaluate_ner_cotr is intended for 'multi_prompt'. Use 'evaluate_ner_cotr_single_prompt' for single_prompt.")
        
            # Format predicted entities
        formatted_predictions = []
            for entity in predicted_entities_raw: # Use the raw predictions from the steps above
                entity_text = entity.get("text", entity.get("entity", ""))
                entity_type = entity.get("type", "")
            formatted_predictions.append({
                "entity": entity_text,
                "type": entity_type
            })
        
        # Calculate metrics
        metrics = calculate_ner_metrics_for_sample(formatted_ground_truth, formatted_predictions)
            runtime_sample = time.time() - start_time_sample
        
        # Store result
        result = {
            "text": text,
            "ground_truth_entities": formatted_ground_truth,
            "predicted_entities": formatted_predictions,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "language": lang_code,
            "pipeline_type": pipeline_type,
            "shot_type": shot_type,
                "intermediate_results": intermediate_results,
                "runtime_seconds": runtime_sample,
                "error_message": None
        }
        results.append(result)

        except Exception as e_sample:
            print(f"ERROR processing sample {idx} for {lang_code} ({pipeline_type}, {shot_type}): {e_sample}")
            import traceback
            traceback.print_exc()
            results.append({
                "text": text,
                "ground_truth_entities": formatted_ground_truth, # Use already formatted GT
                "predicted_entities": [],
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "language": lang_code,
                "pipeline_type": pipeline_type,
                "shot_type": shot_type,
                "intermediate_results": {"error": str(e_sample)},
                "runtime_seconds": time.time() - start_time_sample,
                "error_message": str(e_sample)
            })
            continue # Continue to the next sample
    
    if not results:
        print(f"WARNING: No results were successfully processed for {lang_code} with {model_name} and {pipeline_type} pipeline.")
        return pd.DataFrame()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate average metrics
    avg_precision = results_df["precision"].mean()
    avg_recall = results_df["recall"].mean()
    avg_f1 = results_df["f1_score"].mean()
    
    print(f"\nAverage metrics for {lang_code} with {model_name} and {pipeline_type} pipeline:")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall: {avg_recall:.4f}")
    print(f"  F1 Score: {avg_f1:.4f}")
    
    return results_df

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

class NERCoTRSystem(CoTRSystem):
    def __init__(self, model_name: str, lang_code: str):
        """Initialize the NER CoTR system.

        Args:
            model_name: Name of the model to use
            lang_code: Language code (e.g., "sw", "ha", "en")
        """
        super().__init__(model_name)
        self.lang_code = lang_code
        self.lang_info = get_language_information(lang_code)
        
    def create_prompt(self, text: str, entities_in_english=None) -> str:
        """
        Create a prompt for the NER task that includes language-specific information.
        
        Args:
            text: The input text to analyze for named entities
            entities_in_english: Optional entity descriptions in English
            
        Returns:
            str: The formatted prompt
        """
        # Get language information
        language_name = self.lang_info["language_name"]
        language_native = self.lang_info["language_native"] 
        
        # Create example of NER for the language
        language_examples = self.lang_info.get("ner_examples", [])
        example_text = ""
        if language_examples and len(language_examples) > 0:
            example = language_examples[0]
            example_text = f"""
Example in {language_name}:
Text: {example['text']}
Entities: {example['entities']}
"""
        
        # Define the entity types we want to identify
        entity_types = "person (PER), location (LOC), organization (ORG), date (DATE)"
        
        # Build the prompt
        prompt = f"""You are an expert in {language_name} ({language_native}) natural language processing.

Your task is to identify named entities in the following text in {language_name}. 
Look for the following entity types: {entity_types}.

{example_text}

For each entity you identify, provide the entity text, its entity type (PER, LOC, ORG, DATE), and the character offsets where it appears in the text.

Text: {text}

Think through this step by step:
1. First, scan the text to identify potential named entities.
2. For each potential entity, determine if it's a person, location, organization, or date.
3. Find the exact character position where each entity starts and ends in the text.

Now provide your answer in the following JSON format:
{{
    "entities": [
        {{"text": "entity_text", "type": "entity_type", "start": start_position, "end": end_position}},
        ...
    ]
}}

Ensure that your character positions are accurate. The start position is the index of the first character of the entity, and the end position is the index after the last character of the entity.
"""
        return prompt

    def parse_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the model's response to extract identified entities.
        
        Args:
            response: The model's response to the prompt
            
        Returns:
            List of extracted entities
        """
        # Find JSON-like structure in the response
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        json_matches = re.findall(json_pattern, response)
        
        if not json_matches:
            return []
        
        # Try all potential JSON matches
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if "entities" in data and isinstance(data["entities"], list):
                    return data["entities"]
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON with entities is found, try a more lenient approach
        try:
            # Look for entities in a more flexible way
            entities = []
            entity_pattern = r'"text":\s*"([^"]+)",\s*"type":\s*"([^"]+)",\s*"start":\s*(\d+),\s*"end":\s*(\d+)'
            for match in re.finditer(entity_pattern, response):
                text, entity_type, start, end = match.groups()
                entity = {
                    "text": text,
                    "type": entity_type,
                    "start": int(start),
                    "end": int(end)
                }
                entities.append(entity)
            
            if entities:
                return entities
        except Exception:
            pass
        
        return []

class NEREvaluator:
    """
    A class to evaluate NER performance with different approaches (direct and CoTR).
    Supports both few-shot and zero-shot evaluation.
    """
    
    def __init__(self, lang_code="sw", approach="direct", num_shots=0):
        """
        Initialize the evaluator with the given language and approach.
        
        Args:
            lang_code: Language code ("sw", "ha", etc.)
            approach: Approach to use ("direct", "cotr-single", "cotr-multi")
            num_shots: Number of examples to use (0 for zero-shot)
        """
        self.lang_code = lang_code
        self.approach = approach
        self.num_shots = num_shots
        self.instructions = self.generate_language_specific_instructions()
        
        if num_shots > 0:
            self.few_shot_examples = self._load_few_shot_examples()
        else:
            self.few_shot_examples = []
    
    def format_ner_sample(self, tokens, entities=None, include_markup=True):
        """
        Format a NER sample for prompting.
        
        Args:
            tokens: List of tokens
            entities: List of entity dictionaries with start, end, entity_type
            include_markup: Whether to include entity markup
            
        Returns:
            Formatted sample string
        """
        if entities is None or not include_markup:
            return " ".join(tokens)
        
        # Create a marked-up sample with entity tags
        text_parts = []
        current_idx = 0
        
        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        for entity in sorted_entities:
            start_idx = entity['start']
            end_idx = entity['end']
            entity_type = entity['entity_type']
            
            # Add text before the entity
            if start_idx > current_idx:
                text_parts.append(" ".join(tokens[current_idx:start_idx]))
            
            # Add the entity with markup
            entity_text = " ".join(tokens[start_idx:end_idx])
            text_parts.append(f"[{entity_type}: {entity_text}]")
            
            current_idx = end_idx
        
        # Add any remaining text
        if current_idx < len(tokens):
            text_parts.append(" ".join(tokens[current_idx:]))
        
        return " ".join(text_parts)
    
    def parse_ner_response(self, text, tokens):
        """
        Parse the NER response from the model and align with tokens.
        Now using the improved extract_entities function from the baseline implementation.
        
        Args:
            text: Model response text
            tokens: Original tokens for alignment
            
        Returns:
            List of entity dictionaries with start, end, entity_type
        """
        # Import the improved extract_entities function from the baseline implementation
        from src.experiments.baseline.ner.ner_baseline import extract_entities
        
        # Use the improved function to extract entities
        extracted_entities = extract_entities(text, lang_code=self.lang_code)
        
        # Now convert the extracted entities to the expected format with token indices
        aligned_entities = []
        
        # Convert tokens to a string for easier matching
        text_str = " ".join(tokens).lower()
        
        for entity in extracted_entities:
            entity_text = entity["text"].lower()
            entity_type = entity["type"]
            
            # Find token indices that correspond to this entity
            spans = self.find_entity_spans(entity_text, tokens)
            
            if spans:
                for start, end in spans:
                    aligned_entities.append({
                        "start": start,
                        "end": end,
                        "entity_type": entity_type
                    })
        
        # Filter out invalid or overlapping entities (optional)
        # This could be done here if needed
        
        return aligned_entities
    
    def generate_language_specific_instructions(self):
        """
        Generate language-specific instructions for NER prompts.
        
        Returns:
            Dictionary with instruction components
        """
        # Default English instructions
        base_instruction = """Identify all named entities in the text below. 
Tag each entity with one of these categories: PER (person), ORG (organization), LOC (location), or DATE (date).
For each entity you find, use EXACTLY this format: [TYPE: entity text]

Examples of correct format:
[PER: John Smith]
[ORG: Microsoft]
[LOC: New York]
[DATE: Friday]

Categories explanation:
- PER: Real people, fictional characters, named individuals
- ORG: Companies, institutions, governments, political parties, teams
- LOC: Countries, cities, regions, geographic features, buildings 
- DATE: Specific dates, days of the week, months, years, time periods"""
        
        zero_shot_tips = """IMPORTANT: Follow these guidelines:
1. Read the entire text carefully to understand the context
2. Look for proper nouns (capitalized words) and specific dates
3. Identify people's names (PER), organization names (ORG), place names (LOC), and dates (DATE)
4. Use the exact [TYPE: entity text] format for each entity you find
5. Include complete entity names (e.g., "John Smith" not just "John")
6. If an entity appears multiple times, tag each occurrence
7. Be thorough and don't miss any entities
8. Make sure to identify all dates, even if they're not obvious
9. For organizations, include the full name (e.g., "United Nations" not just "UN")
10. For people with titles, include the title (e.g., "President Joe Biden")"""
        
        example_text = """John Smith from Microsoft visited New York last Friday."""
        example_entities = """[PER: John Smith] [ORG: Microsoft] [LOC: New York] [DATE: Friday]"""
        
        # Language-specific instructions
        if self.lang_code == "sw":  # Swahili
            base_instruction = """Tambua vitu vyote vilivyotajwa kwenye maandishi yafuatayo.
Weka kila kitu katika mojawapo ya makundi haya: PER (mtu), ORG (shirika), LOC (mahali), au DATE (tarehe).
Kwa kila kitu unachopata, panga jibu lako kama [AINA: maandishi ya kitu].

Maelezo ya makundi:
- PER: Watu halisi, wahusika wa hadithi, watu wanaotambulika kwa majina
- ORG: Kampuni, taasisi, serikali, vyama vya siasa, timu
- LOC: Nchi, miji, mikoa, sehemu za kijiografia, majengo
- DATE: Tarehe mahususi, siku za wiki, miezi, miaka, vipindi vya muda"""
            
            zero_shot_tips = """MUHIMU: Fuata miongozo hii:
1. Soma maandishi yote kwa makini kuelewa muktadha
2. Angalia majina mahususi (maneno yenye herufi kubwa) na tarehe mahususi
3. Tambua majina ya watu (PER), majina ya mashirika (ORG), majina ya mahali (LOC), na tarehe (DATE)
4. Tumia muundo sahihi wa [AINA: maandishi ya kitu] kwa kila kitu unachokipata
5. Jumuisha majina kamili ya vitu (mfano, "John Smith" sio "John" tu)
6. Ikiwa kitu kinatokea mara kadhaa, weka lebo kwa kila tokeo
7. Kuwa makini na usikose kitu chochote
8. Hakikisha unatambua tarehe zote, hata kama sio wazi
9. Kwa mashirika, jumuisha jina kamili (mfano, "Umoja wa Mataifa" sio "UN" tu)
10. Kwa watu wenye vyeo, jumuisha cheo (mfano, "Rais William Ruto")"""
            
            example_text = """James Mwangi kutoka Equity Bank alisafiri kwenda Nairobi Jumapili iliyopita."""
            example_entities = """[PER: James Mwangi] [ORG: Equity Bank] [LOC: Nairobi] [DATE: Jumapili iliyopita]"""
        
        elif self.lang_code == "ha":  # Hausa
            base_instruction = """Gano duk wata sunan da aka ambata a cikin rubutun da ke kasa.
Ka sanya kowane suna a cikin daya daga cikin wadannan nau'ikan: PER (mutum), ORG (kungiya), LOC (wuri), ko DATE (kwanan wata).
Domin kowanne suna da ka gano, tsara amsarka kamar [NAWA: rubutun sunan].

Bayani game da nau'ikan:
- PER: Mutanen gaskiya, mutanen labari, mutane masu suna
- ORG: Kamfanoni, cibiyoyi, gwamnatoci, jam'iyyun siyasa, kungiyoyi
- LOC: Kasashe, birane, yankunan, wurare na musamman, gine-gine
- DATE: Kwanan wata, ranakun mako, watanni, shekaru, lokutan musamman"""
            
            zero_shot_tips = """MUHIMMI: Bi wadannan kagaggun:
1. Karanta duk rubutun da kyau don fahimtar jigo
2. Nemo sunaye musamman (kalmomi masu haruffa manyan farko) da kwanan wata musamman
3. Gano sunayen mutane (PER), sunayen kungiyoyi (ORG), sunayen wurare (LOC), da kwanan wata (DATE)
4. Yi amfani da tsarin daidai na [NAWA: rubutun sunan] don kowanne suna da ka gano
5. Hada cikakken sunayen (misali, "John Smith" ba "John" kawai ba)
6. Idan wani suna ya bayyana sau da yawa, sanya lamba ga kowanne bayyanawa
7. Ka yi hankali kada ka rasa wani suna
8. Tabbatar da cewa ka gano duk kwanaki, ko da ba a bayyana su a fili ba
9. Don kungiyoyi, hada sunan cikakke (misali, "Majalisar Dinkin Duniya" ba "UN" kawai ba)
10. Don mutane masu matsayi, hada da matsayin (misali, "Gwamna Abba Yusuf")"""
            
            example_text = """Aliyu Muhammadu daga Dangote Group ya ziyarci Kano ranar Litinin don ganawa da Gwamna Abba Yusuf da wakilin MTN Nigeria."""
            example_entities = """[PER: Aliyu Muhammadu] [ORG: Dangote Group] [LOC: Kano] [DATE: ranar Litinin] [PER: Abba Yusuf] [PER: Gwamna Abba Yusuf] [ORG: MTN Nigeria]"""
        
        # Return dictionary with all instruction components
        return {
            "base_instruction": base_instruction,
            "zero_shot_tips": zero_shot_tips,
            "example_text": example_text,
            "example_entities": example_entities
        }
    
    def find_entity_spans(self, text, tokens):
        """
        Find token indices that correspond to an entity text.
        
        Args:
            text: Entity text to find
            tokens: List of tokens to search in
            
        Returns:
            List of (start, end) token index tuples
        """
        text = text.lower()
        spans = []
        
        # Create a string version of tokens for matching
        token_str = " ".join(tokens).lower()
        
        # For exact matches
        for i in range(len(tokens)):
            for j in range(i+1, len(tokens)+1):
                span_text = " ".join(tokens[i:j]).lower()
                if span_text == text:
                    spans.append((i, j))
        
        # If no exact matches, try fuzzy matching
        if not spans:
            # Simple partial match if token sequence contains entity text
            for i in range(len(tokens)):
                for j in range(i+1, min(len(tokens)+1, i+10)):  # Limit span size for efficiency
                    span_text = " ".join(tokens[i:j]).lower()
                    if text in span_text or span_text in text:
                        # Check if there's substantial overlap
                        if len(text) > 3 and len(span_text) > 3:
                            # Simple overlap heuristic
                            overlap = set(text.split()).intersection(set(span_text.split()))
                            if len(overlap) >= min(2, len(text.split()) // 2):
                                spans.append((i, j))
        
        return spans

# --- Experiment Runner --- #

# Comment out or remove the previous run_ner_cotr_experiment function
# def run_ner_cotr_experiment(...): ...

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
                            pipeline_type=pipeline_type,
                            use_few_shot=use_few_shot,
                            temperature=args.temperature if hasattr(args, 'temperature') else 0.3, # Provide defaults if not in args
                            top_p=args.top_p if hasattr(args, 'top_p') else 0.9,
                            top_k=args.top_k if hasattr(args, 'top_k') else 40,
                            max_tokens=args.max_ner_new_tokens, # Ensure this arg exists or use a default
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

                    # Calculate average metrics (already done inside evaluate_ner_cotr, but we can re-access)
        avg_f1 = lang_results_df["f1_score"].mean()
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
                        # Add COMET score if it's part of lang_results_df and needed here
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

def generate_single_prompt_ner_cotr(text_lrl: str, lang_code: str, use_few_shot: bool = True) -> str:
    """Generate a single prompt for the entire CoTR NER pipeline.
    Instructs the model to translate LRL->EN, find EN entities, translate entities back to LRL.
    
    Args:
        text_lrl: The text in the Low-Resource Language
        lang_code: The language code (e.g., 'sw', 'ha')
        use_few_shot: Whether to include few-shot examples demonstrating the process
        
    Returns:
        Formatted single-prompt CoTR NER prompt
    """
    lrl_name = lang_names.get(lang_code, lang_code)
    
    # Simplified Instructions
    instructions = f"""Text ({lrl_name}): '{text_lrl}'

Your task is to perform Named Entity Recognition on this {lrl_name} text.
Follow these steps carefully:
1. Translate the text to English.
2. Identify entities (PER, ORG, LOC, DATE) in the English text.
3. Translate ONLY the identified English entity texts back to {lrl_name}.
4. Provide ONLY the final list of {lrl_name} entities, each on a new line, in the format [TYPE: {lrl_name} entity text].

Do NOT include the English translation or English entities in your final response. Only output the {lrl_name} entities list."""

    # Few-shot examples showing ONLY the final expected output
    examples = ""
    if use_few_shot:
        if lang_code == 'sw':
            # Example Input (for context, not shown to model): 'James Mwangi kutoka Equity Bank alisafiri kwenda Nairobi Jumapili iliyopita.'
            examples = f"""Example:
[PER: James Mwangi]
[ORG: Equity Bank]
[LOC: Nairobi]
[DATE: Jumapili iliyopita]
"""
        elif lang_code == 'ha':
            # Example Input (for context, not shown to model): 'Aliyu Muhammadu daga Dangote Group ya ziyarci Kano ranar Litinin.'
            examples = f"""Example:
[PER: Aliyu Muhammadu]
[ORG: Dangote Group]
[LOC: Kano]
[DATE: ranar Litinin]
"""
        # Add other language examples if needed

    # Construct the prompt
    if use_few_shot and examples:
        prompt = f"{instructions}\n\n{examples}\n\nFinal Answer ({lrl_name} Entities Only):" # Clarified final answer request
    else:
        prompt = f"{instructions}\n\nFinal Answer ({lrl_name} Entities Only):"
        
    return prompt

def extract_lrl_entities_from_single_prompt(response_text: str, lang_code: str) -> Tuple[str, List[Dict[str, str]], List[Dict[str, str]]]:
    """Extracts English translation, English entities, and LRL entities from the single prompt response.
    
    Args:
        response_text: The full raw response from the model.
        lang_code: The LRL code.
        
    Returns:
        Tuple: (english_translation, english_entities, lrl_entities)
    """
    english_translation = "[Extraction Failed]"
    english_entities_raw = ""
    lrl_entities_raw = ""
    english_entities = []
    lrl_entities = []
    lrl_name = lang_names.get(lang_code, lang_code)

    # Extract sections using regex, allowing for optional whitespace and case insensitivity
    en_trans_match = re.search(r"English Translation:\s*(.*?)(?:\n\n|English Entities:|$)", response_text, re.IGNORECASE | re.DOTALL)
    en_entities_match = re.search(r"English Entities:\s*(.*?)(?:\n\n|{lrl_name} Entities:|$)", response_text, re.IGNORECASE | re.DOTALL)
    lrl_entities_match = re.search(rf"{lrl_name} Entities:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

    if en_trans_match:
        english_translation = en_trans_match.group(1).strip()
    else:
        print(f"WARN: Could not extract 'English Translation:' section.")
        
    if en_entities_match:
        english_entities_raw = en_entities_match.group(1).strip()
        # Use the baseline extract_entities to parse the English entities section
        english_entities = extract_entities_baseline(english_entities_raw, lang_code='en') # Extract from the raw string
    else:
        print(f"WARN: Could not extract 'English Entities:' section.")
        
    if lrl_entities_match:
        lrl_entities_raw = lrl_entities_match.group(1).strip()
        # --- Use robust extract_entities for LRL section --- 
        print(f"DEBUG: Extracted raw LRL entity string: {lrl_entities_raw[:100]}...") # Debug
        lrl_entities = extract_entities_baseline(lrl_entities_raw, lang_code=lang_code)
        print(f"DEBUG: Parsed LRL entities: {lrl_entities}") # Debug
    else:
        print(f"WARN: Could not extract '{lrl_name} Entities:' section. LRL entities list will be empty.")

    # Fallback attempt: If LRL section failed, try parsing the *entire* response for LRL entities
    if not lrl_entities:
        print(f"WARN: LRL section extraction failed. Trying to parse entire response for LRL entities...")
        lrl_entities = extract_entities_baseline(response_text, lang_code=lang_code)
        if lrl_entities:
            print(f"DEBUG: Found LRL entities via fallback: {lrl_entities}")

    return english_translation, english_entities, lrl_entities

# Function definition for evaluate_ner_cotr_single_prompt
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
    repetition_penalty: float = 1.1 # Add missing param
) -> pd.DataFrame:
    """
    Evaluate NER using a single CoTR prompt.
    """
    results = []
    model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
    safe_max_input_length = min(model_max_len, 8192) 
    # max_new_tokens_for_chain = max(512, max_tokens * 6 + 300) # Max tokens controlled by max_tokens arg now

    gen_temperature = temperature
    gen_top_p = top_p
    gen_top_k = top_k
    gen_repetition_penalty = repetition_penalty 
    
    # Apply model specific adjustments if needed (consistent with multi-prompt/baseline)
    if "aya" in model_name.lower():
        gen_temperature = max(0.1, gen_temperature * 0.9)
    elif "qwen" in model_name.lower():
        gen_top_p = max(0.7, gen_top_p * 0.9)
        gen_top_k = 35

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} samples ({model_name} CoTR Single Prompt)"):
        text_lrl = row['text']
        ground_truth_entities = row['entities']
        start_time = time.time()
        
        try:
            prompt = generate_single_prompt_ner_cotr(text_lrl, lang_code, use_few_shot)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=safe_max_input_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens, # Use the max_tokens passed for the whole output
                    do_sample=True,
                    temperature=gen_temperature,
                    top_p=gen_top_p,
                    top_k=gen_top_k,
                    repetition_penalty=gen_repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            
            # Extract results
            english_translation, english_entities, predicted_entities = extract_lrl_entities_from_single_prompt(response_text, lang_code)
            
            runtime = time.time() - start_time

            # Format ground truth for consistency in metrics
            formatted_ground_truth = []
            for entity in ground_truth_entities:
                if "start" in entity and "end" in entity:
                    entity_text = " ".join(row["tokens"][entity["start"]:entity["end"]])
                    entity_type = entity["entity_type"]
                else:
                    entity_text = entity.get("text", entity.get("entity", ""))
                    entity_type = entity.get("type", "")
                formatted_ground_truth.append({"entity": entity_text, "type": entity_type})

            # Calculate metrics for this sample
            metrics = calculate_ner_metrics(formatted_ground_truth, predicted_entities)
            
            results.append({
                "text": text_lrl,
                "ground_truth_entities": formatted_ground_truth,
                "predicted_entities": predicted_entities,
                "english_translation": english_translation,
                "english_entities": english_entities,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "runtime_seconds": runtime,
                "language": lang_code,
                "pipeline_type": "single_prompt",
                "shot_type": "few-shot" if use_few_shot else "zero-shot"
            })

        except Exception as e:
            print(f"ERROR processing sample {row.get('id', idx)} (Single Prompt NER): {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "text": text_lrl,
                "ground_truth_entities": formatted_ground_truth, # Use formatted GT
                "predicted_entities": [], # Empty list on error
                "english_translation": "[Error]",
                "english_entities": [],
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "runtime_seconds": time.time() - start_time,
                "language": lang_code,
                "pipeline_type": "single_prompt",
                "shot_type": "few-shot" if use_few_shot else "zero-shot",
                "error": str(e)
            })
            continue

    if not results:
        print(f"WARNING: No results were successfully processed for {lang_code} with {model_name} using CoTR Single Prompt NER.")
        return pd.DataFrame()
        
    return pd.DataFrame(results) 