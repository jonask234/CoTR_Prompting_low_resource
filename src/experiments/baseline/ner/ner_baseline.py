import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import json
from typing import Dict, List, Any, Tuple, Optional
import time
import logging
from datasets import load_dataset
import gc
from collections import Counter

ENTITY_TYPES = ["PER", "ORG", "LOC", "DATE"]

logger = logging.getLogger(__name__)

# Define global English few-shot examples for NER
ENGLISH_FEW_SHOT_EXAMPLES_NER_TYPE_TEXT_FORMAT = [
    {"text": "Angela Merkel visited Paris on September 1st, 2021 with a delegation from the European Union.", 
     "entities_str": "[PER: Angela Merkel] [LOC: Paris] [DATE: September 1st, 2021] [ORG: European Union]"},
    {"text": "The quick brown fox jumps over the lazy dog in New York.", 
     "entities_str": "[LOC: New York]"},
    {"text": "There are no entities here.",
     "entities_str": "[NO_ENTITIES_FOUND]"}
]

# Define LRL-specific instructions for NER
# LRL_INSTRUCTIONS_NER = {
#     "swa": {
#         "instruction": "Tafadhali toa huluki zilizotajwa (Mtu, Shirika, Mahali, Tarehe) kutoka kwa maandishi yafuatayo. Pato linapaswa kuwa orodha ya kamusi, ambapo kila kamusi inawakilisha huluki na ina funguo za 'entity' na 'type'.",
#         "examples_header": "Hapa kuna mifano (mifano hii ni kwa Kiingereza):",
#         "analyze_header": "Chambua maandishi yafuatayo na utoe huluki:",
#         "text_label": "Maandishi",
#         "entities_label": "Huluki"
#     },
#     "yor": {
#         "instruction": "Jọwọ yọ awọn nkan ti a darukọ (Eniyan, Ajo, Ipo, Ọjọ) jade lati inu ọrọ atẹle. Ijade yẹ ki o jẹ atokọ ti awọn iwe-itumọ, nibiti iwe-itumọ kọọkan duro fun nkan kan ti o ni awọn bọtini 'entity' ati 'type'.",
#         "examples_header": "Eyi ni diẹ ninu awọn apẹẹrẹ (awọn apẹẹrẹ wọnyi wa ni Gẹẹsi):",
#         "analyze_header": "Ṣe itupalẹ ọrọ atẹle ki o yọ awọn nkan jade:",
#         "text_label": "Ọrọ",
#         "entities_label": "Awọn nkan"
#     }
# }

# Define LRL-specific few-shot examples using the [TYPE: Entity Text] format
LRL_FEW_SHOT_EXAMPLES_NER_TYPE_TEXT_FORMAT = {
    "sw": [
        {
            "text": "Rais Samia Suluhu Hassan alitembelea Nairobi mwezi Januari.",
            "entities_str": "[PER: Samia Suluhu Hassan] [LOC: Nairobi] [DATE: mwezi Januari]"
        },
        {
            "text": "Kampuni ya Yetu Microfinance Bank PLC iliorodheshwa katika Soko la Hisa la Dar es Salaam (DSE) mwaka 2015.",
            "entities_str": "[ORG: Yetu Microfinance Bank PLC] [LOC: Soko la Hisa la Dar es Salaam] [ORG: DSE] [DATE: 2015]"
        },
        {
            "text": "Hakuna huluki hapa.",
            "entities_str": "[NO_ENTITIES_FOUND]"
        }
    ],
    "ha": [
    {
            "text": "Shugaba Bola Tinubu zai je Kano ranar Litinin.",
            "entities_str": "[PER: Bola Tinubu] [LOC: Kano] [DATE: ranar Litinin]"
        },
        {
            "text": "Kamfanin Dangote Cement ya sanar da ribar Naira biliyan 300 a shekarar 2022.",
            "entities_str": "[ORG: Dangote Cement] [MONEY: Naira biliyan 300] [DATE: 2022]" # Note: MONEY type might need to be added to ENTITY_TYPES if consistently used
        },
        {
            "text": "Babu wasu abubuwa anan.", # Roughly "No entities here"
            "entities_str": "[NO_ENTITIES_FOUND]"
        }
    ]
    # Add other languages here if LRL examples are available
}

# Fallback English few-shot examples if LRL examples are not defined for a language
# but LRL instructions are requested. This uses the same structure as LRL_FEW_SHOT_EXAMPLES_NER_TYPE_TEXT_FORMAT.
FALLBACK_ENGLISH_FEW_SHOT_EXAMPLES_NER_TYPE_TEXT_FORMAT = ENGLISH_FEW_SHOT_EXAMPLES_NER_TYPE_TEXT_FORMAT

# LRL Instructions (defined globally)
LRL_INSTRUCTIONS = {
    "sw": {
        "task": "Jukumu lako ni kutambua huluki zilizo na majina katika maandishi yafuatayo ya Kiswahili. Huluki zinazojulikana ni: PER (Person), ORG (Organization), LOC (Location), DATE (Date).",
        "format": "Wasilisha matokeo yako YOTE kama orodha ya huluki, kila moja katika muundo wa [AINA: Maandishi ya Huluki], zikitenganishwa na nafasi au mistari mipya.",
        "example_format": "Mfano wa muundo: [PER: Juma Kaseja] [LOC: Dar es Salaam] [DATE: Julai 2024].",
        "no_entities": "Kama hakuna huluki zilizopatikana, andika maneno haya hasa: [NO_ENTITIES_FOUND].",
        "examples_header": "\\n\\nMifano (hakikisha matokeo yako yanafuata muundo huu kikamilifu):\\n",
        "analyze_header": "\\n\\nSasa, changanua maandishi yafuatayo ya Kiswahili na utoe huluki katika muundo uliobainishwa ([AINA: Maandishi ya Huluki] au [NO_ENTITIES_FOUND]):",
        "text_label": "Maandishi (Kiswahili)",
        "entities_label": "Huluki"
    },
    "swa": {
        "task": "Jukumu lako ni kutambua huluki zilizo na majina katika maandishi yafuatayo ya Kiswahili. Huluki zinazojulikana ni: PER (Person), ORG (Organization), LOC (Location), DATE (Date).",
        "format": "Wasilisha matokeo yako YOTE kama orodha ya huluki, kila moja katika muundo wa [AINA: Maandishi ya Huluki], zikitenganishwa na nafasi au mistari mipya.",
        "example_format": "Mfano wa muundo: [PER: Juma Kaseja] [LOC: Dar es Salaam] [DATE: Julai 2024].",
        "no_entities": "Kama hakuna huluki zilizopatikana, andika maneno haya hasa: [NO_ENTITIES_FOUND].",
        "examples_header": "\\n\\nMifano (hakikisha matokeo yako yanafuata muundo huu kikamilifu):\\n",
        "analyze_header": "\\n\\nSasa, changanua maandishi yafuatayo ya Kiswahili na utoe huluki katika muundo uliobainishwa ([AINA: Maandishi ya Huluki] au [NO_ENTITIES_FOUND]):",
        "text_label": "Maandishi (Kiswahili)",
        "entities_label": "Huluki"
    },
    "ha": {
        "task": "Aikin ku shine gano sunayen da aka ambata a cikin rubutun Hausa da ke ƙasa. Ire-iren sunayen da aka sani sune: PER (Person), ORG (Organization), LOC (Location), DATE (Date).",
        "format": "Gabatar da DUKKAN sakamakon ku a matsayin jerin sunaye, kowanne a cikin tsarin [NAU'I: Rubutun Suna], waɗanda aka raba da sarari ko sabbin layuka.",
        "example_format": "Misalin tsari: [PER: Musa Aliyu] [LOC: Kano] [DATE: Yuli 2024].",
        "no_entities": "Idan ba a sami wasu sunaye ba, rubuta wannan jimlar daidai: [NO_ENTITIES_FOUND].",
        "examples_header": "\\n\\nMisalai (tabbatar cewa sakamakon ku ya bi wannan tsarin sosai):\\n",
        "analyze_header": "\\n\\nYanzu, yi nazarin rubutun Hausa da ke tafe kuma samar da sunayen a cikin tsarin da aka ƙayyade ([NAU'I: Rubutun Suna] ko [NO_ENTITIES_FOUND]):",
        "text_label": "Rubutu (Hausa)",
        "entities_label": "Sunaye"
    }
}

def initialize_model(model_name: str) -> Tuple[Any, Any]:
    """
    Initialize a model and tokenizer.
    
    Args:
        model_name: Name of the model to initialize
    
    Returns:
        tokenizer, model
    """
    logger.info(f"Initializing {model_name}...")
    cache_path = "/work/bbd6522/cache_dir"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path
    )
    
    model = None
    try:
        logger.info(f"Attempting to load {model_name} with device_map='auto', torch_dtype=torch.float16, and low_cpu_mem_usage=True.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_path,
            low_cpu_mem_usage=True
        )
        logger.info(f"Successfully loaded {model_name} using device_map='auto' and torch_dtype=torch.float16.")

    except Exception as e_load:
        logger.error(f"CRITICAL: Failed to load model {model_name} with device_map='auto' and float16. Error: {e_load}", exc_info=True)
        if model is not None:
            del model
            model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache after failed model load attempt.")
        raise RuntimeError(f"Could not load model {model_name} due to: {e_load}") from e_load

    # Pad token handling (only if model loaded successfully)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer.pad_token to eos_token ('{tokenizer.eos_token}') for {model_name}")
        else:
            new_pad_token = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': new_pad_token})
            # Resize embeddings only if model is not None
            if model:
                try:
                    model.resize_token_embeddings(len(tokenizer))
                    logger.info(f"Added new pad_token: '{new_pad_token}' and resized embeddings for {model_name}.")
                except Exception as e_resize:
                    logger.error(f"Failed to resize token embeddings for {model_name} after adding new pad_token: {e_resize}")
            else: # Should not happen if loading failed and raised error
                 logger.warning(f"Model for {model_name} is None after loading attempt, cannot resize embeddings for new pad_token.")


    if model.config.pad_token_id is None:
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info(f"Set model.config.pad_token_id to tokenizer.pad_token_id ({tokenizer.pad_token_id}) for {model_name}")
        elif tokenizer.eos_token_id is not None:
            model.config.pad_token_id = tokenizer.eos_token_id
            logger.info(f"Set model.config.pad_token_id to tokenizer.eos_token_id ({tokenizer.eos_token_id}) for {model_name}")
        else:
            # This case should be rare for well-configured models/tokenizers
            logger.warning(f"Could not set model.config.pad_token_id for {model_name} as tokenizer has no pad_token_id or eos_token_id.")
            # Potentially set a default like 0, but this can be risky.
            # model.config.pad_token_id = 0 
            # logger.warning(f"Defaulted model.config.pad_token_id to 0 for {model_name}. VERIFY THIS IS CORRECT.")


    # Final check for pad_token_id's type, crucial for `generate`
    if model.config.pad_token_id is not None and not isinstance(model.config.pad_token_id, int):
        logger.error(f"CRITICAL: model.config.pad_token_id for {model_name} is {model.config.pad_token_id} (type: {type(model.config.pad_token_id)}), which is not an int. This will likely cause errors in model.generate().")
        # Attempt to fix if possible, e.g. if eos_token_id is an int
        if tokenizer.eos_token_id is not None and isinstance(tokenizer.eos_token_id, int):
            model.config.pad_token_id = tokenizer.eos_token_id
            logger.warning(f"Attempted fix: Set model.config.pad_token_id to tokenizer.eos_token_id ({tokenizer.eos_token_id}) for {model_name}.")
        else:
            logger.error(f"Further CRITICAL: Cannot automatically fix model.config.pad_token_id for {model_name} to an integer value.")


    logger.info(f"Model {model_name} initialization finished. Tokenizer pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id}). Model config pad_token_id: {model.config.pad_token_id}")
    return tokenizer, model

def generate_ner_prompt(text: str, lang_code: str = "en", model_name: str = "", use_few_shot: bool = True) -> str:
    """
    Generate an NER prompt with English instructions.
    Focuses on the [TYPE: Entity Text] format for output.
    This is used when prompt_in_lrl is False.
    """
    processed_text = text.strip()
    entity_types_desc = "PER (Person), ORG (Organization), LOC (Location), DATE (Date)"
    no_entities_marker = "[NO_ENTITIES_FOUND]"

    instruction = (
        f"Your task is to identify named entities in the text below. The text language is '{lang_code}'. " # Clarify text language
        f"Extract all entities corresponding to {entity_types_desc}. "
        f"Present your ENTIRE output as a list of entities, each in the format [TYPE: Entity Text], separated by spaces or newlines. "
        f"Example of format: [PER: John Doe] [LOC: Paris] [DATE: July 2024]. "
        f"If no entities are found, output the exact phrase: {no_entities_marker}."
    )

    prompt = instruction
    
    if use_few_shot:
        prompt += "\\n\\nExamples (ensure your output strictly follows this format for the actual task):\\n"
        # English examples using the [TYPE: Entity Text] format
        for ex in ENGLISH_FEW_SHOT_EXAMPLES_NER_TYPE_TEXT_FORMAT: # Updated to use the new example list name
            prompt += f"\\nText (English): '{ex['text']}'\\nEntities: {ex['entities_str']}" # lang_code is 'en' for these examples
    else: # Zero-shot
        prompt += f"\\n\\nNow, analyze the following text and provide the entities in the specified format ([TYPE: Entity Text] or {no_entities_marker}):"

    prompt += f"\\n\\nText ({lang_code}): '{processed_text}'\\n\\nEntities:" # Ensure 'Entities:' is the final cue
    return prompt

# This function needs to be defined or imported if used by generate_ner_prompt
# For now, assuming it's not critical or handled elsewhere.
# def preprocess_text(text: str, lang_code: str) -> str:
#     return text.strip()

def generate_lrl_instruct_ner_prompt(text: str, lang_code: str = "sw", model_name: str = "", use_few_shot: bool = True) -> str:
    """
    Generates an NER prompt with instructions AND few-shot examples in the specified 
    Low-Resource Language (LRL) if available.
    Outputs entities in the format [TYPE: Entity Text].
    This is used when prompt_in_lrl is True.
    """
    processed_text = text.strip()

    # Use the globally defined LRL_INSTRUCTIONS dictionary
    current_instructions = LRL_INSTRUCTIONS.get(lang_code)
    if not current_instructions:
        # Fallback to English instructions if LRL instructions are not defined for the lang_code
        logger.warning(f"LRL instructions not defined for lang_code '{lang_code}'. Falling back to English-instructed prompt structure via generate_ner_prompt but with LRL text still indicated.")
        # Call the English prompt generator, but it will still process LRL text.
        # It will use English few-shot examples by default.
        return generate_ner_prompt(text, lang_code, model_name, use_few_shot)

    instruction = (
        f"{current_instructions['task']} "
        f"{current_instructions['format']} "
        f"{current_instructions['example_format']} "
        f"{current_instructions['no_entities']}"
    )
    
    prompt = instruction

    if use_few_shot:
        prompt += current_instructions['examples_header']
        # Per user requirement, ALWAYS use English few-shot examples for the baseline
        # to create a cross-lingual baseline, even when instructions are in LRL.
        for ex in ENGLISH_FEW_SHOT_EXAMPLES_NER_TYPE_TEXT_FORMAT:
            # The prompt text label should still indicate the example is in English
            prompt += f"\\nText (English): '{ex['text']}'\\n{current_instructions['entities_label']}: {ex['entities_str']}"

    else: # Zero-shot
        prompt += current_instructions['analyze_header']

    prompt += f"\\n\\n{current_instructions['text_label']}: '{processed_text}'\\n\\n{current_instructions['entities_label']}:"
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
) -> Tuple[List[Dict[str, str]], float, str]: # Return type updated
    start_time = time.time()
    
    logger.debug(f"DEBUG process_ner_baseline: Type of tokenizer arg: {type(tokenizer)}")
    logger.debug(f"DEBUG process_ner_baseline: Type of model arg: {type(model)}")

    if prompt_in_lrl:
        prompt = generate_lrl_instruct_ner_prompt(text, lang_code=lang_code, model_name=model_name, use_few_shot=use_few_shot)
    else:
        prompt = generate_ner_prompt(text, lang_code=lang_code, model_name=model_name, use_few_shot=use_few_shot)
        
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    
    # Determine target device for inputs
    target_device = model.device # Default
    if hasattr(model, 'hf_device_map'):
        try:
            # For models loaded with device_map, the embedding layer's device is a good target
            target_device = model.get_input_embeddings().weight.device
            logger.debug(f"Model has hf_device_map. Using embedding layer device: {target_device}")
        except AttributeError:
            logger.debug(f"Model has hf_device_map but get_input_embeddings().weight.device failed. Using model.device: {model.device}")
            pass # Fall back to model.device if get_input_embeddings fails

    inputs_on_device_dict = {k: v.to(target_device) for k, v in inputs.items()}
    
    logger.debug(f"process_ner_baseline: target_device for inputs: {target_device}")
    logger.debug(f"process_ner_baseline: input_ids.device after move: {inputs_on_device_dict['input_ids'].device}")
    logger.debug(f"process_ner_baseline: model.device: {model.device}")


    # Prepare generation kwargs
    # Ensure do_sample is consistent with temperature (common HuggingFace practice)
    effective_do_sample = do_sample if temperature > 1e-5 else False
    
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
        "do_sample": effective_do_sample
    }
    if effective_do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
        generation_kwargs["top_k"] = top_k
    
    raw_output = "[GEN_ERROR: NER_Baseline_Generation_Failed]"
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs_on_device_dict['input_ids'],
                attention_mask=inputs_on_device_dict.get('attention_mask'), # Ensure attention_mask is also on target_device
                **generation_kwargs
            )
        
        # Decode only the newly generated tokens
        input_len = inputs_on_device_dict["input_ids"].shape[1]
        raw_output = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        
    except RuntimeError as e:
        logger.error(f"RuntimeError in process_ner_baseline for {model_name} ({lang_code}): {e}", exc_info=True)
        raw_output = f"[GEN_ERROR: RuntimeError: {e}]"
        # If the error is the device mismatch, this log helps confirm it.
        # For Qwen, the error "Expected all tensors to be on the same device..." indicates inputs might not be on the same device as some model parts.
        # Log devices again right before error if possible, or here.
        logger.error(f"  Device info at error: input_ids_device={inputs_on_device_dict['input_ids'].device}, model_main_device={model.device}, target_device_used={target_device}")
        if hasattr(model, 'hf_device_map'): logger.error(f"  Model hf_device_map: {model.hf_device_map}")

    except Exception as e:
        logger.error(f"Error in process_ner_baseline for {model_name} ({lang_code}): {e}", exc_info=True)
        raw_output = f"[GEN_ERROR: Exception: {e}]"

    extracted_entities = extract_entities(raw_output, lang_code=lang_code, verbose=verbose)
    duration = time.time() - start_time
    
    logger.debug(f"process_ner_baseline ({lang_code}, {model_name}): Raw='{raw_output[:100]}...', Extracted={len(extracted_entities)} ents, Time={duration:.2f}s")
    return extracted_entities, duration, raw_output

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
        predicted_entities_list, runtime_sample, raw_output = process_ner_baseline(
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
            'do_sample_used': do_sample,
            'raw_output': raw_output
        })

    results_df = pd.DataFrame(results)
    return results_df

def calculate_ner_metrics(gold_entities: List[Dict[str, str]], predicted_entities: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Calculate NER metrics (Precision, Recall, F1) at the entity level.
    Assumes entities are dictionaries with 'text' and 'type'.
    This is a strict matching based on exact text and type.
    """
    # Initialize overall counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # For type-specific metrics (optional, can be expanded later if needed)
    true_positives_by_type = Counter()
    false_positives_by_type = Counter()
    false_negatives_by_type = Counter()

    # Create sets of (text, type) tuples for efficient lookup and to handle duplicates from model
    # Gold entities should ideally be unique, but predictions might have duplicates.
    # Using a list for gold to preserve original count for false negatives if multiple identical gold entities exist
    # and model predicts only one.
    # However, for TP/FP, a matched gold entity should probably only count once.
    # Let's use sets for matching to avoid overcounting TP/FP due to duplicate predictions matching the same gold entity.
    
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
    # This block should be de-indented to be at the function's top level of indentation,
    # after populating gt_set and pred_set.
    true_positives = len(gt_set.intersection(pred_set))
    false_positives = len(pred_set) - true_positives
    false_negatives = len(gt_set) - true_positives
    
    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

    # Per-type metrics (optional to return, but calculated)
    # for entity_type in unique_entity_types:
    #     tp = true_positives_by_type[entity_type]
    #     fp = false_positives_by_type[entity_type]
    #     fn = false_negatives_by_type[entity_type]
    #     p_type = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    #     r_type = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    #     f1_type = (2 * p_type * r_type) / (p_type + r_type) if (p_type + r_type) > 0 else 0.0
    #     metrics[f'{entity_type}_precision'] = p_type
    #     metrics[f'{entity_type}_recall'] = r_type
    #     metrics[f'{entity_type}_f1'] = f1_type

    return metrics

def load_masakhaner_samples(lang_code: str, split: str = "test", num_samples: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """
    Load MasakhaNER samples for a specific language from Hugging Face Hub.
    Includes a mapping for common language codes to HF dataset config names.
    Args:
        lang_code: Language code (e.g., 'ha', 'sw').
        split: Dataset split to use ('train', 'validation', 'test').
        num_samples (Optional[int]): The number of samples to return. If None, all samples from the split are returned.
        seed (int): Random seed for shuffling if num_samples is specified.
    Returns:
        DataFrame containing loaded samples with 'id', 'tokens', 'text', 'ner_tags_indices', 'tag_names', and 'entities' columns,
        or empty DataFrame if loading/processing fails or lang_code is not supported.
    """
    dataset_name = "masakhane/masakhaner"
    
    MASAKHANER_LANG_CONFIGS = ["amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "swa", "wol", "yor"]
    lang_config_map = {
        'ha': 'hau',
        'sw': 'swa',
    }
    hf_config_name = lang_config_map.get(lang_code.lower(), lang_code.lower())

    if hf_config_name not in MASAKHANER_LANG_CONFIGS:
        logging.error(f"Unsupported MasakhaNER language config '{hf_config_name}' (from input '{lang_code}'). Supported: {MASAKHANER_LANG_CONFIGS}")
        return pd.DataFrame()

    logging.info(f"Attempting to load '{hf_config_name}' samples from {dataset_name}, split '{split}'...")

    def _convert_raw_tags_to_entities_baseline(tokens: List[str], ner_tags_indices: List[int], tag_names_list: List[str]) -> List[Dict[str, str]]:
        """
        Converts IOB-style NER tags to a list of entity dictionaries.
        Outputs format: [{'entity': 'Entity Text', 'type': 'TYPE'}, ...]
        """
        entities_found = []
        current_entity_tokens_list = []
        current_entity_tag_id_val = -1
        
        for i, token_item in enumerate(tokens):
            tag_id_val = ner_tags_indices[i]
            tag_name_val = tag_names_list[tag_id_val]

            if tag_name_val.startswith("B-"):
                if current_entity_tokens_list: # Finalize previous entity
                    entity_type_str = tag_names_list[current_entity_tag_id_val].split('-')[1]
                    entities_found.append({"entity": " ".join(current_entity_tokens_list), "type": entity_type_str})
                current_entity_tokens_list = [token_item]
                current_entity_tag_id_val = tag_id_val
            elif tag_name_val.startswith("I-"):
                # Continue current entity IF the I- tag matches the B- tag's type
                # (e.g., B-PER followed by I-PER)
                if current_entity_tokens_list and tag_names_list[current_entity_tag_id_val].split('-')[1] == tag_name_val.split('-')[1]:
                    current_entity_tokens_list.append(token_item)
                else: # Unexpected I- tag (no B- or type mismatch)
                    if current_entity_tokens_list: # Finalize previous if any
                        entity_type_str = tag_names_list[current_entity_tag_id_val].split('-')[1]
                        entities_found.append({"entity": " ".join(current_entity_tokens_list), "type": entity_type_str})
                    # Start new entity with this I- tag, treating its type as if it were B-
                    current_entity_tokens_list = [token_item]
                    current_entity_tag_id_val = tag_id_val 
            else: # O tag or unexpected tag
                if current_entity_tokens_list: # Finalize previous entity
                    entity_type_str = tag_names_list[current_entity_tag_id_val].split('-')[1]
                    entities_found.append({"entity": " ".join(current_entity_tokens_list), "type": entity_type_str})
                    current_entity_tokens_list = []
                    current_entity_tag_id_val = -1
        
        if current_entity_tokens_list: # After loop, check for any remaining entity
            entity_type_str = tag_names_list[current_entity_tag_id_val].split('-')[1]
            entities_found.append({"entity": " ".join(current_entity_tokens_list), "type": entity_type_str})
        return entities_found

    all_samples_list = []
    try:
        dataset = load_dataset(dataset_name, name=hf_config_name, split=split, trust_remote_code=True)
        logging.info(f"Successfully loaded dataset for {hf_config_name}, split {split}. Full size: {len(dataset)}")
        
        tag_feature = dataset.features['ner_tags']
        if hasattr(tag_feature, 'feature') and hasattr(tag_feature.feature, 'names'):
            tag_names = tag_feature.feature.names
        else:
            logging.error(f"Could not retrieve NER tag names for {hf_config_name}. Aborting.")
            return pd.DataFrame()

        for i, example in enumerate(dataset):
            tokens = example.get('tokens', [])
            ner_tags_indices = example.get('ner_tags', []) 
            sample_id = example.get('id', f"{hf_config_name}_{split}_{i}")

            if not tokens or not isinstance(ner_tags_indices, list) or len(tokens) != len(ner_tags_indices):
                logging.warning(f"Skipping sample {sample_id} for {hf_config_name} due to missing/mismatched tokens or ner_tags format.")
                continue
            
            # Use the local helper to convert tags to entities
            entities = _convert_raw_tags_to_entities_baseline(tokens, ner_tags_indices, tag_names)
            
            all_samples_list.append({
                'id': sample_id,
                'tokens': tokens,
                'text': " ".join(tokens),
                'ner_tags_indices': ner_tags_indices, 
                'tag_names': tag_names, 
                'entities': entities, # This is the crucial part for baseline evaluation
                'language': lang_code 
            })
        
        if not all_samples_list:
            logging.warning(f"No samples processed for language '{hf_config_name}', split '{split}'.")
            return pd.DataFrame()
                
        all_samples_df = pd.DataFrame(all_samples_list)

        # Convert tokens to text for the 'text' column
        if 'tokens' in all_samples_df.columns:
            all_samples_df['text'] = all_samples_df['tokens'].apply(lambda t: " ".join(t) if isinstance(t, list) else "")
            logger.info(f"First 5 'text' entries after creation in load_masakhaner_samples for {lang_code}:\n{all_samples_df[['tokens', 'text']].head().to_string()}")
            # Check for empty text strings
            empty_texts = all_samples_df[all_samples_df['text'] == ''].shape[0]
            if empty_texts > 0:
                logger.warning(f"{empty_texts}/{len(all_samples_df)} samples have empty 'text' after conversion from 'tokens' in load_masakhaner_samples for {lang_code}.")
                if empty_texts == len(all_samples_df):
                    logger.error(f"CRITICAL: All {len(all_samples_df)} samples have empty 'text' for {lang_code}. This will lead to all samples being skipped.")
        else:
            logger.error(f"'tokens' column not found in data for {lang_code}. Cannot create 'text' column.")
            # Create an empty 'text' column to prevent KeyErrors downstream, though processing will likely fail
            all_samples_df['text'] = ""

        if num_samples is not None:
            if num_samples > 0:
                all_samples_df = all_samples_df.sample(frac=1, random_state=seed).reset_index(drop=True)
                if num_samples >= len(all_samples_df):
                    logging.info(f"Requested {num_samples} samples for {hf_config_name} ({split}), but only {len(all_samples_df)} are available. Using all.")
                    final_samples_df = all_samples_df
                else:
                    final_samples_df = all_samples_df.head(num_samples)
                    logging.info(f"Selected {len(final_samples_df)} samples for {hf_config_name} ({split}) after requesting {num_samples} with seed {seed}.")
            else: 
                logging.warning(f"Requested {num_samples} samples for {hf_config_name} ({split}). Returning empty DataFrame.")
                return pd.DataFrame()
        else:
            final_samples_df = all_samples_df
            logging.info(f"Returning all {len(final_samples_df)} MasakhaNER samples for {hf_config_name} ({split}) as num_samples was None.")
        
        logging.info(f"Loaded and processed {len(final_samples_df)} samples for {hf_config_name}, split '{split}'.")
        return final_samples_df
    
    except Exception as e: # Broader exception catch
        logging.error(f"Error loading MasakhaNER data for {hf_config_name}, split {split}: {e}", exc_info=True)
        logging.warning("Falling back to dummy data due to error...")
        return create_dummy_ner_data(lang_code, num_samples if num_samples is not None and num_samples > 0 else 5)

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
                {"entity": "October 2024", "type": "DATE"},
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