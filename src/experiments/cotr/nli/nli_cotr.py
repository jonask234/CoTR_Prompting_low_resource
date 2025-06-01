from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import re
import os
from typing import Any, Tuple, Dict, List, Optional
import sys
import argparse
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import logging
from collections import Counter

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

# Define language names dictionary at module level (align with old script, ensure all necessary are present)
lang_names = {
    "sw": "Swahili",
    "te": "Telugu", # From old script
    "en": "English",
    "ha": "Hausa",   # From old script
    "ur": "Urdu",
    # Add any other languages used in NLI tasks if they were in old script's map and are relevant
    "yo": "Yoruba", "ig": "Igbo", "hi": "Hindi", # Examples from old script's get_language_name
    "ar": "Arabic", "bn": "Bengali", "fi": "Finnish", "id": "Indonesian", "ko": "Korean",
    "ru": "Russian", "th": "Thai", "zh": "Chinese", "de": "German", "es": "Spanish",
    "fr": "French", "ja": "Japanese", "pt": "Portuguese", "vi": "Vietnamese"
}

# Define label mapping for NLI (consistent with old and new)
NLI_LABELS = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

# Define translations of NLI labels (from new script, can be expanded if old had more)
NLI_TRANSLATIONS = {
    "sw": {
        "entailment": "uambatani", 
        "neutral": "katikati",
        "contradiction": "upinzani"
    },
    "ur": {
        "entailment": "lazmi nateeja",
        "neutral": "ghair janibdar",
        "contradiction": "tazad"
    },
    # From old script - these were identical to English, but good to note
    "ha": {
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction"
    },
    "te": {
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction"
    }
}

# Import the baseline implementation for shared functionality (model init, label extraction if needed)
# from src.experiments.baseline.nli.nli_baseline import initialize_model, extract_nli_label # This was in old, might be superseded

# Import shared functionality
from src.utils.data_loaders.load_xnli import load_xnli_samples, XNLI_LABEL_MAP # XNLI_LABEL_MAP is the int->str map
# calculate_nli_metrics will be handled by the runner, importing from baseline

# Import the COMET related utilities
from evaluation.cotr.translation_metrics import COMET_AVAILABLE, calculate_comet_score

# Logger setup
logger = logging.getLogger(__name__)

def initialize_model(model_name: str, cache_path: Optional[str] = "/work/bbd6522/cache_dir") -> Tuple[AutoTokenizer, AutoModelForCausalLM]: # Type hints added
    """
    Initialize a model for NLI task, with robust pad_token handling.
    (Retaining robust version from current nli_cotr.py, as it's generally better)
    """
    logger.info(f"Loading NLI CoTR model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_path,
        low_cpu_mem_usage=True
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"NLI CoTR: Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token}) for {model_name}")
        else:
            new_pad_token = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': new_pad_token})
            model.resize_token_embeddings(len(tokenizer)) # Important step
            logger.info(f"NLI CoTR: Added new pad_token '{new_pad_token}' and resized model embeddings for {model_name}.")

    if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.info(f"NLI CoTR: Aligned model.config.pad_token_id with tokenizer.pad_token_id ({tokenizer.pad_token_id}) for {model_name}")
    
    logger.info(f"NLI CoTR model {model_name} loaded. Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}. Model config pad_token_id: {model.config.pad_token_id}")
    return tokenizer, model

def get_language_name(lang_code: str) -> str:
    """Get the full language name from a language code (merged from old and new)."""
    return lang_names.get(lang_code, lang_code.capitalize())

def generate_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """
    Generate a prompt for translating text. (Consistent with old and new)
    """
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)
    prompt = f"""Translate the following {source_name} text to {target_name}:

{source_name} text: {text}

{target_name} translation:"""
    return prompt

def generate_nli_prompt(premise: str, hypothesis: str, use_few_shot: bool = True) -> str:
    """
    Generate a prompt for NLI task in English.
    (Adapted from old script's version for basic structure, but with current script's clarity on single-word output,
    and using English few-shot examples as per constraint).
    """
    safe_premise = str(premise).replace("'''", "\'\'\'").replace('"""', '\"\"\"')
    safe_hypothesis = str(hypothesis).replace("'''", "\'\'\'").replace('"""', '\"\"\"')

    instructions = (
        "Given a premise and hypothesis, determine if the hypothesis is an ENTAILMENT, CONTRADICTION, or NEUTRAL with respect to the premise. "
        "Provide your answer as a single word: either ENTAILMENT, NEUTRAL, or CONTRADICTION. Do not add any other text or explanation."
    )

    definitions = """
Definitions:
- ENTAILMENT: The hypothesis definitely follows from the premise.
- CONTRADICTION: The hypothesis definitely contradicts the premise.
- NEUTRAL: The hypothesis might be true or false; there's not enough information to tell.
"""
    
    few_shot_examples_str = ""
    if use_few_shot:
        # English few-shot examples (as per constraint)
        ex1_prem = "The chef is cooking a meal in the kitchen."
        ex1_hyp = "The chef is preparing food."
        ex1_nli = "ENTAILMENT"

        ex2_prem = "The boy is playing soccer in the park."
        ex2_hyp = "The boy is swimming in a pool."
        ex2_nli = "CONTRADICTION"

        ex3_prem = "The woman is walking down the street."
        ex3_hyp = "She is going to the grocery store."
        ex3_nli = "NEUTRAL"

        few_shot_examples_str = f"""
Examples:

Premise: '{ex1_prem}'
Hypothesis: '{ex1_hyp}'
Relationship: {ex1_nli}

Premise: '{ex2_prem}'
Hypothesis: '{ex2_hyp}'
Relationship: {ex2_nli}

Premise: '{ex3_prem}'
Hypothesis: '{ex3_hyp}'
Relationship: {ex3_nli}
"""

    task_prompt_str = f"""
Now, determine the relationship for the following:

Premise: '{safe_premise}'
Hypothesis: '{safe_hypothesis}'
Relationship:"""

    final_prompt = f"""{instructions}
{definitions}
{few_shot_examples_str}
{task_prompt_str}"""
    return final_prompt

def generate_single_prompt_nli_cotr(lrl_premise: str, lrl_hypothesis: str, lang_code: str, use_few_shot: bool = True) -> str:
    """
    Generate a single Chain-of-Thought (CoT) prompt for NLI, inspired by nli_cotr_old.py's structure
    but using English for the CoT steps and examples as per constraints.
    """
    lrl_name = get_language_name(lang_code)
    safe_lrl_premise = str(lrl_premise).replace("'''", "\'\'\'").replace('"""', '\"\"\"')
    safe_lrl_hypothesis = str(lrl_hypothesis).replace("'''", "\'\'\'").replace('"""', '\"\"\"')

    # Instructions based on old script's single prompt but adapted for English CoT
    instructions_core = f"""Perform the following tasks in order for the provided {lrl_name} text and hypothesis:
1. Translate the {lrl_name} Premise to English.
2. Translate the {lrl_name} Hypothesis to English.
3. Based ONLY on your English translations, determine if the English Hypothesis is ENTAILMENT, CONTRADICTION, or NEUTRAL with respect to the English Premise.

Definitions:
- ENTAILMENT: The hypothesis must be true if the text is true.
- CONTRADICTION: The hypothesis cannot be true if the text is true.
- NEUTRAL: The hypothesis might be true or false; the text doesn't provide enough information.

Provide your answer in this exact format:
English Premise: [Your English Translation of the Premise]
English Hypothesis: [Your English Translation of the Hypothesis]
Relationship: [ENTAILMENT/CONTRADICTION/NEUTRAL]"""

    few_shot_examples_str = ""
    if use_few_shot:
        # Few-shot examples with LRL input part, but English CoT (as per constraint)
        # Using Swahili as an example LRL, adapt if other LRL examples from old script are preferred
        if lang_code == "sw":
            ex1_lrl_prem = "Mwanamke anapika chakula jikoni."
            ex1_lrl_hyp = "Mwanamke anatayarisha chakula."
            ex1_en_prem = "A woman is cooking food in the kitchen."
            ex1_en_hyp = "A woman is preparing food."
            ex1_nli = "ENTAILMENT"
        elif lang_code == "ur": # Urdu example from old script
            ex1_lrl_prem = "ایک عورت باورچی خانے میں کھانا پکا رہی ہے۔"
            ex1_lrl_hyp = "عورت کھانا تیار کر رہی ہے۔"
            ex1_en_prem = "A woman is cooking food in the kitchen."
            ex1_en_hyp = "A woman is preparing food."
            ex1_nli = "ENTAILMENT"
        else: # Generic English example to show structure if LRL not sw/ur
            ex1_lrl_prem = f"[Example Premise in {lrl_name}]"
            ex1_lrl_hyp = f"[Example Hypothesis in {lrl_name}]"
            ex1_en_prem = "A representative English premise."
            ex1_en_hyp = "A representative English hypothesis."
            ex1_nli = "NEUTRAL" # Adjust NLI label as appropriate for a generic example

        # Generic structure for a second example
        ex2_lrl_prem_generic = f"[Another Premise in {lrl_name} for contradiction]"
        ex2_lrl_hyp_generic = f"[Another Hypothesis in {lrl_name} for contradiction]"
        ex2_en_prem_generic = "The cat is sleeping."
        ex2_en_hyp_generic = "The cat is awake."
        ex2_nli_generic = "CONTRADICTION"

        few_shot_examples_str = f"""
--- Examples ---

Example 1:
Original {lrl_name} Premise: '{ex1_lrl_prem}'
Original {lrl_name} Hypothesis: '{ex1_lrl_hyp}'

English Premise: {ex1_en_prem}
English Hypothesis: {ex1_en_hyp}
Relationship: {ex1_nli}

Example 2:
Original {lrl_name} Premise: '{ex2_lrl_prem_generic}'
Original {lrl_name} Hypothesis: '{ex2_lrl_hyp_generic}'

English Premise: {ex2_en_prem_generic}
English Hypothesis: {ex2_en_hyp_generic}
Relationship: {ex2_nli_generic}
"""

    task_prompt_str = f"""
--- Your Task ---

Original {lrl_name} Premise: '{safe_lrl_premise}'
Original {lrl_name} Hypothesis: '{safe_lrl_hypothesis}'

Follow the three steps precisely and provide your full response:"""

    final_prompt = instructions_core + "\n" + few_shot_examples_str + "\n" + task_prompt_str
    logger.debug(f"Generated Single NLI CoTR Prompt for {lang_code}:\n{final_prompt[:500]}...")
    return final_prompt

def translate_text( # Retaining current more robust version, ensure params match runner
    model: Any,
    tokenizer: Any,
    text: str,
    source_lang: str,
    target_lang: str,
    # Parameters passed from runner (cli_args mapped to these)
    temperature: float,
    do_sample: bool,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int, # Max tokens for this translation step
    is_label: bool = False, # Retained from current, might be useful
    max_input_length: int = 4096 # Tokenizer max length
) -> str:
    """
    Translate text using the provided model.
    Parameters (temp, top_p etc.) are now passed directly.
    """
    if not text or text.strip() == "": return ""
    if source_lang == target_lang: return text
        
    prompt = generate_translation_prompt(text, source_lang, target_lang)
    # ... (rest of the robust translate_text from current nli_cotr.py, ensuring it uses the passed-in generation parameters) ...
    # For brevity, assuming the existing logic for Aya model and standard tokenization is kept,
    # but critically, the model.generate call must use the passed-in temp, top_p, top_k, etc.

    inputs_dict = {}
    model_name_for_check = getattr(model, 'name_or_path', "").lower()

    # Cohere/Aya specific chat template handling (from current script)
    if "coherelabs/aya" in model_name_for_check:
        logger.debug(f"NLI CoTR Translate: Detected Cohere Aya model ('{model_name_for_check}'). Using apply_chat_template.")
        messages = [{"role": "user", "content": prompt}]
        try:
            input_ids_tensor = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                truncation=True, max_length=max_input_length
            )
            if isinstance(input_ids_tensor, torch.Tensor): inputs_dict = {"input_ids": input_ids_tensor}
            else:
                logger.error(f"NLI CoTR Translate: apply_chat_template non-tensor. Type: {type(input_ids_tensor)}")
                return "[TRANSLATION_ERROR_TOKENIZATION]"
        except Exception as e_chat_template:
            logger.error(f"NLI CoTR Translate: Error apply_chat_template for {model_name_for_check}: {e_chat_template}", exc_info=True)
            return "[TRANSLATION_ERROR_TOKENIZATION]"
    else:
        inputs_dict = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)

    if not inputs_dict or 'input_ids' not in inputs_dict:
        logger.error(f"NLI CoTR Translate: Tokenization failed for model {model_name_for_check}.")
        return "[TRANSLATION_ERROR_TOKENIZATION]"
        
    inputs = {k: v.to(model.device) for k, v in inputs_dict.items()}
    
    effective_do_sample = do_sample if temperature > 0 else False # Explicitly False for temp 0

    try:  # Line 347
        with torch.no_grad():  # Line 348
            outputs = model.generate(  # Indented
            **inputs,
                max_new_tokens=max_new_tokens, # Use the specific max_tokens for translation
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=effective_do_sample, # Use the derived do_sample
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    except Exception as e_generate:
        logger.error(f"NLI CoTR Translate: Error during model.generate for {source_lang}->{target_lang} on text '{text[:50]}...': {e_generate}", exc_info=True)
        return "[TRANSLATION_FAILED_MODEL_ERROR]"
    
    cleaned_response = clean_translation_response(response, target_lang, source_lang)
    logger.debug(f"Raw translation ({source_lang}->{target_lang}): '{response[:100]}...' Cleaned: '{cleaned_response[:100]}...'")
    return cleaned_response

def clean_translation_response(response: str, target_lang: str, source_lang: str) -> str:
    """
    Clean the translation response. (Consistent with old and new)
    """
    prefixes_to_remove = [
        "translation:", 
        f"{get_language_name(target_lang)}:", 
        f"{get_language_name(target_lang)} translation:",
        f"{target_lang.upper()} translation:",
        f"{target_lang} translation:"
    ]
    cleaned = response.strip()
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or \
       (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    return cleaned

def extract_nli_label(output_text: str) -> str:
    """
    Extract NLI label from model output.
    Combines robustness of current script's version with direct checks from old script.
    Ensures a valid label ("entailment", "neutral", "contradiction") or a default.
    """
    original_output_text = output_text.strip()
    text_lower = original_output_text.lower()
    valid_labels = ["entailment", "neutral", "contradiction"]

    # 1. Check for structured "Relationship: LABEL" (from current and old)
    # Made more robust: optional step numbers, colons, "is"
    relationship_match = re.search(r'(?:relationship|step\s*\d+\s*:\s*relationship|label)\s*(?:is)?\s*:\s*\b(entailment|neutral|contradiction)\b', text_lower, re.IGNORECASE)
    if relationship_match:
        return relationship_match.group(1).lower()

    # 2. Check for capitalized labels directly (from current and old)
    if "ENTAILMENT" in original_output_text: return "entailment"
    if "CONTRADICTION" in original_output_text: return "contradiction"
    if "NEUTRAL" in original_output_text: return "neutral"

    # 3. Check for lowercase labels as standalone words or dominant terms (from current)
    # This is more heuristic. Search for the labels as whole words.
    for label in valid_labels:
        if re.search(rf"\b{label}\b", text_lower): # Check for whole word
            # To avoid ambiguity if multiple labels are mentioned (e.g., in definitions),
            # prioritize if it's the only label, or appears in a concluding line.
            lines = [line.strip() for line in text_lower.split('\\n') if line.strip()]
            if lines:
                last_line_lower = lines[-1].lower()
                if re.search(rf"\b{label}\b", last_line_lower): # If in last line
                    # Check if ONLY this label is in the last line to be more certain
                    other_labels_in_last_line = [other for other in valid_labels if other != label and re.search(rf"\b{other}\b", last_line_lower)]
                    if not other_labels_in_last_line:
                        return label # Indented
            # Fallback if not clearly in last line but present
            # Count occurrences to see if one is dominant if multiple are present
            counts = {l: len(re.findall(rf"\b{l}\b", text_lower)) for l in valid_labels}
            if counts[label] > 0 and all(counts[label] >= counts[other] for other in valid_labels if other != label):
                 # If this label is present and no other label has a higher count
                if sum(c > 0 for c in counts.values()) == 1: # Only this label is present
                    return label
                # If multiple labels, but this one is at least as frequent and appears last or in a specific context
                # This part can get complex, for now, if it's found and no other is clearly dominant, it's a candidate.
    
    # 4. Check last line explicitly (from old and current)
    lines = [line.strip() for line in original_output_text.split('\\n') if line.strip()]
    if lines:
        last_line_cleaned = lines[-1].lower().strip(".,;:!?").split()[-1] # Get last word of last line
        if last_line_cleaned in valid_labels:
            return last_line_cleaned
        # Check if the entire last line IS one of the labels
        if lines[-1].lower() in valid_labels:
            return lines[-1].lower()

    # 5. Fallback search for keywords (from old script)
    if any(x in text_lower for x in ["entail", "follows", "must be true"]): return "entailment"
    if any(x in text_lower for x in ["contradict", "cannot be true", "inconsistent"]): return "contradiction"
    # Neutral is often the hardest, "might be true", "not enough information"

    logger.warning(f"NLI CoTR: Could not confidently extract NLI label from: '{original_output_text[:100]}...' Returning 'unknown' as default.")
    return "unknown" # Default to unknown

def extract_english_translations(output_text: str) -> Tuple[str, str]:
    """
    Extract English translations of premise and hypothesis from single-prompt CoT output.
    Adapted from old script's regex, made more robust.
    """
    premise_en = "[PREMISE_EN_EXTRACTION_FAILED]"
    hypothesis_en = "[HYPOTHESIS_EN_EXTRACTION_FAILED]"
    
    # Pattern for "English Premise: [translation]"
    # Looks for "English Premise:" (case insensitive) followed by the translation.
    # Stops at "English Hypothesis:" or "Relationship:" or end of text.
    premise_match = re.search(r"(?:English\s+Premise\s*:\s*)(.*?)(?=(?:English\s+Hypothesis\s*:|Relationship\s*:)|$)", output_text, re.IGNORECASE | re.DOTALL)
    if premise_match:
        premise_en = premise_match.group(1).strip().split('\\n')[0].strip() # Get first line of match
    
    # Pattern for "English Hypothesis: [translation]"
    hypothesis_match = re.search(r"(?:English\s+Hypothesis\s*:\s*)(.*?)(?=(?:Relationship\s*:)|$)", output_text, re.IGNORECASE | re.DOTALL)
    if hypothesis_match:
        hypothesis_en = hypothesis_match.group(1).strip().split('\\n')[0].strip()

    if premise_en == "[PREMISE_EN_EXTRACTION_FAILED]":
        logger.warning(f"NLI CoTR: Failed to extract English Premise from: {output_text[:200]}...")
    if hypothesis_en == "[HYPOTHESIS_EN_EXTRACTION_FAILED]":
        logger.warning(f"NLI CoTR: Failed to extract English Hypothesis from: {output_text[:200]}...")
    
    return premise_en, hypothesis_en

def process_nli_english( # For multi-prompt's NLI step
    model: Any,
    tokenizer: Any,
    premise_en: str,
    hypothesis_en: str,
    use_few_shot: bool,
    # Generation parameters passed from runner (mapped from cli_args)
    temperature: float, 
    do_sample: bool,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int, # Max tokens for the NLI label output
    max_input_length: int = 4096
) -> str: # Returns the predicted English label string
    """
    Process an NLI pair in English using the model. For multi-prompt pipeline.
    Parameters (temp, top_p, etc.) are passed directly.
    """
    prompt = generate_nli_prompt(premise_en, hypothesis_en, use_few_shot=use_few_shot)

    inputs_dict = {} # Initialize inputs_dict
    model_name_for_check = getattr(model, 'name_or_path', "").lower()
    # Cohere/Aya specific handling from current script
    if "coherelabs/aya" in model_name_for_check:
        messages = [{"role": "user", "content": prompt}]
        try:
            input_ids_tensor = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                truncation=True, max_length=max_input_length
            )
            if isinstance(input_ids_tensor, torch.Tensor):
                inputs_dict = {"input_ids": input_ids_tensor}
            else:
                logger.error(f"NLI CoTR process_nli_english: apply_chat_template did not return a tensor for Aya. Type: {type(input_ids_tensor)}")
                return "[NLI_ERROR_TOKENIZATION_AYA_TYPE]"
        except Exception as e_chat_template_nli:
            logger.error(f"NLI CoTR process_nli_english: Error during apply_chat_template for Aya: {e_chat_template_nli}", exc_info=True)
            return "[NLI_ERROR_TOKENIZATION_AYA_EXC]"
    
    if not inputs_dict: # If not an Aya model OR if Aya specific tokenization did not populate inputs_dict
        inputs_dict = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)

    if not inputs_dict or 'input_ids' not in inputs_dict:
        logger.error(f"NLI CoTR process_nli_english: Tokenization failed or resulted in empty input_ids for model {model_name_for_check}.")
        return "[NLI_ERROR_TOKENIZATION_FINAL]"
    inputs = {k: v.to(model.device) for k, v in inputs_dict.items()}

    effective_do_sample = do_sample if temperature > 0 else False
    output_text = "[MODEL_GENERATION_ERROR_NLI_EN]" # Default in case of error
    predicted_label = "neutral" # Default

    try: # Line 530
        with torch.no_grad(): # Line 531
            outputs = model.generate( # Indented
            **inputs,
                max_new_tokens=max_new_tokens, # Specific for label
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=effective_do_sample,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        )
        output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True) # Correctly part of try
        predicted_label = extract_nli_label(output_text) # Correctly part of try
    except Exception as e_generate_nli_en: # Line 544
        logger.error(f"NLI CoTR process_nli_english: Error during model.generate: {e_generate_nli_en}", exc_info=True)
        # output_text and predicted_label will remain as defaults

    logger.debug(f"NLI English Step: Premise='{premise_en[:50]}...', Hyp='{hypothesis_en[:50]}...', RawOut='{output_text[:50]}...', Pred='{predicted_label}'")
    return predicted_label # Return the extracted label (or default if error)

def process_nli_cotr_single_prompt( # For single-prompt pipeline
    model: Any,
    tokenizer: Any,
    premise: str, # LRL Premise
    hypothesis: str, # LRL Hypothesis
    lang_code: str,
    use_few_shot: bool,
    # Generation parameters for the entire CoT chain, passed from runner
    temperature: float, 
    do_sample: bool,
    top_p: float, 
    top_k: int, 
    repetition_penalty: float, 
    max_new_tokens: int, # Max tokens for the entire CoT output
    max_input_length: int = 4096
) -> Dict[str, Any]:
    """
    Process an NLI pair using a single CoTR prompt. Inspired by old script's single_prompt logic.
    Parameters (temp, top_p, etc.) are passed directly for the whole chain.
    """
    start_time = time.time()
    prompt = generate_single_prompt_nli_cotr(premise, hypothesis, lang_code, use_few_shot)

    inputs_dict = {} # Initialize
    model_name_for_check = getattr(model, 'name_or_path', "").lower()
    # Cohere/Aya specific handling
    if "coherelabs/aya" in model_name_for_check:
        messages = [{"role": "user", "content": prompt}]
        try:
            input_ids_tensor = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                truncation=True, max_length=max_input_length
            )
            if isinstance(input_ids_tensor, torch.Tensor):
                inputs_dict = {"input_ids": input_ids_tensor}
            else: # Error case
                logger.error(f"NLI CoTR single_prompt: Aya tokenization did not return tensor. Type: {type(input_ids_tensor)}")
                return {"raw_response": "[TOKENIZATION_ERROR_AYA_TYPE_SP]", "predicted_label_for_accuracy": "neutral", "premise_en": "[Error]", "hypothesis_en": "[Error]", "runtime_seconds": time.time() - start_time}
        except Exception as e_aya_sp: # Error case
             logger.error(f"NLI CoTR single_prompt: Aya tokenization error: {e_aya_sp}", exc_info=True)
             return {"raw_response": "[TOKENIZATION_ERROR_AYA_EXC_SP]", "predicted_label_for_accuracy": "neutral", "premise_en": "[Error]", "hypothesis_en": "[Error]", "runtime_seconds": time.time() - start_time}
    
    if not inputs_dict: # Not Aya, or Aya tokenization failed to populate inputs_dict
        inputs_dict = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)

    if not inputs_dict or 'input_ids' not in inputs_dict:
        logger.error(f"NLI CoTR single_prompt: Tokenization failed or empty input_ids for model {model_name_for_check}.")
        return {"raw_response": "[TOKENIZATION_ERROR_GENERAL_SP]", "predicted_label_for_accuracy": "neutral", "premise_en": "[Error]", "hypothesis_en": "[Error]", "runtime_seconds": time.time() - start_time}

    inputs = {k: v.to(model.device) for k, v in inputs_dict.items()}
    effective_do_sample = do_sample if temperature > 0 else False
    response_text = "[MODEL_GENERATION_ERROR_SP]" # Default in case of error
    premise_en = "[ErrorSP_GenFail]" # Default
    hypothesis_en = "[ErrorSP_GenFail]" # Default
    predicted_english_label = "neutral" # Default

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens, # For the whole chain
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=effective_do_sample,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            )
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        # Extract parts from response after successful generation
        premise_en, hypothesis_en = extract_english_translations(response_text)
        predicted_english_label = extract_nli_label(response_text)
    except Exception as e_generate_sp:
        logger.error(f"NLI CoTR single_prompt: Error during model.generate: {e_generate_sp}", exc_info=True)
        # Values will remain as defaults initialized before try block
    
    runtime = time.time() - start_time
    
    logger.debug(f"Single-Prompt CoTR for {lang_code}: Premise='{premise[:50]}...', LRL_Hyp='{hypothesis[:50]}...'")
    logger.debug(f"  RawResp='{response_text[:150]}...'")
    logger.debug(f"  Extracted EN_Prem='{premise_en[:50]}...', EN_Hyp='{hypothesis_en[:50]}...', Pred_EN_Label='{predicted_english_label}'")

    return {
        "premise_lrl": premise, # Original LRL
        "hypothesis_lrl": hypothesis, # Original LRL
        "premise_en": premise_en, # Use potentially updated value
        "hypothesis_en": hypothesis_en, # Use potentially updated value
        "predicted_english_label_model_intermediate": predicted_english_label, # Use potentially updated value
        "predicted_lrl_label_model_raw": "N/A_SP_EN_eval",
        "predicted_label_for_accuracy": predicted_english_label, # Use potentially updated value
        "raw_response": response_text, # Use potentially updated value
        "runtime_seconds": runtime
    }

def process_nli_cotr_multi_prompt( # For multi-prompt pipeline
    model: Any,
    tokenizer: Any,
    premise: str, # LRL Premise
    hypothesis: str, # LRL Hypothesis
    lang_code: str,
    use_few_shot: bool,
    # Parameter dictionaries passed from runner (mapped from cli_args)
    nli_processing_params: Dict[str, Any],      # For EN NLI step
    text_translation_params: Dict[str, Any],   # For LRL->EN text
    label_translation_params: Dict[str, Any],  # For EN->LRL label (if lang_code != 'en')
    max_input_length: int = 4096
) -> Dict[str, Any]:
    """
    Process an NLI pair using multi-prompt CoTR. Inspired by old script.
    Returns a dictionary with intermediate and final parts.
    COMET scores calculated by the runner.
    """
    start_time = time.time()
    output = {
        "premise_lrl": premise, "hypothesis_lrl": hypothesis,
        "premise_en": "[Translation Error]", "hypothesis_en": "[Translation Error]",
        "predicted_english_label": "[NLI Error]",
        "predicted_lrl_label_raw": "[BackTranslation Error or N/A]", # LRL version of the label
        "predicted_label_for_accuracy": "neutral", # Final label for accuracy (English)
        "premise_translation_response": "", "hypothesis_translation_response": "", "nli_response": "", # Raw responses
        "runtime_seconds": 0.0
    }

        # 1. Translate LRL premise to English
    premise_en_raw = translate_text(model, tokenizer, premise, lang_code, "en", is_label=False, **text_translation_params, max_input_length=max_input_length)
    output["premise_translation_response"] = premise_en_raw # Store raw for debugging if needed
    output["premise_en"] = premise_en_raw # Assuming translate_text returns cleaned
    if "[Error]" in output["premise_en"]: # Check for error marker from translate_text
        output["runtime_seconds"] = time.time() - start_time
        return output # Early exit on critical error

        # 2. Translate LRL hypothesis to English
    hypothesis_en_raw = translate_text(model, tokenizer, hypothesis, lang_code, "en", is_label=False, **text_translation_params, max_input_length=max_input_length)
    output["hypothesis_translation_response"] = hypothesis_en_raw
    output["hypothesis_en"] = hypothesis_en_raw
    if "[Error]" in output["hypothesis_en"]:
        output["runtime_seconds"] = time.time() - start_time
        return output

        # 3. Perform NLI on translated English premise and hypothesis
    # nli_processing_params includes: temperature, do_sample, top_p, top_k, repetition_penalty, max_new_tokens (for label)
    predicted_english_label_raw = process_nli_english(
        model, tokenizer, output["premise_en"], output["hypothesis_en"], use_few_shot,
        **nli_processing_params, max_input_length=max_input_length
    )
    output["nli_response"] = predicted_english_label_raw # This is actually the raw model output for the NLI step
    output["predicted_english_label"] = extract_nli_label(predicted_english_label_raw) # Cleaned English label
    
    # For XNLI, accuracy is typically based on the English label prediction.
    output["predicted_label_for_accuracy"] = output["predicted_english_label"]

    # 4. Translate predicted English label back to LRL (if not English input and if required by constraint)
    # Constraint: "output back translating in LRL and comparing to LRL ground truth... for nli"
    # This implies we need an LRL version of the predicted label if lang_code is not 'en'.
    if lang_code != "en": # Line 705 - Correctly Indented
        if output["predicted_english_label"] not in ["entailment", "neutral", "contradiction", "[NLI Error]"]: # Check if it's a valid label, including error marker
            logger.warning(f"Predicted English label '{output['predicted_english_label']}' is not standard for back-translation. Defaulting LRL label.")
            output["predicted_lrl_label_raw"] = f"[Invalid EN label for backtrans: {output['predicted_english_label']}]"
        elif output["predicted_english_label"] == "[NLI Error]":
            output["predicted_lrl_label_raw"] = "[BackTranslation Error due to NLI Error]"
        else:
            # label_translation_params includes: temperature, do_sample, top_p, top_k, repetition_penalty, max_new_tokens
            output["predicted_lrl_label_raw"] = translate_text(
                model, tokenizer, output["predicted_english_label"], "en", lang_code, is_label=True,
                **label_translation_params, max_input_length=max_input_length
            )
    else: # lang_code is 'en' - Correctly Indented
        output["predicted_lrl_label_raw"] = output["predicted_english_label"] # LRL is same as English

    output["runtime_seconds"] = time.time() - start_time
    logger.debug(f"Multi-Prompt CoTR for {lang_code}: LRL_Prem='{premise[:50]}...' -> EN_Prem='{output['premise_en'][:50]}...'")
    logger.debug(f"  EN_NLI_Label='{output['predicted_english_label']}' -> LRL_Label='{output['predicted_lrl_label_raw'][:50]}...'")
    return output

def evaluate_nli_cotr( # Main multi-prompt evaluation loop (runner calls this)
    model_name: str,
    tokenizer: Any, 
    model: Any,     
    samples_df: pd.DataFrame,
    lang_code: str,
    use_few_shot: bool,
    # Dictionaries of parameters from runner (mapped from cli_args)
    nli_params: Dict[str, Any],
    text_translation_params: Dict[str, Any],
    label_translation_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Evaluate NLI using the MULTI-PROMPT CoTR approach.
    Model and tokenizer are passed in (initialized by runner).
    Parameters are passed as dicts.
    COMET scores are now handled by the runner script after this returns.
    """
    results_list = []
    total_start_time = time.time()

    logger.info(f"Starting Multi-Prompt NLI CoTR evaluation for {model_name} on {lang_code} ({'few-shot' if use_few_shot else 'zero-shot'}).")
    logger.info(f"  NLI Step Params: {nli_params}")
    logger.info(f"  Text Translation Params: {text_translation_params}")
    logger.info(f"  Label Translation Params (for LRL back-translation): {label_translation_params}")

    # Determine a safe max_input_length for tokenization (passed to process_nli_cotr_multi_prompt)
    tokenizer_model_max_length = getattr(tokenizer, 'model_max_length', 4096) 
    safe_max_input_length = min(tokenizer_model_max_length, 4096) 
    safe_max_input_length = max(safe_max_input_length, 256)
    logger.info(f"NLI CoTR Multi-Prompt: Using safe_max_input_length: {safe_max_input_length} for tokenizing prompts in {model_name}")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Multi-Prompt CoTR {lang_code}"):
        premise_lrl = str(row.get('premise', '')) 
        hypothesis_lrl = str(row.get('hypothesis', ''))
        ground_truth_label_str = str(row.get('label', 'unknown')) 
        ground_truth_label_int = row.get('original_label_int', -1)

        base_error_dict_structure = {
            'premise_lrl': premise_lrl,
            'hypothesis_lrl': hypothesis_lrl,
            'original_gt_label_int': ground_truth_label_int,
            'ground_truth_eng_label_str': ground_truth_label_str,
            'premise_en': "[Error]",
            'hypothesis_en': "[Error]",
            'predicted_english_label_model_intermediate': "[Error]",
            'predicted_lrl_label_model_raw': "[Error]",
            'predicted_label_for_accuracy': "neutral",
            'raw_premise_translation_response': "[Error]",
            'raw_hypothesis_translation_response': "[Error]",
            'raw_nli_response': "[Error]",
            'runtime_seconds': 0.0,
            'error_message': None
        }

        if not premise_lrl or not hypothesis_lrl or ground_truth_label_str == 'unknown':
            logger.warning(f"Skipping sample {idx} due to missing premise, hypothesis, or ground truth label.")
            error_entry = base_error_dict_structure.copy()
            error_entry['error_message'] = "Missing input or GT label"
            error_entry['premise_en'] = premise_lrl if premise_lrl else "[Missing]"
            error_entry['hypothesis_en'] = hypothesis_lrl if hypothesis_lrl else "[Missing]"
            results_list.append(error_entry)
            continue

        try:
            processed_data = process_nli_cotr_multi_prompt(
                model, tokenizer, premise_lrl, hypothesis_lrl, lang_code, use_few_shot,
                nli_processing_params=nli_params,
                text_translation_params=text_translation_params,
                label_translation_params=label_translation_params,
                max_input_length=safe_max_input_length
            )

            current_result = {
                'premise_lrl': premise_lrl,
                'hypothesis_lrl': hypothesis_lrl,
                'original_gt_label_int': ground_truth_label_int,
                'ground_truth_eng_label_str': ground_truth_label_str,
                'premise_en': processed_data.get("premise_en", "[ErrorMP_Eval]"),
                'hypothesis_en': processed_data.get("hypothesis_en", "[ErrorMP_Eval]"),
                'predicted_english_label_model_intermediate': processed_data.get("predicted_english_label", "[ErrorMP_Eval]"),
                'predicted_lrl_label_model_raw': processed_data.get("predicted_lrl_label_raw", "[ErrorMP_Eval]"),
                'predicted_label_for_accuracy': processed_data.get("predicted_label_for_accuracy", "neutral"),
                'raw_premise_translation_response': processed_data.get("premise_translation_response", "[ErrorMP_Eval]"),
                'raw_hypothesis_translation_response': processed_data.get("hypothesis_translation_response", "[ErrorMP_Eval]"),
                'raw_nli_response': processed_data.get("nli_response", "[ErrorMP_Eval]"),
                'runtime_seconds': processed_data.get("runtime_seconds", 0.0),
                'error_message': None
            }
            results_list.append(current_result)

        except Exception as e_eval_multi_sample:
            logger.error(f"CRITICAL: Error processing multi-prompt sample {idx} for lang {lang_code} in evaluate_nli_cotr: {e_eval_multi_sample}", exc_info=True)
            error_entry = base_error_dict_structure.copy()
            error_entry['error_message'] = f"Sample processing error in eval: {str(e_eval_multi_sample)}"
            results_list.append(error_entry)

    total_runtime = time.time() - total_start_time
    results_df = pd.DataFrame(results_list)
    # Add columns that runner expects for summary, like model_name, language etc. (runner adds these)
    # The runner script will also add COMET score columns by processing 'premise_lrl' vs 'premise_en', etc.
    
    logger.info(f"Finished Multi-Prompt NLI CoTR eval for {model_name} on {lang_code}. Total runtime: {total_runtime:.2f}s")
    return results_df

def evaluate_nli_cotr_single_prompt( # Main single-prompt evaluation loop (runner calls this)
    model_name: str,
    tokenizer: Any, 
    model: Any,     
    samples_df: pd.DataFrame,
    lang_code: str,
    use_few_shot: bool,
    # Individual generation parameters from runner (mapped from cli_args)
    temperature: float, 
    top_p: float, 
    top_k: int, 
    max_tokens: int, # Max tokens for the entire single prompt output chain
    repetition_penalty: float, 
    do_sample: bool = True # Default from old script, runner controls this
) -> pd.DataFrame:
    """
    Evaluate NLI CoTR SINGLE-PROMPT approach on a dataset.
    Model and tokenizer are passed in. Parameters are individual.
    COMET scores handled by runner.
    """
    results_list = []
    total_start_time = time.time()
    logger.info(f"Starting Single-Prompt NLI CoTR evaluation for {model_name} on {lang_code} ({'few-shot' if use_few_shot else 'zero-shot'}).")
    logger.info(f"  Chain Params: temp={temperature}, top_p={top_p}, top_k={top_k}, rep_pen={repetition_penalty}, max_tok={max_tokens}, do_sample={do_sample}")

    tokenizer_model_max_length = getattr(tokenizer, 'model_max_length', 4096)
    safe_max_input_length = min(tokenizer_model_max_length, 4096)
    safe_max_input_length = max(safe_max_input_length, 256)
    logger.info(f"NLI CoTR Single-Prompt: Using safe_max_input_length: {safe_max_input_length} for tokenizing prompts in {model_name}")
    
    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Single-Prompt CoTR {lang_code}"):
        premise_lrl = str(row.get('premise', ''))
        hypothesis_lrl = str(row.get('hypothesis', ''))
        ground_truth_label_str = str(row.get('label', 'unknown'))
        ground_truth_label_int = row.get('original_label_int', -1)

        base_error_dict_structure = {
            'premise_lrl': premise_lrl,
            'hypothesis_lrl': hypothesis_lrl,
            'original_gt_label_int': ground_truth_label_int,
            'ground_truth_eng_label_str': ground_truth_label_str,
            'premise_en': "[Error]",
            'hypothesis_en': "[Error]",
            'predicted_english_label_model_intermediate': "[Error]",
            'predicted_lrl_label_model_raw': "N/A",
            'predicted_label_for_accuracy': "neutral", # Default/fallback
            'raw_model_response': "[Error]",
            'runtime_seconds': 0.0,
            'error_message': None
        }

        if not premise_lrl or not hypothesis_lrl or ground_truth_label_str == 'unknown':
            logger.warning(f"Skipping sample {idx} due to missing input or GT label.")
            error_entry = base_error_dict_structure.copy()
            error_entry['error_message'] = "Missing input or GT label"
            error_entry['premise_en'] = premise_lrl if premise_lrl else "[Missing]" # Keep original if present
            error_entry['hypothesis_en'] = hypothesis_lrl if hypothesis_lrl else "[Missing]"
            results_list.append(error_entry)
            continue

        try:
            processed_output = process_nli_cotr_single_prompt(
                model, tokenizer, premise_lrl, hypothesis_lrl, lang_code, use_few_shot,
                temperature=temperature, do_sample=do_sample, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty, max_new_tokens=max_tokens,
                max_input_length=safe_max_input_length
            )

            # Merge processed_output with base info, ensuring all keys are present
            current_result = {
                'premise_lrl': premise_lrl,
                'hypothesis_lrl': hypothesis_lrl,
                'original_gt_label_int': ground_truth_label_int,
                'ground_truth_eng_label_str': ground_truth_label_str, # Target for accuracy
                'premise_en': processed_output.get("premise_en", "[ErrorSP_Eval]"),
                'hypothesis_en': processed_output.get("hypothesis_en", "[ErrorSP_Eval]"),
                'predicted_english_label_model_intermediate': processed_output.get("predicted_english_label_model_intermediate", "[ErrorSP_Eval]"),
                'predicted_lrl_label_model_raw': processed_output.get("predicted_lrl_label_model_raw", "N/A_SP_EN_eval"),
                'predicted_label_for_accuracy': processed_output.get("predicted_label_for_accuracy", "neutral"),
                'raw_model_response': processed_output.get("raw_response", "[ErrorSP_Eval]"),
                'runtime_seconds': processed_output.get("runtime_seconds", 0.0),
                'error_message': None 
            }
            results_list.append(current_result)

        except Exception as e_eval_single_sample:
            logger.error(f"CRITICAL: Error processing single-prompt sample {idx} for lang {lang_code} in evaluate_nli_cotr_single_prompt: {e_eval_single_sample}", exc_info=True)
            error_entry = base_error_dict_structure.copy()
            error_entry['error_message'] = f"Sample processing error in eval: {str(e_eval_single_sample)}"
            results_list.append(error_entry)

    total_runtime = time.time() - total_start_time
    results_df = pd.DataFrame(results_list)
    logger.info(f"Finished Single-Prompt NLI CoTR eval for {model_name} on {lang_code}. Total runtime: {total_runtime:.2f}s")
    return results_df

# calculate_nli_metrics is NOT defined here. It's imported by the runner script.
# from src.experiments.baseline.nli.nli_baseline import calculate_nli_metrics
# This was in the new runner. The old nli_cotr.py had a local definition, which we are removing.

