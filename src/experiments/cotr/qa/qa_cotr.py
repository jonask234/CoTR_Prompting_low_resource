from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import time
from tqdm import tqdm
import re
import logging
import json
import sys
import os
import argparse
import numpy as np
from huggingface_hub import login
from config import get_token
from src.utils.data_loaders.load_tydiqa import TYDIQA_LANG_CONFIG_MAP
from evaluation.cotr.translation_metrics import COMET_AVAILABLE, calculate_comet_score as calculate_translation_quality

# --- Global Logger ---
logger = logging.getLogger(__name__)

# Define language names dictionary at module level
LANG_NAMES = {
    "en": "English",
    "sw": "Swahili",
    "fi": "Finnish",
    # Add other languages if they become relevant from TyDiQA or other datasets
    "hi": "Hindi", "vi": "Vietnamese", "bn": "Bengali", "id": "Indonesian",
    "ko": "Korean", "ru": "Russian", "ar": "Arabic", "th": "Thai"
}

def get_language_name(lang_code: str) -> str:
    """Get full language name from language code, using LANG_NAMES."""
    return LANG_NAMES.get(lang_code.lower(), lang_code.capitalize())

def initialize_model(model_name: str, cache_path: Optional[str] = "/work/bbd6522/cache_dir") -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Initialize a model for QA task, with robust pad_token handling."""
    logger.info(f"Loading QA CoTR model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_path,
            low_cpu_mem_usage=True
        )
    except Exception as e_load_auto_fp16:
        logger.error(f"Failed to load model {model_name} with device_map='auto' and float16. Error: {e_load_auto_fp16}")
        try:
            logger.info(f"Attempting to load {model_name} without device_map and with default dtype...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_path
            )
            if torch.cuda.is_available():
                logger.info(f"Manually moving {model_name} to CUDA and attempting float16 conversion.")
                try:
                    model = model.to("cuda").half()
                except Exception as e_cuda_half:
                    logger.error(f"Failed to move to CUDA and convert to float16: {e_cuda_half}. Using full precision on CUDA if possible.")
                    model = model.to("cuda") # Try moving to CUDA without half()
        except Exception as e_load_fallback:
            logger.critical(f"CRITICAL: Fallback loading for {model_name} also failed: {e_load_fallback}. Cannot proceed with this model.")
            raise 

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"QA CoTR: Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token}) for {model_name}")
        else:
            new_pad_token = '[PAD]' 
            tokenizer.add_special_tokens({'pad_token': new_pad_token})
            model.resize_token_embeddings(len(tokenizer)) 
            logger.info(f"QA CoTR: Added new pad_token '{new_pad_token}' and resized model embeddings for {model_name}.")

    if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.info(f"QA CoTR: Aligned model.config.pad_token_id with tokenizer.pad_token_id ({tokenizer.pad_token_id}) for {model_name}")
    
    logger.info(f"QA CoTR model {model_name} loaded. Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}. Model config pad_token_id: {model.config.pad_token_id}")

    if tokenizer.eos_token_id is None:
        logger.warning(f"Tokenizer for {model_name} has no eos_token_id. model.config.eos_token_id is {model.config.eos_token_id}.")

    logger.info(f"Post-configuration check for {model_name}:")
    
    pad_token_id_tok = getattr(tokenizer, 'pad_token_id', 'N/A')
    eos_token_id_tok = getattr(tokenizer, 'eos_token_id', 'N/A')
    bos_token_id_tok = getattr(tokenizer, 'bos_token_id', 'N/A')
    
    pad_token_cfg = getattr(model.config, 'pad_token_id', 'N/A')
    eos_token_cfg = getattr(model.config, 'eos_token_id', 'N/A')
    bos_token_cfg = getattr(model.config, 'bos_token_id', 'N/A')

    logger.info(f"  Tokenizer: PAD ID = {pad_token_id_tok} (type: {type(pad_token_id_tok)}), Value: '{getattr(tokenizer, 'pad_token', 'N/A')}'")
    logger.info(f"  Tokenizer: EOS ID = {eos_token_id_tok} (type: {type(eos_token_id_tok)}), Value: '{getattr(tokenizer, 'eos_token', 'N/A')}'")
    logger.info(f"  Tokenizer: BOS ID = {bos_token_id_tok} (type: {type(bos_token_id_tok)}), Value: '{getattr(tokenizer, 'bos_token', 'N/A')}'")
    
    logger.info(f"  Model Config: PAD ID = {pad_token_cfg} (type: {type(pad_token_cfg)})")
    logger.info(f"  Model Config: EOS ID = {eos_token_cfg} (type: {type(eos_token_cfg)})")
    logger.info(f"  Model Config: BOS ID = {bos_token_cfg} (type: {type(bos_token_cfg)})")

    if pad_token_id_tok is not None and pad_token_id_tok != 'N/A':
        assert isinstance(pad_token_id_tok, int), \
            f"tokenizer.pad_token_id for {model_name} is not an int: {pad_token_id_tok} (type: {type(pad_token_id_tok)})"
    if eos_token_id_tok is not None and eos_token_id_tok != 'N/A':
        assert isinstance(eos_token_id_tok, int), \
            f"tokenizer.eos_token_id for {model_name} is not an int: {eos_token_id_tok} (type: {type(eos_token_id_tok)})"

    effective_pad_id_for_gen = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if effective_pad_id_for_gen is None and hasattr(model.config, 'eos_token_id') and model.config.eos_token_id is not None:
        effective_pad_id_for_gen = model.config.eos_token_id
        logger.warning(f"  Effective PAD ID for generate() for {model_name} is falling back to model.config.eos_token_id: {effective_pad_id_for_gen}")
    elif effective_pad_id_for_gen is None: 
        effective_pad_id_for_gen = 0 
        logger.error(f"  CRITICAL: Effective PAD ID for generate() for {model_name} could not be determined. Defaulting to {effective_pad_id_for_gen}. THIS IS LIKELY WRONG.")

    effective_eos_id_for_gen = tokenizer.eos_token_id
    if effective_eos_id_for_gen is None and hasattr(model.config, 'eos_token_id') and model.config.eos_token_id is not None:
        effective_eos_id_for_gen = model.config.eos_token_id
        logger.warning(f"  Effective EOS ID for generate() for {model_name} is falling back to model.config.eos_token_id: {effective_eos_id_for_gen}")
    elif effective_eos_id_for_gen is None: 
        logger.error(f"  CRITICAL: Effective EOS ID for generate() for {model_name} could not be determined. THIS IS LIKELY WRONG.")

    logger.info(f"  Effective PAD ID for generate(): {effective_pad_id_for_gen} (type: {type(effective_pad_id_for_gen)})")
    logger.info(f"  Effective EOS ID for generate(): {effective_eos_id_for_gen} (type: {type(effective_eos_id_for_gen)})")
    
    if effective_pad_id_for_gen is not None:
        assert isinstance(effective_pad_id_for_gen, int), f"Effective PAD ID for generation for {model_name} is not an int: {effective_pad_id_for_gen} (type: {type(effective_pad_id_for_gen)})"
    if effective_eos_id_for_gen is not None:
        assert isinstance(effective_eos_id_for_gen, int), f"Effective EOS ID for generation for {model_name} is not an int: {effective_eos_id_for_gen} (type: {type(effective_eos_id_for_gen)})"
        
    # If Qwen model, ensure tokenizer.model is not set to the main model object
    # to prevent tokenizer() call from trying to run the full model.
    if "qwen" in model_name.lower():
        if hasattr(tokenizer, 'model') and tokenizer.model is model:
            logger.warning(f"QA CoTR Init: Qwen tokenizer's .model attribute was set to the main model. Setting it to None to ensure tokenizer() only tokenizes.")
            tokenizer.model = None
            
    return tokenizer, model

def _escape_fstring_val(value: Any) -> str:
    """Escapes characters that might break f-string definitions if f-strings use triple quotes."""
    if value is None: return ""
    return str(value).replace("'''", "\\'\\'\\'").replace('"""', '\\"\\"\\"')

def generate_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """Generate a prompt for translation. Based on old script's specialized prompts."""
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)

    # Use specialized prompts from old script for higher quality translations
    if target_lang == 'en':
        if source_lang == 'sw':
            prompt_text = f"""Original Text (Swahili):
'{_escape_fstring_val(text)}'

Instructions:
Translate this Swahili text to English with maximum accuracy and fluency.
Preserve the full meaning, all named entities, numbers, dates, and any specific details critical for answering questions based on this text.
Provide ONLY the English translation.

English Translation:"""
            return prompt_text
        elif source_lang == 'fi': # Added Finnish to English translation prompt
            prompt_text = f"""Original Text (Finnish):
'{_escape_fstring_val(text)}'

Instructions:
Translate this Finnish text to English with maximum accuracy and fluency.
Preserve the full meaning, all named entities, numbers, dates, and any specific details critical for answering questions based on this text.
Provide ONLY the English translation.

English Translation:"""
            return prompt_text
    elif source_lang == 'en': # English to LRL (for answer back-translation)
        if target_lang == 'sw':
            prompt_text = f"""Original Text (English):
'{_escape_fstring_val(text)}'

Instructions:
Translate this English text to natural, fluent, and grammatically correct Swahili.
Preserve all factual content, names, and numerical information exactly.
For yes/no answers, use the standard Swahili equivalent (e.g., 'Ndiyo' or 'Hapana').
Provide ONLY the Swahili translation.

Swahili Translation:"""
            return prompt_text
        elif target_lang == 'fi': # Added English to Finnish translation prompt (for answer)
            prompt_text = f"""Original Text (English):
'{_escape_fstring_val(text)}'

Instructions:
Translate this English text to natural, fluent, and grammatically correct Finnish.
Preserve all factual content, names, and numerical information exactly.
For yes/no answers, use the standard Finnish equivalent (e.g., 'Kyllä' or 'Ei').
Provide ONLY the Finnish translation.

Finnish Translation:"""
            return prompt_text
            
    # Fallback to a more general prompt if specific LRL pair isn't covered above
    return f"""Translate the following text from {source_name} to {target_name}.
{source_name} text: "{_escape_fstring_val(text)}"
{target_name} translation:"""

def generate_qa_prompt(question: str, context_en: str, use_few_shot: bool = True) -> str:
    """Generate the QA prompt (for English QA) with explicit instructions. Based on old script."""
    safe_question = _escape_fstring_val(question)
    safe_context = _escape_fstring_val(context_en)

    # Drastically simplifying examples string to isolate linter error
    examples = "" # Set to empty string for now
    if use_few_shot:
        # A very simple, single example to minimize complexity for the linter
        examples = (
            '\nExamples:\n'
            'Context: \\\'Paris is the capital of France.\\\'\n'
            'Question: \\\'What is the capital of France?\\\'\n'
            'Answer: Paris\n'
        )

    prompt_base = f"""Context: '{safe_context}'
Question: '{safe_question}'

Answer the question accurately and very concisely using ONLY the information provided in the context above.
Follow these rules strictly:
- For yes/no questions: respond with ONLY 'Yes' or 'No'.
- For factual questions: provide ONLY the specific answer found in the context.
- Do NOT add any explanations, disclaimers, or conversational phrases like "The answer is..." or "Based on the context...".
- Your entire response should ideally be a short phrase or name extracted directly from the context.
- If the answer cannot be found in the context, respond with ONLY the exact phrase 'I don\\'t know'.
"""

    if use_few_shot and examples: # Check if examples is not empty
        prompt = f"""{prompt_base}
{examples}
Answer:"""
    else:
        prompt = f"""{prompt_base}
Answer:"""
    return prompt

def generate_single_prompt_qa_cotr(lrl_question: str, lrl_context: str, lang_code: str, use_few_shot: bool = True) -> str:
    """Generate a single CoTR prompt for QA (based on old script, LRL input, English CoT)."""
    lrl_name = get_language_name(lang_code)
    safe_lrl_question = _escape_fstring_val(lrl_question)
    safe_lrl_context = _escape_fstring_val(lrl_context)

    # Core instructions using triple-double-quotes for the f-string.
    # Python's f-string triple quotes handle newlines well.
    # Ensure internal 'I don\\'t know' is properly escaped for a Python string literal.
    
    # Adjust prompt format for English to avoid redundant steps
    if lang_code == 'en':
        format_instruction = """English Question: [Your English Translation of the Question]
English Context: [Your English Translation of the Context]
English Answer: [Your English Answer based on the English Context and Question]"""
    else:
        format_instruction = f"""English Question: [Your English Translation of the Question]
English Context: [Your English Translation of the Context]
English Answer: [Your English Answer based on the English Context and Question]
{lrl_name} Answer: [Your {lrl_name} Translation of the English Answer]"""

    instructions_core = f"""Perform the following tasks in order for the provided {lrl_name} question and context:
1.  Translate the {lrl_name} Question to English.
2.  Translate the {lrl_name} Context to English.
3.  Based ONLY on your English translations of the context and question, answer the English Question. For Yes/No questions, answer 'Yes' or 'No'. If the answer is not in the context, say 'I don\\'t know'.
4.  Translate your English Answer from step 3 back to {lrl_name}.

Provide your answer in this exact format:
{format_instruction}"""

    few_shot_examples_str = "" # Initialize as empty
    if use_few_shot:
        lrl_name_for_prompt = get_language_name(lang_code)
        # Define few-shot examples (CoT in English, LRL parts vary)
        # Example 1: Question about a known fact, answer directly in context
        ex1_lrl_q = f"Nani alikuwa rais wa kwanza wa Marekani?" if lang_code == "sw" else (f"అమెరికా మొదటి అధ్యక్షుడు ఎవరు?" if lang_code == "te" else (f"Kuka oli Yhdysvaltain ensimmäinen presidentti?" if lang_code == "fi" else f"[Question about first US president in {lrl_name_for_prompt}]"))
        ex1_lrl_c = f"George Washington alikuwa rais wa kwanza wa Marekani. Alihudumu kuanzia 1789 hadi 1797." if lang_code == "sw" else (f"జార్జ్ వాషింగ్టన్ అమెరికా మొదటి అధ్యక్షుడు. అతను 1789 నుండి 1797 వరకు పనిచేశారు." if lang_code == "te" else (f"George Washington oli Yhdysvaltain ensimmäinen presidentti. Hän palveli vuosina 1789-1797." if lang_code == "fi" else f"[Context about George Washington in {lrl_name_for_prompt}]"))
        ex1_en_q = "Who was the first president of the United States?"
        ex1_en_c = "George Washington was the first president of the United States. He served from 1789 to 1797."
        ex1_en_a = "George Washington"
        ex1_lrl_a_map = {"sw": "George Washington", "te": "జార్జ్ వాషింగ్టన్", "fi": "George Washington"}
        ex1_lrl_a = ex1_lrl_a_map.get(lang_code, f"[George Washington in {lrl_name_for_prompt}]")

        ex2_lrl_q = f"Rangi ya anga kwenye Mirihi ni ipi?" if lang_code == "sw" else (f"అంగారకుడిపై ఆకాశం రంగు ఏమిటి?" if lang_code == "te" else (f"Mikä on Marsin taivaan väri?" if lang_code == "fi" else f"[Question about Mars sky in {lrl_name_for_prompt}]"))
        ex2_lrl_c = f"Mirihi ni sayari ya nne kutoka Jua." if lang_code == "sw" else (f"అంగారకుడు సూర్యుని నుండి నాల్గవ గ్రహం." if lang_code == "te" else (f"Mars on neljäs planeetta Auringosta." if lang_code == "fi" else f"[Context about Mars in {lrl_name_for_prompt}]"))
        ex2_en_q = "What is the color of the sky on Mars?"
        ex2_en_c = "Mars is the fourth planet from the Sun."
        ex2_en_a = "I don't know"
        ex2_lrl_a_map = {"sw": "Sijui", "fi": "En tiedä"}
        ex2_lrl_a = ex2_lrl_a_map.get(lang_code, f"[Equivalent of I don't know in {lrl_name_for_prompt}]")

        # Handle final answer label for examples based on language
        if lang_code == 'en':
            final_answer_label_ex1 = "English Answer"
            final_answer_val_ex1 = ex1_en_a
            final_answer_label_ex2 = "English Answer"
            final_answer_val_ex2 = ex2_en_a
        else:
            final_answer_label_ex1 = f"{lrl_name} Answer"
            final_answer_val_ex1 = ex1_lrl_a
            final_answer_label_ex2 = f"{lrl_name} Answer"
            final_answer_val_ex2 = ex2_lrl_a

        ex1_lrl_q_esc, ex1_lrl_c_esc, ex1_en_q_esc, ex1_en_c_esc, ex1_en_a_esc, ex1_lrl_a_esc = map(_escape_fstring_val, [ex1_lrl_q, ex1_lrl_c, ex1_en_q, ex1_en_c, ex1_en_a, final_answer_val_ex1])
        ex2_lrl_q_esc, ex2_lrl_c_esc, ex2_en_q_esc, ex2_en_c_esc, ex2_en_a_esc, ex2_lrl_a_esc = map(_escape_fstring_val, [ex2_lrl_q, ex2_lrl_c, ex2_en_q, ex2_en_c, ex2_en_a, final_answer_val_ex2])
        
        # Constructing the few_shot_examples_str using triple-double-quotes f-string.
        # Variables are already escaped by _escape_fstring_val.
        few_shot_examples_str = f"""
--- Examples ---

Example 1:
Original {lrl_name} Question: '{ex1_lrl_q_esc}'
Original {lrl_name} Context: '{ex1_lrl_c_esc}'

English Question: {ex1_en_q_esc}
English Context: {ex1_en_c_esc}
English Answer: {ex1_en_a_esc}
{final_answer_label_ex1}: {ex1_lrl_a_esc}

Example 2:
Original {lrl_name} Question: '{ex2_lrl_q_esc}'
Original {lrl_name} Context: '{ex2_lrl_c_esc}'

English Question: {ex2_en_q_esc}
English Context: {ex2_en_c_esc}
English Answer: {ex2_en_a_esc}
{final_answer_label_ex2}: {ex2_lrl_a_esc}
"""

    # Task prompt also uses triple-double-quotes f-string
    task_prompt_str = f"""
--- Your Task ---
Original {lrl_name} Question: '{safe_lrl_question}'
Original {lrl_name} Context: '{safe_lrl_context}'

Follow all steps and provide your full response as per the format:"""
        
    final_prompt = instructions_core
    if use_few_shot and few_shot_examples_str:
        final_prompt += "\n" + few_shot_examples_str
    final_prompt += "\n" + task_prompt_str
    
    prompt_preview = final_prompt[:600] # Pre-slice the prompt for logging
    logger.debug(f"Generated Single QA CoTR Prompt for {lang_code} (use_few_shot={use_few_shot}):\n{prompt_preview}...")
    return final_prompt

def clean_translation_response(response: str, target_lang: str, source_lang: str) -> str:
    """Clean the translation response (from old script)."""
    cleaned = response.strip()
    target_name = get_language_name(target_lang)
    # source_name = get_language_name(source_lang) # Not used in this version of cleaning
    
    prefixes_to_remove = [
        "translation:", f"{target_name.lower()} translation:", f"{target_name} translation:",
        f"{target_lang.upper()}:", f"{target_lang}:",
        f"{target_name.lower()}:", f"{target_name}:"
    ]
    # Specific prefixes from CoT single prompt output that might bleed into translation parts
    cot_prefixes = ["english question:", "english context:", f"{target_name.lower()} answer:"]
    prefixes_to_remove.extend(cot_prefixes)

    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()

    if (cleaned.startswith('"') and cleaned.endswith('"')) or \
       (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    
    return cleaned if cleaned else "[EMPTY_TRANSLATION]"

def is_yes_no_question(question: str) -> bool:
    """Determine if a question is likely a Yes/No question (from old script)."""
    if not question: return False
    question_lower = question.lower().strip()
    if re.match(r"^(is|are|was|were|do|does|did|has|have|had|can|could|shall|should|will|would|may|might|must)\\b", question_lower):
        return True
    if " or not" in question_lower:
        return True
    if question_lower.startswith("whether"):
        return True
    return False

def extract_answer(response: str, is_yes_no: bool = False, lang_code_for_lrl_answer: Optional[str] = None) -> str:
    """Extracts the answer from the model's raw QA output (based on old script)."""
    original_response = response 
    response = response.strip()
    if not response: return "I don't know"

    if lang_code_for_lrl_answer: # Extracting LRL answer from single-prompt CoT
        lrl_name = get_language_name(lang_code_for_lrl_answer)
        match = re.search(rf"{re.escape(lrl_name)}\\s+Answer\\s*:\\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        if match:
            answer_part = match.group(1).strip()
            # Remove trailing "---" or "Example" or "Original {lrl_name} Question:|English Question:" lines if any captured due to greedy regex
            answer_part = re.split(r"\\n\\s*(?:---|Example|Original {lrl_name} Question:|English Question:)", answer_part, 1)[0].strip()
            return answer_part if answer_part else "I don't know"
        else:
            logger.debug(f"QA CoTR Extract: LRL answer line not found for {lrl_name} in '{original_response[:100]}...'. Trying general extraction on whole response.")
            # Fall through to general extraction, but this is less ideal for LRL.

    # General extraction (primarily for English QA outputs)
    response_lower = response.lower()
    i_dont_know_variants = [
        "i don't know", "i do not know", "don't know", "cannot answer",
        "not in the context", "cannot be determined from the context",
        "answer not found", "no information", "unknown"
    ]
    for idk_variant in i_dont_know_variants:
        if idk_variant in response_lower:
            return "I don't know"

    if is_yes_no:
        if re.search(r"^yes[\\s.,;:!?]*$", response_lower): return "Yes"
        if re.search(r"^no[\\s.,;:!?]*$", response_lower): return "No"
        # LRL Yes/No checks for direct output (less likely for English step)
        if lang_code_for_lrl_answer == "sw":
            if "ndiyo" in response_lower: return "Ndiyo" # Or map to "Yes" if metrics are English-based
            if "hapana" in response_lower: return "Hapana" # Or map to "No"
        elif lang_code_for_lrl_answer == "fi":
            if "kyllä" in response_lower: return "Kyllä" # (yes)
            if "ei" in response_lower: return "Ei"   # (no)


    prefixes_to_strip = [
        "answer:", "the answer is", "based on the context, the answer is", 
        "the context states that", "according to the context", 
        "the final answer is", "concise answer:", "extracted answer:"
    ]
    for prefix in prefixes_to_strip:
        if response_lower.startswith(prefix):
            response = response[len(prefix):].strip()
            response_lower = response.lower()

    answer_lines = response.split('\\n')
    first_line_answer = answer_lines[0].strip()

    if is_yes_no: # If after stripping, it's just "yes" or "no"
        if first_line_answer.lower() == "yes": return "Yes"
        if first_line_answer.lower() == "no": return "No"
        # If it's a Y/N question but the answer isn't explicitly Yes/No/IDK, it's likely a bad extraction or non-compliant answer
        # For TyDiQA, this might mean the model didn't follow instructions.
        # We should return the extracted text and let F1/EM scoring handle it.
        # However, if we MUST categorize as Yes/No/IDK for some internal logic:
        # return "I don't know" # Fallback for unclear Y/N if not already IDK

    return first_line_answer if first_line_answer else "I don't know"

def translate_text( # For multi-prompt pipeline
    model: Any, tokenizer: Any, text: str, source_lang: str, target_lang: str,
    # Generation parameters passed from runner
    temperature: float, do_sample: bool, top_p: float, top_k: int, repetition_penalty: float, max_new_tokens: int,
    max_input_length: int = 4096, # Tokenizer max length
    is_answer_translation: bool = False # Hint if it's an answer being translated
) -> Tuple[str, str, float]: # Returns (translated_text, raw_model_output, duration)
    """Translate text using the provided model. Parameters are passed directly."""
    start_time = time.time()
    if not text or text.strip() == "" or text.strip().lower() == "[error]" or "[translation_error]" in text.strip().lower():
        logger.warning(f"QA CoTR Translate: Empty or error input text for {source_lang}->{target_lang}: '{text}'")
        return "[TRANSLATION_SKIPPED_EMPTY_INPUT]", "[TRANSLATION_SKIPPED_EMPTY_INPUT]", 0.0
    if source_lang == target_lang and not is_answer_translation: # No-op for non-answers if langs are same
         return text, text, 0.0

    prompt = generate_translation_prompt(text, source_lang, target_lang)
    
    inputs_dict = {}
    model_name_for_check = getattr(model, 'name_or_path', "").lower()

    if "coherelabs/aya" in model_name_for_check:
        logger.debug(f"QA CoTR Translate: Detected Cohere Aya model ('{model_name_for_check}'). Using apply_chat_template.")
        messages = [{"role": "user", "content": prompt}]
        try:
            tokenized_output = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                truncation=True, max_length=max_input_length
            )
            if isinstance(tokenized_output, dict) and 'input_ids' in tokenized_output:
                inputs_dict = tokenized_output
            elif isinstance(tokenized_output, torch.Tensor):
                inputs_dict = {"input_ids": tokenized_output}
            else:
                logger.error(f"QA CoTR Translate: apply_chat_template returned unexpected type for Aya. Type: {type(tokenized_output)}. Value: {str(tokenized_output)[:200]}")
                return "[TRANSLATION_ERROR_TOKENIZATION_AYA_UNEXPECTED_TYPE]", f"[RAW_TRANSLATION_ERROR_TOKENIZATION_AYA_UNEXPECTED_TYPE: {type(tokenized_output)}]", time.time() - start_time
        except Exception as e_chat_template:
            logger.error(f"QA CoTR Translate: Error apply_chat_template for {model_name_for_check}: {e_chat_template}", exc_info=True)
            return f"[TRANSLATION_ERROR_TOKENIZATION_AYA_EXC: {e_chat_template}]", f"[RAW_TRANSLATION_ERROR_TOKENIZATION_AYA_EXC: {e_chat_template}]", time.time() - start_time
    else: 
        inputs_dict = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)

    if not inputs_dict or 'input_ids' not in inputs_dict or inputs_dict['input_ids'].numel() == 0:
        logger.error(f"QA CoTR Translate: Tokenization failed or resulted in empty input_ids for model {model_name_for_check} on prompt: '{prompt[:100]}...'")
        return "[TRANSLATION_ERROR_TOKENIZATION]", "[RAW_TOKENIZATION_ERROR_EMPTY]", time.time() - start_time
        
    inputs = {k: v.to(model.device) for k, v in inputs_dict.items()}
    effective_do_sample = do_sample if temperature > 0.01 else False

    raw_model_output = "[TRANSLATION_FAILED_MODEL_ERROR_RAW]"
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=effective_do_sample,
                pad_token_id=model.config.pad_token_id
            )
        raw_model_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    except Exception as e_generate:
        logger.error(f"QA CoTR Translate: Error during model.generate for {source_lang}->{target_lang} on text '{_escape_fstring_val(text)[:50]}...': {e_generate}", exc_info=True)
        return "[TRANSLATION_FAILED_MODEL_ERROR]", raw_model_output, time.time() - start_time
    
    cleaned_response = clean_translation_response(raw_model_output, target_lang, source_lang)
    duration = time.time() - start_time
    logger.debug(f"Raw translation ({source_lang}->{target_lang}): '{raw_model_output[:100]}...' Cleaned: '{cleaned_response[:100]}...' (Duration: {duration:.2f}s)")
    return cleaned_response, raw_model_output, duration

def process_qa_english( # For multi-prompt's English QA step
    model: Any, tokenizer: Any, question_en: str, context_en: str, use_few_shot: bool,
    # Generation parameters passed from runner - CORRECTED ORDER
    temperature: float, do_sample: bool, top_p: float, top_k: int, repetition_penalty: float, max_new_tokens: int,
    max_input_length: int = 4096
) -> Tuple[str, str, float]: # Returns (extracted_answer_en, raw_model_output, duration)
    """Process a QA pair in English using the model. Parameters are passed directly."""
    start_time = time.time()
    if not question_en or not context_en or "[error]" in question_en.lower() or "[error]" in context_en.lower():
        logger.warning(f"QA CoTR process_qa_english: Empty or error input EN question/context. Q: '{question_en[:50]}...', C: '{context_en[:50]}...'")
        return "I don't know", "[QA_SKIPPED_EMPTY_INPUT]", 0.0

    prompt = generate_qa_prompt(question_en, context_en, use_few_shot=use_few_shot)
    
    inputs_dict = {}
    model_name_for_check = getattr(model, 'name_or_path', "").lower()
    if "coherelabs/aya" in model_name_for_check:
        logger.debug(f"QA CoTR English QA: Detected Cohere Aya model ('{model_name_for_check}'). Using apply_chat_template.")
        messages = [{"role": "user", "content": prompt}]
        try:
            tokenized_output = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                truncation=True, max_length=max_input_length
            )
            if isinstance(tokenized_output, dict) and 'input_ids' in tokenized_output:
                inputs_dict = tokenized_output
            elif isinstance(tokenized_output, torch.Tensor):
                inputs_dict = {"input_ids": tokenized_output}
            else:
                logger.error(f"QA CoTR English QA: apply_chat_template returned unexpected type for Aya. Type: {type(tokenized_output)}. Value: {str(tokenized_output)[:200]}")
                return "I don't know", f"[QA_ERROR_TOKENIZATION_AYA_UNEXPECTED_TYPE]", time.time() - start_time
        except Exception as e_chat_template_qa:
            logger.error(f"QA CoTR English QA: Error apply_chat_template for {model_name_for_check}: {e_chat_template_qa}", exc_info=True)
            return "I don't know", f"[QA_ERROR_TOKENIZATION_AYA_EXC: {e_chat_template_qa}]", time.time() - start_time
    else:
        inputs_dict = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)

    if not inputs_dict or 'input_ids' not in inputs_dict or inputs_dict['input_ids'].numel() == 0:
        logger.error(f"QA CoTR English QA: Tokenization failed or empty input_ids for model {model_name_for_check} on prompt: '{prompt[:100]}...'")
        return "I don't know", "[QA_ERROR_TOKENIZATION]", time.time() - start_time

    inputs = {k: v.to(model.device) for k, v in inputs_dict.items()}
    effective_do_sample = do_sample if temperature > 0.01 else False
    
    raw_model_output_en = "[MODEL_GENERATION_ERROR_QA_EN_RAW]"
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens, # Specific for answer
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=effective_do_sample,
                pad_token_id=model.config.pad_token_id
            )
        raw_model_output_en = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    except Exception as e_generate_qa_en:
        logger.error(f"QA CoTR English QA: Error during model.generate: {e_generate_qa_en}", exc_info=True)
        # raw_model_output_en keeps its error state
    
    is_yn_q = is_yes_no_question(question_en)
    extracted_answer_en = extract_answer(raw_model_output_en, is_yes_no=is_yn_q, lang_code_for_lrl_answer=None)
    duration = time.time() - start_time
    
    logger.debug(f"English QA Step: Q='{question_en[:50]}...', RawOut='{raw_model_output_en[:50]}...', ExtractedAns='{extracted_answer_en}' (Duration: {duration:.2f}s)")
    return extracted_answer_en, raw_model_output_en, duration

def evaluate_qa_cotr( # For multi-prompt pipeline (from old script, adapted)
    model: Any, tokenizer: Any, samples_df: pd.DataFrame, lang_code: str, use_few_shot: bool,
    # Generation parameters (already specific per step, passed from runner)
    qa_params: Dict[str, Any], 
    translation_params: Dict[str, Any],
    max_input_length: int = 4096,
    model_name: Optional[str] = None # Added model_name for logging
) -> pd.DataFrame:
    """Evaluate QA using the Multi-Prompt CoTR approach. Adapted from old script."""
    results_list = []
    total_start_time = time.time()
    
    qa_temp, qa_top_p, qa_top_k, qa_max_new_tokens, qa_rep_penalty, qa_do_sample = (
        qa_params['temperature'], qa_params['top_p'], qa_params['top_k'], 
        qa_params['max_new_tokens'], qa_params['repetition_penalty'], qa_params['do_sample']
    )
    trans_temp, trans_top_p, trans_top_k, trans_max_new_tokens, trans_rep_penalty, trans_do_sample = (
        translation_params['temperature'], translation_params['top_p'], translation_params['top_k'],
        translation_params['max_new_tokens'], translation_params['repetition_penalty'], translation_params['do_sample']
    )

    model_name_for_log = model_name if model_name else "UnknownModel"
    results_label = 'few-shot' if use_few_shot else 'zero-shot'
    
    # Log parameters being used more consistently
    logger.info(f"Starting Multi-Prompt QA CoTR evaluation for {getattr(model, 'name_or_path', 'unknown_model')} on {lang_code} ({results_label}).")
    logger.info(f"  QA Step Params: temp={qa_temp}, top_p={qa_top_p}, top_k={qa_top_k}, max_new_tok={qa_max_new_tokens}, rep_pen={qa_rep_penalty}, do_sample={qa_do_sample}")
    logger.info(f"  Translation Params: temp={trans_temp}, top_p={trans_top_p}, top_k={trans_top_k}, max_new_tok={trans_max_new_tokens}, rep_pen={trans_rep_penalty}, do_sample={trans_do_sample}")
    logger.info(f"  Max Input Length for tokenization: {max_input_length}")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Multi-Prompt QA CoTR {lang_code}"):
        sample_start_time = time.time() # For per-sample runtime
        sample_id = str(row.get('example_id', row.get('id', f'sample_{idx}'))) # Keep original sample_id logic
        lrl_question_orig = str(row.get('question', '')) # Get original before strip
        lrl_context_orig = str(row.get('context', ''))   # Get original before strip

        lrl_question = lrl_question_orig.strip()
        lrl_context = lrl_context_orig.strip()

        # Ground truth answers - ensure it's a list of strings
        gt_answers_raw = row.get('answers', {}) # TyDiQA specific, or 'lrl_ground_truth_answers_list' if preprocessed
        lrl_ground_truth_texts_list = []
        if 'lrl_ground_truth_answers_list' in row and isinstance(row['lrl_ground_truth_answers_list'], list):
            lrl_ground_truth_texts_list = [str(ans) for ans in row['lrl_ground_truth_answers_list'] if ans]
        elif isinstance(gt_answers_raw, dict) and 'text' in gt_answers_raw and isinstance(gt_answers_raw['text'], list):
            lrl_ground_truth_texts_list = [str(ans) for ans in gt_answers_raw['text'] if ans]
        elif isinstance(gt_answers_raw, list) and all(isinstance(ans, str) for ans in gt_answers_raw): # If it's already a list of strings
             lrl_ground_truth_texts_list = [str(ans) for ans in gt_answers_raw if ans]
        elif isinstance(gt_answers_raw, str) and gt_answers_raw: # If GT is a single string
             lrl_ground_truth_texts_list = [gt_answers_raw]


        current_result = {
            'id': sample_id, 
            'lrl_question': lrl_question_orig, # Log original
            'lrl_context': lrl_context_orig,   # Log original
            'lrl_ground_truth_answers_list': lrl_ground_truth_texts_list,
            'en_question_model': "[ErrorQTrans]", 'en_context_model': "[ErrorCTrans]",
            'en_answer_model_intermediate': "[ErrorEN_QA]", 'lrl_answer_model_final': "[ErrorAnsBTrans]",
            'comet_lrl_q_to_en': None, 'comet_lrl_c_to_en': None, 'comet_en_a_to_lrl': None,
            'raw_q_translation_output':"", 'raw_c_translation_output':"", 
            'raw_en_qa_output':"", 'raw_a_backtranslation_output':"",
            'error_message': None, 'runtime_sec': 0.0 
        }

        # Skip if LRL question or context is effectively empty AFTER stripping
        if not lrl_question or not lrl_context:
            error_msg_parts = []
            if not lrl_question: 
                error_msg_parts.append("LRL Question is empty or whitespace.")
                current_result['en_question_model'] = '[SKIPPED_EMPTY_LRL_Q]'
            if not lrl_context: 
                error_msg_parts.append("LRL Context is empty or whitespace.")
                current_result['en_context_model'] = '[SKIPPED_EMPTY_LRL_C]'
            
            current_result['error_message'] = " | ".join(error_msg_parts)
            logger.warning(f"Skipping sample {sample_id} ({lang_code}) due to: {current_result['error_message']}")
            current_result['runtime_sec'] = time.time() - sample_start_time
            results_list.append(current_result)
            continue
        
        try:
            # --- 1. Translate LRL Question to English --- #
            en_question, raw_q_trans, _ = translate_text(
                model, tokenizer, lrl_question, lang_code, "en",
                temperature=trans_temp, do_sample=trans_do_sample, top_p=trans_top_p, top_k=trans_top_k,
                repetition_penalty=trans_rep_penalty, max_new_tokens=trans_max_new_tokens, max_input_length=max_input_length
            )
            current_result['en_question_model'] = en_question
            current_result['raw_q_translation_output'] = raw_q_trans
            if "[error]" in en_question.lower() or "[skipped]" in en_question.lower(): raise ValueError(f"QTransFail: {en_question}")

            # --- 2. Translate LRL Context to English --- #
            en_context, raw_c_trans, _ = translate_text(
                model, tokenizer, lrl_context, lang_code, "en",
                temperature=trans_temp, do_sample=trans_do_sample, top_p=trans_top_p, top_k=trans_top_k,
                repetition_penalty=trans_rep_penalty, max_new_tokens=trans_max_new_tokens, max_input_length=max_input_length
            )
            current_result['en_context_model'] = en_context
            current_result['raw_c_translation_output'] = raw_c_trans
            if "[error]" in en_context.lower() or "[skipped]" in en_context.lower(): raise ValueError(f"CTransFail: {en_context}")

            if COMET_AVAILABLE:
                if lrl_question and en_question and "[error]" not in en_question.lower() and calculate_translation_quality:
                    comet_q = calculate_translation_quality(sources=[lrl_question], predictions=[en_question], references=[[lrl_question]])
                    current_result['comet_lrl_q_to_en'] = comet_q['mean_score'] if isinstance(comet_q, dict) else comet_q
                if lrl_context and en_context and "[error]" not in en_context.lower() and calculate_translation_quality:
                    comet_c = calculate_translation_quality(sources=[lrl_context], predictions=[en_context], references=[[lrl_context]])
                    current_result['comet_lrl_c_to_en'] = comet_c['mean_score'] if isinstance(comet_c, dict) else comet_c
            
            # --- 3. Process English QA --- #
            answer_en, raw_qa_output, _ = process_qa_english(
                model, tokenizer, en_question, en_context, use_few_shot,
                # Generation parameters passed from runner - CORRECTED ORDER
                temperature=qa_temp, 
                do_sample=qa_do_sample, 
                top_p=qa_top_p, 
                top_k=qa_top_k, 
                repetition_penalty=qa_rep_penalty,
                max_new_tokens=qa_max_new_tokens
                # max_input_length is not passed from qa_params here, process_qa_english uses its default or takes it separately
            )
            current_result['en_answer_model_intermediate'] = answer_en
            current_result['raw_en_qa_output'] = raw_qa_output
            if "[error]" in answer_en.lower() or "[skipped]" in answer_en.lower(): raise ValueError(f"EN_QA_Fail: {answer_en}")

            # --- 4. Translate English Answer to LRL Answer --- #
            lrl_answer_final = "[BACKTRANSLATION_SKIPPED_OR_FAILED]"
            raw_a_btrans = ""
            if answer_en.strip().lower() not in ["[error]", "i don't know", "[qa_error_tokenization]", "[qa_skipped_empty_input]", "[model_generation_error_qa_en_raw]"]:
                lrl_answer_final, raw_a_btrans, _ = translate_text(
                    model, tokenizer, answer_en, "en", lang_code,
                    temperature=trans_temp, do_sample=trans_do_sample, top_p=trans_top_p, top_k=trans_top_k,
                    repetition_penalty=trans_rep_penalty, max_new_tokens=trans_max_new_tokens, 
                    max_input_length=max_input_length, is_answer_translation=True
                )
                if COMET_AVAILABLE and answer_en and lrl_answer_final and "[error]" not in lrl_answer_final.lower() and calculate_translation_quality:
                    comet_a = calculate_translation_quality(sources=[answer_en], predictions=[lrl_answer_final], references=[[answer_en]])
                    current_result['comet_en_a_to_lrl'] = comet_a['mean_score'] if isinstance(comet_a, dict) else comet_a
            elif answer_en.strip().lower() == "i don't know":
                 lrl_answer_final, raw_a_btrans, _ = translate_text(
                    model, tokenizer, "I don't know", "en", lang_code, 
                    temperature=trans_temp, do_sample=trans_do_sample, top_p=trans_top_p, top_k=trans_top_k,
                    repetition_penalty=trans_rep_penalty, max_new_tokens=trans_max_new_tokens,
                    max_input_length=max_input_length, is_answer_translation=True
                )
            current_result['lrl_answer_model_final'] = lrl_answer_final
            current_result['raw_a_backtranslation_output'] = raw_a_btrans

        except Exception as e_sample:
            logger.error(f"Error processing MP QA sample {sample_id} ({lang_code}): {e_sample}", exc_info=False)
            current_result['error_message'] = str(e_sample)
        
        current_result['runtime_sec'] = time.time() - sample_start_time
        results_list.append(current_result)

    results_df = pd.DataFrame(results_list)
    logger.info(f"Finished Multi-Prompt QA CoTR eval for {getattr(model, 'name_or_path', '?')} on {lang_code}. Runtime: {(time.time() - total_start_time):.2f}s")
    return results_df

def extract_parts_from_single_prompt_qa_response(response_text: str, lang_code: str) -> Dict[str, str]:
    """Extracts intermediate English translations and final LRL answer from single-prompt CoT QA output."""
    parts = {
        "en_question_model": "[SP_Extract_Error_EN_Q]", "en_context_model": "[SP_Extract_Error_EN_C]",
        "en_answer_model_intermediate": "[SP_Extract_Error_EN_A]", "lrl_answer_model_final": "[SP_Extract_Error_LRL_A]"
    }
    lrl_name = get_language_name(lang_code)

    match_q_en = re.search(r"English\s+Question\s*:\s*(.*?)(?=\nEnglish\s+Context\s*:|\nEnglish\s+Answer\s*:|\n" + re.escape(lrl_name) + r"\s+Answer\s*:|$)", response_text, re.IGNORECASE | re.DOTALL)
    if match_q_en: parts["en_question_model"] = match_q_en.group(1).strip()

    match_c_en = re.search(r"English\s+Context\s*:\s*(.*?)(?=\nEnglish\s+Answer\s*:|\n" + re.escape(lrl_name) + r"\s+Answer\s*:|$)", response_text, re.IGNORECASE | re.DOTALL)
    if match_c_en: parts["en_context_model"] = match_c_en.group(1).strip()

    # Handling for English Answer is different for EN vs LRLs
    if lang_code == 'en':
        # For English, the English Answer is the final part. The response might end there.
        match_a_en = re.search(r"English\s+Answer\s*:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
    else:
        # For LRLs, we expect a final LRL answer to follow, so the match should be non-greedy.
        match_a_en = re.search(r"English\s+Answer\s*:\s*(.*?)(?=\n" + re.escape(lrl_name) + r"\s+Answer\s*:|$)", response_text, re.IGNORECASE | re.DOTALL)

    if match_a_en:
        # Clean up potential trailing junk for the greedy english match
        answer_part = match_a_en.group(1).strip()
        answer_part = re.split(r"\\n\\s*(?:---|Example|Original {lrl_name} Question:|English Question:)", answer_part, 1)[0].strip()
        parts["en_answer_model_intermediate"] = answer_part

    # Handling for final LRL Answer
    if lang_code == 'en':
        # For English, the final answer is just the intermediate English answer.
        if parts["en_answer_model_intermediate"] != "[SP_Extract_Error_EN_A]":
             parts["lrl_answer_model_final"] = parts["en_answer_model_intermediate"]
        else:
            logger.warning(f"QA CoTR SP Extract: Could not extract English answer for English sample. Response: '{response_text[:150]}...'")
    else:
        # For LRLs, we must find the LRL answer section.
        match_a_lrl = re.search(rf"{re.escape(lrl_name)}\s+Answer\s*:\s*(.+?)(?=\n---|\nExample|\nOriginal|\nEnglish\s+Question\s*:|$)", response_text, re.IGNORECASE | re.DOTALL)
        if match_a_lrl:
            parts["lrl_answer_model_final"] = match_a_lrl.group(1).strip()
        else:
            logger.warning(f"QA CoTR SP Extract: Final LRL Answer line not found for {lrl_name} in response: '{response_text[:150]}...'")

    if parts["en_answer_model_intermediate"] != "[SP_Extract_Error_EN_A]":
        is_yn_q_intermediate = is_yes_no_question(parts["en_question_model"] if parts["en_question_model"] else "")
        parts["en_answer_model_intermediate"] = extract_answer(parts["en_answer_model_intermediate"], is_yes_no=is_yn_q_intermediate)
    return parts

def evaluate_qa_cotr_single_prompt( 
    model: Any, tokenizer: Any, samples_df: pd.DataFrame, lang_code: str, use_few_shot: bool,
    temperature: float, do_sample: bool, top_p: float, top_k: int, repetition_penalty: float, max_new_tokens: int,
    max_input_length: int = 4096,
    model_name: Optional[str] = None # Added model_name for logging
) -> pd.DataFrame:
    """Evaluate QA using the Single-Prompt CoTR approach. Adapted from old script."""
    results_list = []
    total_start_time = time.time()
    logger.info(f"Starting Single-Prompt QA CoTR evaluation for {getattr(model, 'name_or_path', '?')} on {lang_code} ({'fs' if use_few_shot else 'zs'}).")
    logger.info(f"  Chain Params: temp={temperature}, top_p={top_p}, top_k={top_k}, max_new_tok={max_new_tokens}, rep_pen={repetition_penalty}, do_sample={do_sample}")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Single-Prompt QA CoTR {lang_code}"):
        sample_id = str(row.get('example_id', row.get('id', f'sample_{idx}')))
        lrl_question = str(row['question'])
        lrl_context = str(row['context'])
        gt_answers_raw = row.get('answers', {})
        lrl_ground_truth_texts_list = [] # Initialize as list
        if isinstance(gt_answers_raw, dict) and 'text' in gt_answers_raw and isinstance(gt_answers_raw['text'], list):
            lrl_ground_truth_texts_list = [str(ans) for ans in gt_answers_raw['text'] if ans]
        elif isinstance(gt_answers_raw, list) and all(isinstance(ans, str) for ans in gt_answers_raw):
             lrl_ground_truth_texts_list = [str(ans) for ans in gt_answers_raw if ans]
        elif isinstance(gt_answers_raw, str) and gt_answers_raw:
            lrl_ground_truth_texts_list = [gt_answers_raw]
        
        current_result = {
            'id': sample_id, 'lrl_question': lrl_question, 'lrl_context': lrl_context, 'lrl_ground_truth_answers_list': lrl_ground_truth_texts_list,
            'en_question_model': "[SP_Error]", 'en_context_model': "[SP_Error]",
            'en_answer_model_intermediate': "[SP_Error]", 'lrl_answer_model_final': "[SP_Error]",
            'comet_lrl_q_to_en': None, 'comet_lrl_c_to_en': None, 'comet_en_a_to_lrl': None,
            'raw_model_response_single_prompt': "[SP_MODEL_GEN_ERROR_RAW]",
            'error_message': None, 'runtime_sec': 0.0
        }
        sample_start_time = time.time()

        try:
            prompt = generate_single_prompt_qa_cotr(lrl_question, lrl_context, lang_code, use_few_shot)
            inputs_dict = {}
            model_name_for_check = getattr(model, 'name_or_path', "").lower()

            if "coherelabs/aya" in model_name_for_check:
                messages = [{"role": "user", "content": prompt}]
                try:
                    tokenized_output = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                        truncation=True, max_length=max_input_length
                    )
                    if isinstance(tokenized_output, dict) and 'input_ids' in tokenized_output:
                        inputs_dict = tokenized_output
                    elif isinstance(tokenized_output, torch.Tensor):
                        inputs_dict = {"input_ids": tokenized_output}
                    else:
                        logger.error(f"QA CoTR Single Prompt: apply_chat_template returned unexpected type for Aya. Type: {type(tokenized_output)}. Value: {str(tokenized_output)[:200]}")
                        raise ValueError(f"Aya tokenization error for SP: unexpected type {type(tokenized_output)}")
                except Exception as e_chat_template_qa:
                    logger.error(f"QA CoTR Single Prompt: Aya tokenization error for SP: {e_chat_template_qa}", exc_info=True)
                    raise ValueError(f"Aya tokenization error for SP: {e_chat_template_qa}")
            else:
                inputs_dict = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)

            if not inputs_dict or 'input_ids' not in inputs_dict or inputs_dict['input_ids'].numel() == 0:
                raise ValueError(f"Tokenization failed SP for model {model_name_for_check}")

            inputs = {k: v.to(model.device) for k, v in inputs_dict.items()}
            if 'input_ids' in inputs:
                inputs['input_ids'] = inputs['input_ids'].long() # Ensure it's a LongTensor
            if 'attention_mask' in inputs: # Ensure attention_mask is also on device
                inputs['attention_mask'] = inputs['attention_mask'].to(model.device)

            effective_do_sample = do_sample if temperature > 0.01 else False

            # Debugging token IDs before generation
            logger.debug(f"SP-{sample_id}: Before model.generate:")
            if 'input_ids' in inputs:
                logger.debug(f"  inputs['input_ids'] Shape: {inputs['input_ids'].shape}")
                logger.debug(f"  inputs['input_ids'] Dtype: {inputs['input_ids'].dtype}")
                logger.debug(f"  inputs['input_ids'] Sample Values: {inputs['input_ids'][0, :10]}") # Log first 10 values
            else:
                logger.debug("  inputs['input_ids'] is not present in inputs dict before generate.")

            pad_id_val = tokenizer.pad_token_id
            eos_id_val = tokenizer.eos_token_id

            logger.debug(f"  tokenizer.pad_token_id: {pad_id_val} (type: {type(pad_id_val)})")
            logger.debug(f"  tokenizer.eos_token_id: {eos_id_val} (type: {type(eos_id_val)})")

            effective_pad_id_for_gen = pad_id_val if pad_id_val is not None else eos_id_val
            effective_eos_id_for_gen = eos_id_val

            logger.debug(f"  Effective pad_token_id for generate: {effective_pad_id_for_gen} (type: {type(effective_pad_id_for_gen)})")
            logger.debug(f"  Effective eos_token_id for generate: {effective_eos_id_for_gen} (type: {type(effective_eos_id_for_gen)})")

            if effective_pad_id_for_gen is not None and not isinstance(effective_pad_id_for_gen, int):
                logger.error(f"SP-{sample_id}: CRITICAL - effective_pad_id_for_gen IS NOT AN INT! Value: {effective_pad_id_for_gen}, Type: {type(effective_pad_id_for_gen)}")
            
            if effective_eos_id_for_gen is not None and not isinstance(effective_eos_id_for_gen, int):
                logger.error(f"SP-{sample_id}: CRITICAL - effective_eos_id_for_gen IS NOT AN INT! Value: {effective_eos_id_for_gen}, Type: {type(effective_eos_id_for_gen)}")


            # Generate response
            output_sequences = model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty, do_sample=effective_do_sample,
                pad_token_id=model.config.pad_token_id,
                # eos_token_id=tokenizer.eos_token_id # Ensure EOS is handled if necessary by generate's defaults or specific model needs
            )
            raw_response = tokenizer.decode(output_sequences[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            current_result['raw_model_response_single_prompt'] = raw_response
            
            extracted_parts = extract_parts_from_single_prompt_qa_response(raw_response, lang_code)
            current_result.update(extracted_parts)

            if lang_code != 'en' and COMET_AVAILABLE and calculate_translation_quality:
                if lrl_question and extracted_parts["en_question_model"] and "[error]" not in extracted_parts["en_question_model"].lower():
                    comet_q_sp = calculate_translation_quality(sources=[lrl_question], predictions=[extracted_parts["en_question_model"]], references=[[lrl_question]])
                    current_result['comet_lrl_q_to_en'] = comet_q_sp['mean_score'] if isinstance(comet_q_sp, dict) else comet_q_sp
                if lrl_context and extracted_parts["en_context_model"] and "[error]" not in extracted_parts["en_context_model"].lower():
                    comet_c_sp = calculate_translation_quality(sources=[lrl_context], predictions=[extracted_parts["en_context_model"]], references=[[lrl_context]])
                    current_result['comet_lrl_c_to_en'] = comet_c_sp['mean_score'] if isinstance(comet_c_sp, dict) else comet_c_sp
                if extracted_parts["en_answer_model_intermediate"] and extracted_parts["lrl_answer_model_final"] and \
                   "[error]" not in extracted_parts["en_answer_model_intermediate"].lower() and \
                   "[error]" not in extracted_parts["lrl_answer_model_final"].lower():
                    comet_a_sp = calculate_translation_quality(sources=[extracted_parts["en_answer_model_intermediate"]], predictions=[extracted_parts["lrl_answer_model_final"]], references=[[extracted_parts["en_answer_model_intermediate"]]])
                    current_result['comet_en_a_to_lrl'] = comet_a_sp['mean_score'] if isinstance(comet_a_sp, dict) else comet_a_sp
        
        except Exception as e_sample_sp:
            logger.error(f"Error processing SP QA sample {sample_id} ({lang_code}): {e_sample_sp}", exc_info=True) # Ensure full traceback
            current_result['error_message'] = str(e_sample_sp)

        current_result['runtime_sec'] = time.time() - sample_start_time
        results_list.append(current_result)

    results_df = pd.DataFrame(results_list)
    logger.info(f"Finished Single-Prompt QA CoTR eval for {getattr(model, 'name_or_path', '?')} on {lang_code}. Runtime: {(time.time() - total_start_time):.2f}s")
    return results_df

# Note: calculate_qa_f1 and EM would be called by the runner script (run_qa_cotr.py)
# using the 'lrl_answer_model_final' and 'lrl_ground_truth_answers_list' columns.
# The normalization for F1 (normalize_answer_for_f1) is important for TyDiQA.
# The official TyDiQA evaluation script should be used for final scores if possible.