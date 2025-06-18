import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os
import re
from typing import Any, Dict, List, Tuple, Optional
import logging
import time
import json # Added for robust parsing if needed

# --- Global Logger ---
logger = logging.getLogger(__name__)

# Import COMET related utilities
from evaluation.cotr.translation_metrics import COMET_AVAILABLE
# Ensure calculate_comet_score is imported directly if used by that name, or aliased.
# from evaluation.cotr.translation_metrics import calculate_comet_score as calculate_translation_quality
from evaluation.cotr.translation_metrics import calculate_comet_score # Using direct name

# Define language names globally for access by multiple functions
LANG_NAMES = {
    "en": "English",
    "sw": "Swahili", # Corrected LRL code
    "ha": "Hausa",   # Corrected LRL code
    "am": "Amharic",
    "dz": "Dzongkha",
    "pcm": "Nigerian Pidgin",
    "yo": "Yoruba",
    "ma": "Marathi",
    "multi": "Multilingual",
    "te": "Telugu", # From QA, might be needed if unifying
    "pt": "Portuguese" # Added Portuguese
    # Add other languages from datasets as needed
}

# Define English sentiment labels (typically from the dataset, used as target for English classification)
ENGLISH_SENTIMENT_LABELS = ["positive", "negative", "neutral"]
EXPECTED_LABELS = ENGLISH_SENTIMENT_LABELS # For extraction consistency

# Define LRL translations of these English sentiment labels
# This is crucial for COMET score calculation of label back-translation and for single-prompt examples
SENTIMENT_LABELS_LRL = {
    "sw": {"positive": "chanya", "negative": "hasi", "neutral": "kati"}, # Was "sioegemea", "kati" is common
    "ha": {"positive": "tabbatacce", "negative": "korau", "neutral": "tsaka-tsaki"},
    "am": {"positive": "አዎንታዊ", "negative": "አሉታዊ", "neutral": "ገለልተኛ"},
    "yo": {"positive": "rere", "negative": "búburú", "neutral": "dídọ̀ọ̀dọ́"}, # Added accents for Yoruba
    "pcm": {"positive": "good", "negative": "bad", "neutral": "neutral"}, # Pidgin might use English-like terms
    "pt": {"positive": "positivo", "negative": "negativo", "neutral": "neutro"}, # Added Portuguese
    # For Telugu, if sentiment task is extended:
    # "te": {"positive": "సానుకూల", "negative": "ప్రతికూల", "neutral": "తటస్థ"}
}

# Define SENTIMENT_LABELS_EN and SENTIMENT_LABELS_EN_STR for import by runner script
SENTIMENT_LABELS_EN = ["positive", "negative", "neutral"]
SENTIMENT_LABELS_EN_STR = ", ".join(SENTIMENT_LABELS_EN)

# --- Added Utility Function ---
def _sanitize_for_prompt(text: str) -> str:
    """Basic sanitization for text included in prompts."""
    if not isinstance(text, str):
        text = str(text)
    # Escape backticks and triple quotes to prevent markdown/f-string issues
    text = text.replace('`', '\\`')
    text = text.replace("'''", "'\\''") # Escape existing triple quotes
    text = text.replace('"""', '\"\"\"')
    return text

def get_language_name(lang_code: str) -> str:
    """Helper to get full language name."""
    return LANG_NAMES.get(lang_code, lang_code.capitalize())

def initialize_model(model_name: str) -> tuple:
    """Initialize the model and tokenizer, with robust pad token handling."""
    logger.info(f"Initializing model {model_name} for Sentiment CoTR...")
    cache_path = "/work/bbd6522/cache_dir"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path
    )

    model_load_kwargs = {
        "trust_remote_code": True,
        "cache_dir": cache_path,
    }
    # Model-specific loading adjustments
    if "qwen" in model_name.lower():
        logger.info(f"Applying float16 for Qwen model: {model_name}")
        model_load_kwargs["torch_dtype"] = torch.float16
    elif "aya" in model_name.lower():
        logger.info(f"Using default dtype for Aya model: {model_name}")
        # Aya might not need explicit float16 and can be sensitive to it.

    if torch.cuda.is_available():
        logger.info(f"CUDA available. Attempting to load {model_name} on GPU.")
        # For models not using device_map="auto" by default (like some Qwen configurations)
        # or if we want explicit control.
        # model_load_kwargs["device_map"] = "auto" # Consider this if OOMs persist for large models.
    else:
        logger.info(f"CUDA not available. Loading {model_name} on CPU.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_load_kwargs
    )
    
    if torch.cuda.is_available() and not model_load_kwargs.get("device_map"):
        try:
            model = model.to("cuda")
            logger.info(f"Model {model_name} successfully moved to CUDA.")
        except Exception as e_to_cuda:
            logger.error(f"Error moving {model_name} to CUDA: {e_to_cuda}", exc_info=True)
            # Decide if this is critical or if CPU fallback is acceptable for the run.
            # For now, assume it's critical if CUDA was expected.
            raise
    
    # Robustly set pad_token and model.config.pad_token_id
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")
        else:
            logger.warning("No pad_token or eos_token found. Adding a new [PAD] token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            logger.info("Resized model token embeddings for the new [PAD] token.")

    if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.info(f"Aligned model.config.pad_token_id with tokenizer.pad_token_id ({tokenizer.pad_token_id})")
            
    logger.info(f"Model {model_name} initialized. Device: {next(model.parameters()).device}")
    return tokenizer, model

def generate_translation_prompt(text: str, source_lang: str, target_lang: str, is_label: bool = False) -> str:
    """Generates an English-instructed prompt for translation."""
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)
    safe_text = _sanitize_for_prompt(text) # Ensure _sanitize_for_prompt is defined

    if is_label:
        # Prompt for translating a single sentiment label (e.g., "positive")
        # Instruction must be in ENGLISH
        return f"""Translate the following English sentiment label into {target_name}:
English Sentiment Label: "{safe_text}"
Provide ONLY the {target_name} translation of this label. Do not add explanations or any other text.
{target_name} Sentiment Label:"""
    else:
        # Prompt for translating a block of text
        # Instruction must be in ENGLISH
        # Adding English few-shot examples for LRL to English text translation
        example_lrl_sw = "Nimefurahishwa sana na bidhaa hii."
        example_en_sw = "I am very pleased with this product."
        example_lrl_ha = "Wannan fim din ya kasance mai ban sha'awa."
        example_en_ha = "This film was very interesting."
        example_lrl_pt = "O serviço ao cliente foi péssimo."
        example_en_pt = "The customer service was terrible."

        few_shot_text_trans = ""
        if source_lang == 'sw' and target_lang == 'en':
            few_shot_text_trans = f"Example {source_name} Text: '{example_lrl_sw}'\\nExample English Translation: '{example_en_sw}'"
        elif source_lang == 'ha' and target_lang == 'en':
            few_shot_text_trans = f"Example {source_name} Text: '{example_lrl_ha}'\\nExample English Translation: '{example_en_ha}'"
        elif source_lang == 'pt' and target_lang == 'en':
            few_shot_text_trans = f"Example {source_name} Text: '{example_lrl_pt}'\\nExample English Translation: '{example_en_pt}'"
        # Add more elif for other LRL->EN pairs if needed

        return f"""Original Text ({source_name}):
'{safe_text}'

Instructions:
Translate the {source_name} text above to fluent and accurate English.
Provide ONLY the English translation. Do not add any introductory text, labels, or explanations.

{few_shot_text_trans}

English Translation:"""

def generate_sentiment_prompt_english(text_en: str, use_few_shot: bool = True) -> str:
    """Generates an English-instructed prompt for English sentiment analysis.
       Instructs the model to output ONE of the predefined English labels.
       Uses English few-shot examples.
    """
    safe_text_en = _sanitize_for_prompt(text_en)

    # Instruction must be in ENGLISH
    instruction = f"""Analyze the sentiment of the following English text.
Respond with ONLY ONE of these English labels: {SENTIMENT_LABELS_EN_STR}.
Do not add explanations or any other text.

English Text: "{safe_text_en}"

English Sentiment Label:"""

    few_shot_examples_text = f"""Here are some examples:

Example 1:
English Text: "This is a fantastic product, highly recommended!"
English Sentiment Label: positive

Example 2:
English Text: "I am extremely disappointed with the quality and service."
English Sentiment Label: negative

Example 3:
English Text: "The movie was okay, nothing special."
English Sentiment Label: neutral

---
"""
    if use_few_shot:
        return f"{few_shot_examples_text}\n{instruction}"
    else:
        return instruction

def clean_translation_response(raw_response: str, target_lang: str, is_label: bool = False) -> str:
    """Cleans the raw translation response from the model."""
    cleaned = raw_response.strip()

    # Remove common prefixes/suffixes models might add
    target_lang_name = get_language_name(target_lang)
    prefixes_to_remove = [
        f"{target_lang_name} Translation:",
        f"The {target_lang_name.lower()} translation is:",
        "Translation:",
        "Translated text:",
        "Here is the translation:"
    ]
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()

    # Remove leading/trailing quotes that models often add
    cleaned = cleaned.strip('\'"`')

    if is_label:
        # For labels, we expect a single word or a very short phrase.
        # Take the first significant part.
        parts = cleaned.split()
        if parts:
            cleaned = parts[0] # Take the first word
        else: # If split results in empty, it might have been just spaces or quotes
            cleaned = "" # Fallback to empty if nothing substantive remains
    
    # If the response still contains the prompt structure (e.g., "Original Text: ... Translation: ...")
    # try to isolate just the translation part. This is a heuristic.
    # This is less likely with the improved prompt structure but kept as a safeguard.
    if "Original Text (" in cleaned and target_lang_name + " Translation:" in cleaned:
        # Try to split by the target language header.
        # This regex looks for the header and captures everything after it.
        match = re.search(rf"{re.escape(target_lang_name)}\s*Translation:\s*(.*)", cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
            cleaned = cleaned.strip('\'"`') # Strip quotes again after extraction

    return cleaned

def extract_sentiment_label_cotr(output_text: str, for_lrl_label: bool = False, lang_code: Optional[str] = None) -> str:
    """
    Extracts a sentiment label from model output.
    If for_lrl_label=True, it tries to match known LRL labels first for the given lang_code.
    Otherwise (or as fallback), it tries to match known English labels.
    Adapted from sentiment_cotr_old.py for more robust extraction.
    """
    if not output_text or not isinstance(output_text, str):
        logger.warning(f"extract_sentiment_label_cotr received invalid output_text: {output_text}")
        return "unknown"

    text_to_process = output_text.lower().strip()
    text_to_process = text_to_process.split('\\n')[0].strip() # Take only first line
    text_to_process = text_to_process.rstrip('.!') # Remove trailing punctuation

    # Prefixes to remove (from various model outputs)
    prefixes_to_remove = [
        "english sentiment label:", "lrl sentiment label:", "sentiment label:",
        "the sentiment is", "sentiment:", "label:"
    ]
    if lang_code:
        lrl_name = get_language_name(lang_code)
        prefixes_to_remove.extend([
            f"{lrl_name.lower()} sentiment label:",
            f"the {lrl_name.lower()} sentiment is"
        ])

    for prefix in prefixes_to_remove:
        if text_to_process.startswith(prefix):
            text_to_process = text_to_process[len(prefix):].strip()
            break # Remove only the first matching prefix

    # 1. If for_lrl_label, try to match LRL labels first
    if for_lrl_label and lang_code:
        if lang_code in SENTIMENT_LABELS_LRL:
            lrl_map = SENTIMENT_LABELS_LRL[lang_code]
            for eng_label, lrl_label_val in lrl_map.items():
                if text_to_process == lrl_label_val.lower():
                    logger.debug(f"Extracted LRL label '{lrl_label_val}' for {lang_code}, mapped to English '{eng_label}' from '{output_text}'")
                    return eng_label # Return the English equivalent
    else:
            logger.warning(f"No LRL sentiment labels defined for lang_code: {lang_code} in SENTIMENT_LABELS_LRL")

    # 2. Try to match English labels directly
    # This is the primary target for English sentiment analysis step
    # or fallback if LRL extraction (above) didn't yield a result or wasn't requested.
        for label in ENGLISH_SENTIMENT_LABELS:
        if text_to_process == label: # Exact match after cleaning
            logger.debug(f"Extracted English label '{label}' directly from '{output_text}'")
                return label
        # Check for common variations if needed, e.g., "the sentiment is positive"
        # The prefix removal should handle many of these.

    # 3. Fallback: check for English labels as substrings (less precise)
    # This was in the old script, can be noisy but might catch some cases.
    # Consider if this is too aggressive or should be stricter.
    # For now, retaining similar logic to old script's keyword search.
    if "positive" in text_to_process or "good" in text_to_process :
        logger.debug(f"Extracted English label 'positive' via keyword from '{output_text}'")
            return "positive"
    if "negative" in text_to_process or "bad" in text_to_process :
        logger.debug(f"Extracted English label 'negative' via keyword from '{output_text}'")
            return "negative"
    if "neutral" in text_to_process : # "neutral" is less likely to have synonyms like good/bad
        logger.debug(f"Extracted English label 'neutral' via keyword from '{output_text}'")
            return "neutral"
        
    logger.warning(f"Could not extract standard sentiment from '{output_text}'. Cleaned: '{text_to_process}'. Defaulting to 'unknown'.")
    return "unknown" # Default if no clear label found

def translate_text(
    model: Any, tokenizer: Any, text_to_translate: str, source_lang: str, target_lang: str, 
    is_label: bool, model_name: str,
    # Unified generation parameters (caller provides these)
    temperature: float, top_p: float, top_k: int, 
    max_new_tokens: int, repetition_penalty: float, do_sample: bool,
    max_input_length: int = 2048 # Max length for the prompt tokenization
) -> Tuple[str, str, float]:
    """
    Translate text or a classification label. Uses unified generation parameters.
    Returns: (translated_text, raw_model_output, runtime_seconds)
    """
    start_time = time.time()
    raw_model_output_str = "[Error Initial]"
    final_translation = "[Translation Error]"

    if not text_to_translate or not text_to_translate.strip(): 
        runtime = time.time() - start_time
        logger.warning(f"translate_text received empty input for {source_lang}->{target_lang}.")
        return "[Empty input to translate]", "[N/A - Empty Input]", runtime
    
    prompt = generate_translation_prompt(text_to_translate, source_lang, target_lang, is_label=is_label)
    
    original_tokenizer_config = None # For the hack
    if hasattr(model, 'config'):
        original_tokenizer_config = getattr(tokenizer, 'config', None)
        tokenizer.config = model.config

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs_on_device = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs_on_device,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None, # Temp is irrelevant if not sampling
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        raw_model_output_str = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Clean the raw output
        cleaned_response = clean_translation_response(raw_model_output_str, target_lang, is_label)
        
        if is_label:
            # For labels, extract_sentiment_label_cotr can be used for LRLs to map to known LRL forms
            # or to clean up English labels.
            if target_lang != 'en':
                final_translation = extract_sentiment_label_cotr(cleaned_response, for_lrl_label=True, lang_code=target_lang)
                if final_translation == "unknown" or final_translation == cleaned_response.lower().strip('\'"()[]{}<>`'): # If extraction didn't map to a known LRL label
                    # It might be that the model outputted English, or a variant. Use cleaned_response.
                    final_translation = cleaned_response if cleaned_response else text_to_translate # Fallback to original if cleaning results in empty
                    logger.debug(f"LRL label translation for '{text_to_translate}' to '{target_lang}' resulted in '{final_translation}' after extraction fallback.")

            else: # Target is English label
                final_translation = extract_sentiment_label_cotr(cleaned_response, for_lrl_label=False)
            
            # Ensure label is not empty; fallback to original text if it becomes empty after cleaning/extraction
            if not final_translation.strip() and text_to_translate.strip():
                final_translation = text_to_translate 
                logger.warning(f"Label translation for '{text_to_translate}' became empty, falling back to original.")

        else: # Not a label, general text translation
            final_translation = cleaned_response
            if not final_translation.strip():
                final_translation = "[No translation generated]"
                logger.warning(f"Text translation for '{text_to_translate[:50]}...' resulted in empty string, using fallback placeholder.")

    except Exception as e:
        logger.error(f"Error during translate_text ({model_name}, {source_lang}->{target_lang}): {e}", exc_info=True)
        raw_model_output_str = f"[Exception: {str(e)}]"
        # final_translation remains "[Translation Error]"
    finally:
        if hasattr(model, 'config'): # Only if the hack was applied
            if original_tokenizer_config is not None:
                tokenizer.config = original_tokenizer_config
            elif hasattr(tokenizer, 'config'): # if it was added and wasn't there before
                del tokenizer.config

    runtime = time.time() - start_time
    return final_translation, raw_model_output_str, runtime

def process_sentiment_english(
    model: Any, tokenizer: Any, text_en: str, use_few_shot: bool, model_name: str, 
    temperature: float, top_p: float, top_k: int, 
    max_new_tokens: int, repetition_penalty: float, do_sample: bool,
    max_input_length: int = 2048 # Max length for the prompt tokenization
) -> Tuple[str, str, float]:
    """
    Perform sentiment classification in English using the model. Uses unified generation parameters.
    Returns: (predicted_english_label, raw_model_output, runtime_seconds)
    """
    start_time = time.time()
    raw_model_output_str = "[Error Initial]"
    predicted_english_label = "unknown" # Default to unknown

    if not text_en or text_en.strip() == "" or "[Translation Error]" in text_en or "[Empty input" in text_en:
        logger.warning(f"Skipping English sentiment classification for {model_name} due to invalid input: '{text_en[:100]}'.")
        runtime = time.time() - start_time
        return predicted_english_label, "[N/A - Invalid Input Text]", runtime

    prompt = generate_sentiment_prompt_english(text_en, use_few_shot)
    
    original_tokenizer_config = None # For the hack
    if hasattr(model, 'config'):
        original_tokenizer_config = getattr(tokenizer, 'config', None)
        tokenizer.config = model.config

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs_on_device = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs_on_device,
                max_new_tokens=max_new_tokens, # Labels are short
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        raw_model_output_str = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        predicted_english_label = extract_sentiment_label_cotr(raw_model_output_str, for_lrl_label=False)

    except Exception as e:
        logger.error(f"Error during process_sentiment_english ({model_name}): {e}", exc_info=True)
        raw_model_output_str = f"[Exception: {str(e)}]"
        # predicted_english_label remains "unknown"
    finally:
        if hasattr(model, 'config'): # Only if the hack was applied
            if original_tokenizer_config is not None:
                tokenizer.config = original_tokenizer_config
            elif hasattr(tokenizer, 'config'):
                del tokenizer.config
        
    runtime = time.time() - start_time
    return predicted_english_label, raw_model_output_str, runtime

def evaluate_sentiment_cotr_multi_prompt(
    model_name: str, model: Any, tokenizer: Any, samples_df: pd.DataFrame,
    lang_code: str, use_few_shot: bool,
    # Generation parameters for each step, passed as Dicts
    text_translation_params: Dict[str, Any],
    sentiment_classification_params: Dict[str, Any],
    label_translation_params: Dict[str, Any],
    max_input_length: int = 2048 # Added max_input_length
) -> pd.DataFrame:
    """
    Evaluate Sentiment CoTR MULTI-PROMPT approach.
    Metrics (accuracy, F1) are based on the intermediate predicted English label.
    COMET scores are calculated for all translation steps.
    Ground truth labels from dataset are assumed to be English.
    """
    results_list = []
    lrl_name = get_language_name(lang_code)

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Multi-Prompt Sentiment CoTR {lang_code}"):
        original_text_lrl = str(row['text'])
        ground_truth_english_label = str(row['label']).lower().strip() # Assumed English GT

        # For COMET score of EN label -> LRL label, we need the canonical LRL translation of the English GT
        ground_truth_lrl_label_for_comet = SENTIMENT_LABELS_LRL.get(lang_code, {}).get(ground_truth_english_label, ground_truth_english_label)

        # Initialize fields
        intermediate_en_text, raw_trans_text_out, rt_trans_text = "[Error]", "[Error]", 0.0
        predicted_en_label, raw_classify_out, rt_classify = "unknown", "[Error]", 0.0
        predicted_lrl_label_final, raw_trans_label_out, rt_trans_label = "[Error]", "[Error]", 0.0
        comet_lrl_text_to_en, comet_en_label_to_lrl = None, None

        try:
            # 1. Translate LRL text to English
            intermediate_en_text, raw_trans_text_out, rt_trans_text = translate_text(
                model, tokenizer, original_text_lrl, lang_code, "en", False, model_name,
                max_input_length=max_input_length, **text_translation_params
            )
            if COMET_AVAILABLE and original_text_lrl and intermediate_en_text not in ["[Error]", "[Empty input to translate]", "[No translation generated]"] and lang_code != 'en':
                try:
                    # For text, COMET is typically calculated source-to-prediction, no reference needed if model is DA
                    # If you have human gold translations, they should be used as references.
                    score = calculate_comet_score(predictions=[intermediate_en_text], sources=[original_text_lrl])
                    comet_lrl_text_to_en = score[0] if isinstance(score, list) and score else (score if isinstance(score, float) else None)
                except Exception as e_comet:
                    logger.warning(f"COMET LRL->EN text error (sample {idx}): {e_comet}")

            # 2. Classify English text
            predicted_en_label, raw_classify_out, rt_classify = process_sentiment_english(
                model, tokenizer, intermediate_en_text, use_few_shot, model_name,
                max_input_length=max_input_length, **sentiment_classification_params
            )

            # 3. Translate English sentiment label back to LRL (if not English input)
            if lang_code != "en" and predicted_en_label != "unknown":
                predicted_lrl_label_final, raw_trans_label_out, rt_trans_label = translate_text(
                    model, tokenizer, predicted_en_label, "en", lang_code, True, model_name,
                    max_input_length=max_input_length, **label_translation_params
                )
                if COMET_AVAILABLE and predicted_en_label != "unknown" and predicted_lrl_label_final not in ["[Error]", "[Empty input to translate]", "[No translation generated]"]:
                    # Source for back-translation is the predicted EN label.
                    # Reference is the canonical LRL translation of the *ground_truth_english_label*.
                    if ground_truth_lrl_label_for_comet != ground_truth_english_label: # Only if a known LRL translation exists for GT
                        try:
                            score = calculate_comet_score(
                            predictions=[predicted_lrl_label_final], 
                                sources=[predicted_en_label], # Source of this specific translation
                                references=[[ground_truth_lrl_label_for_comet]] # Gold standard LRL of the original GT English label
                            )
                            comet_en_label_to_lrl = score[0] if isinstance(score, list) and score else (score if isinstance(score, float) else None)
                        except Exception as e_comet:
                            logger.warning(f"COMET EN->LRL label error (sample {idx}): {e_comet}")
            elif lang_code == "en":
                predicted_lrl_label_final = predicted_en_label # No back-translation
                raw_trans_label_out = "[N/A for EN]"
            else: # predicted_en_label was "unknown"
                 predicted_lrl_label_final = "unknown" # or map to LRL unknown if defined
                 raw_trans_label_out = "[Skipped - EN label was unknown]"


        except Exception as e_outer:
            logger.error(f"Outer error in multi-prompt sample {idx}, lang {lang_code}: {e_outer}", exc_info=True)
            # Ensure all variables are assigned error strings if not already.

        results_list.append({
            'original_text_lrl': original_text_lrl,
            'ground_truth_english_label': ground_truth_english_label,
            'intermediate_en_text': intermediate_en_text,
            'predicted_en_label': predicted_en_label, # This is the label used for accuracy/F1
            'predicted_lrl_label_final': predicted_lrl_label_final,
            'comet_lrl_text_to_en': comet_lrl_text_to_en,
            'comet_en_label_to_lrl': comet_en_label_to_lrl,
            'raw_translation_to_en_output': raw_trans_text_out,
            'raw_predicted_en_label_output': raw_classify_out,
            'raw_translation_to_lrl_output': raw_trans_label_out,
            'runtime_translation_to_en_seconds': rt_trans_text,
            'runtime_classification_en_seconds': rt_classify,
            'runtime_translation_to_lrl_seconds': rt_trans_label,
            'language': lang_code,
            'pipeline': 'multi_prompt',
            'shot_type': 'few-shot' if use_few_shot else 'zero-shot'
        })
    return pd.DataFrame(results_list)

# --- Revised Single Prompt Generation ---
def generate_single_prompt_sentiment_cotr(lrl_text: str, lang_code: str, use_few_shot: bool = True) -> str:
    """
    Generates a single comprehensive prompt for the Sentiment CoTR pipeline.
    All instructions are in ENGLISH.
    The model is instructed to:
    1. Translate LRL text to English.
    2. Classify sentiment of the English translation (outputting an English label).
    3. Translate the English sentiment label back to the LRL.
    Few-shot examples (if used) are in English and demonstrate this 3-step process.
    """
    lrl_name = get_language_name(lang_code)
    original_lrl_text_sanitized = _sanitize_for_prompt(lrl_text)

    # Core instructions - ALL IN ENGLISH
    prompt_instructions_template = f"""You are an expert multilingual AI assistant. Your task is to perform sentiment analysis on a given {lrl_name} text by following these three steps precisely:

1.  **Translate to English**: Translate the original '{lrl_name} Text' accurately into English.
    Begin your output for this step with the exact label "English Translation:" followed immediately by the translation.

2.  **Perform English Sentiment Analysis**: Analyze the 'English Translation' from Step 1. Determine if its sentiment is positive, negative, or neutral.
    Begin your output for this step with the exact label "English Sentiment Label:" followed immediately by ONLY ONE of these English labels: {SENTIMENT_LABELS_EN_STR}. Do not add explanations.

3.  **Translate Sentiment Label to {lrl_name}**: Translate the 'English Sentiment Label' from Step 2 accurately into {lrl_name}.
    Begin your output for this step with the exact label "{lrl_name} Sentiment Label:" followed immediately by the {lrl_name} translation of the sentiment label.

Your entire response must follow this three-step structure, with each step clearly delineated by its label."""

    few_shot_section = ""
    if use_few_shot:
        # Few-shot examples are in English text, demonstrating the process.
        # LRL-specific parts are the input LRL text and the final LRL label.
        # The intermediate English translation and English label are shown.

        # Define example variables based on lang_code - ALL EXAMPLES ARE ENGLISH-DRIVEN
        # These examples show the *process* using English as the CoT language.
        # The LRL text and LRL label are specific to the lang_code for the example.
        example_lrl_text_fs = ""
        example_eng_translation_fs = ""
        example_eng_label_fs = "" # This will always be one of "positive", "negative", "neutral"
        example_lrl_label_fs = "" # This will be the LRL translation of example_eng_label_fs
        lrl_name_fs_display = lrl_name # Used in the example header, e.g., "Swahili Text"

        if lang_code == 'sw':
            example_lrl_text_fs = "Huduma hii ni nzuri sana."
            example_eng_translation_fs = "This service is very good."
            example_eng_label_fs = "positive"
            example_lrl_label_fs = SENTIMENT_LABELS_LRL.get('sw', {}).get('positive', 'chanya')
        elif lang_code == 'ha':
            example_lrl_text_fs = "Wannan fim din ya kasance mai ban sha'awa kwarai da gaske."
            example_eng_translation_fs = "This film was very interesting indeed."
            example_eng_label_fs = "positive"
            example_lrl_label_fs = SENTIMENT_LABELS_LRL.get('ha', {}).get('positive', 'tabbatacce')
        elif lang_code == 'pt':
            example_lrl_text_fs = "Estou muito insatisfeito com a compra."
            example_eng_translation_fs = "I am very dissatisfied with the purchase."
            example_eng_label_fs = "negative"
            example_lrl_label_fs = SENTIMENT_LABELS_LRL.get('pt', {}).get('negative', 'negativo')
        else: # Fallback generic English example if lang_code not specified for examples
            example_lrl_text_fs = f"A sample text in {lrl_name} that expresses a clear sentiment."
            example_eng_translation_fs = "A sample text in English that expresses a clear sentiment (this is the translation)."
            example_eng_label_fs = "positive" # Default example label
            # Try to get LRL equivalent, or use a placeholder
            example_lrl_label_fs = SENTIMENT_LABELS_LRL.get(lang_code, {}).get(example_eng_label_fs, f"{example_eng_label_fs}_in_{lrl_name}")

        few_shot_section = f"""

Here is an example of how to perform this task:

--- Example Input ({lrl_name_fs_display} Text) ---
{_sanitize_for_prompt(example_lrl_text_fs)}

--- Example Output ---
English Translation:
{_sanitize_for_prompt(example_eng_translation_fs)}

English Sentiment Label:
{example_eng_label_fs}

{lrl_name_fs_display} Sentiment Label:
{_sanitize_for_prompt(example_lrl_label_fs)}
--- End Example ---
"""

    task_section = f"""

Now, complete the task for the following input:

--- Your Task ---
{lrl_name} Text:
```
{original_lrl_text_sanitized}
```

Your Response (following the 3 steps precisely):
"""

    final_prompt = prompt_instructions_template
    if use_few_shot:
        final_prompt += few_shot_section
    final_prompt += task_section
    
    # Log only a portion of the prompt for brevity
    logger.debug(f"Generated Single Sentiment CoTR Prompt for {lang_code} (use_few_shot={use_few_shot}):\n{final_prompt[:1000]}...")
    return final_prompt

# --- Revised Extraction for Single Prompt ---
def extract_sentiment_intermediates_from_single_prompt_response(response_text: str, lrl_name_for_extraction: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extracts translated English text, English sentiment label, and LRL sentiment label
    from the structured output of a single CoT prompt.
    Returns: (eng_translation, eng_label, lrl_label)
    """
    eng_translation_out: Optional[str] = None
    eng_label_out: Optional[str] = None
    lrl_label_out: Optional[str] = None

    # Regex patterns (case-insensitive, dotall for multiline content)
    # Ensure labels are specific and the capture group is non-greedy (.*?)
    # Lookaheads ensure we capture up to the next expected label or end of string.
    eng_translation_pattern = re.compile(
        r"English Translation:\s*(.*?)(?=English Sentiment Label:|{lrl_name_for_extraction} Sentiment Label:|$)", 
        re.IGNORECASE | re.DOTALL
    )
    eng_label_pattern = re.compile(
        r"English Sentiment Label:\s*(.*?)(?={lrl_name_for_extraction} Sentiment Label:|$)", 
        re.IGNORECASE | re.DOTALL
    )
    lrl_label_pattern = re.compile(
        r"{lrl_name_for_extraction} Sentiment Label:\s*(.*?)(?=$)", 
        re.IGNORECASE | re.DOTALL
    )

    match_eng_text = eng_translation_pattern.search(response_text)
    if match_eng_text:
        eng_translation_out = match_eng_text.group(1).strip()
        # Further clean potential model conversational prefixes if any remain
        common_prefixes = ["Sure, here is the translation:", "Here's the translation:"]
        for prefix in common_prefixes:
            if eng_translation_out.lower().startswith(prefix.lower()):
                eng_translation_out = eng_translation_out[len(prefix):].strip()
        eng_translation_out = eng_translation_out.strip('`') # Remove backticks if model wraps translation in them

    match_eng_label = eng_label_pattern.search(response_text)
    if match_eng_label:
        extracted_eng_label_raw = match_eng_label.group(1).strip()
        # The extract_sentiment_label_cotr function can robustly find the label
        eng_label_out = extract_sentiment_label_cotr(extracted_eng_label_raw, for_lrl_label=False)
        if eng_label_out == "unknown" and extracted_eng_label_raw: # Log if specific extraction failed but there was text
            logger.debug(f"extract_sentiment_label_cotr returned 'unknown' for eng_label part: '{extracted_eng_label_raw}'")


    match_lrl_label = lrl_label_pattern.search(response_text)
    if match_lrl_label:
        extracted_lrl_label_raw = match_lrl_label.group(1).strip()
        # Use extract_sentiment_label_cotr, but indicate it's for an LRL label if lang_code available
        # For single prompt, we don't have lang_code here, so we rely on it being one of the known LRL forms
        # or it being an English label if back-translation failed.
        # The goal here is to get the *English equivalent* of the LRL label the model produced.
        # We need lang_code for robust LRL label extraction.
        # For now, assume extract_sentiment_label_cotr will try its best.
        # This part might need refinement if LRL labels are diverse.
        # A temporary approach: first try direct LRL match if possible, then English.
        temp_lrl_name_lower = lrl_name_for_extraction.lower()
        found_lrl_direct = False
        if temp_lrl_name_lower in SENTIMENT_LABELS_LRL: # e.g. 'swahili' -> 'sw'
            lang_code_for_lrl_extraction = [lc for lc, name in LANG_NAMES.items() if name.lower() == temp_lrl_name_lower]
            if lang_code_for_lrl_extraction:
                lrl_label_out = extract_sentiment_label_cotr(extracted_lrl_label_raw, for_lrl_label=True, lang_code=lang_code_for_lrl_extraction[0])
                if lrl_label_out != "unknown":
                    found_lrl_direct = True
        
        if not found_lrl_direct: # Fallback to treating it as an English label or general extraction
            lrl_label_out = extract_sentiment_label_cotr(extracted_lrl_label_raw, for_lrl_label=False)

        if lrl_label_out == "unknown" and extracted_lrl_label_raw :
             logger.debug(f"extract_sentiment_label_cotr returned 'unknown' for lrl_label part: '{extracted_lrl_label_raw}'")


    if not eng_translation_out and not eng_label_out and not lrl_label_out:
        logger.warning(f"Could not extract any parts from single prompt sentiment output for {lrl_name_for_extraction}. Raw output: {response_text[:300]}...")
        # Add a simple fallback for the English label if others failed
        if not eng_label_out:
            eng_label_candidate = extract_sentiment_label_cotr(response_text, for_lrl_label=False)
            if eng_label_candidate != "unknown":
                eng_label_out = eng_label_candidate
                logger.info(f"Fallback: Extracted English label '{eng_label_out}' from full response.")
    
    logger.debug(f"SP Sentiment Extraction for {lrl_name_for_extraction}: EN Text='{str(eng_translation_out)[:50]}...', EN Label='{eng_label_out}', LRL Label='{lrl_label_out}' (parsed as EN equivalent)")
    return eng_translation_out, eng_label_out, lrl_label_out


def evaluate_sentiment_cotr_single_prompt(
    model_name: str, model: Any, tokenizer: Any, samples_df: pd.DataFrame,
    lang_code: str, use_few_shot: bool,
    # Unified generation parameters for the entire single prompt chain
    temperature: float, top_p: float, top_k: int, 
    max_new_tokens: int, repetition_penalty: float, do_sample: bool,
    max_input_length: int = 2048 # Added max_input_length
) -> pd.DataFrame:
    """
    Evaluate Sentiment CoTR SINGLE-PROMPT approach.
    Ground truth labels from dataset are assumed to be English.
    Metrics (accuracy, F1) are based on the intermediate predicted English label extracted from the CoT.
    """
    results_list = []
    lrl_name = get_language_name(lang_code) # Full language name like "Swahili"

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Single-Prompt Sentiment CoTR {lang_code}"):
        original_text_lrl = str(row['text'])
        ground_truth_english_label = str(row['label']).lower().strip() # Assumed English GT
        
        # For COMET score of EN label -> LRL label (back-translation quality assessment)
        # Reference is the canonical LRL translation of the *ground_truth_english_label*.
        ground_truth_lrl_label_for_comet_ref = SENTIMENT_LABELS_LRL.get(lang_code, {}).get(ground_truth_english_label, ground_truth_english_label)

        # Initialize fields for this sample
        comet_lrl_text_to_en, comet_en_label_to_lrl = None, None
        intermediate_en_text, intermediate_en_label = "[Error Extr EN Txt]", "unknown"
        predicted_lrl_label_final_raw = "[Error Extr LRL Lbl]"
        raw_model_response_str = "[Error Initial]"
        rt_sample = 0.0

        try:
            start_time_sample = time.time()
            prompt = generate_single_prompt_sentiment_cotr(original_text_lrl, lang_code, use_few_shot)
            
            original_tokenizer_config = None # For the hack
            if hasattr(model, 'config'):
                original_tokenizer_config = getattr(tokenizer, 'config', None)
                tokenizer.config = model.config

            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
                inputs_on_device = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                            **inputs_on_device,
                            max_new_tokens=max_new_tokens, # Max tokens for the entire multi-step CoT output
                            temperature=temperature if do_sample else None,
                            top_p=top_p if do_sample else None,
                            top_k=top_k if do_sample else None,
                            repetition_penalty=repetition_penalty,
                        do_sample=do_sample,
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                        )
                    raw_model_response_str = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            finally: # Ensure tokenizer config is restored
                if hasattr(model, 'config'):
                    if original_tokenizer_config is not None: tokenizer.config = original_tokenizer_config
                    elif hasattr(tokenizer, 'config'): del tokenizer.config

            rt_sample = time.time() - start_time_sample

            # Extract intermediates using the more robust parser
            intermediate_en_text_ext, intermediate_en_label_ext, predicted_lrl_label_final_raw_ext = \
                extract_sentiment_intermediates_from_single_prompt_response(raw_model_response_str, lrl_name)

            # Assign extracted values or fallbacks
            intermediate_en_text = intermediate_en_text_ext if intermediate_en_text_ext is not None else "[Extraction Error EN Text]"
            intermediate_en_label = intermediate_en_label_ext if intermediate_en_label_ext is not None and intermediate_en_label_ext in EXPECTED_LABELS else "unknown"
            predicted_lrl_label_final_raw = predicted_lrl_label_final_raw_ext if predicted_lrl_label_final_raw_ext is not None else "[Extraction Error LRL Label]"

            # COMET score for LRL Text -> EN Text (forward translation by model)
            # This requires gold standard English translations as references.
            # If not available, this COMET score might measure fluency/coherence rather than pure accuracy against a gold ref.
            # For now, we calculate it if COMET is available and the necessary texts were extracted.
            if COMET_AVAILABLE and original_text_lrl and intermediate_en_text not in ["[Error Extr EN Txt]", "[Extraction Error EN Text]"] and lang_code != 'en':
                try:
                    # Assuming no gold EN refs for LRL text, COMET here is source-prediction DA score.
                    score = calculate_comet_score(predictions=[intermediate_en_text], sources=[original_text_lrl])
                    comet_lrl_text_to_en = score[0] if isinstance(score, list) and score else (score if isinstance(score, float) else None)
                except Exception as e_comet:
                    logger.warning(f"COMET LRL->EN text (single-prompt, sample {idx}) error: {e_comet}")
            
            # COMET score for EN Label -> LRL Label (model's back-translation quality)
            if COMET_AVAILABLE and intermediate_en_label != "unknown" and \
               predicted_lrl_label_final_raw not in ["[Error Extr LRL Lbl]", "[Extraction Error LRL Label]"] and lang_code != 'en':
                    # Source is the model's predicted English label (intermediate_en_label)
                    # Prediction is the model's LRL translation of that (predicted_lrl_label_final_raw)
                    # Reference is the canonical LRL translation of the *model's predicted English label*
                # This assesses how well the model translates ITS OWN English prediction to LRL.
                reference_lrl_for_models_en_pred = SENTIMENT_LABELS_LRL.get(lang_code, {}).get(intermediate_en_label, intermediate_en_label)
                if reference_lrl_for_models_en_pred != intermediate_en_label: # Check if a known LRL form exists for the model's EN pred
                    try:
                        score = calculate_comet_score(
                            predictions=[predicted_lrl_label_final_raw], 
                            sources=[intermediate_en_label], # Source of this translation step
                            references=[[reference_lrl_for_models_en_pred]] # Canonical LRL of model's EN pred
                        )
                        comet_en_label_to_lrl = score[0] if isinstance(score, list) and score else (score if isinstance(score, float) else None)
                    except Exception as e_comet:
                        logger.warning(f"COMET EN Label->LRL (single-prompt, sample {idx}) error: {e_comet}")

        except Exception as e_outer_sp:
            logger.error(f"Outer error in single-prompt sample {idx}, lang {lang_code}: {e_outer_sp}", exc_info=True)
            rt_sample = time.time() - (start_time_sample if 'start_time_sample' in locals() else time.time())
            # Error strings already initialized for intermediate_en_text, intermediate_en_label, etc.

        results_list.append({
            'original_text_lrl': original_text_lrl,
            'ground_truth_english_label': ground_truth_english_label,
            'intermediate_en_text': intermediate_en_text,
            'intermediate_en_label': intermediate_en_label, # This is the EN label extracted from CoT, used for accuracy/F1
            'predicted_lrl_label_final_raw': predicted_lrl_label_final_raw, # The LRL label from CoT
            'comet_lrl_text_to_en': comet_lrl_text_to_en,
            'comet_en_label_to_lrl': comet_en_label_to_lrl,
            'raw_model_response': raw_model_response_str,
            'runtime_seconds_sample': rt_sample,
            'language': lang_code,
            'pipeline': 'single_prompt',
            'shot_type': 'few-shot' if use_few_shot else 'zero-shot'
        })
    return pd.DataFrame(results_list)

# Placeholder for main if direct execution is needed (typically run via run_sentiment_cotr.py)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Testing Sentiment CoTR direct execution (prompts, parsing, etc.).")
    
    # Example Test for generate_single_prompt_sentiment_cotr
    sample_sw_text = "Huduma hii ni mbaya sana, sitarudi tena!"
    prompt_sw_single_fs = generate_single_prompt_sentiment_cotr(sample_sw_text, 'sw', use_few_shot=True)
    logger.debug(f"Generated Single Prompt (Swahili, Few-Shot):\n{prompt_sw_single_fs}")

    sample_ha_text = "Ina matukar farin ciki da wannan kyauta."
    prompt_ha_single_zs = generate_single_prompt_sentiment_cotr(sample_ha_text, 'ha', use_few_shot=False)
    logger.debug(f"Generated Single Prompt (Hausa, Zero-Shot):\n{prompt_ha_single_zs}")

    # Example Test for extract_sentiment_intermediates_from_single_prompt_response
    mock_response_sw_full = """English Text: This service is very bad, I will not return!
English Sentiment Label: negative
Final Sentiment Label (Swahili): hasi"""
    en_text, en_label, lrl_label = extract_sentiment_intermediates_from_single_prompt_response(mock_response_sw_full, "Swahili")
    logger.debug(f"Extracted Intermediates (Swahili): EN Text='{en_text}', EN Label='{en_label}', LRL Label='{lrl_label}'")

    mock_response_ha_partial = """English Text: I am very happy with this gift.
English Sentiment Label: positive
""" # Missing LRL part
    en_text_ha, en_label_ha, lrl_label_ha = extract_sentiment_intermediates_from_single_prompt_response(mock_response_ha_partial, "Hausa")
    logger.debug(f"Extracted Intermediates (Hausa, Partial): EN Text='{en_text_ha}', EN Label='{en_label_ha}', LRL Label='{lrl_label_ha}'")
    
    logger.info("Direct execution testing complete.") 