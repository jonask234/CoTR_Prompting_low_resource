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
    "te": "Telugu" # From QA, might be needed if unifying
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
    # For Telugu, if sentiment task is extended:
    # "te": {"positive": "సానుకూల", "negative": "ప్రతికూల", "neutral": "తటస్థ"}
}

# Define SENTIMENT_LABELS_EN and SENTIMENT_LABELS_EN_STR for import by runner script
SENTIMENT_LABELS_EN = ["positive", "negative", "neutral"]
SENTIMENT_LABELS_EN_STR = ", ".join(SENTIMENT_LABELS_EN)

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
    """Generate a prompt for translation (text or label) with structured format. All instructions in English."""
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)

    # Escape the text to be included in the f-string
    # Basic escaping for single quotes; f-strings with triple quotes are safer.
    text_escaped = text.replace("'", "\\'") # Escape single quotes

    if is_label:
        # Specific prompt for translating a sentiment label. Target output is a single LRL word.
        return f"""Translate the following English sentiment label to {target_name}: '{text_escaped}'
Provide ONLY the single translated word for the sentiment label in {target_name}.
{target_name} Translation:"""
    else:
        # General text translation prompt, ensuring clarity for the model.
        return f"""Translate the following {source_name} text to {target_name}.
Preserve the original meaning and sentiment accurately.
Provide only the direct translation without any explanations, conversational phrases, or markdown.

Original Text ({source_name}):
'''{text}'''

{target_name} Translation:""" # The model should continue from here

def generate_sentiment_prompt_english(text_en: str, use_few_shot: bool = True) -> str:
    """Generate a prompt for English sentiment classification. All instructions in English."""
    # Escape the English text
    text_en_escaped = text_en.replace("'", "\\'")

    # Clear instructions and output format specification
    base_instruction = f"""Analyze the sentiment of the following English text.
Respond with ONLY one of these English labels: positive, negative, or neutral.
Do NOT add any explanations or other text.

Text:
'''{text_en_escaped}'''
"""
    
    examples_en = """
Examples:
Text:
'''This movie was fantastic, I loved it!'''
Sentiment: positive

Text:
'''I am not happy with the service provided.'''
Sentiment: negative

Text:
'''The meeting is scheduled for 3 PM.'''
Sentiment: neutral
""" # End of examples
    
    if use_few_shot:
        prompt = f"{base_instruction}\n{examples_en}\nSentiment:"
    else:
        prompt = f"{base_instruction}\nSentiment:"
        
    return prompt

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
    Extracts sentiment label (English or LRL) from model output.
    If for_lrl_label is True, it uses LRL keywords for the specified lang_code.
    Otherwise, it extracts standard English labels.
    """
    text_lower = output_text.lower().strip()
    
    # Remove common prefixes models might add, including LRL ones
    common_prefixes = [
        "sentiment:", "the sentiment is", "this text is", "i think the sentiment is", 
        "hisia:", "ra'ayi:", "jibu:", "answer:", "label:", "sentimento:",
        "translation:", "tafsiri:", "fassara:", "traducción:" 
    ]
    if lang_code and lang_code in LANG_NAMES: # Add language-specific prefixes
        lang_name_cap = LANG_NAMES[lang_code]
        common_prefixes.extend([f"{lang_name_cap.lower()} translation:", f"{lang_name_cap.lower()} sentiment:"])

    for prefix in common_prefixes:
        if text_lower.startswith(prefix):
            text_lower = text_lower[len(prefix):].strip()

    # Remove surrounding quotes or brackets that models often add
    text_lower = text_lower.strip('\'"()[]{}<>')

    if for_lrl_label and lang_code:
        # LRL Label Extraction (using defined LRL labels for mapping)
        lrl_labels_for_lang = SENTIMENT_LABELS_LRL.get(lang_code, {})
        # Check for exact LRL label match (e.g., "chanya", "tabbatacce")
        for eng_label, lrl_val in lrl_labels_for_lang.items():
            if text_lower == lrl_val.lower():
                return lrl_val # Return the original LRL form
        # Check if text_lower starts with an LRL label
        for eng_label, lrl_val in lrl_labels_for_lang.items():
            if text_lower.startswith(lrl_val.lower()):
                return lrl_val
        # If no exact LRL match, this function might not be the right place to map back to English.
        # The caller should handle mapping if needed. For now, return "unknown" or the cleaned text.
        logger.debug(f"Could not extract known LRL sentiment label for '{lang_code}' from '{text_lower}'. It might be an unknown variant or English.")
        # Attempt to extract English label as a fallback if LRL is not clearly identified.
        # This helps if the model fails to translate the label to LRL and outputs English.
        for label in ENGLISH_SENTIMENT_LABELS:
            if text_lower == label: return label # Return English label if found
            if text_lower.startswith(label): return label
        return text_lower # Return the cleaned text as is if no known LRL/EN label found
    else:
        # English Label Extraction
        for label in ENGLISH_SENTIMENT_LABELS:
            if text_lower == label:
                return label
        for label in ENGLISH_SENTIMENT_LABELS: # Check startswith after exact matches
            if text_lower.startswith(label): # e.g., "positive." or "positive sentiment"
                # Ensure it's not "positive " followed by "negative" etc.
                # A simple way is to check if the found label is the only word or followed by non-alpha
                parts = text_lower.split(None, 1) # Split into first word and rest
                if parts[0] == label:
                    if len(parts) == 1 or not parts[1][0].isalpha(): # It is the label or followed by punctuation
                        return label
        
        # Fallback keyword check for English (less reliable, use with caution)
        # This is simplified; a more advanced keyword approach might be needed if models are inconsistent.
        if "positive" in text_lower or "good" in text_lower or "happy" in text_lower :
        return "positive"
        if "negative" in text_lower or "bad" in text_lower or "sad" in text_lower :
        return "negative"
        # Neutral is harder with keywords, often default.
        # If the text is very short and none of the above, it might be neutral or unknown.
        if len(text_lower.split()) <= 2 and "neutral" in text_lower: # Check for "neutral" in short responses
        return "neutral"
        
    logger.debug(f"Could not extract standard English sentiment from '{output_text}'. Cleaned: '{text_lower}'. Defaulting to 'unknown'.")
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

def generate_single_prompt_sentiment_cotr(lrl_text: str, lang_code: str, use_few_shot: bool = True) -> str:
    """Generate a single prompt for the entire CoTR Sentiment pipeline. Instructions in English."""
    lrl_name = get_language_name(lang_code)
    
    lrl_labels_str_for_prompt = ", ".join([f"{k} ({v})" for k, v in SENTIMENT_LABELS_LRL.get(lang_code, {}).items()])
    
    # Escape the text to be included in the f-string, robustly handling various quote types.
    lrl_text_escaped = str(lrl_text).replace("'", "\\'").replace("\\\"\\\"\\\"", "\\\\\\\"\\\"\\\"").replace("'''", "\\'\\'\\'")

    instructions_core = (
        f"You are an advanced AI assistant performing sentiment analysis via Chain-of-Thought.\\n"
        f"Given a text in {lrl_name}, follow these steps precisely:\\n\\n"
        f"1.  **Translate to English**: Accurately translate the original {lrl_name} text into English.\\n"
        f"    Output this as:\\n"
        f"    English Text: [Your English Translation]\\n\\n"
        f"2.  **Classify English Sentiment**: Analyze the English text (from Step 1) and determine its sentiment.\\n"
        f"    The ONLY allowed English sentiment labels are: {EXPECTED_LABELS_STR}.\\n"
        f"    Output this as:\\n"
        f"    English Sentiment Label: [One of: {EXPECTED_LABELS_STR}]\\n\\n"
        f"3.  **Translate Sentiment to {lrl_name}**: Translate the English sentiment label (from Step 2) into {lrl_name}.\\n"
        f"    Use the most natural and common single-word translation for the sentiment in {lrl_name}.\\n"
        f"    (Known {lrl_name} labels are: {lrl_labels_str_for_prompt}, but translate naturally even if the English label isn't one that has a direct known mapping shown here, using the closest equivalent sentiment concept in {lrl_name}).\\n"
        f"    Output this as:\\n"
        f"    Final Sentiment Label ({lrl_name}): [The {lrl_name} sentiment label]\\n\\n"
        f"Ensure your response clearly includes each of these labeled steps.\\n"
        f"The \\\"Final Sentiment Label ({lrl_name})\\\" is the ultimate result.\\n\\n"
        f"Original {lrl_name} Text:\\n"
        f"'''{lrl_text_escaped}'''"
    )

    few_shot_examples_str = ""
    if use_few_shot:
        # Few-shot examples must demonstrate the full CoT process.
        # Example 1: Swahili Positive
        ex1_lrl_text_sw = "Kitabu hiki ni kizuri sana na kinanifurahisha!"
        ex1_en_translation_sw = "This book is very good and makes me happy!"
        ex1_en_sentiment_sw = "positive"
        ex1_lrl_sentiment_sw = SENTIMENT_LABELS_LRL.get("sw", {}).get("positive", "chanya") # "chanya"

        # Example 2: Hausa Negative
        ex2_lrl_text_ha = "Wannan labarin ya bata min rai matuka."
        ex2_en_translation_ha = "This news has greatly upset me."
        ex2_en_sentiment_ha = "negative"
        ex2_lrl_sentiment_ha = SENTIMENT_LABELS_LRL.get("ha", {}).get("negative", "korau") # "korau"
        
        current_ex_lrl_text = ex1_lrl_text_sw
        current_ex_en_translation = ex1_en_translation_sw
        current_ex_en_sentiment = ex1_en_sentiment_sw
        current_ex_lrl_sentiment_final = ex1_lrl_sentiment_sw
        example_lrl_name = get_language_name("sw")

        if lang_code == "ha":
            current_ex_lrl_text = ex2_lrl_text_ha
            current_ex_en_translation = ex2_en_translation_ha
            current_ex_en_sentiment = ex2_en_sentiment_ha
            current_ex_lrl_sentiment_final = ex2_lrl_sentiment_ha
            example_lrl_name = get_language_name("ha")
        elif lang_code != "sw": 
            example_lrl_name = lrl_name 
            current_ex_lrl_text = f"Some example text in {example_lrl_name} that is positive."
            current_ex_en_translation = "Some example English text that is positive."
            current_ex_en_sentiment = "positive"
            current_ex_lrl_sentiment_final = SENTIMENT_LABELS_LRL.get(lang_code, {}).get("positive", f"positive_in_{lang_code}")

        # Safely escape content for the example string
        escaped_current_ex_lrl_text = str(current_ex_lrl_text).replace("'", "\\'")
        escaped_current_ex_en_translation = str(current_ex_en_translation).replace("'", "\\'")

        few_shot_examples_str = (
            f"\\nExamples:\\n\\n"
            f"Original {example_lrl_name} Text:\\n"
            f"'''{escaped_current_ex_lrl_text}'''\\n"
            f"English Text: {escaped_current_ex_en_translation}\\n"
            f"English Sentiment Label: {current_ex_en_sentiment}\\n"
            f"Final Sentiment Label ({example_lrl_name}): {current_ex_lrl_sentiment_final}"
        )

    # Combine instructions and examples
    # The model is expected to start its response with "English Text:"
    prompt = f"{instructions_core}\\n{few_shot_examples_str}\\n\\nEnglish Text:" # Cue for the model to start step 1
    return prompt

def extract_sentiment_intermediates_from_single_prompt_response(response_text: str, lrl_name_for_extraction: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extracts English translation, English sentiment, and LRL sentiment
    from a structured single-prompt response. More robust parsing.
    Args:
        response_text: The full raw response from the model.
        lrl_name_for_extraction: The name of the Low-Resource Language (e.g., "Swahili") used in prompt labels.
    Returns: (intermediate_en_text, intermediate_en_label, predicted_lrl_label)
    """
    intermediate_en_text, intermediate_en_label, predicted_lrl_label = None, None, None

    # Regex patterns for each step, allowing for variations in model output
    # Pattern for English Text:
    # Looks for "English Text:" and captures content until "English Sentiment Label:" or other subsequent headers.
    en_text_match = re.search(
        r"English Text:\s*(.*?)(?=\n\s*(?:English Sentiment Label:|Final Sentiment Label \({lrl_name_for_extraction}\):|$))",
        response_text, re.DOTALL | re.IGNORECASE
    )
    if en_text_match:
        intermediate_en_text = en_text_match.group(1).strip()

    # Pattern for English Sentiment Label:
    en_label_match = re.search(
        r"English Sentiment Label:\s*(.*?)(?=\n\s*(?:Final Sentiment Label \({lrl_name_for_extraction}\):|$))",
        response_text, re.DOTALL | re.IGNORECASE
    )
    if en_label_match:
        # Use the existing robust English label extractor
        intermediate_en_label = extract_sentiment_label_cotr(en_label_match.group(1).strip(), for_lrl_label=False)
    
    # Pattern for Final LRL Sentiment Label:
    # This regex also considers slight variations like "Final Sentiment (LRL Name):"
    lrl_label_match = re.search(
        rf"Final Sentiment Label \({re.escape(lrl_name_for_extraction)}\):\s*(.*?)(?=\n\s*\n|$)|Final Sentiment \({re.escape(lrl_name_for_extraction)}\):\s*(.*?)(?=\n\s*\n|$)",
        response_text, re.DOTALL | re.IGNORECASE
    )
    if lrl_label_match:
        # The actual captured group might be group(1) or group(2) depending on which part of the OR matched.
        predicted_lrl_label = (lrl_label_match.group(1) or lrl_label_match.group(2) or "").strip()
        # Further clean the LRL label if necessary (e.g. if model adds quotes)
        predicted_lrl_label = extract_sentiment_label_cotr(predicted_lrl_label, for_lrl_label=True, lang_code=lrl_name_for_extraction.lower()) # Assuming lrl_name is like "Swahili"

    # Log if parts are missing, which can indicate prompt adherence issues
    if intermediate_en_text is None: logger.debug(f"Single-prompt: Could not extract 'English Text' for {lrl_name_for_extraction}.")
    if intermediate_en_label is None: logger.debug(f"Single-prompt: Could not extract 'English Sentiment Label' for {lrl_name_for_extraction}.")
    if predicted_lrl_label is None: logger.debug(f"Single-prompt: Could not extract 'Final Sentiment Label ({lrl_name_for_extraction})'.")

    return intermediate_en_text, intermediate_en_label, predicted_lrl_label


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