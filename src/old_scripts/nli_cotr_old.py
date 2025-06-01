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

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

# Define language names dictionary at module level
lang_names = {
    "sw": "Swahili",
    "te": "Telugu",
    "en": "English",
    "ha": "Hausa",
    "ur": "Urdu"
}

# Define label mapping for NLI
NLI_LABELS = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

# Define translations of NLI labels
NLI_TRANSLATIONS = {
    "sw": {  # Swahili
        "entailment": "entailment",  # Often kept as English in datasets
        "neutral": "neutral",
        "contradiction": "contradiction"
    },
    "ha": {  # Hausa
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction"
    },
    "te": {  # Telugu
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction"
    },
    "ur": {  # Urdu
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction"
    }
}

# Import the baseline implementation for shared functionality
from src.experiments.baseline.nli.nli_baseline import initialize_model, extract_nli_label

# Import shared functionality
# Assuming translate_text and initialize_model might be reusable
# If they are defined in a shared utils module, import from there.
# For now, duplicating/adapting them here for clarity.
from src.utils.data_loaders.load_xnli import load_xnli_samples
from src.experiments.simple_baseline.nli_simple_baselines import calculate_metrics as calculate_nli_metrics

def initialize_model(model_name: str) -> Tuple:
    """
    Initialize a model for NLI task.
    
    Args:
        model_name: Name of the model to initialize
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    return model, tokenizer

def generate_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """
    Generate a prompt for translating text from source to target language.
    
    Args:
        text: The text to translate
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        A formatted translation prompt
    """
    # Get full language names
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)
    
    # Simple translation prompt
    prompt = f"""Translate the following {source_name} text to {target_name}:

{source_name} text: {text}

{target_name} translation:"""
    
    return prompt

def get_language_name(lang_code: str) -> str:
    """Get the full language name from a language code."""
    language_names = {
        "en": "English",
        "sw": "Swahili",
        "ha": "Hausa",
        "yo": "Yoruba",
        "ig": "Igbo",
        "te": "Telugu",
        "hi": "Hindi",
        "ur": "Urdu"
    }
    return language_names.get(lang_code, lang_code)

def generate_nli_prompt(premise: str, hypothesis: str, use_few_shot: bool = True) -> str:
    """
    Generate a prompt for NLI task in English.
    
    Args:
        premise: The premise text in English
        hypothesis: The hypothesis text in English
        use_few_shot: Whether to include few-shot examples
        
    Returns:
        A formatted NLI prompt
    """
    # Basic instructions
    instructions = "Given a premise and hypothesis, determine if the hypothesis is an ENTAILMENT, CONTRADICTION, or NEUTRAL with respect to the premise."
    
    # Add definitions
    definitions = """
ENTAILMENT: The hypothesis definitely follows from the premise.
CONTRADICTION: The hypothesis definitely contradicts the premise.
NEUTRAL: The hypothesis might be true or false; there's not enough information to tell.
"""
    
    # Format the specific prompt for this example
    task_prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"
    
    # Few-shot examples
    few_shot_examples = ""
    
    if use_few_shot:
        few_shot_examples = """
Example 1:
Premise: The chef is cooking a meal in the kitchen.
Hypothesis: The chef is preparing food.
Label: ENTAILMENT

Example 2:
Premise: The boy is playing soccer in the park.
Hypothesis: The boy is swimming in a pool.
Label: CONTRADICTION

Example 3:
Premise: The woman is walking down the street.
Hypothesis: She is going to the grocery store.
Label: NEUTRAL

"""
    
    # Combine everything into a final prompt
    prompt = f"{instructions}\n{definitions}\n{few_shot_examples}\n{task_prompt}"
    
    return prompt

def generate_single_prompt_nli_cotr(premise: str, hypothesis: str, lang_code: str, use_few_shot: bool = True) -> str:
    """
    Generate a single prompt for CoTR NLI (translation + reasoning in one step).
    
    Args:
        premise: The premise text in LRL
        hypothesis: The hypothesis text in LRL
        lang_code: Language code
        use_few_shot: Whether to include few-shot examples
        
    Returns:
        A formatted prompt that combines translation and reasoning
    """
    # Get full language names
    source_name = get_language_name(lang_code)
    
    # Basic instructions for combined translation and NLI with structured output
    instructions = f"""Text: '<PREMISE>' (in {source_name})
Hypothesis: '<HYPOTHESIS>' (in {source_name})

Perform the following tasks in order:
1. Translate the {source_name} text to English
2. Translate the {source_name} hypothesis to English
3. Determine if the hypothesis is ENTAILMENT, CONTRADICTION, or NEUTRAL with respect to the text

Definitions:
- ENTAILMENT: The hypothesis must be true if the text is true
- CONTRADICTION: The hypothesis cannot be true if the text is true
- NEUTRAL: The hypothesis might be true or false; the text doesn't provide enough information

Provide your answer in this exact format:
English text: [Your translation of the {source_name} text]
English hypothesis: [Your translation of the {source_name} hypothesis]
Relationship: [ENTAILMENT/CONTRADICTION/NEUTRAL]"""
    
    # Replace the placeholders with actual premise and hypothesis
    instructions = instructions.replace('<PREMISE>', premise).replace('<HYPOTHESIS>', hypothesis)
    
    # Few-shot examples with clear structure
    few_shot_examples = ""
    
    if use_few_shot:
        if lang_code == "sw":  # Swahili example
            few_shot_examples = """
Example:
Text: 'Mwanamke anapika chakula jikoni.' (in Swahili)
Hypothesis: 'Mwanamke anatayarisha chakula.' (in Swahili)

English text: A woman is cooking food in the kitchen.
English hypothesis: A woman is preparing food.
Relationship: ENTAILMENT

Example:
Text: 'Mtoto anacheza mpira wa miguu uwanjani.' (in Swahili)
Hypothesis: 'Mtoto anaogelea baharini.' (in Swahili)

English text: A child is playing football in the field.
English hypothesis: A child is swimming in the sea.
Relationship: CONTRADICTION

Example:
Text: 'Mwanaume anatembea mtaani.' (in Swahili)
Hypothesis: 'Anaenda dukani.' (in Swahili)

English text: A man is walking in the neighborhood.
English hypothesis: He is going to the store.
Relationship: NEUTRAL
"""
        elif lang_code == "te":  # Telugu example
            few_shot_examples = """
Example:
Text: 'ఒక మహిళ వంటగదిలో వంట చేస్తోంది.' (in Telugu)
Hypothesis: 'మహిళ ఆహారం తయారు చేస్తోంది.' (in Telugu)

English text: A woman is cooking food in the kitchen.
English hypothesis: The woman is preparing food.
Relationship: ENTAILMENT

Example:
Text: 'పిల్లాడు మైదానంలో ఫుట్బాల్ ఆడుతున్నాడు.' (in Telugu)
Hypothesis: 'పిల్లాడు సముద్రంలో ఈదుతున్నాడు.' (in Telugu)

English text: A boy is playing football in the field.
English hypothesis: The boy is swimming in the sea.
Relationship: CONTRADICTION

Example:
Text: 'ఒక వ్యక్తి వీధిలో నడుస్తున్నాడు.' (in Telugu)
Hypothesis: 'అతను దుకాణానికి వెళుతున్నాడు.' (in Telugu)

English text: A man is walking in the street.
English hypothesis: He is going to the shop.
Relationship: NEUTRAL
"""
        elif lang_code == "ur":  # Urdu example
            few_shot_examples = """
Example:
Text: 'ایک عورت باورچی خانے میں کھانا پکا رہی ہے۔' (in Urdu)
Hypothesis: 'عورت کھانا تیار کر رہی ہے۔' (in Urdu)

English text: A woman is cooking food in the kitchen.
English hypothesis: A woman is preparing food.
Relationship: ENTAILMENT

Example:
Text: 'ایک بچہ میدان میں فٹبال کھیل رہا ہے۔' (in Urdu)
Hypothesis: 'بچہ سمندر میں تیر رہا ہے۔' (in Urdu)

English text: A child is playing football in the field.
English hypothesis: A child is swimming in the sea.
Relationship: CONTRADICTION

Example:
Text: 'ایک آدمی گلی میں چل رہا ہے۔' (in Urdu)
Hypothesis: 'وہ دکان پر جا رہا ہے۔' (in Urdu)

English text: A man is walking in the street.
English hypothesis: He is going to the shop.
Relationship: NEUTRAL
"""
        else:  # Default English example for other languages
            few_shot_examples = """
Example:
Text: '[Source language text for: A woman is cooking food in the kitchen.]' (in Other Language)
Hypothesis: '[Source language text for: A woman is preparing food.]' (in Other Language)

English text: A woman is cooking food in the kitchen.
English hypothesis: A woman is preparing food.
Relationship: ENTAILMENT

Example:
Text: '[Source language text for: A child is playing football in the field.]' (in Other Language)
Hypothesis: '[Source language text for: A child is swimming in the sea.]' (in Other Language)

English text: A child is playing football in the field.
English hypothesis: A child is swimming in the sea.
Relationship: CONTRADICTION

Example:
Text: '[Source language text for: A man is walking in the neighborhood.]' (in Other Language)
Hypothesis: '[Source language text for: He is going to the store.]' (in Other Language)

English text: A man is walking in the neighborhood.
English hypothesis: He is going to the store.
Relationship: NEUTRAL
"""
    
    # Combine instructions and examples
    if use_few_shot:
        prompt = f"{instructions}\n\n{few_shot_examples}\n\nYour answer:"
    else:
        prompt = f"{instructions}\n\nYour answer:"
    
    return prompt

def translate_text(
    model: Any,
    tokenizer: Any,
    text: str,
    source_lang: str,
    target_lang: str,
    max_input_length: int = 4096,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    do_sample: bool = False,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.0
) -> str:
    """
    Translate text using the provided model.
    
    Args:
        model: The language model
        tokenizer: The model tokenizer
        text: The text to translate
        source_lang: Source language code
        target_lang: Target language code
        max_input_length: Maximum length of the input sequence
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        do_sample: Whether to use sampling
        top_p: Top-p parameter for sampling
        top_k: Top-k parameter for sampling
        repetition_penalty: Repetition penalty for sampling
        
    Returns:
        The translated text
    """
    # Generate translation prompt
    prompt = generate_translation_prompt(text, source_lang, target_lang)
    
    # Debug: Print the prompt being used
    print(f"\nTranslation Prompt ({source_lang} → {target_lang}):")
    print(f"{prompt[:100]}...")
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Get the generated text
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Clean the response
    cleaned_response = clean_translation_response(response, target_lang, source_lang)
    
    # Debug: Print raw and cleaned translation
    print(f"\nRaw translation: '{response[:100]}...'")
    print(f"Cleaned translation: '{cleaned_response[:100]}...'")

    return cleaned_response

def clean_translation_response(response: str, target_lang: str, source_lang: str) -> str:
    """
    Clean the translation response to extract just the translated text.
    
    Args:
        response: The model response containing translation
        target_lang: Target language code
        source_lang: Source language code
        
    Returns:
        Cleaned translated text
    """
    # Strip common prefixes or suffixes
    prefixes_to_remove = [
        "translation:", 
        f"{get_language_name(target_lang)}:", 
        f"{get_language_name(target_lang)} translation:",
        f"{target_lang.upper()} translation:",
        f"{target_lang} translation:"
    ]
    
    # First, strip any leading/trailing whitespace
    cleaned = response.strip()
    
    # Remove known prefixes
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    
    # Remove quotes if they wrap the entire translation
    if (cleaned.startswith('"') and cleaned.endswith('"')) or \
       (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    
    return cleaned

def extract_nli_label(output_text: str) -> str:
    """
    Extract NLI label from model output text with improved handling of structured outputs.
    
    Args:
        output_text: Raw text generated by the model
        
    Returns:
        Extracted label: one of "entailment", "contradiction", "neutral"
    """
    # Convert to lowercase for matching but keep original for capitalized label detection
    text = output_text.lower().strip()
    
    # Look for the structured format we requested: "Relationship: LABEL"
    relationship_pattern = re.search(r'relationship\s*:\s*(entailment|contradiction|neutral)', text, re.IGNORECASE)
    if relationship_pattern:
        return relationship_pattern.group(1).lower()
    
    # Look for capitalized labels (from the structured prompt response)
    if "ENTAILMENT" in output_text:
        return "entailment"
    elif "CONTRADICTION" in output_text:
        return "contradiction"
    elif "NEUTRAL" in output_text:
        return "neutral"
    
    # Check for exact matches in lowercase
    if "entailment" in text:
        return "entailment"
    elif "contradiction" in text:
        return "contradiction"
    elif "neutral" in text:
        return "neutral"
    
    # Look for a line with just the label (e.g., "Step 4: ENTAILMENT")
    step_pattern = re.search(r'step\s*\d*\s*:\s*(entailment|contradiction|neutral)', text, re.IGNORECASE)
    if step_pattern:
        return step_pattern.group(1).lower()
    
    # Try to find the last line which might contain just the label
    last_line = text.strip().split('\n')[-1].strip().lower()
    if last_line in ["entailment", "contradiction", "neutral"]:
        return last_line
    
    # Try to extract the first word of the last line which might be the label
    last_line_words = last_line.split()
    if last_line_words and last_line_words[0] in ["entailment", "contradiction", "neutral"]:
        return last_line_words[0]
    
    # Check common variations or expansions across the text
    if any(x in text for x in ["entail", "follows", "must be true", "definitely true"]):
        return "entailment"
    elif any(x in text for x in ["contradict", "cannot be true", "definitely false", "inconsistent"]):
        return "contradiction"
    elif any(x in text for x in ["neutral", "might be true", "not enough information", "unknown"]):
        return "neutral"
    
    # Default to neutral if no match is found
    print(f"WARNING: Could not extract label from CoTR output, defaulting to 'neutral'")
    print(f"Output was: '{output_text[:100]}...'")
    return "neutral"

def extract_english_translations(output_text: str) -> Tuple[str, str]:
    """
    Extract English translations of premise and hypothesis from single-prompt output.
    
    Args:
        output_text: Raw output text from the model
        
    Returns:
        Tuple of (premise_en, hypothesis_en)
    """
    # Initialize with defaults
    premise_en = ""
    hypothesis_en = ""
    
    # Look for step patterns in the output
    step1_pattern = r"(?:Step 1:|English Premise:)(.+?)(?:Step 2:|English Hypothesis:)"
    step2_pattern = r"(?:Step 2:|English Hypothesis:)(.+?)(?:Step 3:|Determine)"
    
    # Try to extract English premise
    premise_match = re.search(step1_pattern, output_text, re.DOTALL)
    if premise_match:
        premise_en = premise_match.group(1).strip()
    
    # Try to extract English hypothesis
    hypothesis_match = re.search(step2_pattern, output_text, re.DOTALL)
    if hypothesis_match:
        hypothesis_en = hypothesis_match.group(1).strip()
    
    # If regular patterns failed, look for more generic English indicators
    if not premise_en:
        english_premise_pattern = r"English Premise:(.+?)(?:English Hypothesis:|\n\n)"
        match = re.search(english_premise_pattern, output_text, re.DOTALL)
        if match:
            premise_en = match.group(1).strip()
    
    if not hypothesis_en:
        english_hypothesis_pattern = r"English Hypothesis:(.+?)(?:Step 3|\n\n)"
        match = re.search(english_hypothesis_pattern, output_text, re.DOTALL)
        if match:
            hypothesis_en = match.group(1).strip()
    
    return premise_en, hypothesis_en

def process_nli_cotr_single_prompt(
    model: Any,
    tokenizer: Any,
    premise: str,
    hypothesis: str,
    lang_code: str = "sw",
    max_input_length: int = 4096,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    do_sample: bool = True,
    use_few_shot: bool = True,
    top_p: float = 0.92,
    top_k: int = 40,
    repetition_penalty: float = 1.1
) -> Dict[str, Any]:
    """
    Process a single NLI example using the improved CoTR single-prompt approach.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        premise: The premise text in source language
        hypothesis: The hypothesis text in source language
        lang_code: Language code
        max_input_length: Maximum length of the input sequence
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling
        do_sample: Whether to use sampling
        use_few_shot: Whether to use few-shot prompting
        top_p: Top-p parameter for sampling
        top_k: Top-k parameter for sampling
        repetition_penalty: Repetition penalty for sampling
        
    Returns:
        Dictionary with prediction results
    """
    import time
    
    # Start timing
    start_time = time.time()
    
    # Generate the single-step CoTR prompt
    prompt = generate_single_prompt_nli_cotr(
        premise, 
        hypothesis, 
        lang_code=lang_code, 
        use_few_shot=use_few_shot
    )
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(model.device)
    
    # Add model-specific adjustments for better performance
    # repetition_penalty = 1.1  # REMOVED internal definition
    # top_p = 0.92 # REMOVED internal definition
    # top_k = 40 # REMOVED internal definition
    
    # Adjust parameters by language for better results
    current_repetition_penalty = repetition_penalty # Use passed-in value
    current_temperature = temperature
    current_top_p = top_p
    current_top_k = top_k

    if lang_code != "en":
        current_temperature = max(0.08, temperature * 0.8)
        current_top_p = min(0.95, top_p * 1.05)
        current_repetition_penalty = repetition_penalty * 1.1
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=current_temperature,
            do_sample=do_sample,
            top_p=current_top_p,
            top_k=current_top_k,
            repetition_penalty=current_repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # Get the generated text
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    response = response.strip()
    
    # Extract English translations and NLI label from the response
    try:
        premise_en, hypothesis_en = extract_english_translations(response)
        nli_label = extract_nli_label(response)
    except Exception as e:
        print(f"Error extracting translations or label: {e}")
        premise_en = f"[Error: Could not extract EN translation for: {premise}]"
        hypothesis_en = f"[Error: Could not extract EN translation for: {hypothesis}]"
        nli_label = "neutral"  # Default to neutral on extraction failure
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    return {
        "predicted_label": nli_label,
        "premise_en": premise_en,
        "hypothesis_en": hypothesis_en,
        "raw_response": response,
        "runtime_seconds": runtime,
        "pipeline": "single_prompt",
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty
    }

def process_nli_cotr_multi_prompt(
    model: Any,
    tokenizer: Any,
    premise: str,
    hypothesis: str,
    lang_code: str = "sw",
    max_input_length: int = 4096,
    max_new_tokens: int = 50,
    max_translation_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = True,
    use_few_shot: bool = True,
    trans_temp: float = 0.3,
    trans_top_p: float = 0.9,
    trans_top_k: int = 40,
    trans_repetition_penalty: float = 1.0,
    nli_temp: float = 0.1,
    nli_top_p: float = 0.92,
    nli_top_k: int = 40,
    nli_repetition_penalty: float = 1.1
) -> Dict[str, Any]:
    """
    Process an NLI example using the improved CoTR multi-prompt approach.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        premise: The premise text in source language
        hypothesis: The hypothesis text in source language
        lang_code: Language code
        max_input_length: Maximum length of the input sequence
        max_new_tokens: Maximum number of tokens for the NLI step
        max_translation_tokens: Maximum number of tokens for the translation step
        temperature: Temperature for sampling
        do_sample: Whether to use sampling
        use_few_shot: Whether to use few-shot prompting
        trans_temp: Temperature for translation
        trans_top_p: Top-p parameter for translation
        trans_top_k: Top-k parameter for translation
        trans_repetition_penalty: Repetition penalty for translation
        nli_temp: Temperature for NLI
        nli_top_p: Top-p parameter for NLI
        nli_top_k: Top-k parameter for NLI
        nli_repetition_penalty: Repetition penalty for NLI
        
    Returns:
        Dictionary with prediction results
    """
    import time
    
    # Start timing
    start_time = time.time()
    
    # Step 1: Translate premise to English
    premise_translation_prompt = generate_translation_prompt(premise, lang_code, "en")
    inputs = tokenizer(premise_translation_prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(model.device)
    
    # Translation generation parameters (use passed-in trans_... params)
    # translation_temp = max(0.05, temperature * 0.8)  # REMOVED internal logic
    # translation_top_p = 0.95 # REMOVED internal logic
    # translation_repetition_penalty = 1.0 # REMOVED internal logic
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_translation_tokens,
            temperature=trans_temp,
            do_sample=do_sample,
            top_p=trans_top_p,
            repetition_penalty=trans_repetition_penalty,
            top_k=trans_top_k,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Get and clean the translated premise
    premise_en_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    premise_en = clean_translation_response(premise_en_response, "en", lang_code)
    
    # Step 2: Translate hypothesis to English
    hypothesis_translation_prompt = generate_translation_prompt(hypothesis, lang_code, "en")
    inputs = tokenizer(hypothesis_translation_prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_translation_tokens,
            temperature=trans_temp,
            do_sample=do_sample,
            top_p=trans_top_p,
            repetition_penalty=trans_repetition_penalty,
            top_k=trans_top_k,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Get and clean the translated hypothesis
    hypothesis_en_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    hypothesis_en = clean_translation_response(hypothesis_en_response, "en", lang_code)
    
    # Step 3: Perform NLI on the English translations
    nli_prompt = generate_nli_prompt(premise_en, hypothesis_en, use_few_shot=use_few_shot)
    inputs = tokenizer(nli_prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(model.device)
    
    # NLI inference parameters (use passed-in nli_... params)
    # nli_repetition_penalty = 1.1 # REMOVED internal logic
    # nli_top_p = 0.92 # REMOVED internal logic
    # nli_top_k = 40 # REMOVED internal logic
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=nli_temp,
            do_sample=do_sample,
            top_p=nli_top_p,
            top_k=nli_top_k,
            repetition_penalty=nli_repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Get the NLI prediction
    nli_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    try:
        nli_label = extract_nli_label(nli_response)
    except Exception as e:
        print(f"Error extracting NLI label: {e}")
        nli_label = "neutral"  # Default to neutral on extraction failure
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    return {
        "premise_en": premise_en,
        "hypothesis_en": hypothesis_en,
        "predicted_label": nli_label,
        "premise_translation_response": premise_en_response,
        "hypothesis_translation_response": hypothesis_en_response,
        "nli_response": nli_response,
        "runtime_seconds": runtime,
        "pipeline": "multi_prompt",
        "trans_temp": trans_temp,
        "trans_top_p": trans_top_p,
        "trans_top_k": trans_top_k,
        "trans_repetition_penalty": trans_repetition_penalty,
        "nli_temp": nli_temp,
        "nli_top_p": nli_top_p,
        "nli_top_k": nli_top_k,
        "nli_repetition_penalty": nli_repetition_penalty
    }

def evaluate_nli_cotr(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str,
    temperature: float = 0.3,
    max_new_tokens: int = 50,
    max_translation_tokens: int = 512,
    do_sample: bool = False,
    use_few_shot: bool = True,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.0
) -> pd.DataFrame:
    """
    Evaluate NLI using the multi-prompt CoTR approach (separate translation and reasoning).
    
    Args:
        model_name: Name of the model to use
        samples_df: DataFrame containing NLI samples
        lang_code: Language code
        temperature: Temperature for generation
        max_new_tokens: Maximum new tokens for NLI
        max_translation_tokens: Maximum new tokens for translation
        do_sample: Whether to use sampling
        use_few_shot: Whether to include few-shot examples
        top_p: Top-p parameter for sampling
        top_k: Top-k parameter for sampling
        repetition_penalty: Repetition penalty for sampling
        
    Returns:
        DataFrame with results
    """
    print(f"\nInitializing {model_name} for multi-prompt CoTR NLI evaluation...")
    
    # Initialize model
    model, tokenizer = initialize_model(model_name)
    model.eval()  # Set to evaluation mode
    
    # Process all samples
    results = []
    start_time = time.time()
    
    # Print sample count
    print(f"Processing {len(samples_df)} samples for {lang_code} with multi-prompt CoTR...")
    
    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} samples"):
        premise = row['premise']
        hypothesis = row['hypothesis']
        ground_truth = row['label']  # Use consistent name for gold label
        
        # Debug: Print sample information
        print(f"\nProcessing sample {idx+1}:")
        print(f"Premise: '{premise[:50]}...'")
        print(f"Hypothesis: '{hypothesis[:50]}...'")
        print(f"Ground truth: '{ground_truth}'")
        
        # Process with multi-prompt CoTR
        result = process_nli_cotr_multi_prompt(
            model,
            tokenizer,
            premise,
            hypothesis,
            lang_code=lang_code,
            max_new_tokens=max_new_tokens,
            max_translation_tokens=max_translation_tokens,
            temperature=temperature,
            do_sample=do_sample,
            use_few_shot=use_few_shot,
            trans_temp=temperature,
            trans_top_p=top_p,
            trans_top_k=top_k,
            trans_repetition_penalty=repetition_penalty,
            nli_temp=temperature,
            nli_top_p=top_p,
            nli_top_k=top_k,
            nli_repetition_penalty=repetition_penalty
        )
        
        # Add sample and result information
        result.update({
            "premise": premise,
            "hypothesis": hypothesis,
            "gold_label": ground_truth,
            "language": lang_code,
            "pipeline": "multi_prompt",
            "few_shot": use_few_shot,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty
        })
        
        results.append(result)
    
    # Create results DataFrame
    total_runtime = time.time() - start_time
    results_df = pd.DataFrame(results)
    
    # Add experiment information
    results_df['model'] = model_name
    results_df['temperature'] = temperature
    results_df['max_new_tokens'] = max_new_tokens
    results_df['max_translation_tokens'] = max_translation_tokens
    results_df['do_sample'] = do_sample
    results_df['top_p'] = top_p
    results_df['top_k'] = top_k
    results_df['repetition_penalty'] = repetition_penalty
    results_df['total_runtime_seconds'] = total_runtime
    results_df['runtime_per_sample'] = total_runtime / len(samples_df) if len(samples_df) > 0 else 0
    
    # Debug: Output sample of predictions
    print("\nSample of multi-prompt CoTR predictions:")
    if not results_df.empty:
        sample_size = min(5, len(results_df))
        for i in range(sample_size):
            row = results_df.iloc[i]
            print(f"\nExample {i+1}:")
            print(f"  Original Premise: '{row['premise'][:50]}...'")
            print(f"  Translated Premise: '{row['premise_en'][:50]}...'")
            print(f"  Original Hypothesis: '{row['hypothesis'][:50]}...'")
            print(f"  Translated Hypothesis: '{row['hypothesis_en'][:50]}...'")
            print(f"  Gold Label: '{row['gold_label']}'")
            print(f"  Predicted Label: '{row['predicted_label']}'")
    
    return results_df

def evaluate_nli_cotr_single_prompt(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str,
    temperature: float = 0.3,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    use_few_shot: bool = True,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1
) -> pd.DataFrame:
    """
    Evaluate NLI using the single-prompt CoTR approach (combined translation and reasoning).
    
    Args:
        model_name: Name of the model to use
        samples_df: DataFrame containing NLI samples
        lang_code: Language code
        temperature: Temperature for generation
        max_new_tokens: Maximum new tokens to generate
        do_sample: Whether to use sampling
        use_few_shot: Whether to include few-shot examples
        top_p: Top-p parameter for sampling
        top_k: Top-k parameter for sampling
        repetition_penalty: Repetition penalty for sampling
    
    Returns:
        DataFrame with results
    """
    print(f"\nInitializing {model_name} for single-prompt CoTR NLI evaluation...")
    
    # Initialize model
    model, tokenizer = initialize_model(model_name)
    model.eval()  # Set to evaluation mode
    
    # Process all samples
    results = []
    start_time = time.time()
    
    # Print sample count
    print(f"Processing {len(samples_df)} samples for {lang_code} with single-prompt CoTR...")
    
    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} samples"):
        premise = row['premise']
        hypothesis = row['hypothesis']
        ground_truth = row['label']  # Use consistent name for gold label
        
        # Debug: Print sample information
        print(f"\nProcessing sample {idx+1}:")
        print(f"Premise: '{premise[:50]}...'")
        print(f"Hypothesis: '{hypothesis[:50]}...'")
        print(f"Ground truth: '{ground_truth}'")
        
        # Process with single-prompt CoTR
        result = process_nli_cotr_single_prompt(
            model,
            tokenizer,
            premise,
            hypothesis,
            lang_code=lang_code,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            use_few_shot=use_few_shot,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        
        # Add sample and result information
        result.update({
            "premise": premise,
            "hypothesis": hypothesis,
            "gold_label": ground_truth,
            "language": lang_code,
            "pipeline": "single_prompt",
            "few_shot": use_few_shot,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty
        })
        
        results.append(result)
    
    # Create results DataFrame
    total_runtime = time.time() - start_time
    results_df = pd.DataFrame(results)
    
    # Add experiment information
    results_df['model'] = model_name
    results_df['temperature'] = temperature
    results_df['max_new_tokens'] = max_new_tokens
    results_df['do_sample'] = do_sample
    results_df['top_p'] = top_p
    results_df['top_k'] = top_k
    results_df['repetition_penalty'] = repetition_penalty
    results_df['total_runtime_seconds'] = total_runtime
    results_df['runtime_per_sample'] = total_runtime / len(samples_df) if len(samples_df) > 0 else 0
    
    # Debug: Output sample of predictions
    print("\nSample of single-prompt CoTR predictions:")
    if not results_df.empty:
        sample_size = min(5, len(results_df))
        for i in range(sample_size):
            row = results_df.iloc[i]
            print(f"\nExample {i+1}:")
            print(f"  Original Premise: '{row['premise'][:50]}...'")
            print(f"  Translated Premise: '{row['premise_en'][:50]}...'")
            print(f"  Original Hypothesis: '{row['hypothesis'][:50]}...'")
            print(f"  Translated Hypothesis: '{row['hypothesis_en'][:50]}...'")
            print(f"  Gold Label: '{row['gold_label']}'")
            print(f"  Predicted Label: '{row['predicted_label']}'")
    
    return results_df

def calculate_nli_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate metrics for NLI evaluation.
    
    Args:
        results_df: DataFrame with NLI results
        
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    # Debug: Print the first few rows to check format
    print("\nFirst 5 rows of results DataFrame:")
    print(results_df[['premise', 'premise_en', 'gold_label', 'predicted_label']].head().to_string())
    
    # Count label distribution
    print("\nGold label distribution:")
    print(results_df['gold_label'].value_counts())
    
    print("\nPredicted label distribution:")
    print(results_df['predicted_label'].value_counts())
    
    # Make sure we have valid data
    if 'predicted_label' not in results_df.columns or 'gold_label' not in results_df.columns:
        print("ERROR: Missing 'predicted_label' or 'gold_label' columns")
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'class_metrics': {}}
    
    if results_df['predicted_label'].isnull().all() or results_df['gold_label'].isnull().all():
        print("ERROR: All null values in 'predicted_label' or 'gold_label'")
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'class_metrics': {}}
    
    # Create a copy to avoid modifying the original DataFrame
    eval_df = results_df.copy()
    
    # Check if gold_label is numeric and predicted_label is string
    if pd.api.types.is_numeric_dtype(eval_df['gold_label']) and pd.api.types.is_string_dtype(eval_df['predicted_label']):
        print("Converting numeric gold labels to strings for compatibility...")
        # Map from numeric labels to string labels
        label_map = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }
        # Convert gold_label from numeric to string format
        eval_df['gold_label'] = eval_df['gold_label'].map(label_map)
        print("After conversion, gold label distribution:")
        print(eval_df['gold_label'].value_counts())
    
    # Calculate accuracy
    accuracy = accuracy_score(eval_df['gold_label'], eval_df['predicted_label'])
    
    # Calculate macro F1 score
    macro_f1 = f1_score(
        eval_df['gold_label'], 
        eval_df['predicted_label'], 
        average='macro', 
        zero_division=0
    )
    
    # Get detailed metrics for each class
    class_report = classification_report(
        eval_df['gold_label'], 
        eval_df['predicted_label'], 
        output_dict=True, 
        zero_division=0
    )
    
    # Extract class metrics
    class_metrics = {}
    for label in ["entailment", "neutral", "contradiction"]:
        if label in class_report:
            class_metrics[label] = {
                'precision': class_report[label]['precision'],
                'recall': class_report[label]['recall'],
                'f1': class_report[label]['f1-score']
            }
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(eval_df['gold_label'], eval_df['predicted_label'], zero_division=0))
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'class_metrics': class_metrics
    } 