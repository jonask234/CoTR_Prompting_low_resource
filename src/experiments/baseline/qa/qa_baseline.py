# thesis_project/baseline/qa_baseline.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import time
from collections import Counter
from typing import Any

def initialize_model(model_name):
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
        cache_dir=cache_path # Add cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        cache_dir=cache_path # Add cache_dir
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

def generate_qa_prompt(question: str, context: str = None, lang_code: str = "en", model_name: str = "", use_few_shot: bool = True) -> str:
    """
    Generate a prompt for question answering.
    Instructions are language-specific, but few-shot examples are ALWAYS IN ENGLISH.
    
    Args:
        question: The question to answer
        context: Context to use (if any)
        lang_code: Language code for language-specific prompting
        model_name: Model name for potential model-specific prompt adjustments
        use_few_shot: Whether to include few-shot examples in the prompt
    
    Returns:
        Formatted prompt with instructions and examples
    """
    # Define UNIVERSAL English few-shot examples
    english_few_shot_examples_text = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "When was the United Nations founded?", "answer": "1945"},
        {"question": "Who wrote 'To Kill a Mockingbird'?", "answer": "Harper Lee"},
        {"question": "Is the sky blue?", "answer": "Yes"},
        {"question": "What is the speed of dark?", "answer": "I don't know"}
    ]
    # Language-specific instruction components
    if lang_code == "sw":
        instructions = (
            "Jibu swali kwa usahihi na kwa ufupi sana ukitumia maarifa yako ya ndani (na muktadha uliotolewa ikiwa upo).\\n"
            "Fuata sheria hizi kwa makini:\\n"
            "- Kwa maswali ya ndiyo/hapana: jibu KWA NENO MOJA TU 'Ndio' au 'Hapana'.\\n"
            "- Kwa maswali ya ukweli: toa UKWELI TU, jina, tarehe, au nambari husika.\\n"
            "- USIONGEZE maelezo yoyote, vikanusho, au maneno ya mazungumzo kama \\\"Jibu ni...\\\" au \\\"Kulingana na ninavyojua...\\\".\\n"
            "- Jibu lako lote linapaswa kuwa na maneno 1-5 tu.\\n"
            "- Ikiwa hujui jibu, jibu KWA MANENO HAYA TU: 'Sijui'."
        )
        examples_header = "Mifano (kwa Kiingereza):" # Clarify examples are in English
        question_prefix = "Swali:"
        answer_prefix = "Jibu:"
        context_prefix = "Muktadha:"
    elif lang_code == "te":
        instructions = (
            "మీ అంతర్గత జ్ఞానాన్ని (మరియు అందించినట్లయితే సందర్భాన్ని) ఉపయోగించి ఖచ్చితంగా మరియు చాలా క్లుప్తంగా ప్రశ్నకు సమాధానం ఇవ్వండి.\\n"
            "ఈ నియమాలను ఖచ్చితంగా పాటించండి:\\n"
            "- అవును/కాదు ప్రశ్నలకు: 'అవును' లేదా 'కాదు' అని మాత్రమే ప్రతిస్పందించండి.\\n"
            "- వాస్తవ ప్రశ్నలకు: నిర్దిష్ట వాస్తవం, పేరు, తేదీ లేదా సంఖ్యను మాత్రమే అందించండి.\\n"
            "- \\\"సమాధానం...\\\" లేదా \\\"నాకు తెలిసినంతవరకు...\\\" వంటి వివరణలు, నిరాకరణలు లేదా సంభాషణ పదబంధాలను జోడించవద్దు.\\n"
            "- మీ మొత్తం ప్రతిస్పందన ఆదర్శంగా 1-5 పదాలు ఉండాలి.\\n"
            "- మీకు సమాధానం తెలియకపోతే, 'నాకు తెలియదు' అనే ఖచ్చితమైన పదబంధంతో మాత్రమే ప్రతిస్పందించండి."
        )
        examples_header = "ఉదాహరణలు (ఆంగ్లంలో):" # Clarify examples are in English
        question_prefix = "ప్రశ్న:"
        answer_prefix = "సమాధానం:"
        context_prefix = "సందర్భం:"
    else: # Default to English
        instructions = (
            "Answer the question accurately and very concisely using your internal knowledge (and the provided context if available).\\n"
            "Follow these rules strictly:\\n"
            "- For yes/no questions: respond with ONLY 'Yes' or 'No'.\\n"
            "- For factual questions: provide ONLY the specific fact, name, date, or number.\\n"
            "- Do NOT add any explanations, disclaimers, or conversational phrases like \\\"The answer is...\\\" or \\\"According to my knowledge...\\\".\\n"
            "- Your entire response should ideally be 1-5 words.\\n"
            "- If you do not know the answer, respond with ONLY the exact phrase 'I don\\'t know'."
        )
        examples_header = "Examples:"
        question_prefix = "Question:"
        answer_prefix = "Answer:"
        context_prefix = "Context:"
    
    # Start building the prompt
    prompt = instructions + "\\n\\n"
    
    # Add context if provided
    if context and context.strip(): # Ensure context is not empty or just whitespace
        prompt += f"{context_prefix} {context}\\n\\n"
        
    if use_few_shot:
        prompt += examples_header + "\\n\\n"
        # Few-shot examples are ALWAYS English, but use language-specific Q/A prefixes for the *current* question
        for ex in english_few_shot_examples_text:
            # For the examples themselves, use English "Question:" and "Answer:"
            prompt += f"Question: {ex['question']}\\n" 
            prompt += f"Answer: {ex['answer']}\\n\\n"
    
    # Add the actual question to be answered, using language-specific prefix
    prompt += f"{question_prefix} {question}\\n"
    prompt += f"{answer_prefix} "
    
    return prompt

def extract_answer(text: str, question: str, lang_code: str = "en") -> str:
    """
    Extract the answer from model output with improved language-specific handling.
    
    Args:
        text: The model output text
        question: The original question
        lang_code: Language code for language-specific handling
        
    Returns:
        The cleaned/extracted answer
    """
    # If there's no text, return empty answer
    if not text or text.strip() == "":
        # Return language-specific "no answer" string
        no_answer_map = {
            "en": "[no answer generated]",
            "sw": "Sijui", # Changed from "Hakuna jibu" to match prompt
            "te": "నాకు తెలియదు" # Changed from "సమాధానం లేదు" to match prompt
        }
        return no_answer_map.get(lang_code, "[no answer generated]")
    
    # Clean text from common patterns of noise
    text = text.strip()
    
    # Normalize common "I don't know" phrases to a standard one FIRST for all languages
    # This should catch variations before language-specific yes/no checks
    i_dont_know_patterns = [
        # English
        r"^i don\\'t know\\.?$", r"^i do not know\\.?$", r"^sorry, i don\\'t know\\.?$",
        r"^i am unable to answer\\.?$", r"^i cannot answer that\\.?$",
        r"^i don\\'t have that information\\.?$", r"^i am not sure\\.?$", r"^unknown\\.?$",
        # Swahili (approximations)
        r"^sijui\\.?$", r"^sina uhakika\\.?$", r"^siwezi kujibu\\.?$",
        # Telugu (approximations)
        r"^నాకు తెలియదు\\.?$", r"^నాకు ఖచ్చితంగా తెలియదు\\.?$", r"^చెప్పలేను\\.?$"
    ]
    standard_i_dont_know = {
        "en": "I don't know",
        "sw": "Sijui",
        "te": "నాకు తెలియదు"
    }

    for pattern in i_dont_know_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return standard_i_dont_know.get(lang_code, "I don't know")
    
    # Check if we need to extract the answer part from a longer response
    
    # Language-specific prefix patterns
    prefixes = {
        "en": ["answer:", "the answer is:", "answer is:"],
        "sw": ["jibu:", "jibu ni:", "jibu lake ni:"],
        "te": ["సమాధానం:", "జవాబు:", "సమాధానం ఏమిటంటే:"]
    }
    
    # Get language-specific prefixes or default to English
    lang_prefixes = prefixes.get(lang_code, prefixes["en"])
    
    # Try to extract answer using prefixes
    for prefix in lang_prefixes:
        if prefix in text.lower():
            parts = text.lower().split(prefix, 1)
            if len(parts) > 1:
                text = parts[1].strip()
                # Check for potential end markers
                for end_marker in [".", "\n", ". "]:
                    if end_marker in text:
                        text = text.split(end_marker, 1)[0].strip()
                break
    
    # Handle quotes in the answer
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    elif text.startswith("'") and text.endswith("'"):
        text = text[1:-1].strip()
    
    # Improve handling for yes/no answers
    yes_patterns = {
        "en": ["yes", "yes,", "yes.", "yeah", "correct", "true", "right"],
        "sw": ["ndio", "ndiyo", "naam", "ndivyo", "kweli", "ndiyo", "ndio kweli"],
        "te": ["అవును", "avunu", "ఔను", "ఓను", "సరి", "నిజమే"]
    }
    
    no_patterns = {
        "en": ["no", "no,", "no.", "nope", "not", "false", "incorrect", "wrong"],
        "sw": ["hapana", "la", "sivyo", "si", "siyo", "si kweli", "sikweli"],
        "te": ["కాదు", "లేదు", "కాదండి", "వద్దు", "తప్పు"]
    }
    
    # Get patterns for the language
    yes_terms = yes_patterns.get(lang_code, yes_patterns["en"])
    no_terms = no_patterns.get(lang_code, no_patterns["en"])
    
    # Check for yes/no pattern at the beginning of the answer
    text_lower = text.lower()
    for yes_term in yes_terms:
        if text_lower == yes_term or text_lower.startswith(yes_term + " "):
            if lang_code == "sw":
                return "Ndio"
            elif lang_code == "te":
                return "అవును"
            else:
                return "Yes"
    
    for no_term in no_terms:
        if text_lower == no_term or text_lower.startswith(no_term + " "):
            if lang_code == "sw":
                return "Hapana"
            elif lang_code == "te":
                return "కాదు"
            else:
                return "No"
    
    # Handle common phrase formats
    if lang_code == "sw":
        if "majibu ni " in text.lower():
            text = text.lower().split("majibu ni ", 1)[1].strip()
        elif "jibu ni " in text.lower():
            text = text.lower().split("jibu ni ", 1)[1].strip()
    elif lang_code == "te":
        if "సమాధానం " in text.lower():
            text = text.lower().split("సమాధానం ", 1)[1].strip()
    else:
        if "the answer is " in text.lower():
            text = text.lower().split("the answer is ", 1)[1].strip()
        elif "answer is " in text.lower():
            text = text.lower().split("answer is ", 1)[1].strip()
    
    # Handle period or other punctuation at the end
    if text.endswith(".") or text.endswith(",") or text.endswith(":") or text.endswith(";"):
        text = text[:-1].strip()
    
    # Handling for numeric answers
    if is_numeric_question(question):
        # Extract numeric values with improved regex
        numeric_pattern = r'\b(\d[\d,.]*\d|\d)\b'
        numeric_matches = re.findall(numeric_pattern, text)
        
        if numeric_matches:
            # Return just the first numeric match if found
            return numeric_matches[0].strip()
    
    # Return the first sentence if the answer is too long
    if len(text.split()) > 15:  # If more than 15 words
        sentences = re.split(r'[.!?।]+', text)
        if sentences:
            return sentences[0].strip()
    
    return text.strip()

def is_yes_no_question(question: str) -> bool:
    """
    Determine if a question is a yes/no question with language awareness.
    
    Args:
        question: The question text
        
    Returns:
        Boolean indicating if it's a yes/no question
    """
    # English patterns
    en_patterns = [
        r'^is ', r'^are ', r'^was ', r'^were ', r'^do ', r'^does ', 
        r'^did ', r'^has ', r'^have ', r'^had ', r'^can ', r'^could ', 
        r'^will ', r'^would ', r'^should '
    ]
    
    # Swahili patterns
    sw_patterns = [
        r'^je[,\\s]', r'^je ', r'^ni ', r'^ni[,\\s]', r'^kuna ', 
        r'^kuna[,\\s]', r'^kulikuwa '
    ]
    
    # Telugu patterns
    te_patterns = [
        r'^ఏమి', r'^ఏది', r'^ఉందా', r'^ఉన్నాయా', r'^ఉన్నాడా', 
        r'^ఉన్నారా', r'^అవునా', r'^కాదా'
    ]
    
    # English yes/no words at the end
    en_end_patterns = [r'\?$']
    
    # Combine all patterns
    all_patterns = en_patterns + sw_patterns + te_patterns + en_end_patterns
    
    # Check each pattern
    for pattern in all_patterns:
        if re.search(pattern, question.lower()):
            return True
    
    return False

def is_numeric_question(question: str) -> bool:
    """
    Determine if a question is asking for a numeric answer.
    
    Args:
        question: The question text
        
    Returns:
        Boolean indicating if it's a numeric question
    """
    # English numeric question words
    en_numeric_words = [
        'how many', 'how much', 'how old', 'how long', 'how far',
        'how often', 'how fast', 'how tall', 'how heavy',
        'what year', 'what is the number', 'what is the amount',
        'how high', 'how low', 'how deep', 'how wide',
        'how big', 'how small', 'how expensive', 'how cheap',
    ]
    
    # Swahili numeric question words
    sw_numeric_words = [
        'wangapi', 'ngapi', 'kiasi gani', 'umri gani', 'urefu gani',
        'mara ngapi', 'kasi gani', 'thamani gani', 'bei gani',
        'mwaka gani', 'idadi', 'hesabu', 'namba', 'tarakimu',
    ]
    
    # Telugu numeric question words
    te_numeric_words = [
        'ఎన్ని', 'ఎంత', 'వయసు ఎంత', 'ఎంత దూరం',
        'ఎన్ని సార్లు', 'ఎంత వేగంగా', 'ఎంత ఎత్తు',
        'ఏ సంవత్సరం', 'సంఖ్య ఎంత',
    ]
    
    # Combine all words
    all_numeric_words = en_numeric_words + sw_numeric_words + te_numeric_words
    
    # Check if any numeric question word appears in the question
    for word in all_numeric_words:
        if word in question.lower():
            return True
    
    return False

def process_qa_baseline(tokenizer, model, question, context=None, 
                          max_new_tokens=50,
                          max_input_length=4096,
                          temperature=0.3,
                          top_p=0.85, # Base top_p
                          top_k=40,   # Base top_k, added to params
                          repetition_penalty=1.2, # Base repetition_penalty, added to params
                          do_sample=True, # Base do_sample, added to params
                          lang_code="en",
                          model_name="",
                          use_few_shot: bool = True):
    """
    Process a QA pair with the given model directly (baseline).
    Now with language-specific parameter adjustments and dynamic adjustments based on question type.
    
    Args:
        tokenizer: The model tokenizer
        model: The language model
        question: The question text
        context: The context text
        max_new_tokens: Base maximum number of new tokens to generate for the answer
        max_input_length: Maximum length of the input sequence
        temperature: Base temperature for sampling
        top_p: Base top_p for sampling
        top_k: Base top_k for sampling
        repetition_penalty: Base repetition_penalty
        do_sample: Base do_sample flag
        lang_code: Language code for language-specific handling
        model_name: Name of the model for model-specific parameters
        use_few_shot: Whether to include few-shot examples in the prompt
    
    Returns:
        The model's answer
    """
    # Generate prompt with language-specific examples
    prompt = generate_qa_prompt(question, context, lang_code, model_name, use_few_shot=use_few_shot)
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Initialize generation parameters with base values
    gen_temperature = temperature
    gen_top_p = top_p
    gen_top_k = top_k
    gen_max_tokens = max_new_tokens
    gen_repetition_penalty = repetition_penalty
    gen_do_sample = do_sample
    beam_size = 4 # Default beam_size from old script for Aya

    # Dynamic adjustments based on question type (from old evaluate_qa_baseline)
    if is_yes_no_question(question):
        gen_temperature = max(0.1, temperature - 0.1) # Lower temp for yes/no
        gen_max_tokens = min(30, max_new_tokens)      # Shorter answers
    elif is_numeric_question(question):
        gen_temperature = max(0.1, temperature - 0.15) # Lower temp for numeric
        gen_top_p = max(0.7, top_p - 0.1)             # More focused for numeric
        gen_max_tokens = min(25, max_new_tokens)     # Shorter answers
        
    # Language-specific adjustments (from old process_qa_baseline)
    if lang_code == "sw":
        gen_temperature = 0.2
        gen_top_p = 0.8
        gen_max_tokens = 30
        gen_repetition_penalty = 1.3
        gen_top_k = 30
    elif lang_code == "te":
        gen_temperature = 0.15
        gen_top_p = 0.75
        gen_max_tokens = 25
        gen_repetition_penalty = 1.4
        gen_top_k = 25
    
    # Model-specific adjustments (from old process_qa_baseline)
    is_aya_model = "aya" in model_name.lower()
    is_qwen_model = "qwen" in model_name.lower()
    
    if is_aya_model:
        if lang_code == "sw":
            gen_temperature = 0.15
            gen_repetition_penalty = 1.4
            gen_top_k = 25
        elif lang_code == "te":
            gen_temperature = 0.1
            gen_repetition_penalty = 1.5
            gen_top_k = 20
        else: # English for Aya
            gen_temperature = 0.25
            gen_repetition_penalty = 1.3
            gen_top_k = 30
        
        if lang_code != "en": # Use beam search for Aya with non-English
            gen_do_sample = False
            # beam_size is already set
    
    if is_qwen_model:
        if lang_code == "sw":
            gen_temperature = 0.18
            gen_top_p = 0.7
            gen_repetition_penalty = 1.35
            gen_top_k = 35
        elif lang_code == "te":
            gen_temperature = 0.12
            gen_top_p = 0.65
            gen_repetition_penalty = 1.45
            gen_top_k = 30
        else: # English for Qwen
            gen_temperature = 0.25
            gen_top_p = 0.8
            gen_repetition_penalty = 1.2
            gen_top_k = 40
    
    # Generate answer
    with torch.no_grad():
        if not gen_do_sample and is_aya_model and lang_code != "en": # Condition for beam search
            output_ids = model.generate(
                **inputs,
                max_new_tokens=gen_max_tokens,
                do_sample=False,
                num_beams=beam_size,
                early_stopping=True,
                repetition_penalty=gen_repetition_penalty,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            )
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=gen_max_tokens,
                do_sample=True, # Default to True if not beam searching
                temperature=gen_temperature,
                top_p=gen_top_p,
                top_k=gen_top_k,
                repetition_penalty=gen_repetition_penalty,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            )
    
    # Decode only the newly generated tokens
    output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Extract and clean the answer
    answer = extract_answer(output_text, question, lang_code)
    
    # Special handling for Yes/No questions to avoid answering with explanations
    if is_yes_no_question(question):
        # Language-specific yes/no handling
        if lang_code == "sw":
            if any(term in answer.lower() for term in ["ndio", "ndiyo", "naam"]):
                answer = "Ndio" 
            elif any(term in answer.lower() for term in ["hapana", "la", "sivyo"]):
                answer = "Hapana"
        elif lang_code == "te":
            if any(term in answer.lower() for term in ["అవును", "avunu", "ఔను"]):
                answer = "అవును" 
            elif any(term in answer.lower() for term in ["కాదు", "లేదు", "ledu"]):
                answer = "కాదు"
        else: # English
            # Check if the answer starts with yes/no or is just yes/no
            if answer.lower().startswith("yes") or answer.lower() == "yes":
                 if len(answer) > 5 and not (answer.lower().startswith("yes ") or answer.lower().startswith("yes.")): # Avoid "Yes, it is..."
                     pass # Keep longer answer if it's more than just "Yes" and some punctuation
                 else:
                     answer = "Yes"
            elif answer.lower().startswith("no") or answer.lower() == "no":
                if len(answer) > 4 and not (answer.lower().startswith("no ") or answer.lower().startswith("no.")):
                    pass
                else:
                    answer = "No"

    
    # Final formatting for numeric answers - try to extract just the number if appropriate
    if is_numeric_question(question) and not answer.lower() in ["yes", "no", "ndio", "hapana", "అవును", "కాదు", "[no answer generated]", "i don\'t know", "sijui", "నాకు తెలియదు"]:
        # Enhanced regex that handles different number formats
        numeric_match = re.search(r'\b(\d[\d,.]*\d|\d)\b', answer)
        if numeric_match:
            answer = numeric_match.group(1)
    
    return answer

def calculate_qa_f1(ground_truth, predicted_answer):
    """
    Calculate F1 score between ground truth and predicted answer for QA evaluation.
    
    Args:
        ground_truth: Ground truth answer (string or list/dict with 'text' field)
        predicted_answer: Model's predicted answer (string)
        
    Returns:
        F1 score
    """
    # Handle different input formats
    if isinstance(ground_truth, dict) and 'text' in ground_truth:
        references = ground_truth['text']
    elif isinstance(ground_truth, list):
        references = ground_truth
    else:
        references = [str(ground_truth)]
    
    # Ensure predicted_answer is a string
    prediction = str(predicted_answer)
    
    # Helper function to normalize text
    def normalize_text(text):
        # Convert to lowercase, remove punctuation
        text = text.lower()
        text = re.sub(r'[^\\w\\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\\s+', ' ', text).strip()
        return text
    
    # Helper function to calculate F1 between two strings
    def calculate_f1(a_tokens, b_tokens):
        if not a_tokens or not b_tokens:
            return 0.0
            
        common = Counter(a_tokens) & Counter(b_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
            
        precision = num_common / len(b_tokens) if len(b_tokens) > 0 else 0
        recall = num_common / len(a_tokens) if len(a_tokens) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
            
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    # Calculate F1 scores for each reference
    f1_scores = []
    for reference in references:
        if not reference: # Handles cases where a reference might be None or empty
            # If prediction is also empty/None, consider it a perfect match for this empty ref (F1=1)
            # If prediction has content, it's a mismatch (F1=0)
            if not prediction or prediction == "[no answer generated]" or prediction == "I don't know" or prediction == "Sijui" or prediction == "నాకు తెలియదు":
                 f1_scores.append(1.0) # Both are effectively 'no answer'
            else:
                 f1_scores.append(0.0) # Ground truth is no answer, but model provided one
            continue
            
        # Normalize texts
        norm_reference = normalize_text(reference)
        norm_prediction = normalize_text(prediction)
        
        # Tokenize
        ref_tokens = norm_reference.split()
        pred_tokens = norm_prediction.split()
        
        # Calculate F1
        f1 = calculate_f1(ref_tokens, pred_tokens)
        f1_scores.append(f1)
    
    # Return the highest F1 score if there's any
    if f1_scores:
        return max(f1_scores)
    else: # This case should ideally not be hit if ground_truth always has at least one string.
          # If ground_truth was an empty list, and prediction is also empty, it's F1=1.
          # If ground_truth was empty list, and prediction has content, it's F1=0.
        if not prediction or prediction == "[no answer generated]" or prediction == "I don't know" or prediction == "Sijui" or prediction == "నాకు తెలియదు":
            return 1.0 
        else:
            return 0.0

def evaluate_qa_baseline(
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    use_few_shot: bool = True,
    temperature: float = 0.3, # Base temperature
    top_p: float = 0.9,       # Base top_p
    top_k: float = 40,        # Base top_k
    max_tokens: int = 50,     # Base max_tokens
    # Added repetition_penalty and do_sample to be passed to process_qa_baseline
    repetition_penalty: float = 1.2, 
    do_sample: bool = True
) -> pd.DataFrame:
    """
    Evaluate QA baseline approaches on a dataset.
    Passes base generation parameters to process_qa_baseline, which now handles dynamic adjustments.
    
    Args:
        model_name: Name of the model to use
        tokenizer: The model tokenizer
        model: The language model
        samples_df: DataFrame containing QA samples (must have 'question' and 'ground_truth')
        lang_code: Language code for language-specific handling
        use_few_shot: Whether to include few-shot examples in the prompt
        temperature: Base temperature for sampling
        top_p: Base top_p sampling parameter
        top_k: Base top_k sampling parameter
        max_tokens: Base maximum number of new tokens to generate
        repetition_penalty: Base repetition_penalty
        do_sample: Base do_sample flag
        
    Returns:
        DataFrame with predictions and evaluation metrics
    """
    # Process each sample
    results = []
    shot_type = "few-shot" if use_few_shot else "zero-shot"
    
    print(f"\nProcessing {len(samples_df)} samples with {shot_type} prompting...")
    # Add a counter for Telugu debug prints
    telugu_debug_count = 0
    max_telugu_debug_prints = 5

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df)):
        question = row["question"]
        ground_truth = row["ground_truth"] # Ensure this column name is used
        context = row.get("context", "") # Get context, default to empty string if not present
        
        # Dynamic parameter adjustment logic is now inside process_qa_baseline
        
        try:
            start_time = time.time()
            predicted_answer = process_qa_baseline(
                tokenizer, model, question, context=context, 
                temperature=temperature, 
                top_p=top_p,          
                top_k=top_k,                # Pass base top_k
                max_new_tokens=max_tokens, 
                repetition_penalty=repetition_penalty, # Pass base repetition_penalty
                do_sample=do_sample,            # Pass base do_sample
                lang_code=lang_code,
                model_name=model_name,
                use_few_shot=use_few_shot
            )
            runtime = time.time() - start_time
            
            f1_score_val = calculate_qa_f1(ground_truth, predicted_answer)
            
            result = {
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "f1_score": f1_score_val,
                "language": lang_code,
                "shot_type": shot_type,
                "runtime_seconds": runtime,
                "model": model_name
            }
            results.append(result)

            # Debug print for Telugu
            if lang_code == "te" and telugu_debug_count < max_telugu_debug_prints:
                print(f"\nDEBUG TELUGU (Sample {idx}, Model: {model_name}, Shot: {shot_type}):")
                print(f"  Question: {question}")
                print(f"  Ground Truth: {ground_truth}")
                print(f"  Predicted Answer: {predicted_answer}")
                print(f"  F1 Score: {f1_score_val:.4f}")
                telugu_debug_count += 1
            
        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": "[ERROR]",
                "f1_score": 0.0,
                "language": lang_code,
                "shot_type": shot_type,
                "runtime_seconds": 0.0,
                "model": model_name,
                "error": str(e)
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        avg_f1 = results_df["f1_score"].mean()
        avg_runtime = results_df["runtime_seconds"].mean()
        
        print(f"\nResults for {lang_code} with {model_name} ({shot_type}):")
        print(f"  Average F1 score: {avg_f1:.4f}")
        print(f"  Average runtime: {avg_runtime:.2f} seconds")
    else:
        print(f"No results generated for {lang_code} with {model_name}")
    
    return results_df