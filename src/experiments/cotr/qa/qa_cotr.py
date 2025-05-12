from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any, List
import pandas as pd
import time
from tqdm import tqdm
import re # Import re

# Import the COMET related utilities
from evaluation.cotr.qa_metrics_cotr import COMET_AVAILABLE, calculate_translation_quality

# Define language names dictionary at module level
lang_names = {
    "hi": "Hindi",
    "vi": "Vietnamese",
    "en": "English",
    "bn": "Bengali",
    "sw": "Swahili",
    "id": "Indonesian",
    "te": "Telugu", # Keep others just in case
    "fi": "Finnish",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "th": "Thai"
}

def initialize_model(model_name: str) -> tuple:
    """Initialize the model and tokenizer."""
    print(f"Loading model {model_name}...")
    cache_path = "/work/bbd6522/cache_dir" # Define cache path
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_path # Add cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_path # Add cache_dir
    )
    return model, tokenizer

def generate_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """Generate a prompt for translation with clear structure and output format."""
    # lang_names is now defined globally

    # Special handling for English to English "translation" (no-op)
    if source_lang == 'en' and target_lang == 'en':
        return text  # Simply return the original text for English->English

    # Get language names
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)

    # Enhanced specialized prompts for LRLs → English to improve translation quality
    if target_lang == 'en':
        if source_lang == 'sw':
            return f"""Original Text (Swahili):
'{text}'

Instructions:
Translate this Swahili text to English with maximum accuracy and fluency.
Preserve the full meaning, all named entities, numbers, dates, and any specific details critical for answering questions based on this text.
Provide ONLY the English translation.

English Translation:"""
        elif source_lang == 'te':
            return f"""Original Text (Telugu):
'{text}'

Instructions:
Translate this Telugu text to English with high precision and natural phrasing.
Keep all facts, named entities, numbers, dates, and technical information intact and accurately translated.
Maintain the original meaning without adding any interpretations or losing nuances crucial for later question answering.
Provide ONLY the English translation.

English Translation:"""
    
    # Enhanced specialized prompts for English → LRLs to improve back-translation
    if source_lang == 'en':
        if target_lang == 'sw':
            return f"""Original Text (English):
'{text}'

Instructions:
Translate this English text to natural, fluent, and grammatically correct Swahili.
Preserve all factual content, names, and numerical information exactly.
For yes/no answers, use the standard Swahili equivalent (e.g., 'Ndiyo' or 'Hapana').
Provide ONLY the Swahili translation.

Swahili Translation:"""
        elif target_lang == 'te':
            return f"""Original Text (English):
'{text}'

Instructions:
Translate this English text to accurate and natural-sounding Telugu using standard script.
Maintain all factual information, especially names and numbers, with precision.
For yes/no answers, use the standard Telugu equivalent (e.g., 'అవును' or 'లేదు').
Provide ONLY the Telugu translation.

Telugu Translation:"""

    # Default/original prompt for other language pairs or directions
    return f"""Original Text ({source_name}):
'{text}'

Instructions:
Translate this {source_name} text to {target_name} accurately and clearly.
Preserve all important information, especially names, dates, and numerical values.
Maintain the exact meaning without adding explanations or extra context.
Provide only the direct translation.

{target_name} Translation:"""

def generate_qa_prompt(question: str, context_en: str, use_few_shot: bool = True) -> str:
    """Generate the QA prompt (for English QA) for the model with explicit instructions
    to use the provided context.
    
    Args:
        question: The English question
        context_en: The English context
        use_few_shot: Whether to include few-shot examples
        
    Returns:
        Formatted English QA prompt
    """
    # Context-based few-shot examples
    examples = """
Examples:
Context: 'Paris is the capital and most populous city of France. It is located in the north-central part of the country.'
Question: 'What is the capital of France?'
Answer: Paris

Context: 'The Eiffel Tower was completed in 1889 for the Exposition Universelle. It is located on the Champ de Mars in Paris.'
Question: 'When was the Eiffel Tower finished?'
Answer: 1889

Context: 'Mount Kilimanjaro is a dormant volcano in Tanzania. It has three volcanic cones: Kibo, Mawenzi, and Shira.'
Question: 'What is the highest mountain in Africa?'
Answer: Mount Kilimanjaro

Context: 'The Great Barrier Reef is the world\'s largest coral reef system located off the coast of Queensland, Australia.'
Question: 'Is the Great Barrier Reef in New Zealand?'
Answer: No

Context: 'The Nile is the longest river in Africa and flows through northeastern Africa.'
Question: 'Which continent is the Nile in?'
Answer: Africa
"""
    
    # Enhanced prompt focusing on using ONLY the provided context
    prompt_base = f"""Context: '{context_en}'
Question: '{question}'

Answer the question accurately and very concisely using ONLY the information provided in the context above.
Follow these rules strictly:
- For yes/no questions: respond with ONLY 'Yes' or 'No'.
- For factual questions: provide ONLY the specific answer found in the context.
- Do NOT add any explanations, disclaimers, or conversational phrases like "The answer is..." or "Based on the context...".
- Your entire response should ideally be a short phrase or name extracted directly from the context.
- If the answer cannot be found in the context, respond with ONLY the exact phrase 'I don't know'.
"""

    if use_few_shot:
        prompt = f"{prompt_base}\n\n{examples}\nAnswer:"
    else:
        prompt = f"{prompt_base}\nAnswer:"
        
    return prompt

def generate_single_prompt_qa_cotr(lrl_question: str, lrl_context: str, lang_code: str, use_few_shot: bool = True) -> str:
    """Generate a single prompt for the entire CoTR QA pipeline, now including context.
    Instructs the model to translate LRL->EN (question+context), answer in EN using context, translate EN->LRL answer.
    
    Args:
        lrl_question: The question in the Low-Resource Language
        lrl_context: The context in the Low-Resource Language
        lang_code: The language code (e.g., 'sw', 'te')
        use_few_shot: Whether to include few-shot examples demonstrating the process
        
    Returns:
        Formatted single-prompt CoTR QA prompt
    """
    global lang_names
    lrl_name = lang_names.get(lang_code, lang_code)
    
    # Enhanced base instructions including context
    instructions = f"""Your task is to answer the following question asked in {lrl_name}, based ONLY on the provided {lrl_name} context.

Follow these steps exactly in sequence:
1. Translate the {lrl_name} context into English accurately.
2. Translate the {lrl_name} question into English accurately.
3. Answer the translated English question using ONLY the information from the translated English context. Keep your answer concise. If the answer is not in the context, state 'I don't know'.
4. Translate the English answer back into {lrl_name}.
5. Provide ONLY the final {lrl_name} answer as your response. Do not include any intermediate steps, explanations, or other text.

Context ({lrl_name}): '{lrl_context}'
Question ({lrl_name}): '{lrl_question}'"""

    # Improved few-shot examples demonstrating the process with context
    examples = ""
    if use_few_shot:
        if lang_code == 'sw':
            examples = f"""Examples:

Context (Swahili): 'Paris ndio mji mkuu na jiji lenye watu wengi zaidi Ufaransa. Unapatikana sehemu ya kaskazini-kati ya nchi.'
Question (Swahili): 'Mji mkuu wa Ufaransa ni upi?'
Step 1 (Translate Context): Paris is the capital and most populous city of France. It is located in the north-central part of the country.
Step 2 (Translate Question): What is the capital of France?
Step 3 (Answer from Context): Paris
Step 4 (Translate Answer to Swahili): Paris
Final Answer (Swahili): Paris

Context (Swahili): 'Mlima Kilimanjaro ni volkano iliyolala nchini Tanzania. Una koni tatu za volkano: Kibo, Mawenzi, na Shira.'
Question (Swahili): 'Mlima mrefu zaidi Afrika ni upi?'
Step 1 (Translate Context): Mount Kilimanjaro is a dormant volcano in Tanzania. It has three volcanic cones: Kibo, Mawenzi, and Shira.
Step 2 (Translate Question): What is the highest mountain in Africa?
Step 3 (Answer from Context): Mount Kilimanjaro
Step 4 (Translate Answer to Swahili): Mlima Kilimanjaro
Final Answer (Swahili): Mlima Kilimanjaro
"""
        elif lang_code == 'te':
            examples = f"""Examples:

Context (Telugu): 'పారిస్ ఫ్రాన్స్ రాజధాని మరియు అత్యధిక జనాభా కలిగిన నగరం. ఇది దేశం యొక్క ఉత్తర-మధ్య భాగంలో ఉంది.'
Question (Telugu): 'ఫ్రాన్స్ రాజధాని ఏది?'
Step 1 (Translate Context): Paris is the capital and most populous city of France. It is located in the north-central part of the country.
Step 2 (Translate Question): What is the capital of France?
Step 3 (Answer from Context): Paris
Step 4 (Translate Answer to Telugu): పారిస్
Final Answer (Telugu): పారిస్

Context (Telugu): 'కిలిమంజారో పర్వతం టాంజానియాలో ఒక నిద్రాణమైన అగ్నిపర్వతం. దీనికి మూడు అగ్నిపర్వత శంఖాలు ఉన్నాయి: కిబో, మావెంజి మరియు షిరా.'
Question (Telugu): 'ఆఫ్రికాలో ఎత్తైన పర్వతం ఏది?'
Step 1 (Translate Context): Mount Kilimanjaro is a dormant volcano in Tanzania. It has three volcanic cones: Kibo, Mawenzi, and Shira.
Step 2 (Translate Question): What is the highest mountain in Africa?
Step 3 (Answer from Context): Mount Kilimanjaro
Step 4 (Translate Answer to Telugu): కిలిమంజారో పర్వతం
Final Answer (Telugu): కిలిమంజారో పర్వతం
"""
        # Add other languages here if needed
        
    if use_few_shot and examples:
        prompt = f"{instructions}\n\n{examples}\n\nFinal Answer ({lrl_name}):"
    else:
        prompt = f"{instructions}\n\nFinal Answer ({lrl_name}):"
        
    return prompt

def translate_text(
    model: Any,
    tokenizer: Any,
    text: str,
    source_lang: str,
    target_lang: str,
    max_input_length: int = 4096,
    max_new_tokens: int = 200, # Max tokens for the translated output
    temperature: float = 0.7, # Higher temperature for translation (default)
    top_p: float = 0.9,       # Standard top_p
    model_name: str = ""      # Added model name for model-specific adjustments
) -> str:
    """Translate text from source language to target language using the model.
    Uses more standardized parameters for consistent scientific evaluation.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        max_input_length: Maximum input length
        max_new_tokens: Maximum new tokens for translation
        temperature: Temperature for generation
        top_p: Top-p for generation
        model_name: Model name for potential adjustments
        
    Returns:
        Translated text
    """
    # Special handling for English to English "translation" (no-op)
    if source_lang == 'en' and target_lang == 'en':
        return text  # Simply return the original text for English->English
        
    # --- Direct Handling for Yes/No Back-Translation (e.g., English -> Telugu) ---
    if source_lang == 'en':
        # More comprehensive handling of yes/no cases for cleaner answers
        if text.strip().lower() in ["yes", "true", "correct", "right", "affirmative"]:
            if target_lang == 'te':
                return "అవును"  # "Avunu" in Telugu
            elif target_lang == 'sw':
                return "Ndiyo"   # Swahili affirmative
        elif text.strip().lower() in ["no", "false", "incorrect", "wrong", "negative"]:
            if target_lang == 'te':
                return "లేదు"   # "Ledu" in Telugu
            elif target_lang == 'sw':
                return "Hapana" # Swahili negative

    # Handle empty or whitespace-only text
    if not text or text.strip() == "":
        return ""

    # --- Proceed with Model-based Translation if not handled above ---
    prompt = generate_translation_prompt(text, source_lang, target_lang)
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Use standardized parameters with small adjustments for translation
    # temperature and top_p are passed in and should be used directly.
    # repetition_penalty is managed locally here.
    # gen_temperature = temperature  # No longer re-assign
    # gen_top_p = top_p              # No longer re-assign
    repetition_penalty = 1.2 # Default repetition penalty
    
    # Model-specific adjustments that match baseline approach
    # is_aya_model = "aya" in model_name.lower() # Not needed for temp adjustment anymore
    is_qwen_model = "qwen" in model_name.lower()
    
    # if is_aya_model:
    #     # Aya adjustments - slightly lower temperature - REMOVE THIS BLOCK
    #     gen_temperature *= 0.9 
    if is_qwen_model:
        # Qwen adjustments - slightly higher repetition penalty
        repetition_penalty *= 1.1
    
    # For translation quality, use a single consistent approach
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature, # Use temperature argument directly
            top_p=top_p,             # Use top_p argument directly
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    # Clean up the response
    translation = clean_translation_response(response, target_lang, source_lang)
    
    return translation

def clean_translation_response(response: str, target_lang: str, source_lang: str) -> str:
    """Clean up the translation response.
    
    Args:
        response: Raw translation from the model
        target_lang: Target language code
        source_lang: Source language code
    
    Returns:
        Cleaned translation
    """
    # Handle empty responses
    if not response or response.strip() == "":
        return ""
    
    # Clean up whitespace and quotes
    response = response.strip()
    response = response.strip('"\'')  # Remove surrounding quotes
    
    # Remove common prefixes
    prefixes_to_remove = [
        "Translation:", 
        "The translation is:", 
        "Translated text:", 
        "Here's the translation:", 
        "English translation:", 
        "Swahili translation:", 
        "Telugu translation:"
    ]
    
    # Add language-specific prefixes to remove
    if target_lang == 'sw':
        prefixes_to_remove.extend([
            "Tafsiri:", 
            "Tafsiri ni:", 
            "Matini iliyotafsiriwa:"
        ])
    elif target_lang == 'te':
        prefixes_to_remove.extend([
            "అనువాదం:", 
            "తెలుగు అనువాదం:", 
            "అనువదించిన పాఠ్యం:"
        ])
    
    # Remove any matching prefix
    for prefix in prefixes_to_remove:
        if response.lower().startswith(prefix.lower()):
            response = response[len(prefix):].strip()
    
    # Handle language-specific cleaning
    if target_lang == 'te':
        # Telugu-specific cleaning
        # If response mixes Telugu and English, try to extract just the Telugu part
        if any(ord(c) >= 128 for c in response) and any(c.isascii() and c.isalpha() for c in response):
            # Find where Telugu script begins (first Telugu character)
            telugu_match = re.search(r'[\u0C00-\u0C7F]', response)
            if telugu_match:
                # Extract from first Telugu character
                start_idx = telugu_match.start()
                telugu_text = response[start_idx:]
                
                # Find where Telugu ends (if there's more English after it)
                end_match = re.search(r'[a-zA-Z]', telugu_text)
                if end_match:
                    # Keep only the Telugu portion
                    telugu_text = telugu_text[:end_match.start()].strip()
                
                # Only use the extracted Telugu if it's substantial
                if len(telugu_text) > 0.3 * len(response):
                    response = telugu_text.strip()
    
    elif target_lang == 'sw':
        # Swahili-specific cleaning
        # Remove common Swahili explanations that follow the translation
        if '\n' in response:
            # Take only the first line if multiple lines
            response = response.split('\n')[0].strip()
        
        # Remove explanatory notes that sometimes appear after "--" or similar
        for separator in ['--', '—', '-', ':', ';']:
            if separator in response:
                response = response.split(separator)[0].strip()
    
    # Remove leading/trailing quotes again (after prefix removal)
    response = response.strip('"\'')
    
    # Remove parenthetical clarifications like "(in English)" or similar
    response = re.sub(r'\([^)]*\)', '', response).strip()
    
    # Handle responses that contain instructions or explanations after the translation
    if "\n\n" in response:
        # If there are double newlines, usually the translation is before them
        response = response.split("\n\n")[0].strip()
    
    # For very long responses, try to extract just the initial part
    if len(response) > 200:  # arbitrary threshold
        # Look for sentence boundaries in the first part of the response
        first_sentence_match = re.search(r'^(.*?[.?!])\s', response[:150])
        if first_sentence_match:
            # Use just the first sentence if it's substantial
            first_sentence = first_sentence_match.group(1)
            if len(first_sentence) > 10:  # Avoid tiny responses
                response = first_sentence
    
    return response

def process_qa_pair(
    model: Any,
    tokenizer: Any,
    question: str,
    context_en: str,
    max_input_length: int = 4096,
    max_new_tokens: int = 50,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: float = 40,
    model_name: str = "",
    use_few_shot: bool = True 
) -> str:
    """Process a QA pair with the model in English, now using context."""
    try:
        # Generate enhanced prompt for QA using context
        prompt = generate_qa_prompt(question, context_en, use_few_shot=use_few_shot)
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Parameters for generation - repetition_penalty is managed locally here.
        # temperature, top_p, top_k are passed in and assumed to be final.
        gen_repetition_penalty = 1.2 
        gen_do_sample = True # Assuming we usually want to sample if temp > 0
        
        is_yes_no = is_yes_no_question(question) # Determine if it's a yes/no question
        
        # Model-specific adjustments for temperature, top_p, top_k are REMOVED here.
        # They are expected to be handled by the calling function (e.g., run_experiment in run_qa_cotr.py).
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=gen_do_sample,
                temperature=temperature, # Use temperature argument directly
                top_p=top_p,             # Use top_p argument directly
                top_k=top_k,             # Use top_k argument directly
                repetition_penalty=gen_repetition_penalty,
                pad_token_id=tokenizer.eos_token_id
            )
        
        output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Extract and clean the answer
        answer = extract_answer(output_text, is_yes_no=is_yes_no) # Pass is_yes_no
        
        return answer
    
    except Exception as e:
        import traceback
        print(f"Error in process_qa_pair: {str(e)}")
        traceback.print_exc()
        return "[ERROR in QA processing]"

def analyze_question_type(question: str) -> str:
    """Analyze the question to determine its type for parameter optimization.
    
    Args:
        question: The question text
        
    Returns:
        Question type category (yes_no, factoid, entity, quantity, etc.)
    """
    question_lower = question.lower().strip()
    
    # Yes/no questions (already handled by is_yes_no_question)
    if is_yes_no_question(question_lower):
        return "yes_no"
    
    # Check for "wh" question words and patterns
    who_pattern = r'\b(who|whose)\b'
    what_pattern = r'\b(what|which)\b'
    where_pattern = r'\bwhere\b'
    when_pattern = r'\bwhen\b'
    how_pattern = r'\bhow\b'
    how_many_pattern = r'\bhow many\b'
    how_much_pattern = r'\bhow much\b'
    
    # Entity questions (usually who, which person/organization)
    if re.search(who_pattern, question_lower) or "name the person" in question_lower:
        return "entity"
    
    # Location questions
    if re.search(where_pattern, question_lower) or "location" in question_lower or "place" in question_lower:
        return "entity"
    
    # Temporal questions
    if re.search(when_pattern, question_lower) or "date" in question_lower or "year" in question_lower:
        return "factoid"
    
    # Quantity questions
    if re.search(how_many_pattern, question_lower) or re.search(how_much_pattern, question_lower) or "number of" in question_lower:
        return "quantity"
    
    # General "how" questions that aren't about quantity
    if re.search(how_pattern, question_lower) and not re.search(how_many_pattern, question_lower) and not re.search(how_much_pattern, question_lower):
        return "factoid"
    
    # Definition/explanation questions (what is, what are)
    if re.search(r'\bwhat (is|are)\b', question_lower) or "define" in question_lower:
        return "definition"
    
    # General "what" and "which" questions
    if re.search(what_pattern, question_lower):
        return "factoid"
    
    # Default case
    return "factoid"

def is_yes_no_question(question: str) -> bool:
    """Determine if a question is a yes/no question."""
    question_lower = question.lower()
    
    # Check for common question starters that indicate yes/no questions
    yes_no_starters = [
        "is ", "are ", "was ", "were ", "will ", "would ", "can ", "could ", 
        "does ", "do ", "did ", "has ", "have ", "had ", "should ", "shall "
    ]
    
    for starter in yes_no_starters:
        if question_lower.startswith(starter):
            return True
            
    return False

def extract_answer(response: str, is_yes_no: bool = False) -> str:
    """Extract the answer from model response, cleaning up extra text.
    
    Args:
        response: Model's raw response text
        is_yes_no: Whether this is a yes/no question
        
    Returns:
        Cleaned, extracted answer
    """
    if not response or response.strip() == "":
        return "[No answer generated]"
    
    # Basic cleaning
    answer = response.strip()
    
    # Normalize common "I don't know" phrases to a standard one FIRST
    i_dont_know_patterns = [
        r"^i don't know\.?$",
        r"^i do not know\.?$",
        r"^sorry, i don't know\.?$",
        r"^i am unable to answer\.?$",
        r"^i cannot answer that\.?$",
        r"^i don't have that information\.?$",
        r"^i am not sure\.?$",
        r"^unknown\.?$"
    ]
    for pattern in i_dont_know_patterns:
        if re.match(pattern, answer, re.IGNORECASE):
            return "I don't know" # Standardize
    
    # Remove common prefixes systematically (expanded list)
    prefixes_to_remove = [
        "answer:", "the answer is:", "answer is:",
        "i believe the answer is:", "the answer to this question is:",
        "based on my knowledge,", "according to my knowledge,",
        "based on the information,", "based on the given information,",
        "the answer would be:", "i would say", "i think", "i'd say",
        "the correct answer is:", "correct answer:", "my answer is:",
        "here is the answer:", "the response is:", "the result is:",
        "to answer this question,", "in response to your question,",
        "to summarize,", "in summary,"
    ]
    
    # Try case-insensitive prefix removal (more robust)
    answer_lower = answer.lower()
    for prefix in prefixes_to_remove:
        if answer_lower.startswith(prefix.lower()):
            # Remove the prefix using the exact case from the original string
            prefix_len = len(prefix)
            answer = answer[prefix_len:].strip()
            # After removing prefix, clean up any punctuation residue
            if answer.startswith(':') or answer.startswith(','):
                answer = answer[1:].strip()
    
    # Remove any quotes surrounding the answer
    answer = answer.strip('"\'')
    
    # Special handling for yes/no questions
    if is_yes_no:
        # More comprehensive yes/no detection with stronger bias toward binary answers
        yes_indicators = ["yes", "yeah", "correct", "right", "true", "affirmative", "certainly", "definitely"]
        no_indicators = ["no", "nope", "not", "incorrect", "wrong", "false", "negative", "untrue"]
        
        # First check for exact matches (stronger signal)
        answer_lower = answer.lower()
        if answer_lower in yes_indicators:
            return "Yes"
        if answer_lower in no_indicators:
            return "No"
        
        # Then check for pattern matches (weaker signal)
        if any(re.search(r'\b' + re.escape(indicator) + r'\b', answer_lower) for indicator in yes_indicators):
            return "Yes"
        if any(re.search(r'\b' + re.escape(indicator) + r'\b', answer_lower) for indicator in no_indicators):
            return "No"
        
        # If the answer contains "yes" and "no" (e.g., "yes and no"), check which comes first
        yes_pos = answer_lower.find("yes")
        no_pos = answer_lower.find("no")
        if yes_pos >= 0 and no_pos >= 0:
            return "Yes" if yes_pos < no_pos else "No"
    
    # Remove notes/citations/qualifiers in parentheses and brackets
    answer = re.sub(r'\([^)]*\)', '', answer)
    answer = re.sub(r'\[[^]]*\]', '', answer)
    
    # Split on first obvious sentence boundary for long answers
    if len(answer) > 30:  # Only for longer answers
        first_sent_match = re.search(r'^(.*?[.?!])(?:\s|$)', answer)
        if first_sent_match:
            first_sent = first_sent_match.group(1).strip()
            # Only use the first sentence if it's substantial
            if len(first_sent) > 5:  # Must be at least 5 chars to be meaningful
                answer = first_sent
    
    # Split at coordinating conjunctions for long answers to get main point
    if len(answer) > 30:
        for conj in [", however", ", but", ", although", ", though", ", yet"]:
            if conj in answer.lower():
                answer = answer.split(conj)[0].strip()
                break
    
    # If answer contains multiple lines, take just the first line
    if "\n" in answer:
        answer = answer.split("\n")[0].strip()
    
    # For long answers with commas, often the key part is before the first comma
    if len(answer) > 30 and "," in answer and not is_yes_no:
        # Look for common continuation phrases after a comma
        continuations = ["which", "who", "where", "when", "because", "since", "as", "but", "however"]
        next_part = answer.split(",", 1)[1].strip().lower()
        if any(next_part.startswith(cont) for cont in continuations):
            # If there's an explanation after the comma, take only what's before
            answer = answer.split(",")[0].strip()
        
    # Remove explanations that start with common indicators
    explanation_starters = [
        "because", "since", "which", "this is", "as it", "that is",
        "meaning", "referring to", "specifically", "given that", 
        "considering", "this refers to", "this means", "more specifically",
        "in other words"
    ]
    for starter in explanation_starters:
        pattern = r'\b' + re.escape(starter) + r'\b'
        match = re.search(pattern, answer.lower())
        if match and match.start() > 10:  # Only split if explanation is not at very start
            answer = answer[:match.start()].strip()
    
    # If the answer is still very long, truncate it more intelligently
    if len(answer) > 50:
        # Try to find a clean break point
        last_period = answer[:50].rfind('.')
        last_question = answer[:50].rfind('?')
        last_exclamation = answer[:50].rfind('!')
        
        # Find the latest sentence break
        sentence_end = max(last_period, last_question, last_exclamation)
        
        if sentence_end > 0:
            # Break at sentence boundary
            answer = answer[:sentence_end+1]
        else:
            # Try comma boundary
            last_comma = answer[:50].rfind(',')
            if last_comma > 0:
                answer = answer[:last_comma]
            else:
                # Try space boundary (only if reasonably far in)
                last_space = answer[:50].rfind(' ')
                if last_space > 30:
                    answer = answer[:last_space]
                else:
                    # Last resort: hard truncate
                    answer = answer[:50] + "..."
    
    # Final cleanup
    answer = answer.strip().strip('.,:;-–—')  # Remove trailing punctuation
    
    # Final check for empty answers
    if not answer:
        answer = "[No answer generated]"
    
    return answer

def evaluate_qa_cotr(
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    batch_size: int = 1,
    use_few_shot: bool = True,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: float = 40,
    max_tokens: int = 50,
    max_translation_tokens: int = 200
) -> pd.DataFrame:
    """
    Evaluate QA using Chain of Translation Prompting (CoTR) approach - MULTI-PROMPT.
    Now includes context translation and usage.
    
    Steps:
    1. Translate question from source language to English
    2. Translate context from source language to English
    3. Perform QA in English using translated context
    4. Translate the answer back to the source language
    
    Args:
        model_name: Name of the model to use
        tokenizer: Initialized tokenizer
        model: Initialized model
        samples_df: DataFrame containing the samples
        lang_code: Language code (e.g., 'bn' for Bengali, 'sw' for Swahili)
        batch_size: Number of samples to process at once (currently unused)
        use_few_shot: Whether to use few-shot examples for the English QA step
        temperature: Temperature for generation (using standardized parameter)
        top_p: Top-p for generation (using standardized parameter)
        top_k: Top-k for generation (using standardized parameter)
        max_tokens: Maximum tokens for answer generation (using standardized parameter)
        max_translation_tokens: Maximum tokens for translation
        
    Returns:
        DataFrame with results including all translations and answers, or empty DF on error
    """
    results = []
    
    model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
    safe_max_input_length = min(model_max_len, 8192) 
    print(f"Using max_input_length: {safe_max_input_length}")
    print(f"NOTE: Now using context for QA step.")

    qa_temperature = temperature  
    qa_top_p = top_p         
    qa_top_k = top_k          
    qa_max_tokens = max_tokens
    is_aya_model = "aya" in model_name.lower()
    is_qwen_model = "qwen" in model_name.lower()
    if is_aya_model:
        qa_temperature = max(0.1, qa_temperature - 0.05)
        print(f"Applied Aya-specific adjustment to temperature: {qa_temperature}")
    elif is_qwen_model:
        qa_top_p = max(0.7, qa_top_p - 0.05)
        qa_top_k = 35
        print(f"Applied Qwen-specific adjustments: top_p={qa_top_p}, top_k={qa_top_k}")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} samples ({model_name} CoTR)"):
        try:
            original_question = row['question']
            original_context = row.get('context', '')
            
            if "ground_truth" in row and row["ground_truth"]:
                ground_truth_answers_list = [row["ground_truth"]]
            else:
                ground_truth_answers_list = row.get("answers", {}).get("text", [])
            if not ground_truth_answers_list:
                 print(f"Skipping sample {row.get('id', idx)} due to missing ground truth answers.")
                 continue
            ground_truth_lrl_for_comet = ground_truth_answers_list[0]
            
            question_en = translate_text(model, tokenizer, original_question, lang_code, "en", 
                                       max_input_length=safe_max_input_length, 
                                       max_new_tokens=max_translation_tokens,
                                       temperature=qa_temperature,
                                       top_p=qa_top_p,
                                       model_name=model_name)
            
            context_en = translate_text(model, tokenizer, original_context, lang_code, "en", 
                                      max_input_length=safe_max_input_length, 
                                      max_new_tokens=max_translation_tokens * 2,
                                      temperature=qa_temperature,
                                      top_p=qa_top_p,
                                      model_name=model_name)

            answer_en = process_qa_pair(model, tokenizer, question_en, context_en, 
                                       max_input_length=safe_max_input_length,
                                       max_new_tokens=qa_max_tokens,
                                       temperature=qa_temperature,
                                       top_p=qa_top_p,
                                       top_k=qa_top_k,
                                       model_name=model_name,
                                       use_few_shot=use_few_shot)
            
            if answer_en.lower() in ["yes", "no"]:
                answer_en = "Yes" if answer_en.lower() == "yes" else "No"
            
            predicted_lrl_answer = translate_text(model, tokenizer, answer_en, "en", lang_code, 
                                   max_input_length=safe_max_input_length,
                                   max_new_tokens=max_translation_tokens,
                                   temperature=qa_temperature,
                                   top_p=qa_top_p,
                                   model_name=model_name)
            
            comet_score_en_to_lrl = None
            if COMET_AVAILABLE:
                try:
                    if predicted_lrl_answer and ground_truth_lrl_for_comet and answer_en:
                        comet_results = calculate_translation_quality(
                            predictions=[predicted_lrl_answer],
                            references=[[ground_truth_lrl_for_comet]],
                            sources=[answer_en]
                        )
                        if comet_results and isinstance(comet_results, list) and len(comet_results) > 0:
                            comet_score_en_to_lrl = comet_results[0]
                        elif isinstance(comet_results, float):
                            comet_score_en_to_lrl = comet_results
                    else:
                        print(f"Skipping COMET for sample {row.get('id', idx)} due to empty prediction, reference, or source.")
                except Exception as e:
                    print(f"Error calculating COMET score for sample {row.get('id', idx)}: {e}")
            
            result = {
                'question': original_question,
                'context': original_context[:200] + "..." if len(original_context or '') > 200 else original_context,
                'question_en': question_en,
                'context_en': context_en,
                'answer_en': answer_en,
                'predicted_answer': predicted_lrl_answer,
                'ground_truth': ground_truth_lrl_for_comet,
                'all_ground_truths': ground_truth_answers_list,
                'comet_score_en_to_lrl': comet_score_en_to_lrl,
                'context_used': True,
                'pipeline': 'multi_prompt',
                'few_shot': use_few_shot,
                'temperature': qa_temperature,
                'top_p': qa_top_p,
                'top_k': qa_top_k,
                'max_tokens': qa_max_tokens
            }
            results.append(result)

        except Exception as e:
            print(f"ERROR processing sample {row.get('id', idx)}: {e}")
            continue

    if not results:
        print(f"WARNING: No results were successfully processed for {lang_code} with {model_name} using CoTR.")
        return pd.DataFrame()
        
    return pd.DataFrame(results)

def is_text_english(text: str) -> bool:
    """
    Simple heuristic to check if text is primarily English.
    Useful for determining if we need to translate the context or not.
    """
    # Count ASCII alphabetic characters
    ascii_count = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    # Count total alphabetic characters
    total_alpha = sum(1 for c in text if c.isalpha())
    
    # If the text has no alphabetic characters, consider it as non-English
    if total_alpha == 0:
        return False
        
    # If more than 85% of alphabetic chars are ASCII, it's likely English
    return ascii_count / total_alpha > 0.85

def process_english_qa(model, tokenizer, translated_question, context):
    """
    Process a QA pair in English.
    
    Args:
        model: The transformer model to use
        tokenizer: The tokenizer for the model
        translated_question: The question translated to English
        context: The context text
        
    Returns:
        dict: A dictionary containing the answer and full output
    """
    # Generate prompt for QA in English
    qa_prompt = f"""Answer the following question based only on the provided context.
Context: {context}

Question: {translated_question}

Answer:"""
    
    # Get model response
    try:
        # Tokenize with truncation
        inputs = tokenizer(qa_prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200, 
                do_sample=True,
                temperature=0.5,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        full_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        answer = full_output.strip()
        
        # Simple post-processing to clean the answer
        answer = re.sub(r'^answer:', '', answer, flags=re.IGNORECASE).strip()
        
        return {"answer": answer, "full_output": full_output}
    except Exception as e:
        print(f"Error in process_english_qa: {e}")
        return {"answer": "[Error generating answer]", "full_output": ""}

def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    """
    Generate a response from the model based on the given prompt.
    
    Args:
        model: The transformer model to use
        tokenizer: The tokenizer for the model
        prompt: The prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        
    Returns:
        The generated response text
    """
    try:
        # Tokenize with truncation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        print(f"Error in generate_response: {e}")
        return "[Error generating response]"

# Using a different name for the single-sample evaluation function
def evaluate_qa_cotr_sample(model, tokenizer, sample, lang_code="sw"):
    """
    Evaluate the CoTR model on a single QA sample.
    This is a two-step process:
    1. Translate the LRL question to English
    2. Answer the question in English based on the context
    
    Args:
        model: The transformer model to use
        tokenizer: The tokenizer for the model
        sample: A dictionary containing 'question', 'context', and 'answer_text'
        lang_code: Language code of the original question
    
    Returns:
        dict: A dictionary containing the original question, original answer,
              translated question, model's output, predicted answer, and metrics
    """
    # Collect input info
    lrl_question = sample['question']
    # For TyDiQA, context is already in English
    context = sample['context']
    # Get the ground truth answer - use LRL answers for evaluation
    ground_truth = sample['answer_text']  # Using LRL ground truth from TyDiQA
    
    if not lrl_question or not context or not ground_truth:
        print("Missing required fields in sample.")
        return None
    
    # Step 1: Translate from LRL to English
    translate_prompt = f"""Translate the following {get_language_name(lang_code)} question to English:
{lrl_question}

English translation:"""
    
    try:
        # Get English translation of the question
        translation_response = generate_response(model, tokenizer, translate_prompt)
        translated_question = translation_response.strip()
        
        # Step 2: Answer in English
        qa_result = process_english_qa(model, tokenizer, translated_question, context)
        predicted_answer = qa_result["answer"]
        
        # Collect results - include both the LRL ground truth for native evaluation
        result = {
            "original_question": lrl_question,
            "original_context": context,
            "original_answer": ground_truth,  # LRL ground truth
            "translated_question": translated_question,
            "model_output": qa_result["full_output"],
            "predicted_answer": predicted_answer
        }
        
        return result
        
    except Exception as e:
        print(f"Error in QA CoTR evaluation: {e}")
        return None

def get_language_name(lang_code):
    """Get full language name from language code."""
    language_mapping = {
        "sw": "Swahili",
        "te": "Telugu",
        "en": "English"
        # Add more languages as needed
    }
    return language_mapping.get(lang_code, "unknown language")

# --- New Single-Prompt Evaluation Function --- 
def evaluate_qa_cotr_single_prompt(
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    use_few_shot: bool = True,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: float = 40,
    max_tokens: int = 50 # Note: This now applies to the final LRL answer extraction, not the whole chain
) -> pd.DataFrame:
    """
    Evaluate QA using a single CoTR prompt with standardized parameters.
    Now includes context and improved answer extraction.
    """
    # --- DEBUG ---
    print(f"\n--- DEBUG: ENTERING evaluate_qa_cotr_single_prompt ---")
    print(f"Model: {model_name}, Lang: {lang_code}, FewShot: {use_few_shot}")
    print(f"Params: temp={temperature}, top_p={top_p}, top_k={top_k}, max_tokens={max_tokens}")
    # --- END DEBUG ---
    results = []
    
    model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
    safe_max_input_length = min(model_max_len, 8192) 
    # Increase max_new_tokens for the whole chain generation significantly
    # Adjusted heuristic: Make it larger but maybe not *10
    max_new_tokens_for_chain = max(512, max_tokens * 6 + 300) # Ensure minimum length, scale with steps
    print(f"DEBUG: Calculated max_new_tokens_for_chain: {max_new_tokens_for_chain}") # DEBUG
    
    # gen_temperature = temperature # No longer re-assign
    # gen_top_p = top_p             # No longer re-assign
    # gen_top_k = top_k             # No longer re-assign
    gen_repetition_penalty = 1.2 # This can remain if not passed from caller
    
    # Model-specific adjustments are now assumed to be handled by the caller (run_experiment)
    # is_aya_model = "aya" in model_name.lower()
    # is_qwen_model = "qwen" in model_name.lower()
    # if is_aya_model:
    #     gen_temperature = max(0.1, gen_temperature - 0.05)
    #     print(f"Applied Aya-specific adjustment to temperature: {gen_temperature}")
    # elif is_qwen_model:
    #     gen_top_p = max(0.7, gen_top_p - 0.05)
    #     gen_top_k = 35
    #     print(f"Applied Qwen-specific adjustments: top_p={gen_top_p}, top_k={gen_top_k}")

    # Define regex to capture the final LRL answer from structured output
    lrl_name_pattern = re.escape(lang_names.get(lang_code, lang_code)) # Escape potential special chars
    # Refined Regex: Look for 'Final Answer (<Lang>):' possibly with variations in spacing/case,
    # capture everything after it until the end of the string or maybe a double newline?
    # Making it non-greedy and allowing for more flexible endings.
    final_answer_regex = re.compile(rf"Final Answer\s*\({lrl_name_pattern}\)\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
    print(f"DEBUG: Using regex pattern: {final_answer_regex.pattern}") # DEBUG

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} samples ({model_name} CoTR Single Prompt)"):
        try:
            original_question = row['question']
            original_context = row.get('context', '') # Get context
            
            if "ground_truth" in row and row["ground_truth"]:
                ground_truth = [row["ground_truth"]]
            else:
                ground_truth = row.get("answers", {}).get("text", [])
            if not ground_truth:
                 print(f"Skipping sample {row.get('id', idx)} due to missing ground truth answers.")
                 continue
            original_ground_truth_str = ground_truth[0]
            
            # Generate the single CoTR prompt with context
            prompt = generate_single_prompt_qa_cotr(original_question, original_context, lang_code, use_few_shot)
            # DEBUG: Print prompt for first sample
            if idx == 0:
                print(f"\nDEBUG: Example Single Prompt (Sample 0):\n{prompt[:500]}...\n")

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=safe_max_input_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens_for_chain, # Use larger limit for the whole chain
                    do_sample=True,
                    temperature=temperature, # Use temperature argument directly
                    top_p=top_p,             # Use top_p argument directly
                    top_k=top_k,             # Use top_k argument directly
                    repetition_penalty=gen_repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            # DEBUG: Print raw response for first few samples
            if idx < 2:
                print(f"\nDEBUG: Raw Model Response (Sample {idx}):\n>>>n{response_text}n<<<n")

            # Extract final LRL answer using regex
            match = final_answer_regex.search(response_text)
            if match:
                predicted_lrl_answer = match.group(1).strip()
                # Further clean potential artifacts if needed
                predicted_lrl_answer = predicted_lrl_answer.split('\n')[0].strip() # Take only first line after label
                print(f"DEBUG (Sample {idx}): Extracted answer via regex: '{predicted_lrl_answer}'") # DEBUG
            else:
                # Fallback logic
                print(f"WARN (Sample {idx}): Could not extract final LRL answer via regex. Raw output: {response_text[:100]}...") # DEBUG
            predicted_lrl_answer = response_text.strip()
                if len(predicted_lrl_answer) > max_tokens * 3: # Adjusted fallback length check
                    print(f"DEBUG (Sample {idx}): Fallback answer too long, setting to [Extraction Failed].") # DEBUG
                    predicted_lrl_answer = "[Extraction Failed]"
                else:
                    print(f"DEBUG (Sample {idx}): Using fallback answer (raw response): '{predicted_lrl_answer}'") # DEBUG
            
            result = {
                'question': original_question,
                'ground_truth': original_ground_truth_str,
                'predicted_answer': predicted_lrl_answer,
                'question_en': None, 
                'answer_en': None,
                'context_used': True, # Context is used in this pipeline
                'pipeline': 'single_prompt',
                'few_shot': use_few_shot,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'max_tokens': max_tokens # Store the intended final answer max_tokens
            }
            results.append(result)

        except Exception as e:
            print(f"ERROR processing sample {row.get('id', idx)} (Single Prompt): {e}") # Added Sample ID
            import traceback # DEBUG
            traceback.print_exc() # DEBUG
            continue

    if not results:
        print(f"WARNING: No results were successfully processed for {lang_code} with {model_name} using CoTR Single Prompt.")
        # --- DEBUG ---
        print(f"--- DEBUG: EXITING evaluate_qa_cotr_single_prompt (No Results) ---")
        # --- END DEBUG ---
        return pd.DataFrame()
        
    results_df_final = pd.DataFrame(results)
    # --- DEBUG ---
    print(f"\n--- DEBUG: EXITING evaluate_qa_cotr_single_prompt ---")
    print(f"Returning DataFrame with shape: {results_df_final.shape}")
    if not results_df_final.empty:
        print("Sample of returned data:")
        print(results_df_final.head())
    # --- END DEBUG ---
    return results_df_final