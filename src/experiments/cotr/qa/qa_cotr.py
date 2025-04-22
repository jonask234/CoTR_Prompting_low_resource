from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any, List
import pandas as pd
import time
from tqdm import tqdm
import re # Import re

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
            return f"""Text: '{text}'

Translate this Swahili text to English.
Preserve the exact meaning without adding or removing information.
Provide only the direct translation without explanations.

Translation:"""
        elif source_lang == 'te':
            return f"""Text: '{text}'

Translate this Telugu text to English.
Preserve the exact meaning without adding or removing information.
Provide only the direct translation without explanations.

Translation:"""
    
    # Enhanced specialized prompts for English → LRLs to improve back-translation
    if source_lang == 'en':
        if target_lang == 'sw':
            return f"""Text: '{text}'

Translate this English text to Swahili.
Preserve the exact meaning and be precise with names, numbers, and technical terms.
Provide only the direct translation without explanations.

Translation:"""
        elif target_lang == 'te':
            return f"""Text: '{text}'

Translate this English text to Telugu.
Preserve the exact meaning and be precise with names, numbers, and technical terms.
Provide only the direct translation without explanations.

Translation:"""

    # Default/original prompt for other language pairs or directions
    return f"""Text: '{text}'

Translate this {source_name} text to {target_name}.
Preserve the exact meaning without adding explanations or extra context.

Translation:"""

def generate_qa_prompt(context=None, question=None, prompt_type="default"):
    """Generate the QA prompt for the model with explicit instructions.
    Now ignoring context as per professor's recommendation to use model's parametric knowledge.
    Using more structured format with clear input/output expectations.
    
    Args:
        context: The context (not used anymore, kept for API compatibility)
        question: The question
        prompt_type: Type of prompt to generate
        
    Returns:
        Formatted prompt
    """
    # Enhanced prompt focusing on model's parametric knowledge with clear formatting
    prompt = f"""Question: '{question}'

Answer the question using your own knowledge. Provide your answer in the following format:
- For yes/no questions: respond with only 'Yes' or 'No'
- For factual questions: provide just the specific fact, name, date, or number
- For quantity questions: respond with just the number
- For time questions: respond with just the date or time period
- For person questions: respond with just the person's name
- For location questions: respond with just the place name

Answer:"""
    return prompt

def translate_text(
    model: Any,
    tokenizer: Any,
    text: str,
    source_lang: str,
    target_lang: str,
    max_input_length: int = 4096,
    max_new_tokens: int = 512, # Max tokens for the translated output
    model_name: str = ""  # Added model name for model-specific adjustments
) -> str:
    """Translate text from source language to target language using the model."""
    # Special handling for English to English "translation" (no-op)
    if source_lang == 'en' and target_lang == 'en':
        return text  # Simply return the original text for English->English
        
    # --- Direct Handling for Yes/No Back-Translation (e.g., English -> Telugu) ---
    if source_lang == 'en':
        if target_lang == 'te':
            if text.strip().lower() == "yes":
                return "అవును"  # "Avunu" in Telugu
            elif text.strip().lower() == "no":
                return "లేదు"   # "Ledu" in Telugu
        elif target_lang == 'sw':
            if text.strip().lower() == "yes":
                return "Ndiyo"
            elif text.strip().lower() == "no":
                return "Hapana" # Swahili negative

    # --- Proceed with Model-based Translation if not handled above ---
    prompt = generate_translation_prompt(text, source_lang, target_lang)
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Adjust generation parameters based on language pair and model
    temperature = 0.7  # Default
    top_p = 0.9  # Default
    
    # Customize parameters for specific model+language combinations
    if "aya" in model_name.lower():
        # Aya model adjustments
        if target_lang == 'sw':
            temperature = 0.6  # Lower temp for more reliable Swahili
        elif target_lang == 'te':
            temperature = 0.6  # Lower temp for more reliable Telugu
    elif "qwen" in model_name.lower():
        # Qwen model adjustments
        if target_lang == 'sw':
            temperature = 0.8  # Higher temp for Qwen's Swahili
        elif target_lang == 'te':
            temperature = 0.8  # Higher temp for Qwen's Telugu
    
    # Multiple translation attempts for important cases (text < 100 chars)
    # This helps with short answers that might benefit from consensus
    translations = []
    if len(text) < 100 and (target_lang in ['sw', 'te'] or source_lang in ['sw', 'te']):
        # Generate 3 translations to pick the best/most consistent one
        for _ in range(3):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode only the newly generated tokens
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            # Clean up the response
            clean_translation = clean_translation_response(response, target_lang, source_lang)
            translations.append(clean_translation)
        
        # Pick the most common translation (simple majority vote)
        if len(set(translations)) == 1:
            # All translations are the same
            translation = translations[0]
        else:
            # Count occurrences
            from collections import Counter
            counter = Counter(translations)
            # Get the most common
            translation = counter.most_common(1)[0][0]
    else:
        # For longer text, just do a single translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature, 
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        # Clean up the response
        translation = clean_translation_response(response, target_lang, source_lang)
    
    return translation

def clean_translation_response(response: str, target_lang: str, source_lang: str) -> str:
    """Clean up translation response to extract the actual translation."""
    translation = response.strip()
    
    # Remove potential prompt remnants first (case-insensitive)
    target_name = lang_names.get(target_lang, target_lang)
    translation_prompt_start = f"{target_name} translation:"
    translation = re.sub(f"^{re.escape(translation_prompt_start)}", '', translation, flags=re.IGNORECASE).strip()
    
    # Remove generic QA artifacts as well (case-insensitive)
    translation = re.sub(r'^answer:', '', translation, flags=re.IGNORECASE).strip()
    translation = re.sub(r'^question:', '', translation, flags=re.IGNORECASE).strip()
    translation = re.sub(r'^context:', '', translation, flags=re.IGNORECASE).strip()
    
    # Remove new prompt instructions if they leak
    translation = translation.replace("Based *only* on the context below, answer the question.", "").strip()
    translation = translation.replace("If the answer is a simple yes or no, respond with only the single word 'Yes' or 'No'.", "").strip()
    translation = translation.replace("Otherwise, provide the shortest possible span of text from the context that answers the question.", "").strip()

    # Remove specific artifacts observed (e.g., from Swahili QA)
    translation = translation.replace("[1]", "").strip()
    translation = translation.replace("AlgQuestion", "").strip()
    
    if target_lang == 'sw': # Language-specific cleaning
         translation = re.sub(r'^Je,', '', translation, flags=re.IGNORECASE).strip()
    elif target_lang == 'te': # Add Telugu-specific cleaning
         # Remove potential Telugu artifacts like question markers or repetitive phrases
         translation = re.sub(r'^మరి,', '', translation, flags=re.IGNORECASE).strip()  # Telugu equivalent of "Je," or "Well,"
         translation = re.sub(r'^చెప్పండి,', '', translation, flags=re.IGNORECASE).strip()  # "Tell me," in Telugu
    
    # Basic handling for very obvious, short repetitions
    if len(translation) > 3:
        if target_lang == 'sw': # Apply Swahili repetition check
            if translation.lower().startswith('yesyes') or translation.lower().startswith('ndiyondiyo'):
                 translation = 'Yes' if target_lang == 'en' else 'Ndiyo'
            elif translation.lower().startswith('nono') or translation.lower().startswith('hapahapa'):
                 translation = 'No' if target_lang == 'en' else 'Hapana'
        elif target_lang == 'te': # Apply Telugu repetition check
            if translation.lower().startswith('అవునుఅవును') or translation.lower().startswith('yesyes'):
                 translation = 'Yes' if target_lang == 'en' else 'అవును'
            elif translation.lower().startswith('లేదులేదు') or translation.lower().startswith('nono'):
                 translation = 'No' if target_lang == 'en' else 'లేదు'
    
    # Handle potential empty answers AFTER cleaning
    if not translation:
        translation = "[No translation generated]"
    
    return translation

def process_qa_pair(
    model: Any,
    tokenizer: Any,
    question: str,
    context: str = None,  # Now optional and unused
    max_input_length: int = 4096,
    max_new_tokens: int = 50, # Reduced from 200 to 50 for more concise answers
    temperature: float = 0.3, # Lower temperature for more focused answers
    top_p: float = 0.9,
    model_name: str = ""  # Added model_name parameter to adjust params per model
) -> str:
    """Process a single QA pair using the model.
    Now ignoring context as per professor's recommendation to use model's parametric knowledge.
    """
    prompt = generate_qa_prompt(None, question)
    
    # Check if this is a yes/no question to handle differently
    is_yes_no = is_yes_no_question(question)
    
    # Adjust parameters based on model
    if "aya" in model_name.lower():
        # Aya model needs higher temperature for diverse answers
        if is_yes_no:
            temperature = 0.1  # Keep very low for yes/no questions
            do_sample = False  # Use greedy for yes/no 
        else:
            temperature = 0.4  # Slightly higher for non-yes/no questions
            do_sample = True
    elif "qwen" in model_name.lower():
        # Qwen model tends to be more verbose, so use lower temperature
        if is_yes_no:
            temperature = 0.1
            do_sample = False
        else:
            temperature = 0.25
            do_sample = True
    else:
        # Default parameters for other models
        do_sample = not is_yes_no  # Greedy for yes/no, sampling for others
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        if is_yes_no:
            # For yes/no questions, use lower temperature and no sampling
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,  # Even shorter for yes/no questions
                do_sample=do_sample,
                num_beams=3, # Use beam search for yes/no questions
                temperature=temperature,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            # Non-yes/no questions with customized parameters
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens, 
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
    
    # Decode only the newly generated tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Apply answer extraction to clean up the response
    answer = extract_answer(response, is_yes_no)
    
    return answer

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
    """Extract and clean the answer from the model response."""
    # Clean and normalize
    answer = response.strip()
    
    # First remove "Explanation:" sections which often contain justifications
    if "explanation:" in answer.lower():
        answer = answer.split("explanation:", 1)[0].strip()
    
    # Also remove any "Answer:" prefix
    if "answer:" in answer.lower():
        parts = answer.lower().split("answer:")
        # Take the part after "Answer:" if it exists
        if len(parts) > 1:
            answer = parts[1].strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "answer:", "the answer is:", "answer is:", 
        "context:", "question:", "according to my knowledge,",
        "based on my knowledge,", "as far as i know,",
        "to the best of my knowledge,"
    ]
    
    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Remove notes/citations/qualifiers in parentheses and brackets
    answer = re.sub(r'\([^)]*\)', '', answer)
    answer = re.sub(r'\[[^]]*\]', '', answer)
    
    # Remove bullet points and list markers
    answer = re.sub(r'^\s*[\-\*•]\s+', '', answer)
    answer = re.sub(r'^\s*\d+[\.\)]\s+', '', answer)
    
    # For yes/no questions, extract just the yes or no
    if is_yes_no:
        if answer.lower().startswith("yes"):
            answer = "Yes"
        elif answer.lower().startswith("no"):
            answer = "No"
        # Handle more complex cases where the answer might include explanation
        elif "yes" in answer.lower() and not "no" in answer.lower():
            answer = "Yes"
        elif "no" in answer.lower() and not "yes" in answer.lower():
            answer = "No"
    
    # For non-yes/no questions, try to get just the first sentence if it's a good answer
    elif not is_yes_no and "." in answer:
        first_sentence = answer.split(".")[0].strip() + "."
        if len(first_sentence) > 2 and len(first_sentence) < len(answer) * 0.7:
            # The first sentence is reasonably sized (not too short, not the whole answer)
            answer = first_sentence
    
    # Clean up additional patterns
    answer = re.sub(r'^["\'""'']+', '', answer).strip()
    answer = re.sub(r'["\'""'']+$', '', answer).strip()
    
    # Handle potential empty answers
    if not answer:
        answer = "[No answer generated]"
    
    return answer

def evaluate_qa_cotr(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str,
    batch_size: int = 1 # Batching not implemented here, processes one by one
) -> pd.DataFrame:
    """
    Evaluate QA using Chain of Translation Prompting (CoTR) approach.
    Now ignoring context and using only model's parametric knowledge.
    
    Steps:
    1. Translate question from source language to English
    2. Perform QA in English (using model's knowledge)
    3. Translate the answer back to the source language
    
    Args:
        model_name: Name of the model to use
        samples_df: DataFrame containing the samples
        lang_code: Language code (e.g., 'bn' for Bengali, 'sw' for Swahili)
        batch_size: Number of samples to process at once (currently unused)
        
    Returns:
        DataFrame with results including all translations and answers, or empty DF on error
    """
    try:
        model, tokenizer = initialize_model(model_name)
    except Exception as e:
        print(f"ERROR: Failed to initialize model {model_name}: {e}")
        return pd.DataFrame()
        
    results = []
    
    # Determine max input length based on model if possible, fallback to default
    model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
    # Set a practical limit, considering memory and typical context window usefulness
    safe_max_input_length = min(model_max_len, 8192) 
    print(f"Using max_input_length: {safe_max_input_length}")
    print(f"NOTE: Context is being ignored as per professor's recommendation. Using model's parametric knowledge.")

    # Set language-specific parameters
    max_answer_tokens = 50  # Default
    qa_temperature = 0.3    # Default
    
    # Language-specific adjustments
    if lang_code == 'sw':
        # Swahili tends to need more tokens for equivalent expressions
        max_answer_tokens = 60
        qa_temperature = 0.25  # Lower temperature for more consistent answers
    elif lang_code == 'te':
        # Telugu often needs more tokens as well
        max_answer_tokens = 60
        qa_temperature = 0.25  # Lower temperature for more consistent answers

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} samples ({model_name} CoTR)"):
        try:
            original_question = row['question']
            original_context = row['context']  # Keep for reference but don't use
            
            # Handle both formats for ground truth - direct ground_truth or answers.text
            if "ground_truth" in row and row["ground_truth"]:
                # Fallback data format - use ground_truth directly
                ground_truth = [row["ground_truth"]]
            else:
                # TyDiQA format - extract from answers
                ground_truth = row.get("answers", {}).get("text", [])
                
            if not ground_truth:
                 print(f"Skipping sample {row.get('id', idx)} due to missing ground truth answers.")
                 continue
            
            # Step 1: Translate question to English
            question_en = translate_text(model, tokenizer, original_question, lang_code, "en", 
                                       max_input_length=safe_max_input_length, model_name=model_name)
            
            # Step 2: Process QA in English (without context)
            answer_en = process_qa_pair(model, tokenizer, question_en, None, 
                                       max_input_length=safe_max_input_length,
                                       max_new_tokens=max_answer_tokens,
                                       temperature=qa_temperature,
                                       model_name=model_name)
            
            # Normalize English answer before back-translation
            original_ground_truth_str = ground_truth[0]
            if answer_en.lower() in ["yes", "no"]:
                # Ensure consistent capitalization for yes/no answers
                answer_en = "Yes" if answer_en.lower() == "yes" else "No"
            
            # Step 3: Translate answer back to original language
            answer = translate_text(model, tokenizer, answer_en, "en", lang_code, 
                                   max_input_length=safe_max_input_length,
                                   model_name=model_name)
            
            # Store results
            result = {
                'question': original_question,
                'context': original_context[:200] + "..." if len(original_context) > 200 else original_context,
                'question_en': question_en,
                'context_en': "",  # No context used
                'answer_en': answer_en,
                'predicted_answer': answer,
                'ground_truth': ground_truth[0],
                'context_used': False  # Flag to indicate context was not used
            }
            results.append(result)

        except Exception as e:
            print(f"ERROR processing sample {row.get('id', idx)}: {e}")
            continue # Skip sample on error

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