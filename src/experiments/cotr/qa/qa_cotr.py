from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any, List
import pandas as pd
import time
from tqdm import tqdm

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
    """Generate a prompt for translation."""
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
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)
    
    return f"""Translate the following {source_name} text to {target_name}:

{text}

{target_name} translation:"""

def generate_qa_prompt(question: str, context: str) -> str:
    """Generate the QA prompt for the model."""
    return f"""Context: {context}

Question: {question}

Answer:"""

def translate_text(
    model: Any,
    tokenizer: Any,
    text: str,
    source_lang: str,
    target_lang: str,
    max_input_length: int = 4096,
    max_new_tokens: int = 512 # Max tokens for the translated output
) -> str:
    """Translate text from source language to target language using the model."""
    prompt = generate_translation_prompt(text, source_lang, target_lang)
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7, # Allow some variation for potentially better translation
            top_p=0.9,
            do_sample=True, # Sample for translation
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Extract the translation part (simplified extraction)
    translation = response.strip()
    
    # Basic post-processing: remove potential prompt artifacts if needed
    # This might need refinement based on observed outputs
    if f"{target_lang} translation:" in prompt:
         translation = translation.split(f"{target_lang} translation:")[-1].strip()
         
    # Handle potential empty answers
    if not translation:
        translation = "[No translation generated]"
    
    return translation

def process_qa_pair(
    model: Any,
    tokenizer: Any,
    question: str,
    context: str,
    max_input_length: int = 4096,
    max_new_tokens: int = 50, # Keep QA answers short
    temperature: float = 0.7, # Use lower temp for more factual QA
    top_p: float = 0.9
) -> str:
    """Process a single QA pair using the model."""
    prompt = generate_qa_prompt(question, context)
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens, 
            do_sample=False, # Usually False for QA baseline/CoTR answer generation
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # The decoded text *is* the answer
    answer = response.strip()
    
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
    
    Steps:
    1. Translate question and context from source language to English
    2. Perform QA in English
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

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} samples ({model_name} CoTR)"):
        try:
            original_question = row['question']
            original_context = row['context']
            ground_truth = row.get("answers", {}).get("text", [])
            if not ground_truth:
                 print(f"Skipping sample {row.get('id', idx)} due to missing ground truth answers.")
                 continue
            
            # Step 1: Translate question and context to English
            # print("Step 1: Translating question and context to English...") # Verbose
            question_en = translate_text(model, tokenizer, original_question, lang_code, "en", max_input_length=safe_max_input_length)
            context_en = translate_text(model, tokenizer, original_context, lang_code, "en", max_input_length=safe_max_input_length, max_new_tokens=1024) # Allow longer translation output
            
            # Step 2: Process QA in English
            # print("Step 2: Processing QA in English...") # Verbose
            answer_en = process_qa_pair(model, tokenizer, question_en, context_en, max_input_length=safe_max_input_length)
            
            # Step 3: Translate answer back to original language
            # print("Step 3: Translating answer back to original language...") # Verbose
            answer = translate_text(model, tokenizer, answer_en, "en", lang_code, max_input_length=safe_max_input_length)
            
            # Store results
            result = {
                'question': original_question,
                'context': original_context[:200] + "...", # Truncate original context for storage
                'question_en': question_en,
                'context_en': context_en[:200] + "...", # Truncate translated context for storage
                'answer_en': answer_en,
                'predicted_answer': answer,
                'ground_truth': ground_truth[0] # Assuming first ground truth is primary
            }
            results.append(result)
            
            # Optional: Add delay to avoid rate limiting or excessive GPU heat
            # time.sleep(1) 

        except Exception as e:
            print(f"ERROR processing sample {row.get('id', idx)}: {e}")
            continue # Skip sample on error

    if not results:
        print(f"WARNING: No results were successfully processed for {lang_code} with {model_name} using CoTR.")
        return pd.DataFrame()
        
    return pd.DataFrame(results)