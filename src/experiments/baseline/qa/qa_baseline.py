# thesis_project/baseline/qa_baseline.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

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

def generate_qa_prompt(question, context):
    """
    Generate a prompt for the QA task.
    
    Args:
        question: The question
        context: The context/passage
    
    Returns:
        Formatted prompt
    """
    prompt = f"""Context: {context}

Question: {question}

Answer:"""
    return prompt

def process_qa_baseline(tokenizer, model, question, context, max_new_tokens=100, max_input_length=4096):
    """
    Process a QA pair with the given model directly (baseline approach).
    
    Args:
        tokenizer: The model tokenizer
        model: The language model
        question: The question text
        context: The context text
        max_new_tokens: Maximum number of new tokens to generate for the answer
        max_input_length: Maximum length of the input sequence (context + question + prompt)
    
    Returns:
        The model's answer
    """
    prompt = generate_qa_prompt(question, context)
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate answer
    with torch.no_grad():
        # Ensure max_new_tokens doesn't exceed model limits if possible, but prioritize user setting
        # Some models might have a hard limit, generate might handle this.
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False, # Keep False for deterministic baseline
            # temperature=0.7, # Temperature ignored if do_sample=False
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens
    output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # The decoded text *is* the answer, no need to split prompt again
    answer = output_text.strip()
    
    # Handle potential empty answers
    if not answer:
        answer = "[No answer generated]"
        
    return answer

def evaluate_qa_baseline(model_name, samples_df, lang_code):
    """
    Evaluate the baseline QA approach on the given samples.
    
    Args:
        model_name: Name of the model to use
        samples_df: DataFrame containing the samples
        lang_code: Language code for reporting
    
    Returns:
        DataFrame with predictions and metrics, or empty DataFrame if critical error
    """
    try:
        tokenizer, model = initialize_model(model_name)
    except Exception as e:
        print(f"ERROR: Failed to initialize model {model_name}: {e}")
        return pd.DataFrame() # Return empty if model fails to load
        
    results = []
    
    # Determine max input length based on model if possible, fallback to default
    # Using a potentially safer value than max_position_embeddings directly
    model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
    # Set a practical limit, considering memory and typical context window usefulness
    safe_max_input_length = min(model_max_len, 8192) 
    
    print(f"Using max_input_length: {safe_max_input_length}")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} samples ({model_name})"):
        try:
            question = row["question"]
            context = row["context"]
            # Ensure answers field is correctly accessed
            ground_truth_answers = row.get("answers", {}).get("text", []) 
            if not ground_truth_answers: # Skip if no ground truth answers
                print(f"Skipping sample {row.get('id', idx)} due to missing ground truth answers.")
                continue
            
            # Get model prediction with truncation awareness
            predicted_answer = process_qa_baseline(
                tokenizer, model, question, context, 
                max_input_length=safe_max_input_length
            )
            
            # Store result
            result = {
                "question": question,
                "context": context[:200] + "...",  # Truncated context for display/storage
                "ground_truth_answers": ground_truth_answers,
                "predicted_answer": predicted_answer,
                "language": lang_code
            }
            results.append(result)
        except Exception as e:
            print(f"ERROR processing sample {row.get('id', idx)}: {e}")
            # Optionally add a placeholder result or just skip
            continue # Skip sample on error
    
    if not results:
        print(f"WARNING: No results were successfully processed for {lang_code} with {model_name}.")
        return pd.DataFrame()
        
    results_df = pd.DataFrame(results)
    return results_df