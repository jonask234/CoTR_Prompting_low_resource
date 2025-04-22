# thesis_project/baseline/qa_baseline.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re

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

def generate_qa_prompt(question, context=None):
    """
    Generate a prompt for the QA task with explicit instructions.
    Now ignoring context as per professor's recommendation to use model's parametric knowledge.
    
    Args:
        question: The question
        context: The context/passage (not used anymore, kept for API compatibility)
    
    Returns:
        Formatted prompt
    """
    prompt = f"""Answer the following question using your own knowledge.
Keep your answer as short and direct as possible.
- If the answer is a number, respond with just the number.
- If the answer is a date, respond with just the date.
- If the answer is a name, respond with just the name.
- If the answer is a short phrase, respond with just that phrase.
- If the answer is Yes or No, respond with only "Yes" or "No".
- Do not add explanations, notes, or citations.
- Do not include text like "Answer:" in your response.

Question: {question}

Answer:"""
    return prompt

def extract_answer(output_text, question, is_aya_model=False):
    """
    Extract and clean the model's answer.
    
    Args:
        output_text: Raw output text from the model
        question: The original question (for answer verification)
        is_aya_model: Flag for model-specific processing
        
    Returns:
        Cleaned answer
    """
    # Normalize whitespace and remove leading/trailing spaces
    answer = output_text.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "answer:", "the answer is:", "answer is:", 
        "context:", "question:", "according to my knowledge,",
        "based on my knowledge,"
    ]
    
    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Remove notes/citations/qualifiers in parentheses and brackets
    answer = re.sub(r'\([^)]*\)', '', answer)
    answer = re.sub(r'\[[^]]*\]', '', answer)
    answer = re.sub(r'^\s*-\s+', '', answer)  # Remove leading bullet points
    
    # Remove common verbose qualifiers
    qualifiers = [
        "this is", "the answer is", "it is", "i believe",
        "according to my knowledge", "based on the information",
        "based on my understanding", "as far as i know",
        "to the best of my knowledge", "historically",
        "note:", "please note"
    ]
    
    for qualifier in qualifiers:
        if answer.lower().startswith(qualifier):
            answer = answer[len(qualifier):].strip()
            # Remove any colons or similar characters after removing qualifiers
            if answer.startswith(":") or answer.startswith(",") or answer.startswith("-"):
                answer = answer[1:].strip()
    
    # Split the answer on the first sentence boundary and take the first part only
    # This helps with removing explanations that follow the actual answer
    first_sent_match = re.search(r'^(.*?[.?!])(?:\s|$)', answer)
    if first_sent_match:
        first_sent = first_sent_match.group(1).strip()
        # Only use the first sentence if it's not too short compared to the full answer
        if len(first_sent) > len(answer) * 0.3:  # At least 30% of the full answer
            answer = first_sent
    
    # Clean up Yes/No answers
    if answer.lower().startswith("yes"):
        # Check if the answer is a clear yes with explanation
        if re.match(r'^yes\W', answer.lower()):
            answer = "Yes"
    elif answer.lower().startswith("no"):
        # Check if the answer is a clear no with explanation
        if re.match(r'^no\W', answer.lower()):
            answer = "No"
            
    # If we're still left with a multiline answer, take just the first line
    if "\n" in answer:
        answer = answer.split("\n")[0].strip()
    
    # If answer starts with a long quote mark or similar, clean it
    answer = re.sub(r'^["\'""'']+', '', answer).strip()
    answer = re.sub(r'["\'""'']+$', '', answer).strip()
    
    # Final sanity check - if answer is still very long (more than 50 chars),
    # and has sentence-looking content, try to extract just the key part
    if len(answer) > 50 and "," in answer:
        # For long answers with commas, often the key answer is before the first comma
        potential_short = answer.split(",")[0].strip()
        if 2 < len(potential_short) < 30:  # Reasonable short answer length
            answer = potential_short
    
    # If the answer is extremely long (likely incorrect), truncate it
    max_answer_length = 100
    if len(answer) > max_answer_length:
        answer = answer[:max_answer_length].strip()
        
    # Final check for empty answers
    if not answer:
        answer = "[No answer generated]"
        
    return answer

def process_qa_baseline(tokenizer, model, question, context=None, 
                          max_new_tokens=50,  # Reduced from 200
                          max_input_length=4096,
                          temperature=0.3,  # Reduced from 0.5 for more focus
                          top_p=0.85):  # Slightly reduced
    """
    Process a QA pair with the given model directly (baseline).
    Now ignoring context as per professor's recommendation to use model's parametric knowledge.
    
    Args:
        tokenizer: The model tokenizer
        model: The language model
        question: The question text
        context: The context text (not used anymore, kept for API compatibility)
        max_new_tokens: Maximum number of new tokens to generate for the answer
        max_input_length: Maximum length of the input sequence
        temperature: Temperature for sampling
        top_p: Top_p for sampling
    
    Returns:
        The model's answer
    """
    prompt = generate_qa_prompt(question)
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Determine if this is the Aya model for specialized processing
    is_aya_model = "aya" in model.config._name_or_path.lower()
    
    # Model-specific parameter adjustments
    gen_temperature = temperature
    gen_top_p = top_p
    gen_max_tokens = max_new_tokens
    gen_do_sample = True
    
    # Aya model specific adjustments
    if is_aya_model:
        # Aya seems to respond better with slightly higher temperature
        gen_temperature = 0.4  
        # Use beam search with Aya for more focused answers
        gen_do_sample = False  # Disable sampling, use beam search
        beam_size = 3
    
    # Generate answer
    with torch.no_grad():
        if is_aya_model and not gen_do_sample:
            # Beam search for Aya model
            output_ids = model.generate(
                **inputs,
                max_new_tokens=gen_max_tokens,
                do_sample=False,
                num_beams=beam_size,
                early_stopping=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            # Temperature sampling for other models or Aya (if sampling enabled)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=gen_max_tokens,
                do_sample=gen_do_sample,
                temperature=gen_temperature,
                top_p=gen_top_p,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
    
    # Decode only the newly generated tokens
    output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Extract and clean the answer
    answer = extract_answer(output_text, question, is_aya_model=is_aya_model)
    
    # Special handling for Yes/No questions to avoid answering with explanations
    if is_yes_no_question(question):
        if "yes" in answer.lower() and len(answer) > 5:
            answer = "Yes"
        elif "no" in answer.lower() and len(answer) > 5:
            answer = "No"
    
    # Final formatting for numeric answers - try to extract just the number if appropriate
    if is_numeric_question(question) and not answer.lower() in ["yes", "no", "[no answer generated]"]:
        numeric_match = re.search(r'\b(\d[\d,.]*\d|\d)\b', answer)
        if numeric_match:
            answer = numeric_match.group(1)
    
    return answer

def is_yes_no_question(question):
    """Check if a question is likely a yes/no question."""
    question_lower = question.lower()
    
    # Check for common yes/no question patterns
    yes_no_starters = [
        "is ", "are ", "was ", "were ", "will ", "would ", "can ", "could ", 
        "does ", "do ", "did ", "has ", "have ", "had ", "should ", "shall ",
        "might ", "may "
    ]
    
    for starter in yes_no_starters:
        if question_lower.startswith(starter):
            return True
    
    return False

def is_numeric_question(question):
    """Check if a question is likely expecting a numeric answer."""
    question_lower = question.lower()
    
    numeric_patterns = [
        "how many", "how much", "what year", "what date", "when", 
        "how old", "how long", "how far", "what is the number", 
        "what is the value", "what is the amount", "what is the percentage"
    ]
    
    for pattern in numeric_patterns:
        if pattern in question_lower:
            return True
    
    return False

def evaluate_qa_baseline(model_name, samples_df, lang_code):
    """
    Evaluate the baseline QA approach on the given samples.
    Now using model's parametric knowledge instead of provided context.
    
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
    print(f"NOTE: Context is being ignored as per professor's recommendation. Using model's parametric knowledge.")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} samples ({model_name})"):
        try:
            question = row["question"]
            context = row["context"]  # Still read context for reference/reporting
            
            # Handle both formats for ground truth - direct ground_truth or answers.text
            if "ground_truth" in row and row["ground_truth"]:
                # Fallback data format - use ground_truth directly
                ground_truth_answer = row["ground_truth"]
                ground_truth_answers = [ground_truth_answer]
            else:
                # TyDiQA format - extract from answers
                ground_truth_answers = row.get("answers", {}).get("text", [])
                if not ground_truth_answers:
                    print(f"Skipping sample {row.get('id', idx)} due to missing ground truth answers.")
                    continue
            
            # Get model prediction WITHOUT using context
            predicted_answer = process_qa_baseline(
                tokenizer, model, question, None,  # Pass None for context 
                max_input_length=safe_max_input_length
            )
            
            # Store result
            result = {
                "question": question,
                "context": context[:200] + "...",  # Kept for reference only
                "ground_truth_answers": ground_truth_answers,
                "ground_truth": ground_truth_answers[0] if ground_truth_answers else None,  # Add direct ground_truth
                "predicted_answer": predicted_answer,
                "language": lang_code,
                "context_used": False  # Flag to indicate context was not used
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