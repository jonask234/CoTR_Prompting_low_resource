# thesis_project/baseline/qa_baseline.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm

def initialize_model(model_name):
    """
    Initialize a model and tokenizer.
    
    Args:
        model_name: Name of the model to initialize
    
    Returns:
        tokenizer, model
    """
    print(f"Initializing {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
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

def process_qa_baseline(tokenizer, model, question, context, max_length=100):
    """
    Process a QA pair with the given model directly (baseline approach).
    
    Args:
        tokenizer: The model tokenizer
        model: The language model
        question: The question text
        context: The context text
        max_length: Maximum length of the generated answer
    
    Returns:
        The model's answer
    """
    prompt = generate_qa_prompt(question, context)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate answer
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract just the answer part (after "Answer:")
    answer_parts = output_text.split("Answer:")
    if len(answer_parts) > 1:
        answer = answer_parts[1].strip()
    else:
        answer = output_text  # Fallback if format isn't as expected
    
    return answer

def evaluate_qa_baseline(model_name, samples_df, lang_code):
    """
    Evaluate the baseline QA approach on the given samples.
    
    Args:
        model_name: Name of the model to use
        samples_df: DataFrame containing the samples
        lang_code: Language code for reporting
    
    Returns:
        DataFrame with predictions and metrics
    """
    tokenizer, model = initialize_model(model_name)
    
    results = []
    
    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} samples"):
        question = row["question"]
        context = row["context"]
        ground_truth_answers = [answer["text"] for answer in row["answers"]["text"]]
        
        # Get model prediction
        predicted_answer = process_qa_baseline(tokenizer, model, question, context)
        
        # Store result
        result = {
            "question": question,
            "context": context[:100] + "...",  # Truncated for display
            "ground_truth_answers": ground_truth_answers,
            "predicted_answer": predicted_answer,
            "language": lang_code
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    return results_df