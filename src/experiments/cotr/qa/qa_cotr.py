from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any, List
import pandas as pd
import time

def initialize_model(model_name: str) -> tuple:
    """Initialize the model and tokenizer."""
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def generate_qa_prompt(question: str, context: str) -> str:
    """Generate the QA prompt for the model."""
    return f"""Context: {context}

Question: {question}

Answer:"""

def process_qa_pair(
    model: Any,
    tokenizer: Any,
    question: str,
    context: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """Process a single QA pair using the model."""
    prompt = generate_qa_prompt(question, context)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the answer part after "Answer:"
    answer = response.split("Answer:")[-1].strip()
    
    return answer

def evaluate_qa_cotr(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str,
    batch_size: int = 1
) -> pd.DataFrame:
    """
    Evaluate QA using CoTR approach.
    
    Args:
        model_name: Name of the model to use
        samples_df: DataFrame containing the samples
        lang_code: Language code (e.g., 'hi' for Hindi)
        batch_size: Number of samples to process at once
        
    Returns:
        DataFrame with results
    """
    model, tokenizer = initialize_model(model_name)
    results = []
    
    for idx, row in samples_df.iterrows():
        print(f"\nProcessing sample {idx + 1}/{len(samples_df)}")
        
        # Get answer directly in the original language
        answer = process_qa_pair(model, tokenizer, row['question'], row['context'])
        
        # Store results
        result = {
            'question': row['question'],
            'context': row['context'],
            'predicted_answer': answer,
            'ground_truth': row['answers']['text'][0]
        }
        results.append(result)
        
        # Optional: Add delay to avoid rate limiting
        time.sleep(1)
    
    return pd.DataFrame(results)