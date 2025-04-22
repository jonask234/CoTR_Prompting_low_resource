import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
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
    cache_path = "/work/bbd6522/cache_dir"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        cache_dir=cache_path
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_path
    )
    return tokenizer, model

def generate_summarization_prompt(text, lang_code="en", prompt_in_lrl=False):
    """
    Generate a prompt for the summarization task with structured format.
    
    Args:
        text: The text to summarize
        lang_code: Language code
        prompt_in_lrl: Whether to generate the prompt in the LRL
    
    Returns:
        Formatted prompt
    """
    # Truncate long articles to 8000 chars to avoid token limits
    if len(text) > 8000:
        text = text[:8000] + "..."
        
    # English prompt (default) - more structured format
    en_prompt = f"""Text: '{text}'

Summarize the above text in 2-3 sentences. Capture the main points only.
Provide your summary in a direct, concise format without additional explanation.

Summary:"""

    # For English input, always use English prompt
    if lang_code == "en":
        return en_prompt
        
    # LRL prompts if requested
    if prompt_in_lrl:
        if lang_code == "sw":
            # Swahili prompt - structured format
            return f"""Maandishi: '{text}'

Fupisha maandishi hapo juu katika sentensi 2-3. Chukua pointi kuu tu.
Toa muhtasari wako kwa njia ya moja kwa moja, bila maelezo ya ziada.

Muhtasari:"""
        elif lang_code == "te":
            # Telugu prompt - structured format
            return f"""పాఠ్యం: '{text}'

పైన ఉన్న పాఠ్యాన్ని 2-3 వాక్యాలలో సంక్షిప్తీకరించండి. ప్రధాన అంశాలను మాత్రమే పట్టుకోండి.
అదనపు వివరణ లేకుండా నేరుగా, సంక్షిప్తమైన ఫార్మాట్‌లో మీ సారాంశాన్ని అందించండి.

సారాంశం:"""
        else:
            print(f"Warning: No specific {lang_code} prompt available, falling back to English prompt.")
    
    # Default to English prompt
    return en_prompt

def extract_summary(output_text):
    """
    Extract and clean the generated summary.
    
    Args:
        output_text: Raw output text from the model
        
    Returns:
        Cleaned summary
    """
    # Remove prompt residue if present
    summary = output_text.strip()
    
    # Remove "Summary:" prefix if present
    if summary.lower().startswith("summary:"):
        summary = summary[8:].strip()
        
    # Remove bullet points if present
    summary = re.sub(r'^\s*[\-\*•]\s+', '', summary)
    
    # Remove numbered points if present
    summary = re.sub(r'^\s*\d+[\.\)]\s+', '', summary)
    
    # Remove quotes if they wrap the entire output
    if (summary.startswith('"') and summary.endswith('"')) or \
       (summary.startswith("'") and summary.endswith("'")):
        summary = summary[1:-1].strip()
    
    return summary

def process_summarization_baseline(tokenizer, model, text, lang_code="en", prompt_in_lrl=False,
                                max_input_length=4096, max_new_tokens=100):
    """
    Process a summarization task with the given model directly (baseline).
    
    Args:
        tokenizer: The model tokenizer
        model: The language model
        text: The text to summarize
        lang_code: Language code
        prompt_in_lrl: Whether to generate the prompt in the LRL
        max_input_length: Maximum length of the input sequence
        max_new_tokens: Maximum number of new tokens to generate
    
    Returns:
        The model's summary
    """
    # Generate appropriate prompt
    prompt = generate_summarization_prompt(text, lang_code, prompt_in_lrl)
    
    # Get the device of the model
    device = next(model.parameters()).device
    
    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Determine if this is the Aya model for specialized processing
    is_aya_model = "aya" in model.config._name_or_path.lower()
    
    # Generate summary
    with torch.no_grad():
        if is_aya_model:
            # Aya tends to respond better with beam search for structured outputs
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            # For other models, use sampling with temperature
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
    
    # Decode only the newly generated tokens
    output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Extract and clean the summary
    summary = extract_summary(output_text)
    
    return summary

def evaluate_summarization_baseline(model_name, samples_df, lang_code, prompt_in_lrl=False):
    """
    Evaluate the baseline summarization approach on the given samples.
    
    Args:
        model_name: Name of the model to use
        samples_df: DataFrame containing the samples (text and summary columns)
        lang_code: Language code for reporting
        prompt_in_lrl: Whether to use prompts in the LRL
    
    Returns:
        DataFrame with predictions and original summaries
    """
    try:
        tokenizer, model = initialize_model(model_name)
    except Exception as e:
        print(f"ERROR: Failed to initialize model {model_name}: {e}")
        return pd.DataFrame()
        
    results = []
    
    prompt_type = "LRL prompt" if prompt_in_lrl else "English prompt"
    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), 
                         desc=f"Processing {lang_code} samples ({model_name}, {prompt_type})"):
        try:
            article_text = row["text"]
            reference_summary = row["summary"]
            
            # Truncate very long articles (XL-Sum can have lengthy articles)
            # Better to get a summary on a partial article than none at all
            if len(article_text) > 8000:
                article_text = article_text[:8000]
            
            # Get model prediction
            predicted_summary = process_summarization_baseline(
                tokenizer, model, article_text, lang_code, prompt_in_lrl,
                max_new_tokens=100  # Adjust based on desired summary length
            )
            
            # Store result
            result = {
                "article": article_text[:500] + "...",  # First 500 chars for reference
                "reference_summary": reference_summary,
                "predicted_summary": predicted_summary,
                "language": lang_code,
                "prompt_type": "LRL prompt" if prompt_in_lrl else "English prompt"
            }
            
            # Add the row ID if available
            if "id" in row:
                result["id"] = row["id"]
                
            results.append(result)
            
        except Exception as e:
            print(f"ERROR processing sample {idx}: {e}")
            continue
    
    if not results:
        print(f"WARNING: No results were successfully processed for {lang_code} with {model_name}.")
        return pd.DataFrame()
        
    results_df = pd.DataFrame(results)
    return results_df 