import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
import time

# Erwartete Labels
EXPECTED_LABELS = ["positive", "negative", "neutral"]

def initialize_model(model_name):
    # Initialisiert das Modell und den Tokenizer
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
        cache_dir=cache_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def generate_sentiment_prompt(text, use_few_shot=True):
    # Erzeugt einen einfachen Prompt für die Sentiment-Analyse
    
    prompt = f"Analyze the sentiment of the following text. Respond with only one word: positive, negative, or neutral.\\n\\nText: '{text}'"

    if use_few_shot:
        examples = """
Examples:
Text: 'This movie was fantastic, I loved it!'
Sentiment: positive

Text: 'I am not happy with the service provided.'
Sentiment: negative

Text: 'The meeting is scheduled for 3 PM.'
Sentiment: neutral"""
        prompt += examples
            
    prompt += "\nSentiment:"
    return prompt

def extract_label(output_text):
    # Extrahiert das Label aus der Ausgabe des Modells
    if not output_text:
        return "unknown"

    text_lower = output_text.lower().strip()

    if "positive" in text_lower:
        return "positive"
    elif "negative" in text_lower:
        return "negative"
    elif "neutral" in text_lower:
        return "neutral"
    
    return "unknown"

def process_sentiment_baseline(
    model,
    tokenizer,
    text,
    use_few_shot,
    max_tokens=10,
    temperature=0.3,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.1,
    do_sample=True
):
    # Verarbeitet einen einzelnen Text zur Sentiment-Analyse
    start_time = time.time()
    
    prompt = generate_sentiment_prompt(text, use_few_shot=use_few_shot)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
        
    raw_model_output = ""
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
        
        raw_model_output = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        predicted_label = extract_label(raw_model_output)
    
    except Exception as e:
        print(f"Error during sentiment processing: {e}")
        predicted_label = "error"
            
    runtime = time.time() - start_time
    return predicted_label, runtime, raw_model_output

def evaluate_sentiment_baseline(
    model_name, 
    tokenizer,
    model,
    samples_df, 
    lang_code,
    prompt_in_lrl,
    use_few_shot,
    temperature,
    top_p,
    top_k,
    max_tokens,
    repetition_penalty
):
    # Wertet die Sentiment-Analyse für einen Datensatz aus
    results = []

    print(f"Evaluating {model_name} on {lang_code}...")
    for idx, row in samples_df.iterrows():
        text = row['text']
        ground_truth_label = row['label'] 

        predicted_label, runtime, raw_model_output = process_sentiment_baseline(
            model=model, 
            tokenizer=tokenizer, 
            text=text,
            use_few_shot=use_few_shot,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True
        )

        results.append({
            'id': row.get('id', idx), 
            'text': text,
            'ground_truth_label': ground_truth_label,
            'predicted_label': predicted_label,
            'language': lang_code,
            'runtime_seconds': runtime,
            'raw_model_output': raw_model_output
        })

    return pd.DataFrame(results)
