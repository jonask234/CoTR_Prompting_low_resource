import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os

# Define expected labels (adjust if dataset uses different ones)
EXPECTED_LABELS = ["positive", "negative", "neutral"]

def initialize_model(model_name):
    """
    Initialize a model and tokenizer, specifying cache directory.
    """
    print(f"Initializing {model_name}...")
    cache_path = "/work/bbd6522/cache_dir" # Define cache path
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path,
        torch_dtype=torch.float16, # Use float16 for potential memory savings
        device_map="auto" # Automatically map model to available GPUs
    )
    # No explicit .to("cuda") needed with device_map="auto"
    return tokenizer, model

def generate_sentiment_prompt(text: str) -> str:
    """
    Generate a zero-shot prompt for sentiment classification.
    Instructs model to output only one of the expected labels.
    """
    # Simple instruction-based prompt
    prompt = f"""Analyze the sentiment of the following text. Respond with only one word: positive, negative, or neutral. 

Text: {text}

Sentiment:"""
    return prompt

def process_sentiment_baseline(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    text: str,
    max_new_tokens: int = 5, # Labels are short
    max_input_length: int = 1024 # Adjust as needed, tweets are short
) -> str:
    """
    Process a text sample for sentiment classification directly (baseline).
    Returns the predicted label string (or an error indicator).
    """
    prompt = generate_sentiment_prompt(text)

    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate prediction
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False, # Deterministic output
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the newly generated tokens
    output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    # Post-process the output to extract the label
    predicted_label = output_text.strip().lower()

    # Find the first expected label mentioned in the output
    final_label = "[Unknown]" # Default if no valid label found
    for label in EXPECTED_LABELS:
        if label in predicted_label:
            final_label = label
            break # Take the first match

    return final_label

def evaluate_sentiment_baseline(model_name: str, samples_df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
    """
    Evaluate the baseline sentiment analysis approach on the given samples.
    """
    try:
        tokenizer, model = initialize_model(model_name)
    except Exception as e:
        print(f"ERROR: Failed to initialize model {model_name}: {e}")
        return pd.DataFrame()

    results = []

    model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
    # Use a reasonable default, sentiment texts are usually shorter
    safe_max_input_length = min(model_max_len, 1024)
    print(f"Using max_input_length: {safe_max_input_length}")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} baseline sentiment ({model_name})"):
        try:
            text = row["text"]
            ground_truth_label = row["label"]

            predicted_label = process_sentiment_baseline(
                tokenizer, model, text,
                max_input_length=safe_max_input_length
            )

            result = {
                "id": row.get("id", idx),
                "text": text,
                "ground_truth_label": ground_truth_label,
                "predicted_label": predicted_label,
                "language": lang_code
            }
            results.append(result)
        except Exception as e:
            print(f"ERROR processing sample {row.get('id', idx)}: {e}")
            continue

    if not results:
        print(f"WARNING: No results were successfully processed for {lang_code} with {model_name}.")
        return pd.DataFrame()

    return pd.DataFrame(results) 