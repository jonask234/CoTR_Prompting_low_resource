import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os
from typing import List

# Define expected labels based on MasakhaNEWS dataset exploration
# This should ideally match the labels found by the data loader
EXPECTED_LABELS = ['business', 'health', 'nigeria', 'politics', 'religion', 'sports', 'world']

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
    return tokenizer, model

def generate_classification_prompt(text: str, labels: List[str]) -> str:
    """
    Generate a zero-shot prompt for text classification.
    Instructs model to output only one of the expected labels.
    """
    label_string = ", ".join(labels)
    prompt = f"""Classify the following text into one of these categories: {label_string}.
Respond with only the category name.

Text: {text}

Category:"""
    return prompt

def process_classification_baseline(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    text: str,
    labels: List[str],
    max_new_tokens: int = 10, # Labels are short words
    max_input_length: int = 1024 # MasakhaNEWS texts can be longer
) -> str:
    """
    Process a text sample for classification directly (baseline).
    Returns the predicted label string (or an error indicator).
    """
    prompt = generate_classification_prompt(text, labels)

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
    predicted_label_raw = output_text.strip().lower()
    # Clean potential extra text around the label
    predicted_label_lines = predicted_label_raw.split('\n')
    predicted_label = predicted_label_lines[0].strip()

    # Find the first expected label mentioned in the cleaned output
    final_label = "[Unknown]" # Default if no valid label found
    # Check against the provided possible labels for this run
    for label in labels:
        # Check for exact match or if prediction starts with the label
        # (e.g., handles cases like 'sports.')
        if label == predicted_label or predicted_label.startswith(label):
            final_label = label
            break # Take the first match

    # Fallback: Check if the raw output contained the label name clearly
    if final_label == "[Unknown]":
         for label in labels:
             if label in predicted_label_raw:
                 final_label = label
                 break

    return final_label

def evaluate_classification_baseline(model_name: str, samples_df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
    """
    Evaluate the baseline classification approach on the given samples.
    Assumes samples_df has 'text' and 'label' columns.
    """
    try:
        tokenizer, model = initialize_model(model_name)
    except Exception as e:
        print(f"ERROR: Failed to initialize model {model_name}: {e}")
        return pd.DataFrame()

    results = []

    # Determine possible labels from the dataset itself and convert to strings
    possible_labels = sorted([str(label) for label in samples_df['label'].unique()])
    print(f"Using labels found in dataset for prompts: {possible_labels}")

    model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
    # Use a reasonable default, allow for longer news text
    safe_max_input_length = min(model_max_len, 1024)
    print(f"Using max_input_length: {safe_max_input_length}")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} baseline classification ({model_name})"):
        try:
            text = row["text"]
            ground_truth_label = row["label"]

            predicted_label = process_classification_baseline(
                tokenizer, model, text,
                labels=possible_labels,
                max_input_length=safe_max_input_length
            )

            result = {
                "id": row.get("id", idx),
                "text": text, # Keep text for potential analysis
                "ground_truth_label": str(ground_truth_label),
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