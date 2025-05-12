import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os
import re
import time
import traceback
from typing import Tuple, Dict, List, Any # Added List, Any
import logging

# Define expected labels (adjust if dataset uses different ones)
EXPECTED_LABELS = ["positive", "negative", "neutral"]

def initialize_model(model_name: str) -> Tuple[Any, Any]: # Added type hints
    """
    Initialize a model and tokenizer, specifying cache directory.
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
        cache_dir=cache_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def preprocess_text(text: str, lang_code: str) -> str:
    """
    Preprocess text based on language-specific characteristics.
    """
    processed_text = text.strip()
    if lang_code == 'sw':
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
                                   u"\U0001F700-\U0001F77F" u"\U0001F780-\U0001F7FF" u"\U0001F800-\U0001F8FF"
                                   u"\U0001F900-\U0001F9FF" u"\U0001FA00-\U0001FA6F" u"\U0001FA70-\U0001FAFF"
                                   u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251" 
                                   "]+")
        processed_text = re.sub(emoji_pattern, lambda m: f" {m.group(0)} ", processed_text)
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text

def generate_sentiment_prompt(text: str, lang_code: str = "en", model_name: str = "", use_few_shot: bool = True) -> str:
    """Generate a prompt for sentiment classification, with EN instructions and optional few-shot examples."""
    processed_text = preprocess_text(text, lang_code) # Preprocess based on actual text lang
    
    # Instructions are in English
    base_instruction = "Analyze the sentiment of the text below.\nRespond with only one of these labels: positive, negative, or neutral."
    
    prompt_parts = [f"Text: '{processed_text}'\n", base_instruction]

    if use_few_shot:
        examples_en = """
Examples:
Text: 'This movie was fantastic, I loved it!'
Sentiment: positive

Text: 'I am not happy with the service provided.'
Sentiment: negative

Text: 'The meeting is scheduled for 3 PM.'
Sentiment: neutral

Text: 'I'm really happy with the service! 😊' 
Sentiment: positive

Text: 'I was very disappointed with this product.'
Sentiment: negative

Text: 'I'm going to the store tomorrow.'
Sentiment: neutral

Text: 'This experience has been terrible and I lost money. 😢'
Sentiment: negative"""
        prompt_parts.append("\n" + examples_en)
            
    prompt_parts.append("\nSentiment:")
    return "\n".join(prompt_parts)

def generate_lrl_instruct_sentiment_prompt(text: str, lang_code: str, model_name: str = "", use_few_shot: bool = True) -> str:
    """Generate a prompt for sentiment classification with LRL instructions and optional few-shot examples."""
    processed_text = preprocess_text(text, lang_code)
    en_labels_for_prompt = "positive, negative, or neutral" # Model should output these EN labels

    instruction = ""
    examples = ""

    if lang_code == 'sw':
        instruction = f"Chunguza kwa makini hisia katika maandishi yaliyotolewa, ukizingatia pia emoji na ishara zingine.\nJibu kwa Kiingereza kwa kutumia MOJA TU kati ya maneno haya: {en_labels_for_prompt}.\nHakikisha majibu yako ni sahihi na sio tu kupendelea hisia chanya."
        if use_few_shot:
            examples = """
Mifano:
Maandishi: 'Ninafurahia sana hali ya hewa leo! 😊'
Hisia: positive

Maandishi: 'Sikufurahishwa na huduma hii kabisa, ilikuwa mbaya sana. 😠'
Hisia: negative

Maandishi: 'Labda nitaenda dukani kesho, sijaamua bado.'
Hisia: neutral

Maandishi: 'Nilipata hasara kubwa katika biashara hii. 😢'
Hisia: negative"""
        prompt_format = f"Maandishi: '{processed_text}'\n\n{instruction}"
        if use_few_shot:
            prompt_format += f"\n\n{examples}"
        prompt_format += "\n\nHisia:" # Model should output English label here
        return prompt_format

    elif lang_code == 'ha':
        instruction = f"Bincika yanayin rubutu a sama. Ka amsa da ɗaya daga cikin kalmomi na turanci kawai: {en_labels_for_prompt}."
        if use_few_shot:
            examples = """
Misalai:
Rubutu: 'Na yi farin ciki da jin labarin nasara.'
Ra'ayi: positive

Rubutu: 'Ba na son yadda aka yi wannan abu ba.'
Ra'ayi: negative

Rubutu: 'Zan tafi gobe.'
Ra'ayi: neutral

Rubutu: 'Ban ji dadi da labarin da na samu ba.'
Ra'ayi: negative"""
        prompt_format = f"Rubutu: '{processed_text}'\n\n{instruction}"
        if use_few_shot:
            prompt_format += f"\n\n{examples}"
        prompt_format += "\n\nRa'ayi:" # Model should output English label here
        return prompt_format
        
    else: # Fallback to English instructions if LRL not defined
        print(f"WARN: LRL instructions not specifically defined for {lang_code}. Using English instruction prompt structure.")
        return generate_sentiment_prompt(text, 'en', model_name, use_few_shot) # lang_code='en' to get EN examples

def extract_label(output_text: str, lang_code: str = "en", model_name: str = "") -> str:
    """Extracts sentiment label from model output, prioritizing exact matches."""
    text_lower = output_text.lower().strip()
    
    common_prefixes = ["sentiment:", "the sentiment is", "this text is", "i think the sentiment is", "hisia:", "ra'ayi:"]
    for prefix in common_prefixes:
        if text_lower.startswith(prefix):
            text_lower = text_lower[len(prefix):].strip()

    # Check for exact labels first (case-insensitive)
    for label in EXPECTED_LABELS:
        if text_lower == label:
            return label
    
    # Check if the cleaned text starts with one of the expected labels
    for label in EXPECTED_LABELS:
        if text_lower.startswith(label):
            return label
        
    # Fallback for very short responses containing the label
    if len(text_lower.split()) < 5:
        for label in EXPECTED_LABELS:
            if label in text_lower:
                return label
                
    print(f"WARN: Could not reliably extract label for lang '{lang_code}', model '{model_name}'. Output: '{output_text}'. Defaulting to neutral.")
    return "neutral"

def process_sentiment_baseline(
    model: Any, # Added type hint
    tokenizer: Any, # Added type hint
    text: str, 
    lang_code: str, 
    model_name: str,
    generation_params: Dict,
    use_few_shot: bool,
    prompt_in_lrl: bool = False,
    max_input_length: int = 2048 # Increased default
) -> Tuple[str, float]:
    """Process a text sample for sentiment classification."""
    start_time = time.time()
    predicted_label = "error" 
    
    try:
        if prompt_in_lrl and lang_code != 'en': # LRL instructions only for LRL text
            prompt = generate_lrl_instruct_sentiment_prompt(text, lang_code, model_name, use_few_shot=use_few_shot)
        else: # English instructions for English text, or if LRL instructions are off for LRL text
            prompt = generate_sentiment_prompt(text, lang_code, model_name, use_few_shot=use_few_shot)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        gen_temp = generation_params.get("temperature", 0.3) # Defaulted to lower for classification
        gen_rep_penalty = generation_params.get("repetition_penalty", 1.1)
        gen_top_p = generation_params.get("top_p", 0.9)
        gen_top_k = generation_params.get("top_k", 40) 
        gen_max_new_tokens = generation_params.get("max_tokens", 10) 

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_max_new_tokens,
                temperature=gen_temp,
                top_p=gen_top_p,
                top_k=gen_top_k,
                repetition_penalty=gen_rep_penalty,
                do_sample=True if gen_temp > 0.01 else False, # Sample if temp is not effectively zero
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        response = response.strip()
        predicted_label = extract_label(response, lang_code, model_name) # Pass lang_code
    
    except Exception as e:
        print(f"Error during sentiment processing for text: '{text[:50]}...'")
        print(traceback.format_exc())
        # predicted_label remains "error"
            
    runtime = time.time() - start_time
    return predicted_label, runtime


def evaluate_sentiment_baseline(
    model_name: str, 
    tokenizer: Any, # Added tokenizer
    model: Any,     # Added model
    samples_df: pd.DataFrame, 
    lang_code: str,
    prompt_in_lrl: bool = False, # Added prompt_in_lrl
    use_few_shot: bool = True,   # Added use_few_shot
    generation_params: Dict = None # Added generation_params
) -> pd.DataFrame:
    """Evaluate sentiment classification baseline on a dataset."""
    
    if generation_params is None: 
        generation_params = {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_tokens": 10, "repetition_penalty": 1.1}

    # Model initialization is now expected to happen in the runner script
    # tokenizer, model = initialize_model(model_name) 
    results = []

    shot_description = "few-shot" if use_few_shot else "zero-shot"
    prompt_lang_description = "LRL-instruct" if prompt_in_lrl and lang_code != 'en' else "EN-instruct"

    print(f"Evaluating {model_name} on {lang_code} ({prompt_lang_description}, {shot_description})...")
    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} Sentiment Baseline"):
        text = row['text']
        ground_truth_label = row['label'] 

            predicted_label, runtime = process_sentiment_baseline(
            model, tokenizer, text, lang_code, model_name,
            generation_params=generation_params,
            use_few_shot=use_few_shot,
                prompt_in_lrl=prompt_in_lrl
            )

        results.append({
            'id': row.get('id', idx), 
            'text': text,
            'ground_truth_label': ground_truth_label,
            'predicted_label': predicted_label,
            'language': lang_code,
            'runtime_seconds': runtime,
            'prompt_language': prompt_lang_description, # Use descriptive term
            'shot_type': shot_description # Use descriptive term
        })

    results_df = pd.DataFrame(results)
    # Model cleanup should happen in the runner script after all experiments for that model are done
    # del model
    # del tokenizer
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    return results_df

def run_sentiment_baseline(model_name: str, samples_df: pd.DataFrame, lang_code: str, prompt_in_lrl: bool = False, use_few_shot: bool = True, generation_params: Dict = None):
    """Run sentiment classification baseline experiments for a given model and dataset."""
    tokenizer, model = initialize_model(model_name)
    results_df = evaluate_sentiment_baseline(model_name, tokenizer, model, samples_df, lang_code, prompt_in_lrl, use_few_shot, generation_params)
    return results_df

def main():
    # This is a placeholder for the data loader and experiment orchestration.
    # Example usage (you'll need to adapt this based on your actual data loading and argument parsing):
    # args = parse_your_arguments() # You would define an argument parser
    # model_name_to_test = "CohereLabs/aya-expanse-8b" # Example model
    # lang_to_test = "sw" # Example language
    # num_samples_to_test = 10 # Example number of samples
    
    # logging.info(f"Running baseline sentiment analysis for {model_name_to_test} on {lang_to_test}")
    
    # Placeholder: Load your sentiment dataset
    # Make sure this function is correctly implemented and returns a DataFrame with 'text' and 'label'
    # from src.utils.data_loaders.load_sentiment_data import load_sentiment_dataset 
    # samples = load_sentiment_dataset(lang_code=lang_to_test, num_samples=num_samples_to_test, seed=42) # Example call
    
    # if not samples.empty:
    #     generation_params_example = {"temperature": 0.2, "top_p": 0.9, "max_tokens": 5}
    #     results = run_sentiment_baseline(
    #         model_name_to_test, 
    #         samples, 
    #         lang_to_test,
    #         prompt_in_lrl=False, # Example: English instructions
    #         use_few_shot=True,   # Example: Few-shot
    #         generation_params=generation_params_example
    #     )
    #     if not results.empty:
    #         print("\\nSample of Results:")
    #         print(results.head())
    #         # Further processing: calculate metrics, save results, etc.
    #         # from evaluation.sentiment_metrics import calculate_sentiment_metrics
    #         # metrics = calculate_sentiment_metrics(results)
    #         # print(f"\\nMetrics for {lang_to_test} ({model_name_to_test}):")
    #         # for key, value in metrics.items():
    #         #     print(f"  {key}: {value:.4f}")
    #     else:
    #         logging.info(f"No results obtained for {lang_to_test} with {model_name_to_test}.")
    # else:
    #     logging.info(f"No samples loaded for {lang_to_test}. Skipping baseline evaluation.")
    
    logging.info("Placeholder main function executed. Implement data loading and experiment calls.")
    logging.info("\n====== Sentiment Baseline Script Finished (Placeholder Main) ======")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Add basic logging config
    main()