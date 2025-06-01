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
    """
    Generates a sentiment analysis prompt, ALWAYS with English instructions and English few-shot examples.
    Model is always expected to output English sentiment labels: 'positive', 'negative', or 'neutral'.
    The lang_code parameter here is mostly for preprocess_text, as instructions are fixed to English.
    """
    processed_text = preprocess_text(text, lang_code) # lang_code used for text preprocessing
    en_labels_for_prompt = "positive, negative, or neutral"

    base_instruction = f"Analyze the sentiment of the text provided. Your entire response MUST be ONLY one of the following words: {en_labels_for_prompt}. Do NOT add any other text, explanation, or punctuation."
    prompt_parts = [f"Text: '{processed_text}'\n", base_instruction]

    if use_few_shot:
        # Consistent English few-shot examples
        english_few_shot_examples_text = """
Examples:
Text: 'This movie was fantastic, I loved it!'
Sentiment: positive

Text: 'I am not happy with the service provided.'
Sentiment: negative

Text: 'The meeting is scheduled for 3 PM.'
Sentiment: neutral

Text: 'The product is okay, neither good nor bad.'
Sentiment: neutral"""
        prompt_parts.append("\n" + english_few_shot_examples_text)
            
    prompt_parts.append("\nSentiment:")
    return "\n".join(prompt_parts)

def generate_lrl_instruct_sentiment_prompt(text: str, lang_code: str, model_name: str = "", use_few_shot: bool = True) -> str:
    """Generate a prompt for sentiment classification with LRL instructions and optional few-shot examples."""
    processed_text = preprocess_text(text, lang_code)
    en_labels_for_prompt = "positive, negative, or neutral" # Model should output these EN labels

    instruction = ""
    examples_content = "" # Renamed from 'examples' to avoid conflict

    # Define English few-shot examples (original, for reference or if LRL examples are not available)
    # english_few_shot_examples_text = """
    # Examples:
    # Text: 'I am very happy with this product, it works perfectly! 😊'
    # Sentiment: positive
    # ... (rest of English examples)
    # """

    if lang_code == 'sw':
        instruction = f"Chunguza kwa makini hisia katika maandishi yaliyotolewa. Jibu lako lote lazima liwe MOJA TU kati ya maneno haya ya Kiingereza: {en_labels_for_prompt}. USIANDIKE maneno mengine yoyote, maelezo, au alama za uandishi."
        if use_few_shot:
            # Swahili text examples, English labels
            sw_few_shot_examples = """
Mifano:
Maandishi: 'Huduma hii ni nzuri sana, nimefurahishwa kweli!'
Hisia: positive

Maandishi: 'Nimechukizwa na bidhaa hii, haifanyi kazi vizuri.'
Hisia: negative

Maandishi: 'Mkutano utaanza saa tisa alasiri.'
Hisia: neutral
"""
            examples_content = sw_few_shot_examples
        prompt_format = f"Maandishi: '{processed_text}'\n\n{instruction}"
        if examples_content: # Check if examples_content is not empty
            prompt_format += f"\n\n{examples_content}"
        prompt_format += "\n\nHisia:" # Model should output English label here
        return prompt_format

    elif lang_code == 'ha':
        instruction = f"Bincika yanayin rubutu a sama. Duk amsarka DOLE ta zama DAYA KAWAI daga cikin waɗannan kalmomi na Turanci: {en_labels_for_prompt}. KADA KA ƙara wasu kalmomi, bayani, ko alamun rubutu."
        if use_few_shot:
            # Hausa text examples, English labels
            ha_few_shot_examples = """
Misalai:
Rubutu: 'Wannan abinci yana da daɗi ƙwarai, na gamsu sosai!'
Ra'ayi: positive

Rubutu: 'Na ji haushin wannan fim, bai yi kyau ba ko kaɗan.'
Ra'ayi: negative

Rubutu: 'Yanayin yau ba sanyi, ba zafi.'
Ra'ayi: neutral
"""
            examples_content = ha_few_shot_examples
        prompt_format = f"Rubutu: '{processed_text}'\n\n{instruction}"
        if examples_content: # Check if examples_content is not empty
            prompt_format += f"\n\n{examples_content}"
        prompt_format += "\n\nRa'ayi:" # Model should output English label here
        return prompt_format
    else: # Fallback for other LRLs or if prompt_in_lrl is False (though this func is for LRL-instruct)
        logging.warning(f"LRL instructions for '{lang_code}' using generate_lrl_instruct_sentiment_prompt. Custom LRL examples not defined; falling back to English prompt structure via generate_sentiment_prompt.")
        # Fallback to the standard English prompt function if specific LRL is not handled
        return generate_sentiment_prompt(text, "en", model_name, use_few_shot)

def extract_label(output_text: str, lang_code: str = "en", model_name: str = "") -> str:
    """
    Extracts the sentiment label from the model's output text.
    Prioritizes direct matches, then checks for keywords.
    The function now aims to return one of "positive", "neutral", "negative", or "unknown".
    """
    processed_output = output_text.lower().strip()

    # Define keywords for each sentiment in English (can be expanded for LRL keywords if needed)
    # These are indicative and might need adjustment based on model behavior
    positive_keywords = ["positive", "positive sentiment", "good", "happy", "joyful", "excellent"]
    negative_keywords = ["negative", "negative sentiment", "bad", "sad", "angry", "terrible"]
    neutral_keywords = ["neutral", "neutral sentiment", "objective", "no sentiment", "neither positive nor negative"]

    # --- Stricter matching for direct labels first ---
    # Check for exact label words, possibly surrounded by non-alphanumeric characters or at string ends
    if re.search(r"\bpositive\b", processed_output):
        return "positive"
    if re.search(r"\bnegative\b", processed_output):
        return "negative"
    if re.search(r"\bneutral\b", processed_output):
        return "neutral"

    # --- If no direct match, check for keywords ---
    # Count keyword occurrences for each sentiment
    positive_score = sum(1 for kw in positive_keywords if kw in processed_output)
    negative_score = sum(1 for kw in negative_keywords if kw in processed_output)
    neutral_score = sum(1 for kw in neutral_keywords if kw in processed_output)

    # Determine label based on scores
    if positive_score > negative_score and positive_score > neutral_score:
        return "positive"
    elif negative_score > positive_score and negative_score > neutral_score:
        return "negative"
    elif neutral_score > positive_score and neutral_score > negative_score:
        return "neutral"
    
    # --- Fallback and Ambiguity Handling ---
    # If scores are tied, or no keywords strongly indicate a sentiment,
    # or if the output is very short and non-indicative, consider it "unknown".
    # Example: if output is "I don't know" or just "The text is about..."

    # More specific checks for model-specific "cannot determine" phrases
    # (These should be adapted based on observed model outputs)
    unknown_phrases_common = [
        "cannot determine", "unable to classify", "not enough information",
        "no clear sentiment", "sentiment is unclear", "uncertain",
        "i don't know", "it's unclear", "no sentiment expressed",
        # LRL equivalents if known
        "sijui", # Swahili for "I don't know"
        "ban sani ba" # Hausa for "I don't know"
    ]
    
    model_specific_unknown_phrases = []
    if "aya" in model_name.lower():
        model_specific_unknown_phrases.extend([
            # "Aya specific unknown phrase 1",
        ])
    elif "qwen" in model_name.lower():
        model_specific_unknown_phrases.extend([
            # "Qwen specific unknown phrase 1",
        ])

    all_unknown_phrases = unknown_phrases_common + model_specific_unknown_phrases
    for phrase in all_unknown_phrases:
        if phrase in processed_output:
            return "unknown"

    # If after all checks, no clear label is found, return "unknown".
    # This also handles cases where scores might be tied (e.g., 1 positive, 1 negative, 0 neutral)
    # or if all scores are 0.
    
    # If only one class has a non-zero score, and others are zero, pick that class.
    # This is a refinement after the initial scoring to catch less ambiguous cases.
    if positive_score > 0 and negative_score == 0 and neutral_score == 0:
        return "positive"
    if negative_score > 0 and positive_score == 0 and neutral_score == 0:
        return "negative"
    if neutral_score > 0 and positive_score == 0 and negative_score == 0:
        return "neutral"
        
    # Final fallback
    return "unknown"

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
) -> Tuple[str, float, str]: # Return type updated to include raw_output
    """Process a single text for sentiment analysis using the baseline approach."""
    start_time = time.time()
    raw_model_output_for_return = "[No output]"
    predicted_label = "error" # Default in case of issues

    try:
        # Determine which prompt generation function to use
        if prompt_in_lrl and lang_code != 'en':
            prompt = generate_lrl_instruct_sentiment_prompt(
                text, lang_code, model_name, use_few_shot
            )
            logging.debug(f"Using LRL-instruct prompt for {lang_code}")
        else:
            prompt = generate_sentiment_prompt(
                text, lang_code, model_name, use_few_shot
            )
            logging.debug(f"Using EN-instruct prompt for {lang_code}")
        
        logging.debug(f"Generated prompt (first 300 chars): {prompt[:300]}...")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        # Use generation parameters passed in, with a fallback for max_tokens if not present
        gen_max_tokens = generation_params.get("max_tokens", 30) # Fallback to 30 if not in params
        gen_temperature = generation_params.get("temperature", 0.3)
        gen_top_p = generation_params.get("top_p", 0.9)
        gen_top_k = generation_params.get("top_k", 40)
        gen_repetition_penalty = generation_params.get("repetition_penalty", 1.1)
        # Determine do_sample based on temperature: if temp is very low (e.g. <=0.01), greedy is better.
        # However, allow explicit override from generation_params if present.
        gen_do_sample = generation_params.get("do_sample", gen_temperature > 0.01)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_max_tokens,
                temperature=gen_temperature,
                top_p=gen_top_p,
                top_k=gen_top_k,
                repetition_penalty=gen_repetition_penalty,
                do_sample=gen_do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        response = response.strip()
        raw_model_output_for_return = response
        predicted_label = extract_label(response, lang_code, model_name) # Pass lang_code
    
    except Exception as e:
        print(f"Error during sentiment processing for text: '{text[:50]}...'")
        print(traceback.format_exc())
        # predicted_label remains "error"
            
    runtime = time.time() - start_time
    return predicted_label, runtime, raw_model_output_for_return


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

        predicted_label, runtime, raw_model_output = process_sentiment_baseline(
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
            'shot_type': shot_description, # Use descriptive term
            'raw_model_output': raw_model_output
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