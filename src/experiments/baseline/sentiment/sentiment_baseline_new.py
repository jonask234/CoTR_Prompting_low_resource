import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os
import re
import time
import traceback
from typing import Tuple, Dict, List, Any, Optional # Added List, Any, Optional
import logging
import json

# Initialize logger at the module level
logger = logging.getLogger(__name__) # Added logger initialization

# Define expected labels (adjust if dataset uses different ones)
EXPECTED_LABELS = ["positive", "negative", "neutral"]

# All few-shot examples are now consistently in English across all configurations

# --- Added Global Constants and Helper ---
LANG_NAMES = {
    "en": "English", "sw": "Swahili", "ha": "Hausa", "am": "Amharic", "dz": "Dzongkha",
    "pcm": "Nigerian Pidgin", "yo": "Yoruba", "ma": "Marathi", "multi": "Multilingual",
    "te": "Telugu", "pt": "Portuguese"
}

ENGLISH_SENTIMENT_LABELS = ["positive", "negative", "neutral"]
SENTIMENT_LABELS_EN_STR = ", ".join(ENGLISH_SENTIMENT_LABELS)

# Define LRL translations - crucial for mapping if model outputs LRL label by mistake
SENTIMENT_LABELS_LRL = {
    "sw": {"positive": "chanya", "negative": "hasi", "neutral": "kati", "unknown": "haijulikani"},
    "ha": {"positive": "tabbatacce", "negative": "korau", "neutral": "tsaka-tsaki", "unknown": "ba'a sani ba"},
    "yo": {"positive": "rere", "negative": "búburú", "neutral": "dídọ̀ọ̀dọ́", "unknown": "aimọ"},
    "am": {"positive": "አዎንታዊ", "negative": "አሉታዊ", "neutral": "ገለልተኛ", "unknown": "ያልታወቀ"},
    "pcm": {"positive": "good", "negative": "bad", "neutral": "neutral", "unknown": "unknown"},
    "pt": {"positive": "positivo", "negative": "negativo", "neutral": "neutro", "unknown": "desconhecido"},
}

def get_language_name(lang_code: str) -> str:
    """Helper to get full language name."""
    return LANG_NAMES.get(lang_code, lang_code.capitalize())
# --- End of Added Global Constants ---

# Ensure SENTIMENT_LABELS_EN is defined or imported if used elsewhere in this file
# For consistency with sentiment_cotr.py, let's define it.
SENTIMENT_LABELS_EN = ["positive", "negative", "neutral"]

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
        # Standardized to 3 examples for consistency across all tasks
        english_few_shot_examples_text = """
Examples:
Text: 'This movie was fantastic, I loved it!'
Sentiment: positive

Text: 'I am not happy with the service provided.'
Sentiment: negative

Text: 'The meeting is scheduled for 3 PM.'
Sentiment: neutral"""
        prompt_parts.append("\n" + english_few_shot_examples_text)
            
    prompt_parts.append("\nSentiment:")
    return "\n".join(prompt_parts)

def generate_lrl_instruct_sentiment_prompt(text: str, lang_code: str, model_name: str = "", use_few_shot: bool = True) -> str:
    """
    Generates a sentiment classification prompt IN THE LOW-RESOURCE LANGUAGE (LRL),
    instructing the model to output an English sentiment label.
    Few-shot examples use English text and English sentiment labels for consistency with other tasks.
    """
    lrl_name = get_language_name(lang_code)
    text_escaped = text.replace("'", "\\'") # Keep existing escaping

    # --- English Few-Shot Examples (English text, English label) ---
    # Changed to use English examples for consistency with other tasks - STANDARDIZED TO 3
    english_example_positive_text = "This movie was fantastic, I loved it!"
    english_example_negative_text = "I am not happy with the service provided."
    english_example_neutral_text = "The meeting is scheduled for 3 PM."

    # --- LRL Instruction Segment ---
    # IMPORTANT: This section needs to be translated into each respective LRL.
    # For now, using placeholders or a direct English instruction with a note.
    # The core instruction is to classify the LRL text and respond with an ENGLISH label.

    instruction_segment_lrl = ""
    if lang_code == "sw":
        instruction_segment_lrl = f"Wewe ni mtaalamu wa uchanganuzi wa hisia. Kazi yako ni kuainisha hisia za maandishi yafuatayo ya {lrl_name}. Jibu kwa MOJA tu ya lebo hizi za Kiingereza: {SENTIMENT_LABELS_EN_STR}. Usiongeze maelezo."
    elif lang_code == "ha":
        instruction_segment_lrl = f"Kai ƙwararren masanin nazarin ra'ayi ne. Ayyukanka shine ka rarraba ra'ayin rubutun {lrl_name} mai zuwa. Ka amsa da DAYA kawai daga cikin waɗannan alamun Turanci: {SENTIMENT_LABELS_EN_STR}. Kada ka ƙara bayani."
    elif lang_code == "pt":
        instruction_segment_lrl = f"Você é um especialista em análise de sentimentos. Sua tarefa é classificar o sentimento do seguinte texto em {lrl_name}. Responda com APENAS UM dos seguintes rótulos em inglês: {SENTIMENT_LABELS_EN_STR}. Não adicione explicações."
    else:
        # Fallback to English instruction if LRL not defined, with a clear logger warning
        logger.warning(f"LRL instruction segment not defined for lang_code '{lang_code}'. Using English instructions as a fallback for the main prompt body.")
        instruction_segment_lrl = f"You are a sentiment analysis expert. Your task is to classify the sentiment of the following {lrl_name} text. Respond with ONLY one of the English labels: {SENTIMENT_LABELS_EN_STR}. Do not add explanations."

    prompt = f"{instruction_segment_lrl}\n\n"

    if use_few_shot:
        # The few-shot examples section header should also be in LRL.
        few_shot_header_lrl = f"Hapa kuna mifano kadhaa katika {lrl_name} na lebo zao za hisia za Kiingereza:" # Swahili
        if lang_code == "ha":
            few_shot_header_lrl = f"Ga wasu misalai a cikin {lrl_name} tare da alamun ra'ayinsu na Turanci:"
        elif lang_code == "pt":
            few_shot_header_lrl = f"Aqui estão alguns exemplos em {lrl_name} com seus rótulos de sentimento em inglês:"
        else: # Fallback for header
            few_shot_header_lrl = f"Here are some examples in {lrl_name} with their English sentiment labels:"
            
        prompt += f"{few_shot_header_lrl}\n"
        prompt += f"Mfano 1:\nKiingereza Nakala: '{english_example_positive_text}'\nLebo ya Hisia ya Kiingereza: positive\n\n" # Swahili example structure
        if lang_code == "ha":
            prompt += f"Misali 1:\nTuranci Rubutu: '{english_example_positive_text}'\nAlamar Ra'ayi ta Turanci: positive\n\n"
        elif lang_code == "pt":
             prompt += f"Exemplo 1:\nTexto em Inglês: '{english_example_positive_text}'\nRótulo de Sentimento em Inglês: positive\n\n"
        else: # Fallback structure
            prompt += f"Example 1:\nEnglish Text: '{english_example_positive_text}'\nEnglish Sentiment Label: positive\n\n"


        prompt += f"Mfano 2:\nKiingereza Nakala: '{english_example_negative_text}'\nLebo ya Hisia ya Kiingereza: negative\n\n" # Swahili
        if lang_code == "ha":
            prompt += f"Misali 2:\nTuranci Rubutu: '{english_example_negative_text}'\nAlamar Ra'ayi ta Turanci: negative\n\n"
        elif lang_code == "pt":
            prompt += f"Exemplo 2:\nTexto em Inglês: '{english_example_negative_text}'\nRótulo de Sentimento em Inglês: negative\n\n"
        else: # Fallback
            prompt += f"Example 2:\nEnglish Text: '{english_example_negative_text}'\nEnglish Sentiment Label: negative\n\n"


        prompt += f"Mfano 3:\nKiingereza Nakala: '{english_example_neutral_text}'\nLebo ya Hisia ya Kiingereza: neutral\n\n" # Swahili
        if lang_code == "ha":
            prompt += f"Misali 3:\nTuranci Rubutu: '{english_example_neutral_text}'\nAlamar Ra'ayi ta Turanci: neutral\n\n"
        elif lang_code == "pt":
            prompt += f"Exemplo 3:\nTexto em Inglês: '{english_example_neutral_text}'\nRótulo de Sentimento em Inglês: neutral\n\n"
        else: # Fallback
            prompt += f"Example 3:\nEnglish Text: '{english_example_neutral_text}'\nEnglish Sentiment Label: neutral\n\n"


    # The final part asking to classify the current text should also be in LRL.
    classify_text_header_lrl = f"Sasa, ainisha maandishi yafuatayo ya {lrl_name}:" # Swahili
    text_label_lrl = f"{lrl_name} Nakala:" # Swahili
    output_label_lrl = "Lebo ya Hisia ya Kiingereza:" # Swahili (instructing for English output label)

    if lang_code == "ha":
        classify_text_header_lrl = f"Yanzu, rarraba rubutun {lrl_name} mai zuwa:"
        text_label_lrl = f"{lrl_name} Rubutu:"
        output_label_lrl = "Alamar Ra'ayi ta Turanci:"
    elif lang_code == "pt":
        classify_text_header_lrl = f"Agora, classifique o seguinte texto em {lrl_name}:"
        text_label_lrl = f"{lrl_name} Texto:"
        output_label_lrl = "Rótulo de Sentimento em Inglês:"
    # Fallback for other languages not explicitly defined
    elif lang_code != 'sw': # to avoid repeating swahili default
        classify_text_header_lrl = f"Now, classify the following {lrl_name} text:"
        text_label_lrl = f"{lrl_name} Text:"
        output_label_lrl = "English Sentiment Label:"


    prompt += f"{classify_text_header_lrl}\n"
    prompt += f"{text_label_lrl}\n'{text_escaped}'\n"
    prompt += f"{output_label_lrl}"
    return prompt

def extract_label(output_text: str, lang_code: str = "en", model_name: str = "") -> str:
    """
    Extracts a sentiment label from the model output.
    Prioritizes English labels. If LRL label is found, maps it to English.
    """
    if not output_text or not isinstance(output_text, str):
        # Ensure output_text is a string for logging, or provide a placeholder
        log_output_text = str(output_text) if output_text else "[empty_output]"
        logger.warning(f"extract_label received invalid output_text: {log_output_text}")
        return "unknown"

    text_to_process = output_text.lower().strip()

    # Define initial prefixes
    prefixes_to_remove = [
        "english sentiment label:", "sentiment label:", "sentiment:", "label:",
        "the sentiment is", "this text is", "output:", "answer:",
        "based on the text, the sentiment is", "i would classify this as",
        "hisia:", "ra'ayi:", "sentimento:", "lebo ya kiingereza:", # LRL examples
    ]
    # Dynamically add language-specific prefixes
    if lang_code and lang_code in LANG_NAMES:
        lang_name_lower = get_language_name(lang_code).lower()
        prefixes_to_remove.extend([
            f"{lang_name_lower} sentiment label:",
            f"the english label for the {lang_name_lower} text is:",
        ])

    for prefix in prefixes_to_remove:
        if text_to_process.startswith(prefix):
            text_to_process = text_to_process[len(prefix):].strip()
    
    # Strip various quote and bracket characters
    # Using a raw string for the strip characters just in case, though not strictly necessary here.
    text_to_process = text_to_process.strip(r'''\'\"`[](){}<>* ''') 
    
    # Remove trailing punctuation that might be attached to the label
    if text_to_process.endswith(('.', '!', '?')):
        if len(text_to_process) > 1: # Avoid stripping if it's only a punctuation mark
            text_to_process = text_to_process[:-1].strip()

    # 1. Check for direct English label match (whole string)
    if text_to_process in ENGLISH_SENTIMENT_LABELS:
        return text_to_process

    # 2. Check if the processed text *starts with* an English label
    for label in ENGLISH_SENTIMENT_LABELS:
        if text_to_process.startswith(label):
            # Ensure it's the full word or followed by a non-alphabetic character
            if len(text_to_process) == len(label) or not text_to_process[len(label)].isalpha():
                    return label
    
    # 3. If lang_code is not 'en', check for LRL labels and map them to English
    if lang_code != "en" and lang_code in SENTIMENT_LABELS_LRL:
        current_lrl_map = SENTIMENT_LABELS_LRL.get(lang_code, {})
        # Create a reverse map from LRL translation (lowercase) to English label
        lrl_to_en_map = {}
        for en_label, lrl_translation in current_lrl_map.items():
            if en_label != "unknown": # Don't map LRL "unknown" to English "unknown" via this path
                lrl_to_en_map[lrl_translation.lower()] = en_label
        
        # Check for direct LRL label match (whole string)
        if text_to_process in lrl_to_en_map:
            return lrl_to_en_map[text_to_process] # Map to English

        # Check if the processed text *starts with* an LRL label
        for lrl_label_text_lower, en_equivalent in lrl_to_en_map.items():
            if text_to_process.startswith(lrl_label_text_lower):
                if len(text_to_process) == len(lrl_label_text_lower) or not text_to_process[len(lrl_label_text_lower)].isalpha():
                    return en_equivalent # Map to English
    
    # 4. Fallback: if any English label is a substring (less precise, check parts)
    parts = re.split(r'\\s+|\\.|,|;|!', text_to_process) # Split by common delimiters
    for part in parts:
        cleaned_part = part.strip(r'''\'\"`[](){}<>* ''') # Strip again after splitting
        if cleaned_part in ENGLISH_SENTIMENT_LABELS:
            return cleaned_part
            
    # Use more robust f-string formatting for logging complex strings
    log_original_output = repr(output_text) 
    log_processed_text = repr(text_to_process)
    logger.debug(
        f"Could not extract a recognized sentiment label. Original: {log_original_output}, Processed: {log_processed_text}, Model: {model_name}, Lang: {lang_code}. Defaulting to 'unknown'."
    )
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