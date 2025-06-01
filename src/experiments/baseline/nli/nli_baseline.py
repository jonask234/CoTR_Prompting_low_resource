from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import re
import os
from typing import Tuple, Dict, List, Any, Optional
import logging
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Define NLI labels (model is expected to output these English strings)
EXPECTED_NLI_LABELS = ["entailment", "neutral", "contradiction", "unknown"]

# Mapping from XNLI dataset numeric labels (if needed by data loader)
# 0: entailment, 1: neutral, 2: contradiction
NLI_LABEL_MAP_FROM_NUMERIC = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

def initialize_model(model_name: str) -> Tuple[Any, Any]:
    """
    Initialize a model and tokenizer, specifying cache directory and robust pad_token handling.
    """
    print(f"Initializing NLI baseline model: {model_name}...")
    cache_path = "/work/bbd6522/cache_dir"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto", 
        trust_remote_code=True,
        cache_dir=cache_path,
        # Add low_cpu_mem_usage to potentially help with large models
        low_cpu_mem_usage=True 
    )
    
    # Robustly set pad_token and model.config.pad_token_id
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logging.info(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token}) for {model_name}")
        else:
            # If no eos_token, add a new pad token
            logging.warning(f"No pad_token or eos_token found for {model_name}. Adding a new [PAD] token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Important: Resize model embeddings if a new token is added
            model.resize_token_embeddings(len(tokenizer))
            logging.info(f"Resized model token embeddings for the new [PAD] token in {model_name}.")

    # Ensure model.config.pad_token_id is also aligned with the tokenizer
    if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id
        logging.info(f"Aligned model.config.pad_token_id with tokenizer.pad_token_id ({tokenizer.pad_token_id}) for {model_name}")

    logging.info(f"Successfully loaded {model_name}")
    return tokenizer, model

def generate_nli_prompt(
    premise: str,
    hypothesis: str,
    instruction_lang_code: str = "en", # Language for the prompt instructions & examples
    use_few_shot: bool = True
) -> str:
    """
    Generate a prompt for NLI task based on the old script's structure.
    The `instruction_lang_code` determines the language of instructions.
    Few-shot examples are ALWAYS in English.
    The model is always asked to output one of the English labels: 'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'.
    """
    
    # English few-shot examples (used regardless of instruction_lang_code)
    english_few_shot_examples_list = [
        {"premise": "The chef is cooking a meal in the kitchen.", "hypothesis": "The chef is preparing food.", "answer": "ENTAILMENT"},
        {"premise": "The boy is playing soccer in the park.", "hypothesis": "The boy is swimming in a pool.", "answer": "CONTRADICTION"},
        {"premise": "The woman is walking down the street.", "hypothesis": "She is going to the grocery store.", "answer": "NEUTRAL"}
    ]
    english_example_header = "Examples (English text, English answer):"
    english_premise_key = "Text"
    english_hypothesis_key = "Hypothesis"
    english_answer_key = "Answer"
    
    # Select instruction block based on instruction_lang_code
    if instruction_lang_code == "sw":  # Swahili
        instruction_block = """Maandishi: '<PREMISE>'
Hipothesia: '<HYPOTHESIS>'

Amua kama hipothesia ni ENTAILMENT, CONTRADICTION, au NEUTRAL kuhusiana na maandishi.

Ufafanuzi:
- ENTAILMENT: Hipothesia lazima iwe kweli ikiwa maandishi ni kweli.
- CONTRADICTION: Hipothesia haiwezi kuwa kweli ikiwa maandishi ni kweli.
- NEUTRAL: Hipothesia inaweza kuwa kweli au si kweli kutokana na dhana.

Jibu kwa kutumia MOJA TU ya lebo hizi za Kiingereza: 'ENTAILMENT', 'CONTRADICTION', au 'NEUTRAL'.
Jibu lako lote lazima liwe neno MOJA TU kati ya hayo matatu ya Kiingereza. Hakuna maandishi mengine, maelezo, au alama za uakifishaji zinazoruhusiwa."""
        # Use English examples even for Swahili instructions
        example_header_to_use = english_example_header
        premise_key_for_examples = english_premise_key
        hypothesis_key_for_examples = english_hypothesis_key
        answer_key_for_examples = english_answer_key
        few_shot_examples_to_use = english_few_shot_examples_list

    elif instruction_lang_code == "ur":  # Urdu
        instruction_block = """متن: '<PREMISE>'
مفروضہ: '<HYPOTHESIS>'

یہ تعین کریں کہ مفروضہ متن کے حوالے سے ENTAILMENT، CONTRADICTION، یا NEUTRAL ہے۔

تعریفیں:
- ENTAILMENT: اگر متن سچ ہے تو مفروضہ بھی ضرور سچ ہے۔
- CONTRADICTION: اگر متن سچ ہے تو مفروضہ سچ نہیں ہو سکتا۔
- NEUTRAL: مفروضہ سچ یا غلط ہو سکتا ہے؛ متن کافی معلومات فراہم نہیں کرتا۔

اپنا جواب صرف ان تین انگریزی لیبلز میں سے ایک کے طور پر دیں: 'ENTAILMENT'، 'CONTRADICTION'، یا 'NEUTRAL'۔
آپ کا پورا جواب صرف ان تین الفاظ میں سے ایک ہونا چاہیے۔ کسی دوسرے متن، وضاحت یا رموز اوقاف کی اجازت نہیں ہے۔"""
        # Use English examples even for Urdu instructions
        example_header_to_use = english_example_header
        premise_key_for_examples = english_premise_key
        hypothesis_key_for_examples = english_hypothesis_key
        answer_key_for_examples = english_answer_key
        few_shot_examples_to_use = english_few_shot_examples_list

    else:  # Default to English instructions and English examples
        instruction_block = """Text: '<PREMISE>'
Hypothesis: '<HYPOTHESIS>'

Determine if the hypothesis is ENTAILMENT, CONTRADICTION, or NEUTRAL with respect to the text. 

Definitions:
- ENTAILMENT: The hypothesis must be true if the text is true.
- CONTRADICTION: The hypothesis cannot be true if the text is true.
- NEUTRAL: The hypothesis might be true or false; the text doesn't provide enough information.

Provide your answer as EXACTLY one of these three English labels: 'ENTAILMENT', 'CONTRADICTION', or 'NEUTRAL'.
Your entire response must be only one of these three words. No other text, explanation, or punctuation is allowed."""
        example_header_to_use = english_example_header
        premise_key_for_examples = english_premise_key
        hypothesis_key_for_examples = english_hypothesis_key
        answer_key_for_examples = english_answer_key
        few_shot_examples_to_use = english_few_shot_examples_list

    # Replace placeholders in the instruction block with actual premise/hypothesis
    # Ensure premise and hypothesis are escaped for string formatting if they contain quotes
    safe_premise = premise.replace("'", "\'")
    safe_hypothesis = hypothesis.replace("'", "\'")
    current_instruction = instruction_block.replace('<PREMISE>', safe_premise).replace('<HYPOTHESIS>', safe_hypothesis)

    full_prompt = [current_instruction]

    if use_few_shot:
        full_prompt.append(f"\n{example_header_to_use}")
        for ex in few_shot_examples_to_use:
            # Ensure example premise/hypothesis are also escaped
            safe_ex_premise = ex['premise'].replace("'", "\'")
            safe_ex_hypothesis = ex['hypothesis'].replace("'", "\'")
            full_prompt.append(f"{premise_key_for_examples}: '{safe_ex_premise}'\n{hypothesis_key_for_examples}: '{safe_ex_hypothesis}'\n{answer_key_for_examples}: {ex['answer']}\n")
    
    # The final query part (Text to classify and Answer prompt) should match the language of the main instruction block.
    # For NLI, the prompt structure usually ends with "Label:" or similar, which should be in the instruction_lang_code.
    if instruction_lang_code == "sw":
        final_answer_prompt_key = "Jibu"
    elif instruction_lang_code == "ur":
        final_answer_prompt_key = "جواب" # Ensure this is the correct Urdu for "Answer/Label"
    else: # English
        final_answer_prompt_key = "Answer"

    full_prompt.append(f"\n{final_answer_prompt_key}:") # Final prompt for the model to complete
    
    return "\n".join(full_prompt)

def extract_nli_label(output_text: str) -> str:
    """
    Extract NLI label from model output text, adapted from the old script.
    Prioritizes exact matches to 'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL' (case-insensitive for matching).
    """
    text_cleaned = output_text.strip()
    text_lower = text_cleaned.lower()

    # Priority 1: Check for exact full-string matches of the expected English labels (case-insensitive)
    for label in EXPECTED_NLI_LABELS:
        if text_lower == label:
            logging.debug(f"Extracted NLI label (exact match): '{label}' from '{text_cleaned}'")
            return label

    # Priority 2: Check if the raw output *contains* the capitalized versions (as requested by prompt)
    # This can catch cases where the model might add minimal extra tokens like a period.
    if "ENTAILMENT" in text_cleaned:
        logging.debug(f"Extracted NLI label (contains capitalized): 'entailment' from '{text_cleaned}'")
        return "entailment"
    elif "CONTRADICTION" in text_cleaned:
        logging.debug(f"Extracted NLI label (contains capitalized): 'contradiction' from '{text_cleaned}'")
        return "contradiction"
    elif "NEUTRAL" in text_cleaned:
        logging.debug(f"Extracted NLI label (contains capitalized): 'neutral' from '{text_cleaned}'")
        return "neutral"
    
    # Priority 3: Find the label closest to the start of the lowercase response if multiple are present
    label_positions = {}
    for label in EXPECTED_NLI_LABELS:
        pos = text_lower.find(label)
        if pos != -1:
            label_positions[label] = pos
    
    if label_positions:
        # Return the label that appears first in the text
        best_match_by_pos = min(label_positions.items(), key=lambda x: x[1])[0]
        logging.debug(f"Extracted NLI label (first occurrence): '{best_match_by_pos}' from '{text_cleaned}'")
        return best_match_by_pos
    
    # Fallback: Old script had language-specific keywords. Retaining some simple ones for English as a last resort.
    # These are less reliable as the prompt strictly asks for the three main labels.
    if any(term in text_lower for term in ['yes', 'follows', 'must be true', 'is true', 'has to be true']):
        logging.debug(f"Extracted NLI label (synonym): 'entailment' from '{text_cleaned}'")
        return 'entailment'
    elif any(term in text_lower for term in ['no', 'not true', 'cannot be true', 'opposite', 'disagree']):
        logging.debug(f"Extracted NLI label (synonym): 'contradiction' from '{text_cleaned}'")
        return 'contradiction'
    
    logging.warning(f"Could not reliably extract NLI label from '{text_cleaned[:100]}...'. Defaulting to 'unknown'.")
    return "unknown" # Default to unknown if no clear label is found

def process_nli_baseline(
    model: Any,
    tokenizer: Any, 
    premise: str, 
    hypothesis: str, 
    instruction_lang_code: str, # Language for the prompt structure
    use_few_shot: bool,
    generation_params: Dict[str, Any],
    model_name_for_logging: str # For more informative logs
) -> Dict[str, Any]:
    """Process a single NLI example."""
    start_time = time.time()
    
    prompt = generate_nli_prompt(
        premise, 
        hypothesis, 
        instruction_lang_code=instruction_lang_code, 
        use_few_shot=use_few_shot
    )
    
    # Log the first 300 characters of the prompt for inspection
    logging.debug(f"NLI Prompt ({instruction_lang_code}, Few-shot: {use_few_shot}) for {model_name_for_logging}:\n{prompt[:300]}...")
        
    # Ensure prompt is a single string
    if not isinstance(prompt, str):
        logging.error(f"CRITICAL: Prompt is not a string! Type: {type(prompt)}. Value: {prompt}")
        # Handle error appropriately, e.g., by returning an error state
        return {"predicted_label": "error_prompt_type", "raw_model_output": "Prompt was not a string", "runtime": 0.0}

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Move all tensors in inputs to the model's device
    # And ensure they are indeed tensors of the correct type
    if 'input_ids' not in inputs or not hasattr(inputs['input_ids'], 'to'):
        logging.error(f"CRITICAL: 'input_ids' not found in tokenizer output or not a tensor. Output: {inputs}")
        return {"predicted_label": "error_tokenization", "raw_model_output": "Tokenization failed to produce input_ids tensor", "runtime": 0.0}

    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    # --- DEBUG PRINTS ---
    print(f"DEBUG NLI BASELINE (process_nli_baseline): type(prompt): {type(prompt)}")
    print(f"DEBUG NLI BASELINE (process_nli_baseline): input_ids type: {type(input_ids)}, shape: {input_ids.shape}, dtype: {input_ids.dtype}, device: {input_ids.device}")
    print(f"DEBUG NLI BASELINE (process_nli_baseline): attention_mask type: {type(attention_mask)}, shape: {attention_mask.shape}, dtype: {attention_mask.dtype}, device: {attention_mask.device}")
    # print(f"DEBUG NLI BASELINE: input_ids content (first 50 tokens): {input_ids[:, :50]}") # Can be very verbose
    # --- END DEBUG PRINTS ---

    # Use generation parameters
    actual_do_sample = generation_params.get("do_sample", True) # Default to True if not specified
    if generation_params.get("temperature", 0.1) <= 0.01 and "do_sample" not in generation_params : # if temp is very low and do_sample not explicitly set
         actual_do_sample = False # Override to False for greedy decoding if temp is effectively 0

    raw_model_response = "[Generation Error]"
    predicted_label = "error" # Default in case of issues

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids, # Pass the tensor directly
                attention_mask=attention_mask, # Pass the tensor directly
                max_new_tokens=generation_params.get("max_tokens", 10), # Default if not provided
                temperature=generation_params.get("temperature", 0.1),
                do_sample=actual_do_sample,
                top_p=generation_params.get("top_p", 0.9), # Added top_p
                top_k=generation_params.get("top_k", 40),   # Added top_k
                repetition_penalty=generation_params.get("repetition_penalty", 1.0), # Added repetition_penalty
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        raw_model_response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        logging.debug(f"Raw NLI model response from {model_name_for_logging}: '{raw_model_response}'")
        predicted_label = extract_nli_label(raw_model_response)
        logging.debug(f"Extracted NLI label for {model_name_for_logging}: '{predicted_label}'")

    except Exception as e:
        logging.error(f"Error during NLI processing for {model_name_for_logging}, premise '{premise[:50]}...': {e}", exc_info=True)
        # predicted_label remains "error", raw_model_response might hold error string or last value
        if isinstance(e, torch.cuda.OutOfMemoryError):
            raw_model_response = "[CUDA OOM Error]"
            torch.cuda.empty_cache() # Attempt to clear cache
        else:
            raw_model_response = f"[Exception: {str(e)}]"
        
    runtime = time.time() - start_time
    
    return {
        "predicted_label": predicted_label,
        "raw_model_output": raw_model_response, # Changed key for consistency
        "runtime_seconds": runtime
    }

def evaluate_nli_baseline(
    model_name: str,
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    data_lang_code: str, # The actual language of the premise/hypothesis in samples_df
    prompt_in_lrl: bool,
    use_few_shot: bool,  
    generation_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Evaluate NLI performance using the baseline approach.
    `data_lang_code` is the language of the text samples.
    `prompt_in_lrl` determines if instructions are in `data_lang_code` or English.
    """
    results = []
    
    # Determine the language for prompt instructions
    # If prompt_in_lrl is True, instructions are in data_lang_code.
    # Otherwise, instructions are in English.
    instruction_lang_for_prompt = data_lang_code if prompt_in_lrl and data_lang_code != "en" else "en"
    
    shot_description = "few-shot" if use_few_shot else "zero-shot"
    prompt_lang_description = "LRL-instruct" if prompt_in_lrl and data_lang_code != 'en' else "EN-instruct"

    logging.info(f"Evaluating NLI for Model: {model_name}, Data Lang: {data_lang_code}, Prompt Instruct Lang: {instruction_lang_for_prompt}, Shot: {shot_description}")
    
    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing NLI {data_lang_code} ({prompt_lang_description}, {shot_description})"):
        premise = str(row['premise'])
        hypothesis = str(row['hypothesis'])
        
        # Ensure ground_truth_label is string (e.g., "entailment")
        # Input df might have numeric labels (0,1,2) or string labels.
        if 'label' in row:
            gt_val = row['label']
            if isinstance(gt_val, (int, np.integer)): # if it's numeric (0,1,2)
                ground_truth_label = NLI_LABEL_MAP_FROM_NUMERIC.get(gt_val, "unknown")
            else: # assume it's already a string label
                ground_truth_label = str(gt_val).lower().strip()
        else:
            ground_truth_label = "unknown" # Fallback if no label column

        if not premise or not hypothesis:
            logging.warning(f"Skipping sample {idx} due to missing premise or hypothesis.")
            results.append({
                'id': row.get('id', idx),
                'premise': premise,
                'hypothesis': hypothesis,
                'ground_truth_label': ground_truth_label,
                'predicted_label': "error_empty_input",
                'raw_model_output': "Input premise or hypothesis was empty.",
                'runtime_seconds': 0,
                'data_language': data_lang_code,
                'prompt_instruction_language': instruction_lang_for_prompt,
                'shot_type': shot_description
            })
            continue
            
        result_dict = process_nli_baseline(
            model,
            tokenizer, 
            premise, 
            hypothesis, 
            instruction_lang_code=instruction_lang_for_prompt,
            use_few_shot=use_few_shot,
            generation_params=generation_params,
            model_name_for_logging=model_name # Pass the string model name
        )
            
        results.append({
            'id': row.get('id', idx), # Assuming an 'id' column might exist
            'premise': premise,
            'hypothesis': hypothesis,
            'ground_truth_label': ground_truth_label,
            'predicted_label': result_dict["predicted_label"],
            'raw_model_output': result_dict["raw_model_output"],
            'runtime_seconds': result_dict["runtime_seconds"],
            'data_language': data_lang_code, # Actual language of the text
            'prompt_instruction_language': instruction_lang_for_prompt, # Language of the prompt's instructions
            'shot_type': shot_description
        })
    
    return pd.DataFrame(results)

def calculate_nli_metrics(results_df):
    """
    Calculate accuracy and F1 scores for NLI predictions.
    
    Args:
        results_df: DataFrame with 'predicted_label' and 'gold_label' columns
    
    Returns:
        Dictionary with accuracy and F1 metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    # Debug: Print the first few rows to check format
    print("\nFirst 5 rows of results DataFrame:")
    print(results_df[['premise', 'hypothesis', 'gold_label', 'predicted_label']].head().to_string())
    
    # Count label distribution for both gold and predicted
    print("\nGold label distribution:")
    print(results_df['gold_label'].value_counts())
    
    print("\nPredicted label distribution:")
    print(results_df['predicted_label'].value_counts())
    
    # Make sure we have valid labels to calculate metrics
    if 'predicted_label' not in results_df.columns or 'gold_label' not in results_df.columns:
        print("ERROR: Missing 'predicted_label' or 'gold_label' columns in results DataFrame")
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'class_metrics': {}}
    
    if results_df['predicted_label'].isnull().all() or results_df['gold_label'].isnull().all():
        print("ERROR: All null values in 'predicted_label' or 'gold_label' columns")
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'class_metrics': {}}
    
    # Create a copy to avoid modifying the original DataFrame
    eval_df = results_df.copy()
    
    # Check if gold_label is numeric and predicted_label is string
    if pd.api.types.is_numeric_dtype(eval_df['gold_label']) and pd.api.types.is_string_dtype(eval_df['predicted_label']):
        print("Converting numeric gold labels to strings for compatibility...")
        # Map from numeric labels to string labels
        label_map = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }
        # Convert gold_label from numeric to string format
        eval_df['gold_label'] = eval_df['gold_label'].map(label_map)
        print("After conversion, gold label distribution:")
        print(eval_df['gold_label'].value_counts())
    
    # Calculate accuracy
    accuracy = accuracy_score(eval_df['gold_label'], eval_df['predicted_label'])
    
    # Calculate macro F1 score
    macro_f1 = f1_score(
        eval_df['gold_label'], 
        eval_df['predicted_label'], 
        average='macro', 
        zero_division=0
    )
    
    # Get detailed metrics for each class
    class_report = classification_report(
        eval_df['gold_label'], 
        eval_df['predicted_label'], 
        output_dict=True, 
        zero_division=0
    )
    
    # Extract class metrics
    class_metrics = {}
    for label in ["entailment", "neutral", "contradiction"]:
        if label in class_report:
            class_metrics[label] = {
                'precision': class_report[label]['precision'],
                'recall': class_report[label]['recall'],
                'f1': class_report[label]['f1-score']
            }
    
    # Print classification report for debugging
    print("\nDetailed Classification Report:")
    print(classification_report(eval_df['gold_label'], eval_df['predicted_label'], zero_division=0))
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'class_metrics': class_metrics
    } 