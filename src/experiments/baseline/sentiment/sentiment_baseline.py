import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os
import re

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

def preprocess_text(text: str, lang_code: str) -> str:
    """
    Preprocess text based on language-specific characteristics.
    
    Args:
        text: The input text to preprocess
        lang_code: Language code (e.g., 'sw' for Swahili, 'ha' for Hausa)
        
    Returns:
        Preprocessed text
    """
    # Common preprocessing for all languages
    processed_text = text.strip()
    
    # Language-specific preprocessing
    if lang_code == 'sw':  # Swahili-specific processing
        # Handle emojis - preserve them as they're important for sentiment
        # But add spaces around them to help tokenization
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F700-\U0001F77F"  # alchemical symbols
                                   u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                                   u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                   u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                   u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                   u"\U00002702-\U000027B0"  # Dingbats
                                   u"\U000024C2-\U0001F251" 
                                   "]+")
        # Add spaces around emojis to help tokenization
        processed_text = re.sub(emoji_pattern, lambda m: f" {m.group(0)} ", processed_text)
        # Clean up multiple spaces
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
    elif lang_code == 'ha':  # Hausa-specific processing
        # Hausa typically has fewer emojis but may have specific characters
        # Normalize any Hausa-specific characters if needed
        pass
    
    return processed_text

def generate_sentiment_prompt(text: str, lang_code: str = "en") -> str:
    """Generate a zero-shot prompt for sentiment classification, with structured format."""
    # Preprocess text based on language
    processed_text = preprocess_text(text, lang_code)
    
    # Enhanced prompt with language-specific context when known
    if lang_code == 'sw':  # Swahili-specific prompt in English
        prompt = f"""Text: '{processed_text}'

Analyze the sentiment of the Swahili text above.
Consider Swahili linguistic patterns and social media expressions (including emojis).
Respond with only one of these labels: positive, negative, or neutral.

Examples:
Text: 'Ninafurahia sana hali ya hewa leo! 😊' 
Sentiment: positive

Text: 'Sikufurahishwa na huduma hii kabisa. 😡'
Sentiment: negative

Text: 'Mimi ni mwanafunzi wa chuo kikuu.'
Sentiment: neutral

Sentiment:"""
    elif lang_code == 'ha':  # Hausa-specific prompt in English
        prompt = f"""Text: '{processed_text}'

Analyze the sentiment of the Hausa text above.
Consider Hausa expressions and linguistic patterns.
Respond with only one of these labels: positive, negative, or neutral.

Examples:
Text: 'Na yi murna da jin wannan labari.'
Sentiment: positive

Text: 'Ban yarda da wannan lamari ba.'
Sentiment: negative

Text: 'Ina zuwa kasuwa gobe.'
Sentiment: neutral

Sentiment:"""
    else:
        # Standard prompt for other languages
        prompt = f"""Text: '{processed_text}'

Analyze the sentiment of the text above.
Respond with only one of these labels: positive, negative, or neutral.

Sentiment:"""
    return prompt

def generate_lrl_instruct_sentiment_prompt(text: str, lang_code: str) -> str:
    """Generate a prompt in the low-resource language with structured format."""
    # Preprocess text based on language
    processed_text = preprocess_text(text, lang_code)
    
    # Explicitly mention required English labels
    en_labels = "positive, negative, or neutral" 
    
    if lang_code == 'ha': # Hausa
        instructions = f"Text: '{processed_text}'\n\nBincika yanayin rubutu a sama. Ka amsa da ɗaya daga cikin kalmomi na turanci kawai: {en_labels}."
        
        # Add few-shot examples for Hausa
        prompt = f"""{instructions}

Misalai:
Text: 'Na yi farin ciki da jin labarin nasara.'
Sentiment: positive

Text: 'Ba na son yadda aka yi wannan abu ba.'
Sentiment: negative

Text: 'Zan tafi gobe.'
Sentiment: neutral

Sentiment:"""
        
    elif lang_code == 'sw': # Swahili
        # More explicit instructions for Swahili with emphasis on accuracy and handling emojis
        instructions = f"""Text: '{processed_text}'

Chunguza kwa makini hisia katika maandishi yaliyotolewa, ukizingatia pia emoji na ishara zingine.
Jibu kwa Kiingereza kwa kutumia MOJA TU kati ya maneno haya: {en_labels}.
Hakikisha majibu yako ni sahihi na sio tu kupendelea hisia chanya."""
        
        # Add balanced examples with emojis for Swahili
        prompt = f"""{instructions}

Mifano:
Text: 'Ninafurahia sana hali ya hewa leo! 😊'
Sentiment: positive

Text: 'Sikufurahishwa na huduma hii kabisa, ilikuwa mbaya sana. 😠'
Sentiment: negative

Text: 'Labda nitaenda dukani kesho, sijaamua bado.'
Sentiment: neutral

Text: 'Nilipata hasara kubwa katika biashara hii. 😢'
Sentiment: negative

Text: 'Mimi ni mwanafunzi wa chuo kikuu.'
Sentiment: neutral

Sentiment:"""
        
    elif lang_code == 'eng':
         print(f"WARN: LRL instructions requested for English text. Using standard English prompt.")
         return generate_sentiment_prompt(processed_text)
    else:
        print(f"WARN: LRL instructions not defined for {lang_code}. Falling back to English prompt.")
        return generate_sentiment_prompt(processed_text)
    
    return prompt

def process_sentiment_baseline(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    text: str,
    prompt_in_lrl: bool = False,
    lang_code: str = 'en',
    max_new_tokens: int = 10,
    max_input_length: int = 1024
) -> str:
    """
    Process a text sample for sentiment classification directly (baseline).
    Returns the predicted label string (or an error indicator).
    """
    # Choose prompt based on flag
    if prompt_in_lrl:
        prompt = generate_lrl_instruct_sentiment_prompt(text, lang_code)
    else:
        prompt = generate_sentiment_prompt(text, lang_code)

    # Tokenize with truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate prediction with improved sampling parameters
    with torch.no_grad():
        # Language-specific generation strategy
        if "aya" in model.config._name_or_path.lower():
            # Multiple generations for more reliability
            all_labels = []
            # More generations for Swahili with emojis
            num_generations = 5 if lang_code == 'sw' else 3
            
            for _ in range(num_generations):
                # Adjust temperature based on language characteristics
                temp = 0.9 if lang_code == 'sw' else 0.8
                
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )
                # Decode only the newly generated tokens
                output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                label = extract_label(output_text, lang_code, text)  # Also pass original text for context
                all_labels.append(label)
            
            # Count occurrences of each label
            label_counts = {}
            for label in all_labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
            
            # Take the most frequent label
            predicted_label = max(label_counts.items(), key=lambda x: x[1])[0]
            
            # If there's a tie or all are [Unknown], use first non-[Unknown] label
            if predicted_label == "[Unknown]" or max(label_counts.values()) == 1:
                for label in all_labels:
                    if label != "[Unknown]":
                        predicted_label = label
                        break
        else:
            # Standard generation for other cases
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            # Decode only the newly generated tokens
            output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            predicted_label = extract_label(output_text, lang_code, text)
                
    return predicted_label

def extract_label(output_text: str, lang_code: str = "en", original_text: str = "") -> str:
    """
    Extract sentiment label from model output with improved logic.
    
    Args:
        output_text: The model's generated output text
        lang_code: Language code for language-specific handling
        original_text: The original input text (for emoji analysis etc.)
    """
    # Normalize the output text
    predicted_text = output_text.strip().lower()
    final_label = "[Unknown]"
    
    # Remove common prefixes that might appear in responses
    prefixes_to_remove = ["sentiment:", "the sentiment is", "answer:"]
    for prefix in prefixes_to_remove:
        if predicted_text.startswith(prefix):
            predicted_text = predicted_text[len(prefix):].strip()
    
    # First try exact matches or matches at the beginning of the response
    for label in EXPECTED_LABELS:
        if predicted_text == label or predicted_text.startswith(label + " "):
            final_label = label
            break
    
    # If no match found yet, try more flexible matching
    if final_label == "[Unknown]":
        if "positiv" in predicted_text:
            final_label = "positive"
        elif "negativ" in predicted_text:
            final_label = "negative"
        elif "neutr" in predicted_text:
            final_label = "neutral"
    
    # Still no match? Look for any occurrence of the labels
    if final_label == "[Unknown]":
        for label in EXPECTED_LABELS:
            if label in predicted_text:
                final_label = label
                break
    
    # Language-specific corrections for known biases
    if lang_code == 'sw':  # Swahili-specific handling
        # For Swahili with Aya, if we detect "neutral" with low confidence, check for positive signals
        if final_label == "neutral" and any(word in predicted_text for word in ["good", "nzuri", "furah"]):
            final_label = "positive"
        
        # Also correct common signals in Swahili that might be missed
        if final_label == "[Unknown]":
            # Positive Swahili signals
            if any(word in predicted_text for word in ["nzuri", "furah", "pend", "cheka"]):
                final_label = "positive"
            # Negative Swahili signals
            elif any(word in predicted_text for word in ["mbaya", "huzun", "chuki", "sikufurah"]):
                final_label = "negative"
            # Neutral Swahili signals
            elif any(word in predicted_text for word in ["katikati", "kawaida"]):
                final_label = "neutral"
                
        # Check for emojis in original Swahili text and use them to influence prediction
        # if model's confidence is low (demonstrated by neutral or unknown predictions)
        if (final_label == "neutral" or final_label == "[Unknown]") and original_text:
            # Define emoji patterns
            positive_emojis = re.compile("["
                                        u"\U0001F600-\U0001F607"  # happy faces
                                        u"\U0001F60A-\U0001F60E"  # smiling faces
                                        u"\U0001F60D"  # heart eyes
                                        u"\U0001F917"  # hugging face
                                        u"\U0001F970"  # smiling face with hearts
                                        u"\U0001F929"  # star-struck
                                        u"\U0001F618"  # blowing kiss
                                        u"\U0001F44D"  # thumbs up
                                        u"\U0001F495-\U0001F49F"  # hearts
                                        u"\U0001F525"  # fire
                                        u"\U0001F389"  # party popper
                                        "]+")
                                        
            negative_emojis = re.compile("["
                                        u"\U0001F61E-\U0001F61F"  # sad faces
                                        u"\U0001F620-\U0001F623"  # angry faces
                                        u"\U0001F625"  # disappointed face
                                        u"\U0001F62D"  # crying face
                                        u"\U0001F631"  # screaming face
                                        u"\U0001F624"  # face with steam
                                        u"\U0001F44E"  # thumbs down
                                        u"\U0001F494"  # broken heart
                                        u"\U0001F4A9"  # pile of poo
                                        u"\U0001F92C"  # face with symbols on mouth
                                        "]+")
                                        
            # Check for presence of emojis
            if re.search(positive_emojis, original_text):
                final_label = "positive"
            elif re.search(negative_emojis, original_text):
                final_label = "negative"
                
    elif lang_code == 'ha':  # Hausa-specific handling
        # Add Hausa-specific label extraction logic similar to Swahili if needed
        if final_label == "[Unknown]":
            # Positive Hausa signals
            if any(word in predicted_text for word in ["farin ciki", "murna", "dadi"]):
                final_label = "positive"
            # Negative Hausa signals  
            elif any(word in predicted_text for word in ["bakin ciki", "jin haushin", "babu"]):
                final_label = "negative"
            # Neutral Hausa signals
            elif any(word in predicted_text for word in ["kare", "kawai"]):
                final_label = "neutral"
    
    # If still unknown, default to a more appropriate fallback
    if final_label == "[Unknown]":
        if lang_code == 'sw' and "aya" in output_text:
            # Aya model with Swahili language tends to overpredict neutral
            final_label = "positive"  # Change default for Swahili with Aya to counter neutral bias
        else:
            final_label = "neutral"  # Default fallback for truly ambiguous cases
                
    return final_label

def evaluate_sentiment_baseline(
    model_name: str, 
    samples_df: pd.DataFrame, 
    lang_code: str,
    prompt_in_lrl: bool = False
) -> pd.DataFrame:
    """Evaluates baseline sentiment, selecting prompt based on flag."""
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

    prompt_desc = "LRLInstruct" if prompt_in_lrl else "EnInstruct" # For logging
    
    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} baseline ({prompt_desc}) sentiment ({model_name})"):
        try:
            text = row["text"]
            ground_truth_label = row["label"]

            predicted_label = process_sentiment_baseline(
                tokenizer, model, text,
                prompt_in_lrl=prompt_in_lrl,
                lang_code=lang_code,
                max_input_length=safe_max_input_length
            )

            result = {
                "id": row.get("id", idx),
                "text": text,
                "ground_truth_label": ground_truth_label,
                "predicted_label": predicted_label,
                "language": lang_code,
                "prompt_language": "LRL" if prompt_in_lrl else "EN"
            }
            results.append(result)
        except Exception as e:
            print(f"ERROR processing sample {row.get('id', idx)}: {e}")
            continue

    if not results:
        print(f"WARNING: No results were successfully processed for {lang_code} with {model_name}.")
        return pd.DataFrame()

    return pd.DataFrame(results)