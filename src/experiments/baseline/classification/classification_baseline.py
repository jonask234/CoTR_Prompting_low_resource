import torch
import pandas as pd
from tqdm import tqdm
import os
import sys
from typing import Any, List

# Add project root to path to find model_initialization
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from model_initialization import initialize_model

def generate_lrl_classification_prompt(text: str, lang_code: str, labels: List[str]) -> str:
    """Generate a few-shot prompt (ENGLISH INSTRUCTIONS) for LRL text classification."""
    label_string = ", ".join(labels) # labels are English
    
    # Add few-shot examples (English examples for consistency)
    examples = ""
    if "sports" in labels:
        examples += """Example 1:
Text: The basketball team scored 92 points in their victory over the defending champions.
Category: sports

"""
    if "politics" in labels:
        examples += """Example 2:
Text: The president announced new economic policies during yesterday's press conference.
Category: politics

"""
    if "health" in labels:
        examples += """Example 3:
Text: Doctors recommend at least 30 minutes of exercise daily for cardiovascular health.
Category: health

"""
    if "entertainment" in labels:
        examples += """Example 4:
Text: The new movie broke box office records with its opening weekend sales.
Category: entertainment

"""
    if "business" in labels:
        examples += """Example 5:
Text: The company announced its quarterly earnings report, showing a 20% increase in revenue.
Category: business

"""
    
    prompt = f"""Classify the following {lang_code} text into one of these categories: {label_string}.
Respond with only the category name in English. Do not translate the text, just classify it directly.

{examples}Now classify this:
Text ({lang_code}): {text}

Category:"""
    return prompt

def generate_lrl_instruct_classification_prompt(text: str, lang_code: str, labels: List[str]) -> str:
    """Generate a few-shot prompt (LRL INSTRUCTIONS) for LRL text classification."""
    # Explicitly mention required English labels
    en_labels_string = ", ".join(labels) 
    
    # Create examples in the LRL with English label outputs
    examples = ""
    
    if lang_code == 'hau': # Hausa
        instructions = f"Karkasa rubutun da ke gaba zuwa ɗaya daga cikin wadannan ɗakunan Turanci: {en_labels_string}. Ka amsa da sunan ɗakin ne kawai a Turanci, ba tare da ƙarin bayani ba."
        
        # Add Hausa examples with more categories
        if "sports" in labels:
            examples += """Misali 1:
Text: Kungiyar Real Madrid ta doke Barcelona da ci 3-0 a wasan karshe na gasar.
Category: sports

"""
        if "politics" in labels:
            examples += """Misali 2:
Text: Shugaban kasa ya sanar da sabuwar manufofin tattalin arziki da zai sa kasar ta ci gaba.
Category: politics

"""
        if "health" in labels:
            examples += """Misali 3:
Text: Likitoci sun bayar da shawarar yin aikin motsa jiki na minti 30 kowace rana don lafiyar jiki.
Category: health

"""
        if "entertainment" in labels:
            examples += """Misali 4:
Text: Sabon fim din ya karya tarihin sayar da tiketi a satin farko na nunawa.
Category: entertainment

"""
        if "business" in labels:
            examples += """Misali 5:
Text: Kamfanin ya sanar da rahoton kudin shiga na kwatan nan, inda aka nuna karuwar kashi 20%.
Category: business

"""
    
    elif lang_code == 'swa': # Swahili
        instructions = f"Ainisha maandishi yafuatayo katika mojawapo ya vikundi hivi vya Kiingereza: {en_labels_string}. Jibu kwa kutumia jina la kikundi cha Kiingereza pekee, bila maelezo zaidi."
        
        # Add comprehensive Swahili examples
        if "sports" in labels:
            examples += """Mfano 1:
Text: Timu ya mpira wa miguu ya Simba imeshinda mchezo dhidi ya Yanga kwa mabao 2-0 katika fainali.
Category: sports

"""
        if "politics" in labels:
            examples += """Mfano 2:
Text: Rais ametangaza sera mpya ya uchumi katika mkutano wa waandishi wa habari jana.
Category: politics

"""
        if "health" in labels:
            examples += """Mfano 3:
Text: Madaktari wanapendekeza kufanya mazoezi kwa dakika 30 kila siku kwa afya ya moyo.
Category: health

"""
        if "entertainment" in labels:
            examples += """Mfano 4:
Text: Filamu mpya imevunja rekodi ya mauzo ya tiketi katika wiki ya kwanza ya kuonyeshwa.
Category: entertainment

"""
        if "business" in labels:
            examples += """Mfano 5:
Text: Kampuni imetangaza ripoti ya mapato ya robo mwaka, ikionyesha ongezeko la asilimia 20.
Category: business

"""
    
    elif lang_code == 'eng':
        print(f"WARN: LRL instructions requested for English text. Using standard English prompt.")
        return generate_lrl_classification_prompt(text, lang_code, labels)
    else:
        print(f"WARN: LRL instructions not defined for {lang_code}. Falling back to English prompt.")
        return generate_lrl_classification_prompt(text, lang_code, labels)
        
    prompt = f"""{instructions}

{examples}Text ({lang_code}): {text}

Category:"""
    return prompt

def process_classification_output(
    response: str,
    possible_labels: List[str] # These are the English labels
) -> str:
    """Process the model's output to extract an ENGLISH classification label."""
    predicted_label_raw = response.strip().lower()
    predicted_label_lines = predicted_label_raw.split('\n')
    
    # Get first non-empty line
    predicted_label = ""
    for line in predicted_label_lines:
        if line.strip():
            predicted_label = line.strip()
            break
    
    if not predicted_label:
        return "[Unknown]"
        
    # Remove typical prefixes that models might include
    predicted_label = predicted_label.replace("category:", "").replace("class:", "").strip()
    
    # Create normalized versions of possible labels
    normalized_labels = {label.lower(): label for label in possible_labels}
    
    # Check for exact match with case normalization
    if predicted_label.lower() in normalized_labels:
        return normalized_labels[predicted_label.lower()]
            
    # Check if prediction contains any of the labels
    for raw_label, original_label in normalized_labels.items():
        if raw_label in predicted_label.lower():
            return original_label
    
    # Check for partial matches (e.g. "polit" for "politics")
    for raw_label, original_label in normalized_labels.items():
        # Get first 5 chars or the whole label if shorter
        prefix_length = min(5, len(raw_label))
        if prefix_length > 0 and raw_label[:prefix_length] in predicted_label.lower():
            return original_label
    
    # Final check for common misspellings or partial matches
    common_variants = {
        "sport": "sports",
        "polit": "politics",
        "tech": "technology",
        "entertainment": "entertainment",
        "entertain": "entertainment",
        "health": "health",
        "relig": "religion",
        "busin": "business"
    }
    
    for variant, label in common_variants.items():
        if variant in predicted_label.lower() and label in possible_labels:
            return label
            
    return "[Unknown]"

def evaluate_classification_baseline(
    model_name: str,
    samples_df: pd.DataFrame,
    lang_code: str,
    use_lrl_ground_truth: bool, # Keep for consistency, though GT is EN here
    prompt_in_lrl: bool = False # Add flag
) -> pd.DataFrame:
    """Evaluates baseline classification, selecting prompt based on flag."""
    try:
        model, tokenizer = initialize_model(model_name)
    except Exception as e:
        print(f"ERROR: Failed to initialize model {model_name}: {e}")
        return pd.DataFrame()

    results = []
    
    # Determine possible labels from the dataset itself
    possible_labels = sorted(list(samples_df['label'].unique()))
    print(f"Using labels found in dataset for prompts: {possible_labels}")
    
    model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
    safe_max_input_length = min(model_max_len, 1024) # Adjust as needed
    max_new_tokens = 25 # Increased slightly to allow more output
    print(f"Using max_input_length: {safe_max_input_length}, max_new_tokens: {max_new_tokens}")

    # Clear description for logging which prompt template is used
    prompt_desc = "LRLInstruct" if prompt_in_lrl else "EnInstruct" 
    print(f"Using {prompt_desc} prompts for {lang_code}")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} baseline ({prompt_desc}) classification ({model_name})"):
        try:
            original_text = row['text']
            ground_truth_label = row['label'] # Assumed to be English

            # Generate prompt based on flag
            if prompt_in_lrl and lang_code in ['swa', 'hau']:
                prompt = generate_lrl_instruct_classification_prompt(original_text, lang_code, possible_labels)
                prompt_language = "LRL"
            else:
                prompt = generate_lrl_classification_prompt(original_text, lang_code, possible_labels)
                prompt_language = "EN"
            
            # Debug: print prompt occasionally to verify correct template is used
            if idx % 100 == 0:
                print(f"\nSample prompt for {lang_code} using {prompt_language}:")
                print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
                print("\n")
            
            # Tokenize and predict
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=safe_max_input_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True, # Changed to True to allow some variability
                    temperature=0.3, # Low temperature for more focused responses
                    top_p=0.9, # Add top_p sampling 
                    repetition_penalty=1.2, # Add repetition penalty
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            
            # Process the output to extract the ENGLISH label
            predicted_label = process_classification_output(response, possible_labels)

            # Store results
            result = {
                'id': row.get('id', idx),
                'original_text': original_text,
                'ground_truth_label': ground_truth_label,
                'predicted_label': predicted_label, 
                'language': lang_code,
                'lrl_evaluation': use_lrl_ground_truth, # Store flag for consistency
                'prompt_language': prompt_language, 
                'raw_response': response[:100] # Store first 100 chars of raw response for debugging
            }
            results.append(result)

        except Exception as e:
            print(f"ERROR processing sample {row.get('id', idx)}: {e}")
            continue

    if not results:
        print(f"WARNING: No results successfully processed for {lang_code} with {model_name} using Baseline.")
        return pd.DataFrame()

    # Print debugging statistics
    results_df = pd.DataFrame(results)
    unknown_count = sum(results_df['predicted_label'] == "[Unknown]")
    print(f"Unknown predictions: {unknown_count}/{len(results_df)} ({unknown_count/len(results_df)*100:.1f}%)")
    print("Label distribution:", results_df['predicted_label'].value_counts().to_dict())

    return results_df
