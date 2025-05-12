import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import json
from typing import Dict, List, Any, Tuple

def initialize_model(model_name: str) -> Tuple[Any, Any]:
    """
    Initialize a model and tokenizer.
    
    Args:
        model_name: Name of the model to initialize
    
    Returns:
        tokenizer, model
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
        cache_dir=cache_path
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

def generate_ner_prompt(text: str, lang_code: str = "en", model_name: str = "", use_few_shot: bool = True) -> str:
    """
    Generate a prompt for NER with language-specific examples and improved instructions.
    
    Args:
        text: The text to analyze for named entities
        lang_code: Language code for language-specific examples
        model_name: Model name for potential model-specific adjustments
        use_few_shot: Whether to include few-shot examples in the prompt
        
    Returns:
        Formatted prompt
    """
    # Base instructions - English
    if lang_code == "en":
        base_instruction = """Identify all named entities in the text below. 
Tag each entity with one of these categories: PER (person), ORG (organization), LOC (location), or DATE (date).
For each entity you find, use EXACTLY this format: [TYPE: entity text]

Examples of correct format:
[PER: John Smith]
[ORG: Microsoft]
[LOC: New York]
[DATE: Friday]

Categories explanation:
- PER: Real people, fictional characters, named individuals
- ORG: Companies, institutions, governments, political parties, teams
- LOC: Countries, cities, regions, geographic features, buildings 
- DATE: Specific dates, days of the week, months, years, time periods"""
        
        # Example text and entities - English
        example_text = """John Smith from Microsoft visited New York last Friday to meet with Sarah Johnson and representatives from Google."""
        example_entities = """[PER: John Smith]
[ORG: Microsoft]
[LOC: New York]
[DATE: Friday]
[PER: Sarah Johnson]
[ORG: Google]"""
        
    # Swahili instructions and examples
    elif lang_code == "sw":
        base_instruction = """Tambua vitu vyote vilivyotajwa kwenye maandishi yafuatayo.
Weka kila kitu katika mojawapo ya makundi haya: PER (mtu), ORG (shirika), LOC (mahali), au DATE (tarehe).
Kwa kila kitu unachopata, panga jibu lako kama [AINA: maandishi ya kitu].
JIBU KWA ORODHA TU ya vitu vilivyotambuliwa katika muundo wa [AINA: maandishi ya kitu], kila kimoja kwenye mstari mpya. Usijumuishe maandishi mengine ya mazungumzo, maelezo, au utangulizi.

Maelezo ya makundi:
- PER: Watu halisi, wahusika wa hadithi, watu wanaotambulika kwa majina
- ORG: Kampuni, taasisi, serikali, vyama vya siasa, timu
- LOC: Nchi, miji, mikoa, sehemu za kijiografia, majengo
- DATE: Tarehe mahususi, siku za wiki, miezi, miaka, vipindi vya muda"""

        # Example text and entities - Swahili (expanded with more examples)
        example_text_sw1 = "James Mwangi kutoka Equity Bank alisafiri kwenda Nairobi Jumapili iliyopita kuhudhuria mkutano na Rais William Ruto na maafisa wa Safaricom."
        example_entities_sw1 = """[PER: James Mwangi]
[ORG: Equity Bank]
[LOC: Nairobi]
[DATE: Jumapili iliyopita]
[PER: William Ruto]
[PER: Rais William Ruto]
[ORG: Safaricom]"""
        example_text_sw2 = "Shirika la Afya Duniani (WHO) liliripoti mnamo Juni 5, 2023 kwamba kesi za malaria zimeongezeka nchini Kenya."
        example_entities_sw2 = """[ORG: Shirika la Afya Duniani]
[ORG: WHO]
[DATE: Juni 5, 2023]
[LOC: Kenya]"""
        
        # Combined examples for few-shot prompt
        example_text = f"Example 1:\\nText: {example_text_sw1}\\nEntities:\\n{example_entities_sw1}\\n\\nExample 2:\\nText: {example_text_sw2}\\nEntities:\\n{example_entities_sw2}"
        # The `example_entities` variable is not directly used if `example_text` contains the full structure.
        # However, to maintain compatibility with how the prompt is constructed later, we can set example_entities to an empty string
        # as the full examples are now in `example_text`. Or adjust prompt construction.
        # For now, let's assume the prompt construction uses `example_text` as the complete few-shot block.
        example_entities = "" # Signifies entities are within example_text

    # Hausa instructions and examples
    elif lang_code == "ha":
        base_instruction = """Gano duk wata sunan da aka ambata a cikin rubutun da ke kasa.
Ka sanya kowane suna a cikin daya daga cikin wadannan nau'ikan: PER (mutum), ORG (kungiya), LOC (wuri), ko DATE (kwanan wata).
Domin kowanne suna da ka gano, tsara amsarka kamar [NAU\'I: rubutun sunan].
AMSA KAWAI da jerin sunayen da aka gano cikin tsarin `[NAU\\'I: rubutun suna]`, kowanne a sabon layi. Kada ka hada da wani rubutu na zance, bayani, ko gabatarwa.

Bayani game da nau'ikan:
- PER: Mutanen gaskiya, mutanen labari, mutane masu suna
- ORG: Kamfanoni, cibiyoyi, gwamnatoci, jam\'iyyun siyasa, kungiyoyi
- LOC: Kasashe, birane, yankunan, wurare na musamman, gine-gine
- DATE: Kwanan wata, ranakun mako, watanni, shekaru, lokutan musamman"""

        # Example text and entities - Hausa (expanded with more examples)
        example_text_ha1 = """Aliyu Muhammadu daga Dangote Group ya ziyarci Kano ranar Litinin don ganawa da Gwamna Abba Yusuf da wakilin MTN Nigeria."""
        example_entities_ha1 = """[PER: Aliyu Muhammadu]
[ORG: Dangote Group]
[LOC: Kano]
[DATE: ranar Litinin]
[PER: Abba Yusuf]
[PER: Gwamna Abba Yusuf]
[ORG: MTN Nigeria]"""
        example_text_ha2 = """Bankin Duniya ya amince da sabon aiki a Najeriya a ranar Alhamis da ta wuce."""
        example_entities_ha2 = """[ORG: Bankin Duniya]
[LOC: Najeriya]
[DATE: ranar Alhamis da ta wuce]"""

        # Combined examples for few-shot prompt
        example_text = f"Example 1:\\nText: {example_text_ha1}\\nEntities:\\n{example_entities_ha1}\\n\\nExample 2:\\nText: {example_text_ha2}\\nEntities:\\n{example_entities_ha2}"
        example_entities = "" # Signifies entities are within example_text
    
    # Default to English for unknown languages
    else:
        print(f"Warning: Unknown language code '{lang_code}'. Defaulting to English prompt.")
        return generate_ner_prompt(text, "en", model_name, use_few_shot)
    
    # Construct final prompt based on whether few-shot or zero-shot
    if use_few_shot:
        prompt = f"{base_instruction}\\n\\n{example_text}\\n\\nNow identify all entities in this text:\\n{text}\\n\\nEntities:"
    else:
        # Zero-shot prompt with more detailed instructions
        if lang_code == "en":
            prompt = f"""You are an expert in Named Entity Recognition.
{base_instruction}

IMPORTANT: Follow these guidelines:
1. Read the entire text carefully to understand the context
2. Look for proper nouns (capitalized words) and specific dates
3. Identify people's names (PER), organization names (ORG), place names (LOC), and dates (DATE)
4. Use the exact [TYPE: entity text] format for each entity you find
5. Include complete entity names (e.g., "John Smith" not just "John")
6. If an entity appears multiple times, tag each occurrence
7. Be thorough and don't miss any entities
8. Make sure to identify all dates, even if they're not obvious
9. For organizations, include the full name (e.g., "United Nations" not just "UN")
10. For people with titles, include the title (e.g., "President Joe Biden")

Text to analyze: {text}

Entities:"""
        elif lang_code == "sw":
            prompt = f"""Wewe ni mtaalam wa Utambuzi wa Majina.
{base_instruction}

MUHIMU: Fuata miongozo hii:
1. Soma maandishi yote kwa makini kuelewa muktadha
2. Angalia majina mahususi (maneno yenye herufi kubwa) na tarehe mahususi
3. Tambua majina ya watu (PER), majina ya mashirika (ORG), majina ya mahali (LOC), na tarehe (DATE)
4. Tumia muundo sahihi wa [AINA: maandishi ya kitu] kwa kila kitu unachokipata
5. Jumuisha majina kamili ya vitu (mfano, "John Smith" sio "John" tu)
6. Ikiwa kitu kinatokea mara kadhaa, weka lebo kwa kila tokeo
7. Kuwa makini na usikose kitu chochote
8. Hakikisha unatambua tarehe zote, hata kama sio wazi
9. Kwa mashirika, jumuisha jina kamili (mfano, "Umoja wa Mataifa" sio "UN" tu)
10. Kwa watu wenye vyeo, jumuisha cheo (mfano, "Rais William Ruto")

Maandishi ya kuchambua: {text}

Vitu vilivyotajwa:"""
        elif lang_code == "ha":
            prompt = f"""Kai masanin gano sunaye cikin rubutu.
{base_instruction}

MUHIMMI: Bi wadannan kagaggun:
1. Karanta duk rubutun da kyau don fahimtar jigo
2. Nemo sunaye musamman (kalmomi masu haruffa manyan farko) da kwanan wata musamman
3. Gano sunayen mutane (PER), sunayen kungiyoyi (ORG), sunayen wurare (LOC), da kwanan wata (DATE)
4. Yi amfani da tsarin daidai na [NAWA: rubutun sunan] don kowanne suna da ka gano
5. Hada cikakken sunayen (misali, "John Smith" ba "John" kawai ba)
6. Idan wani suna ya bayyana sau da yawa, sanya lamba ga kowanne bayyanawa
7. Ka yi hankali kada ka rasa wani suna
8. Tabbatar da cewa ka gano duk kwanaki, ko da ba a bayyana su a fili ba
9. Don kungiyoyi, hada sunan cikakke (misali, "Majalisar Dinkin Duniya" ba "UN" kawai ba)
10. Don mutane masu matsayi, hada da matsayin (misali, "Gwamna Abba Yusuf")

Rubutun da za a bincika: {text}

Sunaye:"""
    
    return prompt

def generate_lrl_instruct_ner_prompt(text: str, lang_code: str = "sw", model_name: str = "", use_few_shot: bool = True) -> str:
    """
    Generate a prompt for NER with instructions in the target low-resource language.
    This is the same as generate_ner_prompt but ensures we use the correct language version.
    
    Args:
        text: The text to analyze for named entities
        lang_code: Language code for language-specific instructions
        model_name: Model name for potential model-specific adjustments
        use_few_shot: Whether to include few-shot examples in the prompt
        
    Returns:
        Formatted prompt in the target language
    """
    # Just use the same function since we've already implemented language-specific prompts
    return generate_ner_prompt(text, lang_code, model_name, use_few_shot)

def extract_entities(text: str, lang_code: str = "en", verbose: bool = False):
    """
    Extract named entities from model output text with improved robustness.
    
    Args:
        text: Model-generated text with entity mentions
        lang_code: Language code
        verbose: Whether to print debug information
        
    Returns:
        List of entity dictionaries with text and type
    """
    import re
    
    # If text is None or empty, return empty list
    if text is None or text.strip() == "":
        return []
        
    entities = []
    
    # Clean up the text - remove unnecessary parts
    text = text.strip()
    
    # Debug: Print the first part of the text before extraction
    if verbose:
        print(f"--- Extracting Entities (Verbose) ---")
        print(f"Raw Input Text (start): {text[:200]}...")
        print(f"-------------------------------------")

    # Pattern 1: Strict format [TYPE: entity text] (prioritized)
    strict_pattern = r'\[(PER|ORG|LOC|DATE)\s*:\s*([^\\\]]+)\]'
    if verbose: print(f"Trying pattern 1: {strict_pattern}")
    matches = re.findall(strict_pattern, text, re.IGNORECASE)
    if matches:
        if verbose: print(f"  Found {len(matches)} matches with pattern 1.")
        for entity_type, entity_text in matches:
            entity_type_norm = normalize_entity_type(entity_type.strip().upper())
            entity_text_clean = entity_text.strip().strip('"\'')
            if len(entity_text_clean) > 1 and entity_type_norm in ["PER", "ORG", "LOC", "DATE"]:
                entities.append({"text": entity_text_clean, "type": entity_type_norm})
                if verbose: print(f"    Added: {entity_text_clean} [{entity_type_norm}]")
            elif verbose:
                 print(f"    Skipped (pattern 1): Type='{entity_type_norm}', Text='{entity_text_clean}'")

    # Pattern 2: Allow variations like TYPE: entity text (outside brackets)
    colon_pattern = r'\b(PER|ORG|LOC|DATE)\b\s*:\s*([^\n,\[\(\]\)]+?)(?:\n|,|\(|\[|\)|]|$)'
    if verbose: print(f"Trying pattern 2: {colon_pattern}")
    matches = re.findall(colon_pattern, text, re.IGNORECASE)
    if matches:
        if verbose: print(f"  Found {len(matches)} matches with pattern 2.")
        for entity_type, entity_text in matches:
            entity_type_norm = normalize_entity_type(entity_type.strip().upper())
            entity_text_clean = entity_text.strip().strip('"\'')
            if len(entity_text_clean) > 1 and entity_type_norm in ["PER", "ORG", "LOC", "DATE"]:
                entities.append({"text": entity_text_clean, "type": entity_type_norm})
                if verbose: print(f"    Added: {entity_text_clean} [{entity_type_norm}]")
            elif verbose:
                 print(f"    Skipped (pattern 2): Type='{entity_type_norm}', Text='{entity_text_clean}'")

    # Pattern 3: TYPE: Entity text
    type_colon_pattern = r'(PER|ORG|LOC|DATE):\\s*([^\\n,[\]\\(\\)]{2,})'
    # Pattern 4: Entity text (TYPE)
    paren_after_pattern = r'([^\n,\[\]\(\)\\]+?)\s+\((PER|ORG|LOC|DATE)\)'
    if verbose: print(f"Trying pattern 4: {paren_after_pattern}")
    matches = re.findall(paren_after_pattern, text, re.IGNORECASE)
    if matches:
        if verbose: print(f"  Found {len(matches)} matches with pattern 4.")
        for entity_text, entity_type in matches:
            entity_type_norm = normalize_entity_type(entity_type.strip().upper())
            entity_text_clean = entity_text.strip().strip('\"\'')
            if len(entity_text_clean) > 1 and entity_type_norm in ["PER", "ORG", "LOC", "DATE"]:
                entities.append({"text": entity_text_clean, "type": entity_type_norm})
                if verbose: print(f"    Added: {entity_text_clean} [{entity_type_norm}]")
            elif verbose:
                 print(f"    Skipped (pattern 4): Type='{entity_type_norm}', Text='{entity_text_clean}'")

    # Fallback: Section-based extraction if few/no structured entities found
    if len(entities) < 2: # Only try this if structured patterns yielded few results
        if verbose: print(f"Trying fallback: Section-based extraction (found {len(entities)} so far)")
        section_patterns = [
            (r'(?:PERSON|PER|PEOPLE)s?:\s*([^\n]+(?:\n(?!\s*(?:ORGANIZATION|ORG|LOCATION|LOC|DATE))[^{}:]+)*)', "PER"),
            (r'(?:ORGANIZATION|ORG)s?:\s*([^\n]+(?:\n(?!\s*(?:PERSON|PER|LOCATION|LOC|DATE))[^{}:]+)*)', "ORG"),
            (r'(?:LOCATION|LOC|PLACE)s?:\s*([^\n]+(?:\n(?!\s*(?:PERSON|PER|ORGANIZATION|ORG|DATE))[^{}:]+)*)', "LOC"),
            (r'(?:DATE|TIME|DATETIME)s?:\s*([^\n]+(?:\n(?!\s*(?:PERSON|PER|ORGANIZATION|ORG|LOCATION|LOC))[^{}:]+)*)', "DATE")
        ]
        
        for pattern, entity_type in section_patterns:
            if verbose: print(f"  Trying section pattern for {entity_type}")
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                 if verbose: print(f"    Found {len(matches)} potential section(s) for {entity_type}")
            for match_group in matches:
                # Match group might contain multiple lines/items
                items_text = match_group.strip()
                # Split by likely delimiters: newline, comma, semicolon, bullet points
                potential_entities = re.split(r'\n|\s*,\s*|\s*;\s*|\s*•\s*', items_text)
                if verbose: print(f"      Potential entities in section: {potential_entities}")
                for entity_text in potential_entities:
                    entity_text_clean = entity_text.strip().strip('"\'.-')
                    # Basic validation
                    if entity_text_clean and len(entity_text_clean) > 1 and not entity_text_clean.lower().startswith('example') and not entity_text_clean.lower().startswith('text:'):
                        entities.append({"text": entity_text_clean, "type": entity_type})
                        if verbose: print(f"        Added from section: {entity_text_clean} [{entity_type}]")
                    elif verbose:
                        print(f"        Skipped section item: '{entity_text_clean}'")
    
    # Remove duplicates while preserving order (case-insensitive text match for same type)
    unique_entities = []
    seen = set()
    for entity in entities:
        # Create a unique key for each entity (text + type)
        key = (entity["text"].lower(), entity["type"])
        if key not in seen:
            # Additional check: Ensure type is standard
            if entity["type"] in ["PER", "ORG", "LOC", "DATE"]:
                seen.add(key)
                unique_entities.append(entity)
            elif verbose:
                 print(f"  Filtering out entity with non-standard type '{entity['type']}' during deduplication: {entity['text']}")
        elif verbose:
            print(f"  Removing duplicate: {entity['text']} [{entity['type']}]")

    if verbose: print(f"  Found {len(unique_entities)} unique entities after deduplication.")
    return unique_entities

def normalize_entity_type(entity_type):
    """
    Normalize entity type to standard NER categories.
    
    Args:
        entity_type: Raw entity type from model output
        
    Returns:
        Normalized entity type
    """
    entity_type = entity_type.upper().strip()
    
    # Map to standard types with more comprehensive matching
    if any(t in entity_type for t in ["PERSON", "PER", "INDIVIDUAL", "HUMAN", "NAME"]):
        return "PER"
    elif any(t in entity_type for t in ["ORG", "ORGANIZATION", "COMPANY", "INSTITUTION", "AGENCY", "GROUP"]):
        return "ORG"
    elif any(t in entity_type for t in ["LOC", "LOCATION", "PLACE", "AREA", "REGION", "COUNTRY", "CITY"]):
        return "LOC"
    elif any(t in entity_type for t in ["DATE", "TIME", "DATETIME", "PERIOD", "DAY", "MONTH", "YEAR"]):
        return "DATE"
    else:
        return entity_type  # Return as is if no match

def process_ner_baseline(
    tokenizer: Any, 
    model: Any, 
    text: str, 
    max_input_length: int = 4096,
    max_new_tokens: int = 200,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    lang_code: str = "en",
    model_name: str = "",
    prompt_in_lrl: bool = False,
    use_few_shot: bool = True,
    verbose: bool = False
) -> List[Dict[str, str]]:
    """
    Process a text for NER with the given model directly (baseline).
    Now accepts and uses top_k, repetition_penalty, do_sample.
    """
    # --- Add Type Check ---
    print(f"DEBUG process_ner_baseline: Type of tokenizer arg: {type(tokenizer)}")
    print(f"DEBUG process_ner_baseline: Type of model arg: {type(model)}")
    # --- End Type Check ---
    try:
        # Generate prompt based on language and instruction preference
        if prompt_in_lrl:
            prompt = generate_lrl_instruct_ner_prompt(text, lang_code, model_name, use_few_shot)
        else:
            prompt = generate_ner_prompt(text, "en", model_name, use_few_shot)  # English instructions
        
        # Tokenize with truncation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        
        # Move to GPU
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # --- Debug Print ---
        print(f"DEBUG: Inputs passed to model.generate: {inputs}")
        if 'input_ids' in inputs:
            print(f"DEBUG: Type of input_ids: {type(inputs['input_ids'])}")
        # --- End Debug Print ---
            
        # Use the generation parameters passed to the function
        gen_temperature = temperature
        gen_top_p = top_p
        gen_top_k = top_k
        gen_repetition_penalty = repetition_penalty
        gen_do_sample = do_sample
        
        # Model-specific adjustments (optional, could be simplified)
        is_aya_model = "aya" in model_name.lower()
        is_qwen_model = "qwen" in model_name.lower()
        
        # Note: Parameter adjustments here might override intended experiment parameters.
        # Consider applying adjustments *before* calling this function if needed.
        # if is_aya_model:
        #     gen_temperature = 0.3 
        #     gen_repetition_penalty = 1.3
        # elif is_qwen_model:
        #     gen_temperature = 0.2 
        #     gen_top_p = 0.8
        
        # Generate with optimized parameters
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=gen_do_sample,
                temperature=gen_temperature,
                top_p=gen_top_p,
                top_k=gen_top_k,
                repetition_penalty=gen_repetition_penalty,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # TEMPORARY DEBUG: Print raw model output for specific cases
        # Adjust the condition to target the failing case, e.g., Swahili zero-shot with Aya
        if lang_code == 'sw' and not use_few_shot and "aya" in model_name.lower():
            print(f"\nDEBUG SWahili ZERO-SHOT (Aya) RAW OUTPUT for text: {text[:100]}...")
            print(f"RAW MODEL OUTPUT: >>>{output_text}<<<")
            
        # DEBUG: Print model output for inspection (only for first few samples)
        static_sample_counter = getattr(process_ner_baseline, "sample_counter", 0)
        if verbose and static_sample_counter < 3:  # Print only first 3 samples to avoid flooding logs
            print(f"\n=== DEBUG: MODEL OUTPUT (Sample {static_sample_counter + 1}) ===")
            print(f"Text: {text[:50]}...")
            print(f"Raw model output: {output_text[:500]}...")
            print("=======================================")
            process_ner_baseline.sample_counter = static_sample_counter + 1
        
        # Extract entities
        entities = extract_entities(output_text, lang_code, verbose=verbose)
        
        # DEBUG: Print extracted entities
        if verbose and static_sample_counter <= 3:
            print(f"Extracted entities: {entities}")
        
        return entities
    except Exception as e:
        import traceback
        print(f"Error in process_ner_baseline: {str(e)}")
        traceback.print_exc()
        return []

# Add a static counter to the function
process_ner_baseline.sample_counter = 0

def evaluate_ner_baseline(
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    model_name: str,
    prompt_in_lrl: bool = False,
    use_few_shot: bool = True,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 40,
    max_new_tokens: int = 200,
    repetition_penalty: float = 1.1,
    do_sample: bool = True
) -> pd.DataFrame:
    """
    Evaluate NER performance using the baseline approach.
    Now accepts tokenizer, model, and generation parameters.
    It no longer initializes the model internally.
    """
    # --- Add Type Check at Entry ---
    print(f"DEBUG evaluate_ner_baseline ENTRY: Type of tokenizer param: {type(tokenizer)}")
    print(f"DEBUG evaluate_ner_baseline ENTRY: Type of model param: {type(model)}")
    # --- End Type Check ---
    results = []
    print(f"DEBUG: Columns in samples_df passed to evaluate_ner_baseline: {samples_df.columns.tolist()}")
    
    # --- Create 'text' column OUTSIDE the if block --- 
    if 'tokens' in samples_df.columns:
        samples_df['text'] = samples_df['tokens'].apply(lambda tokens: ' '.join(tokens))
    else:
        print("ERROR: 'tokens' column missing. Cannot create 'text' column.")
        return pd.DataFrame() # Return empty if tokens are missing
        
    # --- Convert entities OUTSIDE the if block ---
    if 'entities' in samples_df.columns:
        # --- Define convert_entities helper function --- 
        def convert_entities(row):
            formatted_entities = []
            # Ensure entities column contains lists (handle potential loading issues)
            entities_list = row['entities']
            if not isinstance(entities_list, list):
                print(f"Warning: 'entities' data is not a list for a row: {type(entities_list)}. Skipping entity conversion for this row.")
                return [] 
                
            for entity_info in entities_list:
                # Add checks for expected keys in entity_info dict
                if not all(k in entity_info for k in ('start', 'end', 'entity_type')):
                    print(f"Warning: Skipping malformed entity_info: {entity_info}")
                    continue
                try:
                    start_idx = entity_info['start']
                    end_idx = entity_info['end']
                    entity_type = entity_info['entity_type']
                    # Ensure indices are within bounds
                    if start_idx >= 0 and end_idx <= len(row['tokens']):
                        entity_text = ' '.join(row['tokens'][start_idx:end_idx])
                        formatted_entities.append({
                            "entity": entity_text,
                            "type": entity_type
                        })
                    else:
                        print(f"Warning: Entity indices [{start_idx}:{end_idx}] out of bounds for tokens length {len(row['tokens'])}")
                except Exception as e_conv:
                    print(f"Warning: Error converting entity: {entity_info}. Error: {e_conv}")
            return formatted_entities
            
        # Apply the conversion function
        samples_df['entities'] = samples_df.apply(convert_entities, axis=1)
    else:
        print("ERROR: 'entities' column missing. Cannot process entities.")
        return pd.DataFrame() # Return empty if entities are missing
    
    print(f"DEBUG: Columns in samples_df *before returning* from load_masakhaner_samples: {samples_df.columns.tolist()}") # ADD DEBUG PRINT
    print(f"  Loaded and prepared {len(samples_df)} samples for {lang_code}.") # Updated print message
    return samples_df

def calculate_ner_metrics(gold_entities: List[Dict[str, str]], predicted_entities: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for NER predictions.
    
    Args:
        gold_entities: List of ground truth entities
        predicted_entities: List of predicted entities
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    # Convert to sets of (entity, type) tuples for comparison
    gt_set = set()
    for e in gold_entities:
        if "entity" in e:
            entity_text = e["entity"].lower()
        elif "text" in e:
            entity_text = e["text"].lower()
        else:
            continue  # Skip entities with unrecognized format
            
        entity_type = e.get("type", e.get("entity_type", ""))
        if entity_type:  # Only add if entity_type is not empty
            gt_set.add((entity_text, entity_type))
    
    pred_set = set()
    for e in predicted_entities:
        if "entity" in e:
            entity_text = e["entity"].lower()
        elif "text" in e:
            entity_text = e["text"].lower()
        else:
            continue  # Skip entities with unrecognized format
            
        entity_type = e.get("type", e.get("entity_type", ""))
        if entity_type:  # Only add if entity_type is not empty
            pred_set.add((entity_text, entity_type))
        
        # Calculate true positives, false positives, false negatives
        true_positives = len(gt_set.intersection(pred_set))
        false_positives = len(pred_set) - true_positives
        false_negatives = len(gt_set) - true_positives
        
        # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def load_masakhaner_samples(lang_code: str, split: str = "test", num_samples: int = None, seed: int = 42) -> pd.DataFrame:
    """
    Load samples from MasakhaNER dataset.
    
    Args:
        lang_code: Language code (e.g., 'sw', 'ha', 'en')
        split: Dataset split ('train', 'dev', 'test')
        num_samples: Number of samples to load (None for all)
        seed: Random seed for sampling
        
    Returns:
        DataFrame with text and entities
    """
    # Import the HuggingFace loader
    from src.utils.data_loaders.load_masakhaner import load_masakhaner_samples as load_from_hf

    try:
        # Load data from HuggingFace
        samples_df = load_from_hf(lang_code, num_samples=num_samples, split=split, seed=seed)
        
        if samples_df.empty:
            print(f"No samples found for {lang_code} in MasakhaNER split '{split}'. Falling back to dummy data...")
            return create_dummy_ner_data(lang_code, num_samples or 5)
        
        # --- Create 'text' column OUTSIDE the if block --- 
        if 'tokens' in samples_df.columns:
            samples_df['text'] = samples_df['tokens'].apply(lambda tokens: ' '.join(tokens))
        else:
            print("ERROR: 'tokens' column missing. Cannot create 'text' column.")
            return pd.DataFrame() # Return empty if tokens are missing
            
        # --- Convert entities OUTSIDE the if block ---
        if 'entities' in samples_df.columns:
            # --- Define convert_entities helper function --- 
            def convert_entities(row):
                formatted_entities = []
                # Ensure entities column contains lists (handle potential loading issues)
                entities_list = row['entities']
                if not isinstance(entities_list, list):
                    print(f"Warning: 'entities' data is not a list for a row: {type(entities_list)}. Skipping entity conversion for this row.")
                    return [] 
                    
                for entity_info in entities_list:
                    # Add checks for expected keys in entity_info dict
                    if not all(k in entity_info for k in ('start', 'end', 'entity_type')):
                        print(f"Warning: Skipping malformed entity_info: {entity_info}")
                        continue
                    try:
                        start_idx = entity_info['start']
                        end_idx = entity_info['end']
                        entity_type = entity_info['entity_type']
                        # Ensure indices are within bounds
                        if start_idx >= 0 and end_idx <= len(row['tokens']):
                            entity_text = ' '.join(row['tokens'][start_idx:end_idx])
                            formatted_entities.append({
                                "entity": entity_text,
                                "type": entity_type
                            })
                        else:
                            print(f"Warning: Entity indices [{start_idx}:{end_idx}] out of bounds for tokens length {len(row['tokens'])}")
                    except Exception as e_conv:
                        print(f"Warning: Error converting entity: {entity_info}. Error: {e_conv}")
                return formatted_entities
            
            # Apply the conversion function
            samples_df['entities'] = samples_df.apply(convert_entities, axis=1)
        else:
            print("ERROR: 'entities' column missing. Cannot process entities.")
            return pd.DataFrame() # Return empty if entities are missing
        
        print(f"DEBUG: Columns in samples_df *before returning* from load_masakhaner_samples: {samples_df.columns.tolist()}") # ADD DEBUG PRINT
        print(f"  Loaded and prepared {len(samples_df)} samples for {lang_code}.") # Updated print message
        return samples_df
    
    except Exception as e:
        print(f"Error loading MasakhaNER data from HuggingFace: {e}")
        print("Falling back to dummy data...")
        return create_dummy_ner_data(lang_code, num_samples or 5)

def create_dummy_ner_data(lang_code: str, num_samples: int = 5) -> pd.DataFrame:
    """
    Create dummy NER data for testing.
    
    Args:
        lang_code: Language code
        num_samples: Number of samples to create
        
    Returns:
        DataFrame with text and entities
    """
    samples = []
    
    # English dummy data
    if lang_code == "en":
        texts = [
            "The President of the United States, Joe Biden, visited Berlin last Tuesday.",
            "Apple Inc. announced a new partnership with Microsoft Corporation in January 2023.",
            "Mount Kilimanjaro is located in Tanzania and is the highest mountain in Africa.",
            "The United Nations was founded on October 24, 1945, in San Francisco, California.",
            "Elon Musk is the CEO of Tesla and SpaceX, and he recently acquired Twitter."
        ]
        
        entities = [
            [
                {"entity": "Joe Biden", "type": "PER"},
                {"entity": "United States", "type": "LOC"},
                {"entity": "Berlin", "type": "LOC"},
                {"entity": "Tuesday", "type": "DATE"}
            ],
            [
                {"entity": "Apple Inc.", "type": "ORG"},
                {"entity": "Microsoft Corporation", "type": "ORG"},
                {"entity": "January 2023", "type": "DATE"}
            ],
            [
                {"entity": "Mount Kilimanjaro", "type": "LOC"},
                {"entity": "Tanzania", "type": "LOC"},
                {"entity": "Africa", "type": "LOC"}
            ],
            [
                {"entity": "United Nations", "type": "ORG"},
                {"entity": "October 24, 1945", "type": "DATE"},
                {"entity": "San Francisco", "type": "LOC"},
                {"entity": "California", "type": "LOC"}
            ],
            [
                {"entity": "Elon Musk", "type": "PER"},
                {"entity": "Tesla", "type": "ORG"},
                {"entity": "SpaceX", "type": "ORG"},
                {"entity": "Twitter", "type": "ORG"}
            ]
        ]
    
    # Swahili dummy data
    elif lang_code == "sw":
        texts = [
            "Rais wa Marekani, Joe Biden, alitembelea Berlin Jumanne iliyopita.",
            "Kampuni ya Apple Inc. ilitangaza ushirikiano mpya na Kampuni ya Microsoft mwezi Januari 2023.",
            "Mlima Kilimanjaro upo Tanzania na ni mlima mrefu zaidi Afrika.",
            "Umoja wa Mataifa ulianzishwa tarehe 24 Oktoba, 1945, huko San Francisco, California.",
            "Elon Musk ni Mkurugenzi Mtendaji wa Tesla na SpaceX, na hivi karibuni alinunua Twitter."
        ]
        
        entities = [
            [
                {"entity": "Joe Biden", "type": "PER"},
                {"entity": "Marekani", "type": "LOC"},
                {"entity": "Berlin", "type": "LOC"},
                {"entity": "Jumanne", "type": "DATE"}
            ],
            [
                {"entity": "Apple Inc.", "type": "ORG"},
                {"entity": "Kampuni ya Microsoft", "type": "ORG"},
                {"entity": "Januari 2023", "type": "DATE"}
            ],
            [
                {"entity": "Mlima Kilimanjaro", "type": "LOC"},
                {"entity": "Tanzania", "type": "LOC"},
                {"entity": "Afrika", "type": "LOC"}
            ],
            [
                {"entity": "Umoja wa Mataifa", "type": "ORG"},
                {"entity": "24 Oktoba, 1945", "type": "DATE"},
                {"entity": "San Francisco", "type": "LOC"},
                {"entity": "California", "type": "LOC"}
            ],
            [
                {"entity": "Elon Musk", "type": "PER"},
                {"entity": "Tesla", "type": "ORG"},
                {"entity": "SpaceX", "type": "ORG"},
                {"entity": "Twitter", "type": "ORG"}
            ]
        ]
    
    # Hausa dummy data
    elif lang_code == "ha":
        texts = [
            "Shugaban Amurka, Joe Biden, ya ziyarci Berlin ranar Talata da ta gabata.",
            "Kamfanin Apple Inc. ya sanar da sabon hadin gwiwa da Kamfanin Microsoft a watan Janairu 2023.",
            "Dutsen Kilimanjaro yana a Tanzania kuma shine dutsen da ya fi tsawo a Afrika.",
            "Majalisar Dinkin Duniya an kafa ta ne a ranar 24 ga Oktoba, 1945, a San Francisco, California.",
            "Elon Musk shine Babban Daraktan Tesla da SpaceX, kuma kwanannan ya sayo Twitter."
        ]
        
        entities = [
            [
                {"entity": "Joe Biden", "type": "PER"},
                {"entity": "Amurka", "type": "LOC"},
                {"entity": "Berlin", "type": "LOC"},
                {"entity": "Talata", "type": "DATE"}
            ],
            [
                {"entity": "Apple Inc.", "type": "ORG"},
                {"entity": "Kamfanin Microsoft", "type": "ORG"},
                {"entity": "Janairu 2023", "type": "DATE"}
            ],
            [
                {"entity": "Dutsen Kilimanjaro", "type": "LOC"},
                {"entity": "Tanzania", "type": "LOC"},
                {"entity": "Afrika", "type": "LOC"}
            ],
            [
                {"entity": "Majalisar Dinkin Duniya", "type": "ORG"},
                {"entity": "24 ga Oktoba, 1945", "type": "DATE"},
                {"entity": "San Francisco", "type": "LOC"},
                {"entity": "California", "type": "LOC"}
            ],
            [
                {"entity": "Elon Musk", "type": "PER"},
                {"entity": "Tesla", "type": "ORG"},
                {"entity": "SpaceX", "type": "ORG"},
                {"entity": "Twitter", "type": "ORG"}
            ]
        ]
    
    # Default to English if language not supported
    else:
        return create_dummy_ner_data("en", num_samples)
    
    # Create samples
    for i in range(min(num_samples, len(texts))):
        samples.append({
            "text": texts[i],
            "entities": entities[i]
        })
    
    return pd.DataFrame(samples) 