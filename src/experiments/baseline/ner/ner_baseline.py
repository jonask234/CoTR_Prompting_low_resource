import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import json
from typing import Dict, List, Any, Tuple, Optional
import time
import logging
from datasets import load_dataset
import gc
from collections import Counter

ENTITY_TYPES = ["PER", "ORG", "LOC", "DATE"]

logger = logging.getLogger(__name__)

# Definiert globale englische Few-Shot-Beispiele für NER
ENGLISH_FEW_SHOT_EXAMPLES_NER_TYPE_TEXT_FORMAT = [
    {"text": "Angela Merkel visited Paris on September 1st, 2021 with a delegation from the European Union.", 
     "entities_str": "[PER: Angela Merkel] [LOC: Paris] [DATE: September 1st, 2021] [ORG: European Union]"},
    {"text": "The quick brown fox jumps over the lazy dog in New York.", 
     "entities_str": "[LOC: New York]"},
    {"text": "There are no entities here.",
     "entities_str": "[NO_ENTITIES_FOUND]"}
]

# Definiert LRL-spezifische Anweisungen für NER
# LRL_INSTRUCTIONS_NER = {
#     "swa": {
#         "instruction": "Tafadhali toa huluki zilizotajwa (Mtu, Shirika, Mahali, Tarehe) kutoka kwa maandishi yafuatayo. Pato linapaswa kuwa orodha ya kamusi, ambapo kila kamusi inawakilisha huluki na ina funguo za 'entity' na 'type'.",
#         "examples_header": "Hapa kuna mifano (mifano hii ni kwa Kiingereza):",
#         "analyze_header": "Chambua maandishi yafuatayo na utoe huluki:",
#         "text_label": "Maandishi",
#         "entities_label": "Huluki"
#     },
#     "yor": {
#         "instruction": "Jọwọ yọ awọn nkan ti a darukọ (Eniyan, Ajo, Ipo, Ọjọ) jade lati inu ọrọ atẹle. Ijade yẹ ki o jẹ atokọ ti awọn iwe-itumọ, nibiti iwe-itumọ kọọkan duro fun nkan kan ti o ni awọn bọtini 'entity' ati 'type'.",
#         "examples_header": "Eyi ni diẹ ninu awọn apẹẹrẹ (awọn apẹẹrẹ wọnyi wa ni Gẹẹsi):",
#         "analyze_header": "Ṣe itupalẹ ọrọ atẹle ki o yọ awọn nkan jade:",
#         "text_label": "Ọrọ",
#         "entities_label": "Awọn nkan"
#     }
# }

# Alle Few-Shot-Beispiele sind jetzt über alle Konfigurationen hinweg konsistent auf Englisch

# LRL-Anweisungen (global definiert)
LRL_INSTRUCTIONS = {
    "sw": {
        "task": "Jukumu lako ni kutambua huluki zilizo na majina katika maandishi yafuatayo ya Kiswahili. Huluki zinazojulikana ni: PER (Person), ORG (Organization), LOC (Location), DATE (Date).",
        "format": "Wasilisha matokeo yako YOTE kama orodha ya huluki, kila moja katika muundo wa [AINA: Maandishi ya Huluki], zikitenganishwa na nafasi au mistari mipya.",
        "example_format": "Mfano wa muundo: [PER: Juma Kaseja] [LOC: Dar es Salaam] [DATE: Julai 2024].",
        "no_entities": "Kama hakuna huluki zilizopatikana, andika maneno haya hasa: [NO_ENTITIES_FOUND].",
        "examples_header": "\\n\\nMifano (hakikisha matokeo yako yanafuata muundo huu kikamilifu):\\n",
        "analyze_header": "\\n\\nSasa, changanua maandishi yafuatayo ya Kiswahili na utoe huluki katika muundo uliobainishwa ([AINA: Maandishi ya Huluki] au [NO_ENTITIES_FOUND]):",
        "text_label": "Maandishi (Kiswahili)",
        "entities_label": "Huluki"
    },
    "swa": {
        "task": "Jukumu lako ni kutambua huluki zilizo na majina katika maandishi yafuatayo ya Kiswahili. Huluki zinazojulikana ni: PER (Person), ORG (Organization), LOC (Location), DATE (Date).",
        "format": "Wasilisha matokeo yako YOTE kama orodha ya huluki, kila moja katika muundo wa [AINA: Maandishi ya Huluki], zikitenganishwa na nafasi au mistari mipya.",
        "example_format": "Mfano wa muundo: [PER: Juma Kaseja] [LOC: Dar es Salaam] [DATE: Julai 2024].",
        "no_entities": "Kama hakuna huluki zilizopatikana, andika maneno haya hasa: [NO_ENTITIES_FOUND].",
        "examples_header": "\\n\\nMifano (hakikisha matokeo yako yanafuata muundo huu kikamilifu):\\n",
        "analyze_header": "\\n\\nSasa, changanua maandishi yafuatayo ya Kiswahili na utoe huluki katika muundo uliobainishwa ([AINA: Maandishi ya Huluki] au [NO_ENTITIES_FOUND]):",
        "text_label": "Maandishi (Kiswahili)",
        "entities_label": "Huluki"
    },
    "hau": {
        "task": "Aikin ku shine gano sunayen da aka ambata a cikin rubutun Hausa da ke ƙasa. Ire-iren sunayen da aka sani sune: PER (Person), ORG (Organization), LOC (Location), DATE (Date).",
        "format": "Gabatar da DUKKAN sakamakon ku a matsayin jerin sunaye, kowanne a cikin tsarin [NAU'I: Rubutun Suna], waɗanda aka raba da sarari ko sabbin layuka.",
        "example_format": "Misalin tsari: [PER: Musa Aliyu] [LOC: Kano] [DATE: Yuli 2024].",
        "no_entities": "Idan ba a sami wasu sunaye ba, rubuta wannan jimlar daidai: [NO_ENTITIES_FOUND].",
        "examples_header": "\\n\\nMisalai (tabbatar cewa sakamakon ku ya bi wannan tsarin sosai):\\n",
        "analyze_header": "\\n\\nYanzu, yi nazarin rubutun Hausa da ke tafe kuma samar da sunayen a cikin tsarin da aka ƙayyade ([NAU'I: Rubutun Suna] ko [NO_ENTITIES_FOUND]):",
        "text_label": "Rubutu (Hausa)",
        "entities_label": "Sunaye"
    },
    "ha": {
        "task": "Aikin ku shine gano sunayen da aka ambata a cikin rubutun Hausa da ke ƙasa. Ire-iren sunayen da aka sani sune: PER (Person), ORG (Organization), LOC (Location), DATE (Date).",
        "format": "Gabatar da DUKKAN sakamakon ku a matsayin jerin sunaye, kowanne a cikin tsarin [NAU'I: Rubutun Suna], waɗanda aka raba da sarari ko sabbin layuka.",
        "example_format": "Misalin tsari: [PER: Musa Aliyu] [LOC: Kano] [DATE: Yuli 2024].",
        "no_entities": "Idan ba a sami wasu sunaye ba, rubuta wannan jimlar daidai: [NO_ENTITIES_FOUND].",
        "examples_header": "\\n\\nMisalai (tabbatar cewa sakamakon ku ya bi wannan tsarin sosai):\\n",
        "analyze_header": "\\n\\nYanzu, yi nazarin rubutun Hausa da ke tafe kuma samar da sunayen a cikin tsarin da aka ƙayyade ([NAU'I: Rubutun Suna] ko [NO_ENTITIES_FOUND]):",
        "text_label": "Rubutu (Hausa)",
        "entities_label": "Sunaye"
    }
}

def initialize_model(model_name: str) -> Tuple[Any, Any]:
    """
    Initialisiert ein Modell und einen Tokenizer.
    
    Args:
        model_name: Name des zu initialisierenden Modells
    
    Returns:
        tokenizer, model
    """
    logger.info(f"Initializing {model_name}...")
    cache_path = "/work/bbd6522/cache_dir"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path
    )
    
    model = None
    try:
        logger.info(f"Attempting to load {model_name} with device_map='auto', torch_dtype=torch.float16, and low_cpu_mem_usage=True.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_path,
            low_cpu_mem_usage=True
        )
        logger.info(f"Successfully loaded {model_name} using device_map='auto' and torch_dtype=torch.float16.")

    except Exception as e_load:
        logger.error(f"CRITICAL: Failed to load model {model_name} with device_map='auto' and float16. Error: {e_load}", exc_info=True)
        if model is not None:
            del model
            model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache after failed model load attempt.")
        raise RuntimeError(f"Could not load model {model_name} due to: {e_load}") from e_load

    # Behandlung des Pad-Tokens (nur wenn das Modell erfolgreich geladen wurde)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer.pad_token to eos_token ('{tokenizer.eos_token}') for {model_name}")
        else:
            new_pad_token = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': new_pad_token})
            # Ändert die Größe der Embeddings nur, wenn das Modell nicht None ist
            if model:
                try:
                    model.resize_token_embeddings(len(tokenizer))
                    logger.info(f"Added new pad_token: '{new_pad_token}' and resized embeddings for {model_name}.")
                except Exception as e_resize:
                    logger.error(f"Failed to resize token embeddings for {model_name} after adding new pad_token: {e_resize}")
            else: # Sollte nicht passieren, wenn das Laden fehlgeschlagen ist und einen Fehler ausgelöst hat
                 logger.warning(f"Model for {model_name} is None after loading attempt, cannot resize embeddings for new pad_token.")


    if model.config.pad_token_id is None:
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info(f"Set model.config.pad_token_id to tokenizer.pad_token_id ({tokenizer.pad_token_id}) for {model_name}")
        elif tokenizer.eos_token_id is not None:
            model.config.pad_token_id = tokenizer.eos_token_id
            logger.info(f"Set model.config.pad_token_id to tokenizer.eos_token_id ({tokenizer.eos_token_id}) for {model_name}")
        else:
            # Dieser Fall sollte bei gut konfigurierten Modellen/Tokenizern selten sein
            logger.warning(f"Could not set model.config.pad_token_id for {model_name} as tokenizer has no pad_token_id or eos_token_id.")
            # Potenziell einen Standardwert wie 0 setzen, was aber riskant sein kann.
            # model.config.pad_token_id = 0 
            # logger.warning(f"Defaulted model.config.pad_token_id to 0 for {model_name}. VERIFY THIS IS CORRECT.")


    # Endgültige Überprüfung des Typs von pad_token_id, entscheidend für `generate`
    if model.config.pad_token_id is not None and not isinstance(model.config.pad_token_id, int):
        logger.error(f"CRITICAL: model.config.pad_token_id for {model_name} is {model.config.pad_token_id} (type: {type(model.config.pad_token_id)}), which is not an int. This will likely cause errors in model.generate().")
        # Versucht eine Korrektur, wenn möglich, z.B. wenn eos_token_id ein int ist
        if tokenizer.eos_token_id is not None and isinstance(tokenizer.eos_token_id, int):
            model.config.pad_token_id = tokenizer.eos_token_id
            logger.warning(f"Attempted fix: Set model.config.pad_token_id to tokenizer.eos_token_id ({tokenizer.eos_token_id}) for {model_name}.")
        else:
            logger.error(f"Further CRITICAL: Cannot automatically fix model.config.pad_token_id for {model_name} to an integer value.")


    logger.info(f"Model {model_name} initialization finished. Tokenizer pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id}). Model config pad_token_id: {model.config.pad_token_id}")
    return tokenizer, model

def generate_ner_prompt(text: str, lang_code: str = "en", model_name: str = "", use_few_shot: bool = True) -> str:
    """
    Generiert einen NER-Prompt mit englischen Anweisungen.
    Konzentriert sich auf das Format [TYP: Entitätstext] für die Ausgabe.
    Dies wird verwendet, wenn prompt_in_lrl False ist.
    """
    processed_text = text.strip()
    entity_types_desc = "PER (Person), ORG (Organization), LOC (Location), DATE (Date)"
    no_entities_marker = "[NO_ENTITIES_FOUND]"

    instruction = (
        f"Your task is to identify named entities in the text below. The text language is '{lang_code}'. " # Klärt die Textsprache
        f"Extract all entities corresponding to {entity_types_desc}. "
        f"Present your ENTIRE output as a list of entities, each in the format [TYPE: Entity Text], separated by spaces or newlines. "
        f"Example of format: [PER: John Doe] [LOC: Paris] [DATE: July 2024]. "
        f"If no entities are found, output the exact phrase: {no_entities_marker}."
    )

    prompt = instruction
    
    if use_few_shot:
        prompt += "\\n\\nExamples (ensure your output strictly follows this format for the actual task):\\n"
        # Englische Beispiele im Format [TYP: Entitätstext]
        for ex in ENGLISH_FEW_SHOT_EXAMPLES_NER_TYPE_TEXT_FORMAT: # Aktualisiert, um den neuen Namen der Beispiel-Liste zu verwenden
            prompt += f"\\nText (English): '{ex['text']}'\\nEntities: {ex['entities_str']}" # lang_code ist 'en' für diese Beispiele
    else: # Zero-shot
        prompt += f"\\n\\nNow, analyze the following text and provide the entities in the specified format ([TYPE: Entity Text] or {no_entities_marker}):"

    prompt += f"\\n\\nText ({lang_code}): '{processed_text}'\\n\\nEntities:" # Stellt sicher, dass 'Entities:' der letzte Hinweis ist
    return prompt

# Diese Funktion muss definiert oder importiert werden, wenn sie von generate_ner_prompt verwendet wird
# Vorerst wird angenommen, dass sie nicht kritisch ist oder an anderer Stelle behandelt wird.
# def preprocess_text(text: str, lang_code: str) -> str:
#     return text.strip()

def generate_lrl_instruct_ner_prompt(text: str, lang_code: str = "sw", model_name: str = "", use_few_shot: bool = True) -> str:
    """
    Generiert einen NER-Prompt mit Anweisungen in der angegebenen Low-Resource Language (LRL),
    falls verfügbar, aber mit englischen Few-Shot-Beispielen.
    Gibt Entitäten im Format [TYP: Entitätstext] aus.
    Dies wird verwendet, wenn prompt_in_lrl True ist.
    """
    processed_text = text.strip()

    # Verwendet das global definierte LRL_INSTRUCTIONS-Wörterbuch
    current_instructions = LRL_INSTRUCTIONS.get(lang_code)
    if not current_instructions:
        # Fällt auf englische Anweisungen zurück, wenn keine LRL-Anweisungen für den lang_code definiert sind
        logger.warning(f"LRL instructions not defined for lang_code '{lang_code}'. Falling back to English-instructed prompt structure via generate_ner_prompt but with LRL text still indicated.")
        # Ruft den englischen Prompt-Generator auf, der aber immer noch LRL-Text verarbeitet.
        # Er verwendet standardmäßig englische Few-Shot-Beispiele.
        return generate_ner_prompt(text, lang_code, model_name, use_few_shot)

    instruction = (
        f"{current_instructions['task']} "
        f"{current_instructions['format']} "
        f"{current_instructions['example_format']} "
        f"{current_instructions['no_entities']}"
    )
    
    prompt = instruction

    if use_few_shot:
        prompt += current_instructions['examples_header']
        # Gemäß Anforderung des Benutzers werden IMMER englische Few-Shot-Beispiele für die Baseline verwendet,
        # um eine sprachübergreifende Baseline zu erstellen, auch wenn die Anweisungen in LRL sind.
        for ex in ENGLISH_FEW_SHOT_EXAMPLES_NER_TYPE_TEXT_FORMAT:
            # Das Prompt-Text-Label sollte weiterhin darauf hinweisen, dass das Beispiel auf Englisch ist
            prompt += f"\\nText (English): '{ex['text']}'\\n{current_instructions['entities_label']}: {ex['entities_str']}"

    else: # Zero-shot
        prompt += current_instructions['analyze_header']

    prompt += f"\\n\\n{current_instructions['text_label']}: '{processed_text}'\\n\\n{current_instructions['entities_label']}:"
    return prompt

def extract_entities(text: str, lang_code: str = "en", verbose: bool = False):
    """
    Extrahiert benannte Entitäten aus dem Text des Modells mit verbesserter Robustheit.
    
    Args:
        text: Text des Modells mit Entitätsanmerkungen
        lang_code: Sprachcode
        verbose: Ob Debug-Informationen gedruckt werden sollen
        
    Returns:
        Eine Liste von Entitätswörterbüchern mit Text und Typ.
    """
    extracted_entities = []
    # Regex zum Finden von Entitäten im Format [TYP: Entitätstext]
    # Es erlaubt Leerzeichen um den Typ, Doppelpunkt und den Entitätstext.
    # Es erfasst den Typ und den Entitätstext.
    entity_pattern = r"\[\s*([A-Z]+)\s*:\s*([^]]+?)\s*\]"

    # Sprachspezifische "keine Entitäten gefunden" Marker
    no_entities_markers = {
        "en": "[NO_ENTITIES_FOUND]",
        "sw": "[HAKUNA_VITU_VILIVYOPATIKANA]",
        "ha": "[BABU_SUNAYEN_DA_AKA_SAMU]"
    }
    no_entities_marker = no_entities_markers.get(lang_code, "[NO_ENTITIES_FOUND]") # Standardmäßig auf Englisch, falls lang_code unbekannt

    # Prüft, ob das Modell-Output anzeigt, dass keine Entitäten gefunden wurden
    if no_entities_marker in text:
        if verbose:
            logging.info(f"Marker '{no_entities_marker}' gefunden. Keine Entitäten extrahiert.")
        return []

    # Weitere robuste Prüfung für Variationen wie "[NONE](NONE)" oder ähnlich, hauptsächlich für ältere Prompt-Kompatibilität
    # Dies sollte weniger notwendig sein, da die neuen spezifischen Marker verwendet werden, aber für Sicherheit beibehalten.
    none_patterns = [
        r"\[NONE\]\(NONE\)", # Exakte Übereinstimmung für [NONE](NONE)
        r"\[HAKUNA\]\(HAKUNA\)", # Swahili [HAKUNA](HAKUNA)
        r"\[BABU\]\(BABU\)"    # Hausa [BABU](BABU)
    ]
    for none_pattern_str in none_patterns:
        if re.search(none_pattern_str, text, re.IGNORECASE):
            if verbose:
                logging.info(f"Legacy-keine-Entität-Marker über Regex gefunden: {none_pattern_str}. Keine Entitäten extrahiert.")
            return []

    matches = re.finditer(entity_pattern, text)
    for match in matches:
        entity_type = match.group(1)
        entity_text = match.group(2)
        entity_type_norm = normalize_entity_type(entity_type.strip().upper())
        entity_text_clean = entity_text.strip().strip('"\'')
        if len(entity_text_clean) > 1 and entity_type_norm in ["PER", "ORG", "LOC", "DATE"]:
            extracted_entities.append({"text": entity_text_clean, "type": entity_type_norm})
            if verbose: print(f"    Added: {entity_text_clean} [{entity_type_norm}]")
        elif verbose:
            print(f"    Skipped (pattern): Type='{entity_type_norm}', Text='{entity_text_clean}'")

    # Entfernt Duplikate, während die Reihenfolge (Fall-Sensitivität für Text) beibehalten wird
    unique_entities = []
    seen = set()
    for entity in extracted_entities:
        # Erstellt einen eindeutigen Schlüssel für jede Entität (Text + Typ)
        key = (entity["text"].lower(), entity["type"])
        if key not in seen:
            # Zusätzliche Prüfung: Stellt sicher, dass der Typ standardmäßig ist
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
    Normalisiert den Entitätstyp auf Standard-NER-Kategorien.
    
    Args:
        entity_type: Roher Entitätstyp vom Modell-Output
        
    Returns:
        Normalisierter Entitätstyp
    """
    entity_type = entity_type.upper().strip()
    
    # Zuordnung zu Standard-Typen mit umfassenderer Abdeckung
    if any(t in entity_type for t in ["PERSON", "PER", "INDIVIDUAL", "HUMAN", "NAME"]):
        return "PER"
    elif any(t in entity_type for t in ["ORG", "ORGANIZATION", "COMPANY", "INSTITUTION", "AGENCY", "GROUP"]):
        return "ORG"
    elif any(t in entity_type for t in ["LOC", "LOCATION", "PLACE", "AREA", "REGION", "COUNTRY", "CITY"]):
        return "LOC"
    elif any(t in entity_type for t in ["DATE", "TIME", "DATETIME", "PERIOD", "DAY", "MONTH", "YEAR"]):
        return "DATE"
    else:
        return entity_type  # Gibt als ist zurück, falls keine Übereinstimmung

def process_ner_baseline(
    tokenizer: Any, 
    model: Any, 
    text: str, 
    max_input_length: int = 4096,
    max_new_tokens: int = 200, # Max. Tokens für die NER-Ausgabe selbst
    # Vereinheitlichte Generierungsparameter (erwartet, dass sie übergeben werden)
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    do_sample: bool = True, # Behält do_sample bei, da es basierend auf der Temperatur festgelegt werden könnte
    lang_code: str = "en",
    model_name: str = "",
    prompt_in_lrl: bool = False,
    use_few_shot: bool = True,
    verbose: bool = False
) -> Tuple[List[Dict[str, str]], float, str]: # Rückgabetyp aktualisiert
    """
    Verarbeitet einen einzelnen Text für NER unter Verwendung des Baseline-Ansatzes.
    
    Args:
        tokenizer: Der Tokenizer.
        model: Das Modell.
        text: Der Text, der verarbeitet werden soll.
        max_input_length: Maximale Token-Länge für die Modelleingabe.
        max_new_tokens: Maximale neue Tokens, die generiert werden sollen.
        ... (andere Generierungsparameter)
        
    Returns:
        Ein Tupel mit (Liste der Entitäten, Laufzeit, rohe Antwort).
    """
    start_time = time.time()
    
    # Wählt den Prompt-Generierungsmodus basierend auf der Einstellung prompt_in_lrl.
    if prompt_in_lrl:
        prompt = generate_lrl_instruct_ner_prompt(text, lang_code, model_name, use_few_shot)
    else:
        prompt = generate_ner_prompt(text, lang_code, model_name, use_few_shot)
    
    # Erzwingt do_sample=False, wenn die Temperatur sehr niedrig ist, für mehr deterministische Ausgaben.
    # Dies ist nützlich für die Fehlersuche bei bestimmten Modellen.
    if temperature < 0.01:
        do_sample = False

    # Stellt sicher, dass die Eingabelänge die maximale Länge des Modells nicht überschreitet.
    # Dies ist eine Sicherheitsmaßnahme.
    model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
    # Begrenzt auch die maximale Länge, um sicherzustellen, dass sie nicht übermäßig ist.
    safe_max_input_length = min(model_max_len, 8192) 
    
    try:
        # Tokenisiert den Prompt und stellt sicher, dass er auf das richtige Gerät verschoben wird.
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=safe_max_input_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generierungsargumente als Wörterbuch für Sauberkeit.
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": tokenizer.eos_token_id
        }

        # Passt die Generierungsargumente basierend auf dem Sampling-Modus an.
        if do_sample:
            generation_args.update({
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": True
            })
        else: # Verwende gierige Dekodierung
            generation_args["do_sample"] = False
            
        # Generiert die Antwortsequenz.
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_args)
        
        # Dekodiert nur die neu generierten Tokens, um den Prompt auszuschließen.
        raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Extrahiert Entitäten aus der rohen Ausgabe.
        # Der verbose-Parameter hilft bei der Fehlersuche bei Extraktionsfehlern.
        entities = extract_entities(raw_output, lang_code, verbose=verbose)
        
        runtime = time.time() - start_time
        return entities, runtime, raw_output

    except Exception as e:
        # Protokolliert alle Fehler, die während der Verarbeitung auftreten.
        logger.error(f"Error processing text for NER baseline: {e}", exc_info=True)
        runtime = time.time() - start_time
        return [], runtime, f"Error: {e}"

# Add a static counter to the function
process_ner_baseline.sample_counter = 0

def evaluate_ner_baseline(
    tokenizer: Any,
    model: Any,
    samples_df: pd.DataFrame,
    lang_code: str,
    model_name: str,
    prompt_in_lrl: bool,
    use_few_shot: bool,
    # Vereinheitlichte Generierungsparameter (erwartet vom Runner-Skript)
    temperature: float, 
    top_p: float, 
    top_k: int, 
    max_tokens: int,
    repetition_penalty: float, 
    do_sample: bool = True
) -> pd.DataFrame:
    """
    Evaluates the NER baseline approach on a dataset.
    Accepts unified generation parameters.
    
    Returns:
        A DataFrame with detailed results for each sample.
    """
    results = []
    # The input samples_df should already have 'text' and 'entities' (as ground_truth) correctly formatted by load_masakhaner_samples.
    # If 'tokens' is present, it can be used, but 'text' is primary for process_ner_baseline.

    shot_description = "few-shot" if use_few_shot else "zero-shot"
    prompt_lang_description = "LRL-instruct" if prompt_in_lrl else "EN-instruct"

    logging.info(f"Starting NER baseline evaluation for {model_name} on {lang_code} ({prompt_lang_description}, {shot_description}).")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing NER baseline ({lang_code}, {shot_description})"):
        text_to_process = row.get('text')
        # Ground truth entities should be in row['entities'] and already be a list of dicts
        # e.g., [{'entity': 'text', 'type': 'PER'}, ...]
        ground_truth_entities = row.get('entities', []) 

        if not text_to_process:
            logging.warning(f"Sample {idx} has no text. Skipping.")
            # Optionally, append a row with error or skip
            results.append({
                'id': row.get('id', idx),
                'text': text_to_process,
                'ground_truth_entities': ground_truth_entities,
                'predicted_entities': [], # No prediction possible
                'error_message': 'Missing text',
                # Add other relevant fields for consistency in the DataFrame
                'language': lang_code,
                'model_name': model_name,
                'shot_type': shot_description,
                'prompt_language': prompt_lang_description
            })
            continue

        start_time_sample = time.time()
        predicted_entities_list, runtime_sample, raw_output = process_ner_baseline(
            tokenizer=tokenizer,
            model=model,
            text=text_to_process,
            lang_code=lang_code, # lang_code of the text
            model_name=model_name,
            prompt_in_lrl=prompt_in_lrl, # Whether to use LRL instructions
            use_few_shot=use_few_shot,
            # Pass unified generation parameters
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_tokens, # max_tokens is for NER output length
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            verbose=False # Set to True for detailed extraction logs from process_ner_baseline
        )
        runtime_sample = time.time() - start_time_sample

        results.append({
            'id': row.get('id', idx),
            'text': text_to_process,
            'ground_truth_entities': ground_truth_entities, # This should be the pre-processed list of dicts
            'predicted_entities': predicted_entities_list,
            'runtime_seconds': runtime_sample,
            'language': lang_code,
            'model_name': model_name, 
            'shot_type': shot_description,
            'prompt_language': prompt_lang_description,
            # Optionally log the exact generation parameters used for this sample
            'temperature_used': temperature,
            'top_p_used': top_p,
            'top_k_used': top_k,
            'max_tokens_used': max_tokens,
            'repetition_penalty_used': repetition_penalty,
            'do_sample_used': do_sample,
            'raw_output': raw_output
        })

    results_df = pd.DataFrame(results)
    return results_df

def calculate_ner_metrics(gold_entities: List[Dict[str, str]], predicted_entities: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Berechnet NER-Metriken (Präzision, Recall, F1) auf Entitätsebene.
    Annahme: Entitäten sind Wörterbücher mit 'text' und 'type'.
    Dies ist eine strenge Übereinstimmung basierend auf exaktem Text und Typ.
    
    Returns:
        Ein Wörterbuch mit Präzision, Recall und F1-Score.
    """
    
    # Stellt sicher, dass die Eingaben Listen sind, um Fehler zu vermeiden.
    if not isinstance(gold_entities, list): gold_entities = []
    if not isinstance(predicted_entities, list): predicted_entities = []

    # Konvertiert Listen von Wörterbüchern in Sets von Tupeln (entity, type) für einen effizienten Vergleich.
    # Konvertiert Text zur Konsistenz in Kleinbuchstaben.
    gold_set = set((d.get('entity', d.get('text', '')).lower().strip(), d['type']) for d in gold_entities if d.get('entity', d.get('text')))
    predicted_set = set((d.get('entity', d.get('text', '')).lower().strip(), d['type']) for d in predicted_entities if d.get('entity', d.get('text')))

    # Berechnet True Positives, False Positives und False Negatives.
    tp = len(gold_set.intersection(predicted_set))
    fp = len(predicted_set) - tp
    fn = len(gold_set) - tp

    # Berechnet Präzision, Recall und F1-Score.
    # Behandelt den Fall der Division durch Null, um Fehler zu vermeiden.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }

def load_masakhaner_samples(lang_code: str, split: str = "test", num_samples: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """
    Lädt MasakhaNER-Samples für eine bestimmte Sprache von Hugging Face Hub.
    Enthält eine Zuordnung für häufige Sprachcodes zu HF-Dataset-Konfigurationsnamen.
    Args:
        lang_code: Sprachcode (z.B., 'ha', 'sw').
        split: Datensplitt zum Verwenden ('train', 'validation', 'test').
        num_samples (Optional[int]): Die Anzahl der zu ladenden Samples. Wenn None, werden alle Samples geladen.
        seed: Der Zufalls-Seed für das Sampling.
    
    Returns:
        Ein Pandas DataFrame mit den geladenen Samples.
    """
    dataset_name = "masakhane/masakhaner"
    
    MASAKHANER_LANG_CONFIGS = ["amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "swa", "wol", "yor"]
    lang_config_map = {
        'ha': 'hau',
        'sw': 'swa',
    }
    hf_config_name = lang_config_map.get(lang_code.lower(), lang_code.lower())

    if hf_config_name not in MASAKHANER_LANG_CONFIGS:
        logging.error(f"Unsupported MasakhaNER language config '{hf_config_name}' (from input '{lang_code}'). Supported: {MASAKHANER_LANG_CONFIGS}")
        return pd.DataFrame()

    logging.info(f"Attempting to load '{hf_config_name}' samples from {dataset_name}, split '{split}'...")

    def _convert_raw_tags_to_entities_baseline(tokens: List[str], ner_tags_indices: List[int], tag_names_list: List[str]) -> List[Dict[str, str]]:
        """
        Konvertiert IOB-style NER-Tags in eine Liste von Entitätswörterbüchern.
        Ausgabeformat: [{'entity': 'Entitätstext', 'type': 'TYP'}, ...]
        """
        entities_found = []
        current_entity_tokens_list = []
        current_entity_tag_id_val = -1
        
        for i, token_item in enumerate(tokens):
            tag_id_val = ner_tags_indices[i]
            tag_name_val = tag_names_list[tag_id_val]

            if tag_name_val.startswith("B-"):
                if current_entity_tokens_list: # Vorherige Entität abschließen
                    entity_type_str = tag_names_list[current_entity_tag_id_val].split('-')[1]
                    entities_found.append({"entity": " ".join(current_entity_tokens_list), "type": entity_type_str})
                current_entity_tokens_list = [token_item]
                current_entity_tag_id_val = tag_id_val
            elif tag_name_val.startswith("I-"):
                # Fortsetzung der aktuellen Entität, wenn das I-Tag mit dem B-Tag übereinstimmt
                # (z.B., B-PER gefolgt von I-PER)
                if current_entity_tokens_list and tag_names_list[current_entity_tag_id_val].split('-')[1] == tag_name_val.split('-')[1]:
                    current_entity_tokens_list.append(token_item)
                else: # Ungültiges I-Tag (kein B- oder Typ-Mismatch)
                    if current_entity_tokens_list: # Vorherige abschließen, falls vorhanden
                        entity_type_str = tag_names_list[current_entity_tag_id_val].split('-')[1]
                        entities_found.append({"entity": " ".join(current_entity_tokens_list), "type": entity_type_str})
                    # Neue Entität mit diesem I-Tag beginnen, als ob es ein B- wäre
                    current_entity_tokens_list = [token_item]
                    current_entity_tag_id_val = tag_id_val 
            else: # O-Tag oder unerwartetes Tag
                if current_entity_tokens_list: # Vorherige Entität abschließen
                    entity_type_str = tag_names_list[current_entity_tag_id_val].split('-')[1]
                    entities_found.append({"entity": " ".join(current_entity_tokens_list), "type": entity_type_str})
                    current_entity_tokens_list = []
                    current_entity_tag_id_val = -1
        
        if current_entity_tokens_list: # Nach Schleife, überprüft, ob noch eine Entität vorhanden ist
            entity_type_str = tag_names_list[current_entity_tag_id_val].split('-')[1]
            entities_found.append({"entity": " ".join(current_entity_tokens_list), "type": entity_type_str})
        return entities_found

    all_samples_list = []
    try:
        dataset = load_dataset(dataset_name, name=hf_config_name, split=split, trust_remote_code=True)
        logging.info(f"Successfully loaded dataset for {hf_config_name}, split {split}. Full size: {len(dataset)}")
        
        tag_feature = dataset.features['ner_tags']
        if hasattr(tag_feature, 'feature') and hasattr(tag_feature.feature, 'names'):
            tag_names = tag_feature.feature.names
        else:
            logging.error(f"Could not retrieve NER tag names for {hf_config_name}. Aborting.")
            return pd.DataFrame()

        for i, example in enumerate(dataset):
            tokens = example.get('tokens', [])
            ner_tags_indices = example.get('ner_tags', []) 
            sample_id = example.get('id', f"{hf_config_name}_{split}_{i}")

            if not tokens or not isinstance(ner_tags_indices, list) or len(tokens) != len(ner_tags_indices):
                logging.warning(f"Skipping sample {sample_id} for {hf_config_name} due to missing/mismatched tokens or ner_tags format.")
                continue
            
            # Verwendet die lokale Hilfsfunktion, um Tags in Entitäten umzuwandeln
            entities = _convert_raw_tags_to_entities_baseline(tokens, ner_tags_indices, tag_names)
            
            all_samples_list.append({
                'id': sample_id,
                'tokens': tokens,
                'text': " ".join(tokens),
                'ner_tags_indices': ner_tags_indices, 
                'tag_names': tag_names, 
                'entities': entities, # Dies ist der entscheidende Teil für die Baseline-Auswertung
                'language': lang_code 
            })
        
        if not all_samples_list:
            logging.warning(f"No samples processed for language '{hf_config_name}', split '{split}'.")
            return pd.DataFrame()
                
        all_samples_df = pd.DataFrame(all_samples_list)

        # Konvertiert Tokens in Text für die 'text' Spalte
        if 'tokens' in all_samples_df.columns:
            all_samples_df['text'] = all_samples_df['tokens'].apply(lambda t: " ".join(t) if isinstance(t, list) else "")
            logger.info(f"First 5 'text' entries after creation in load_masakhaner_samples for {lang_code}:\n{all_samples_df[['tokens', 'text']].head().to_string()}")
            # Prüft auf leere Textzeichenfolgen
            empty_texts = all_samples_df[all_samples_df['text'] == ''].shape[0]
            if empty_texts > 0:
                logger.warning(f"{empty_texts}/{len(all_samples_df)} samples have empty 'text' after conversion from 'tokens' in load_masakhaner_samples for {lang_code}.")
                if empty_texts == len(all_samples_df):
                    logger.error(f"CRITICAL: All {len(all_samples_df)} samples have empty 'text' for {lang_code}. This will lead to all samples being skipped.")
        else:
            logger.error(f"'tokens' column not found in data for {lang_code}. Cannot create 'text' column.")
            # Erstellt eine leere 'text' Spalte, um KeyErrors im Downstream zu vermeiden, obwohl die Verarbeitung wahrscheinlich fehlschlägt
            all_samples_df['text'] = ""

        if num_samples is not None:
            if num_samples > 0:
                all_samples_df = all_samples_df.sample(frac=1, random_state=seed).reset_index(drop=True)
                if num_samples >= len(all_samples_df):
                    logging.info(f"Requested {num_samples} samples for {hf_config_name} ({split}), but only {len(all_samples_df)} are available. Using all.")
                    final_samples_df = all_samples_df
                else:
                    final_samples_df = all_samples_df.head(num_samples)
                    logging.info(f"Selected {len(final_samples_df)} samples for {hf_config_name} ({split}) after requesting {num_samples} with seed {seed}.")
            else: 
                logging.warning(f"Requested {num_samples} samples for {hf_config_name} ({split}). Returning empty DataFrame.")
                return pd.DataFrame()
        else:
            final_samples_df = all_samples_df
            logging.info(f"Returning all {len(final_samples_df)} MasakhaNER samples for {hf_config_name} ({split}) as num_samples was None.")
        
        logging.info(f"Loaded and processed {len(final_samples_df)} samples for {hf_config_name}, split '{split}'.")
        return final_samples_df
    
    except Exception as e: # Breitere Ausnahmebehandlung
        logging.error(f"Error loading MasakhaNER data for {hf_config_name}, split {split}: {e}", exc_info=True)
        logging.warning("Falling back to dummy data due to error...")
        return create_dummy_ner_data(lang_code, num_samples if num_samples is not None and num_samples > 0 else 5)

def create_dummy_ner_data(lang_code: str, num_samples: int = 5) -> pd.DataFrame:
    """
    Erstellt Dummy-NER-Daten für Tests.
    
    Args:
        lang_code: Sprachcode
        num_samples: Die Anzahl der zu generierenden Dummy-Samples.
        
    Returns:
        Ein Pandas DataFrame mit Dummy-NER-Daten.
    """
    dummy_data = []
    # Beispiel-Entitäten für jede Sprache.
    entities_map = {
        "swa": [
            ("Juma", "PER"), ("Dar es Salaam", "LOC"), ("Benki ya CRDB", "ORG"), ("jana", "DATE")
        ],
        "hau": [
            ("Musa", "PER"), ("Kano", "LOC"), ("Kamfanin Dangote", "ORG"), ("gobe", "DATE")
        ]
    }
    
    # Beispiel-Sätze für jede Sprache.
    sentences_map = {
        "swa": "{} alikwenda {} na kufanya kazi na {} {}",
        "hau": "{} ya tafi {} don yin aiki da {} a {}",
    }
    
    # Wählt die richtigen Entitäten und Sätze basierend auf dem Sprachcode aus.
    entities = entities_map.get(lang_code, entities_map['swa']) # Standardmäßig Swahili
    sentence_template = sentences_map.get(lang_code, sentences_map['swa'])

    # Generiert die angegebene Anzahl von Dummy-Samples.
    for i in range(min(num_samples, len(sentences_map))): # Anzahl der Sätze ist die Anzahl der Dummy-Samples
        # Wählt zufällige Entitäten für jeden Slot aus.
        entity_tuple = (
            entities[i % len(entities)][0], # Person
            entities[(i + 1) % len(entities)][0], # Ort
            entities[(i + 2) % len(entities)][0], # Organisation
            entities[(i + 3) % len(entities)][0] # Datum
        )
        
        # Erstellt den Dummy-Text.
        text = sentence_template.format(*entity_tuple)
        
        # Erstellt die Ground-Truth-Entitäten-Struktur.
        ground_truth_entities = [
            {"entity": entity_tuple[0], "type": "PER"},
            {"entity": entity_tuple[1], "type": "LOC"},
            {"entity": entity_tuple[2], "type": "ORG"},
            {"entity": entity_tuple[3], "type": "DATE"}
        ]
        
        # Fügt die Dummy-Daten zur Liste hinzu.
        dummy_data.append({
            "text": text,
            "entities": ground_truth_entities
        })
        
    # Gibt ein DataFrame aus den Dummy-Daten zurück.
    return pd.DataFrame(dummy_data)


# Hauptfunktion zum Ausführen von Tests, wenn das Skript direkt ausgeführt wird
def main():
    """
    Hauptfunktion zum Testen der Baseline-NER-Pipeline.
    """
    import argparse
    # Definiert Befehlszeilenargumente für einfache Tests.
    parser = argparse.ArgumentParser(description="Test NER Baseline")
    parser.add_argument("--model-name", type=str, default="CohereForAI/aya-23-8B", help="Model to test")
    parser.add_argument("--lang-code", type=str, default="swa", help="Language code for MasakhaNER (e.g., swa, hau)")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--prompt-in-lrl", action="store_true", help="Use instructions in the LRL")
    parser.add_argument("--use-few-shot", action="store_true", help="Use few-shot examples")
    parser.add_argument("--save-results", action="store_true", help="Save results to a CSV file")
    
    args = parser.parse_args()

    # Initialisiert das Modell und den Tokenizer.
    try:
        tokenizer, model = initialize_model(args.model_name)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}", exc_info=True)
        return

    # Lädt Test-Samples aus dem MasakhaNER-Datensatz.
    samples_df = load_masakhaner_samples(args.lang_code, num_samples=args.num_samples)
    
    if samples_df.empty:
        print(f"No samples found for language {args.lang_code}. Exiting.")
        return

    # Führt die Baseline-NER-Evaluierung aus.
    # Übergibt die fest codierten Generierungsparameter für den Test.
    results_df = evaluate_ner_baseline(
        tokenizer,
        model,
        samples_df,
        lang_code=args.lang_code,
        model_name=args.model_name,
        prompt_in_lrl=args.prompt_in_lrl,
        use_few_shot=args.use_few_shot,
        # Harte Codierung der Parameter für den direkten Skriptlauf
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        max_tokens=200,
        repetition_penalty=1.1,
        do_sample=True
    )

    # Druckt die Ergebnisse in die Konsole.
    print(f"Results for {args.model_name} on {args.lang_code} ({'LRL-instruct' if args.prompt_in_lrl else 'EN-instruct'}, {'few-shot' if args.use_few_shot else 'zero-shot'}):")
    print(results_df[['text', 'ground_truth_entities', 'predicted_entities', 'f1', 'precision', 'recall']].head())
    
    # Berechnet und druckt die durchschnittlichen Metriken.
    avg_f1 = results_df['f1'].mean()
    avg_precision = results_df['precision'].mean()
    avg_recall = results_df['recall'].mean()

    print(f"\\nAverage F1 Score: {avg_f1:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")

    # Speichert die Ergebnisse in einer CSV-Datei, falls angefordert.
    if args.save_results:
        # Erstellt ein Ausgabeverzeichnis.
        output_dir = "results/ner_baseline_test"
        os.makedirs(output_dir, exist_ok=True)
        
        # Erstellt einen Dateinamen, der die Experimentkonfiguration widerspiegelt.
        shot_type = "fs" if args.use_few_shot else "zs"
        prompt_type = "lrl" if args.prompt_in_lrl else "en"
        model_short_name = args.model_name.split('/')[-1]
        
        file_name = f"test_{model_short_name}_{args.lang_code}_{prompt_type}_{shot_type}.csv"
        output_path = os.path.join(output_dir, file_name)
        
        # Speichert das DataFrame.
        results_df.to_csv(output_path, index=False)
        print(f"\\nResults saved to {output_path}")


if __name__ == "__main__":
    # Konfiguriert das Logging, um detaillierte Informationen anzuzeigen.
    logging.basicConfig(level=logging.INFO)
    main() 