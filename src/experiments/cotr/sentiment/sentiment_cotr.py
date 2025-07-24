
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os
import re
from typing import Any, Dict, List, Tuple, Optional
import logging
import time
import json # Für robustes Parsen hinzugefügt

# --- Globaler Logger ---
logger = logging.getLogger(__name__)

# COMET-Dienstprogramme entfernt - die Übersetzungsevaluierung erfolgt separat mit NLLB

# Sprachnamen global definieren für den Zugriff durch mehrere Funktionen
LANG_NAMES = {
    "en": "English",
    "sw": "Swahili", # Korrigierter LRL-Code
    "ha": "Hausa",   # Korrigierter LRL-Code
    "am": "Amharic",
    "dz": "Dzongkha",
    "pcm": "Nigerian Pidgin",
    "yo": "Yoruba",
    "ma": "Marathi",
    "multi": "Multilingual",
    "te": "Telugu", # Aus QA, könnte bei Vereinheitlichung benötigt werden
    "pt": "Portuguese" # Portugiesisch hinzugefügt
    # Weitere Sprachen aus den Datensätzen nach Bedarf hinzufügen
}

# Englische Sentiment-Labels definieren (typischerweise aus dem Datensatz, als Ziel für die englische Klassifizierung verwendet)
ENGLISH_SENTIMENT_LABELS = ["positive", "negative", "neutral"]
EXPECTED_LABELS = ENGLISH_SENTIMENT_LABELS # Zur Konsistenz der Extraktion

# LRL-Übersetzungen dieser englischen Sentiment-Labels definieren
# Dies ist entscheidend für die Berechnung des COMET-Scores der Label-Rückübersetzung und für Single-Prompt-Beispiele
SENTIMENT_LABELS_LRL = {
    "sw": {"positive": "chanya", "negative": "hasi", "neutral": "kati"}, # War "sioegemea", "kati" ist gebräuchlich
    "ha": {"positive": "tabbatacce", "negative": "korau", "neutral": "tsaka-tsaki"},
    "am": {"positive": "አዎንታዊ", "negative": "አሉታዊ", "neutral": "ገለልተኛ"},
    "yo": {"positive": "rere", "negative": "búburú", "neutral": "dídọ̀ọ̀dọ́"}, # Akzente für Yoruba hinzugefügt
    "pcm": {"positive": "good", "negative": "bad", "neutral": "neutral"}, # Pidgin könnte englischähnliche Begriffe verwenden
    "pt": {"positive": "positivo", "negative": "negativo", "neutral": "neutro"}, # Portugiesisch hinzugefügt
    # Für Telugu, falls die Sentiment-Aufgabe erweitert wird:
    # "te": {"positive": "సానుకూల", "negative": "ప్రతికూల", "neutral": "తటస్థ"}
}

# SENTIMENT_LABELS_EN und SENTIMENT_LABELS_EN_STR für den Import durch das Runner-Skript definieren
SENTIMENT_LABELS_EN = ["positive", "negative", "neutral"]
SENTIMENT_LABELS_EN_STR = ", ".join(SENTIMENT_LABELS_EN)

# --- Hilfsfunktion hinzugefügt ---
def _sanitize_for_prompt(text: str) -> str:
    """Grundlegende Bereinigung für Text, der in Prompts enthalten ist."""
    if not isinstance(text, str):
        text = str(text)
    # Backticks und dreifache Anführungszeichen escapen, um Markdown/f-string-Probleme zu vermeiden
    text = text.replace('`', '\\`')
    text = text.replace("'''", "'\\''") # Vorhandene dreifache Anführungszeichen escapen
    text = text.replace('"""', '\"\"\"')
    return text

def get_language_name(lang_code: str) -> str:
    """Hilfsfunktion, um den vollständigen Sprachnamen zu erhalten."""
    return LANG_NAMES.get(lang_code, lang_code.capitalize())

def initialize_model(model_name: str) -> tuple:
    """Initialisiert das Modell und den Tokenizer mit robuster Handhabung des Pad-Tokens."""
    logger.info(f"Initializing model {model_name} for Sentiment CoTR...")
    cache_path = "/work/bbd6522/cache_dir"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_path
    )

    model_load_kwargs = {
        "trust_remote_code": True,
        "cache_dir": cache_path,
    }
    # Modellspezifische Ladeanpassungen
    if "qwen" in model_name.lower():
        logger.info(f"Applying float16 for Qwen model: {model_name}")
        model_load_kwargs["torch_dtype"] = torch.float16
    elif "aya" in model_name.lower():
        logger.info(f"Using default dtype for Aya model: {model_name}")
        # Aya benötigt möglicherweise kein explizites float16 und kann darauf empfindlich reagieren.

    if torch.cuda.is_available():
        logger.info(f"CUDA available. Attempting to load {model_name} on GPU.")
        # Für Modelle, die standardmäßig nicht device_map="auto" verwenden (wie einige Qwen-Konfigurationen)
        # oder wenn wir explizite Kontrolle wünschen.
        # model_load_kwargs["device_map"] = "auto" # Dies in Betracht ziehen, wenn OOMs bei großen Modellen bestehen bleiben.
    else:
        logger.info(f"CUDA not available. Loading {model_name} on CPU.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_load_kwargs
    )
    
    if torch.cuda.is_available() and not model_load_kwargs.get("device_map"):
        try:
            model = model.to("cuda")
            logger.info(f"Model {model_name} successfully moved to CUDA.")
        except Exception as e_to_cuda:
            logger.error(f"Error moving {model_name} to CUDA: {e_to_cuda}", exc_info=True)
            # Entscheiden, ob dies kritisch ist oder ob ein CPU-Fallback für den Lauf akzeptabel ist.
            # Vorerst wird angenommen, dass es kritisch ist, wenn CUDA erwartet wurde.
            raise
    
    # Pad-Token und model.config.pad_token_id robust setzen
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")
        else:
            logger.warning("No pad_token or eos_token found. Adding a new [PAD] token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            logger.info("Resized model token embeddings for the new [PAD] token.")

    if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.info(f"Aligned model.config.pad_token_id with tokenizer.pad_token_id ({tokenizer.pad_token_id})")
            
    logger.info(f"Model {model_name} initialized. Device: {next(model.parameters()).device}")
    return tokenizer, model

def generate_translation_prompt(text: str, source_lang: str, target_lang: str, is_label: bool = False) -> str:
    """Erzeugt einen auf Englisch instruierten Prompt für die Übersetzung."""
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)
    safe_text = _sanitize_for_prompt(text) # Sicherstellen, dass _sanitize_for_prompt definiert ist

    if is_label:
        # Prompt für die Übersetzung eines einzelnen Sentiment-Labels (z.B. "positive")
        # Anweisung muss auf ENGLISCH sein
        return f"""Translate the following English sentiment label into {target_name}:
English Sentiment Label: "{safe_text}"
Provide ONLY the {target_name} translation of this label. Do not add explanations or any other text.
{target_name} Sentiment Label:"""
    else:
        # Prompt für die Übersetzung eines Textblocks
        # Anweisung muss auf ENGLISCH sein
        # Englische Few-Shot-Beispiele für LRL-zu-Englisch-Textübersetzung hinzufügen
        example_lrl_sw = "Nimefurahishwa sana na bidhaa hii."
        example_en_sw = "I am very pleased with this product."
        example_lrl_ha = "Wannan fim din ya kasance mai ban sha'awa."
        example_en_ha = "This film was very interesting."
        example_lrl_pt = "O serviço ao cliente foi péssimo."
        example_en_pt = "The customer service was terrible."

        few_shot_text_trans = ""
        if source_lang == 'sw' and target_lang == 'en':
            few_shot_text_trans = f"Example {source_name} Text: '{example_lrl_sw}'\\nExample English Translation: '{example_en_sw}'"
        elif source_lang == 'ha' and target_lang == 'en':
            few_shot_text_trans = f"Example {source_name} Text: '{example_lrl_ha}'\\nExample English Translation: '{example_en_ha}'"
        elif source_lang == 'pt' and target_lang == 'en':
            few_shot_text_trans = f"Example {source_name} Text: '{example_lrl_pt}'\\nExample English Translation: '{example_en_pt}'"
        # Bei Bedarf weitere elif für andere LRL->EN-Paare hinzufügen

        return f"""Original Text ({source_name}):
'{safe_text}'

Instructions:
Translate the {source_name} text above to fluent and accurate English.
Provide ONLY the English translation. Do not add any introductory text, labels, or explanations.

{few_shot_text_trans}

English Translation:"""

def generate_sentiment_prompt_english(text_en: str, use_few_shot: bool = True) -> str:
    """Erzeugt einen auf Englisch instruierten Prompt für die englische Sentiment-Analyse.
       Weist das Modell an, EINES der vordefinierten englischen Labels auszugeben.
       Verwendet englische Few-Shot-Beispiele.
    """
    safe_text_en = _sanitize_for_prompt(text_en)

    # Anweisung muss auf ENGLISCH sein
    instruction = f"""Analyze the sentiment of the following English text.
Respond with ONLY ONE of these English labels: {SENTIMENT_LABELS_EN_STR}.
Do not add explanations or any other text.

English Text: "{safe_text_en}"

English Sentiment Label:"""

    few_shot_examples_text = f"""Here are some examples:

Example 1:
English Text: "This movie was fantastic, I loved it!"
English Sentiment Label: positive

Example 2:
English Text: "I am not happy with the service provided."
English Sentiment Label: negative

Example 3:
English Text: "The meeting is scheduled for 3 PM."
English Sentiment Label: neutral

---
"""
    if use_few_shot:
        return f"{few_shot_examples_text}\n{instruction}"
    else:
        return instruction

def clean_translation_response(raw_response: str, target_lang: str, is_label: bool = False) -> str:
    """Bereinigt die rohe Übersetzungsantwort des Modells."""
    cleaned = raw_response.strip()

    # Übliche Präfixe/Suffixe entfernen, die Modelle hinzufügen könnten
    target_lang_name = get_language_name(target_lang)
    prefixes_to_remove = [
        f"{target_lang_name} Translation:",
        f"The {target_lang_name.lower()} translation is:",
        "Translation:",
        "Translated text:",
        "Here is the translation:"
    ]
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()

    # Führende/nachgestellte Anführungszeichen entfernen, die Modelle oft hinzufügen
    cleaned = cleaned.strip('\'"`')

    if is_label:
        # Bei Labels erwarten wir ein einzelnes Wort oder eine sehr kurze Phrase.
        # Nimm den ersten signifikanten Teil.
        parts = cleaned.split()
        if parts:
            cleaned = parts[0] # Nimm das erste Wort
        else: # Wenn das Splitten zu einem leeren Ergebnis führt, waren es möglicherweise nur Leerzeichen oder Anführungszeichen
            cleaned = "" # Fallback auf leer, wenn nichts Substantielles übrig bleibt
    
    # Wenn die Antwort immer noch die Prompt-Struktur enthält (z.B. "Original Text: ... Translation: ...")
    # versuche, nur den Übersetzungsteil zu isolieren. Dies ist eine Heuristik.
    # Dies ist bei der verbesserten Prompt-Struktur weniger wahrscheinlich, wird aber als Schutzmaßnahme beibehalten.
    if "Original Text (" in cleaned and target_lang_name + " Translation:" in cleaned:
        # Versuche, nach dem Header der Zielsprache zu splitten.
        # Dieses Regex sucht nach dem Header und erfasst alles danach.
        match = re.search(rf"{re.escape(target_lang_name)}\s*Translation:\s*(.*)", cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
            cleaned = cleaned.strip('\'"`') # Anführungszeichen nach der Extraktion erneut entfernen

    return cleaned

def extract_sentiment_label_cotr(output_text: str, for_lrl_label: bool = False, lang_code: Optional[str] = None) -> str:
    """
    Extrahiert ein Sentiment-Label aus der Modellausgabe.
    Wenn for_lrl_label=True, wird zuerst versucht, bekannte LRL-Labels für den gegebenen lang_code zu finden.
    Andernfalls (oder als Fallback) wird versucht, bekannte englische Labels zu finden.
    Angepasst von sentiment_cotr_old.py für eine robustere Extraktion.
    """
    if not output_text or not isinstance(output_text, str):
        logger.warning(f"extract_sentiment_label_cotr received invalid output_text: {output_text}")
        return "unknown"

    text_to_process = output_text.lower().strip()
    text_to_process = text_to_process.split('\\n')[0].strip() # Nur erste Zeile nehmen
    text_to_process = text_to_process.rstrip('.!') # Nachgestellte Satzzeichen entfernen

    # Zu entfernende Präfixe (aus verschiedenen Modellausgaben)
    prefixes_to_remove = [
        "english sentiment label:", "lrl sentiment label:", "sentiment label:",
        "the sentiment is", "sentiment:", "label:"
    ]
    if lang_code:
        lrl_name = get_language_name(lang_code)
        prefixes_to_remove.extend([
            f"{lrl_name.lower()} sentiment label:",
            f"the {lrl_name.lower()} sentiment is"
        ])

    for prefix in prefixes_to_remove:
        if text_to_process.startswith(prefix):
            text_to_process = text_to_process[len(prefix):].strip()
            break # Nur den ersten passenden Präfix entfernen

    # 1. Wenn for_lrl_label, zuerst versuchen, LRL-Labels zu finden
    if for_lrl_label and lang_code:
        if lang_code in SENTIMENT_LABELS_LRL:
            lrl_map = SENTIMENT_LABELS_LRL[lang_code]
            for eng_label, lrl_label_val in lrl_map.items():
                if text_to_process == lrl_label_val.lower():
                    logger.debug(f"Extracted LRL label '{lrl_label_val}' for {lang_code}, mapped to English '{eng_label}' from '{output_text}'")
                    return eng_label # Das englische Äquivalent zurückgeben
        else:
            logger.warning(f"No LRL sentiment labels defined for lang_code: {lang_code} in SENTIMENT_LABELS_LRL")

    # 2. Versuchen, englische Labels direkt zu finden
    # Dies ist das primäre Ziel für den englischen Sentiment-Analyse-Schritt
    # oder Fallback, wenn die LRL-Extraktion (oben) kein Ergebnis lieferte oder nicht angefordert wurde.
    for label in ENGLISH_SENTIMENT_LABELS:
        if text_to_process == label: # Exakte Übereinstimmung nach Bereinigung
            logger.debug(f"Extracted English label '{label}' directly from '{output_text}'")
            return label
        # Bei Bedarf gängige Variationen prüfen, z.B. "the sentiment is positive"
        # Die Präfixentfernung sollte viele davon abdecken.

    # 3. Fallback: Englische Labels als Substrings prüfen (weniger präzise)
    # Dies war im alten Skript enthalten, kann verrauscht sein, aber einige Fälle abfangen.
    # Überlegen, ob dies zu aggressiv ist oder strenger sein sollte.
    # Vorerst wird eine ähnliche Logik wie bei der Schlüsselwortsuche des alten Skripts beibehalten.
    if "positive" in text_to_process or "good" in text_to_process :
        logger.debug(f"Extracted English label 'positive' via keyword from '{output_text}'")
        return "positive"
    if "negative" in text_to_process or "bad" in text_to_process :
        logger.debug(f"Extracted English label 'negative' via keyword from '{output_text}'")
        return "negative"
    if "neutral" in text_to_process : # "neutral" hat weniger wahrscheinlich Synonyme wie gut/schlecht
        logger.debug(f"Extracted English label 'neutral' via keyword from '{output_text}'")
        return "neutral"
        
    logger.warning(f"Could not extract standard sentiment from '{output_text}'. Cleaned: '{text_to_process}'. Defaulting to 'unknown'.")
    return "unknown" # Standard, wenn kein klares Label gefunden wurde

def translate_text(
    model: Any, tokenizer: Any, text_to_translate: str, source_lang: str, target_lang: str, 
    is_label: bool, model_name: str,
    # Vereinheitlichte Generierungsparameter (Aufrufer stellt diese bereit)
    temperature: float, top_p: float, top_k: int, 
    max_new_tokens: int, repetition_penalty: float, do_sample: bool,
    max_input_length: int = 2048 # Maximale Länge für die Prompt-Tokenisierung
) -> Tuple[str, str, float]:
    """
    Übersetzt Text oder ein Klassifizierungslabel. Verwendet vereinheitlichte Generierungsparameter.
    Gibt zurück: (übersetzter_text, rohe_modellausgabe, laufzeit_sekunden)
    """
    start_time = time.time()
    raw_model_output_str = "[Error Initial]"
    final_translation = "[Translation Error]"

    if not text_to_translate or not text_to_translate.strip(): 
        runtime = time.time() - start_time
        logger.warning(f"translate_text received empty input for {source_lang}->{target_lang}.")
        return "[Empty input to translate]", "[N/A - Empty Input]", runtime
    
    prompt = generate_translation_prompt(text_to_translate, source_lang, target_lang, is_label=is_label)
    
    original_tokenizer_config = None # Für den Hack
    if hasattr(model, 'config'):
        original_tokenizer_config = getattr(tokenizer, 'config', None)
        tokenizer.config = model.config

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs_on_device = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs_on_device,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None, # Temp ist irrelevant, wenn nicht gesampelt wird
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        
        # Nur die neu generierten Tokens dekodieren
        raw_model_output_str = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Die rohe Ausgabe bereinigen
        cleaned_response = clean_translation_response(raw_model_output_str, target_lang, is_label)
        
        if is_label:
            # Für Labels kann extract_sentiment_label_cotr verwendet werden, um LRLs bekannten LRL-Formen zuzuordnen
            # oder um englische Labels zu bereinigen.
            if target_lang != 'en':
                final_translation = extract_sentiment_label_cotr(cleaned_response, for_lrl_label=True, lang_code=target_lang)
                if final_translation == "unknown" or final_translation == cleaned_response.lower().strip('\'"()[]{}<>`'): # Wenn die Extraktion keinem bekannten LRL-Label zugeordnet wurde
                    # Es könnte sein, dass das Modell Englisch oder eine Variante ausgegeben hat. Verwende cleaned_response.
                    final_translation = cleaned_response if cleaned_response else text_to_translate # Fallback auf Original, wenn die Bereinigung zu einem leeren Ergebnis führt
                    logger.debug(f"LRL label translation for '{text_to_translate}' to '{target_lang}' resulted in '{final_translation}' after extraction fallback.")

            else: # Ziel ist ein englisches Label
                final_translation = extract_sentiment_label_cotr(cleaned_response, for_lrl_label=False)
            
            # Sicherstellen, dass das Label nicht leer ist; Fallback auf Originaltext, wenn es nach der Bereinigung/Extraktion leer wird
            if not final_translation.strip() and text_to_translate.strip():
                final_translation = text_to_translate 
                logger.warning(f"Label translation for '{text_to_translate}' became empty, falling back to original.")

        else: # Keine Label, allgemeine Textübersetzung
            final_translation = cleaned_response
            if not final_translation.strip():
                final_translation = "[No translation generated]"
                logger.warning(f"Text translation for '{text_to_translate[:50]}...' resulted in empty string, using fallback placeholder.")

    except Exception as e:
        logger.error(f"Error during translate_text ({model_name}, {source_lang}->{target_lang}): {e}", exc_info=True)
        raw_model_output_str = f"[Exception: {str(e)}]"
        # final_translation bleibt "[Translation Error]"
    finally:
        if hasattr(model, 'config'): # Nur wenn der Hack angewendet wurde
            if original_tokenizer_config is not None:
                tokenizer.config = original_tokenizer_config
            elif hasattr(tokenizer, 'config'): # wenn es hinzugefügt wurde und vorher nicht da war
                del tokenizer.config

    runtime = time.time() - start_time
    return final_translation, raw_model_output_str, runtime

def process_sentiment_english(
    model: Any, tokenizer: Any, text_en: str, use_few_shot: bool, model_name: str, 
    temperature: float, top_p: float, top_k: int, 
    max_new_tokens: int, repetition_penalty: float, do_sample: bool,
    max_input_length: int = 2048 # Maximale Länge für die Prompt-Tokenisierung
) -> Tuple[str, str, float]:
    """
    Führt die Sentiment-Klassifizierung auf Englisch mit dem Modell durch. Verwendet vereinheitlichte Generierungsparameter.
    Gibt zurück: (vorhergesagtes_englisches_label, rohe_modellausgabe, laufzeit_sekunden)
    """
    start_time = time.time()
    raw_model_output_str = "[Error Initial]"
    predicted_english_label = "unknown" # Standardmäßig unbekannt

    if not text_en or text_en.strip() == "" or "[Translation Error]" in text_en or "[Empty input" in text_en:
        logger.warning(f"Skipping English sentiment classification for {model_name} due to invalid input: '{text_en[:100]}'.")
        runtime = time.time() - start_time
        return predicted_english_label, "[N/A - Invalid Input Text]", runtime

    prompt = generate_sentiment_prompt_english(text_en, use_few_shot)
    
    original_tokenizer_config = None # Für den Hack
    if hasattr(model, 'config'):
        original_tokenizer_config = getattr(tokenizer, 'config', None)
        tokenizer.config = model.config

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs_on_device = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs_on_device,
                max_new_tokens=max_new_tokens, # Labels sind kurz
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        raw_model_output_str = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        predicted_english_label = extract_sentiment_label_cotr(raw_model_output_str, for_lrl_label=False)

    except Exception as e:
        logger.error(f"Error during process_sentiment_english ({model_name}): {e}", exc_info=True)
        raw_model_output_str = f"[Exception: {str(e)}]"
        # predicted_english_label bleibt "unknown"
    finally:
        if hasattr(model, 'config'): # Nur wenn der Hack angewendet wurde
            if original_tokenizer_config is not None:
                tokenizer.config = original_tokenizer_config
            elif hasattr(tokenizer, 'config'):
                del tokenizer.config
        
    runtime = time.time() - start_time
    return predicted_english_label, raw_model_output_str, runtime

def evaluate_sentiment_cotr_multi_prompt(
    model_name: str, model: Any, tokenizer: Any, samples_df: pd.DataFrame,
    lang_code: str, use_few_shot: bool,
    # Generierungsparameter für jeden Schritt, als Dicts übergeben
    text_translation_params: Dict[str, Any],
    sentiment_classification_params: Dict[str, Any],
    label_translation_params: Dict[str, Any],
    max_input_length: int = 2048 # max_input_length hinzugefügt
) -> pd.DataFrame:
    """
    Evaluiert den Sentiment CoTR MULTI-PROMPT Ansatz.
    Metriken (Genauigkeit, F1) basieren auf dem zwischenzeitlich vorhergesagten englischen Label.
    COMET-Scores werden für alle Übersetzungsschritte berechnet.
    Es wird angenommen, dass die Ground-Truth-Labels aus dem Datensatz Englisch sind.
    """
    results_list = []
    lrl_name = get_language_name(lang_code)

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Multi-Prompt Sentiment CoTR {lang_code}"):
        original_text_lrl = str(row['text'])
        ground_truth_english_label = str(row['label']).lower().strip() # Angenommenes englisches GT

        # Für den COMET-Score von EN-Label -> LRL-Label benötigen wir die kanonische LRL-Übersetzung des englischen GT
        ground_truth_lrl_label_for_comet = SENTIMENT_LABELS_LRL.get(lang_code, {}).get(ground_truth_english_label, ground_truth_english_label)

        # Felder initialisieren
        intermediate_en_text, raw_trans_text_out, rt_trans_text = "[Error]", "[Error]", 0.0
        predicted_en_label, raw_classify_out, rt_classify = "unknown", "[Error]", 0.0
        predicted_lrl_label_final, raw_trans_label_out, rt_trans_label = "[Error]", "[Error]", 0.0
        comet_lrl_text_to_en, comet_en_label_to_lrl = None, None

        try:
            # 1. LRL-Text ins Englische übersetzen
            intermediate_en_text, raw_trans_text_out, rt_trans_text = translate_text(
                model, tokenizer, original_text_lrl, lang_code, "en", False, model_name,
                max_input_length=max_input_length, **text_translation_params
            )
            # COMET-Score-Berechnung entfernt - Übersetzungsevaluierung wird separat mit NLLB durchgeführt

            # 2. Englischen Text klassifizieren
            predicted_en_label, raw_classify_out, rt_classify = process_sentiment_english(
                model, tokenizer, intermediate_en_text, use_few_shot, model_name,
                max_input_length=max_input_length, **sentiment_classification_params
            )

            # 3. Englisches Sentiment-Label zurück ins LRL übersetzen (wenn keine englische Eingabe)
            if lang_code != "en" and predicted_en_label != "unknown":
                predicted_lrl_label_final, raw_trans_label_out, rt_trans_label = translate_text(
                    model, tokenizer, predicted_en_label, "en", lang_code, True, model_name,
                    max_input_length=max_input_length, **label_translation_params
                )
                # COMET-Score-Berechnung entfernt - Übersetzungsevaluierung wird separat mit NLLB durchgeführt
            elif lang_code == "en":
                predicted_lrl_label_final = predicted_en_label # Keine Rückübersetzung
                raw_trans_label_out = "[N/A for EN]"
            else: # predicted_en_label war "unknown"
                 predicted_lrl_label_final = "unknown" # oder auf LRL-unbekannt abbilden, falls definiert
                 raw_trans_label_out = "[Skipped - EN label was unknown]"


        except Exception as e_outer:
            logger.error(f"Outer error in multi-prompt sample {idx}, lang {lang_code}: {e_outer}", exc_info=True)
            # Sicherstellen, dass alle Variablen Fehlerstrings zugewiesen bekommen, falls noch nicht geschehen.

        results_list.append({
            'original_text_lrl': original_text_lrl,
            'ground_truth_label': ground_truth_english_label,  # Fehlende Spalte für Metriken hinzufügen
            'predicted_label': predicted_en_label,  # Fehlende Spalte für Metriken hinzufügen - wird für das Scoring dem englischen Label zugeordnet
            'ground_truth_english_label': ground_truth_english_label,
            'intermediate_en_text': intermediate_en_text,
            'predicted_en_label': predicted_en_label, # Dies ist das Label, das für Genauigkeit/F1 verwendet wird
            'predicted_lrl_label_final': predicted_lrl_label_final,
            'comet_lrl_text_to_en': comet_lrl_text_to_en,
            'comet_en_label_to_lrl': comet_en_label_to_lrl,
            'raw_translation_to_en_output': raw_trans_text_out,
            'raw_predicted_en_label_output': raw_classify_out,
            'raw_translation_to_lrl_output': raw_trans_label_out,
            'runtime_translation_to_en_seconds': rt_trans_text,
            'runtime_classification_en_seconds': rt_classify,
            'runtime_translation_to_lrl_seconds': rt_trans_label,
            'language': lang_code,
            'pipeline': 'multi_prompt',
            'shot_type': 'few-shot' if use_few_shot else 'zero-shot'
        })
    return pd.DataFrame(results_list)

# --- Überarbeitete Single-Prompt-Generierung ---
def generate_single_prompt_sentiment_cotr(lrl_text: str, lang_code: str, use_few_shot: bool = True) -> str:
    """
    Erzeugt einen einzelnen umfassenden Prompt für die Sentiment-CoTR-Pipeline.
    Alle Anweisungen sind auf ENGLISCH.
    Das Modell wird angewiesen:
    1. LRL-Text ins Englische zu übersetzen.
    2. Das Sentiment der englischen Übersetzung zu klassifizieren (Ausgabe eines englischen Labels).
    3. Das englische Sentiment-Label zurück in die LRL zu übersetzen.
    Few-Shot-Beispiele (falls verwendet) sind auf Englisch und demonstrieren diesen 3-Schritte-Prozess.
    """
    lrl_name = get_language_name(lang_code)
    lrl_name_fs_display = lrl_name  # Korrektur für NameError
    original_lrl_text_sanitized = _sanitize_for_prompt(lrl_text)

    # Kernanweisungen - ALLE AUF ENGLISCH
    prompt_instructions_template = f"""You are an expert multilingual AI assistant. Your task is to perform sentiment analysis on a given {lrl_name} text by following these three steps precisely:

1.  **Translate to English**: Translate the original '{lrl_name} Text' accurately into English.
    Begin your output for this step with the exact label "English Translation:" followed immediately by the translation.

2.  **Perform English Sentiment Analysis**: Analyze the 'English Translation' from Step 1. Determine if its sentiment is positive, negative, or neutral.
    Begin your output for this step with the exact label "English Sentiment Label:" followed immediately by ONLY ONE of these English labels: {SENTIMENT_LABELS_EN_STR}. Do not add explanations.

3.  **Translate Sentiment Label to {lrl_name}**: Translate the 'English Sentiment Label' from Step 2 accurately into {lrl_name}.
    Begin your output for this step with the exact label "{lrl_name} Sentiment Label:" followed immediately by the {lrl_name} translation of the sentiment label.

Your entire response must follow this three-step structure, with each step clearly delineated by its label."""

    few_shot_section = ""
    if use_few_shot:
        # Standardisiert auf 3 Beispiele, die mit Baseline- und Multi-Prompt-Ansätzen übereinstimmen
        # Beispiel 1: Positiv
        example_lrl_text_fs_1 = "This movie was fantastic, I loved it!"
        example_eng_translation_fs_1 = "This movie was fantastic, I loved it!"
        example_eng_label_fs_1 = "positive"
        example_lrl_label_fs_1 = SENTIMENT_LABELS_LRL.get(lang_code, {}).get('positive', 'positive')

        # Beispiel 2: Negativ
        example_lrl_text_fs_2 = "I am not happy with the service provided."
        example_eng_translation_fs_2 = "I am not happy with the service provided."
        example_eng_label_fs_2 = "negative"
        example_lrl_label_fs_2 = SENTIMENT_LABELS_LRL.get(lang_code, {}).get('negative', 'negative')

        # Beispiel 3: Neutral
        example_lrl_text_fs_3 = "The meeting is scheduled for 3 PM."
        example_eng_translation_fs_3 = "The meeting is scheduled for 3 PM."
        example_eng_label_fs_3 = "neutral"
        example_lrl_label_fs_3 = SENTIMENT_LABELS_LRL.get(lang_code, {}).get('neutral', 'neutral')

        few_shot_section = f"""

Here are examples of how to perform this task:

--- Example 1 ({lrl_name_fs_display} Text) ---
{_sanitize_for_prompt(example_lrl_text_fs_1)}

--- Example 1 Output ---
English Translation:
{_sanitize_for_prompt(example_eng_translation_fs_1)}

English Sentiment Label:
{example_eng_label_fs_1}

{lrl_name_fs_display} Sentiment Label:
{_sanitize_for_prompt(example_lrl_label_fs_1)}

--- Example 2 ({lrl_name_fs_display} Text) ---
{_sanitize_for_prompt(example_lrl_text_fs_2)}

--- Example 2 Output ---
English Translation:
{_sanitize_for_prompt(example_eng_translation_fs_2)}

English Sentiment Label:
{example_eng_label_fs_2}

{lrl_name_fs_display} Sentiment Label:
{_sanitize_for_prompt(example_lrl_label_fs_2)}

--- Example 3 ({lrl_name_fs_display} Text) ---
{_sanitize_for_prompt(example_lrl_text_fs_3)}

--- Example 3 Output ---
English Translation:
{_sanitize_for_prompt(example_eng_translation_fs_3)}

English Sentiment Label:
{example_eng_label_fs_3}

{lrl_name_fs_display} Sentiment Label:
{_sanitize_for_prompt(example_lrl_label_fs_3)}
--- End Examples ---
"""

    task_section = f"""

Now, complete the task for the following input:

--- Your Task ---
{lrl_name} Text:
```
{original_lrl_text_sanitized}
```

Your Response (following the 3 steps precisely):
"""

    final_prompt = prompt_instructions_template
    if use_few_shot:
        final_prompt += few_shot_section
    final_prompt += task_section
    
    # Nur einen Teil des Prompts zur Kürze loggen
    logger.debug(f"Generated Single Sentiment CoTR Prompt for {lang_code} (use_few_shot={use_few_shot}):\n{final_prompt[:1000]}...")
    return final_prompt

# --- Überarbeitete Extraktion für Single Prompt ---
def extract_sentiment_intermediates_from_single_prompt_response(response_text: str, lrl_name_for_extraction: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extrahiert übersetzten englischen Text, englisches Sentiment-Label und LRL-Sentiment-Label
    aus der strukturierten Ausgabe eines einzelnen CoT-Prompts.
    Gibt zurück: (eng_translation, eng_label, lrl_label)
    """
    eng_translation_out: Optional[str] = None
    eng_label_out: Optional[str] = None
    lrl_label_out: Optional[str] = None

    # Regex-Muster (case-insensitive, dotall für mehrzeiligen Inhalt)
    # Sicherstellen, dass Labels spezifisch sind und die Erfassungsgruppe nicht gierig ist (.*?)
    # Lookaheads stellen sicher, dass wir bis zum nächsten erwarteten Label oder zum Ende des Strings erfassen.
    eng_translation_pattern = re.compile(
        r"English Translation:\s*(.*?)(?=English Sentiment Label:|{lrl_name_for_extraction} Sentiment Label:|$)", 
        re.IGNORECASE | re.DOTALL
    )
    eng_label_pattern = re.compile(
        r"English Sentiment Label:\s*(.*?)(?={lrl_name_for_extraction} Sentiment Label:|$)", 
        re.IGNORECASE | re.DOTALL
    )
    lrl_label_pattern = re.compile(
        r"{lrl_name_for_extraction} Sentiment Label:\s*(.*?)(?=$)", 
        re.IGNORECASE | re.DOTALL
    )

    match_eng_text = eng_translation_pattern.search(response_text)
    if match_eng_text:
        eng_translation_out = match_eng_text.group(1).strip()
        # Weitere mögliche Modell-Konversationspräfixe entfernen, falls vorhanden
        common_prefixes = ["Sure, here is the translation:", "Here's the translation:"]
        for prefix in common_prefixes:
            if eng_translation_out.lower().startswith(prefix.lower()):
                eng_translation_out = eng_translation_out[len(prefix):].strip()
        eng_translation_out = eng_translation_out.strip('`') # Backticks entfernen, wenn das Modell die Übersetzung darin einwickelt

    match_eng_label = eng_label_pattern.search(response_text)
    if match_eng_label:
        extracted_eng_label_raw = match_eng_label.group(1).strip()
        # Die Funktion extract_sentiment_label_cotr kann das Label robust finden
        eng_label_out = extract_sentiment_label_cotr(extracted_eng_label_raw, for_lrl_label=False)
        if eng_label_out == "unknown" and extracted_eng_label_raw: # Loggen, wenn die spezifische Extraktion fehlgeschlagen ist, aber Text vorhanden war
            logger.debug(f"extract_sentiment_label_cotr returned 'unknown' for eng_label part: '{extracted_eng_label_raw}'")


    match_lrl_label = lrl_label_pattern.search(response_text)
    if match_lrl_label:
        extracted_lrl_label_raw = match_lrl_label.group(1).strip()
        # extract_sentiment_label_cotr verwenden, aber angeben, dass es sich um ein LRL-Label handelt, falls lang_code verfügbar ist
        # Für einen einzelnen Prompt haben wir hier keinen lang_code, also verlassen wir uns darauf, dass es eine der bekannten LRL-Formen ist
        # oder ein englisches Label, wenn die Rückübersetzung fehlgeschlagen ist.
        # Das Ziel hier ist, das *englische Äquivalent* des vom Modell produzierten LRL-Labels zu erhalten.
        # Wir benötigen lang_code für eine robuste LRL-Label-Extraktion.
        # Vorerst wird angenommen, dass extract_sentiment_label_cotr sein Bestes versucht.
        # Dieser Teil muss möglicherweise verfeinert werden, wenn die LRL-Labels vielfältig sind.
        # Ein temporärer Ansatz: zuerst versuchen, direkte LRL-Übereinstimmung zu finden, falls möglich, dann Englisch.
        temp_lrl_name_lower = lrl_name_for_extraction.lower()
        found_lrl_direct = False
        if temp_lrl_name_lower in SENTIMENT_LABELS_LRL: # z.B. 'swahili' -> 'sw'
            lang_code_for_lrl_extraction = [lc for lc, name in LANG_NAMES.items() if name.lower() == temp_lrl_name_lower]
            if lang_code_for_lrl_extraction:
                lrl_label_out = extract_sentiment_label_cotr(extracted_lrl_label_raw, for_lrl_label=True, lang_code=lang_code_for_lrl_extraction[0])
                if lrl_label_out != "unknown":
                    found_lrl_direct = True
        
        if not found_lrl_direct: # Fallback auf die Behandlung als englisches Label oder allgemeine Extraktion
            lrl_label_out = extract_sentiment_label_cotr(extracted_lrl_label_raw, for_lrl_label=False)

        if lrl_label_out == "unknown" and extracted_lrl_label_raw :
             logger.debug(f"extract_sentiment_label_cotr returned 'unknown' for lrl_label part: '{extracted_lrl_label_raw}'")


    if not eng_translation_out and not eng_label_out and not lrl_label_out:
        logger.warning(f"Could not extract any parts from single prompt sentiment output for {lrl_name_for_extraction}. Raw output: {response_text[:300]}...")
        # Einen einfachen Fallback für das englische Label hinzufügen, wenn andere fehlgeschlagen sind
        if not eng_label_out:
            eng_label_candidate = extract_sentiment_label_cotr(response_text, for_lrl_label=False)
            if eng_label_candidate != "unknown":
                eng_label_out = eng_label_candidate
                logger.info(f"Fallback: Extracted English label '{eng_label_out}' from full response.")
    
    logger.debug(f"SP Sentiment Extraction for {lrl_name_for_extraction}: EN Text='{str(eng_translation_out)[:50]}...', EN Label='{eng_label_out}', LRL Label='{lrl_label_out}' (parsed as EN equivalent)")
    return eng_translation_out, eng_label_out, lrl_label_out


def evaluate_sentiment_cotr_single_prompt(
    model_name: str, model: Any, tokenizer: Any, samples_df: pd.DataFrame,
    lang_code: str, use_few_shot: bool,
    # Vereinheitlichte Generierungsparameter für die gesamte Single-Prompt-Kette
    temperature: float, top_p: float, top_k: int, 
    max_new_tokens: int, repetition_penalty: float, do_sample: bool,
    max_input_length: int = 2048 # max_input_length hinzugefügt
) -> pd.DataFrame:
    """
    Evaluiert den Sentiment CoTR SINGLE-PROMPT Ansatz.
    Es wird angenommen, dass die Ground-Truth-Labels aus dem Datensatz Englisch sind.
    Metriken (Genauigkeit, F1) basieren auf dem zwischenzeitlich vorhergesagten englischen Label, das aus dem CoT extrahiert wurde.
    """
    results_list = []
    lrl_name = get_language_name(lang_code) # Vollständiger Sprachname wie "Swahili"

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Single-Prompt Sentiment CoTR {lang_code}"):
        original_text_lrl = str(row['text'])
        ground_truth_english_label = str(row['label']).lower().strip() # Angenommenes englisches GT
        
        # Für den COMET-Score von EN-Label -> LRL-Label (Bewertung der Qualität der Rückübersetzung)
        # Referenz ist die kanonische LRL-Übersetzung des *ground_truth_english_label*.
        ground_truth_lrl_label_for_comet_ref = SENTIMENT_LABELS_LRL.get(lang_code, {}).get(ground_truth_english_label, ground_truth_english_label)

        # Felder für dieses Beispiel initialisieren
        comet_lrl_text_to_en, comet_en_label_to_lrl = None, None
        intermediate_en_text, intermediate_en_label = "[Error Extr EN Txt]", "unknown"
        predicted_lrl_label_final_raw = "[Error Extr LRL Lbl]"
        raw_model_response_str = "[Error Initial]"
        rt_sample = 0.0

        try:
            start_time_sample = time.time()
            prompt = generate_single_prompt_sentiment_cotr(original_text_lrl, lang_code, use_few_shot)
            
            original_tokenizer_config = None # Für den Hack
            if hasattr(model, 'config'):
                original_tokenizer_config = getattr(tokenizer, 'config', None)
                tokenizer.config = model.config

            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
                inputs_on_device = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                            **inputs_on_device,
                            max_new_tokens=max_new_tokens, # Max. Tokens für die gesamte mehrstufige CoT-Ausgabe
                            temperature=temperature if do_sample else None,
                            top_p=top_p if do_sample else None,
                            top_k=top_k if do_sample else None,
                            repetition_penalty=repetition_penalty,
                        do_sample=do_sample,
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                        )
                    raw_model_response_str = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            finally: # Sicherstellen, dass die Tokenizer-Konfiguration wiederhergestellt wird
                if hasattr(model, 'config'):
                    if original_tokenizer_config is not None: tokenizer.config = original_tokenizer_config
                    elif hasattr(tokenizer, 'config'): del tokenizer.config

            rt_sample = time.time() - start_time_sample

            # Zwischenergebnisse mit dem robusteren Parser extrahieren
            intermediate_en_text_ext, intermediate_en_label_ext, predicted_lrl_label_final_raw_ext = \
                extract_sentiment_intermediates_from_single_prompt_response(raw_model_response_str, lrl_name)

            # Extrahierte Werte oder Fallbacks zuweisen
            intermediate_en_text = intermediate_en_text_ext if intermediate_en_text_ext is not None else "[Extraction Error EN Text]"
            intermediate_en_label = intermediate_en_label_ext if intermediate_en_label_ext is not None and intermediate_en_label_ext in EXPECTED_LABELS else "unknown"
            predicted_lrl_label_final_raw = predicted_lrl_label_final_raw_ext if predicted_lrl_label_final_raw_ext is not None else "[Extraction Error LRL Label]"

            # COMET-Score-Berechnung entfernt - Übersetzungsevaluierung wird separat mit NLLB durchgeführt

        except Exception as e_outer_sp:
            logger.error(f"Outer error in single-prompt sample {idx}, lang {lang_code}: {e_outer_sp}", exc_info=True)
            rt_sample = time.time() - (start_time_sample if 'start_time_sample' in locals() else time.time())
            # Fehlerstrings bereits für intermediate_en_text, intermediate_en_label usw. initialisiert.

        results_list.append({
            'original_text_lrl': original_text_lrl,
            'ground_truth_label': ground_truth_english_label,  # Fehlende Spalte für Metriken hinzufügen
            'predicted_label': intermediate_en_label,  # Fehlende Spalte für Metriken hinzufügen - wird für das Scoring dem englischen Label zugeordnet
            'ground_truth_english_label': ground_truth_english_label,
            'intermediate_en_text': intermediate_en_text,
            'intermediate_en_label': intermediate_en_label, # Dies ist das aus dem CoT extrahierte EN-Label, das für Genauigkeit/F1 verwendet wird
            'predicted_lrl_label_final_raw': predicted_lrl_label_final_raw, # Das LRL-Label aus dem CoT
            'comet_lrl_text_to_en': comet_lrl_text_to_en,
            'comet_en_label_to_lrl': comet_en_label_to_lrl,
            'raw_model_response': raw_model_response_str,
            'runtime_seconds_sample': rt_sample,
            'language': lang_code,
            'pipeline': 'single_prompt',
            'shot_type': 'few-shot' if use_few_shot else 'zero-shot'
        })
    return pd.DataFrame(results_list)

# Platzhalter für main, wenn eine direkte Ausführung erforderlich ist (normalerweise über run_sentiment_cotr.py ausgeführt)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Testing Sentiment CoTR direct execution (prompts, parsing, etc.).")
    
    # Beispieltest für generate_single_prompt_sentiment_cotr
    sample_sw_text = "Huduma hii ni mbaya sana, sitarudi tena!"
    prompt_sw_single_fs = generate_single_prompt_sentiment_cotr(sample_sw_text, 'sw', use_few_shot=True)
    logger.debug(f"Generated Single Prompt (Swahili, Few-Shot):\n{prompt_sw_single_fs}")

    sample_ha_text = "Ina matukar farin ciki da wannan kyauta."
    prompt_ha_single_zs = generate_single_prompt_sentiment_cotr(sample_ha_text, 'ha', use_few_shot=False)
    logger.debug(f"Generated Single Prompt (Hausa, Zero-Shot):\n{prompt_ha_single_zs}")

    # Beispieltest für extract_sentiment_intermediates_from_single_prompt_response
    mock_response_sw_full = """English Text: This service is very bad, I will not return!
English Sentiment Label: negative
Final Sentiment Label (Swahili): hasi"""
    en_text, en_label, lrl_label = extract_sentiment_intermediates_from_single_prompt_response(mock_response_sw_full, "Swahili")
    logger.debug(f"Extracted Intermediates (Swahili): EN Text='{en_text}', EN Label='{en_label}', LRL Label='{lrl_label}'")

    mock_response_ha_partial = """English Text: I am very happy with this gift.
English Sentiment Label: positive
""" # Fehlender LRL-Teil
    en_text_ha, en_label_ha, lrl_label_ha = extract_sentiment_intermediates_from_single_prompt_response(mock_response_ha_partial, "Hausa")
    logger.debug(f"Extracted Intermediates (Hausa, Partial): EN Text='{en_text_ha}', EN Label='{en_label_ha}', LRL Label='{lrl_label_ha}'")
    
    logger.info("Direct execution testing complete.") 