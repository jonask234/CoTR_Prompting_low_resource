
import json
import ast
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Optional
from sklearn.metrics import accuracy_score, f1_score
import argparse
import logging
from collections import defaultdict
import os

# Logging einrichten
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_entities_correctly(entities_str: str) -> List[Tuple[str, str]]:
    if not entities_str or pd.isna(entities_str) or entities_str.strip() == '':
        return []
    
    entities = []
    entities_str = str(entities_str).strip()
    
    try:
        if entities_str.startswith('[{') and entities_str.endswith('}]'):
            try:
                entities_list = json.loads(entities_str)
                for e in entities_list:
                    if isinstance(e, dict):
                        text_key = e.get('text') or e.get('entity')
                        type_key = e.get('type')
                        if text_key and type_key:
                            entities.append((str(text_key).lower().strip(), str(type_key).upper()))
                return entities
            except json.JSONDecodeError:
                try:
                    entities_list = ast.literal_eval(entities_str)
                    for e in entities_list:
                        if isinstance(e, dict):
                            text_key = e.get('text') or e.get('entity')
                            type_key = e.get('type')
                            if text_key and type_key:
                                entities.append((str(text_key).lower().strip(), str(type_key).upper()))
                    return entities
                except:
                    pass
        
        bracket_pattern = r'\[([A-Z]+):\s*([^\]]+)\]'
        matches = re.findall(bracket_pattern, entities_str)
        if matches:
            return [(match[1].strip().lower(), match[0].upper()) for match in matches]
            
        if entities_str.startswith('[') and entities_str.endswith(']'):
            try:
                simple_list = ast.literal_eval(entities_str)
                if isinstance(simple_list, list):
                    return [(str(item).lower().strip(), 'MISC') for item in simple_list if item]
            except:
                pass
                
    except Exception as e:
        logger.debug(f"Error parsing entities '{entities_str}': {e}")
    
    return []

def find_all_result_files(base_dir):
    result_files = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, base_dir)
                result_files.append(relative_path)
    
    return result_files

def extract_file_info(file_path):
    """Extrahiert Aufgabe, Ansatz, Sprache, Modell und Shot-Typ aus dem Dateipfad."""
    
    if 'qa_new/baseline/' in file_path:
        parts = file_path.split('/')
        shot_type = parts[2].lower()  # ZeroShot/FewShot
        language = parts[3]
        filename = parts[4]
        model_match = re.search(r'_(aya-23-8B|Qwen2\.5-7B-Instruct)\.csv$', filename)
        model = model_match.group(1) if model_match else 'unknown'
        return 'qa', 'baseline', language, model, shot_type
    
    elif 'qa_new/cotr/' in file_path and 'results_cotr_' in file_path:
        filename = os.path.basename(file_path)
        match = re.search(r'results_cotr_(mp|sp)_(fs|zs)_qa_tydiqa_([a-z]+)_(aya-23-8B|Qwen2\.5-7B-Instruct)\.csv', filename)
        if match:
            pipeline, shot, language, model = match.groups()
            shot_type = 'fewshot' if shot == 'fs' else 'zeroshot'
            approach = f'cotr_{pipeline}'
            return 'qa', approach, language, model, shot_type
    
    elif 'ner_new/baseline/' in file_path and 'results_baseline_' in file_path:
        filename = os.path.basename(file_path)
        match = re.search(r'results_baseline_(fs|zs)_ner_([a-z]+)_(aya-23-8B|Qwen2\.5-7B-Instruct)\.csv', filename)
        if match:
            shot, language, model = match.groups()
            shot_type = 'fewshot' if shot == 'fs' else 'zeroshot'
            return 'ner', 'baseline', language, model, shot_type
    
    elif 'ner_new/baseline/' in file_path and '_baseline_' in file_path:
        filename = os.path.basename(file_path)
        match = re.search(r'_baseline_(fs|zs)_ner_([a-z]+)_(aya-23-8B|Qwen2\.5-7B-Instruct)\.csv', filename)
        if match:
            shot, language, model = match.groups()
            shot_type = 'fewshot' if shot == 'fs' else 'zeroshot'
            return 'ner', 'baseline', language, model, shot_type
    
    # Behandle NER-CoTR-Muster: ner_new/cotr/results_cotr_mp_zs_ner_ha_aya-23-8B.csv
    elif 'ner_new/cotr/' in file_path and 'results_cotr_' in file_path:
        filename = os.path.basename(file_path)
        match = re.search(r'results_cotr_(mp|sp)_(fs|zs)_ner_([a-z]+)_(aya-23-8B|Qwen2\.5-7B-Instruct)\.csv', filename)
        if match:
            pipeline, shot, language, model = match.groups()
            shot_type = 'fewshot' if shot == 'fs' else 'zeroshot'
            approach = f'cotr_{pipeline}'
            return 'ner', approach, language, model, shot_type
    
    # Behandle NER-CoTR-Muster: ner_new/cotr/_cotr_mp_fs_ner_sw_Qwen2.5-7B-Instruct.csv
    elif 'ner_new/cotr/' in file_path and '_cotr_' in file_path:
        filename = os.path.basename(file_path)
        match = re.search(r'_cotr_(mp|sp)_(fs|zs)_ner_([a-z]+)_(aya-23-8B|Qwen2\.5-7B-Instruct)\.csv', filename)
        if match:
            pipeline, shot, language, model = match.groups()
            shot_type = 'fewshot' if shot == 'fs' else 'zeroshot'
            approach = f'cotr_{pipeline}'
            return 'ner', approach, language, model, shot_type
    
    # Behandle Klassifizierungs-Baseline-Muster: classification_new/baseline/*/results_baseline_classification_*.csv
    elif 'classification_new/baseline/' in file_path and 'results_baseline_classification_' in file_path:
        # Extrahiere die Sprache aus der Pfadstruktur
        path_parts = file_path.split('/')
        # Suche nach dem Sprachcode in den Pfadteilen
        language = 'unknown'
        for part in path_parts:
            if part in ['en', 'sw', 'ha', 'fr', 'pt', 'ur']:
                language = part
                break
        
        filename = os.path.basename(file_path)
        model_match = re.search(r'(aya-23-8B|Qwen2\.5-7B-Instruct)', file_path)
        model = model_match.group(1) if model_match else 'unknown'
        
        # Bestimme den Shot-Typ aus dem Pfad
        shot_type = 'fewshot' if '/fs/' in file_path else 'zeroshot'
        
        return 'classification', 'baseline', language, model, shot_type
    
    # Behandle Klassifizierungs-Baseline-Muster: classification_new/baseline/*/_baseline_classification_*.csv
    elif 'classification_new/baseline/' in file_path and '_baseline_classification_' in file_path:
        # Extrahiere die Sprache aus der Pfadstruktur
        path_parts = file_path.split('/')
        # Suche nach dem Sprachcode in den Pfadteilen
        language = 'unknown'
        for part in path_parts:
            if part in ['en', 'sw', 'ha', 'fr', 'pt', 'ur']:
                language = part
                break
        
        filename = os.path.basename(file_path)
        model_match = re.search(r'(aya-23-8B|Qwen2\.5-7B-Instruct)', file_path)
        model = model_match.group(1) if model_match else 'unknown'
        
        # Bestimme den Shot-Typ aus dem Pfad
        shot_type = 'fewshot' if '/fs/' in file_path else 'zeroshot'
        
        return 'classification', 'baseline', language, model, shot_type
    
    # Behandle Klassifizierungs-CoTR-Muster: classification_new/cotr/*/results_cotr_classification_*.csv
    elif 'classification_new/cotr/' in file_path and 'results_cotr_classification_' in file_path:
        # Extrahiere die Sprache aus der Pfadstruktur und dem Dateinamen
        path_parts = file_path.split('/')
        language = 'unknown'
        for part in path_parts:
            if part in ['en', 'sw', 'ha', 'fr', 'pt', 'ur']:
                language = part
                break
        
        # Bestimme Pipeline und Shot-Typ aus dem Pfad
        pipeline = 'sp' if 'single_prompt' in file_path else 'mp'  # single_prompt oder multi_prompt
        shot_type = 'fewshot' if '/fs/' in file_path else 'zeroshot'
        
        model_match = re.search(r'(aya-23-8B|Qwen2\.5-7B-Instruct)', file_path)
        model = model_match.group(1) if model_match else 'unknown'
        
        approach = f'cotr_{pipeline}'
        return 'classification', approach, language, model, shot_type
    
    # Behandle NLI-Baseline-Muster: results_nli_baseline_en_Qwen_Qwen2.5-7B-Instruct_few_shot_EN-instruct.csv
    elif 'nli_new/baseline/' in file_path and 'results_nli_baseline_' in file_path:
        filename = os.path.basename(file_path)
        match = re.search(r'results_nli_baseline_([a-z]+)_(CohereLabs_aya-23-8B|Qwen_Qwen2\.5-7B-Instruct)_(zero_shot|few_shot)_', filename)
        if match:
            language, model_raw, shot = match.groups()
            model = 'aya-23-8B' if 'aya' in model_raw else 'Qwen2.5-7B-Instruct'
            shot_type = shot.replace('_', '')
            return 'nli', 'baseline', language, model, shot_type
    
    # Behandle NLI-CoTR-Muster: results_cotr_single_prompt_fs_nli_sw_aya_23_8B.csv
    elif 'nli_new/cotr/' in file_path and 'results_cotr_' in file_path:
        filename = os.path.basename(file_path)
        match = re.search(r'results_cotr_(single_prompt|multi_prompt)_(fs|zs)_nli_([a-z]+)_(aya_23_8B|Qwen2_5_7B_Instruct)\.csv', filename)
        if match:
            pipeline, shot, language, model_raw = match.groups()
            model = 'aya-23-8B' if 'aya' in model_raw else 'Qwen2.5-7B-Instruct'
            shot_type = 'fewshot' if shot == 'fs' else 'zeroshot'
            approach = f'cotr_{pipeline.replace("_prompt", "")}'  # cotr_single oder cotr_multi
            return 'nli', approach, language, model, shot_type
    
    # Behandle Sentiment-Baseline-Muster: results_sentiment_baseline_new_ha_LRL-instruct_few_shot.csv
    elif 'sentiment_new/baseline/' in file_path and 'results_sentiment_baseline_new_' in file_path:
        filename = os.path.basename(file_path)
        match = re.search(r'results_sentiment_baseline_new_([a-z]+)_[^_]+_(zero_shot|few_shot)\.csv', filename)
        if match:
            language, shot = match.groups()
            # Extrahiere das Modell aus der Pfadstruktur
            model = 'aya-23-8B' if 'aya' in file_path else 'Qwen2.5-7B-Instruct'
            shot_type = shot.replace('_', '')
            return 'sentiment', 'baseline', language, model, shot_type
    
    # Behandle Sentiment-CoTR-Muster: results_sentiment_cotr_sp_fs_ha_aya-23-8B.csv
    elif 'sentiment_new/cotr/' in file_path and 'results_sentiment_cotr_' in file_path:
        filename = os.path.basename(file_path)
        match = re.search(r'results_sentiment_cotr_(sp|mp)_(fs|zs)_([a-z]+)_(aya-23-8B|Qwen2\.5-7B-Instruct)\.csv', filename)
        if match:
            pipeline, shot, language, model = match.groups()
            shot_type = 'fewshot' if shot == 'fs' else 'zeroshot'
            approach = f'cotr_{pipeline}'  # cotr_sp oder cotr_mp 
            return 'sentiment', approach, language, model, shot_type
    
    # Wenn kein Muster übereinstimmt, versuche, grundlegende Informationen zu extrahieren
    logging.warning(f"Unknown file pattern: {file_path}")
    return 'unknown', 'unknown', 'unknown', 'unknown', 'unknown'

def process_qa_file(file_path: str, relative_path: str) -> Optional[Dict[str, Any]]:
    try:
        df = pd.read_csv(file_path)
        
        # QA-Baseline-Dateien verwenden diese Spaltennamen
        if 'ground_truth' in df.columns and 'predicted_answer' in df.columns:
            f1_scores = []
            for _, row in df.iterrows():
                f1 = calculate_qa_f1_score(str(row['ground_truth']), str(row['predicted_answer']))
                f1_scores.append(f1)
            
            avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
            return {
                'f1_score': avg_f1,
                'sample_count': len(df)
            }
            
        # QA-CoTR-Dateien verwenden andere Spaltennamen
        elif 'lrl_ground_truth_answers_list' in df.columns and 'lrl_answer_model_final' in df.columns:
            f1_scores = []
            for _, row in df.iterrows():
                try:
                    # Parse die Ground-Truth-Liste (normalerweise eine String-Repräsentation einer Liste)
                    import ast
                    gt_list = ast.literal_eval(str(row['lrl_ground_truth_answers_list']))
                    gt_text = gt_list[0] if gt_list and isinstance(gt_list, list) else str(row['lrl_ground_truth_answers_list'])
                    pred_text = str(row['lrl_answer_model_final']) if not pd.isna(row['lrl_answer_model_final']) else ""
                    
                    f1 = calculate_qa_f1_score(gt_text, pred_text)
                    f1_scores.append(f1)
                except:
                    f1_scores.append(0.0)
            
            avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
            return {
                'f1_score': avg_f1,
                'sample_count': len(df)
            }
        else:
            logging.warning(f"QA file {file_path} doesn't have expected columns")
            return None
            
    except Exception as e:
        logging.error(f"Error processing QA file {file_path}: {e}")
        return None

def calculate_qa_f1_score(ground_truth: str, predicted: str) -> float:
    if not ground_truth or pd.isna(ground_truth):
        ground_truth = ""
    if not predicted or pd.isna(predicted):
        predicted = ""
        
    gt_normalized = str(ground_truth).lower().strip()
    pred_normalized = str(predicted).lower().strip()
    
    if not gt_normalized and not pred_normalized:
        return 1.0
    if not gt_normalized or not pred_normalized:
        return 0.0
    
    # Token-basierter F1
    gt_tokens = set(gt_normalized.split())
    pred_tokens = set(pred_normalized.split())
    
    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0
    
    common = gt_tokens.intersection(pred_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1

def process_ner_file(file_path: str, relative_path: str) -> Optional[Dict[str, Any]]:
    try:
        df = pd.read_csv(file_path)
        
        if 'ground_truth_entities' in df.columns and 'predicted_entities' in df.columns:
            f1_scores = []
            for _, row in df.iterrows():
                gt_entities = parse_entities_correctly(str(row['ground_truth_entities']))
                pred_entities = parse_entities_correctly(str(row['predicted_entities']))
                
                f1 = calculate_ner_f1_score(gt_entities, pred_entities)
                f1_scores.append(f1)
            
            avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
            return {
                'f1_score': avg_f1,
                'sample_count': len(df)
            }
        else:
            logging.warning(f"NER file {file_path} doesn't have expected columns")
            return None
            
    except Exception as e:
        logging.error(f"Error processing NER file {file_path}: {e}")
        return None

def calculate_ner_f1_score(gt_entities: List[Tuple[str, str]], pred_entities: List[Tuple[str, str]]) -> float:
    """Calculate F1 score for NER entities."""
    gt_set = set(gt_entities)
    pred_set = set(pred_entities)
    
    # Wenn beide leer sind, perfekte Übereinstimmung
    if not gt_set and not pred_set:
        return 1.0
    
    # Wenn eine leer ist, die andere aber nicht, keine Übereinstimmung
    if not gt_set or not pred_set:
        return 0.0
    
    common = gt_set.intersection(pred_set)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gt_set)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1

def process_classification_file(file_path: str, relative_path: str) -> Optional[Dict[str, Any]]:
    try:
        df = pd.read_csv(file_path)
        
        # Suche nach Ground-Truth- und Vorhersagespalten
        ground_truth_col = None
        prediction_col = None
        
        # Klassifizierungs-Baseline verwendet diese Spalten
        for col in ['ground_truth', 'true_label', 'label', 'actual', 'target', 'ground_truth_label']:
            if col in df.columns:
                ground_truth_col = col
                break
        
        # Klassifizierung single_prompt CoTR verwendet andere Spaltennamen
        if not ground_truth_col:
            for col in ['label_lrl_ground_truth']:
                if col in df.columns:
                    ground_truth_col = col
                    break
        
        # Klassifizierung multi_prompt CoTR verwendet andere Spaltennamen
        if not ground_truth_col:
            for col in ['ground_truth_label_eng']:
                if col in df.columns:
                    ground_truth_col = col
                    break
        
        for col in ['predicted_label', 'prediction', 'pred', 'predicted', 'final_predicted_label']:
            if col in df.columns:
                prediction_col = col
                break
        
        # Klassifizierung single_prompt CoTR verwendet andere Spaltennamen
        if not prediction_col:
            for col in ['label_lrl_predicted_final']:
                if col in df.columns:
                    prediction_col = col
                    break
        
        # Klassifizierung multi_prompt CoTR verwendet andere Spaltennamen
        if not prediction_col:
            for col in ['predicted_label_eng_model']:
                if col in df.columns:
                    prediction_col = col
                    break
        
        if ground_truth_col and prediction_col:
            ground_truths = []
            predictions = []
            
            for _, row in df.iterrows():
                gt = str(row[ground_truth_col]).lower().strip() if not pd.isna(row[ground_truth_col]) else ""
                pred = str(row[prediction_col]).lower().strip() if not pd.isna(row[prediction_col]) else ""
                
                # Versuche, die Klassifizierung aus der Roh-Ausgabe zu extrahieren, wenn die Vorhersage leer oder unbekannt ist
                if not pred or pred in ['[unknown label]', 'unknown', '']:
                    if 'raw_model_output' in df.columns:
                        raw_output = str(row['raw_model_output']) if not pd.isna(row['raw_model_output']) else ""
                        pred = extract_classification_label(raw_output)
                    elif 'raw_classification_output' in df.columns:
                        raw_output = str(row['raw_classification_output']) if not pd.isna(row['raw_classification_output']) else ""
                        pred = extract_classification_label(raw_output)
                
                if gt:  # Nur nicht-leere Ground-Truths einschließen
                    ground_truths.append(gt)
                    predictions.append(pred)
            
            if ground_truths and len(ground_truths) == len(predictions):
                accuracy = accuracy_score(ground_truths, predictions)
                f1 = f1_score(ground_truths, predictions, average='weighted', zero_division=0)
                
                return {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'sample_count': len(ground_truths)
                }
        else:
            logging.warning(f"Classification file {file_path} doesn't have expected columns")
            return None
            
    except Exception as e:
        logging.error(f"Error processing Classification file {file_path}: {e}")
        return None

def extract_classification_label(raw_output: str) -> str:
    if not raw_output or pd.isna(raw_output):
        return "unknown"
    
    raw_lower = str(raw_output).lower()
    
    # Gängige Klassifizierungslabels
    common_labels = [
        'politics', 'business', 'technology', 'entertainment', 'sports', 'health',
        'science', 'education', 'travel', 'lifestyle', 'finance', 'economy',
        'society', 'culture', 'environment', 'international', 'national', 'local'
    ]
    
    # Suche nach expliziten Vorhersagen
    for label in common_labels:
        if f'prediction: {label}' in raw_lower or f'answer: {label}' in raw_lower:
            return label
        if f'category: {label}' in raw_lower or f'class: {label}' in raw_lower:
            return label
    
    # Suche nach Labels am Ende des Textes
    words = raw_lower.split()
    if words:
        last_word = words[-1].strip('.,!?:')
        if last_word in common_labels:
            return last_word
    
    # Suche nach jedem Vorkommen von gängigen Labels
    for label in common_labels:
        if label in raw_lower:
            return label
    
    return "unknown"

def process_nli_file(file_path: str, relative_path: str) -> Optional[Dict[str, Any]]:
    try:
        df = pd.read_csv(file_path)
        
        # Suche nach Ground-Truth- und Vorhersagespalten für NLI
        ground_truth_col = None
        prediction_col = None
        
        # NLI-Baseline-Dateien verwenden diese Spalten
        for col in ['ground_truth', 'true_label', 'label', 'actual', 'target', 'gold_label', 'ground_truth_label']:
            if col in df.columns:
                ground_truth_col = col
                break
        
        # NLI-CoTR-Dateien verwenden andere Spaltennamen
        if not ground_truth_col:
            for col in ['original_gt_label_int']:
                if col in df.columns:
                    ground_truth_col = col
                    break
        
        for col in ['predicted_label', 'prediction', 'pred', 'predicted']:
            if col in df.columns:
                prediction_col = col
                break
        
        # NLI-CoTR-Dateien verwenden andere Spaltennamen
        if not prediction_col:
            for col in ['predicted_label_for_accuracy']:
                if col in df.columns:
                    prediction_col = col
                    break
        
        if ground_truth_col and prediction_col:
            ground_truths = []
            predictions = []
            
            for _, row in df.iterrows():
                gt = str(row[ground_truth_col]).lower().strip() if not pd.isna(row[ground_truth_col]) else ""
                pred = str(row[prediction_col]).lower().strip() if not pd.isna(row[prediction_col]) else ""
                
                # Konvertiere Integer-Labels in Text für CoTR-Dateien
                if ground_truth_col == 'original_gt_label_int':
                    gt_int = int(float(gt)) if gt.replace('.','').isdigit() else -1
                    if gt_int == 0:
                        gt = "entailment"
                    elif gt_int == 1:
                        gt = "neutral"
                    elif gt_int == 2:
                        gt = "contradiction"
                    else:
                        gt = ""
                
                if gt:  # Nur nicht-leere Ground-Truths einschließen
                    ground_truths.append(gt)
                    predictions.append(pred)
            
            if ground_truths and len(ground_truths) == len(predictions):
                accuracy = accuracy_score(ground_truths, predictions)
                f1 = f1_score(ground_truths, predictions, average='weighted', zero_division=0)
                
                return {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'sample_count': len(ground_truths)
                }
        else:
            logging.warning(f"NLI file {file_path} doesn't have expected columns")
            return None
            
    except Exception as e:
        logging.error(f"Error processing NLI file {file_path}: {e}")
        return None

def process_sentiment_file(file_path: str, relative_path: str) -> Optional[Dict[str, Any]]:
    try:
        df = pd.read_csv(file_path)
        
        # Suche nach Ground-Truth- und Vorhersagespalten für Sentiment
        ground_truth_col = None
        prediction_col = None
        
        for col in ['ground_truth', 'true_label', 'label', 'actual', 'target', 'sentiment', 'ground_truth_label']:
            if col in df.columns:
                ground_truth_col = col
                break
        
        for col in ['predicted_label', 'prediction', 'pred', 'predicted', 'predicted_sentiment']:
            if col in df.columns:
                prediction_col = col
                break
        
        if ground_truth_col and prediction_col:
            ground_truths = []
            predictions = []
            
            for _, row in df.iterrows():
                gt = str(row[ground_truth_col]).lower().strip() if not pd.isna(row[ground_truth_col]) else ""
                pred = str(row[prediction_col]).lower().strip() if not pd.isna(row[prediction_col]) else ""
                
                if gt:  # Nur nicht-leere Ground-Truths einschließen
                    ground_truths.append(gt)
                    predictions.append(pred)
            
            if ground_truths and len(ground_truths) == len(predictions):
                accuracy = accuracy_score(ground_truths, predictions)
                f1 = f1_score(ground_truths, predictions, average='weighted', zero_division=0)
                
                return {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'sample_count': len(ground_truths)
                }
        else:
            logging.warning(f"Sentiment file {file_path} doesn't have expected columns")
            return None
            
    except Exception as e:
        logging.error(f"Error processing Sentiment file {file_path}: {e}")
        return None

def process_results_files(base_results_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    all_files = find_all_result_files(base_results_dir)
    logging.info(f"Found {len(all_files)} CSV files total")
    
    task_results = defaultdict(list)
    task_stats = defaultdict(lambda: {'total_files': 0, 'parsed_files': 0, 'failed_files': 0})
    
    for file_path in all_files:
        try:
            task, approach, language, model, shot_type = extract_file_info(file_path)
            
            if task == 'unknown':
                task_stats[task]['failed_files'] += 1
                continue
                
            task_stats[task]['total_files'] += 1
            
            full_file_path = os.path.join(base_results_dir, file_path)
            
            if task == 'qa':
                metrics = process_qa_file(full_file_path, file_path)
                if metrics:
                    task_stats[task]['parsed_files'] += 1
                    logging.info(f"QA - {file_path}: F1={metrics['f1_score']:.4f}")
                    task_results[task].append({
                        'file_path': file_path,
                        'approach': approach,
                        'language': language,
                        'model': model,
                        'shot_type': shot_type,
                        **metrics
                    })
                else:
                    task_stats[task]['failed_files'] += 1
                    
            elif task == 'ner':
                metrics = process_ner_file(full_file_path, file_path)
                if metrics:
                    task_stats[task]['parsed_files'] += 1
                    logging.info(f"NER - {file_path}: F1={metrics['f1_score']:.4f}")
                    task_results[task].append({
                        'file_path': file_path,
                        'approach': approach,
                        'language': language,
                        'model': model,
                        'shot_type': shot_type,
                        **metrics
                    })
                else:
                    task_stats[task]['failed_files'] += 1
                    
            elif task == 'classification':
                metrics = process_classification_file(full_file_path, file_path)
                if metrics:
                    task_stats[task]['parsed_files'] += 1
                    logging.info(f"Classification - {file_path}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
                    task_results[task].append({
                        'file_path': file_path,
                        'approach': approach,
                        'language': language,
                        'model': model,
                        'shot_type': shot_type,
                        **metrics
                    })
                else:
                    task_stats[task]['failed_files'] += 1
                    
            elif task == 'nli':
                metrics = process_nli_file(full_file_path, file_path)
                if metrics:
                    task_stats[task]['parsed_files'] += 1
                    logging.info(f"NLI - {file_path}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
                    task_results[task].append({
                        'file_path': file_path,
                        'approach': approach,
                        'language': language,
                        'model': model,
                        'shot_type': shot_type,
                        **metrics
                    })
                else:
                    task_stats[task]['failed_files'] += 1
                    
            elif task == 'sentiment':
                metrics = process_sentiment_file(full_file_path, file_path)
                if metrics:
                    task_stats[task]['parsed_files'] += 1
                    logging.info(f"Sentiment - {file_path}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
                    task_results[task].append({
                        'file_path': file_path,
                        'approach': approach,
                        'language': language,
                        'model': model,
                        'shot_type': shot_type,
                        **metrics
                    })
                else:
                    task_stats[task]['failed_files'] += 1
                    
        except Exception as e:
            task = extract_file_info(file_path)[0] if extract_file_info(file_path)[0] != 'unknown' else 'unknown'
            task_stats[task]['failed_files'] += 1
            logging.error(f"Error processing {file_path}: {e}")
    
    return dict(task_results), dict(task_stats)

def generate_summary_report(metrics: Dict[str, Any], task_stats: Dict[str, Any], output_dir: Path) -> None:
    summary_file = output_dir / "comprehensive_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("=== COMPREHENSIVE METRICS REPORT ===\n")
        f.write("Generated from corrected analysis\n\n")
        
        # Gesamtstatistik
        f.write("=== OVERALL STATISTICS ===\n\n")
        total_files = sum(stats['total_files'] for stats in task_stats.values() if 'total_files' in stats)
        total_parsed = sum(stats['parsed_files'] for stats in task_stats.values() if 'parsed_files' in stats)
        total_failed = sum(stats['failed_files'] for stats in task_stats.values() if 'failed_files' in stats)
        overall_success_rate = (total_parsed / total_files * 100) if total_files > 0 else 0
        
        f.write(f" **TOTAL FILES FOUND**: {total_files}\n")
        f.write(f" **SUCCESSFULLY PARSED**: {total_parsed}\n")
        f.write(f" **FAILED TO PARSE**: {total_failed}\n")
        f.write(f" **OVERALL SUCCESS RATE**: {overall_success_rate:.1f}%\n\n")
        
        f.write("=== SUMMARY BY TASK ===\n\n")
        
        # Aufgaben-spezifische Zusammenfassungen
        for task, results in metrics.items():
            if not results:
                continue
                
            stats = task_stats.get(task, {})
            success_rate = (stats.get('parsed_files', 0) / stats.get('total_files', 1) * 100) if stats.get('total_files', 0) > 0 else 0
            
            f.write(f" **{task.upper()} TASK**\n")
            f.write(f"  Configurations processed: {len(results)}\n")
            f.write(f"  Files found: {stats.get('total_files', 0)}\n")
            f.write(f"  Success rate: {success_rate:.1f}%\n")
            
            if task == 'qa':
                f1_scores = [r['f1_score'] for r in results if 'f1_score' in r]
                if f1_scores:
                    avg_f1 = np.mean(f1_scores)
                    best_result = max(results, key=lambda x: x.get('f1_score', 0))
                    f.write(f"  Average F1: {avg_f1:.4f}\n")
                    f.write(f"  Best F1: {best_result['f1_score']:.4f} ({best_result['file_path']})\n")
                    
            elif task in ['ner']:
                f1_scores = [r['f1_score'] for r in results if 'f1_score' in r]
                if f1_scores:
                    avg_f1 = np.mean(f1_scores)
                    best_result = max(results, key=lambda x: x.get('f1_score', 0))
                    f.write(f"  Average F1: {avg_f1:.4f}\n")
                    f.write(f"  Best F1: {best_result['f1_score']:.4f} ({best_result['file_path']})\n")
                    
            elif task in ['classification', 'nli', 'sentiment']:
                accuracies = [r['accuracy'] for r in results if 'accuracy' in r]
                f1_scores = [r['f1_score'] for r in results if 'f1_score' in r]
                if accuracies and f1_scores:
                    avg_acc = np.mean(accuracies)
                    avg_f1 = np.mean(f1_scores)
                    best_acc_result = max(results, key=lambda x: x.get('accuracy', 0))
                    best_f1_result = max(results, key=lambda x: x.get('f1_score', 0))
                    f.write(f"  Average Accuracy: {avg_acc:.4f}\n")
                    f.write(f"  Average F1: {avg_f1:.4f}\n")
                    f.write(f"  Best Accuracy: {best_acc_result['accuracy']:.4f} ({best_acc_result['file_path']})\n")
                    f.write(f"  Best F1: {best_f1_result['f1_score']:.4f} ({best_f1_result['file_path']})\n")
            
            f.write("\n")
        
        # Detaillierte Aufschlüsselung nach Konfiguration
        f.write("=== DETAILED BREAKDOWN ===\n\n")
        for task, results in metrics.items():
            if not results:
                continue
                
            f.write(f"## {task.upper()} DETAILED RESULTS\n")
            for result in sorted(results, key=lambda x: x.get('f1_score', x.get('accuracy', 0)), reverse=True):
                f.write(f"  {result['file_path']}\n")
                f.write(f"     Approach: {result['approach']}, Language: {result['language']}, Model: {result['model']}, Shot: {result['shot_type']}\n")
                if 'f1_score' in result:
                    f.write(f"     F1: {result['f1_score']:.4f}")
                if 'accuracy' in result:
                    f.write(f"     Accuracy: {result['accuracy']:.4f}")
                f.write(f", Samples: {result.get('sample_count', 'N/A')}\n\n")
            f.write("\n")
        
        # Parsing-Fehler hinzufügen, falls vorhanden
        if any(stats.get('failed_files', 0) > 0 for stats in task_stats.values()):
            f.write("=== PARSING FAILURES ===\n\n")
            for task, stats in task_stats.items():
                if stats.get('failed_files', 0) > 0:
                    f.write(f"**{task.upper()}**: {stats['failed_files']} files failed to parse\n")
    
    logging.info(f"Summary report written to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Calculate comprehensive metrics")
    parser.add_argument("--base_results_dir", required=True, help="Base directory containing results")
    parser.add_argument("--output_dir", required=True, help="Output directory for metrics")
    
    args = parser.parse_args()
    
    # Ausgabeverzeichnis erstellen
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Alle Ergebnisse verarbeiten
    logger.info("Starting comprehensive metrics calculation...")
    metrics, task_stats = process_results_files(args.base_results_dir)
    
    # Detaillierte Metriken speichern
    detailed_file = output_dir / "comprehensive_metrics.json"
    with open(detailed_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Zusammenfassungsbericht erstellen
    generate_summary_report(metrics, task_stats, output_dir)
    
    logger.info("Comprehensive metrics calculation completed successfully!")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 