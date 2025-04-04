import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from comet import download_model, load_from_checkpoint
import torch

# Initialize COMET model
try:
    # Try to load the COMET model
    COMET_MODEL_PATH = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(COMET_MODEL_PATH)
    print("COMET model loaded successfully!")
    COMET_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load COMET model: {str(e)}")
    print("Falling back to simpler metrics for translation quality.")
    COMET_AVAILABLE = False

def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s: str) -> set:
    """Get set of tokens from string."""
    return set(normalize_answer(s).split())

def calculate_qa_f1(row: Dict[str, Any]) -> float:
    """
    Calculate F1 score for a QA pair.
    
    Args:
        row: Dictionary containing 'predicted_answer' and 'ground_truth' keys
        
    Returns:
        float: F1 score between 0 and 1
    """
    pred_tokens = get_tokens(row['predicted_answer'])
    truth_tokens = get_tokens(row['ground_truth'])
    
    if not pred_tokens or not truth_tokens:
        return 0.0
        
    common = pred_tokens.intersection(truth_tokens)
    
    if not common:
        return 0.0
        
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)

def calculate_qa_f1_english(row: Dict[str, Any]) -> float:
    """
    Calculate F1 score for the English QA pair (intermediate step in CoTR).
    
    Args:
        row: Dictionary containing 'answer_en' and optional 'ground_truth_en' keys
        
    Returns:
        float: F1 score between 0 and 1, or None if ground_truth_en is not available
    """
    if 'ground_truth_en' not in row:
        return None
        
    pred_tokens = get_tokens(row['answer_en'])
    truth_tokens = get_tokens(row['ground_truth_en'])
    
    if not pred_tokens or not truth_tokens:
        return 0.0
        
    common = pred_tokens.intersection(truth_tokens)
    
    if not common:
        return 0.0
        
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)

def calculate_token_overlap(original_text: str, translated_text: str) -> float:
    """
    Calculate token overlap between two texts.
    Used as a fallback when COMET is not available.
    """
    orig_tokens = set(normalize_answer(original_text).split())
    trans_tokens = set(normalize_answer(translated_text).split())
    
    if not orig_tokens or not trans_tokens:
        return 0.0
        
    common = orig_tokens.intersection(trans_tokens)
    # Handle division by zero if both sets are empty
    max_len = max(len(orig_tokens), len(trans_tokens))
    return len(common) / max_len if max_len > 0 else 0.0 

def calculate_comet_score(source_texts: List[str], hypothesis_texts: List[str], reference_texts: List[str]=None) -> List[float]:
    """
    Calculate COMET scores for translations.
    
    Args:
        source_texts: List of source texts
        hypothesis_texts: List of translated texts
        reference_texts: List of reference translations (optional)
        
    Returns:
        List of COMET scores (one per input pair)
    """
    if not COMET_AVAILABLE:
        # Fallback if COMET is not available
        print("  COMET not available, returning 0.0 scores.")
        return [0.0] * len(source_texts)
        
    # Ensure inputs are lists of strings
    source_texts = [str(s) for s in source_texts]
    hypothesis_texts = [str(h) for h in hypothesis_texts]
    if reference_texts is not None:
        reference_texts = [str(r) for r in reference_texts]
        
    # Prepare data for COMET
    data = []
    for i in range(len(source_texts)):
        item = {
            "src": source_texts[i],
            "mt": hypothesis_texts[i] # 'mt' is the hypothesis (machine translation)
        }
        if reference_texts is not None:
            # Only add 'ref' key if references are actually provided
            item["ref"] = reference_texts[i] 
        else:
            # Add a dummy ref if none is provided, as the model seems to expect it
            item["ref"] = ""
        data.append(item)
    
    try:
        # Get COMET scores
        # Use gpus=1 if available, otherwise it will use CPU
        # Handle potential errors during prediction
        model_output = comet_model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0) 
        
        # Always extract segment-level scores from the 'scores' attribute
        scores = model_output.get('scores', [0.0] * len(source_texts)) # Default to 0.0 if 'scores' not found
        
        # Ensure scores is a list
        if not isinstance(scores, list):
            print(f"  Warning: Unexpected COMET output format. Scores: {scores}")
            return [0.0] * len(source_texts)
            
        # Ensure the number of scores matches the number of inputs
        if len(scores) != len(source_texts):
             print(f"  Warning: Mismatch between number of COMET scores ({len(scores)}) and inputs ({len(source_texts)}).")
             # Try to return 0.0 for safety, or handle based on expected behavior
             return [0.0] * len(source_texts)
             
        return scores
        
    except KeyError as e:
        # Specific handling if the KeyError for 'ref' persists
        print(f"  KeyError during COMET prediction (likely related to reference handling): {e}")
        print("  Falling back to 0.0 scores for this batch.")
        return [0.0] * len(source_texts)
    except Exception as e:
        # Catch other potential errors during prediction
        print(f"  Error during COMET prediction: {e}")
        return [0.0] * len(source_texts)

def calculate_translation_quality(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate translation quality metrics for the CoTR approach.
    
    Args:
        row: Dictionary containing original and translated text
        
    Returns:
        Dictionary containing translation quality metrics
    """
    results = {
        'question_translation_quality': 0.0,
        'answer_translation_quality': 0.0,
        'average_translation_quality': 0.0,
        'comet_source_to_en': float('nan'), # Use NaN as default for raw scores
        'comet_en_to_source': float('nan')
    }
    
    if COMET_AVAILABLE:
        # Use COMET for translation quality
        try:
            # Source (original) to English translation quality (reference-less)
            comet_src_en_scores = calculate_comet_score(
                [row['question']], 
                [row['question_en']]
            )
            # English to source (original) translation quality (reference-less)
            comet_en_src_scores = calculate_comet_score(
                [row['answer_en']], 
                [row['predicted_answer']]
            )
            
            # Check if scores were returned successfully (list with one element)
            if comet_src_en_scores and isinstance(comet_src_en_scores, list):
                results['comet_source_to_en'] = comet_src_en_scores[0]
                results['question_translation_quality'] = max(0, (results['comet_source_to_en'] + 1) / 2) # Normalize COMET score to 0-1
            else:
                 print(f"  Warning: Failed to get valid COMET score for question translation (ID: {row.get('id')}).")

            if comet_en_src_scores and isinstance(comet_en_src_scores, list):
                results['comet_en_to_source'] = comet_en_src_scores[0]
                results['answer_translation_quality'] = max(0, (results['comet_en_to_source'] + 1) / 2) # Normalize COMET score to 0-1
            else:
                 print(f"  Warning: Failed to get valid COMET score for answer translation (ID: {row.get('id')}).")
                 
            # Calculate average only if both scores are valid numbers
            if not pd.isna(results['comet_source_to_en']) and not pd.isna(results['comet_en_to_source']):
                results['average_translation_quality'] = (results['question_translation_quality'] + results['answer_translation_quality']) / 2
            else:
                results['average_translation_quality'] = 0.0 # Fallback average

        except Exception as e:
            print(f"  Warning: COMET evaluation failed for row ID {row.get('id')}: {str(e)}")
            print("  Falling back to token overlap metrics for this row.")
            # Fall back to token overlap only if COMET calculation itself failed
            results['question_translation_quality'] = calculate_token_overlap(row['question'], row['question_en'])
            results['answer_translation_quality'] = calculate_token_overlap(row['predicted_answer'], row['answer_en'])
            results['average_translation_quality'] = (results['question_translation_quality'] + results['answer_translation_quality']) / 2
            results['comet_source_to_en'] = float('nan')
            results['comet_en_to_source'] = float('nan')
            
    else:
        # Fallback to token overlap method if COMET is unavailable globally
        results['question_translation_quality'] = calculate_token_overlap(row['question'], row['question_en'])
        results['answer_translation_quality'] = calculate_token_overlap(row['predicted_answer'], row['answer_en'])
        results['average_translation_quality'] = (results['question_translation_quality'] + results['answer_translation_quality']) / 2
        
    return results

def evaluate_cotr_results(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate CoTR results focusing on F1 scores and translation quality.
    
    Args:
        results_df: DataFrame containing CoTR results
        
    Returns:
        Dictionary containing F1 score and translation quality metrics
    """
    # Calculate F1 scores
    results_df['f1_score'] = results_df.apply(calculate_qa_f1, axis=1)
    
    # Calculate translation quality metrics
    translation_metrics_list = results_df.apply(calculate_translation_quality, axis=1).tolist()
    
    # Populate DataFrame columns from the list of metric dictionaries
    results_df['question_translation_quality'] = [m.get('question_translation_quality', 0.0) for m in translation_metrics_list]
    results_df['answer_translation_quality'] = [m.get('answer_translation_quality', 0.0) for m in translation_metrics_list]
    results_df['average_translation_quality'] = [m.get('average_translation_quality', 0.0) for m in translation_metrics_list]
    
    # Check if COMET scores were successfully calculated for at least one row
    has_comet_scores = any(not pd.isna(m.get('comet_source_to_en')) for m in translation_metrics_list)
    
    if has_comet_scores:
        results_df['comet_source_to_en'] = [m.get('comet_source_to_en', float('nan')) for m in translation_metrics_list]
        results_df['comet_en_to_source'] = [m.get('comet_en_to_source', float('nan')) for m in translation_metrics_list]
    
    # Calculate average metrics, ignoring NaNs for COMET scores
    metrics = {
        'f1_score': results_df['f1_score'].mean(),
        'question_translation_quality': results_df['question_translation_quality'].mean(),
        'answer_translation_quality': results_df['answer_translation_quality'].mean(),
        'average_translation_quality': results_df['average_translation_quality'].mean()
    }
    
    if has_comet_scores:
        # Use nanmean to calculate average ignoring potential NaNs if some rows failed COMET
        metrics['comet_source_to_en'] = np.nanmean(results_df['comet_source_to_en'].astype(float))
        metrics['comet_en_to_source'] = np.nanmean(results_df['comet_en_to_source'].astype(float))
        
        # Add a note if fallback metrics were used for some rows
        if any(pd.isna(m.get('comet_source_to_en')) for m in translation_metrics_list):
             print("  Note: Some COMET scores failed; averages calculated excluding failed rows.")

    elif not COMET_AVAILABLE:
         print("  Note: COMET model not loaded; using token overlap for translation quality.")
    
    return metrics 