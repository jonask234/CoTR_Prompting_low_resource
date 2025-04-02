import re
from typing import Dict, Any
import pandas as pd

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

def calculate_translation_quality(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate translation quality metrics for the CoTR approach.
    
    Args:
        row: Dictionary containing original and translated text
        
    Returns:
        Dictionary containing translation quality metrics
    """
    # Normalize all texts
    orig_question = normalize_answer(row['question'])
    orig_answer = normalize_answer(row['predicted_answer'])
    
    # Calculate token overlap ratios
    def calculate_overlap(orig: str, trans: str) -> float:
        orig_tokens = set(orig.split())
        trans_tokens = set(trans.split())
        if not orig_tokens or not trans_tokens:
            return 0.0
        common = orig_tokens.intersection(trans_tokens)
        return len(common) / max(len(orig_tokens), len(trans_tokens))
    
    # Calculate metrics
    question_overlap = calculate_overlap(orig_question, row['question_en'])
    answer_overlap = calculate_overlap(orig_answer, row['answer_en'])
    
    return {
        'question_translation_quality': question_overlap,
        'answer_translation_quality': answer_overlap,
        'average_translation_quality': (question_overlap + answer_overlap) / 2
    }

def evaluate_cotr_results(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate CoTR results focusing on F1 scores.
    
    Args:
        results_df: DataFrame containing CoTR results
        
    Returns:
        Dictionary containing F1 score metrics
    """
    # Calculate F1 scores
    results_df['f1_score'] = results_df.apply(calculate_qa_f1, axis=1)
    
    # Calculate average metrics
    metrics = {
        'f1_score': results_df['f1_score'].mean()
    }
    
    return metrics 