import re
from typing import Dict, Any

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
    # Use the ground_truth column directly (should be a string now)
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