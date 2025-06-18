from typing import List, Optional, Dict, Any

try:
    import evaluate
    import torch
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

def calculate_comet_score(
    sources: List[str],
    predictions: List[str],
    references: List[str],
    model_name: str = "Unbabel/wmt22-comet-da"
) -> Optional[Dict[str, Any]]:
    """
    Calculates the COMET score for a batch of translations.

    Args:
        sources (List[str]): A list of source sentences.
        predictions (List[str]): A list of predicted (translated) sentences.
        references (List[str]): A list of reference sentences.
        model_name (str): The name of the COMET model to use.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the mean score and a list of individual scores,
                                  or None if COMET is not available.
    """
    if not COMET_AVAILABLE:
        print("COMET metric is not available. Please install it with: pip install evaluate unbabel-comet")
        return None

    try:
        comet_metric = evaluate.load('comet', config_name=model_name)
        results = comet_metric.compute(
            predictions=predictions,
            references=references,
            sources=sources,
            gpus=1 if torch.cuda.is_available() else 0,
            progress_bar=True
        )
        return results
    except Exception as e:
        print(f"Error calculating COMET score: {e}")
        return None 