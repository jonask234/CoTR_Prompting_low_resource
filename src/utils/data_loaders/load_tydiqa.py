import pandas as pd
from datasets import load_dataset
from typing import Optional
import os
import json
from huggingface_hub import hf_hub_download
import tempfile

def load_tydiqa_local(lang_code: str, split: str = 'validation', num_samples: Optional[int] = None):
    """
    Load TyDiQA from a local JSON file.
    
    Args:
        lang_code: Language code (e.g., 'sw' for Swahili).
        split: Split to load - maps 'validation' to 'dev' for TyDiQA's naming convention.
        num_samples: Number of samples to load, or None for all.
        
    Returns:
        DataFrame containing loaded samples, or empty DataFrame if loading fails.
    """
    # Map validation to dev for TyDiQA naming convention
    file_split = "dev" if split == "validation" else split
    
    # Possible local file paths
    local_paths = [
        f"/work/bbd6522/cache_dir/tydiqa/tydiqa-goldp-v1.1-{file_split}.json",
        os.path.expanduser(f"~/tydiqa-goldp-v1.1-{file_split}.json"),
        os.path.join(os.getcwd(), f"tydiqa-goldp-v1.1-{file_split}.json")
    ]
    
    # Try each possible path
    local_file = None
    for path in local_paths:
        if os.path.exists(path):
            local_file = path
            break
    
    if not local_file:
        print(f"No local TyDiQA file found. Tried paths: {local_paths}")
        return pd.DataFrame()
    
    try:
        print(f"Loading TyDiQA from local file: {local_file}")
        with open(local_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"JSON file loaded. Looking for {lang_code} examples...")
        
        # Map language codes to expected prefixes or strings in the JSON
        lang_id_map = {
            'ar': 'arabic',
            'bn': 'bengali',
            'en': 'english',
            'fi': 'finnish',
            'id': 'indonesian',
            'ko': 'korean',
            'ru': 'russian',
            'sw': 'swahili',
            'te': 'telugu',
        }
        
        # Get the language string to look for
        lang_str = lang_id_map.get(lang_code, lang_code)
        
        # Extract samples for the specified language
        all_samples = []
        
        # TyDiQA-GoldP has a specific structure - either 'data' list or direct 'version' + other fields
        if 'data' in data:
            # Standard SQuAD-like format
            for item in data['data']:
                if isinstance(item, dict) and 'paragraphs' in item:
                    for para in item['paragraphs']:
                        context = para.get('context', '')
                        qas = para.get('qas', [])
                        
                        for qa in qas:
                            # Check if this question belongs to our language
                            # Most reliable way is to check the ID prefix which contains language
                            qa_id = qa.get('id', '')
                            if qa_id.startswith(lang_str):
                                sample = {
                                    'id': qa_id,
                                    'context': context,
                                    'question': qa.get('question', ''),
                                    'answers': {
                                        'text': [a.get('text', '') for a in qa.get('answers', [])],
                                        'answer_start': [a.get('answer_start', 0) for a in qa.get('answers', [])]
                                    },
                                    'language': lang_code
                                }
                                all_samples.append(sample)
        elif 'version' in data:
            # Look at debug info to understand the structure
            print(f"Found version field: {data.get('version')}")
            keys = list(data.keys())
            print(f"Top-level keys: {keys}")
            
            # It may have articles/paragraphs/questions in a different structure
            # For now, attempt to extract from a structure similar to SQuAD
            for key in keys:
                if isinstance(data[key], list) and len(data[key]) > 0:
                    for item in data[key]:
                        if isinstance(item, dict):
                            # Look for text content and questions
                            if 'context' in item and 'qas' in item:
                                context = item.get('context', '')
                                qas = item.get('qas', [])
                            elif 'paragraphs' in item:
                                for para in item['paragraphs']:
                                    context = para.get('context', '')
                                    qas = para.get('qas', [])
                            else:
                                continue
                            
                            for qa in qas:
                                qa_id = qa.get('id', '')
                                if lang_str in qa_id:
                                    sample = {
                                        'id': qa_id,
                                        'context': context,
                                        'question': qa.get('question', ''),
                                        'answers': {
                                            'text': [a.get('text', '') for a in qa.get('answers', [])],
                                            'answer_start': [a.get('answer_start', 0) for a in qa.get('answers', [])]
                                        },
                                        'language': lang_code
                                    }
                                    all_samples.append(sample)
        else:
            # Directly search for any structure with question/answer pairs
            print("Non-standard format. Searching for question/answer pairs...")
            
            def extract_qa_pairs(obj, context="", path=""):
                """Recursively search for question/answer pairs in JSON structure"""
                if isinstance(obj, dict):
                    # Check if this looks like a QA item
                    if 'question' in obj and 'answers' in obj and 'id' in obj:
                        qa_id = obj.get('id', '')
                        if lang_str in qa_id:
                            # Found a QA pair for our language
                            return [{
                                'id': qa_id,
                                'context': context,  # Use parent context if available
                                'question': obj.get('question', ''),
                                'answers': {
                                    'text': [a.get('text', '') for a in obj.get('answers', [])],
                                    'answer_start': [a.get('answer_start', 0) for a in obj.get('answers', [])]
                                },
                                'language': lang_code
                            }]
                    
                    # If this dict has a context, update it for children
                    if 'context' in obj:
                        context = obj['context']
                    
                    # Recursively search in dict values
                    results = []
                    for k, v in obj.items():
                        results.extend(extract_qa_pairs(v, context, f"{path}.{k}"))
                    return results
                
                elif isinstance(obj, list):
                    # Recursively search in list items
                    results = []
                    for i, item in enumerate(obj):
                        results.extend(extract_qa_pairs(item, context, f"{path}[{i}]"))
                    return results
                
                return []  # Base case: not a dict or list
            
            # Start recursive search
            all_samples = extract_qa_pairs(data)
            
        if not all_samples:
            print(f"No samples found for language '{lang_code}' in local file.")
            return pd.DataFrame()
                
        # Convert to DataFrame
        samples_df = pd.DataFrame(all_samples)
        
        # Sample if requested
        if num_samples is not None and num_samples < len(samples_df):
            samples_df = samples_df.sample(n=num_samples, random_state=42)
        
        # Extract the first answer text as ground_truth
        samples_df['ground_truth'] = samples_df['answers'].apply(
            lambda ans: ans['text'][0] if ans['text'] else None
        )
        
        # Drop rows with missing ground truth
        samples_df.dropna(subset=['ground_truth'], inplace=True)
        
        print(f"Successfully loaded {len(samples_df)} samples from local file for {lang_code}.")
        return samples_df
    except Exception as e:
        print(f"Error loading from local file: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def load_tydiqa_samples(
    lang_code: str, 
    num_samples: Optional[int] = None, 
    split: str = 'validation'
) -> pd.DataFrame:
    """
    Loads question answering samples from the TyDiQA-GoldP dataset for a specific language.
    This loader specifically preserves the original LRL ground truth answers.
    
    Args:
        lang_code: Language code (e.g., 'sw' for Swahili).
        num_samples: Number of samples to load. Loads all if None.
        split: Dataset split to use ('train', 'validation'). Default is 'validation'.
    
    Returns:
        DataFrame with 'id', 'context', 'question', 'answers', 'ground_truth' columns.
        'ground_truth' contains the LRL answer string from the original dataset.
    """
    # First try loading from local file
    local_samples = load_tydiqa_local(lang_code, split, num_samples)
    if not local_samples.empty:
        return local_samples
        
    # Check for cached JSON
    cache_dir = "/work/bbd6522/cache_dir"
    local_file_path = os.path.join(cache_dir, f"tydiqa_{lang_code}_{split}.json")
    
    if os.path.exists(local_file_path):
        print(f"Loading TyDiQA from local cache: {local_file_path}")
        try:
            with open(local_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            samples_df = pd.DataFrame(data)
            print(f"Successfully loaded {len(samples_df)} samples from local cache for {lang_code}.")
            return samples_df
        except Exception as e:
            print(f"Error loading from local cache: {e}. Trying online sources...")
    
    # Try loading from Hugging Face Hub
    dataset = None
    last_error = None
    
    # Try several different dataset names that might work
    dataset_names = [
        "tydiqa",
        "google/tydiqa", 
        "allenai/tydiqa"
    ]
    config_name = "goldp"  # Use the "GoldP" (gold passage) configuration
    
    for dataset_name in dataset_names:
        try:
            print(f"Trying to load {dataset_name} dataset for language: {lang_code}, split: {split}...")
            # First try with the specific config
            dataset = load_dataset(dataset_name, config_name, cache_dir=cache_dir)
            
            # Extract the specified split
            if split in dataset:
                dataset = dataset[split]
                break
            else:
                print(f"Split '{split}' not found in {dataset_name}-{config_name}. Available splits: {list(dataset.keys())}")
                continue
        except Exception as e:
            print(f"Could not load {dataset_name}: {e}")
            last_error = e
            continue
    
    # If we still don't have the dataset, we'll need to use a fallback approach
    if dataset is None:
        print(f"ERROR: Could not load TyDiQA dataset from any known sources. Last error: {last_error}")
        print("Using fallback sample data...")
        # Create a minimal fallback dataset with a few examples
        fallback_data = create_fallback_samples(lang_code)
        if fallback_data:
            return pd.DataFrame(fallback_data)
        return pd.DataFrame()
    
    try:
        # Filter to include only the specified language
        dataset = dataset.filter(lambda example: example['language'] == lang_code)
        
        print(f"  Found {len(dataset)} examples for language '{lang_code}' in split '{split}'")
        
        if len(dataset) == 0:
            print(f"ERROR: No samples found for language '{lang_code}' in TyDiQA, split '{split}'")
            return pd.DataFrame()
        
        # Verify 'answers' column has expected structure and 'text' contains answers
        if 'answers' not in dataset.column_names or 'text' not in dataset[0]['answers']:
            print(f"ERROR: Dataset missing expected 'answers.text' format in TyDiQA for {lang_code}")
            print(f"Available columns: {dataset.column_names}")
            return pd.DataFrame()
            
        # Sample if requested
        if num_samples is not None:
            if num_samples < len(dataset):
                print(f"  Sampling {num_samples} examples (from {len(dataset)} total)")
                dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
            else:
                print(f"  Requested {num_samples} samples, but only {len(dataset)} available. Using all.")
        
        # Convert to DataFrame
        samples_df = dataset.to_pandas()
        
        # Extract the first answer text as ground_truth - IMPORTANT: This is in the original LRL
        samples_df['ground_truth'] = samples_df['answers'].apply(
            lambda ans: ans['text'][0] if isinstance(ans, dict) and 'text' in ans and ans['text'] else None
        )
        
        # Drop rows with missing ground truth
        samples_df.dropna(subset=['ground_truth'], inplace=True)
        
        if samples_df.empty:
            print(f"WARNING: No valid samples with ground truth answers found for {lang_code}.")
            return pd.DataFrame()
            
        print(f"Successfully loaded {len(samples_df)} samples with LRL ground truth answers for {lang_code}.")
        
        # Cache to local file for future use
        try:
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            samples_df.to_json(local_file_path, orient='records')
            print(f"Cached dataset to {local_file_path} for future use.")
        except Exception as e:
            print(f"Failed to cache dataset: {e}")
            
        # Return the final DataFrame
        return samples_df
    
    except Exception as e:
        print(f"ERROR processing TyDiQA dataset for language {lang_code}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def create_fallback_samples(lang_code: str) -> list:
    """Create a minimal set of fallback samples for testing when the dataset cannot be loaded."""
    # Different samples for different languages
    if lang_code == "sw":  # Swahili
        return [
            {
                "id": "sw_1",
                "question": "Ni nani aliyeandika kitabu cha 'Things Fall Apart'?",
                "context": "Things Fall Apart ni kitabu kilichoandikwa na Chinua Achebe mnamo mwaka 1958.",
                "ground_truth": "Chinua Achebe",
                "answers": {"text": ["Chinua Achebe"], "answer_start": [40]}
            },
            {
                "id": "sw_2",
                "question": "Jiji kuu la Kenya ni lipi?",
                "context": "Nairobi ni jiji kuu la Kenya. Iko katika Afrika Mashariki.",
                "ground_truth": "Nairobi",
                "answers": {"text": ["Nairobi"], "answer_start": [0]}
            }
        ]
    elif lang_code == "te":  # Telugu
        return [
            {
                "id": "te_1",
                "question": "భారతదేశం యొక్క రాజధాని ఏమిటి?",
                "context": "న్యూఢిల్లీ భారతదేశం యొక్క రాజధాని మరియు ఒక కేంద్రపాలిత ప్రాంతం.",
                "ground_truth": "న్యూఢిల్లీ",
                "answers": {"text": ["న్యూఢిల్లీ"], "answer_start": [0]}
            },
            {
                "id": "te_2",
                "question": "తెలుగు ఏ రాష్ట్రంలో మాట్లాడతారు?",
                "context": "తెలుగు ప్రధానంగా ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో మాట్లాడే భాష.",
                "ground_truth": "ఆంధ్రప్రదేశ్ మరియు తెలంగాణ",
                "answers": {"text": ["ఆంధ్రప్రదేశ్ మరియు తెలంగాణ"], "answer_start": [20]}
            }
        ]
    elif lang_code == "en":  # English
        return [
            {
                "id": "en_1",
                "question": "Who wrote 'To Kill a Mockingbird'?",
                "context": "To Kill a Mockingbird is a novel by Harper Lee published in 1960.",
                "ground_truth": "Harper Lee",
                "answers": {"text": ["Harper Lee"], "answer_start": [28]}
            },
            {
                "id": "en_2",
                "question": "What is the capital of France?",
                "context": "Paris is the capital and most populous city of France.",
                "ground_truth": "Paris",
                "answers": {"text": ["Paris"], "answer_start": [0]}
            }
        ]
    else:
        print(f"No fallback samples available for language code: {lang_code}")
        return []

# Example usage
if __name__ == "__main__":
    # Test with Swahili
    sw_samples = load_tydiqa_samples('sw', num_samples=5)
    if not sw_samples.empty:
        print("\nSwahili Examples:")
        for i, row in sw_samples.head().iterrows():
            print(f"Question: {row['question']}")
            print(f"Answer (LRL): {row['ground_truth']}")
            print("-" * 50)
    
    # Available in TyDiQA-GoldP: 'ar', 'bn', 'en', 'fi', 'id', 'ko', 'ru', 'sw', 'te' 