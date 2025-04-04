import pandas as pd
import json
import os
from typing import Optional, List
import requests
from tqdm import tqdm
import gzip

def download_tydiqa_data(cache_dir: str = None) -> str:
    """
    Download TyDi QA dataset if not already downloaded.
    
    Args:
        cache_dir: Directory to save the dataset
        
    Returns:
        Path to the dataset directory
    """
    if cache_dir is None:
        # Use default cache directory
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache", "tydiqa")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define URLs for the dataset files
    urls = {
        "train": "https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-train.jsonl.gz",
        "dev": "https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-dev.jsonl.gz"
    }
    
    # Download files if they don't exist
    for split_name, url in urls.items(): # Use split_name instead of split
        output_path = os.path.join(cache_dir, f"tydiqa-v1.0-{split_name}.jsonl.gz")
        if not os.path.exists(output_path):
            print(f"Downloading TyDi QA {split_name} split...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=output_path
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"Downloaded {output_path}")
        else:
            print(f"Found cached file: {output_path}")
    
    return cache_dir

def _load_samples_from_split(data_path: str, tydiqa_lang: str, num_samples_to_load: Optional[int]) -> List[dict]:
    """Helper function to load samples from a specific split file."""
    samples = []
    count = 0
    processed_lines = 0
    lang_lines_found = 0
    print(f"Attempting to load {tydiqa_lang} samples from {data_path}...")
    
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            processed_lines += 1
            example = None # Initialize example to None
            try:
                example = json.loads(line.strip())
                example_id = example.get('example_id', f'line_{line_num}') # Use line number if ID missing
                
                # Check language
                if example.get('language') != tydiqa_lang:
                    continue
                
                lang_lines_found += 1
                
                question = example.get('question_text', '')
                if not question:
                    continue # Skip if no question
                    
                context = None 
                answers = {"text": [], "answer_start": []}
                found_answer = False
                passage_index = -1 # Track the best passage index found

                # --- Find Answer First --- 
                for annotation in example.get('annotations', []):
                    if annotation.get('minimal_answers'):
                        for minimal_answer in annotation['minimal_answers']:
                            if minimal_answer.get('plaintext'):
                                answers['text'].append(minimal_answer['plaintext'])
                                answers['answer_start'].append(minimal_answer.get('start_byte', -1)) # Store absolute start byte
                                found_answer = True
                    elif annotation.get('yes_no_answer') and annotation['yes_no_answer'] != "NONE":
                        answers['text'].append(annotation['yes_no_answer'])
                        answers['answer_start'].append(-1)
                        found_answer = True
                        
                    # If we found an answer, store the passage index from this annotation if available
                    if found_answer:
                        current_passage_index = annotation.get('passage_answer_candidate_index', -1)
                        if current_passage_index >= 0:
                            passage_index = current_passage_index # Prioritize passage index from answer annotation
                        break # Stop searching for answers once found

                # Skip if no answer was found across all annotations
                if not found_answer:
                    continue
                    
                # --- Determine Context --- 
                document_plaintext = example.get('document_plaintext', '')
                if passage_index >= 0 and passage_index < len(example.get('document_plaintext', [])):
                    # Use the specific passage if index is valid
                    context = example['document_plaintext'][passage_index]
                elif document_plaintext: 
                    # Fallback: Use the full document if passage index is invalid but an answer was found
                    print(f"    Warning: No valid passage index for {example_id}. Using full document as context.")
                    context = document_plaintext
                else:
                    # Skip if no context can be determined
                    continue 
                    
                # Ensure context is not empty
                if not context:
                    continue

                # --- Adjust Answer Start Indices --- 
                passage_start_byte = document_plaintext.find(context) if context and document_plaintext else -1
                adjusted_answers = {"text": answers["text"], "answer_start": []}
                for start_byte in answers['answer_start']:
                    relative_start = start_byte - passage_start_byte if start_byte >= 0 and passage_start_byte >= 0 else -1
                    adjusted_answers['answer_start'].append(relative_start)
                
                # Create and append the sample
                sample = {
                    'context': context, 
                    'question': question,
                    'answers': adjusted_answers, # Use adjusted answers
                    'id': example_id
                }
                samples.append(sample)
                count += 1
                
                # Check if requested number of samples is reached
                if num_samples_to_load is not None and count >= num_samples_to_load:
                    break
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                err_id = example.get('example_id', f'line_{line_num}') if example else f'line_{line_num}'
                print(f"Error processing example {err_id}: {e}") 
                continue
    
    print(f"  Finished processing {processed_lines} lines from {data_path}.")
    print(f"  Found {lang_lines_found} lines matching language '{tydiqa_lang}'.")
    print(f"  Successfully extracted {len(samples)} valid samples.")
    return samples

def load_tydiqa_samples(lang_code: str, num_samples: Optional[int] = None, split: str = "dev") -> pd.DataFrame:
    """
    Load TyDi QA samples for a specific language.
    Tries the specified split first, then falls back to 'train' if the required
    number of samples (`num_samples`) is not met.
    
    Args:
        lang_code: Language code (e.g., 'bn' for Bengali, 'sw' for Swahili, 'id' for Indonesian)
        num_samples: Target number of samples to load (None for all available in the first split tried).
                     If specified, the function attempts to load this many samples,
                     potentially combining results from 'dev' and 'train' splits.
        split: Preferred dataset split to use first ('dev' or 'train')
        
    Returns:
        DataFrame containing the samples
    """
    # Download or use cached data
    cache_dir = download_tydiqa_data()
    
    # Map language code to TyDi QA language name
    lang_code_map = {
        'bn': 'bengali',
        'te': 'telugu', 
        'sw': 'swahili',
        'id': 'indonesian',
        'bengali': 'bengali',
        'telugu': 'telugu',
        'swahili': 'swahili',
        'indonesian': 'indonesian'
    }
    tydiqa_lang = lang_code_map.get(lang_code)
    if not tydiqa_lang:
        print(f"ERROR: Unsupported language code '{lang_code}' for TyDi QA loader.")
        # Return empty DataFrame if language code is invalid
        return pd.DataFrame({
            'context': [],
            'question': [],
            'answers': [],
            'id': []
        })

    
    # Try the preferred split first
    preferred_data_path = os.path.join(cache_dir, f"tydiqa-v1.0-{split}.jsonl.gz")
    all_samples = _load_samples_from_split(preferred_data_path, tydiqa_lang, num_samples)
    
    # If a specific number of samples was requested and not met, try the 'train' split
    samples_needed = 0
    if num_samples is not None and len(all_samples) < num_samples:
        samples_needed = num_samples - len(all_samples)
        print(f"Found {len(all_samples)}/{num_samples} samples in '{split}' split. Trying 'train' split for {samples_needed} more...")
        
        # Avoid trying train split if it was the preferred split already
        if split != "train":
            train_data_path = os.path.join(cache_dir, "tydiqa-v1.0-train.jsonl.gz")
            # Try to load the remaining needed samples from the train split
            train_samples = _load_samples_from_split(train_data_path, tydiqa_lang, samples_needed)
            all_samples.extend(train_samples) # Combine samples from both splits

    print(f"Loaded {len(all_samples)} {tydiqa_lang} samples in total.")
    
    # Return DataFrame
    if not all_samples:
        print(f"WARNING: No {tydiqa_lang} samples found in any split!")
        return pd.DataFrame({
            'context': [],
            'question': [],
            'answers': [],
            'id': []
        })
        
    # Ensure we don't return more samples than requested if num_samples was specified
    if num_samples is not None and len(all_samples) > num_samples:
        all_samples = all_samples[:num_samples]
        print(f"  Truncated to requested {num_samples} samples.")
    
    return pd.DataFrame(all_samples) 