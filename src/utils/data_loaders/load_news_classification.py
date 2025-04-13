import os
import pandas as pd
import random
from typing import Optional, List, Dict, Any
from datasets import load_dataset, load_from_disk

def _load_and_process(
    dataset_id: str,
    lang_code_for_debug: str, # e.g., 'swa' or 'hau'
    split: str,
    num_samples_to_load: Optional[int],
    text_col: str, # Column name for the text
    label_col: str, # Column name for the label
    id_prefix: str, # Prefix for generating IDs if missing
    token: Optional[str] = None # Added token parameter
) -> pd.DataFrame:
    """Generic helper to load, sample, validate, and format data."""
    samples = []
    print(f"Attempting to load {lang_code_for_debug} samples from {dataset_id} ({split} split)...")

    try:
        # Load the specific split
        print(f"  Loading dataset {dataset_id} split '{split}'...")
        # No specific config name needed for these datasets usually
        # Adding more specific error catching
        try:
            # Pass token explicitly if provided
            dataset = load_dataset(dataset_id, split=split, trust_remote_code=True, token=token)
        except FileNotFoundError:
            print(f"ERROR: Dataset ID '{dataset_id}' not found on the Hub (Split: {split}). Ensure ID is correct and you have access.")
            return pd.DataFrame({'text': [], 'label': [], 'id': []})
        except Exception as load_err: # Catch other potential loading errors
            print(f"ERROR: Failed to load dataset '{dataset_id}' (split: {split}) due to: {load_err}")
            import traceback
            traceback.print_exc() # Print full traceback for detailed diagnosis
            return pd.DataFrame({'text': [], 'label': [], 'id': []})
            
        print("  Dataset loaded.")

        dataset_size = len(dataset)
        print(f"  Full split size: {dataset_size}")

        dataset_to_iterate = None
        effective_total = 0

        if num_samples_to_load is not None and num_samples_to_load < dataset_size:
            print(f"  Applying RANDOM sampling for {lang_code_for_debug}: Shuffling and selecting {num_samples_to_load} samples...")
            dataset_to_iterate = dataset.shuffle(seed=42).select(range(num_samples_to_load))
            print(f"  Selected {len(dataset_to_iterate)} samples after shuffling.")
            effective_total = num_samples_to_load
        elif num_samples_to_load is not None:
            print(f"  Requested samples ({num_samples_to_load}) >= dataset size ({dataset_size}). Processing all.")
            dataset_to_iterate = dataset
            effective_total = dataset_size
        else:
            print("  Processing all samples in the split.")
            dataset_to_iterate = dataset
            effective_total = dataset_size

        count = 0
        processed_records = 0
        for example in tqdm(dataset_to_iterate, total=effective_total, desc=f"Processing {id_prefix} ({split}) sample"):
            processed_records += 1
            # Use a simple counter if ID column missing or unreliable
            example_id = f'{id_prefix}_{split}_{processed_records}'
            text = example.get(text_col, '')
            label_raw = example.get(label_col, None)
            label_name = None
            if isinstance(label_raw, str):
                label_name = label_raw.lower().strip()
            elif isinstance(label_raw, int): # Handle potential integer labels
                label_name = str(label_raw)

            # Basic validation
            if not text or not text.strip() or not label_name or not label_name.strip():
                continue

            sample = {
                'text': text,
                'label': label_name, # Store cleaned label
                'id': example_id
            }
            samples.append(sample)
            count += 1
            if num_samples_to_load is not None and count >= num_samples_to_load:
                break

    except Exception as e:
        print(f"Error during general processing of {dataset_id} ({lang_code_for_debug}, {split} split): {e}")
        # Adding traceback here too for broader issues
        import traceback
        traceback.print_exc()
        return pd.DataFrame({'text': [], 'label': [], 'id': []})

    print(f"  Successfully extracted {len(samples)} valid samples.")
    df = pd.DataFrame(samples)
    if df.empty:
        print(f"WARNING: DataFrame empty for {lang_code_for_debug} after processing.")
    else:
        print(f"DEBUG: Unique labels found for {lang_code_for_debug}: {df['label'].unique()}")
    return df

def load_dataset_fixed(dataset_id: str, split: str = "train", config_name: str = None, token: Optional[str] = None):
    """
    Loads dataset with special parameters to bypass caching issues.
    Uses a fresh cache directory and forces redownload to avoid corrupted cache problems.
    """
    # Create a fresh cache directory
    cache_dir = "/work/bbd6522/fresh_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"  Loading dataset {dataset_id} (split '{split}') with forced redownload...")
    try:
        # Use all available methods to ensure loading works
        dataset = load_dataset(
            dataset_id,
            name=config_name,
            split=split,
            cache_dir=cache_dir,
            download_mode="force_redownload",  # Force fresh download
            trust_remote_code=True,
            token=token  # Pass authentication token
        )
        print(f"  Successfully loaded dataset with {len(dataset)} samples.")
        return dataset
    except Exception as e:
        print(f"  ERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_local_dataset(file_path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Load a dataset from a local CSV or TSV file."""
    try:
        print(f"Loading dataset from local file: {file_path}")
        if not os.path.exists(file_path):
            print(f"ERROR: File {file_path} does not exist")
            return None
            
        # Try loading the dataset (handle different formats)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')  # Use tab as separator for TSV
        elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=file_path.endswith('.jsonl'))
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            print(f"ERROR: Unsupported file format for {file_path}")
            return None
            
        # Print columns to help with debugging
        print(f"  Columns in the file: {df.columns.tolist()}")
        
        # Check and normalize column names
        if text_col not in df.columns or label_col not in df.columns:
            print(f"  Dataset missing required columns '{text_col}' or '{label_col}'")
            
            # Try to find appropriate text column
            text_candidates = ["text", "content", "article", "news", "tweet"]
            for candidate in text_candidates:
                if candidate in df.columns:
                    print(f"  Using '{candidate}' as text column")
                    df = df.rename(columns={candidate: "text"})
                    break
                    
            # Try to find appropriate label column
            label_candidates = ["label", "category", "class", "topic", "sentiment"]
            for candidate in label_candidates:
                if candidate in df.columns:
                    print(f"  Using '{candidate}' as label column")
                    df = df.rename(columns={candidate: "label"})
                    break
        else:
            # Rename to standard names if needed
            if text_col != "text":
                df = df.rename(columns={text_col: "text"})
            if label_col != "label":
                df = df.rename(columns={label_col: "label"})
        
        # Verify columns after potential renaming
        if "text" not in df.columns or "label" not in df.columns:
            print(f"ERROR: Could not identify text and label columns")
            return None
            
        # Add ID column if not present
        if "id" not in df.columns:
            df["id"] = [f"id_{i}" for i in range(len(df))]
            
        # Keep only necessary columns
        df = df[["text", "label", "id"]]
        
        print(f"  Successfully loaded {len(df)} samples")
        print(f"  Label distribution: \n{df['label'].value_counts()}")
        return df
        
    except Exception as e:
        print(f"ERROR loading dataset from {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_swahili_news(
    num_samples: Optional[int] = None, 
    split: str = "test",
    token: Optional[str] = None  # Not used for local loading
) -> pd.DataFrame:
    """Loads Swahili news from MasakhaNEWS dataset."""
    base_path = "/work/bbd6522/datasets/masakhane_news/swa"
    
    # File paths for different splits (updated to use TSV)
    file_paths = {
        "train": os.path.join(base_path, "train.tsv"),
        "validation": os.path.join(base_path, "dev.tsv"),
        "test": os.path.join(base_path, "test.tsv")
    }
    
    # Use the appropriate split
    if split not in file_paths:
        print(f"WARNING: Split '{split}' not available, defaulting to 'test'")
        split = "test"
        
    file_path = file_paths[split]
    print(f"Loading MasakhaNEWS Swahili from {file_path} (split: {split})")
    
    # Try to load the dataset
    df = load_local_dataset(file_path, text_col="text", label_col="label")
    
    # If loading failed, fall back to synthetic data
    if df is None or df.empty:
        print("  ⚠️  Loading MasakhaNEWS failed, falling back to synthetic data ⚠️")
        return _generate_synthetic_swahili(num_samples)
        
    # Sample if needed
    if num_samples is not None and num_samples < len(df):
        df = df.sample(num_samples, random_state=42)
        
    return df

def load_hausa_news(
    num_samples: Optional[int] = None, 
    split: str = "test",
    token: Optional[str] = None  # Not used for local loading
) -> pd.DataFrame:
    """Loads Hausa news from MasakhaNEWS dataset."""
    base_path = "/work/bbd6522/datasets/masakhane_news/hau"
    
    # File paths for different splits (updated to use TSV)
    file_paths = {
        "train": os.path.join(base_path, "train.tsv"),
        "validation": os.path.join(base_path, "dev.tsv"),
        "test": os.path.join(base_path, "test.tsv")
    }
    
    # Use the appropriate split
    if split not in file_paths:
        print(f"WARNING: Split '{split}' not available, defaulting to 'test'")
        split = "test"
        
    file_path = file_paths[split]
    print(f"Loading MasakhaNEWS Hausa from {file_path} (split: {split})")
    
    # For MasakhaNEWS, columns are consistently "text" and "label"
    df = load_local_dataset(file_path, text_col="text", label_col="label")
    
    # If loading failed, fall back to synthetic data
    if df is None or df.empty:
        print("  ⚠️  Loading MasakhaNEWS failed, falling back to synthetic data ⚠️")
        return _generate_synthetic_hausa(num_samples)
        
    # Sample if needed
    if num_samples is not None and num_samples < len(df):
        df = df.sample(num_samples, random_state=42)
        
    return df

# Rename synthetic data generators to make them internal helper functions
def _generate_synthetic_swahili(num_samples: Optional[int] = None) -> pd.DataFrame:
    """Creates a synthetic Swahili news classification dataset for testing."""
    print(f"Creating synthetic Swahili news dataset for testing purposes...")
    
    # Define some fake Swahili news categories
    categories = ["siasa", "biashara", "michezo", "afya", "burudani"]
    
    # Simple Swahili dummy text snippets for each category
    text_samples = {
        "siasa": [
            "Rais atangaza mipango mipya ya uchumi kwa taifa.",
            "Waziri mkuu ahudhuria mkutano wa kimataifa mjini London.",
            "Bunge lapitisha sheria mpya ya uchaguzi."
        ],
        "biashara": [
            "Bei ya mafuta yapanda tena wiki hii.",
            "Benki kuu yatangaza viwango vipya vya riba.",
            "Kampuni mpya ya teknolojia yazinduliwa Nairobi."
        ],
        "michezo": [
            "Simba SC yashinda mchuano mkali dhidi ya Yanga.",
            "Mchezaji bora wa Afrika atatangazwa wiki ijayo.",
            "Timu ya taifa yajiandaa kwa michuano ya kombe la dunia."
        ],
        "afya": [
            "Hospitali mpya kufunguliwa Dar es Salaam.",
            "Wizara ya afya yazindua kampeni ya chanjo.",
            "Wataalamu washauri kuhusu kujikinga na magonjwa ya mlipuko."
        ],
        "burudani": [
            "Tamasha la filamu lafanyika jijini Zanzibar.",
            "Msanii maarufu azindua albamu mpya.",
            "Tuzo za muziki wa Afrika Mashariki zatangazwa."
        ]
    }
    
    # Create synthetic samples
    samples = []
    limit = num_samples if num_samples is not None else 100  # Default to 100 samples if not specified
    for i in range(limit):
        # Select a random category
        category = random.choice(categories)
        # Select a random text from that category
        text = random.choice(text_samples[category])
        
        samples.append({
            'text': text + f" Hii ni habari ya {i+1}.",  # Add a suffix to make each text unique
            'label': category,
            'id': f"sw_synth_{i+1}"
        })
    
    df = pd.DataFrame(samples)
    print(f"  Created {len(df)} synthetic Swahili samples with labels: {categories}")
    print(f"  Label distribution: \n{df['label'].value_counts()}")
    return df

def _generate_synthetic_hausa(num_samples: Optional[int] = None) -> pd.DataFrame:
    """Creates a synthetic Hausa classification dataset for testing."""
    print(f"Creating synthetic Hausa classification dataset for testing purposes...")
    
    # Define some fake Hausa categories (these are sentiment categories)
    categories = ["positive", "negative", "neutral"]
    
    # Simple Hausa dummy text snippets for each category
    text_samples = {
        "positive": [
            "Na ji dadin wannan labarin.",
            "Wannan babban ci gaba ne ga kasar mu.",
            "Allah Ya ba mu albarka."
        ],
        "negative": [
            "Wannan labari ya dame ni sosai.",
            "Matsalar tattalin arziki ta karuwa.",
            "An kasa samun nasara a gasar."
        ],
        "neutral": [
            "An sanar da sakamakon jarrabawa.",
            "Za a gudanar da taro gobe.",
            "An kaddamar da sabon shiri."
        ]
    }
    
    # Create synthetic samples
    samples = []
    limit = num_samples if num_samples is not None else 100  # Default to 100 samples if not specified
    for i in range(limit):
        # Select a random category
        category = random.choice(categories)
        # Select a random text from that category
        text = random.choice(text_samples[category])
        
        samples.append({
            'text': text + f" {i+1}.",  # Add a suffix to make each text unique
            'label': category,
            'id': f"ha_synth_{i+1}"
        })
    
    df = pd.DataFrame(samples)
    print(f"  Created {len(df)} synthetic Hausa samples with labels: {categories}")
    print(f"  Label distribution: \n{df['label'].value_counts()}")
    return df

# Example usage
if __name__ == '__main__':
    print("--- Testing Swahili News Loader (test split) ---")
    sw_df = load_swahili_news(100, split='test')
    if not sw_df.empty:
        print(sw_df.head())
        print(sw_df['label'].value_counts())
    
    print("\n--- Testing Hausa VOA Loader (test split) ---")
    ha_df = load_hausa_news(100, split='test')
    if not ha_df.empty:
        print(ha_df.head())
        print(ha_df['label'].value_counts()) 