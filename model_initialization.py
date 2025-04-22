# thesis_project/models/test_models/test_initialization.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Create cache directory if it doesn't exist
cache_path = "/work/bbd6522/cache_dir"
os.makedirs(cache_path, exist_ok=True)

# Set environment variables for caching
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

def initialize_model(model_name):
    """
    Initialize a model and tokenizer with proper caching and memory optimization.
    
    Args:
        model_name: Name of the model to load from HuggingFace
        
    Returns:
        model: The loaded model
        tokenizer: The loaded tokenizer
    """
    print(f"Initializing model: {model_name}")
    
    try:
        # Initialize tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_path
        )
        
        # Initialize model with memory optimization options
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_path,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto"  # Automatically distribute model across available devices
        )
        
        print(f"Successfully loaded {model_name}")
        return model, tokenizer  # Return model first, then tokenizer
        
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        raise