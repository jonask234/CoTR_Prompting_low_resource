# -*- coding: utf-8 -*-
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Cache-Verzeichnis erstellen, falls es nicht existiert.
cache_path = "/work/bbd6522/cache_dir"
os.makedirs(cache_path, exist_ok=True)

# Umgebungsvariablen für das Caching setzen.
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

def initialize_model(model_name):
    # Initialisiert ein Modell und einen Tokenizer.
    
    try:
        # Zuerst den Tokenizer initialisieren.
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_path
        )
        
        # Das Modell mit Speicheroptimierung initialisieren.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        raise