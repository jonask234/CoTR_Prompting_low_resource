# thesis_project/data/load_mlqa.py

from datasets import load_dataset
import pandas as pd
import random

def load_mlqa_samples(language_pair, num_samples=100):
  
    print(f"Loading {language_pair} dataset...")
    dataset = load_dataset("mlqa", language_pair)
    
    # Get validation split as this is commonly used for evaluation
    validation_data = dataset["validation"]
    
    # Convert to DataFrame
    df = validation_data.to_pandas()
    
    # If there are more samples than requested, take a random subset
    if len(df) > num_samples:
        df = df.sample(num_samples, random_state=42)
    
    print(f"Loaded {len(df)} samples from {language_pair}")
    return df

if __name__ == "__main__":
    # Load samples for Hindi and Vietnamese
    hindi_samples = load_mlqa_samples("mlqa.hi.en", 100)
    vietnamese_samples = load_mlqa_samples("mlqa.vi.en", 100)
    
    # Display first few rows to verify data
    print("\nHindi samples:")
    print(hindi_samples[["question", "context", "answers"]].head(2))
    
    print("\nVietnamese samples:")
    print(vietnamese_samples[["question", "context", "answers"]].head(2))