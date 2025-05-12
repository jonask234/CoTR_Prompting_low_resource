import os
import time
import json
import random
import openai
from openai import OpenAI

# --- API Key Setup ---
API_KEY = os.environ.get('OPENAI_API_KEY')
if not API_KEY:
    # Try reading from file if environment variable is not set
    try:
        key_path = os.path.expanduser("~/.openai_api_key")
        if os.path.exists(key_path):
            with open(key_path, 'r') as f:
                API_KEY = f.read().strip()
                if API_KEY:
                     os.environ['OPENAI_API_KEY'] = API_KEY # Set env var for consistency
                     print("Loaded OpenAI API key from file.")
                else:
                     print("Warning: Found API key file but it was empty.")
        # else: # Removed redundant warning, will be caught below
        #     print("Warning: OpenAI API key file not found (~/.openai_api_key).")
    except Exception as e:
        print(f"Warning: Could not read OpenAI API key from file: {e}")

# Check if API key is available after trying env var and file
if not API_KEY:
    print("CRITICAL ERROR: OpenAI API key is not set.")
    print("Please set the OPENAI_API_KEY environment variable or place it in ~/.openai_api_key")
    # Optionally raise an error to stop execution immediately
    # raise ValueError("OpenAI API key is required but not found.")

# Initialize OpenAI client (it will use the key from env var)
# Only initialize if key is found to avoid errors later
client = None
if API_KEY:
    try:
        client = OpenAI()
        print("OpenAI client initialized.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        client = None # Ensure client is None if init fails
else:
    print("OpenAI client not initialized because API key is missing.")

# --- API Call Function ---

def call_llm_api(prompt, model_name="gpt-3.5-turbo", temperature=0.0, max_tokens=1000, retry_count=3, retry_delay=5):
    """
    Call an LLM API with the given prompt.
    Uses the globally initialized client.
    """
    global client
    if not client:
        # Raise error or return a specific error message if client wasn't initialized
        raise ConnectionError("OpenAI client is not initialized. API key might be missing.")

    for attempt in range(retry_count + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error calling OpenAI API (Attempt {attempt + 1}/{retry_count + 1}): {e}")
            if attempt < retry_count:
                print(f"Retrying in {retry_delay * (attempt + 1)} seconds...")
                time.sleep(retry_delay * (attempt + 1)) # Exponential backoff
            else:
                print(f"Failed to call API after {retry_count + 1} attempts.")
                raise # Re-raise the last exception

# --- Placeholder for Open Source Models ---

def call_opensource_llm(prompt, model_name="llama-2-7b", temperature=0.0, max_tokens=1000):
    """
    Placeholder for calling an open source LLM.
    """
    print(f"Placeholder: Calling open source model {model_name} (not implemented)")
    return "Placeholder response from open source model." 