import os
from pathlib import Path

# home directory
home_dir = str(Path.home())

# config directory
CONFIG_DIR = os.path.join(home_dir, ".cotr_config")
TOKEN_FILE = os.path.join(CONFIG_DIR, "huggingface_token.txt")

def get_token():
    """
    Get the Hugging Face token from the config file.
    If the token file doesn't exist, prompt the user to enter it.
    """
    # Create config erstellen
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Check if token file exists
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            token = f.read().strip()
            if token:
                return token
    
    # wenn kein token vorhanden, bitte eingeben
    print("\nNo Hugging Face token found. Please enter your token to continue.")
    print("You can get your token from: https://huggingface.co/settings/tokens")
    token = input("Enter your Hugging Face token: ").strip()
    
    # speichere den token
    with open(TOKEN_FILE, 'w') as f:
        f.write(token)
    
    # nur eigener Benutzer
    os.chmod(TOKEN_FILE, 0o600)
    
    return token

def set_token(token):
    """
    Set the Hugging Face token in the config file.
    """
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(TOKEN_FILE, 'w') as f:
        f.write(token)
    os.chmod(TOKEN_FILE, 0o600) 