# -*- coding: utf-8 -*-
import os
from pathlib import Path

# Home-Verzeichnis
home_dir = str(Path.home())

# Konfigurationsverzeichnis
CONFIG_DIR = os.path.join(home_dir, ".cotr_config")
TOKEN_FILE = os.path.join(CONFIG_DIR, "huggingface_token.txt")

def get_token():
    # Holt den Hugging Face Token aus der Konfigurationsdatei.
    # Erstellt das Konfigurationsverzeichnis, falls es nicht existiert.
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Überprüft, ob die Token-Datei existiert.
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            token = f.read().strip()
            if token:
                return token
    
    # Wenn kein Token gefunden wird, den Benutzer zur Eingabe auffordern.
    token = input("Enter your Hugging Face token: ").strip()
    
    # Speichert den Token in der Datei.
    with open(TOKEN_FILE, 'w') as f:
        f.write(token)
    
    return token

def set_token(token):
    # Setzt den Hugging Face Token in der Konfigurationsdatei.
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(TOKEN_FILE, 'w') as f:
        f.write(token)
    os.chmod(TOKEN_FILE, 0o600) 