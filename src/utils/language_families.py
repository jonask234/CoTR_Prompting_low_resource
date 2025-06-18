import logging

# Configure logger for this module
logger = logging.getLogger(__name__)

# Primary source: Glottolog, with simplifications for high-level analysis.
# This mapping covers languages from AfriSenti, MasakhaNEWS, MasakhaNER, XNLI, and TyDiQA.
LANGUAGE_FAMILY_MAP = {
    # Niger-Congo -> Atlantic-Congo -> Benue-Congo -> Bantu
    "sw": "Bantu",           # Swahili
    "lin": "Bantu",          # Lingala
    "lug": "Bantu",          # Luganda
    "run": "Bantu",          # Rundi
    "sna": "Bantu",          # Shona
    "xho": "Bantu",          # Xhosa

    # Niger-Congo -> Atlantic-Congo -> Volta-Niger
    "yo": "Volta-Niger",     # Yoruba
    "ig": "Volta-Niger",     # Igbo

    # Afro-Asiatic -> Chadic
    "ha": "Afro-Asiatic (Chadic)",    # Hausa

    # Afro-Asiatic -> Semitic
    "am": "Afro-Asiatic (Semitic)",    # Amharic
    "tir": "Afro-Asiatic (Semitic)",   # Tigrinya

    # Afro-Asiatic -> Cushitic
    "orm": "Afro-Asiatic (Cushitic)",  # Oromo
    "som": "Afro-Asiatic (Cushitic)",  # Somali
    
    # Indo-European -> Germanic
    "en": "Indo-European (Germanic)", # English

    # Indo-European -> Romance
    "fr": "Indo-European (Romance)",  # French
    "pt": "Indo-European (Romance)",  # Portuguese

    # Indo-European -> Indo-Aryan
    "ma": "Indo-European (Indo-Aryan)", # Marathi

    # Dravidian
    "te": "Dravidian",       # Telugu

    # Sino-Tibetan
    "dz": "Sino-Tibetan",    # Dzongkha

    # Creole
    "pcm": "Creole",         # Nigerian Pidgin (English-based)

    # For multi-language datasets where a specific family isn't applicable
    "multi": "Multilingual"
}

def get_language_family(lang_code: str) -> str:
    """
    Retrieves the language family for a given language code.

    Args:
        lang_code (str): The two- or three-letter code for the language (e.g., 'sw', 'ha').

    Returns:
        str: The name of the language family or 'Unknown' if not found.
    """
    family = LANGUAGE_FAMILY_MAP.get(lang_code.lower())
    if family is None:
        logger.warning(f"Language code '{lang_code}' not found in LANGUAGE_FAMILY_MAP. Returning 'Unknown'.")
        return "Unknown"
    return family

def get_major_language_family(lang_code: str) -> str:
    """
    Retrieves the major, high-level language family for a given language code.
    For example, maps 'Afro-Asiatic (Semitic)' to 'Afro-Asiatic'.

    Args:
        lang_code (str): The two- or three-letter code for the language.

    Returns:
        str: The high-level language family name or 'Unknown'.
    """
    family = get_language_family(lang_code)
    if '(' in family:
        return family.split('(')[0].strip()
    return family

if __name__ == '__main__':
    # Example usage and testing
    test_languages = ['sw', 'ha', 'am', 'en', 'pcm', 'yo', 'nonexistent']
    print("--- Testing Language Family Retrieval ---")
    for lang in test_languages:
        print(f"Language: {lang}")
        print(f"  -> Family: {get_language_family(lang)}")
        print(f"  -> Major Family: {get_major_language_family(lang)}") 