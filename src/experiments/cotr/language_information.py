from typing import Dict, Any

def get_language_information(lang_code: str) -> Dict[str, Any]:
    """Provide basic information about a language based on its code.

    Args:
        lang_code: The language code (e.g., 'sw', 'ha', 'yo', 'en').

    Returns:
        A dictionary containing language details.
    """
    # Example information (expand as needed)
    lang_info = {
        "sw": {
            "language_name": "Swahili",
            "language_native": "Kiswahili",
            "ner_examples": [
                {
                    "text": "Rais wa Marekani Joe Biden alitembelea Berlin Jumanne iliyopita.",
                    "entities": "Joe Biden [PER], Marekani [LOC], Berlin [LOC], Jumanne iliyopita [DATE]"
                }
            ]
            # Add other potential keys like common_entities, typical_sentence_structure etc.
        },
        "ha": {
            "language_name": "Hausa",
            "language_native": "Harshen Hausa",
            "ner_examples": [
                {
                    "text": "Shugaban Amurka Joe Biden ya ziyarci Berlin ranar Talatar da ta gabata.",
                    "entities": "Joe Biden [PER], Amurka [LOC], Berlin [LOC], Talatar da ta gabata [DATE]"
                }
            ]
        },
        "yo": {
            "language_name": "Yoruba",
            "language_native": "Èdè Yorùbá",
            "ner_examples": [
                {
                    "text": "Ààrẹ Amẹ́ríkà Joe Biden lọ sí Berlin ní ọjọ́ Tuesday tó kọjá.",
                    "entities": "Joe Biden [PER], Amẹ́ríkà [LOC], Berlin [LOC], Tuesday tó kọjá [DATE]"
                }
            ]
        },
        "en": {
            "language_name": "English",
            "language_native": "English",
            "ner_examples": [
                {
                    "text": "The President of the United States, Joe Biden, visited Berlin last Tuesday.",
                    "entities": "Joe Biden [PER], United States [LOC], Berlin [LOC], Tuesday [DATE]"
                }
            ]
        }
        # Add other supported languages here
    }

    # Return the info for the requested language, or a default structure
    return lang_info.get(lang_code, {
        "language_name": lang_code.upper(),
        "language_native": lang_code.upper(),
        "ner_examples": []
    }) 