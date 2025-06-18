from typing import Dict, Any

def get_language_information(lang_code: str) -> Dict[str, Any]:
    """
    Provides a dictionary of information for a given language code.
    This includes language name, native name, and NER examples.
    """
    lang_info = {
        "sw": {
            "language_name": "Swahili",
            "language_native": "Kiswahili",
            "ner_examples": [
                {
                    "text": "James Mwangi kutoka Equity Bank alisafiri kwenda Nairobi Jumapili iliyopita.",
                    "entities": "[PER: James Mwangi] [ORG: Equity Bank] [LOC: Nairobi] [DATE: Jumapili iliyopita]"
                }
            ]
        },
        "ha": {
            "language_name": "Hausa",
            "language_native": "Harshen Hausa",
            "ner_examples": [
                {
                    "text": "Aliyu Muhammadu daga Dangote Group ya ziyarci Kano ranar Litinin.",
                    "entities": "[PER: Aliyu Muhammadu] [ORG: Dangote Group] [LOC: Kano] [DATE: ranar Litinin]"
                }
            ]
        },
        "en": {
            "language_name": "English",
            "language_native": "English",
            "ner_examples": [
                {
                    "text": "John Smith from Microsoft visited New York last Friday.",
                    "entities": "[PER: John Smith] [ORG: Microsoft] [LOC: New York] [DATE: Friday]"
                }
            ]
        }
    }
    return lang_info.get(lang_code, {}) 