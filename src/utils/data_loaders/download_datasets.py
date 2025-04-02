
from datasets import load_dataset
import pandas as pd

def load_and_preprocess_datasets():
    # Load datasets for each task
    datasets = {
        "QA": {
            "MLQA": ["mlqa.hi.en", "mlqa.vi.en"],
            "XQuAD": ["xquad.tr", "xquad.vi"]
        },
        "Summarization": {
            "csebuetnlp/xlsum": ["indonesian", "oromo"],
            "GEM/wiki_lingua": ["th", "ar"]
        },
        "NER": {
            "MasakhaNER": ["yor", "hau"],
            "unimelb-nlp/wikiann": ["km", "jv"]
        },
        "Sentiment Analysis": {
            "cardiffnlp/tweet_sentiment_multilingual": ["hindi", "arabic"]
        }
    }

    for task, task_datasets in datasets.items():
        for dataset_name, languages in task_datasets.items():
            for lang in languages:
                print(f"Loading {dataset_name} for {lang}")
                dataset = load_dataset(dataset_name.lower(), lang)
                
                # Check available splits
                available_splits = dataset.keys()
                print(f"Available splits for {dataset_name} in {lang}: {available_splits}")
                
                # Access the first available split
                for split in available_splits:
                    print(f"Processing split: {split}")
                    df = dataset[split].to_pandas()
                    print(df.head())

if __name__ == "__main__":
    load_and_preprocess_datasets()
