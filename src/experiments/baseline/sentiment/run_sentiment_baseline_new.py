import sys
import os
import argparse
import time
import pandas as pd
import numpy as np
import torch

# Tokenizer-Parallelität deaktivieren
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Projekt-Root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from huggingface_hub import login
from config import get_token

# Funktionen aus dem Baseline-Skript importieren
from src.experiments.baseline.sentiment.sentiment_baseline_new import (
    initialize_model,
    evaluate_sentiment_baseline
)
from src.utils.data_loaders.load_afrisenti import load_afrisenti_samples
from evaluation.sentiment_metrics import calculate_sentiment_metrics

# Generierungsparameter
GENERATION_PARAMS = {
    "temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_tokens": 10, "repetition_penalty": 1.1
}

def main():
    parser = argparse.ArgumentParser(description="Runs sentiment analysis experiments.")
    parser.add_argument("--models", type=str, default="CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct", help="Model names, separated by comma.")
    parser.add_argument("--langs", type=str, default="sw,ha,pt", help="Language codes, separated by comma.")
    parser.add_argument("--num_samples", type=int, default=80, help="Number of samples per language.")
    parser.add_argument("--data_split", type=str, default="test", help="Dataset split (train, validation, test).")
    parser.add_argument("--shot_settings", nargs='+', default=['zero_shot', 'few_shot'], help="Prompting strategies.")
    parser.add_argument("--base_output_dir", type=str, default="/work/bbd6522/results/sentiment_new/baseline", help="Base output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--overwrite_results", action='store_true', help="Overwrites existing results.")
    parser.add_argument("--test_mode", action='store_true', help="Runs the script with a few samples for testing.")

    args = parser.parse_args()

    # Seed setzen
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # HF Login
    token = get_token()
    if token:
        try:
            login(token=token)
            print("Successfully logged into Hugging Face Hub.")
        except Exception as e:
            print(f"Hugging Face login failed: {e}")
    else:
        print("HuggingFace token not provided.")

    models_list = [m.strip() for m in args.models.split(',')]
    langs_list = [lang.strip() for lang in args.langs.split(',')]
    all_summaries = []

    for model_name in models_list:
        print(f"\n===== Initializing Model: {model_name} =====")
        tokenizer, model = None, None
        try:
            tokenizer, model = initialize_model(model_name)
            print(f"Model {model_name} initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize model {model_name}: {e}. Skipping.")
            continue

        for lang_code in langs_list:
            print(f"\n--- Processing Language: {lang_code} for model {model_name} ---")
            
            num_samples_to_load = 5 if args.test_mode else args.num_samples
            
            try:
                samples_df = load_afrisenti_samples(lang_code=lang_code, split=args.data_split, num_samples=num_samples_to_load)
                if samples_df.empty:
                    print(f"Warning: No samples loaded for {lang_code}. Skipping.")
                    continue
                print(f"Loaded {len(samples_df)} samples for {lang_code}.")
            except Exception as e:
                print(f"Error loading data for {lang_code}: {e}. Skipping.")
                continue

            for shot_setting in args.shot_settings:
                use_few_shot = (shot_setting == 'few_shot')
                prompt_in_lrl = (lang_code != 'en')
                
                print(f"  Running: Shot={shot_setting}, LRL-Prompt={prompt_in_lrl}")

                results_dir = os.path.join(args.base_output_dir, "results")
                os.makedirs(results_dir, exist_ok=True)
                model_name_safe = model_name.replace("/", "_")
                results_file = os.path.join(results_dir, f"results_sentiment_{lang_code}_{model_name_safe}_{shot_setting}.csv")

                if not args.overwrite_results and os.path.exists(results_file):
                    print(f"Results file exists, skipping: {results_file}")
                    continue

                results_df = evaluate_sentiment_baseline(
                    model_name=model_name,
                    tokenizer=tokenizer,
                    model=model,
                    samples_df=samples_df,
                    lang_code=lang_code,
                    prompt_in_lrl=prompt_in_lrl,
                    use_few_shot=use_few_shot,
                    temperature=GENERATION_PARAMS["temperature"],
                    top_p=GENERATION_PARAMS["top_p"],
                    top_k=GENERATION_PARAMS["top_k"],
                    max_tokens=GENERATION_PARAMS["max_tokens"],
                    repetition_penalty=GENERATION_PARAMS["repetition_penalty"]
                )

                if not results_df.empty:
                    results_df.to_csv(results_file, index=False, float_format='%.4f')
                    print(f"Detailed results saved to {results_file}")

                    metrics = calculate_sentiment_metrics(results_df)
                    
                    summary_data = {
                        'model': model_name,
                        'language': lang_code,
                        'shot_type': shot_setting,
                        'accuracy': metrics.get('accuracy'),
                        'macro_f1': metrics.get('macro_f1'),
                        'samples_processed': len(results_df)
                    }
                    all_summaries.append(summary_data)
                else:
                    print(f"Evaluation returned no results for {model_name}, lang {lang_code}, shot {shot_setting}.")
        
        print(f"Unloading model: {model_name}")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summaries_dir = os.path.join(args.base_output_dir, "summaries")
        os.makedirs(summaries_dir, exist_ok=True)
        summary_file = os.path.join(summaries_dir, f"sentiment_baseline_summary_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        summary_df.to_csv(summary_file, index=False, float_format='%.4f')
        print(f"\nOverall summary saved to {summary_file}")
        print(summary_df.to_string())
    else:
        print("No experiments were completed successfully.")

    print("\nAll Sentiment Baseline experiments finished!")

if __name__ == "__main__":
    main() 