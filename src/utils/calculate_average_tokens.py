import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login
from config import get_token


# Eine Liste von Modellen
MODEL_CONFIGURATIONS = [
    {
        'hf_name': "Qwen/Qwen2.5-7B-Instruct",
        'display_name': "Qwen2.5-7B",
        'trust_remote_code': True
    },
    {
        'hf_name': "CohereLabs/aya-expanse-8b",
        'display_name': "Aya-23-8B",
        'trust_remote_code': True
    },
    {
        'hf_name': "gpt2",
        'display_name': "GPT-2",
        'trust_remote_code': False
    }
]

# Sprachen 
LANGUAGES_AND_DATA_SOURCES = {
    "English": {
        'hf_dataset_name': "wikitext",
        'hf_config_name': "wikitext-103-raw-v1",
        'split': "test",
        'text_column': "text"
    },
    "Swahili": {
        'hf_dataset_name': "masakhaner",
        'hf_config_name': "swa",
        'split': "train",
        'text_column': "tokens"
    },
    "Hausa": {
        'hf_dataset_name': "masakhaner",
        'hf_config_name': "hau",
        'split': "train",
        'text_column': "tokens"
    },
    "Urdu": {
        'hf_dataset_name': "oscar", 
        'hf_config_name': "unshuffled_deduplicated_ur",
        'split': "train",
        'text_column': "text"
    }
}

# Anzahl der Textbeispiele pro Sprache.
NUM_SAMPLES_PER_LANGUAGE = 100

# Maximale Zeichenlänge pro Beispiel.
MAX_CHARS_PER_SAMPLE = 1000

# speziellen Tokens.
ADD_SPECIAL_TOKENS = True

# Dateiname 
OUTPUT_PLOT_FILENAME = "average_tokens_generated_plot.png"

# Reihenfolge der Sprachen 
LANGUAGE_PLOT_ORDER = ["English", "Swahili", "Hausa", "Urdu"]

# Farbpalette.
CUSTOM_COLOR_PALETTE = ['#2d5477', '#4e8a86', '#76a990', '#316e7e', '#40726f', '#254562', '#628c77', '#285b68']


def main():
    
    # HUGGING FACE LOGIN
    token = get_token()
    if token:
        login(token=token)

    all_processing_results = []

    for model_config in MODEL_CONFIGURATIONS:
        model_hf_name = model_config['hf_name']
        model_display_name = model_config['display_name']
        trust_remote_tokenizer = model_config['trust_remote_code']

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_hf_name,
                trust_remote_code=trust_remote_tokenizer
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"ERROR: Could not load tokenizer for {model_display_name}: {e}")
            continue

        for lang_display_name, data_source_info in LANGUAGES_AND_DATA_SOURCES.items():
            dataset_name = data_source_info['hf_dataset_name']
            config_name = data_source_info['hf_config_name']
            split_name = data_source_info['split']
            text_col = data_source_info['text_column']

            try:
                dataset = load_dataset(
                    dataset_name,
                    config_name,
                    split=split_name,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"ERROR: Could not load dataset {dataset_name} for {lang_display_name}: {e}")
                continue

            if not dataset or len(dataset) == 0:
                continue

            num_available_samples = len(dataset)
            actual_samples_to_take = min(NUM_SAMPLES_PER_LANGUAGE, num_available_samples)

            if actual_samples_to_take == 0:
                continue
            
            if num_available_samples < NUM_SAMPLES_PER_LANGUAGE:
                random_indices = list(range(num_available_samples))
            else:
                random_indices = random.sample(range(num_available_samples), actual_samples_to_take)
            
            selected_samples = dataset.select(random_indices)
            
            token_counts_for_lang_model = []
            char_counts_for_lang_model = []
            word_counts_for_lang_model = []
            num_samples_processed_successfully = 0

            for sample_idx, item in enumerate(selected_samples):
                try:
                    text_content = item[text_col]

                    if isinstance(text_content, list) and text_col == "tokens":
                        text_content = " ".join(text_content)
                    
                    if not isinstance(text_content, str):
                        text_content = str(text_content)

                    if not text_content or not text_content.strip():
                        continue

                    if MAX_CHARS_PER_SAMPLE and MAX_CHARS_PER_SAMPLE > 0 and len(text_content) > MAX_CHARS_PER_SAMPLE:
                        text_content = text_content[:MAX_CHARS_PER_SAMPLE]

                    tokenized_output = tokenizer(
                        text_content,
                        add_special_tokens=ADD_SPECIAL_TOKENS,
                    )
                    
                    num_tokens = len(tokenized_output['input_ids'])
                    num_chars = len(text_content)
                    num_words = len(text_content.split())

                    token_counts_for_lang_model.append(num_tokens)
                    char_counts_for_lang_model.append(num_chars)
                    word_counts_for_lang_model.append(num_words)
                    num_samples_processed_successfully += 1

                except Exception as e_sample:
                    print(f"ERROR processing sample {sample_idx} for {lang_display_name} with {model_display_name}: {e_sample}")

            if token_counts_for_lang_model:
                average_tokens_raw = sum(token_counts_for_lang_model) / len(token_counts_for_lang_model)
                total_tokens = sum(token_counts_for_lang_model)
                total_chars = sum(char_counts_for_lang_model)
                total_words = sum(word_counts_for_lang_model)

                avg_chars_sample = total_chars / num_samples_processed_successfully if num_samples_processed_successfully > 0 else 0
                avg_words_sample = total_words / num_samples_processed_successfully if num_samples_processed_successfully > 0 else 0
                
                avg_tokens_per_100_words = (total_tokens / total_words) * 100 if total_words > 0 else 0
                avg_tokens_per_1000_chars = (total_tokens / total_chars) * 1000 if total_chars > 0 else 0
                
                all_processing_results.append({
                    'Model': model_display_name,
                    'Language': lang_display_name,
                    'AverageRawTokensPerSample': average_tokens_raw,
                    'AverageCharsPerSample': avg_chars_sample,
                    'AverageWordsPerSample': avg_words_sample,
                    'AverageTokensPer100Words': avg_tokens_per_100_words,
                    'AverageTokensPer1000Chars': avg_tokens_per_1000_chars,
                    'NumTextSamples': num_samples_processed_successfully
                })

    if not all_processing_results:
        print("No results were generated. Exiting.")
        return

    results_df = pd.DataFrame(all_processing_results)

    results_df['Language'] = pd.Categorical(results_df['Language'], categories=LANGUAGE_PLOT_ORDER, ordered=True)
    results_df = results_df.sort_values('Language')

    max_y1 = results_df['AverageTokensPer100Words'].max()
    max_y2 = results_df['AverageTokensPer1000Chars'].max()
    global_max_y = max(max_y1, max_y2) * 1.1

    create_and_save_plot(results_df, 'AverageTokensPer100Words', "Average Tokens per 100 Words", "tokens_per_100_words", global_max_y)
    create_and_save_plot(results_df, 'AverageTokensPer1000Chars', "Average Tokens per 1000 Chars", "tokens_per_1000_chars", global_max_y)
    
    print("Script finished. Plots saved.")


def create_and_save_plot(dataframe, y_metric, y_label, plot_filename_suffix, y_axis_limit=None):
    # Erstellt ein Balkendiagramm 
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.barplot(
        data=dataframe,
        x='Language',
        y=y_metric,
        hue='Model',
        palette=CUSTOM_COLOR_PALETTE,
        ax=ax
    )
    
    ax.set_ylabel(y_label, fontsize=16, weight='normal')
    ax.set_xlabel("", fontsize=16, weight='bold')

    if y_axis_limit is not None:
        ax.set_ylim(0, y_axis_limit)

    ax.tick_params(axis='x', labelsize=14, rotation=0)
    ax.tick_params(axis='y', labelsize=14)

    ax.legend(
        title="",
        loc='upper left',
        fontsize=12,
        frameon=False
    )
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', linestyle='', alpha=0)
    sns.despine(left=False, bottom=False, right=False, top=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    fig.tight_layout()
    
    final_filename = f"{os.path.splitext(OUTPUT_PLOT_FILENAME)[0]}_{plot_filename_suffix}.png"
    plt.savefig(final_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main() 