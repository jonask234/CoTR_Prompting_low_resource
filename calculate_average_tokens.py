import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from datasets import load_dataset

# --- Configuration Section ---

# Define the list of model configurations to test
# Each dictionary specifies:
#   'hf_name': Hugging Face Hub identifier for the tokenizer.
#   'display_name': User-friendly name for plotting.
#   'trust_remote_code': Boolean, set to True if the tokenizer requires it.
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
        'hf_name': "mistralai/Mistral-7B-v0.1",
        'display_name': "Mistral-7B",
        'trust_remote_code': False
    },
    {
        'hf_name': "gpt2",
        'display_name': "GPT-2",
        'trust_remote_code': False
    }
]

# Define languages and their corresponding data sources from Hugging Face datasets
# Each key is a language display name (for plotting).
# Each value is a dictionary specifying:
#   'hf_dataset_name': Hugging Face dataset identifier.
#   'hf_config_name': Specific configuration/subset for the language (None if not applicable).
#   'split': Dataset split to use (e.g., "test", "validation", "train").
#   'text_column': Name of the column containing the text.
LANGUAGES_AND_DATA_SOURCES = {
    "English": {
        'hf_dataset_name': "wikitext",
        'hf_config_name': "wikitext-103-raw-v1", # Example: Using wikitext for English
        'split': "test",
        'text_column': "text"
    },
    "Swahili": {
        'hf_dataset_name': "masakhaner", # Contains Swahili text
        'hf_config_name': "swa", # Configuration for Swahili
        'split': "train", # Using train split as it's larger
        'text_column': "tokens" # This dataset has 'tokens' which is a list of words, will need to join
    },
    "Hausa": {
        'hf_dataset_name': "masakhaner",
        'hf_config_name': "hau",
        'split': "train",
        'text_column': "tokens"
    },
    "Telugu": {
        'hf_dataset_name': "oscar", 
        'hf_config_name': "unshuffled_deduplicated_te",
        'split': "train",
        'text_column': "text"
    },
    "Urdu": {
        'hf_dataset_name': "oscar", 
        'hf_config_name': "unshuffled_deduplicated_ur",
        'split': "train",
        'text_column': "text"
    }
}

# Number of text samples to randomly select and average over for each language
NUM_SAMPLES_PER_LANGUAGE = 100

# Optional: Truncate long text samples to a maximum number of characters
# Set to None or 0 for no truncation.
MAX_CHARS_PER_SAMPLE = 1000

# Control whether special tokens are added during tokenization
ADD_SPECIAL_TOKENS = True

# Filename for the saved plot
OUTPUT_PLOT_FILENAME = "average_tokens_generated_plot.png"

# Define the order of languages on the x-axis of the plot
LANGUAGE_PLOT_ORDER = ["English", "Swahili", "Hausa", "Telugu", "Urdu"]

# Define the custom color palette from the user's file
CUSTOM_COLOR_PALETTE = ['#2d5477', '#4e8a86', '#76a990', '#316e7e', '#40726f', '#254562', '#628c77', '#285b68']


# --- Core Logic ---

def main():
    """
    Main function to orchestrate token calculation and plotting.
    """
    all_processing_results = []
    print("Starting token calculation process...")

    for model_config in MODEL_CONFIGURATIONS:
        model_hf_name = model_config['hf_name']
        model_display_name = model_config['display_name']
        trust_remote_tokenizer = model_config['trust_remote_code']
        print(f"\\nProcessing Model: {model_display_name} ({model_hf_name})")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_hf_name,
                trust_remote_code=trust_remote_tokenizer
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"  Tokenizer for {model_display_name} did not have a pad_token. Set to eos_token: {tokenizer.eos_token}")

        except Exception as e:
            print(f"  ERROR: Could not load tokenizer for {model_display_name}: {e}")
            continue # Skip to the next model

        for lang_display_name, data_source_info in LANGUAGES_AND_DATA_SOURCES.items():
            print(f"  Processing Language: {lang_display_name} for model {model_display_name}")

            dataset_name = data_source_info['hf_dataset_name']
            config_name = data_source_info['hf_config_name']
            split_name = data_source_info['split']
            text_col = data_source_info['text_column']

            try:
                # Load the dataset
                # For datasets like masakhaner that might need specific handling for remote code:
                dataset = load_dataset(
                    dataset_name,
                    config_name,
                    split=split_name,
                    trust_remote_code=True # Often needed for community datasets
                )
                print(f"    Successfully loaded dataset '{dataset_name}' (config: {config_name}, split: {split_name})")

            except Exception as e:
                print(f"    ERROR: Could not load dataset {dataset_name} for {lang_display_name}: {e}")
                continue # Skip to the next language

            if not dataset or len(dataset) == 0:
                print(f"    WARNING: Dataset for {lang_display_name} is empty or failed to load properly.")
                continue

            # Randomly select samples
            num_available_samples = len(dataset)
            actual_samples_to_take = min(NUM_SAMPLES_PER_LANGUAGE, num_available_samples)

            if actual_samples_to_take == 0:
                print(f"    WARNING: No samples available to process for {lang_display_name}.")
                continue
            
            print(f"    Selecting {actual_samples_to_take} random samples out of {num_available_samples} available.")
            
            # Ensure we handle potential issues if dataset is smaller than requested samples
            if num_available_samples < NUM_SAMPLES_PER_LANGUAGE:
                print(f"    Note: Requested {NUM_SAMPLES_PER_LANGUAGE} samples, but only {num_available_samples} are available for {lang_display_name}.")
                random_indices = list(range(num_available_samples)) # Take all available
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

                    # Special handling for 'tokens' column (list of words)
                    if isinstance(text_content, list) and text_col == "tokens":
                        text_content = " ".join(text_content)
                    
                    if not isinstance(text_content, str):
                        # print(f"    WARNING: Sample {sample_idx} for {lang_display_name} has non-string content in '{text_col}'. Type: {type(text_content)}. Skipping.")
                        text_content = str(text_content) # Attempt to cast to string

                    if not text_content or not text_content.strip():
                        # print(f"    WARNING: Sample {sample_idx} for {lang_display_name} is empty or whitespace only. Skipping.")
                        continue

                    # Truncate text if configured
                    if MAX_CHARS_PER_SAMPLE and MAX_CHARS_PER_SAMPLE > 0 and len(text_content) > MAX_CHARS_PER_SAMPLE:
                        text_content = text_content[:MAX_CHARS_PER_SAMPLE]

                    # Tokenize
                    tokenized_output = tokenizer(
                        text_content,
                        add_special_tokens=ADD_SPECIAL_TOKENS,
                        # truncation=True, # Optional: if you want to enforce model's max length
                        # max_length=tokenizer.model_max_length 
                    )
                    
                    num_tokens = len(tokenized_output['input_ids'])
                    num_chars = len(text_content)
                    num_words = len(text_content.split()) # Simple word count

                    token_counts_for_lang_model.append(num_tokens)
                    char_counts_for_lang_model.append(num_chars)
                    word_counts_for_lang_model.append(num_words)
                    num_samples_processed_successfully += 1

                except Exception as e_sample:
                    print(f"    ERROR processing sample {sample_idx} for {lang_display_name} with {model_display_name}: {e_sample}")
                    # print(f"      Problematic text (first 50 chars): {str(item.get(text_col, 'N/A'))[:50]}")


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
                print(f"    Model: {model_display_name}, Language: {lang_display_name}, Avg Raw Tokens: {average_tokens_raw:.2f}, Avg Chars: {avg_chars_sample:.2f}, Avg Words: {avg_words_sample:.2f}, Tokens/100Words: {avg_tokens_per_100_words:.2f}, Tokens/1000Chars: {avg_tokens_per_1000_chars:.2f} (from {num_samples_processed_successfully} samples)")
            else:
                print(f"    No samples successfully tokenized for Model: {model_display_name}, Language: {lang_display_name}")

    # --- Data Handling and Plotting ---
    if not all_processing_results:
        print("\\nNo results were generated. Exiting.")
        return

    results_df = pd.DataFrame(all_processing_results)

    # Order languages for plotting
    results_df['Language'] = pd.Categorical(results_df['Language'], categories=LANGUAGE_PLOT_ORDER, ordered=True)
    results_df = results_df.sort_values('Language')

    # Determine the maximum y-value across both plots for a consistent scale
    max_y1 = results_df['AverageTokensPer100Words'].max()
    max_y2 = results_df['AverageTokensPer1000Chars'].max()
    global_max_y = max(max_y1, max_y2) * 1.1  # Add a 10% buffer

    # Generate and save the plots for the two specified metrics
    create_and_save_plot(results_df, 'AverageTokensPer100Words', "Average Tokens per 100 Words", "tokens_per_100_words", global_max_y)
    create_and_save_plot(results_df, 'AverageTokensPer1000Chars', "Average Tokens per 1000 Chars", "tokens_per_1000_chars", global_max_y)
    
    print(f"\\nPlots saved successfully.")
    print("Script finished.")


def create_and_save_plot(dataframe, y_metric, y_label, plot_filename_suffix, y_axis_limit=None):
    """
    Creates a bar plot from the dataframe for a given metric and saves it.
    
    Args:
        dataframe (pd.DataFrame): The data to plot.
        y_metric (str): The column name for the y-axis values.
        y_label (str): The label for the y-axis.
        plot_filename_suffix (str): A suffix to append to the output plot filename.
        y_axis_limit (float, optional): The upper limit for the y-axis. Defaults to None.
    """
    plt.style.use('seaborn-v0_8-whitegrid') # A clean style to start with
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.barplot(
        data=dataframe,
        x='Language',
        y=y_metric,
        hue='Model',
        palette=CUSTOM_COLOR_PALETTE,
        ax=ax
    )

    # --- Styling Customizations ---
    
    # Set plot labels and title
    ax.set_ylabel(y_label, fontsize=14, weight='normal')
    ax.set_xlabel("", fontsize=14, weight='bold') # Remove x-axis title
    # fig.suptitle(f'{y_label} Across Languages and Models', fontsize=18, weight='bold')

    # Set a consistent y-axis limit if provided
    if y_axis_limit is not None:
        ax.set_ylim(0, y_axis_limit)

    # Customize ticks
    ax.tick_params(axis='x', labelsize=12, rotation=0) # Ensure labels are not rotated
    ax.tick_params(axis='y', labelsize=12)

    # Customize legend
    legend = ax.legend(
        title="", # No legend title
        loc='upper right', # Place legend inside
        fontsize=11,
        frameon=False # No frame
    )
    
    # Customize grid and frame
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', linestyle='', alpha=0)
    sns.despine(left=False, bottom=False, right=False, top=False) # Remove frame/spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)


    # Adjust layout and save
    fig.tight_layout() # Adjust for suptitle
    
    final_filename = f"{os.path.splitext(OUTPUT_PLOT_FILENAME)[0]}_{plot_filename_suffix}.png"
    plt.savefig(final_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved as: {final_filename}")

# --- Execution ---

if __name__ == "__main__":
    main() 