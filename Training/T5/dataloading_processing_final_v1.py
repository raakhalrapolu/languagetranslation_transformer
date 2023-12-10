import json
import torch
from transformers import T5Tokenizer
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# Paths for data files
ukr_file_path = '/home/pandu3011/project/data/wmt22-uken/train.ukr'
eng_file_path = '/home/pandu3011/project/data/wmt22-uken/train.eng'
preprocessed_data_path = "/home/pandu3011/project/data/preprocessed_data.json"

# Function to load data from a file
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
            print(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"An error occurred while loading data from {file_path}: {e}")
        return []

# Function to preprocess data
def preprocess_data(english_sentences,ukrainian_sentences):
    data = []
    for eng, ukr in zip(english_sentences , ukrainian_sentences):
        input_text = f"translate English to Ukrainian : {eng.strip()}"
        target_text = ukr.strip()
        data.append({"input_text": input_text, "target_text": target_text})
    return data

# Function to tokenize data
def tokenize_data(item, tokenizer):
    return (
        tokenizer.encode(item['input_text'], return_tensors='pt', truncation=True, padding='max_length', max_length=512),
        tokenizer.encode(item['target_text'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    )

# Function to process a batch of data
def process_batch(batch, tokenizer):
    return [tokenize_data(item, tokenizer) for item in batch]

# Main execution for tokenization
if __name__ == "__main__":
    # Load the raw data
    ukr_data = load_data(ukr_file_path)[:50000]  # Load only the first 50,000 sentences
    eng_data = load_data(eng_file_path)[:50000]  # Load only the first 50,000 sentences

    # Preprocess the data
    preprocessed_data = preprocess_data(eng_data, ukr_data)
    print(f"Preprocessing complete. Total records: {len(preprocessed_data)}")

    # Save the preprocessed data
    with open(preprocessed_data_path, "w", encoding="utf-8") as f:
        json.dump(preprocessed_data, f, ensure_ascii=False, indent=4)
    print(f"Saved preprocessed data to {preprocessed_data_path}")

    # Initialize the tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    combined_tokenized_data = []

    num_processes = 4  # Number of processes
    batch_size = 2500
    batches = [preprocessed_data[i:i + batch_size] for i in range(0, len(preprocessed_data), batch_size)]

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        func = partial(process_batch, tokenizer=tokenizer)
        for tokenized_batch in tqdm(executor.map(func, batches), total=len(batches), desc="Tokenizing in parallel"):
            combined_tokenized_data.extend(tokenized_batch)
    print("Tokenization complete.")

    torch.save(combined_tokenized_data, "/home/pandu3011/project/data/combined_tokenized_data.pt")
    print("Combined tokenized data saved.")

