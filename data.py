import torch
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import os
import time

# Ensure the output directory exists
output_dir = 'token_batches'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset from Hugging Face
print("Loading the dataset...")
dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split='train', streaming=True)

# Initialize the GPT-2 tokenizer
print("Initializing the GPT-2 tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Tokenization process
print("Starting tokenization...")
tokens_per_file = 10000000  # Save after processing 10 million tokens
all_tokens = []
token_count = 0
file_count = 0
start_time = time.time()

try:
    for sample in tqdm(dataset, desc="Processing texts", unit="texts"):
        text = sample['text']
        newly_encoded_tokens = tokenizer.encode(text, truncation=True, padding=False)
        all_tokens.extend(newly_encoded_tokens)
        token_count += len(newly_encoded_tokens)
        
        if token_count >= tokens_per_file:
            file_path = os.path.join(output_dir, f'tokens_batch_{file_count}.pt')
            torch.save(torch.tensor(all_tokens, dtype=torch.long), file_path)
            print(f"Saved {file_path} with {len(all_tokens)} tokens.")
            file_count += 1
            all_tokens = []  # Clear the list to free memory
            token_count = 0

    # Save any remaining tokens after the loop
    if all_tokens:
        file_path = os.path.join(output_dir, f'tokens_batch_{file_count}.pt')
        torch.save(torch.tensor(all_tokens, dtype=torch.long), file_path)
        print(f"Saved {file_path} with {len(all_tokens)} tokens.")
except Exception as e:
    print(f"An error occurred: {e}")

elapsed_time = time.time() - start_time
print(f"Tokenization completed in {elapsed_time:.2f} seconds.")
print("All tokens saved in 'token_batches' directory.")