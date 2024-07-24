import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import time
from model import GPT, GPTConfig, TokenizedDataset
from torch import nn
from torch.cuda.amp import GradScaler, autocast

BATCH_SIZE = 32  # 16
ACCUMULATION_STEPS = 4  # 4 Accumulate gradients over this many steps
PRINT_EVERY = 200  # Print training loss every this many batches

# Set num_workers to the number of logical processors
num_workers = os.cpu_count()

def save_model_and_optimizer(model, optimizer, epoch, file_idx):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f'checkpoints/gpt2_epoch_{epoch + 1}_file_{file_idx + 1}.pt'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'file_idx': file_idx
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_model_and_optimizer(model, optimizer):
    start_epoch = 0
    if os.path.exists('checkpoints'):
        checkpoint_files = [f for f in os.listdir('checkpoints') if f.startswith('gpt2_epoch_')]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: (int(x.split('_')[2]), int(x.split('_')[4].split('.')[0])))
            checkpoint = torch.load(f'checkpoints/{latest_checkpoint}')
            if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Model and optimizer loaded from checkpoint '{latest_checkpoint}'")
                return start_epoch
            else:
                print("Checkpoint file is missing required keys")
    return start_epoch

def generate_text(inputText):
    model.eval()  # Set model to evaluation mode
    input_ids = tokenizer.encode(inputText, return_tensors="pt").to(device)
    for _ in range(40):
        with autocast():  # Mixed precision inference
            logits = model(input_ids)
        next_token = torch.multinomial(nn.functional.softmax(logits[:, -1, :], dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    generated_text = tokenizer.decode(input_ids[0].tolist())
    print(f"Generated text: {generated_text}")

def train(model, loader, optimizer):
    start_epoch = load_model_and_optimizer(model, optimizer)
    print(f"Starting training from epoch {start_epoch + 1}")
    scaler = GradScaler()
    model.train()
    start_time = time.time()
    last_print_time = time.time()  # Initialize last print time

    # Open the loss file in append mode
    with open('loss.txt', 'a') as loss_file:
        for epoch in range(start_epoch, start_epoch + 5):
            for batch_idx, (input_ids, labels) in enumerate(loader, start=1):
                input_ids, labels = input_ids.to(device), labels.to(device)
                optimizer.zero_grad()

                with autocast():
                    logits = model(input_ids)
                    loss = nn.CrossEntropyLoss()(logits.view(-1, config.vocab_size), labels.view(-1))

                scaler.scale(loss).backward()

                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                tokens_per_batch = input_ids.numel()  # Total number of tokens in the batch
                current_time = time.time()
                elapsed_time = current_time - last_print_time  # Time since last print
                tokens_per_second = PRINT_EVERY * tokens_per_batch / elapsed_time if elapsed_time > 0 else 0

                if batch_idx % PRINT_EVERY == 0:
                    time_since_start = current_time - start_time
                    print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    print(f"Tokens in batch: {tokens_per_batch}, Elapsed time: {elapsed_time:.4f} sec, Tokens/sec: {tokens_per_second:.2f}")
                    print(f"Time Elapsed since start: {time_since_start:.2f} sec")
                    last_print_time = current_time  # Update last print time

                    # Save the loss to the file
                    loss_file.write(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}\n")

                if batch_idx % 1000 == 0:
                    save_model_and_optimizer(model, optimizer, epoch, 0)
                    generate_text("I am a")

    print("Training complete.")

if __name__ == "__main__":
    # Define the global counter in the main script and pass it to the dataset class
    counter0 = 0

    # Initialization and data loading
    token_files = [os.path.join('token_batches', f) for f in os.listdir('token_batches') if f.startswith('tokens_batch_')]
    config = GPTConfig(block_size=128, vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
    dataset = TokenizedDataset(token_files, config.block_size, batch_size=BATCH_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Setup model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    model = GPT(config).to(device)
    optimizer = AdamW(model.parameters(), lr=0.0003)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    train(model, loader, optimizer)
