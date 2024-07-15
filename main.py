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

BATCH_SIZE = 16
ACCUMULATION_STEPS = 4  # Accumulate gradients over this many steps

# Set num_workers to the number of logical processors
num_workers = os.cpu_count()

def save_model_and_optimizer(model, optimizer, epoch):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, f'checkpoints/gpt2_epoch_{epoch + 1}.pt')

def load_model_and_optimizer(model, optimizer):
    if os.path.exists('checkpoints'):
        checkpoint_files = [f for f in os.listdir('checkpoints') if f.startswith('gpt2_epoch_')]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint = torch.load(f'checkpoints/{latest_checkpoint}')
            if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'epoch' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                print(f"Model and optimizer loaded from checkpoint '{latest_checkpoint}'")
                return start_epoch
            else:
                print("Checkpoint file is missing required keys")
    return 0

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
    global counter0
    start_epoch = load_model_and_optimizer(model, optimizer)
    scaler = GradScaler()  # Initialize scaler

    model.train()  # Set model to training mode
    start_time = time.time()  # Start time for the entire training

    for epoch in range(start_epoch, start_epoch + 5):  # Adjust as needed for more epochs
        batch_time = time.time()  # Start time for batch processing
        optimizer.zero_grad()  # Reset gradients
        for batch_idx, (input_ids, labels) in enumerate(loader, 1):
            input_ids, labels = input_ids.to(device), labels.to(device)

            # Starts the autocast context
            with autocast():
                logits = model(input_ids)
                loss = nn.CrossEntropyLoss()(logits.view(-1, config.vocab_size), labels.view(-1))

            # Scales loss and calls backward to create scaled gradients
            scaler.scale(loss).backward()

            if batch_idx % ACCUMULATION_STEPS == 0:
                # Unscales the gradients and step optimizer
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Reset gradients

            # Calculate tokens per second
            tokens_per_batch = input_ids.size(0) * input_ids.size(1)  # batch size * sequence length
            elapsed_time = time.time() - batch_time

            # Reset batch timer for accurate measurement of next batch processing time
            batch_time = time.time()

            if batch_idx % 200 == 0:
                tokens_per_second = tokens_per_batch / elapsed_time if elapsed_time > 0 else 0
                time_since_start = time.time() - start_time  # Calculate time since training started
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Tokens/sec: {tokens_per_second:.2f}, Time Elapsed: {time_since_start:.2f} sec")

            # Save model and print sample every 1000 iterations
            if batch_idx % 1000 == 0:
                save_model_and_optimizer(model, optimizer, epoch)
                generate_text("I am a")

        counter0 += 1

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