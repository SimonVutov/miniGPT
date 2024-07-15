import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import IterableDataset

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int

class TokenizedDataset(IterableDataset):
    def __init__(self, token_files, block_size, batch_size):
        super().__init__()
        self.token_files = token_files
        self.block_size = block_size
        self.batch_size = batch_size

    def read_tokens(self, file_path):
        tokens = torch.load(file_path)
        for i in range(0, len(tokens), self.block_size + 1):
            yield tokens[i:i+self.block_size+1]

    def __iter__(self):
        for file_path in self.token_files:
            tokens = torch.load(file_path)
            total_tokens = len(tokens)
            total_batches = (total_tokens // (self.block_size + 1)) // self.batch_size
            print("Path: ", file_path, "Total tokens in file:", total_tokens, "Total batches in file will be:", total_batches)
            token_generator = self.read_tokens(file_path)
            for tokens in token_generator:
                if len(tokens) == self.block_size + 1:
                    input_ids = tokens[:-1].clone().detach().to(dtype=torch.long)
                    labels = tokens[1:].clone().detach().to(dtype=torch.long)
                    yield input_ids, labels

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(embed_dim=config.n_embd, num_heads=config.n_head)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding[:, :x.size(1), :]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
