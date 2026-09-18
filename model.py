from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 256
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.block_size, self.vocab_size, self.n_layer, self.n_head, self.n_embd) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("Embedding dimension must be divisible by head count")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")


class TokenizedDataset(IterableDataset):
    def __init__(self, token_files, block_size, batch_size=None, vocab_size=None):
        super().__init__()
        self.token_files = [Path(path) for path in token_files]
        if not self.token_files or block_size <= 0:
            raise ValueError("Need token files and a positive block size")
        if any(not path.is_file() for path in self.token_files):
            raise FileNotFoundError("A token shard is missing")
        self.block_size = block_size
        self.vocab_size = vocab_size

    def read_tokens(self, path):
        tokens = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1 or tokens.dtype != torch.long:
            raise ValueError(f"Expected a 1D int64 token tensor: {path}")
        if tokens.numel() and (tokens.min() < 0 or
                              (self.vocab_size is not None and tokens.max() >= self.vocab_size)):
            raise ValueError(f"Token outside vocabulary: {path}")
        for start in range(0, len(tokens)-self.block_size, self.block_size):
            yield tokens[start:start+self.block_size+1]

    def __iter__(self):
        worker = get_worker_info()
        files = self.token_files if worker is None else self.token_files[worker.id::worker.num_workers]
        for path in files:
            for tokens in self.read_tokens(path):
                yield tokens[:-1], tokens[1:]


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head,
                                          dropout=config.dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4*config.n_embd), nn.GELU(),
                                 nn.Linear(4*config.n_embd, config.n_embd), nn.Dropout(config.dropout))
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("causal_mask", torch.ones(config.block_size, config.block_size,
                                                       dtype=torch.bool).triu(1), persistent=False)

    def forward(self, x):
        normalized = self.ln1(x)
        attention = self.attn(normalized, normalized, normalized, need_weights=False,
                              attn_mask=self.causal_mask[:x.size(1), :x.size(1)])[0]
        x = x + self.dropout(attention)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Parameter(torch.empty(1, config.block_size, config.n_embd))
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._initialize)
        nn.init.normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.size(0) == 0 or not 0 < tokens.size(1) <= self.config.block_size:
            raise ValueError("Expected nonempty [batch, time] tokens within the context window")
        if tokens.dtype != torch.long:
            raise ValueError("Token indices must be int64")
        x = self.token_embedding(tokens) + self.position_embedding[:, :tokens.size(1)]
        return self.head(self.ln_f(self.blocks(x)))

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=100, temperature=1.0, top_k=None):
        if max_new_tokens < 0 or temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
            raise ValueError("Invalid generation length or temperature")
        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k must be within the vocabulary")
        if tokens.ndim != 2 or tokens.size(0) == 0 or tokens.size(1) == 0:
            raise ValueError("Prompt must contain at least one token")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits = self(tokens[:, -self.config.block_size:])[:, -1] / temperature
                if top_k is not None:
                    threshold = torch.topk(logits, top_k).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                tokens = torch.cat((tokens, next_token), dim=1)
            return tokens
        finally:
            self.train(was_training)
