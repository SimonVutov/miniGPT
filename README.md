# miniGPT

A small decoder-only transformer and training loop built with PyTorch. Includes
causal attention, token-weighted gradient accumulation, held-out evaluation,
checkpoint resumption, and autoregressive text generation. Trains from random
initialization; it does not load pretrained GPT-2 weights.

## Quick start

Requires Python 3.10+; CPU is sufficient. Install the appropriate
[PyTorch build](https://pytorch.org/get-started/locally/) for CUDA if needed.

```sh
git clone --depth 1 --branch codex/portfolio-polish https://github.com/SimonVutov/miniGPT.git
cd miniGPT
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python data.py --input examples/tiny.txt --output runs/tiny-data
python main.py train --data runs/tiny-data --output runs/tiny-model --steps 200 --block-size 16 --embedding 32 --heads 2 --layers 1 --device cpu
python main.py generate --checkpoint runs/tiny-model/last.pt --prompt "The " --tokens 80 --top-k 20
```

The command above selects the corrected review branch; after merging it into
`main`, the `--branch` option can be omitted.
The local sample is synthetic and verifies the workflow, not language quality.
Tests and this quick start require no dataset or tokenizer downloads.
`miniGPT.ipynb` runs the same code; open it from this directory using a Jupyter
kernel with these dependencies installed.

## Reproduce a language-model run

The explicit download below uses the ~1 MB
[Tiny Shakespeare corpus from char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).
Byte tokenization needs no pretrained tokenizer. The last 10% of bytes form the
validation split; training and validation never share token windows.

```sh
python data.py --shakespeare --output token_batches/shakespeare
python main.py train --data token_batches/shakespeare --output runs/shakespeare --steps 1000 --device cpu
python main.py generate --checkpoint runs/shakespeare/last.pt --prompt "First Citizen:" --tokens 160 --temperature 0.8 --top-k 30
```

Measured locally on Apple M5 CPU, PyTorch 2.14, four threads, seed 42:

| Metric | Result |
| --- | ---: |
| Parameters | 478,720 |
| Architecture | 2 layers, 4 heads, width 128, context 128 |
| Training / validation bytes | 1,003,854 / 111,540 |
| Optimizer steps | 1,000 |
| Initial / final validation loss | 5.532 / 2.167 |
| Final byte-level perplexity | 8.735 |
| Training throughput | ~91,000 bytes/s |

Validation uses every complete validation window. Perplexity is per byte and is
not directly comparable to GPT-2-token perplexity. Throughput measures training
updates, including transfers/backpropagation/optimizer work, excluding data
loading, evaluation, and checkpoint writes. It is hardware-specific. This tiny
model produces imperfect text; no conversational or state-of-the-art claim is made.
[Recorded run](examples/shakespeare_metrics.json). Settings, source hash, runtime versions, losses, and timings are written to
`metrics.json`. Model/optimizer/scaler state and data position are saved in `last.pt`.

## Resume, configure, and use your own data

```sh
python main.py train --data token_batches/shakespeare --output runs/shakespeare --resume runs/shakespeare/last.pt --steps 1200
python data.py --input your_text.txt --output token_batches/custom
python main.py train --help
python main.py generate --help
```

`--steps` is the total target optimizer-step count, including earlier steps.
Resume restores architecture, optimizer/scaler, data offset, and random states;
repeat the same batch size, accumulation, worker count, precision, learning rate,
and seed. Changed data is rejected. Exact CPU resumption is regression-tested;
results need not be identical across devices or PyTorch versions. Legacy
checkpoints are incompatible with the corrected causal architecture.

Use `--accumulation 4` for four microbatches per update. The final partial group
is flushed with correct token weighting. `--workers N` partitions files among
workers; a single shard cannot use multiple workers effectively. Short shard
tails that cannot form a complete context window are discarded.

Devices: `--device cpu`, `cuda`, or `mps`; `auto` selects CUDA when available,
otherwise CPU. CUDA supports `--precision fp16` with gradient scaling or `bf16`
on capable hardware; CPU/MPS use fp32. CUDA paths require hardware testing and
are skipped in the CPU test suite.

Optional GPT-2 tokenization and bounded FineWeb streaming:

```sh
python -m pip install -r requirements-data.txt
python data.py --fineweb --max-documents 1000 --max-characters 1000000 --tokenizer gpt2 --output token_batches/fineweb
```

FineWeb/GPT-2 options download third-party data/tokenizer files explicitly. Use
their respective terms; the repository does not bundle datasets, checkpoints,
or virtual environments. Prepare into an empty directory to avoid mixing runs.

## Tests and code

- `model.py`: architecture, generation, worker-sharded token dataset.
- `main.py`: training, validation, checkpoints, command-line interface.
- `data.py`: local/downloaded text preparation and disjoint splits.
- `tests/`: causal/batch isolation, accumulation equivalence, worker coverage,
  checkpoint resumption, generation, input validation, and optional CUDA checks.

GitHub Actions runs CPU checks on Linux, macOS, and Windows. Historical loss logs
from the earlier noncausal implementation are not valid benchmark evidence.
