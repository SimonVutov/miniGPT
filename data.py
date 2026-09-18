"""Explicit, bounded text preparation; importing this module does no work."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import torch

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare(text, output, validation_fraction=0.1, shard_tokens=100_000, tokenizer="byte"):
    if not 0 < validation_fraction < 1 or shard_tokens < 2:
        raise ValueError("Invalid validation fraction or shard size")
    if tokenizer == "byte":
        tokens = list(text.encode("utf-8"))
        vocab_size = 256
    elif tokenizer == "gpt2":
        from transformers import GPT2TokenizerFast
        encoder = GPT2TokenizerFast.from_pretrained("gpt2")
        tokens = encoder.encode(text, add_special_tokens=False)
        vocab_size = len(encoder)
    else:
        raise ValueError("Unsupported tokenizer")
    split = int(len(tokens)*(1-validation_fraction))
    if min(split, len(tokens)-split) < 2:
        raise ValueError("Text is too short for train/validation splits")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty to avoid mixing datasets")
    metadata = {"tokenizer": tokenizer, "vocab_size": vocab_size,
                "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "train_tokens": split, "validation_tokens": len(tokens)-split}
    for name, values in [("train", tokens[:split]), ("val", tokens[split:])]:
        directory = output/name
        directory.mkdir()
        for i, start in enumerate(range(0, len(values), shard_tokens)):
            torch.save(torch.tensor(values[start:start+shard_tokens], dtype=torch.long), directory/f"tokens_{i:05d}.pt")
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare separate training and validation token shards")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Local UTF-8 text file")
    source.add_argument("--shakespeare", action="store_true", help="Download the ~1 MB Tiny Shakespeare corpus")
    source.add_argument("--fineweb", action="store_true", help="Stream a bounded FineWeb sample; needs requirements-data.txt")
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument("--max-characters", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=Path("token_batches"))
    parser.add_argument("--tokenizer", choices=["byte", "gpt2"], default="byte")
    parser.add_argument("--shard-tokens", type=int, default=100_000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    try:
        if args.max_documents <= 0 or args.max_characters <= 0:
            raise ValueError("Download limits must be positive")
        if args.input:
            text = args.input.read_text(encoding="utf-8")
        elif args.shakespeare:
            with urllib.request.urlopen(SHAKESPEARE_URL, timeout=60) as response:
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ValueError("Unexpectedly large Shakespeare download")
                text = raw.decode("utf-8")
        else:
            from datasets import load_dataset
            documents, remaining = [], args.max_characters
            dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
            for i, item in enumerate(dataset):
                piece = item["text"][:remaining]
                documents.append(piece)
                remaining -= len(piece)+2
                if remaining <= 0 or i+1 >= args.max_documents:
                    break
            text = "\n\n".join(documents)[:args.max_characters]
        print(json.dumps(prepare(text, args.output, args.validation_fraction, args.shard_tokens, args.tokenizer)))
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
