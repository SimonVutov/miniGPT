import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
from pickle import UnpicklingError
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import GPT, GPTConfig, TokenizedDataset


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("Mixed precision currently requires CUDA; use fp32 on CPU/MPS")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16")
    return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def train_update(model, optimizer, batches, device, scaler, precision="fp32", clip=1.0):
    if not batches:
        raise ValueError("An optimizer update needs at least one microbatch")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count = sum(y.numel() for _, y in batches)
    if count == 0:
        raise ValueError("Empty training batch")
    total_loss = 0.0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        with autocast_context(device, precision):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                                                y.reshape(-1), reduction="sum") / count
        if not torch.isfinite(loss):
            raise ValueError("Training produced a non-finite loss")
        scaler.scale(loss).backward()
        total_loss += loss.detach().float().item()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=not scaler.is_enabled())
    scaler.step(optimizer)
    scaler.update()
    return total_loss, count


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            logits = model(x.to(device))
            total += nn.functional.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                                                 y.to(device).reshape(-1), reduction="sum").item()
            count += y.numel()
    finally:
        model.train(was_training)
    if not count:
        raise ValueError("Dataset has no complete context windows; reduce block size or prepare more text")
    loss = total/count
    if not math.isfinite(loss):
        raise ValueError("Non-finite validation loss")
    return loss


def dataset_signature(directory):
    digest = hashlib.sha256()
    paths = [directory/"metadata.json"] + sorted(directory.glob("train/*.pt")) + sorted(directory.glob("val/*.pt"))
    for path in paths:
        digest.update(path.relative_to(directory).as_posix().encode())
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024*1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path, model, optimizer, scaler, state):
    checkpoint = {"format_version": 1, "config": asdict(model.config), "model": model.state_dict(),
                  "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                  "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                  "mps_rng": torch.mps.get_rng_state() if next(model.parameters()).device.type == "mps" else None,
                  "state": state}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint; old checkpoints used an incompatible attention layout")
    model = GPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return torch.device(name)


def generate_text(checkpoint, prompt, *, max_new_tokens=100, device="cpu", temperature=0.8, top_k=None, seed=42):
    device = choose_device(device)
    model, saved = load_checkpoint(checkpoint, device)
    tokenizer = saved["state"]["tokenizer"]
    if tokenizer == "byte":
        indices = list(prompt.encode("utf-8"))
        decode = lambda values: bytes(values).decode("utf-8", errors="replace")
    elif tokenizer == "gpt2":
        from transformers import GPT2TokenizerFast
        encoder = GPT2TokenizerFast.from_pretrained("gpt2")
        indices, decode = encoder.encode(prompt), encoder.decode
    else:
        raise ValueError("Unknown checkpoint tokenizer")
    if not indices:
        raise ValueError("Prompt must not be empty")
    torch.manual_seed(seed)
    tokens = torch.tensor([indices], dtype=torch.long, device=device)
    generated = model.generate(tokens, max_new_tokens, temperature, top_k)
    return decode(generated[0].tolist())


def train(args):
    device = choose_device(args.device)
    autocast_context(device, args.precision)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    metadata = json.loads((args.data/"metadata.json").read_text())
    if metadata["tokenizer"] not in ("byte", "gpt2") or metadata["vocab_size"] != (256 if metadata["tokenizer"] == "byte" else 50257):
        raise ValueError("Invalid tokenizer metadata")
    signature = dataset_signature(args.data)
    settings = {key: getattr(args, key) for key in ("batch_size", "accumulation", "workers", "precision", "learning_rate", "seed")}
    state = {"step": 0, "epoch": 0, "batches_seen": 0, "tokens_seen": 0, "training_seconds": 0.0,
             "tokenizer": metadata["tokenizer"], "dataset_signature": signature, "settings": settings, "history": []}
    checkpoint = None
    if args.resume:
        model, checkpoint = load_checkpoint(args.resume, device)
        state = checkpoint["state"]
        if state["dataset_signature"] != signature or state["settings"] != settings:
            raise ValueError("Resume requires the same data, batch size, accumulation, workers, precision, learning rate, and seed")
        if args.steps <= state["step"]:
            raise ValueError("--steps must exceed the saved total optimizer step count")
    else:
        if (args.output/"last.pt").exists():
            raise ValueError("Output already contains a checkpoint; use --resume or a new directory")
        model = GPT(GPTConfig(args.block_size, metadata["vocab_size"], args.layers, args.heads, args.embedding, args.dropout)).to(device)
    config = model.config
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if device.type == "mps" and checkpoint.get("mps_rng") is not None:
            torch.mps.set_rng_state(checkpoint["mps_rng"])
    dataset = TokenizedDataset(sorted((args.data/"train").glob("*.pt")), config.block_size, vocab_size=config.vocab_size)
    validation = TokenizedDataset(sorted((args.data/"val").glob("*.pt")), config.block_size, vocab_size=config.vocab_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(validation, batch_size=args.batch_size, generator=torch.Generator().manual_seed(args.seed))
    args.output.mkdir(parents=True, exist_ok=True)
    if "initial_validation_loss" not in state:
        state["initial_validation_loss"] = evaluate(model, val_loader, device, args.eval_batches)
    initial_validation = state["initial_validation_loss"]
    print(json.dumps({"device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
                      "initial_validation_loss": initial_validation}), flush=True)
    while state["step"] < args.steps:
        iterator = iter(loader)
        for _ in range(state["batches_seen"]):
            if next(iterator, None) is None:
                raise ValueError("Saved batch offset exceeds dataset")
        updates = 0
        while state["step"] < args.steps:
            batches = list(itertools.islice(iterator, args.accumulation))
            if not batches:
                break
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            loss, tokens = train_update(model, optimizer, batches, device, scaler, args.precision)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if device.type == "mps":
                torch.mps.synchronize()
            state["training_seconds"] += time.perf_counter()-start
            state["step"] += 1
            state["batches_seen"] += len(batches)
            state["tokens_seen"] += tokens
            updates += 1
            if state["step"] % args.eval_every == 0 or state["step"] == args.steps:
                validation_loss = evaluate(model, val_loader, device, args.eval_batches)
                row = {"step": state["step"], "train_loss": loss, "validation_loss": validation_loss,
                       "validation_perplexity": math.exp(validation_loss),
                       "tokens_per_second": state["tokens_seen"]/state["training_seconds"]}
                state["history"].append(row)
                print(json.dumps(row), flush=True)
                save_checkpoint(args.output/"last.pt", model, optimizer, scaler, state)
        if not updates and state["batches_seen"] == 0:
            raise ValueError("Training data has no complete context windows")
        if state["step"] < args.steps:
            state["epoch"] += 1
            state["batches_seen"] = 0
    report = {**state, "config": asdict(config), "parameters": sum(p.numel() for p in model.parameters()),
              "torch": str(torch.__version__), "python": platform.python_version(), "platform": platform.platform(),
              "device": str(device), "eval_batches": args.eval_batches}
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train or sample a small causal transformer")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, default=Path("token_batches"))
    training.add_argument("--output", type=Path, default=Path("runs/demo"))
    training.add_argument("--resume", type=Path)
    training.add_argument("--steps", type=int, default=200)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--accumulation", type=int, default=1)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--threads", type=int, default=4)
    training.add_argument("--block-size", type=int, default=128)
    training.add_argument("--layers", type=int, default=2)
    training.add_argument("--heads", type=int, default=4)
    training.add_argument("--embedding", type=int, default=128)
    training.add_argument("--dropout", type=float, default=0.1)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    training.add_argument("--eval-every", type=int, default=50)
    training.add_argument("--eval-batches", type=int, default=0)
    generation = commands.add_parser("generate")
    generation.add_argument("--checkpoint", type=Path, required=True)
    generation.add_argument("--prompt", default="First Citizen:")
    generation.add_argument("--tokens", type=int, default=100)
    generation.add_argument("--temperature", type=float, default=0.8)
    generation.add_argument("--top-k", type=int)
    for subparser in (training, generation):
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    try:
        if args.seed < 0:
            raise ValueError("Seed must be nonnegative")
        if args.command == "train":
            if min(args.steps, args.batch_size, args.accumulation, args.threads, args.eval_every) <= 0 or min(args.workers, args.eval_batches) < 0:
                raise ValueError("Invalid training counts")
            if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
                raise ValueError("Learning rate must be finite and positive")
            train(args)
        else:
            print(generate_text(args.checkpoint, args.prompt, max_new_tokens=args.tokens, device=args.device,
                                temperature=args.temperature, top_k=args.top_k, seed=args.seed))
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, UnpicklingError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
