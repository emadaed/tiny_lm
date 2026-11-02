"""
train.py  (Step 3: experiment runner)

Trains the tiny character-level Transformer from model.py and saves
everything about the run into runs/<name>/ :

    loss.csv      step, train_loss, val_loss  (one row per evaluation)
    config.json   hyperparameters + results (params, best/final loss, time)
    best.pt       checkpoint with the LOWEST validation loss
    final.pt      checkpoint at the last step

The best checkpoint is also copied to tiny_lm.pt so generate.py keeps
working without any changes.

Usage:
    python train.py --name exp0_baseline --lr 3e-4 --steps 3000 --embed 32 --layers 2
    python train.py --name exp1_overfit  --lr 1e-3 --steps 5000 --embed 64 --layers 4
"""

import argparse
import csv
import json
import re
import shutil
import time
from pathlib import Path

import torch

from model import ModelConfig, TinyLanguageModel


HERE = Path(__file__).resolve().parent
DEVICE = "cpu"


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train the tiny character-level LM.")

    p.add_argument("--name", required=True,
                   help="Run name, e.g. exp0_baseline (letters, digits, _ and - only)")
    p.add_argument("--data", default="data.txt", help="Training text file")

    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--steps", type=int, default=3000, help="Training steps")
    p.add_argument("--embed", type=int, default=32, help="Embedding size")
    p.add_argument("--layers", type=int, default=2, help="Transformer blocks")
    p.add_argument("--heads", type=int, default=4, help="Attention heads")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--block", type=int, default=64, help="Context length")

    p.add_argument("--eval-interval", type=int, default=100,
                   help="Evaluate every N steps")
    p.add_argument("--eval-iters", type=int, default=50,
                   help="Batches averaged per evaluation")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow re-using an existing run folder")

    return p.parse_args()


# ============================================================
# Helpers
# ============================================================

def get_batch(source, batch_size, block_size):
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args):
    out = {}
    model.eval()

    for split_name, source in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(source, args.batch, args.block)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()

    model.train()
    return out


def save_checkpoint(path, model, config, stoi, itos):
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
        },
        path,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_\-]+", args.name):
        raise SystemExit("--name may only contain letters, digits, _ and -")

    run_dir = HERE / "runs" / args.name
    if run_dir.exists() and not args.overwrite:
        raise SystemExit(
            f"Run folder already exists: {run_dir}\n"
            f"Pick a new --name, or add --overwrite to replace it."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load text and build vocabulary --------------------
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = HERE / data_path

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # ---- Train / validation split --------------------------
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    if len(train_data) <= args.block + 1 or len(val_data) <= args.block + 1:
        raise SystemExit(
            f"{data_path.name} is too small for --block {args.block}.\n"
            f"Total characters: {len(data)} "
            f"({len(train_data)} train / {len(val_data)} val).\n"
            f"Add more text, or lower --block."
        )

    # ---- Build model ---------------------------------------
    torch.manual_seed(args.seed)

    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=args.block,
        embed_size=args.embed,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
    )

    model = TinyLanguageModel(config).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print()
    print(f"Run:          {args.name}")
    print(f"Data:         {data_path.name} ({len(text)} characters)")
    print(f"Vocabulary:   {vocab_size}")
    print(f"Parameters:   {num_params}")
    print(f"Settings:     lr={args.lr}  steps={args.steps}  embed={args.embed}  "
          f"layers={args.layers}  heads={args.heads}  dropout={args.dropout}")
    print()

    # ---- Training loop -------------------------------------
    best_val = float("inf")
    best_step = -1
    last = {"train": float("nan"), "val": float("nan")}
    start_time = time.time()

    with open(run_dir / "loss.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "train_loss", "val_loss"])

        for step in range(args.steps):

            x, y = get_batch(train_data, args.batch, args.block)
            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.eval_interval == 0 or step == args.steps - 1:
                last = estimate_loss(model, train_data, val_data, args)

                writer.writerow([step, f"{last['train']:.4f}", f"{last['val']:.4f}"])
                csv_file.flush()

                marker = ""
                if last["val"] < best_val:
                    best_val = last["val"]
                    best_step = step
                    save_checkpoint(run_dir / "best.pt", model, config, stoi, itos)
                    marker = "  <- best so far"

                print(
                    f"step {step:5d} | "
                    f"train loss {last['train']:.4f} | "
                    f"val loss {last['val']:.4f}{marker}"
                )

    elapsed = time.time() - start_time

    # ---- Save final checkpoint, default checkpoint, config --
    save_checkpoint(run_dir / "final.pt", model, config, stoi, itos)
    shutil.copyfile(run_dir / "best.pt", HERE / "tiny_lm.pt")

    summary = {
        "name": args.name,
        "data_file": data_path.name,
        "data_characters": len(text),
        "vocab_size": vocab_size,
        "parameters": num_params,
        "lr": args.lr,
        "steps": args.steps,
        "embed": args.embed,
        "layers": args.layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "batch": args.batch,
        "block": args.block,
        "seed": args.seed,
        "best_val_loss": round(best_val, 4),
        "best_step": best_step,
        "final_train_loss": round(last["train"], 4),
        "final_val_loss": round(last["val"], 4),
        "seconds": round(elapsed, 1),
    }

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---- Final report --------------------------------------
    print()
    print("Training complete.")
    print(f"Best val loss:   {best_val:.4f} at step {best_step}")
    print(f"Final val loss:  {last['val']:.4f} "
          f"({last['val'] - best_val:+.4f} vs best)")
    print(f"Run folder:      {run_dir}")
    print(f"tiny_lm.pt now holds this run's BEST checkpoint (step {best_step}).")


if __name__ == "__main__":
    main()
