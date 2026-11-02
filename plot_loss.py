"""
plot_loss.py

Reads every runs/<name>/loss.csv written by train.py and saves into docs/ :

    loss_<name>.png        train vs validation loss for one run
    loss_comparison.png    all runs on one chart
    results_table.md       a markdown table you can paste into the README

Usage:
    python plot_loss.py
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to files only, no window needed
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
DOCS_DIR = HERE / "docs"


def load_run(run_dir):
    csv_path = run_dir / "loss.csv"
    if not csv_path.exists():
        return None

    steps, train, val = [], [], []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))

    if not steps:
        return None

    config = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    return {"name": run_dir.name, "steps": steps, "train": train, "val": val,
            "config": config}


def plot_single(run):
    steps, train, val = run["steps"], run["train"], run["val"]
    config = run["config"]

    best_i = min(range(len(val)), key=lambda i: val[i])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train, label="train loss", linewidth=2)
    ax.plot(steps, val, label="validation loss", linewidth=2)

    ax.scatter([steps[best_i]], [val[best_i]], color="black", zorder=5)
    ax.annotate(
        f"best val {val[best_i]:.2f} @ step {steps[best_i]}",
        (steps[best_i], val[best_i]),
        textcoords="offset points", xytext=(10, 12), fontsize=9,
    )

    vocab = config.get("vocab_size")
    if vocab:
        ax.axhline(math.log(vocab), color="gray", linestyle=":", linewidth=1,
                   label=f"random guess (ln {vocab} = {math.log(vocab):.2f})")

    title = run["name"]
    if config.get("parameters"):
        title += f"  ({config['parameters']:,} parameters)"
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = DOCS_DIR / f"loss_{run['name']}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_comparison(runs):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, run in enumerate(runs):
        color = colors[i % len(colors)]
        ax.plot(run["steps"], run["train"], linestyle="--", color=color,
                alpha=0.8, label=f"{run['name']} - train")
        ax.plot(run["steps"], run["val"], linestyle="-", color=color,
                linewidth=2, label=f"{run['name']} - val")

    ax.set_title("Train (dashed) vs validation (solid) loss")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = DOCS_DIR / "loss_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_table(runs):
    header = ("| Run | Params | LR | Steps | Embed | Layers | "
              "Best val (step) | Final train | Final val |")
    divider = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, divider]

    for run in runs:
        c = run["config"]
        val = run["val"]
        best_i = min(range(len(val)), key=lambda i: val[i])

        params = f"{c['parameters']:,}" if "parameters" in c else "-"
        lines.append(
            f"| {run['name']} | {params} | {c.get('lr', '-')} | "
            f"{c.get('steps', '-')} | {c.get('embed', '-')} | "
            f"{c.get('layers', '-')} | "
            f"{val[best_i]:.2f} ({run['steps'][best_i]}) | "
            f"{run['train'][-1]:.2f} | {val[-1]:.2f} |"
        )

    return "\n".join(lines)


def main():
    if not RUNS_DIR.exists():
        raise SystemExit(f"No runs folder found at {RUNS_DIR}. Run train.py first.")

    runs = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if run_dir.is_dir():
            run = load_run(run_dir)
            if run:
                runs.append(run)

    if not runs:
        raise SystemExit("No runs with a loss.csv found. Run train.py first.")

    DOCS_DIR.mkdir(exist_ok=True)

    for run in runs:
        print("Saved:", plot_single(run))

    print("Saved:", plot_comparison(runs))

    table = build_table(runs)
    table_path = DOCS_DIR / "results_table.md"
    table_path.write_text(table + "\n", encoding="utf-8")
    print("Saved:", table_path)

    print()
    print(table)


if __name__ == "__main__":
    main()
