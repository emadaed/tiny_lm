"""
generate.py

Loads a trained tiny_lm.pt checkpoint (produced by train.py) and
generates text from a prompt.

Run:
    python generate.py
"""

import torch

from model import TinyLanguageModel


DEVICE = "cpu"


checkpoint = torch.load("tiny_lm.pt", map_location=DEVICE, weights_only=False)

config = checkpoint["config"]  # fix #3: hyperparameters loaded, not hardcoded
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

model = TinyLanguageModel(config).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()


def encode(text):
    """Fix #4: warn instead of silently dropping unknown characters."""
    tokens = []
    skipped = set()

    for c in text:
        if c in stoi:
            tokens.append(stoi[c])
        else:
            skipped.add(c)

    if skipped:
        print(
            f"Warning: these characters aren't in the training vocabulary "
            f"and were skipped: {sorted(skipped)!r}"
        )

    return tokens


def decode(tokens):
    return "".join(itos[int(t)] for t in tokens)


def generate(prompt, max_new_tokens=200):
    tokens = encode(prompt)

    if not tokens:
        raise ValueError(
            "Prompt contains no characters from the training vocabulary. "
            "Try a prompt using only characters that appear in data.txt."
        )

    idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    idx = model.generate(idx, max_new_tokens)

    return decode(idx[0].tolist())


if __name__ == "__main__":
    prompt = input("Prompt: ")
    print()
    print(generate(prompt))
