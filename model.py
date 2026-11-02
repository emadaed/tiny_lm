"""
model.py

Shared Transformer architecture for the tiny character-level language
model. Both train.py and generate.py import the classes from this one
file, so there is a single source of truth for the architecture.

Why this file exists (fix #6 from the review):
Previously the model classes were copy-pasted into train.py AND
generate.py. If you changed EMBED_SIZE in one file and forgot the
other, load_state_dict() would fail with a confusing shape-mismatch
error, or worse, silently load into the wrong shapes. Now there is
only one definition, driven by a single ModelConfig object that gets
saved inside the checkpoint (see train.py) and reloaded automatically
(see generate.py).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 64
    embed_size: int = 32
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


# ============================================================
# Self-attention head
# ============================================================

class Head(nn.Module):
    """One self-attention head."""

    def __init__(self, config: ModelConfig, head_size: int):
        super().__init__()

        self.key = nn.Linear(config.embed_size, head_size, bias=False)
        self.query = nn.Linear(config.embed_size, head_size, bias=False)
        self.value = nn.Linear(config.embed_size, head_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(k.size(-1))

        # Causal mask: position t cannot attend to positions > t
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)

        return weights @ v


# ============================================================
# Multi-head attention
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, config: ModelConfig, head_size: int):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(config, head_size) for _ in range(config.num_heads)
        ])

        self.projection = nn.Linear(
            config.num_heads * head_size,
            config.embed_size
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        return self.dropout(out)


# ============================================================
# Feed-forward network
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(config.embed_size, 4 * config.embed_size),
            nn.GELU(),
            nn.Linear(4 * config.embed_size, config.embed_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Transformer block
# ============================================================

class Block(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        head_size = config.embed_size // config.num_heads
        assert head_size * config.num_heads == config.embed_size, (
            "embed_size must be divisible by num_heads"
        )

        self.attention = MultiHeadAttention(config, head_size)
        self.feed_forward = FeedForward(config)

        self.norm1 = nn.LayerNorm(config.embed_size)
        self.norm2 = nn.LayerNorm(config.embed_size)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


# ============================================================
# Full tiny language model
# ============================================================

class TinyLanguageModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.position_embedding = nn.Embedding(config.block_size, config.embed_size)
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_layers)]
        )

        self.final_norm = nn.LayerNorm(config.embed_size)
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding(idx)

        positions = torch.arange(T, device=idx.device)
        position_embeddings = self.position_embedding(positions)

        x = self.embedding_dropout(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressively sample max_new_tokens characters after idx."""
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        if was_training:
            self.train()

        return idx
