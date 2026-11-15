"""Reproducible CPU-only toy comparisons, not evidence of training improvements."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from zeta_scaling_toolkit import (
    AsymptoticLRScheduler,
    NonCollapsingAttention,
    make_corrected_linear,
)

SEED = 42
STEPS = 300
BATCH_SIZE = 32
SEQ_LEN = 16
VOCAB_SIZE = 32
D_MODEL = 32
HEADS = 4
LR = 1e-3


class TransformerBlock(nn.Module):
    def __init__(self, corrected: bool):
        super().__init__()
        self.attention = nn.MultiheadAttention(D_MODEL, HEADS, batch_first=True)
        self.corrected_attention = NonCollapsingAttention(D_MODEL // HEADS) if corrected else None
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.feed_forward = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 2), nn.ReLU(), nn.Linear(D_MODEL * 2, D_MODEL)
        )

    def forward(self, x):
        normalized = self.norm1(x)
        if self.corrected_attention is None:
            attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        else:
            qkv = F.linear(
                normalized, self.attention.in_proj_weight, self.attention.in_proj_bias
            ).chunk(3, dim=-1)
            q, k, v = (
                t.reshape(t.shape[0], t.shape[1], HEADS, D_MODEL // HEADS).transpose(1, 2)
                for t in qkv
            )
            attended = self.corrected_attention(q, k, v)
            attended = attended.transpose(1, 2).contiguous().view_as(normalized)
            attended = self.attention.out_proj(attended)
        x = x + attended
        return x + self.feed_forward(self.norm2(x))


class CopyTransformer(nn.Module):
    def __init__(self, corrected: bool):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.position = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.Sequential(*(TransformerBlock(corrected) for _ in range(2)))
        self.output = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embedding(tokens) + self.position(positions)
        return self.output(self.blocks(x))


def train_transformer(corrected, batches):
    torch.manual_seed(SEED)
    model = CopyTransformer(corrected)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = (
        AsymptoticLRScheduler(optimizer)
        if corrected
        else CosineAnnealingLR(optimizer, T_max=len(batches))
    )
    losses = []
    model.train()
    for tokens in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tokens.reshape(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return losses


class DeepMLP(nn.Module):
    def __init__(self, corrected: bool):
        super().__init__()
        linear = make_corrected_linear if corrected else nn.Linear
        self.layers = nn.ModuleList(linear(D_MODEL, D_MODEL) for _ in range(30))
        self.output = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x), x


def train_mlp(corrected, batches, evaluation_inputs):
    torch.manual_seed(SEED)
    model = DeepMLP(corrected)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for inputs in batches:
        optimizer.zero_grad(set_to_none=True)
        outputs, _ = model(inputs)
        loss = F.mse_loss(outputs, inputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, activations = model(evaluation_inputs)
    return activations.std().item()


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(SEED)
    token_batches = torch.randint(VOCAB_SIZE, (STEPS, BATCH_SIZE, SEQ_LEN), generator=generator)
    mlp_batches = torch.randn(STEPS, BATCH_SIZE, D_MODEL, generator=generator)
    mlp_evaluation_inputs = torch.randn(BATCH_SIZE, D_MODEL, generator=generator)

    standard_losses = train_transformer(False, token_batches)
    corrected_losses = train_transformer(True, token_batches)
    print(f"Standard attention + cosine final loss: {standard_losses[-1]:.6f}")
    print(f"Corrected attention + asymptotic final loss: {corrected_losses[-1]:.6f}")

    plt.plot(range(1, STEPS + 1), standard_losses, label="Standard attention + cosine")
    plt.plot(range(1, STEPS + 1), corrected_losses, label="Corrected attention + asymptotic")
    plt.xlabel("Training step")
    plt.ylabel("Token cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curves.png")
    plt.close()
    print("Saved loss_curves.png")

    standard_std = train_mlp(False, mlp_batches, mlp_evaluation_inputs)
    corrected_std = train_mlp(True, mlp_batches, mlp_evaluation_inputs)
    print(f"Standard linear final activation std: {standard_std:.6f}")
    print(f"Corrected linear final activation std: {corrected_std:.6f}")


if __name__ == "__main__":
    main()
