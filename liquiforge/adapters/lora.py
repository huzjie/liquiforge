# -*- coding: utf-8 -*-
"""Low-Rank Adaptation (LoRA): freeze the base weights, learn A@B residuals.

This is the engineering trick behind "one base backbone -> six fine-tuned models":
you keep the giant shared trunk frozen and only train a tiny low-rank delta per
task, so each variant costs a fraction of the full fine-tune.
"""
from __future__ import annotations

from ..utils.rng import stable_matrix, stable_float


class LoRALinear:
    """A linear layer W (in x out) with a frozen base + learnable A@B delta."""

    def __init__(self, W: list[list[float]], rank: int, seed: int = 0):
        self.W = W
        self.in_dim = len(W[0]) if W else 0
        self.out_dim = len(W)
        self.rank = min(rank, self.in_dim, self.out_dim)
        self.alpha = stable_float("lora-alpha-%d" % seed, 0.5, 2.0)
        self.A = stable_matrix("lora-A-%d" % seed, self.out_dim, self.rank, -0.1, 0.1)
        self.B = stable_matrix("lora-B-%d" % seed, self.rank, self.in_dim, -0.1, 0.1)

    def forward(self, x: list[float]) -> list[float]:
        base = [sum(self.W[i][j] * x[j] for j in range(self.in_dim)) for i in range(self.out_dim)]
        # h = B @ x  (rank,)
        h = [sum(self.B[r][j] * x[j] for j in range(self.in_dim)) for r in range(self.rank)]
        # delta = A @ h
        scale = self.alpha / self.rank if self.rank else 0.0
        delta = [scale * sum(self.A[i][r] * h[r] for r in range(self.rank)) for i in range(self.out_dim)]
        return [base[i] + delta[i] for i in range(self.out_dim)]


class LoraAdapter:
    """Wraps a head's linear projection with a LoRA delta (task-specific tuning)."""

    def __init__(self, W: list[list[float]], rank: int = 8, seed: int = 0):
        self.lora = LoRALinear(W, rank, seed)

    def forward(self, x: list[float]) -> list[float]:
        return self.lora.forward(x)
