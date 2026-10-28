# -*- coding: utf-8 -*-
"""RMSNorm + helpers shared by backbone sub-layers."""
from __future__ import annotations

import math

from ..utils.rng import stable_vector


class RMSNorm:
    """Root-mean-square layer norm (the norm used across the tri-hybrid block)."""

    def __init__(self, dim: int, eps: float = 1e-5, seed: int = 0):
        self.eps = eps
        self.weight = stable_vector("rmsnorm-%d-%d" % (dim, seed), dim, 0.9, 1.1)

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        out = []
        for vec in x:
            ms = sum(v * v for v in vec) / len(vec)
            inv = 1.0 / math.sqrt(ms + self.eps)
            out.append([v * inv * w for v, w in zip(vec, self.weight)])
        return out


def elu(x: float) -> float:
    return x if x >= 0 else math.exp(x) - 1.0


def matvec(W: list[list[float]], x: list[float]) -> list[float]:
    return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]
