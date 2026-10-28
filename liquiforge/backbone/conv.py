# -*- coding: utf-8 -*-
"""Depthwise causal 1-D short convolution: local context in linear attention stacks."""
from __future__ import annotations

from ..utils.rng import stable_vector


class DepthwiseConv:
    """Causal depthwise convolution over the sequence dimension.

    Each output position only sees past positions (causal), capturing local
    structure that linear attention's global recurrence under-weights.
    """

    def __init__(self, dim: int, kernel: int = 5, seed: int = 0):
        self.dim = dim
        self.kernel = kernel
        # per-channel kernel weights (dim x kernel), deterministic
        self.w = [stable_vector("conv-w-%d-%d" % (d, seed), kernel, -0.2, 0.2) for d in range(dim)]
        self.b = stable_vector("conv-b-%d" % seed, dim, -0.05, 0.05)

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        n = len(x)
        out = []
        for t in range(n):
            vec = [0.0] * self.dim
            for d in range(self.dim):
                acc = self.b[d]
                for i in range(self.kernel):
                    src = t - i
                    if src >= 0:
                        acc += self.w[d][i] * x[src][d]
                vec[d] = acc
            out.append(vec)
        return out
