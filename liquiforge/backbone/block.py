# -*- coding: utf-8 -*-
"""The tri-hybrid block: linear attention (global) + depthwise conv (local) + MoE (capacity)."""
from __future__ import annotations

from ..utils.rng import stable_float
from .linear_attention import LinearAttention
from .conv import DepthwiseConv
from .moe import SparseMoE
from .norm import RMSNorm


class TriHybridBlock:
    """One layer of the omni backbone.

    All three sub-layers run in parallel on the same normalized input and are
    gated before the residual add — a "tri-hybrid" mix of global, local and
    high-capacity inductive biases in a single block.
    """

    def __init__(self, dim: int, n_heads: int = 4, n_experts: int = 8, top_k: int = 2,
                 conv_kernel: int = 5, ffn_mult: int = 4, seed: int = 0):
        self.dim = dim
        self.norm1 = RMSNorm(dim, seed=seed)
        self.norm2 = RMSNorm(dim, seed=seed + 1000)
        self.attn = LinearAttention(dim, n_heads, seed)
        self.conv = DepthwiseConv(dim, conv_kernel, seed)
        self.moe = SparseMoE(dim, n_experts, top_k, ffn_mult, seed)
        self.gate_a = stable_float("gate-a-%d" % seed, 0.4, 1.0)
        self.gate_c = stable_float("gate-c-%d" % seed, 0.2, 0.6)
        self.gate_m = stable_float("gate-m-%d" % seed, 0.4, 1.0)

    def forward(self, x: list[list[float]]) -> tuple[list[list[float]], float]:
        h = self.norm1.forward(x)
        a = self.attn.forward(h)
        c = self.conv.forward(h)
        m, lb = self.moe.forward(self.norm2.forward(x))
        n = len(x)
        out = []
        for t in range(n):
            vec = [0.0] * self.dim
            for d in range(self.dim):
                vec[d] = x[t][d] + self.gate_a * a[t][d] + self.gate_c * c[t][d] + self.gate_m * m[t][d]
            out.append(vec)
        return out, lb
