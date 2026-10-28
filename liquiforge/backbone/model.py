# -*- coding: utf-8 -*-
"""OmniBackbone: stacked tri-hybrid blocks over a unified embedding table."""
from __future__ import annotations

from ..utils.rng import stable_matrix
from .block import TriHybridBlock
from .norm import RMSNorm


class OmniBackbone:
    """The shared trunk: (modality, token) -> embeddings -> N tri-hybrid blocks.

    This is the single backbone every task head plugs into. Only the heads and
    light adapters change between text/vision/audio/omni/decision variants.
    """

    def __init__(self, dim: int = 128, n_layers: int = 6, n_heads: int = 4,
                 n_experts: int = 8, top_k: int = 2, conv_kernel: int = 5,
                 ffn_mult: int = 4, seed: int = 0):
        self.dim = dim
        self.n_layers = n_layers
        self.seed = seed
        self.blocks = [
            TriHybridBlock(dim, n_heads, n_experts, top_k, conv_kernel, ffn_mult, seed + i)
            for i in range(n_layers)
        ]
        self.final_norm = RMSNorm(dim, seed=seed + 5000)

    def forward(self, embs: list[list[float]]) -> tuple[list[list[float]], list[float]]:
        """embs: (n, dim) -> hidden states (n, dim) and per-layer LB losses."""
        x = embs
        lbs = []
        for blk in self.blocks:
            x, lb = blk.forward(x)
            lbs.append(lb)
        return self.final_norm.forward(x), lbs

    def load_balance_loss(self, lbs: list[float]) -> float:
        return sum(lbs) / len(lbs) if lbs else 0.0
