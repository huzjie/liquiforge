# -*- coding: utf-8 -*-
"""Sparse Mixture-of-Experts with top-k routing + auxiliary load-balancing loss."""
from __future__ import annotations

import math

from ..utils.rng import stable_matrix, stable_vector


def _softmax(v: list[float]) -> list[float]:
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


class SparseMoE:
    """A gated FFN where only the top-k experts run per token.

    Capacity scales with expert count while compute stays ~top-k — the key
    trick for huge omni backbones (LFM3-style) staying cheap at inference.
    """

    def __init__(self, dim: int, n_experts: int = 8, top_k: int = 2,
                 ffn_mult: int = 4, seed: int = 0):
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.ffn = dim * ffn_mult
        self.gate = stable_matrix("moe-gate-%d" % seed, dim, n_experts, -0.1, 0.1)
        self.noise_scale = 0.05
        self._experts = []
        for e in range(n_experts):
            w1 = stable_matrix("moe-e%d-w1-%d" % (e, seed), dim, self.ffn, -0.1, 0.1)
            w2 = stable_matrix("moe-e%d-w2-%d" % (e, seed), self.ffn, dim, -0.1, 0.1)
            self._experts.append((w1, w2))

    def _expert(self, e: int, x: list[float]) -> list[float]:
        w1, w2 = self._experts[e]
        h = [max(0.0, sum(w1[j][i] * x[j] for j in range(self.dim))) for i in range(self.ffn)]
        return [sum(w2[j][i] * h[j] for j in range(self.ffn)) for i in range(self.dim)]

    def forward(self, x: list[list[float]]) -> tuple[list[list[float]], float]:
        n = len(x)
        outputs = []
        expert_frac = [0.0] * self.n_experts
        expert_prob = [0.0] * self.n_experts
        for vec in x:
            logits = [sum(self.gate[j][i] * vec[j] for j in range(self.dim)) for i in range(self.n_experts)]
            # deterministic load-balancing noise (seeded by token content hash)
            h = hash(tuple(round(v, 4) for v in vec)) & 0xFFFFFF
            noisy = [logits[i] + self.noise_scale * (((h >> (i * 3)) & 7) / 7.0 - 0.5) for i in range(self.n_experts)]
            probs = _softmax(noisy)
            for i in range(self.n_experts):
                expert_prob[i] += probs[i] / n
            order = sorted(range(self.n_experts), key=lambda i: -probs[i])
            chosen = order[: self.top_k]
            gate = _softmax([probs[i] for i in chosen])
            out = [0.0] * self.dim
            for gi, ei in zip(gate, chosen):
                expert_frac[ei] += 1.0 / (n * self.top_k)
                eo = self._expert(ei, vec)
                for d in range(self.dim):
                    out[d] += gi * eo[d]
            outputs.append(out)

        # auxiliary load-balancing loss (Shazeer et al.): encourages uniform routing
        lb = self.n_experts * sum(expert_frac[i] * expert_prob[i] for i in range(self.n_experts))
        return outputs, lb
