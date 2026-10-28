# -*- coding: utf-8 -*-
"""Linearized (feature-map) attention: O(n) per token, no quadratic softmax.

Softmax attention is replaced with a positive feature map phi(x)=ELU(x)+1, so
the recurrence
    S_i = S_{i-1} + phi(k_i) (v_i)^T,   z_i = z_{i-1} + phi(k_i)
    o_i = phi(q_i) S_i / (phi(q_i) . z_i)
gives causal attention in linear time and constant memory w.r.t. sequence length.
"""
from __future__ import annotations

from ..utils.rng import stable_matrix, stable_vector
from .norm import elu


class LinearAttention:
    def __init__(self, dim: int, n_heads: int = 4, seed: int = 0):
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.Wq = stable_matrix("la-Wq-%d" % seed, dim, dim, -0.1, 0.1)
        self.Wk = stable_matrix("la-Wk-%d" % seed, dim, dim, -0.1, 0.1)
        self.Wv = stable_matrix("la-Wv-%d" % seed, dim, dim, -0.1, 0.1)
        self.Wo = stable_matrix("la-Wo-%d" % seed, dim, dim, -0.1, 0.1)

    @staticmethod
    def _phi(v: list[float]) -> list[float]:
        return [elu(x) + 1.0 for x in v]

    def _project(self, W, x):
        return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        n = len(x)
        q = [self._project(self.Wq, t) for t in x]
        k = [self._project(self.Wk, t) for t in x]
        v = [self._project(self.Wv, t) for t in x]

        outputs = []
        for h in range(self.n_heads):
            lo, hi = h * self.hd, (h + 1) * self.hd
            # state: hd x hd (S), hd (z)
            S = [[0.0] * self.hd for _ in range(self.hd)]
            z = [0.0] * self.hd
            head_out = []
            for t in range(n):
                qh = q[t][lo:hi]
                kh = self._phi(k[t][lo:hi])
                vh = v[t][lo:hi]
                # o = phi(q) . S  (hd,)
                phi_q = self._phi(qh)
                o = [0.0] * self.hd
                for d in range(self.hd):
                    acc = 0.0
                    for c in range(self.hd):
                        acc += phi_q[c] * S[c][d]
                    o[d] = acc
                denom = sum(phi_q[c] * z[c] for c in range(self.hd)) + 1e-6
                head_out.append([o[d] / denom for d in range(self.hd)])
                # update state S += kh outer vh ; z += kh
                for c in range(self.hd):
                    z[c] += kh[c]
                    for d in range(self.hd):
                        S[c][d] += kh[c] * vh[d]
            outputs.append(head_out)

        # concat heads + output projection
        out = []
        for t in range(n):
            cat = []
            for h in range(self.n_heads):
                cat.extend(outputs[h][t])
            out.append(self._project(self.Wo, cat))
        return out
