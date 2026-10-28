# -*- coding: utf-8 -*-
"""Omni head: cross-modal retrieval via pooled embeddings + cosine similarity."""
from __future__ import annotations

import math

from ..utils.rng import stable_matrix


class OmniHead:
    """Projects any modality's hidden states into a shared retrieval space.

    A single backbone serves text/vision/audio; the omni head learns one shared
    projection so cross-modal pairs (e.g. image<->caption) are close in cosine.
    """

    def __init__(self, dim: int, out_dim: int = 64, seed: int = 0):
        self.dim = dim
        self.out_dim = out_dim
        self.proj = stable_matrix("omni-head-%d" % seed, dim, out_dim, -0.1, 0.1)

    def embed(self, h: list[list[float]]) -> list[float]:
        pooled = [sum(h[t][d] for t in range(len(h))) / len(h) for d in range(self.dim)] if h else [0.0] * self.dim
        return [sum(self.proj[j][i] * pooled[j] for j in range(self.dim)) for i in range(self.out_dim)]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) + 1e-9
        nb = math.sqrt(sum(x * x for x in b)) + 1e-9
        return dot / (na * nb)

    def similarity(self, h_a: list[list[float]], h_b: list[list[float]]) -> float:
        return self.cosine(self.embed(h_a), self.embed(h_b))

    def retrieval_accuracy(self, queries: list[list[list[float]]],
                           candidates: list[list[list[float]]],
                           gt_index: list[int]) -> float:
        """queries vs candidates; gt_index[i] is the true candidate for query i."""
        if not queries:
            return 0.0
        correct = 0
        q_emb = [self.embed(h) for h in queries]
        c_emb = [self.embed(h) for h in candidates]
        for i, q in enumerate(q_emb):
            sims = [self.cosine(q, c) for c in c_emb]
            pred = max(range(len(sims)), key=lambda j: sims[j])
            if pred == gt_index[i]:
                correct += 1
        return correct / len(queries)
