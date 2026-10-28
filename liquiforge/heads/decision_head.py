# -*- coding: utf-8 -*-
"""Decision head: actor policy over a discrete action space (for RL fine-tuning)."""
from __future__ import annotations

import math

from ..utils.rng import stable_vector, stable_matrix
from .common import softmax


class DecisionHead:
    """A stochastic policy: pooled state -> action distribution.

    Trained by the RL loop in ``liquiforge/rl`` to maximise a deterministic
    reward — the "decision" modality of the omni model.
    """

    def __init__(self, dim: int, n_actions: int = 16, seed: int = 0):
        self.dim = dim
        self.n_actions = n_actions
        self.W = stable_matrix("decision-head-%d" % seed, dim, n_actions, -0.1, 0.1)

    def _pool(self, h: list[list[float]]) -> list[float]:
        return [sum(h[t][d] for t in range(len(h))) / len(h) for d in range(self.dim)] if h else [0.0] * self.dim

    def logits(self, h: list[list[float]]) -> list[float]:
        pooled = self._pool(h)
        return [sum(self.W[j][i] * pooled[j] for j in range(self.dim)) for i in range(self.n_actions)]

    def policy(self, h: list[list[float]]) -> list[float]:
        return softmax(self.logits(h))

    def sample(self, h: list[list[float]], rng=None) -> tuple[int, list[float]]:
        probs = self.policy(h)
        r = (rng or __import__("random").Random(0)).random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r < acc:
                return i, probs
        return self.n_actions - 1, probs
