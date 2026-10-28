# -*- coding: utf-8 -*-
"""Deterministic reward model for the decision head."""
from __future__ import annotations

import math

from ..utils.rng import stable_vector


class RewardModel:
    """Scores an action given the pooled state, with a deterministic 'correct' action.

    The single ground-truth correct action lives here (an oracle), so the RL loop
    has a well-defined optimum and can be validated end-to-end.
    """

    def __init__(self, dim: int, n_actions: int, seed: int = 0):
        self.dim = dim
        self.n_actions = n_actions
        self.state_weight = stable_vector("reward-state-%d" % seed, dim, -1.0, 1.0)

    def correct_action(self, pooled: list[float]) -> int:
        # deterministic oracle: sign of a weighted sum maps into an action bin
        s = sum(self.state_weight[d] * pooled[d] for d in range(self.dim))
        idx = int((math.tanh(s) + 1.0) / 2.0 * (self.n_actions - 1) + 0.5)
        return max(0, min(self.n_actions - 1, idx))

    def reward(self, pooled: list[float], action: int) -> float:
        return 1.0 if action == self.correct_action(pooled) else 0.0
