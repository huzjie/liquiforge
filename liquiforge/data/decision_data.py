# -*- coding: utf-8 -*-
"""Decision dataset: yields (state, correct_action) pairs via the reward oracle."""
from __future__ import annotations

from .base import Dataset
from ..utils.rng import stable_vector
from ..rl.reward import RewardModel


class DecisionDataset(Dataset):
    def __init__(self, n: int = 16, dim: int = 64, n_actions: int = 16, seed: int = 0):
        self.n = n
        self.dim = dim
        self.reward = RewardModel(dim, n_actions, seed)
        self.states = [stable_vector("decision-ds-%d" % i, dim, -1.0, 1.0) for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = self.states[i]
        return s, self.reward.correct_action(s)
