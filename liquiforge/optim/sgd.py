# -*- coding: utf-8 -*-
"""SGD optimizer + cosine learning-rate scheduler."""
from __future__ import annotations

import math


class SGD:
    def __init__(self, params: list[list[list[float]]], lr: float = 1e-2, momentum: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.vel = [[[0.0] * len(p) for p in mat] for mat in params]

    def step(self, grads: list[list[list[float]]]):
        for pmat, gmat, vmat in zip(self.params, grads, self.vel):
            for p, g, v in zip(pmat, gmat, vmat):
                for i in range(len(p)):
                    v[i] = self.momentum * v[i] - self.lr * g[i]
                    p[i] += v[i]


class CosineScheduler:
    def __init__(self, lr0: float, total_steps: int, lr_min: float = 0.0):
        self.lr0 = lr0
        self.total_steps = max(total_steps, 1)
        self.lr_min = lr_min

    def __call__(self, step: int) -> float:
        t = min(step, self.total_steps) / self.total_steps
        return self.lr_min + 0.5 * (self.lr0 - self.lr_min) * (1.0 + math.cos(math.pi * t))
