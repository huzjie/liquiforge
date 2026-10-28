# -*- coding: utf-8 -*-
"""AdamW optimizer (decoupled weight decay) for dense parameter lists."""
from __future__ import annotations

import math


class AdamW:
    def __init__(self, params: list[list[list[float]]], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [[[0.0] * len(p) for p in mat] for mat in params]
        self.v = [[[0.0] * len(p) for p in mat] for mat in params]

    def zero_grad(self):
        pass

    def step(self, grads: list[list[list[float]]]):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for pmat, gmat, mmat, vmat in zip(self.params, grads, self.m, self.v):
            for p, g, m, v in zip(pmat, gmat, mmat, vmat):
                for i in range(len(p)):
                    m[i] = self.b1 * m[i] + (1 - self.b1) * g[i]
                    v[i] = self.b2 * v[i] + (1 - self.b2) * g[i] * g[i]
                    mh = m[i] / bc1
                    vh = v[i] / bc2
                    p[i] -= self.lr * (mh / (math.sqrt(vh) + self.eps) + self.wd * p[i])
