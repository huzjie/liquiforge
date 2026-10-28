# -*- coding: utf-8 -*-
"""Shared head helpers."""
from __future__ import annotations

import math

from ..utils.rng import stable_matrix


def softmax(v: list[float]) -> list[float]:
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def log_softmax(v: list[float]) -> list[float]:
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x - m - math.log(s) for x in v]


class LinearHead:
    """A single learned linear projection head (hidden -> out_dim)."""

    def __init__(self, dim: int, out_dim: int, key: str, seed: int = 0):
        self.dim = dim
        self.out_dim = out_dim
        self.W = stable_matrix("%s-%d" % (key, seed), dim, out_dim, -0.1, 0.1)

    def forward(self, h: list[float]) -> list[float]:
        return [sum(self.W[j][i] * h[j] for j in range(self.dim)) for i in range(self.out_dim)]
