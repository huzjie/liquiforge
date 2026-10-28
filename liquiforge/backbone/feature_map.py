# -*- coding: utf-8 -*-
"""Feature maps for linearized attention (what you multiply Q/K by).

Linear attention replaces softmax(QK^T) with phi(Q) phi(K)^T. The choice of
phi controls numerical stability:

- ``elu_plus_one`` (default): strictly positive, recommended — guarantees a
  positive normalizer and avoids division by zero.
- ``relu``: non-negative but has zeros, which can drop tokens.
- ``identity``: can produce negative inner products -> unstable normalizer.
"""
from __future__ import annotations

import math


def elu_plus_one(x: list[float]) -> list[float]:
    return [(v if v >= 0 else math.exp(v) - 1.0) + 1.0 for v in x]


def relu(x: list[float]) -> list[float]:
    return [max(0.0, v) for v in x]


def identity(x: list[float]) -> list[float]:
    return list(x)


def phi(name: str, x: list[float]) -> list[float]:
    return {"elu": elu_plus_one, "relu": relu, "identity": identity}[name](x)
