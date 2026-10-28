# -*- coding: utf-8 -*-
"""Text LM head: hidden states -> next-token logits + perplexity."""
from __future__ import annotations

import math

from .common import LinearHead, log_softmax


class TextHead:
    """Language-modeling head (causal next-token prediction)."""

    def __init__(self, dim: int, vocab_size: int = 256, seed: int = 0):
        self.dim = dim
        self.vocab_size = vocab_size
        self.proj = LinearHead(dim, vocab_size, "text-head", seed)

    def logits(self, h: list[list[float]]) -> list[list[float]]:
        return [self.proj.forward(v) for v in h]

    def next_token(self, h: list[float]) -> tuple[int, float]:
        """Argmax next token + its probability."""
        l = self.proj.forward(h)
        best = max(range(len(l)), key=lambda i: l[i])
        s = sum(math.exp(x - max(l)) for x in l)
        return best, math.exp(l[best] - max(l)) / s

    def perplexity(self, h: list[list[float]], targets: list[int]) -> float:
        """Mean negative log-likelihood over targets (exp -> perplexity)."""
        if not targets:
            return float("inf")
        nll = 0.0
        for i, t in enumerate(targets):
            if i >= len(h):
                break
            logp = log_softmax(self.proj.forward(h[i]))
            nll -= logp[t]
        return math.exp(nll / len(targets))
