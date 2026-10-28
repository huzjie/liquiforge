# -*- coding: utf-8 -*-
"""Vision head: pooled [first-token] representation -> class logits + accuracy."""
from __future__ import annotations

from .common import LinearHead, softmax


class VisionHead:
    """Image classification head over the backbone's patch-token sequence."""

    def __init__(self, dim: int, n_classes: int = 10, seed: int = 0):
        self.dim = dim
        self.n_classes = n_classes
        self.proj = LinearHead(dim, n_classes, "vision-head", seed)

    def classify(self, h: list[list[float]]) -> tuple[int, float]:
        pooled = h[0] if h else [0.0] * self.dim
        logits = self.proj.forward(pooled)
        best = max(range(len(logits)), key=lambda i: logits[i])
        p = softmax(logits)
        return best, p[best]

    def accuracy(self, h_batch: list[list[list[float]]], labels: list[int]) -> float:
        if not labels:
            return 0.0
        correct = 0
        for h, y in zip(h_batch, labels):
            pred, _ = self.classify(h)
            if pred == y:
                correct += 1
        return correct / len(labels)
