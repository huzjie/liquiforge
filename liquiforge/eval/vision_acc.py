# -*- coding: utf-8 -*-
"""Vision accuracy: fraction of correctly classified images."""
from __future__ import annotations


def vision_accuracy(predictions: list[int], labels: list[int]) -> float:
    if not labels:
        return 0.0
    return sum(1 for p, y in zip(predictions, labels) if p == y) / len(labels)
