# -*- coding: utf-8 -*-
"""Text perplexity: how well the LM head predicts next tokens."""
from __future__ import annotations


def text_perplexity(log_probs: list[float]) -> float:
    if not log_probs:
        return float("inf")
    nll = -sum(log_probs) / len(log_probs)
    return math.exp(nll)
