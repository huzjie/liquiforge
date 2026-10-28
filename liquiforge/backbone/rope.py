# -*- coding: utf-8 -*-
"""Rotary Position Embedding (RoPE) utilities.

Provides cos/sin tables for relative-position-aware attention. Not used by the
default linear-attention path (which is inherently position-free), but useful
when mixing RoPE into a softmax attention head for absolute/relative position.
"""
from __future__ import annotations

import math


def rope_tables(seq_len: int, dim: int, base: float = 10000.0) -> tuple[list[float], list[float]]:
    """Return (cos, sin) tables: each is (seq_len x dim/2) of rotation angles."""
    half = dim // 2
    freqs = [1.0 / (base ** (2 * i / dim)) for i in range(half)]
    cos = []
    sin = []
    for pos in range(seq_len):
        c = [math.cos(pos * f) for f in freqs]
        s = [math.sin(pos * f) for f in freqs]
        cos.append(c)
        sin.append(s)
    return cos, sin


def apply_rope(x: list[float], pos: int, cos: list[list[float]], sin: list[list[float]]) -> list[float]:
    half = len(x) // 2
    out = [0.0] * len(x)
    for i in range(half):
        c, s = cos[pos][i], sin[pos][i]
        out[2 * i] = x[2 * i] * c - x[2 * i + 1] * s
        out[2 * i + 1] = x[2 * i] * s + x[2 * i + 1] * c
    return out
