# -*- coding: utf-8 -*-
"""Text utilities: word tokenization and Levenshtein edit distance (for WER)."""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff]+")


def tokenize_words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def edit_distance(a: str, b: str) -> int:
    """Classic Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = tokenize_words(reference)
    hyp = tokenize_words(hypothesis)
    if not ref:
        return 1.0 if hyp else 0.0
    return edit_distance(" ".join(ref), " ".join(hyp)) / len(ref)
