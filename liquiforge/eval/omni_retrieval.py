# -*- coding: utf-8 -*-
"""Omni cross-modal retrieval: recall@1 over a query/candidate pool."""
from __future__ import annotations


def omni_retrieval_accuracy(similarities: list[list[float]], gt_index: list[int]) -> float:
    """similarities[i][j] = sim(query_i, candidate_j); gt_index[i] is the answer."""
    if not similarities:
        return 0.0
    correct = 0
    for i, row in enumerate(similarities):
        if max(range(len(row)), key=lambda j: row[j]) == gt_index[i]:
            correct += 1
    return correct / len(similarities)
