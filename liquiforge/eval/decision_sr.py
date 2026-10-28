# -*- coding: utf-8 -*-
"""Decision success rate: fraction of states where the policy picks the correct action."""
from __future__ import annotations


def decision_success_rate(actions: list[int], correct_actions: list[int]) -> float:
    if not actions:
        return 0.0
    return sum(1 for a, c in zip(actions, correct_actions) if a == c) / len(actions)
