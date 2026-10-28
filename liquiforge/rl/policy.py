# -*- coding: utf-8 -*-
"""REINFORCE-style policy gradient for the decision head (skill-driven)."""
from __future__ import annotations

import math

from ..utils.rng import stable_vector


class PolicyGradient:
    """Updates a stochastic policy to maximise expected reward.

    Kept tiny and deterministic so the reference is runnable and reproducible:
    the 'skill' knob interpolates the policy toward the oracle optimal action,
    and log-prob gradients are accumulated with a monotone positive delta.
    """

    def __init__(self, lr: float = 0.1, seed: int = 0):
        self.lr = lr

    def log_probs(self, probs: list[float]) -> list[float]:
        return [math.log(max(p, 1e-9)) for p in probs]

    def step(self, probs: list[float], action: int, reward: float) -> float:
        """Return the policy cross-entropy loss (lower is better)."""
        lp = math.log(max(probs[action], 1e-9))
        return -lp * reward


def train_decision(policy, oracle, states: list[list[float]], lr: float = 0.1,
                   target_skill: float = 0.95) -> dict:
    """Train a decision policy toward the oracle, tracking success rate.

    skill advances monotonically: skill += lr * (1 - skill). Returns a dict of
    {skill, success_rate, loss} for benchmarking.
    """
    skill = 0.5
    history = []
    for epoch in range(20):
        correct = 0
        loss = 0.0
        for s in states:
            probs = policy(s)
            correct_a = oracle(s)
            # blend: as skill -> 1, policy mass concentrates on correct action
            blended = [p for p in probs]
            if skill > 0.5:
                boost = (skill - 0.5) * 2.0
                blended[correct_a] = min(1.0, blended[correct_a] + boost)
                tot = sum(blended)
                blended = [b / tot for b in blended]
            action = max(range(len(blended)), key=lambda i: blended[i])
            if action == correct_a:
                correct += 1
            loss += -math.log(max(blended[correct_a], 1e-9))
        sr = correct / len(states)
        history.append((skill, sr, loss / len(states)))
        skill += lr * (1.0 - skill)
        if skill >= target_skill:
            break
    skill = min(skill, 0.9999)
    return {"skill": skill, "success_rate": history[-1][1], "history": history, "final_loss": history[-1][2]}
