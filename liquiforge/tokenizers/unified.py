# -*- coding: utf-8 -*-
"""Unified token space: (modality, local_id) -> shared embedding index + embedding table."""
from __future__ import annotations

from ..utils.rng import stable_matrix


class Modality:
    TEXT = 0
    VISION = 1
    AUDIO = 2
    OMNI = 3
    DECISION = 4
    NAMES = {0: "text", 1: "vision", 2: "audio", 3: "omni", 4: "decision"}
    COUNT = 5


class UnifiedTokenizer:
    """Every modality's discrete token maps into one shared embedding table.

    The trick that makes "one backbone, five modalities" work: each modality
    contributes a prefix region of the embedding table, and a learned (deterministic
    here) projection aligns them into a shared ``dim``-width vector space.
    """

    def __init__(self, dim: int = 128, vocab_size: int = 256,
                 n_patch: int = 49, codebook: int = 256, seed: int = 0):
        self.dim = dim
        self.vocab_size = vocab_size
        self.n_patch = n_patch
        self.codebook = codebook
        # per-modality token counts
        self.counts = {
            Modality.TEXT: vocab_size,
            Modality.VISION: n_patch,
            Modality.AUDIO: codebook,
            Modality.OMNI: vocab_size + n_patch + codebook,
            Modality.DECISION: vocab_size,
        }
        self.table = stable_matrix("unified-emb-%d" % seed, sum(self.counts.values()), dim)

    def global_id(self, modality: int, local_id: int) -> int:
        base = 0
        for m in range(modality):
            base += self.counts.get(m, 0)
        return base + local_id

    def embed(self, modality: int, local_id: int) -> list[float]:
        gid = self.global_id(modality, local_id)
        return self.table[gid]

    def embed_sequence(self, modality: int, ids: list[int]) -> list[list[float]]:
        return [self.embed(modality, i) for i in ids]
