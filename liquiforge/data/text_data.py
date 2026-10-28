# -*- coding: utf-8 -*-
"""Text corpus dataset: yields (token_ids, target_ids) for causal LM."""
from __future__ import annotations

from .base import Dataset
from ..tokenizers.bpe import BPETokenizer


class TextDataset(Dataset):
    def __init__(self, corpus: list[str], vocab_size: int = 256, seed: int = 0):
        self.bpe = BPETokenizer(vocab_size, seed)
        self.bpe.train(corpus)
        self.samples = [self.bpe.encode(s) for s in corpus if s]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ids = self.samples[i]
        return ids[:-1], ids[1:]
