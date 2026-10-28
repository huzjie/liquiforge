# -*- coding: utf-8 -*-
"""Minimal Dataset / DataLoader (batching + shuffle) with zero dependencies."""
from __future__ import annotations

import random


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 4, shuffle: bool = False, seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __iter__(self):
        n = len(self.dataset)
        order = list(range(n))
        if self.shuffle:
            self.rng.shuffle(order)
        for start in range(0, n, self.batch_size):
            idxs = order[start:start + self.batch_size]
            yield [self.dataset[i] for i in idxs]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
