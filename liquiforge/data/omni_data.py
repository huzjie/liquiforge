# -*- coding: utf-8 -*-
"""Omni dataset: yields (query_modality, candidate_modality) cross-modal pairs."""
from __future__ import annotations

from .base import Dataset
from ..tokenizers.vision_patch import synthetic_image


class OmniPairsDataset(Dataset):
    def __init__(self, n: int = 16):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        caption = "sample caption number %d" % i
        image = synthetic_image(28, "omni-ds-%d" % i)
        return caption, image
