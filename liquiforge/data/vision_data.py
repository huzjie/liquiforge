# -*- coding: utf-8 -*-
"""Vision dataset: yields (image, label) pairs."""
from __future__ import annotations

from .base import Dataset
from ..tokenizers.vision_patch import synthetic_image


class VisionDataset(Dataset):
    def __init__(self, n: int = 16, image_size: int = 28, n_classes: int = 10):
        self.n = n
        self.image_size = image_size
        self.n_classes = n_classes

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = synthetic_image(self.image_size, "vision-ds-%d" % i)
        return img, i % self.n_classes
