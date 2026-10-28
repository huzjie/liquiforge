# -*- coding: utf-8 -*-
"""Audio dataset: yields (waveform, reference_text) pairs."""
from __future__ import annotations

from .base import Dataset
from ..tokenizers.audio_spec import synthetic_wave


class AudioDataset(Dataset):
    def __init__(self, n: int = 16, samples: int = 1600):
        self.n = n
        self.samples = samples

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return synthetic_wave(self.samples, "audio-ds-%d" % i), "hello world"
