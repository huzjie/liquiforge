# -*- coding: utf-8 -*-
"""Audio tokenizer: waveform -> STFT -> mel filterbank -> vector-quantized tokens."""
from __future__ import annotations

import math

from ..utils.rng import stable_matrix, stable_float, stable_vector


def _dft(x: list[float]) -> list[complex]:
    n = len(x)
    out = []
    for k in range(n):
        re = im = 0.0
        for t in range(n):
            ang = -2.0 * math.pi * k * t / n
            re += x[t] * math.cos(ang)
            im += x[t] * math.sin(ang)
        out.append(complex(re, im))
    return out


class AudioSpecTokenizer:
    """STFT -> mel-scale filterbank -> VQ codebook tokens (deterministic)."""

    def __init__(self, n_fft: int = 256, hop: int = 64, n_mels: int = 80,
                 codebook: int = 256, sample_rate: int = 16000, seed: int = 0):
        self.n_fft = n_fft
        self.hop = hop
        self.n_mels = n_mels
        self.codebook = codebook
        self.sample_rate = sample_rate
        self.mel_filter = self._build_mel_filter(n_fft, n_mels, sample_rate)
        # deterministic codebook: codebook x n_mels
        self.centroids = stable_matrix("audio-codebook-%d" % seed, codebook, n_mels, 0.0, 1.0)

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filter(self, n_fft: int, n_mels: int, sr: int) -> list[list[float]]:
        f_min, f_max = 0.0, sr / 2.0
        mel_min, mel_max = self._hz_to_mel(f_min), self._hz_to_mel(f_max)
        mels = [mel_min + (mel_max - mel_min) * i / (n_mels + 1) for i in range(n_mels + 2)]
        hz = [self._mel_to_hz(m) for m in mels]
        bins = [int(math.floor((n_fft + 1) * h / sr)) for h in hz]
        filters = []
        for i in range(1, n_mels + 1):
            f = [0.0] * (n_fft // 2 + 1)
            lo, c, hi = bins[i - 1], bins[i], bins[i + 1]
            for j in range(lo, c):
                if c > lo:
                    f[j] = (j - lo) / (c - lo)
            for j in range(c, hi):
                if hi > c:
                    f[j] = (hi - j) / (hi - c)
            filters.append(f)
        return filters

    def _stft(self, wave: list[float]) -> list[list[float]]:
        frames = []
        n = len(wave)
        for start in range(0, n - self.n_fft + 1, self.hop):
            frame = wave[start:start + self.n_fft]
            spec = _dft(frame)
            mag = [abs(c) for c in spec[: self.n_fft // 2 + 1]]
            frames.append(mag)
        return frames

    def _mel(self, frame: list[float]) -> list[float]:
        return [sum(frame[j] * f[j] for j in range(len(frame))) for f in self.mel_filter]

    def _vq(self, mel_vec: list[float]) -> int:
        best, best_d = 0, float("inf")
        for i, c in enumerate(self.centroids):
            d = sum((mel_vec[j] - c[j]) ** 2 for j in range(len(mel_vec)))
            if d < best_d:
                best, best_d = i, d
        return best

    def encode(self, wave: list[float]) -> list[int]:
        """waveform -> VQ token ids (one per STFT frame)."""
        if len(wave) < self.n_fft:
            return []
        frames = self._stft(wave)
        out = []
        for frame in frames:
            mel = self._mel(frame)
            out.append(self._vq(mel))
        return out


def synthetic_wave(n: int = 1600, key: str = "audio-demo") -> list[float]:
    """Deterministic pseudo-speech waveform (sum of sines + seeded noise)."""
    wave = []
    for t in range(n):
        s = 0.3 * math.sin(2 * math.pi * 220 * t / 16000)
        s += 0.2 * math.sin(2 * math.pi * 440 * t / 16000)
        s += 0.1 * stable_float("%s-%d" % (key, t), -1.0, 1.0)
        wave.append(s)
    return wave
