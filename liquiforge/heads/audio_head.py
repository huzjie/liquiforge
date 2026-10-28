# -*- coding: utf-8 -*-
"""Audio head: acoustic frames -> text-vocab logits (transcription-style)."""
from __future__ import annotations

from ..utils.text import word_error_rate
from .common import LinearHead


class AudioHead:
    """Maps audio-frame hidden states to text tokens for ASR-style decoding."""

    def __init__(self, dim: int, vocab_size: int = 256, seed: int = 0):
        self.dim = dim
        self.vocab_size = vocab_size
        self.proj = LinearHead(dim, vocab_size, "audio-head", seed)

    def decode(self, h: list[list[float]]) -> list[int]:
        out = []
        for v in h:
            l = self.proj.forward(v)
            out.append(max(range(len(l)), key=lambda i: l[i]))
        return out

    def decode_text(self, h: list[list[float]], id_to_byte: dict[int, int] | None = None) -> str:
        ids = self.decode(h)
        if id_to_byte:
            return "".join(chr(id_to_byte.get(i, 63)) for i in ids)
        return "".join(chr(i) if 32 <= i < 127 else " " for i in ids)

    def wer(self, h: list[list[float]], reference: str,
            id_to_byte: dict[int, int] | None = None) -> float:
        hyp = self.decode_text(h, id_to_byte)
        return word_error_rate(reference, hyp)
