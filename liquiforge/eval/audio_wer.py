# -*- coding: utf-8 -*-
"""Audio WER: word error rate of the transcription against a reference."""
from __future__ import annotations

from ..utils.text import word_error_rate


def audio_wer(reference: str, hypothesis: str) -> float:
    return word_error_rate(reference, hypothesis)
