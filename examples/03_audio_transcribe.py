# -*- coding: utf-8 -*-
"""Audio: encode a synthetic waveform into VQ tokens and transcribe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liquiforge.train import SkillModel
from liquiforge.tokenizers.audio_spec import synthetic_wave
from liquiforge.tokenizers.unified import Modality

m = SkillModel()
wave = synthetic_wave(1600, "example-audio")
ids = m.audio_tok.encode(wave)
embs = [m.unified.embed(Modality.AUDIO, i) for i in ids]
hidden, _ = m.backbone.forward(embs)
hyp = m.audio_head.decode_text(hidden)
print("frames:", len(ids))
print("transcript:", hyp)
