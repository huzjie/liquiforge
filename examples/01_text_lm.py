# -*- coding: utf-8 -*-
"""Text LM: tokenize a sentence, run the backbone, predict the next token."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liquiforge.train import SkillModel
from liquiforge.tokenizers.unified import Modality

m = SkillModel()
text = "the quick brown fox"
ids = m.bpe.encode(text)
embs = [m.unified.embed(Modality.TEXT, i) for i in ids]
hidden, _ = m.backbone.forward(embs)
logits = m.text_head.proj.forward(hidden[-1])
best = max(range(len(logits)), key=lambda i: logits[i])
print("input :", text)
print("tokens:", ids)
print("next token id:", best, "(prob-mass top-1)")
