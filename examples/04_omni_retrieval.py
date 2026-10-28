# -*- coding: utf-8 -*-
"""Omni: cross-modal retrieval — rank vision candidates for a text query."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liquiforge.train import SkillModel
from liquiforge.tokenizers.vision_patch import synthetic_image
from liquiforge.tokenizers.unified import Modality

m = SkillModel()
# text query
q_embs = [m.unified.embed(Modality.TEXT, 7)] * 3
q_hidden, _ = m.backbone.forward(q_embs)
q_emb = m.omni_head.embed(q_hidden)

# vision candidates
scores = []
for i in range(4):
    toks = m.vision_tok.encode(synthetic_image(28, "retrieval-%d" % i))
    c_embs = [m.unified.embed(Modality.VISION, min(j, 48)) for j in range(len(toks))]
    c_hidden, _ = m.backbone.forward(c_embs)
    c_emb = m.omni_head.embed(c_hidden)
    scores.append(m.omni_head.cosine(q_emb, c_emb))

print("cosine similarities:", [round(s, 3) for s in scores])
print("best candidate:", max(range(len(scores)), key=lambda i: scores[i]))
