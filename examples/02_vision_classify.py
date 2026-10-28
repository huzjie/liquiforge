# -*- coding: utf-8 -*-
"""Vision: encode a synthetic image into patches and classify it."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liquiforge.train import SkillModel
from liquiforge.tokenizers.vision_patch import synthetic_image
from liquiforge.tokenizers.unified import Modality

m = SkillModel()
img = synthetic_image(28, "example-vision")
toks = m.vision_tok.encode(img)
embs = [m.unified.embed(Modality.VISION, min(i, 48)) for i in range(len(toks))]
hidden, _ = m.backbone.forward(embs)
pred, prob = m.vision_head.classify(hidden)
print("patches:", len(toks))
print("predicted class:", pred, "conf=%.3f" % prob)
