# -*- coding: utf-8 -*-
"""End-to-end evaluator tying a backbone + all heads together."""
from __future__ import annotations

from ..backbone.model import OmniBackbone
from ..heads.text_head import TextHead
from ..heads.vision_head import VisionHead
from ..heads.audio_head import AudioHead
from ..heads.omni_head import OmniHead
from ..heads.decision_head import DecisionHead


class Evaluator:
    """Runs a standardized evaluation across the five modalities.

    Each metric is computed on deterministic synthetic inputs so the numbers
    are reproducible and can be tracked across skill levels in the benchmark.
    """

    def __init__(self, cfg: dict):
        m = cfg["model"]
        self.dim = m["dim"]
        self.backbone = OmniBackbone(
            dim=m["dim"], n_layers=m["n_layers"], n_heads=m["n_heads"],
            n_experts=m["n_experts"], top_k=m["top_k"],
            conv_kernel=m["conv_kernel"], ffn_mult=m["ffn_mult"], seed=cfg["training"]["seed"],
        )
        self.text = TextHead(m["dim"], m["vocab_size"], cfg["training"]["seed"])
        self.vision = VisionHead(m["dim"], 10, cfg["training"]["seed"])
        self.audio = AudioHead(m["dim"], m["vocab_size"], cfg["training"]["seed"])
        self.omni = OmniHead(m["dim"], 64, cfg["training"]["seed"])
        self.decision = DecisionHead(m["dim"], m["n_actions"], cfg["training"]["seed"])
