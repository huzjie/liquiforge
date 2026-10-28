# -*- coding: utf-8 -*-
"""LoRA: freeze the base head projection and learn a low-rank delta."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liquiforge.adapters.lora import LoRALinear
from liquiforge.utils.rng import stable_matrix

W = stable_matrix("example-base", 16, 32)  # (out=16, in=32)
lora = LoRALinear(W, rank=8)
x = [0.5] * 32
base = [sum(W[i][j] * x[j] for j in range(32)) for i in range(16)]
adapted = lora.forward(x)
print("base[0]    :", round(base[0], 4))
print("adapted[0] :", round(adapted[0], 4))
print("delta[0]   :", round(adapted[0] - base[0], 6))
