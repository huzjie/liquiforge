# -*- coding: utf-8 -*-
"""Full deterministic training + benchmark across all five modalities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liquiforge.train import train
from liquiforge.bench import run_all, summarize

print(summarize(run_all()))
print()
res = train(epochs=20, target=0.95)
print("trained final skill=%.4f" % res["final"]["skill"])
