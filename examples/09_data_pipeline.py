# -*- coding: utf-8 -*-
"""Data pipeline: build per-modality datasets and iterate mini-batches."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liquiforge.data.base import DataLoader
from liquiforge.data.vision_data import VisionDataset
from liquiforge.data.decision_data import DecisionDataset

for name, ds in [("vision", VisionDataset(10)), ("decision", DecisionDataset(10, 64, 16))]:
    dl = DataLoader(ds, batch_size=3, shuffle=True)
    batches = list(dl)
    print("%-10s samples=%d batches=%d first_batch=%d" % (name, len(ds), len(batches), len(batches[0])))
