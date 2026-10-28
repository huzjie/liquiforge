# -*- coding: utf-8 -*-
"""Benchmark: measure all five metrics at skill=0.5 (baseline) and skill~1 (trained)."""
from __future__ import annotations

from .train import SkillModel, train
from .config import load_config


def run_all(cfg: dict | None = None) -> dict:
    """Return {metric: (baseline, trained)} — every metric should improve monotonically."""
    cfg = cfg or load_config()
    model = SkillModel(cfg)
    baseline = model.metrics()

    # jump to trained state via the deterministic training loop
    result = train(cfg)
    trained = result["final"]

    keys = ["text_perplexity", "vision_accuracy", "audio_wer", "omni_retrieval", "decision_success"]
    out = {}
    for k in keys:
        out[k] = (baseline.get(k), trained.get(k))
    out["skill"] = (baseline.get("skill"), trained.get("skill"))
    return out


def summarize(results: dict) -> str:
    lines = []
    lines.append("=" * 62)
    lines.append("liquiforge benchmark  (lower is better for perplexity/WER)")
    lines.append("=" * 62)
    names = {
        "text_perplexity": "text  perplexity",
        "vision_accuracy": "vision accuracy ",
        "audio_wer": "audio  WER      ",
        "omni_retrieval": "omni  recall@1  ",
        "decision_success": "decision succ.  ",
    }
    for k, (b, t) in results.items():
        if k == "skill":
            lines.append("  skill            %.4f -> %.4f" % (b, t))
            continue
        arrow = "UP" if t >= b else "DOWN"
        lines.append("  %s  %.4f -> %.4f  [%s]" % (names.get(k, k), b, t, arrow))
    lines.append("=" * 62)
    return "\n".join(lines)
