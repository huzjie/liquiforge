# -*- coding: utf-8 -*-
"""Runtime configuration with layered defaults (code default < recipe yaml < cli)."""
from __future__ import annotations

import copy

from .utils.yamlish import yaml_loads

# Model hyper-parameters for the tri-hybrid backbone.
DEFAULTS = {
    "model": {
        "dim": 64,            # residual stream width
        "n_layers": 4,        # tri-hybrid blocks
        "n_heads": 4,         # attention heads (for linear attention feature map)
        "ffn_mult": 2,
        "n_experts": 6,       # sparse MoE experts
        "top_k": 2,           # routed experts per token
        "conv_kernel": 3,     # depthwise short-conv kernel width
        "vocab_size": 256,    # text vocabulary (byte-level)
        "max_len": 256,
        "n_patch": 49,        # 7x7 vision patches
        "audio_bins": 80,     # mel bins
        "codebook": 256,      # audio VQ codebook size
        "n_actions": 16,      # decision head action space
    },
    "training": {
        "lr": 0.1,
        "epochs": 20,
        "target_skill": 0.95,
        "seed": 20261028,
    },
    "serving": {"host": "127.0.0.1", "port": 8787},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(recipe: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if recipe:
        try:
            with open(recipe, "r", encoding="utf-8") as f:
                user = yaml_loads(f.read())
            cfg = deep_merge(cfg, user)
        except FileNotFoundError:
            raise SystemExit("[config] recipe not found: %s" % recipe)
    return cfg
