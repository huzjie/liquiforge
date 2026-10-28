# -*- coding: utf-8 -*-
"""liquiforge: LFM3-style tri-hybrid omni-modal unified backbone (reference implementation).

A single tri-hybrid backbone (linear attention + depthwise conv + sparse MoE)
is shared across five modalities/tasks (text / vision / audio / omni / decision),
each produced by fine-tuning one lightweight task head from the same base.
Pure Python, zero third-party dependencies.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
