# -*- coding: utf-8 -*-
"""Tri-hybrid backbone: linear attention + depthwise short conv + sparse MoE."""
from .linear_attention import LinearAttention
from .conv import DepthwiseConv
from .moe import SparseMoE
from .block import TriHybridBlock
from .model import OmniBackbone

__all__ = ["LinearAttention", "DepthwiseConv", "SparseMoE", "TriHybridBlock", "OmniBackbone"]
