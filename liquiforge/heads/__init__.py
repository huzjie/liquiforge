# -*- coding: utf-8 -*-
"""Five lightweight task heads, each fine-tuned from the same omni backbone."""
from .text_head import TextHead
from .vision_head import VisionHead
from .audio_head import AudioHead
from .omni_head import OmniHead
from .decision_head import DecisionHead

__all__ = ["TextHead", "VisionHead", "AudioHead", "OmniHead", "DecisionHead"]
