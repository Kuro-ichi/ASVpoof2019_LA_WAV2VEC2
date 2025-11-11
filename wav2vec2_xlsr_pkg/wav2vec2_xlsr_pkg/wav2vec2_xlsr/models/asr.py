from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import torch
from transformers import AutoProcessor, AutoModelForCTC

@dataclass
class ASRBundle:
    processor: Any
    model: torch.nn.Module

def build_asr(model_name: str = "facebook/wav2vec2-base-960h") -> ASRBundle:
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name)
    return ASRBundle(processor=processor, model=model)
