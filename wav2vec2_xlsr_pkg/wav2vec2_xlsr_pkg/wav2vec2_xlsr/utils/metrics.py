from __future__ import annotations
from typing import List
from jiwer import wer

def compute_wer(refs: List[str], hyps: List[str]) -> float:
    return wer(refs, hyps)
