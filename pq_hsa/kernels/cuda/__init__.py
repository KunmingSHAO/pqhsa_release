"""CUDA fused decode path for H20. Default OFF: ``PQ_HSA_CUDA_FUSED=0``."""

from __future__ import annotations

import os

def cuda_fused_enabled() -> bool:
    return os.environ.get("PQ_HSA_CUDA_FUSED", "0") == "1"
