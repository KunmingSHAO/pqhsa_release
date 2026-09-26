from __future__ import annotations

import torch

ALLOWED_KERNEL_BACKENDS = frozenset({"auto", "torch", "triton", "h20"})
# Backends that may launch a GPU LUT / list-exp kernel.
GPU_LUT_BACKENDS = frozenset({"auto", "triton", "h20"})


def is_hopper(device: torch.device | None = None) -> bool:
    """True on Hopper (sm_90), including H20."""
    if not torch.cuda.is_available():
        return False
    idx = 0 if device is None else (device.index or 0)
    major, _minor = torch.cuda.get_device_capability(idx)
    return major >= 9


def resolve_lut_backend(backend: str, device: torch.device) -> str:
    """Map kernel_backend to an executable LUT implementation."""
    if backend not in ALLOWED_KERNEL_BACKENDS:
        raise ValueError("kernel_backend must be 'auto', 'torch', 'triton', or 'h20'")
    if backend != "auto":
        return backend
    if device.type == "cuda":
        from pq_hsa.kernels.triton_lut_scan import is_triton_available

        if is_triton_available() and is_hopper(device):
            return "h20"
        if is_triton_available():
            return "triton"
    return "torch"
