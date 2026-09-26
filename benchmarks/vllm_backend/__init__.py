"""PQ-HSA ↔ vLLM V1 integration (vLLM 0.8.5.post1; 0.29 via compat_v029).

No vLLM source edits. Decode-only wrap. The host is FlashAttentionImpl
(FlashInfer prefill is numerically unsafe on the 0.8.5 stack).

Imports are lazy so sitecustomize / worker spawn can load ``worker_hook``
without initializing CUDA.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "install_pq_hsa_flashattn_backend",
    "install_pq_hsa_decode_backend",
    "install_pq_hsa_vllm_backend",
    "pq_hsa_runtime_stats",
    "reset_pq_hsa_runtime_stats",
    "apply_worker_hook_env",
    "install_in_process",
    "vllm_general_plugin",
]


def __getattr__(name: str) -> Any:
    if name in {
        "install_pq_hsa_flashattn_backend",
        "pq_hsa_runtime_stats",
        "reset_pq_hsa_runtime_stats",
    }:
        from benchmarks.vllm_backend.pq_hsa_flashattn_wrap import (
            install_pq_hsa_flashattn_backend,
            pq_hsa_runtime_stats,
            reset_pq_hsa_runtime_stats,
        )

        mapping = {
            "install_pq_hsa_flashattn_backend": install_pq_hsa_flashattn_backend,
            "pq_hsa_runtime_stats": pq_hsa_runtime_stats,
            "reset_pq_hsa_runtime_stats": reset_pq_hsa_runtime_stats,
        }
        return mapping[name]
    if name == "install_pq_hsa_decode_backend":
        from benchmarks.vllm_backend.pq_hsa_flashinfer_wrap import (
            install_pq_hsa_decode_backend,
        )

        return install_pq_hsa_decode_backend
    if name == "install_pq_hsa_vllm_backend":
        from benchmarks.vllm_backend.pq_hsa_vllm_backend import (
            install_pq_hsa_vllm_backend,
        )

        return install_pq_hsa_vllm_backend
    if name in {"apply_worker_hook_env", "install_in_process", "vllm_general_plugin"}:
        from benchmarks.vllm_backend import worker_hook

        return getattr(worker_hook, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
