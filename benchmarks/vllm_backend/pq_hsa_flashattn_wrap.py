"""Monkey-patch FlashAttentionImpl.forward to run PQ-HSA decode (no vLLM source edit).

Why FLASH_ATTN rather than FLASHINFER:
  On this stack (vLLM 0.8.5.post1 V1 + FlashInfer 0.2.5 + H20/sm90 + fp16) the
  FlashInfer *prefill* path collapses to NaN logits for some prompt lengths
  (256 and 32768 fail standalone, 4096 passes) with no error raised. FLASH_ATTN
  is vLLM's default V1 backend on sm90 and is correct across those lengths, so it
  is both the reliable host and the more defensible same-stack dense baseline.

Set ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the patch lives in the same process.

Persist / flush live in ``pq_hsa_decode_runtime``. This wrap is the
fallback when ``PQ_HSA_NATIVE_BACKEND=0``. Native registration is
``install_pq_hsa_vllm_backend``.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from benchmarks.vllm_backend.pq_hsa_decode_runtime import (
    STATS as _STATS,
    install_flashattn_metadata_runner_hook,
    pq_hsa_runtime_stats,
    pq_or_dense_forward,
    reset_pq_hsa_layer_state,
    reset_pq_hsa_runtime_stats,
    run_pq_decode,
    should_pq_decode,
)


def _run_pq_decode(impl, query, key, value, kv_cache, attn_metadata) -> torch.Tensor:
    """Compat alias (layer unknown → key by id(impl))."""
    return run_pq_decode(impl, None, query, key, value, kv_cache, attn_metadata)


def install_pq_hsa_flashattn_backend() -> None:
    if os.environ.get("PQ_HSA_NATIVE_BACKEND", "0") == "1":
        from benchmarks.vllm_backend.pq_hsa_vllm_backend import install_pq_hsa_vllm_backend

        install_pq_hsa_vllm_backend()
        return

    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    install_flashattn_metadata_runner_hook()

    if getattr(FlashAttentionImpl.forward, "_pq_hsa_wrapped", False):
        _STATS["installed"] = True
        _STATS["backend"] = "wrap"
        return

    orig = FlashAttentionImpl.forward

    def wrapped(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: Optional[torch.Tensor] = None,
        **extra,
    ):
        # (vLLM >= 0.10): the call site passes output_scale= /
        # output_block_scale= keywords; forward them to the dense path
        # untouched. On 0.8.5.post1 ``extra`` is always empty, so the dense
        # call below is byte-identical to the original wrap.
        return pq_or_dense_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            dense_forward=lambda: orig(
                self, layer, query, key, value, kv_cache, attn_metadata, output, **extra
            ),
        )

    wrapped._pq_hsa_wrapped = True  # type: ignore[attr-defined]
    FlashAttentionImpl.forward = wrapped
    _STATS["installed"] = True
    _STATS["backend"] = "wrap"
    _STATS["sidecar"] = os.environ.get("PQ_HSA_SIDECAR", "fast")
    print(
        f"[pq_hsa] FlashAttentionImpl.forward wrapped (decode → PQ-HSA "
        f"sidecar={_STATS['sidecar']}, cuda_graph={os.environ.get('PQ_USE_CUDA_GRAPH', '0')})",
        flush=True,
    )


__all__ = [
    "install_pq_hsa_flashattn_backend",
    "pq_hsa_runtime_stats",
    "reset_pq_hsa_runtime_stats",
    "reset_pq_hsa_layer_state",
    "should_pq_decode",
]
