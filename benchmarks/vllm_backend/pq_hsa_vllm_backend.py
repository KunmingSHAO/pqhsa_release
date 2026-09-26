"""Native vLLM AttentionBackend for PQ-HSA (no site-packages edit).

Registration: wrap ``current_platform.get_attn_backend_cls`` so
``get_attn_backend`` / ``resolve_obj_by_qualname`` loads ``PQHSABackend``.
Prefill and ``PQ_HSA_ENABLE!=1`` call real ``FlashAttentionImpl.forward`` (FA3).
Decode uses the shared sidecar runtime.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)

from benchmarks.vllm_backend.pq_hsa_decode_runtime import (
    STATS,
    pq_or_dense_forward,
    pq_hsa_runtime_stats,
    reset_pq_hsa_layer_state,
    reset_pq_hsa_runtime_stats,
)

PQHSA_BACKEND_QUALNAME = "benchmarks.vllm_backend.pq_hsa_vllm_backend.PQHSABackend"

_NATIVE_INSTALLED = False


class PQHSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Same paged FA metadata, plus runner handle for prefix persist."""

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int):
        metadata = super().build(
            num_reqs, num_actual_tokens, max_query_len, common_prefix_len
        )
        metadata._pq_runner = self.runner
        return metadata


class PQHSABackend(FlashAttentionBackend):
    """FLASH_ATTN V1 layout / builder; PQ decode impl."""

    pq_hsa_native = True

    @staticmethod
    def get_name() -> str:
        # Keep the in-tree enum name so vLLM's backend_name_to_enum still works.
        return "FLASH_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls():
        return PQHSAImpl

    @staticmethod
    def get_builder_cls():
        return PQHSAMetadataBuilder


class PQHSAImpl(FlashAttentionImpl):
    """Prefill / dense = FA3. Decode + PQ_HSA_ENABLE=1 = sidecar runtime."""

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return pq_or_dense_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            dense_forward=lambda: FlashAttentionImpl.forward(
                self, layer, query, key, value, kv_cache, attn_metadata, output
            ),
        )


def install_pq_hsa_vllm_backend() -> None:
    """Wrap platform backend selection. Must run before the first LLM()."""
    global _NATIVE_INSTALLED
    if _NATIVE_INSTALLED:
        STATS["installed"] = True
        STATS["backend"] = "native"
        return

    from vllm.platforms import current_platform

    plat_cls = type(current_platform)
    orig = plat_cls.get_attn_backend_cls
    if getattr(orig, "_pq_hsa_native", False):
        _NATIVE_INSTALLED = True
        STATS["installed"] = True
        STATS["backend"] = "native"
        STATS["sidecar"] = os.environ.get("PQ_HSA_SIDECAR", "fast")
        return
    orig_func = orig.__func__ if hasattr(orig, "__func__") else orig

    @classmethod
    def _wrapped(cls, selected_backend, head_size, dtype, kv_cache_dtype,
                 block_size, use_v1, use_mla):
        if os.environ.get("PQ_HSA_NATIVE_BACKEND", "0") == "1" and not use_mla:
            return PQHSA_BACKEND_QUALNAME
        return orig_func(
            cls, selected_backend, head_size, dtype, kv_cache_dtype,
            block_size, use_v1, use_mla,
        )

    _wrapped._pq_hsa_native = True  # type: ignore[attr-defined]
    plat_cls.get_attn_backend_cls = _wrapped

    try:
        from vllm.attention.selector import _cached_get_attn_backend

        _cached_get_attn_backend.cache_clear()
    except Exception:
        pass

    _NATIVE_INSTALLED = True
    STATS["installed"] = True
    STATS["backend"] = "native"
    STATS["sidecar"] = os.environ.get("PQ_HSA_SIDECAR", "fast")
    print(
        "[pq_hsa] current_platform.get_attn_backend_cls -> "
        f"{PQHSA_BACKEND_QUALNAME} "
        f"(sidecar={STATS['sidecar']}, cuda_graph={os.environ.get('PQ_USE_CUDA_GRAPH', '0')}, "
        f"fused={os.environ.get('PQ_HSA_FUSED_DECODE', '0')})",
        flush=True,
    )


__all__ = [
    "PQHSABackend",
    "PQHSAImpl",
    "PQHSAMetadataBuilder",
    "install_pq_hsa_vllm_backend",
    "pq_hsa_runtime_stats",
    "reset_pq_hsa_runtime_stats",
    "reset_pq_hsa_layer_state",
]
