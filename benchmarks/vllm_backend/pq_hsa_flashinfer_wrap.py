"""Monkey-patch FlashInferImpl.decode to run PQ-HSA (no vLLM source edit).

Registration note (vLLM 0.8.5 V1):
  ``VLLM_ATTENTION_BACKEND`` only maps onto the in-tree ``_Backend`` enum.
  There is no out-of-tree plugin hook for a new AttentionBackend class.
  This wrap keeps FLASHINFER as the registered backend and replaces decode.
  Set ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the patch lives in-process.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch

from benchmarks.vllm_backend.gqa_sidecar import GQASidecar
from benchmarks.vllm_backend.paged_kv import flashinfer_seq_len, gather_flashinfer_kv
from pq_hsa.attention.sparse_attention import SparseAttentionConfig
from pq_hsa.index.ivfpq import IVFPQConfig

_STATS: dict[str, Any] = {
    "installed": False,
    "pq_decode_calls": 0,
    "pq_build_calls": 0,
    "pq_build_s_total": 0.0,
    "pq_forward_s_total": 0.0,
    "flashinfer_fallback_calls": 0,
    "last_error": None,
    "layers_built": 0,
}


def pq_hsa_runtime_stats() -> dict[str, Any]:
    return dict(_STATS)


def reset_pq_hsa_runtime_stats() -> None:
    _STATS["pq_decode_calls"] = 0
    _STATS["pq_build_calls"] = 0
    _STATS["pq_build_s_total"] = 0.0
    _STATS["pq_forward_s_total"] = 0.0
    _STATS["flashinfer_fallback_calls"] = 0
    _STATS["last_error"] = None
    _STATS["layers_built"] = 0


def _paper_index_config() -> IVFPQConfig:
    return IVFPQConfig(
        num_lists=int(os.environ.get("PQ_HSA_NUM_LISTS", "512")),
        nprobe=int(os.environ.get("PQ_HSA_NPROBE", "64")),
        num_subspaces=int(os.environ.get("PQ_HSA_SUBSPACES", "8")),
        num_bits=int(os.environ.get("PQ_HSA_BITS", "4")),
        residual=True,
        coarse_max_iter=int(os.environ.get("PQ_HSA_COARSE_ITER", "1")),
        pq_max_iter=int(os.environ.get("PQ_HSA_PQ_ITER", "2")),
        pack_codes=True,
        topk_block_size=int(os.environ.get("PQ_HSA_TOPK_BLOCK", "1024")),
        kernel_backend=os.environ.get("PQ_HSA_KERNEL", "h20"),
        rotation="none",
        direction_normalize=os.environ.get("PQ_HSA_DIR_NORM", "1") == "1",
        seed=0,
    )


def _paper_attention_config(scale: float) -> SparseAttentionConfig:
    return SparseAttentionConfig(
        sink_tokens=int(os.environ.get("PQ_HSA_SINK", "4")),
        local_window=int(os.environ.get("PQ_HSA_LOCAL", "128")),
        retrieval_topk=100,
        retrieval_top_fraction=float(os.environ.get("PQ_HSA_TOP_FRAC", "0.01")),
        nprobe=int(os.environ.get("PQ_HSA_NPROBE", "64")),
        candidate_budget=int(os.environ.get("PQ_HSA_CAND_BUDGET", "4096")),
        exact_rerank=True,
        scale=float(scale),
        mode="hybrid",
        hybrid_value_mode="centroid",
        hybrid_topk_source="all_pq",
        hybrid_denominator_source="all_pq",
        index_update_interval=int(os.environ.get("PQ_HSA_UPD_INTERVAL", "256")),
        index_update_strategy="deferred",
        kv_storage="device",
        collect_attention_details=False,
        profile_attention_components=False,
    )


def _min_pq_seq() -> int:
    sink = int(os.environ.get("PQ_HSA_SINK", "4"))
    local = int(os.environ.get("PQ_HSA_LOCAL", "128"))
    return sink + local + 16


def _should_pq_decode(attn_metadata) -> bool:
    if attn_metadata is None:
        return False
    if getattr(attn_metadata, "use_cascade", False):
        return False
    if int(getattr(attn_metadata, "num_prefills", 1)) != 0:
        return False
    if int(getattr(attn_metadata, "num_decodes", 0)) != 1:
        return False
    if int(getattr(attn_metadata, "num_decode_tokens", 0)) != 1:
        return False
    try:
        seq_len = flashinfer_seq_len(attn_metadata, 0)
    except Exception:
        return False
    return seq_len >= _min_pq_seq()


def _run_pq_decode(
    impl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
) -> torch.Tensor:
    seq_len = flashinfer_seq_len(attn_metadata, 0)
    prev = getattr(impl, "_pq_seq_len", None)
    adapter: GQASidecar | None = getattr(impl, "_pq_sidecar", None)
    need_build = (
        adapter is None
        or prev is None
        or seq_len != int(prev) + 1
    )
    q = query[:1]
    if need_build:
        keys, values, gathered = gather_flashinfer_kv(kv_cache, attn_metadata, 0)
        if gathered != seq_len:
            raise RuntimeError(f"gather seq {gathered} != metadata seq {seq_len}")
        t0 = time.perf_counter()
        adapter = GQASidecar(
            _paper_index_config(),
            _paper_attention_config(impl.scale),
            num_query_heads=impl.num_heads,
            num_kv_heads=impl.num_kv_heads,
        ).build_cache(keys, values)
        _STATS["pq_build_s_total"] += time.perf_counter() - t0
        _STATS["pq_build_calls"] += 1
        _STATS["layers_built"] = int(_STATS["layers_built"]) + 1
        impl._pq_sidecar = adapter
        impl._pq_seq_len = seq_len
        t1 = time.perf_counter()
        ctx = adapter.forward(q)
        _STATS["pq_forward_s_total"] += time.perf_counter() - t1
    else:
        t1 = time.perf_counter()
        adapter.append(key[:1], value[:1])
        ctx = adapter.forward(q)
        _STATS["pq_forward_s_total"] += time.perf_counter() - t1
        impl._pq_seq_len = seq_len
    _STATS["pq_decode_calls"] += 1
    return ctx


def install_pq_hsa_decode_backend() -> None:
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    if getattr(FlashInferImpl.forward, "_pq_hsa_wrapped", False):
        _STATS["installed"] = True
        return

    orig = FlashInferImpl.forward
    skip_dense = os.environ.get("PQ_HSA_SKIP_DENSE_DECODE", "1") == "1"
    allow_fallback = os.environ.get("PQ_HSA_FALLBACK", "0") == "1"

    def wrapped(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: Optional[torch.Tensor] = None,
    ):
        if os.environ.get("PQ_HSA_ENABLE", "0") != "1" or not _should_pq_decode(attn_metadata):
            return orig(self, layer, query, key, value, kv_cache, attn_metadata, output)

        try:
            if skip_dense:
                assert output is not None
                if attn_metadata is None:
                    return output
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key,
                    value,
                    kv_cache[:, 0],
                    kv_cache[:, 1],
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )
                num_actual = int(attn_metadata.num_actual_tokens)
                ctx = _run_pq_decode(self, query, key, value, kv_cache, attn_metadata)
                output[:num_actual].copy_(ctx.to(dtype=output.dtype))
                return output

            out = orig(self, layer, query, key, value, kv_cache, attn_metadata, output)
            ctx = _run_pq_decode(self, query, key, value, kv_cache, attn_metadata)
            out[: int(attn_metadata.num_decode_tokens)].copy_(ctx.to(dtype=out.dtype))
            return out
        except Exception as exc:
            _STATS["last_error"] = f"{type(exc).__name__}: {exc}"
            _STATS["flashinfer_fallback_calls"] += 1
            if not allow_fallback:
                raise
            return orig(self, layer, query, key, value, kv_cache, attn_metadata, output)

    wrapped._pq_hsa_wrapped = True  # type: ignore[attr-defined]
    FlashInferImpl.forward = wrapped
    _STATS["installed"] = True
    print("[pq_hsa] FlashInferImpl.forward wrapped (decode → PQ-HSA sidecar)", flush=True)
