"""Python wrapper for the CUDA fused decode path. Default OFF."""

from __future__ import annotations

import os
from typing import Optional

import torch

_MOD = None
_LOAD_ERR: str | None = None


def cuda_fused_enabled() -> bool:
    return os.environ.get("PQ_HSA_CUDA_FUSED", "0") == "1"


def _mod():
    global _MOD, _LOAD_ERR
    if _MOD is not None:
        return _MOD
    if _LOAD_ERR is not None:
        raise RuntimeError(_LOAD_ERR)
    try:
        from pq_hsa.kernels.cuda.compile import compile_extension

        _MOD = compile_extension("pq_hsa_fused_h20_l", ["pq_fused_h20.cu"])
        return _MOD
    except Exception as exc:
        _LOAD_ERR = f"{type(exc).__name__}: {exc}"
        raise


def is_cuda_fused_available() -> bool:
    try:
        _mod()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Extra GQA-group instances of the SAME translation unit.
#
# `pq_fused_h20.cu` specialises on the GQA group at compile time (`kG`).  Until
# now only G=4 existed, so every G!=4 model silently fell back to the all-aten
# epilogue.  We now compile the same source again with
# -DPQ_HSA_KG=<G> under a distinct extension name, one build per G.
#
# The default build is untouched: name `pq_hsa_fused_h20_l`, no -D flag, so the
# preprocessed source and the cached .so for G=4 are byte-identical to before.
# The extra instances are reached ONLY when PQ_HSA_CUDA_GQA_MULTI=1 (default 0),
# i.e. they are opt-in on top of the already opt-in PQ_HSA_CUDA_ATTEND_ONLY.
# ---------------------------------------------------------------------------
_G_INSTANCES = (4, 5, 8, 16)  # +16 (Qwen3-235B, 64Q/4KV)
_MODS: dict = {}
_MOD_ERRS: dict = {}


def gqa_multi_enabled() -> bool:
    return os.environ.get("PQ_HSA_CUDA_GQA_MULTI", "0") == "1"


def supported_groups() -> tuple:
    """GQA groups this build is willing to instantiate (env-overridable)."""
    raw = os.environ.get("PQ_HSA_CUDA_GQA_SET", "").replace(",", " ").split()
    if raw:
        try:
            return tuple(sorted({int(x) for x in raw}))
        except ValueError:
            pass
    return _G_INSTANCES


def _mod_g(G: int):
    """Module instance compiled for GQA group ``G``.  G=4 is the default build."""
    G = int(G)
    if G == 4:
        return _mod()
    if G in _MODS:
        return _MODS[G]
    if G in _MOD_ERRS:
        raise RuntimeError(_MOD_ERRS[G])
    if G not in supported_groups():
        raise ValueError(f"no CUDA instance for GQA group G={G}")
    try:
        from pq_hsa.kernels.cuda.compile import compile_extension

        _extra = [f"-DPQ_HSA_KG={G}"]
        if G > 8:
            # Scan dynamic smem ~ kG * kMaxTile; halve the tile so the
            # kG=16 instance stays under the 227 KiB H20 opt-in limit.
            _extra.append("-DPQ_HSA_MAXTILE=2048")
        m = compile_extension(
            f"pq_hsa_fused_h20_l_g{G}",
            ["pq_fused_h20.cu"],
            extra_cuda=_extra,
        )
        _MODS[G] = m
        return m
    except Exception as exc:
        _MOD_ERRS[G] = f"{type(exc).__name__}: {exc}"
        raise


def gqa_instance_available(G: int) -> bool:
    try:
        _mod_g(int(G))
        return True
    except Exception:
        return False


def _resolve_mod(G: int):
    """G=4 -> default module.  G!=4 -> instance, but only under the opt-in env.

    Returns None when the caller must fall back (exactly the previous
    behaviour whenever PQ_HSA_CUDA_GQA_MULTI is not 1).
    """
    G = int(G)
    if G == 4:
        return _mod()
    if not gqa_multi_enabled():
        return None
    return _mod_g(G)


def _prep(packed_codes: torch.Tensor, lut: torch.Tensor, list_scores: torch.Tensor):
    from pq_hsa.kernels.triton_lut_scan_h20 import _build_pair_tables

    if packed_codes.ndim != 3 or packed_codes.shape[-1] != 4:
        raise ValueError("packed codes must be [H,N,4] uint8")
    if lut.shape[2] != 8:
        raise ValueError("CUDA fused path requires M=8")
    if int(lut.shape[1]) != 4 and not (
        gqa_multi_enabled() and int(lut.shape[1]) in supported_groups()
    ):
        raise ValueError(f"CUDA fused path has no instance for G={int(lut.shape[1])}")
    codes = packed_codes.contiguous()
    if codes.stride(-1) != 1 or codes.stride(1) != 4:
        codes = codes.contiguous()
    packed_u32 = codes.view(torch.int32).reshape(codes.shape[0], codes.shape[1]).contiguous()
    pair_lut, list_scores_t = _build_pair_tables(lut.contiguous(), list_scores.contiguous())
    return packed_u32.contiguous(), pair_lut.contiguous(), list_scores_t.contiguous()


def _mass_mode() -> int:
    raw = os.environ.get("PQ_HSA_CUDA_MASS", "a").strip().lower()
    if raw in ("1", "b"):
        return 1
    if raw in ("2", "c"):
        return 2
    return 0


def _merge_threads() -> int:
    return int(os.environ.get("PQ_HSA_CUDA_MERGETH", "1024"))


def _attend_splits() -> int:
    return int(os.environ.get("PQ_HSA_CUDA_SPLITS", "8"))


def _attend_mode() -> int:
    return int(os.environ.get("PQ_HSA_CUDA_ATTEND_MODE", "0"))


def cuda_scan_select(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    list_ids: torch.Tensor,
    list_scores: torch.Tensor,
    *,
    num_subspaces: int,
    k: int,
    token_scale: torch.Tensor | None = None,
    num_ctas: int = 0,
    mass_mode: int | None = None,
    merge_threads: int | None = None,
    write_scores: bool = False,
    inv_offsets: torch.Tensor | None = None,
) -> dict:
    if num_subspaces != 8:
        raise ValueError("cuda_scan_select needs M=8")
    packed_u32, pair_lut, list_t = _prep(packed_codes, lut, list_scores)
    lids = list_ids.contiguous()
    if lids.dtype != torch.int32:
        lids = lids.to(torch.int32)
    ts = token_scale.contiguous() if token_scale is not None else packed_codes.new_empty(0)
    if token_scale is not None and ts.dtype != torch.float16:
        ts = ts.to(torch.float16)
    mm = _mass_mode() if mass_mode is None else int(mass_mode)
    mt = _merge_threads() if merge_threads is None else int(merge_threads)
    H, N = int(packed_u32.shape[0]), int(packed_u32.shape[1])
    _G = int(lut.shape[1])
    scores = packed_codes.new_empty(0)
    if write_scores or mm == 1:
        scores = torch.empty(H, _G, N, device=packed_u32.device, dtype=torch.float16)
    outs = _mod_g(_G).pq_scan_select(
        packed_u32, pair_lut, lids, list_t, ts, int(k), int(num_ctas),
        int(mm), int(mt), scores,
    )
    out_val, out_idx, row_max, list_mass, ret_exp = outs[:5]
    if mm == 1 and inv_offsets is not None:
        from pq_hsa.kernels.triton_lut_scan import list_exp_sums_sorted_multihead_triton

        list_mass = list_exp_sums_sorted_multihead_triton(
            scores.float(), row_max, inv_offsets, num_lists=int(inv_offsets.shape[1] - 1)
        )
    extra = {}
    if len(outs) >= 8:
        extra = {"list_partial": outs[5], "block_max": outs[6], "block_expsum": outs[7]}
    if len(outs) >= 9:
        extra["clocks"] = outs[8]
    if scores.numel() > 0:
        extra["scores"] = scores
    return {
        "topk_val": out_val,
        "topk_idx": out_idx.to(torch.int64),
        "row_max": row_max,
        "list_mass": list_mass,
        "ret_exp_sum": ret_exp,
        **extra,
    }


def cuda_scan_only(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    list_ids: torch.Tensor,
    list_scores: torch.Tensor,
    *,
    num_subspaces: int,
    k: int,
    token_scale: torch.Tensor | None = None,
    num_ctas: int = 0,
    mass_mode: int = 0,
    write_scores: bool = False,
) -> dict:
    packed_u32, pair_lut, list_t = _prep(packed_codes, lut, list_scores)
    lids = list_ids.contiguous()
    if lids.dtype != torch.int32:
        lids = lids.to(torch.int32)
    ts = token_scale.contiguous() if token_scale is not None else packed_codes.new_empty(0)
    if token_scale is not None and ts.dtype != torch.float16:
        ts = ts.to(torch.float16)
    H, N = int(packed_u32.shape[0]), int(packed_u32.shape[1])
    _G = int(lut.shape[1])
    scores = packed_codes.new_empty(0)
    if write_scores or mass_mode == 1:
        scores = torch.empty(H, _G, N, device=packed_u32.device, dtype=torch.float16)
    outs = _mod_g(_G).pq_scan_only(
        packed_u32, pair_lut, lids, list_t, ts, int(k), int(num_ctas), int(mass_mode), scores,
    )
    rec = {
        "cand_idx": outs[0],
        "cand_val": outs[1],
        "block_max": outs[2],
        "block_expsum": outs[3],
        "list_partial": outs[4],
        "clocks": outs[5],
    }
    if scores.numel() > 0:
        rec["scores"] = scores
    return rec


def cuda_merge_only(
    cand_idx: torch.Tensor,
    cand_val: torch.Tensor,
    block_max: torch.Tensor,
    block_expsum: torch.Tensor,
    list_partial: torch.Tensor,
    *,
    L: int,
    merge_threads: int = 1024,
    do_mass: bool = True,
) -> dict:
    outs = _mod().pq_merge_only(
        cand_idx, cand_val, block_max, block_expsum, list_partial,
        int(L), int(merge_threads), 1 if do_mass else 0,
    )
    return {
        "topk_val": outs[0],
        "topk_idx": outs[1].to(torch.int64),
        "row_max": outs[2],
        "list_mass": outs[3],
        "ret_exp_sum": outs[4],
    }


def cuda_fused_decode(
    q_hg: torch.Tensor,
    lut: torch.Tensor,
    list_scores: torch.Tensor,
    packed_codes: torch.Tensor,
    list_ids: torch.Tensor,
    *,
    num_subspaces: int,
    k: int,
    token_scale: torch.Tensor | None,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    mask: torch.Tensor,
    retrieval_global: torch.Tensor,
    shared_base_k: torch.Tensor,
    shared_base_v: torch.Tensor,
    shared_base_cap: int,
    value_centroids: torch.Tensor,
    scale: float,
    inv_offsets: torch.Tensor | None = None,
    n_splits: int | None = None,
    attend_mode: int | None = None,
    mass_mode: int | None = None,
    num_ctas: int = 0,
) -> Optional[torch.Tensor]:
    if shared_base_k is None or int(shared_base_cap) <= 0:
        return None
    sel = cuda_scan_select(
        packed_codes, lut, list_ids, list_scores,
        num_subspaces=num_subspaces, k=k, token_scale=token_scale,
        num_ctas=num_ctas, mass_mode=mass_mode, inv_offsets=inv_offsets,
    )
    idx = sel["topk_idx"].to(torch.int32)
    idx_s, order = torch.sort(idx.to(torch.int64), dim=-1)
    val_s = torch.gather(sel["topk_val"], 2, order)
    sbk = shared_base_k.reshape(int(shared_base_k.shape[0]) * int(shared_base_cap), -1).contiguous()
    sbv = shared_base_v.reshape(int(shared_base_v.shape[0]) * int(shared_base_cap), -1).contiguous()
    ns = _attend_splits() if n_splits is None else int(n_splits)
    md = _attend_mode() if attend_mode is None else int(attend_mode)
    ctx = _mod_g(int(q_hg.shape[1])).pq_exact_attend(
        q_hg.contiguous(),
        full_k.contiguous(),
        full_v.contiguous(),
        mask.contiguous().float(),
        idx_s.to(torch.int32).contiguous(),
        val_s.contiguous(),
        retrieval_global.contiguous().to(torch.int64),
        sbk.to(q_hg.dtype),
        sbv.to(q_hg.dtype),
        list_ids.contiguous().to(torch.int32),
        sel["list_mass"].contiguous(),
        value_centroids.contiguous(),
        sel["row_max"].contiguous(),
        sel["ret_exp_sum"].contiguous(),
        int(shared_base_cap),
        float(scale),
        int(ns),
        int(md),
    )
    H, G, D = int(q_hg.shape[0]), int(q_hg.shape[1]), int(q_hg.shape[2])
    return ctx.to(q_hg.dtype).reshape(1, H * G, 1, D)


# ---------------------------------------------------------------------------
# "attend-only" path.  Keep the (fastest-known) Triton pair-LUT scan
# and aten mbtopk, and replace ONLY the epilogue -- exact gather + full region +
# hybrid softmax + centroid background -- with pq_exact_attend, whose reduce
# stage is the multi-warp kernel (PQ_HSA_CUDA_REDWARPS, default 32).
#
# Switch: PQ_HSA_CUDA_ATTEND_ONLY=1 (default 0 -> the aten epilogue is untouched).
# ---------------------------------------------------------------------------


def cuda_attend_only_enabled() -> bool:
    return os.environ.get("PQ_HSA_CUDA_ATTEND_ONLY", "0") == "1"


def _attend_sort() -> bool:
    """Sort the selected local indices before the gather.

    measured sort ON at +0.039 ms/layer for identical output (the CUDA
    gather is random-access anyway, and sorting the LOCAL indices does not sort
    the GLOBAL ones it dereferences).  Default OFF.
    """
    return os.environ.get("PQ_HSA_CUDA_ATTEND_SORT", "0") == "1"


def _static_views(bh: dict, dtype: torch.dtype, cap: int):
    """Cache the dtype/layout-converted views of the SNAPSHOT-static tensors.

    ``shared_base_*`` / ``list_ids`` / ``retrieval_global`` are refreshed
    in-place by ``_inplace_refresh_snapshot``, so a *view* stays valid, but a
    dtype *copy* would go stale.  We therefore only cache zero-copy views and
    assert the source dtype instead of casting.  Anything that would need a copy
    is cached with a generation counter keyed on the tensor's data_ptr.
    """
    key = "_t8_3o_static"
    gen = (
        bh["shared_base_keys"].data_ptr(),
        bh["list_ids"].data_ptr(),
        bh["retrieval_global"].data_ptr(),
        int(bh["N"]),
        int(cap),
    )
    cached = bh.get(key)
    if cached is not None and cached["gen"] == gen:
        return cached
    H = int(bh["H"])
    sbk = bh["shared_base_keys"].reshape(H * int(cap), -1)
    sbv = bh["shared_base_vals"].reshape(H * int(cap), -1)
    if not sbk.is_contiguous():
        sbk = sbk.contiguous()
    if not sbv.is_contiguous():
        sbv = sbv.contiguous()
    if sbk.dtype != dtype:
        sbk = sbk.to(dtype)
    if sbv.dtype != dtype:
        sbv = sbv.to(dtype)
    lids = bh["list_ids"]
    lids32 = lids if lids.dtype == torch.int32 else lids.to(torch.int32)
    retg = bh["retrieval_global"]
    retg64 = retg if retg.dtype == torch.int64 else retg.to(torch.int64)
    cents = bh["value_centroids"]
    if not cents.is_contiguous():
        cents = cents.contiguous()
    out = {
        "gen": gen,
        "sbk": sbk.contiguous(),
        "sbv": sbv.contiguous(),
        "lids32": lids32.contiguous(),
        "retg": retg64.contiguous(),
        "cents": cents,
        # True when every cached tensor is a zero-copy view of the live buffer.
        "aliased": (
            lids32.data_ptr() == lids.data_ptr()
            and retg64.data_ptr() == retg.data_ptr()
            and sbk.data_ptr() == bh["shared_base_keys"].data_ptr()
        ),
    }
    bh[key] = out
    return out


def cuda_attend_only(
    bh: dict,
    q_hg: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    mask: torch.Tensor,
    topk_idx: torch.Tensor,      # [H,G,K] int64 local indices (aten topk)
    topk_val: torch.Tensor,      # [H,G,K] approx logits for those indices
    list_mass: torch.Tensor,     # [H,G,L] float32, accumulated at ret_row_max
    ret_row_max: torch.Tensor,   # [H,G] retrieval row max
    ret_exp_sum: torch.Tensor,   # [H,G] retrieval exp sum at ret_row_max
    scale: float,
    *,
    n_splits: int | None = None,
    attend_mode: int | None = None,
) -> Optional[torch.Tensor]:
    """Fused hybrid epilogue on top of the existing aten scan+top-k.

    Returns context ``[1, H*G, 1, D]`` or None (caller falls back to aten).
    """
    cap = int(bh.get("shared_base_cap", 0) or 0)
    if cap <= 0 or bh.get("shared_base_keys") is None:
        return None
    H, G, D = int(q_hg.shape[0]), int(q_hg.shape[1]), int(q_hg.shape[2])
    # Pick the instance compiled for this GQA group.  Without
    # PQ_HSA_CUDA_GQA_MULTI=1 this is exactly the old `if G != 4: return None`.
    try:
        _m = _resolve_mod(G)
    except Exception:
        return None
    if _m is None:
        return None
    st = _static_views(bh, q_hg.dtype, cap)

    if _attend_sort():
        idx_s, order = torch.sort(topk_idx, dim=-1)
        val_s = torch.gather(topk_val, 2, order)
    else:
        idx_s, val_s = topk_idx, topk_val

    ns = _attend_splits() if n_splits is None else int(n_splits)
    md = _attend_mode() if attend_mode is None else int(attend_mode)
    # The two dtype casts the epilogue needs (int64 indices -> int32,
    # fp16 logits -> fp32) are two aten launches; fold them into one.  Bit-identical.
    _idx32 = _val32 = None
    # The radix top-k / gsr_expand paths already hand over int32 indices and
    # fp32 logits -> no cast launch at all.
    if idx_s.dtype == torch.int32 and val_s.dtype == torch.float32:
        _idx32, _val32 = idx_s.contiguous(), val_s.contiguous()
    elif os.environ.get("PQ_HSA_ATTEND_RAWIDX", "0") == "1":
        try:
            _idx32, _val32 = cast_topk(idx_s, val_s)
        except Exception:
            _idx32 = _val32 = None
    if _idx32 is None:
        _idx32 = idx_s.to(torch.int32).contiguous()
        _val32 = val_s.float().contiguous()
    ctx = _m.pq_exact_attend(
        q_hg.contiguous(),
        full_k,
        full_v,
        mask if mask.dtype == torch.float32 else mask.float(),
        _idx32,
        _val32,
        st["retg"],
        st["sbk"],
        st["sbv"],
        st["lids32"],
        list_mass.float().contiguous(),
        st["cents"],
        ret_row_max.float().contiguous(),
        ret_exp_sum.float().contiguous(),
        cap,
        float(scale),
        int(ns),
        int(md),
    )
    return ctx.to(q_hg.dtype).reshape(1, H * G, 1, D)


# ---------------------------------------------------------------------------
# Paged-KV variant of the "attend-only" epilogue (opt-in,
# PQ_HSA_PAGED_ATTEND=1). Same frozen-kernel-family epilogue as
# cuda_attend_only() above (exact gather + full region + hybrid softmax +
# centroid background), except the exact-gather K/V rows are read straight
# out of vLLM's own paged kv_cache via block_table, instead of the sidecar's
# flat [H*CAP,D] shared_base_keys/vals duplicate. See
# pq_exact_attend_paged_kernel in pq_fused_h20.cu for the kernel-level design
# (only the exact-gather reader is paged; the sink+local full-region buffer is
# left untouched, same as cuda_attend_only, because it is small/fixed-size and
# not the sidecar's raw-K/V-duplicate memory item).
#
# PQ_HSA_CUDA_SPLITS>1 (the adopted production value is 8) is now
# honoured -- pq_exact_attend_paged's host wrapper dispatches to
# pq_attend_partial_nw_paged_kernel + the SAME unmodified bg/reduce kernels
# used by the non-paged S>1 path (PQ_HSA_CUDA_BGSPLITS/BGWARPS/BGUNROLL/
# REDWARPS/PARTWARPS/PARTUNROLL are read the same way, inside the CUDA
# wrapper).
# ---------------------------------------------------------------------------
def paged_attend_available(G: int = 4) -> bool:
    try:
        m = _resolve_mod(int(G))
        return m is not None and hasattr(m, "pq_exact_attend_paged")
    except Exception:
        return False


def cuda_attend_only_paged(
    bh: dict,
    q_hg: torch.Tensor,
    kv_cache: torch.Tensor,       # vLLM <=0.8.5: [2, num_blocks, block_size, num_kv_heads, D]; >=0.10: [num_blocks, num_kv_heads, block_size, 2*D]
    block_table_row: torch.Tensor,  # this request's row, [max_blocks]
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    mask: torch.Tensor,
    topk_idx: torch.Tensor,      # [H,G,K] int64 local indices (aten topk)
    topk_val: torch.Tensor,      # [H,G,K] approx logits for those indices
    list_mass: torch.Tensor,     # [H,G,L] float32, accumulated at ret_row_max
    ret_row_max: torch.Tensor,   # [H,G] retrieval row max
    ret_exp_sum: torch.Tensor,   # [H,G] retrieval exp sum at ret_row_max
    scale: float,
    *,
    n_splits: int | None = None,
    attend_mode: int | None = None,
) -> Optional[torch.Tensor]:
    """Paged-KV twin of cuda_attend_only(). Returns None -> caller falls back."""
    cap = int(bh.get("shared_base_cap", 0) or 0)
    if cap <= 0 or bh.get("shared_base_keys") is None:
        return None
    H, G, D = int(q_hg.shape[0]), int(q_hg.shape[1]), int(q_hg.shape[2])
    try:
        _m = _resolve_mod(G)
    except Exception:
        return None
    if _m is None or not hasattr(_m, "pq_exact_attend_paged"):
        return None
    # Reuses the same static-view cache as cuda_attend_only: retg/lids32/cents
    # do not depend on sb_k/sb_v, which this path simply does not read.
    st = _static_views(bh, q_hg.dtype, cap)
    ns = _attend_splits() if n_splits is None else int(n_splits)
    md = _attend_mode() if attend_mode is None else int(attend_mode)
    if topk_idx.dtype == torch.int32 and topk_val.dtype == torch.float32:
        idx32, val32 = topk_idx.contiguous(), topk_val.contiguous()
    else:
        idx32 = topk_idx.to(torch.int32).contiguous()
        val32 = topk_val.float().contiguous()
    bt32 = block_table_row.to(torch.int32).contiguous()
    # A 4-D (vLLM>=0.10) page is read in place through its strides -- never copy it.
    kvc = kv_cache if (kv_cache.dim() == 4 or kv_cache.is_contiguous()) else kv_cache.contiguous()
    ctx = _m.pq_exact_attend_paged(
        q_hg.contiguous(),
        full_k,
        full_v,
        mask if mask.dtype == torch.float32 else mask.float(),
        idx32,
        val32,
        st["retg"],
        kvc,
        bt32,
        st["lids32"],
        list_mass.float().contiguous(),
        st["cents"],
        ret_row_max.float().contiguous(),
        ret_exp_sum.float().contiguous(),
        float(scale),
        int(ns),
        int(md),
    )
    return ctx.to(q_hg.dtype).reshape(1, H * G, 1, D)


# ---------------------------------------------------------------------------
# Fused append-prep (opt-in, PQ_HSA_LEAN_APPEND=1)
# ---------------------------------------------------------------------------
def append_prep_available() -> bool:
    try:
        return hasattr(_mod(), "pq_append_prep")
    except Exception:
        return False


def append_prep(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    shared_k: torch.Tensor,
    shared_v: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    mask: torch.Tensor,
    pos: int,
    spos: int,
    q_src: Optional[torch.Tensor] = None,
    q_dst: Optional[torch.Tensor] = None,
) -> None:
    """One launch: shared-base + graph full/mask write (+ optional q staging).

    Pure data movement; bit-identical with the Triton scatter + ``q.copy_`` it
    replaces.  Caller guarantees contiguity/dtype (fp16 kv, fp32 mask).
    """
    _mod().pq_append_prep(
        new_k, new_v, shared_k, shared_v, full_k, full_v, mask, q_src, q_dst,
        int(pos), int(spos),
    )


# ---------------------------------------------------------------------------
# Fused graph-body helpers (opt-in).
# ---------------------------------------------------------------------------
def lut_prep_available() -> bool:
    try:
        return hasattr(_mod(), "pq_lut_prep")
    except Exception:
        return False


def lut_prep(
    q_pq: torch.Tensor,       # [H, G, D] fp16, contiguous
    codebooks: torch.Tensor,  # [H, M, 16, subdim] fp16
    coarse: torch.Tensor,     # [H, L, D] fp16
    scale: float,
):
    """R2: pair LUT [H,PAIRS,256,G] + list-score table [H,L,G] in ONE launch.

    Replaces fp16 GEMM(lut) + scale + fp16 GEMM(list) + scale + the Triton
    pair-build kernel (5 launches).  The intermediate [H,G,M,16] / [H,G,L]
    tensors are never materialised.
    """
    _m = _resolve_mod(int(q_pq.shape[1]))
    if _m is None:
        # No instance for this GQA group -> caller keeps the aten/Triton LUT build.
        raise ValueError(f"no CUDA lut_prep instance for G={int(q_pq.shape[1])}")
    out = _m.pq_lut_prep(
        q_pq.contiguous(), codebooks.contiguous(), coarse.contiguous(), float(scale)
    )
    return out[0], out[1]


def block_reduce_available() -> bool:
    try:
        return hasattr(_mod(), "pq_block_reduce")
    except Exception:
        return False


def block_reduce(block_max: torch.Tensor, block_expsum: torch.Tensor):
    """R3: fold per-block stats into (row_max, exp_sum) in ONE launch."""
    out = _mod().pq_block_reduce(block_max.contiguous(), block_expsum.contiguous())
    return out[0], out[1]


def cast_topk(idx64: torch.Tensor, val16: torch.Tensor):
    """R4: int64->int32 indices and fp16->fp32 logits in ONE launch."""
    out = _mod().pq_cast_topk(idx64.contiguous(), val16.contiguous())
    return out[0], out[1]


# ---------------------------------------------------------------------------
# (opt-in): TP-scalable pieces.
#   PQ_HSA_TOPK_RADIX=1     single-kernel exact top-k (int32 idx / fp32 val)
#   PQ_HSA_REDUCE_FUSED=1   split reduce folded into the partial kernel (read in C++)
#   PQ_HSA_GSR_LEAN=1       kG=1 lut_prep + one-launch GSR broadcast
# ---------------------------------------------------------------------------
def topk_radix_enabled() -> bool:
    return os.environ.get("PQ_HSA_TOPK_RADIX", "0") == "1"


def topk_radix_available() -> bool:
    try:
        return hasattr(_mod(), "pq_topk_radix")
    except Exception:
        return False


def topk_radix(scores: torch.Tensor, k: int):
    """Exact top-k over the last dim of fp16 ``scores`` -> (idx int32, val fp32).

    Same selected SET as torch.topk(scores, k, sorted=False) (tie order may differ).
    One launch, one CTA per row; independent of the GQA instance (uses the default build).
    """
    out = _mod().pq_topk_radix(scores, int(k))
    return out[1], out[0]   # (val, idx) -- mirrors torch.topk's (values, indices) order


def gsr_lean_enabled() -> bool:
    return os.environ.get("PQ_HSA_GSR_LEAN", "0") == "1"


def lut_prep_g1(q_pq: torch.Tensor, codebooks: torch.Tensor, coarse: torch.Tensor, scale: float):
    """kG=1 pair-LUT + list-score table for the group-mean query (GSR)."""
    out = _mod().pq_lut_prep_g1(q_pq.contiguous(), codebooks.contiguous(), coarse.contiguous(), float(scale))
    return out[0], out[1]


def gsr_expand(idx, val, row_max, exp_sum, mass, G: int):
    """Broadcast [H,1,*] retrieval results to [H,G,*] in ONE launch (int32/fp32 out)."""
    _m = _resolve_mod(int(G))
    if _m is None:
        raise ValueError(f"no CUDA instance for G={G}")
    out = _m.pq_gsr_expand(idx, val, row_max, exp_sum, mass)
    return out[0], out[1], out[2], out[3], out[4]
