"""Scan ⊕ tile top-k ⊕ list-mass without materializing ``[H,G,N]``.

Triton 3.2.0 has no ``tl.topk``. Tile top-k is exact iterative argmax
(``min(k, BLOCK)`` per tile) plus a tiny aten merge on ``n_blocks * min(k, BLOCK)``
candidates. List masses are a second pair-LUT pass with ``atomic_add`` against
a known ``row_max`` from pass 1 (so hybrid denom does not need full logits).

Default OFF: ``PQ_HSA_FUSED_FULL=0``. The epilogue-only kernel is not used.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

from pq_hsa.kernels.triton_lut_scan_h20 import _build_pair_tables, _h20_pair_heuristics


def fused_full_enabled() -> bool:
    return os.environ.get("PQ_HSA_FUSED_FULL", "0") == "1"


def is_fused_full_available() -> bool:
    return triton is not None and tl is not None and _tile_scan_kernel is not None


if triton is not None and tl is not None:

    @triton.jit
    def _tile_scan_topk_kernel(
        packed_codes,
        pair_lut,
        list_ids,
        list_scores_t,
        token_scale,
        cand_val,
        cand_idx,
        block_max,
        block_expsum,
        num_vectors,
        num_blocks,
        packed_head_stride,
        packed_row_stride,
        list_ids_head_stride,
        num_lists,
        token_scale_head_stride,
        kt,
        PAIRS: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK: tl.constexpr,
        KT_CAP: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr,
        PACKED_U32: tl.constexpr,
    ):
        """Pair-LUT tile: stats + exact per-tile top-``kt`` (no ``[H,G,N]`` store)."""
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_vectors
        garange = tl.arange(0, GROUPS)
        mask2 = mask[:, None]
        log2e = 1.4426950408889634

        acc = tl.zeros((BLOCK, GROUPS), dtype=tl.float32)
        lut_base = pair_lut + head_id * (PAIRS * 256 * GROUPS)
        if PACKED_U32:
            word = tl.load(
                packed_codes + head_id * packed_head_stride + offsets * packed_row_stride,
                mask=mask, other=0, cache_modifier=".cg",
            ).to(tl.uint32)
            for pair in tl.static_range(0, PAIRS):
                byte = ((word >> (8 * pair)) & 0xFF).to(tl.int32)
                acc += tl.load(
                    lut_base + pair * (256 * GROUPS) + byte[:, None] * GROUPS + garange[None, :],
                    cache_modifier=".ca",
                )
        else:
            for pair in tl.static_range(0, PAIRS):
                byte = tl.load(
                    packed_codes + head_id * packed_head_stride + offsets * packed_row_stride + pair,
                    mask=mask, other=0, cache_modifier=".cg",
                ).to(tl.int32)
                acc += tl.load(
                    lut_base + pair * (256 * GROUPS) + byte[:, None] * GROUPS + garange[None, :],
                    cache_modifier=".ca",
                )

        list_id = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask, other=0, cache_modifier=".cg",
        ).to(tl.int32)
        acc += tl.load(
            list_scores_t + head_id * (num_lists * GROUPS) + list_id[:, None] * GROUPS + garange[None, :],
            cache_modifier=".ca",
        ).to(tl.float32)
        if HAS_TOKEN_SCALE:
            scale = tl.load(
                token_scale + head_id * token_scale_head_stride + offsets,
                mask=mask, other=1.0, cache_modifier=".cg",
            ).to(tl.float32)
            acc = acc * scale[:, None]

        neg_inf = tl.full((BLOCK, GROUPS), -float("inf"), dtype=tl.float32)
        scores = tl.where(mask2, acc, neg_inf)
        bmax = tl.max(scores, axis=0)
        bexp = tl.sum(
            tl.where(mask2, tl.math.exp2((acc - bmax[None, :]) * log2e), 0.0),
            axis=0,
        )
        rows = head_id * GROUPS + garange
        tl.store(block_max + rows * num_blocks + block_id, bmax)
        tl.store(block_expsum + rows * num_blocks + block_id, bexp)

        # Exact tile top-kt: iterative argmax. ``kt`` is runtime, ``kt <= BLOCK``.
        work = scores
        ar = tl.arange(0, BLOCK)
        for ki in range(0, kt):
            mx = tl.max(work, axis=0)
            is_max = work == mx[None, :]
            idx = tl.min(tl.where(is_max, ar[:, None], BLOCK), axis=0)
            base = (
                (head_id * GROUPS + garange) * num_blocks * KT_CAP
                + block_id * KT_CAP
                + ki
            )
            tl.store(cand_val + base, mx)
            tl.store(cand_idx + base, (block_id * BLOCK + idx).to(tl.int32))
            work = tl.where(is_max, neg_inf, work)

    @triton.jit
    def _list_mass_kernel(
        packed_codes,
        pair_lut,
        list_ids,
        list_scores_t,
        token_scale,
        row_max,
        list_sum,
        num_vectors,
        packed_head_stride,
        packed_row_stride,
        list_ids_head_stride,
        num_lists,
        token_scale_head_stride,
        PAIRS: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK: tl.constexpr,
        LISTS: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr,
        PACKED_U32: tl.constexpr,
    ):
        """Second pair-LUT pass: atomic ``sum(exp(s - row_max))`` per list."""
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_vectors
        garange = tl.arange(0, GROUPS)
        log2e = 1.4426950408889634

        acc = tl.zeros((BLOCK, GROUPS), dtype=tl.float32)
        lut_base = pair_lut + head_id * (PAIRS * 256 * GROUPS)
        if PACKED_U32:
            word = tl.load(
                packed_codes + head_id * packed_head_stride + offsets * packed_row_stride,
                mask=mask, other=0, cache_modifier=".cg",
            ).to(tl.uint32)
            for pair in tl.static_range(0, PAIRS):
                byte = ((word >> (8 * pair)) & 0xFF).to(tl.int32)
                acc += tl.load(
                    lut_base + pair * (256 * GROUPS) + byte[:, None] * GROUPS + garange[None, :],
                    cache_modifier=".ca",
                )
        else:
            for pair in tl.static_range(0, PAIRS):
                byte = tl.load(
                    packed_codes + head_id * packed_head_stride + offsets * packed_row_stride + pair,
                    mask=mask, other=0, cache_modifier=".cg",
                ).to(tl.int32)
                acc += tl.load(
                    lut_base + pair * (256 * GROUPS) + byte[:, None] * GROUPS + garange[None, :],
                    cache_modifier=".ca",
                )

        list_id = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask, other=0, cache_modifier=".cg",
        ).to(tl.int32)
        acc += tl.load(
            list_scores_t + head_id * (num_lists * GROUPS) + list_id[:, None] * GROUPS + garange[None, :],
            cache_modifier=".ca",
        ).to(tl.float32)
        if HAS_TOKEN_SCALE:
            scale = tl.load(
                token_scale + head_id * token_scale_head_stride + offsets,
                mask=mask, other=1.0, cache_modifier=".cg",
            ).to(tl.float32)
            acc = acc * scale[:, None]

        rmax = tl.load(row_max + head_id * GROUPS + garange).to(tl.float32)
        mass = tl.where(mask[:, None], tl.math.exp2((acc - rmax[None, :]) * log2e), 0.0)
        ptrs = (
            list_sum
            + (head_id * GROUPS + garange)[None, :] * LISTS
            + list_id[:, None]
        )
        tl.atomic_add(ptrs, mass, mask=mask[:, None])

    _tile_scan_kernel = _tile_scan_topk_kernel
    _list_mass_kernel_impl = _list_mass_kernel
else:
    _tile_scan_kernel = None
    _list_mass_kernel_impl = None


def _choose_block(num_vectors: int, k: int) -> int:
    """Tile width so ``k < BLOCK`` when possible (exact tile top-k shrinks candidates)."""
    if k < 256:
        return 256
    if k < 512:
        return 512
    if k < 1024:
        return 1024
    return 2048


def fused_scan_topk_listmass_h20(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    list_ids: torch.Tensor,
    list_scores: torch.Tensor,
    *,
    num_subspaces: int,
    k: int,
    token_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(topk_val, topk_idx, row_max, list_exp_sums, ret_exp_sum)``.

    No ``[H,G,N]`` score tensor is allocated. ``topk_*`` are ``[H,G,k]``.
    ``list_exp_sums`` is ``[H,G,L]`` at ``row_max`` (approx-only; caller still
    folds exact logits into the hybrid row-max).
    """
    if not is_fused_full_available():
        raise RuntimeError("fused full H20 kernel requires Triton 3.x")
    if packed_codes.ndim != 3 or lut.ndim != 4:
        raise ValueError("packed [H,N,W] and lut [H,G,M,16] required")
    heads, num_vectors, packed_width = packed_codes.shape
    groups = int(lut.shape[1])
    num_lists = int(list_scores.shape[-1])
    k = min(max(int(k), 0), num_vectors)
    if k <= 0:
        raise ValueError("k must be positive")
    # Triton 3.2 has no tl.topk. Iterative tile argmax is exact but O(k*BLOCK)
    # per tile: 16K (k=163,B=256) is fine; 128K (k=1302,B=2048) measured 261 ms
    # vs aten 0.66 ms. Callers should refuse and keep the aten graph.
    if k >= int(os.environ.get("PQ_HSA_FUSED_FULL_MAX_K", "256")):
        raise RuntimeError(
            f"fused full tile-topk refuses k={k} (>= PQ_HSA_FUSED_FULL_MAX_K); "
            "iterative argmax is not competitive with aten::topk at 128K"
        )
    if groups & (groups - 1) != 0:
        raise ValueError("G must be a power of two")
    if num_subspaces % 2 != 0:
        raise ValueError("pair-LUT needs even M")

    packed_u32 = (
        num_subspaces == 8
        and packed_width == 4
        and packed_codes.stride(-1) == 1
        and packed_codes.stride(1) == packed_width
    )
    codes_launch = packed_codes
    packed_head_stride = packed_codes.stride(0)
    packed_row_stride = packed_codes.stride(1)
    if packed_u32:
        codes_launch = packed_codes.view(torch.int32)
        packed_head_stride = codes_launch.stride(0)
        packed_row_stride = codes_launch.stride(1)

    block = _choose_block(num_vectors, k)
    kt = min(k, block)
    num_blocks = triton.cdiv(num_vectors, block)
    pair_lut, list_scores_t = _build_pair_tables(lut, list_scores)
    pairs = num_subspaces // 2
    device = packed_codes.device
    _ts = token_scale if token_scale is not None else packed_codes

    cand_val = torch.full(
        (heads, groups, num_blocks, block),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    cand_idx = torch.zeros(heads, groups, num_blocks, block, device=device, dtype=torch.int32)
    block_max = torch.empty(heads * groups, num_blocks, device=device, dtype=torch.float32)
    block_expsum = torch.empty_like(block_max)

    grid = (num_blocks, heads)
    with torch.cuda.device(device):
        _tile_scan_kernel[grid](
            codes_launch,
            pair_lut,
            list_ids,
            list_scores_t,
            _ts,
            cand_val.reshape(-1),
            cand_idx.reshape(-1),
            block_max,
            block_expsum,
            num_vectors,
            num_blocks,
            packed_head_stride,
            packed_row_stride,
            list_ids.stride(0),
            num_lists,
            token_scale.stride(0) if token_scale is not None else 0,
            kt,
            PAIRS=pairs,
            GROUPS=groups,
            BLOCK=block,
            KT_CAP=block,
            HAS_TOKEN_SCALE=token_scale is not None,
            PACKED_U32=packed_u32,
            num_warps=4,
            num_stages=2,
        )

    bm = block_max.view(heads, groups, num_blocks)
    be = block_expsum.view(heads, groups, num_blocks)
    row_max = bm.amax(dim=-1)
    ret_exp_sum = (be * torch.exp(bm - row_max.unsqueeze(-1))).sum(-1)

    # Merge exact tile winners. Candidate count is n_blocks * kt << N when k < BLOCK.
    flat_v = cand_val[:, :, :, :kt].reshape(heads, groups, num_blocks * kt)
    flat_i = cand_idx[:, :, :, :kt].reshape(heads, groups, num_blocks * kt).to(torch.int64)
    if num_blocks * kt > k:
        topk_val, loc = torch.topk(flat_v, k=k, dim=-1, sorted=False)
        topk_idx = torch.gather(flat_i, 2, loc)
    else:
        topk_val, topk_idx = flat_v, flat_i
    topk_idx = topk_idx.clamp(0, num_vectors - 1)

    list_sum = torch.zeros(heads, groups, num_lists, device=device, dtype=torch.float32)
    with torch.cuda.device(device):
        _list_mass_kernel_impl[grid](
            codes_launch,
            pair_lut,
            list_ids,
            list_scores_t,
            _ts,
            row_max.reshape(-1),
            list_sum.reshape(-1),
            num_vectors,
            packed_head_stride,
            packed_row_stride,
            list_ids.stride(0),
            num_lists,
            token_scale.stride(0) if token_scale is not None else 0,
            PAIRS=pairs,
            GROUPS=groups,
            BLOCK=block,
            LISTS=num_lists if num_lists == 512 else 512,
            HAS_TOKEN_SCALE=token_scale is not None,
            PACKED_U32=packed_u32,
            num_warps=4,
            num_stages=2,
        )
    if num_lists != 512:
        # LISTS constexpr is 512; extra tails stay zero if L<512. If L>512 reject.
        if num_lists > 512:
            raise ValueError("fused full list-mass supports L<=512")
        list_sum = list_sum[:, :, :num_lists]
    return topk_val, topk_idx, row_max, list_sum, ret_exp_sum


def fused_full_decode_h20(
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
) -> Optional[torch.Tensor]:
    """Scan+topk+list-mass then earlier-style epilogue. Context ``[1,H*G,1,D]`` or None."""
    if not is_fused_full_available():
        return None
    if shared_base_k is None or shared_base_v is None or int(shared_base_cap) <= 0:
        return None
    k = min(max(int(k), 0), int(packed_codes.shape[1]))
    if k >= int(os.environ.get("PQ_HSA_FUSED_FULL_MAX_K", "256")):
        return None
    try:
        from pq_hsa.kernels.fused_decode_h20 import fused_hybrid_epilogue_h20
    except Exception:
        return None

    topk_val, topk_idx, row_max, list_mass, ret_exp = fused_scan_topk_listmass_h20(
        packed_codes,
        lut,
        list_ids,
        list_scores,
        num_subspaces=num_subspaces,
        k=k,
        token_scale=token_scale,
    )
    return fused_hybrid_epilogue_h20(
        q_hg,
        full_k,
        full_v,
        mask,
        topk_idx,
        topk_val,
        retrieval_global,
        shared_base_k,
        shared_base_v,
        None,
        int(shared_base_cap),
        list_ids,
        list_mass,
        value_centroids,
        row_max,
        ret_exp,
        scale,
    )
