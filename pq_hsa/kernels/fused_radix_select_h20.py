"""V2: exact top-k via 8+8 radix-select (no heap, no ``[H,G,N]``).

Triton 3.2 has no ``tl.topk``. Pair-LUT scores stay in registers; each score is
mapped to an order-preserving uint16 key (IEEE fp16 bit trick). Pass A builds a
256-bin high-byte histogram and online softmax stats. A tiny tensor cumsum
finds the bin that contains the k-th largest key.

Two collect strategies (``PQ_HSA_RADIX_VARIANT``):

* ``bin_aten`` (default): Pass B writes every token with ``hi >= b*`` into a
  small buffer, then ``torch.topk`` on that buffer (``n_cand ≪ N``).
* ``full``: Pass B histograms the low 8 bits inside ``b*``; Pass C collects
  ``key > t*`` plus a capped number of ``key == t*``.

``PQ_HSA_FUSED_RADIX=0`` (default). New behaviour stays off until verified.
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


def fused_radix_enabled() -> bool:
    return os.environ.get("PQ_HSA_FUSED_RADIX", "0") == "1"


def radix_variant() -> str:
    v = os.environ.get("PQ_HSA_RADIX_VARIANT", "bin_aten")
    return v if v in ("bin_aten", "full") else "bin_aten"


def is_fused_radix_available() -> bool:
    return triton is not None and tl is not None and _pass_a_kernel is not None


def fp16_sort_key_torch(scores: torch.Tensor) -> torch.Tensor:
    """Host/device reference for the kernel's order-preserving uint16 key."""
    bits = scores.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    sign = (bits & 0x8000) != 0
    return torch.where(sign, (~bits) & 0xFFFF, bits ^ 0x8000)


def find_bin_from_hist(hist: torch.Tensor, k: torch.Tensor | int) -> tuple[torch.Tensor, torch.Tensor]:
    """``hist [..., 256]`` high-byte counts. Return ``b_star, c_above``.

    ``k`` may be a Python int or a tensor broadcastable to ``hist[..., 0]``.
    Walks bins from 255 downward; ``c_above`` is the count in bins strictly
    above ``b_star``.
    """
    if not torch.is_tensor(k):
        k = torch.full(hist.shape[:-1], int(k), device=hist.device, dtype=hist.dtype)
    else:
        k = k.to(device=hist.device, dtype=hist.dtype)
        if k.ndim < hist.ndim:
            k = k.expand(hist.shape[:-1])
    rev = hist.flip(-1)
    cs = rev.cumsum(-1)
    ge = cs >= k.unsqueeze(-1)
    idx_flip = ge.to(torch.int32).argmax(dim=-1)
    b_star = 255 - idx_flip
    prev = (cs - rev).gather(-1, idx_flip.unsqueeze(-1)).squeeze(-1)
    return b_star.to(torch.int32), prev.to(torch.int32)


if triton is not None and tl is not None:

    @triton.jit
    def _pair_lut_acc(
        packed_codes, pair_lut, list_ids, list_scores_t, token_scale,
        head_id, offsets, mask,
        packed_head_stride, packed_row_stride, list_ids_head_stride,
        num_lists, token_scale_head_stride,
        PAIRS: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr, PACKED_U32: tl.constexpr,
    ):
        garange = tl.arange(0, GROUPS)
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
        lid = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask, other=0, cache_modifier=".cg",
        ).to(tl.int32)
        acc += tl.load(
            list_scores_t + head_id * (num_lists * GROUPS) + lid[:, None] * GROUPS + garange[None, :],
            cache_modifier=".ca",
        ).to(tl.float32)
        if HAS_TOKEN_SCALE:
            scale = tl.load(
                token_scale + head_id * token_scale_head_stride + offsets,
                mask=mask, other=1.0, cache_modifier=".cg",
            ).to(tl.float32)
            acc = acc * scale[:, None]
        return acc

    @triton.jit
    def _fp16_sort_key(acc):
        bits16 = acc.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        sign = (bits16 & 0x8000) != 0
        return tl.where(sign, (~bits16) & 0xFFFF, bits16 ^ 0x8000)

    @triton.jit
    def _pass_a_hist_kernel(
        packed_codes, pair_lut, list_ids, list_scores_t, token_scale,
        hist, block_max, block_expsum,
        num_vectors, num_blocks,
        packed_head_stride, packed_row_stride, list_ids_head_stride,
        num_lists, token_scale_head_stride,
        PAIRS: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr, PACKED_U32: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_vectors
        garange = tl.arange(0, GROUPS)
        log2e = 1.4426950408889634
        acc = _pair_lut_acc(
            packed_codes, pair_lut, list_ids, list_scores_t, token_scale,
            head_id, offsets, mask,
            packed_head_stride, packed_row_stride, list_ids_head_stride,
            num_lists, token_scale_head_stride,
            PAIRS, GROUPS, BLOCK, HAS_TOKEN_SCALE, PACKED_U32,
        )
        neg_inf = tl.full((BLOCK, GROUPS), -float("inf"), dtype=tl.float32)
        scores = tl.where(mask[:, None], acc, neg_inf)
        bmax = tl.max(scores, axis=0)
        bexp = tl.sum(
            tl.where(mask[:, None], tl.math.exp2((acc - bmax[None, :]) * log2e), 0.0),
            axis=0,
        )
        rows = head_id * GROUPS + garange
        tl.store(block_max + rows * num_blocks + block_id, bmax)
        tl.store(block_expsum + rows * num_blocks + block_id, bexp)

        key = _fp16_sort_key(acc)
        hi = key >> 8
        ptrs = hist + rows[None, :] * 256 + hi
        tl.atomic_add(ptrs, tl.where(mask[:, None], 1, 0).to(tl.int32))

    @triton.jit
    def _pass_b_collect_kernel(
        packed_codes, pair_lut, list_ids, list_scores_t, token_scale,
        b_star, row_max,
        cand_val, cand_idx, count, overflow, list_sum,
        num_vectors, cap,
        packed_head_stride, packed_row_stride, list_ids_head_stride,
        num_lists, token_scale_head_stride,
        PAIRS: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr,
        LISTS: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr, PACKED_U32: tl.constexpr,
        DO_LIST_MASS: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_vectors
        garange = tl.arange(0, GROUPS)
        log2e = 1.4426950408889634
        acc = _pair_lut_acc(
            packed_codes, pair_lut, list_ids, list_scores_t, token_scale,
            head_id, offsets, mask,
            packed_head_stride, packed_row_stride, list_ids_head_stride,
            num_lists, token_scale_head_stride,
            PAIRS, GROUPS, BLOCK, HAS_TOKEN_SCALE, PACKED_U32,
        )
        lid = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask, other=0, cache_modifier=".cg",
        ).to(tl.int32)
        key = _fp16_sort_key(acc)
        hi = key >> 8
        bs = tl.load(b_star + head_id * GROUPS + garange).to(tl.int32)
        keep = mask[:, None] & (hi >= bs[None, :])
        # Expand [1,G] counter pointers to [BLOCK,G] so atomic_add matches keep.
        cnt_ptr = count + (head_id * GROUPS + garange)[None, :] + (hi * 0)
        slots = tl.atomic_add(cnt_ptr, tl.where(keep, 1, 0).to(tl.int32))
        in_cap = keep & (slots < cap)
        ov = tl.max(tl.where(keep & (slots >= cap), 1, 0))
        tl.atomic_add(overflow + head_id, ov)

        base = (head_id * GROUPS + garange) * cap
        tl.store(cand_val + base[None, :] + slots, acc, mask=in_cap)
        tl.store(
            cand_idx + base[None, :] + slots,
            offsets[:, None].to(tl.int32),
            mask=in_cap,
        )
        if DO_LIST_MASS:
            rmax = tl.load(row_max + head_id * GROUPS + garange).to(tl.float32)
            mass = tl.where(mask[:, None], tl.math.exp2((acc - rmax[None, :]) * log2e), 0.0)
            ptrs = list_sum + (head_id * GROUPS + garange)[None, :] * LISTS + lid[:, None]
            tl.atomic_add(ptrs, mass, mask=mask[:, None])

    @triton.jit
    def _pass_b_lo_hist_kernel(
        packed_codes, pair_lut, list_ids, list_scores_t, token_scale,
        b_star, hist_lo, row_max, list_sum,
        num_vectors,
        packed_head_stride, packed_row_stride, list_ids_head_stride,
        num_lists, token_scale_head_stride,
        PAIRS: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr,
        LISTS: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr, PACKED_U32: tl.constexpr,
        DO_LIST_MASS: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_vectors
        garange = tl.arange(0, GROUPS)
        log2e = 1.4426950408889634
        acc = _pair_lut_acc(
            packed_codes, pair_lut, list_ids, list_scores_t, token_scale,
            head_id, offsets, mask,
            packed_head_stride, packed_row_stride, list_ids_head_stride,
            num_lists, token_scale_head_stride,
            PAIRS, GROUPS, BLOCK, HAS_TOKEN_SCALE, PACKED_U32,
        )
        lid = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask, other=0, cache_modifier=".cg",
        ).to(tl.int32)
        key = _fp16_sort_key(acc)
        hi = key >> 8
        lo = key & 0xFF
        bs = tl.load(b_star + head_id * GROUPS + garange).to(tl.int32)
        in_bin = mask[:, None] & (hi == bs[None, :])
        ptrs = hist_lo + (head_id * GROUPS + garange)[None, :] * 256 + lo
        tl.atomic_add(ptrs, tl.where(in_bin, 1, 0).to(tl.int32))
        if DO_LIST_MASS:
            rmax = tl.load(row_max + head_id * GROUPS + garange).to(tl.float32)
            mass = tl.where(mask[:, None], tl.math.exp2((acc - rmax[None, :]) * log2e), 0.0)
            lptrs = list_sum + (head_id * GROUPS + garange)[None, :] * LISTS + lid[:, None]
            tl.atomic_add(lptrs, mass, mask=mask[:, None])

    @triton.jit
    def _pass_c_collect_kernel(
        packed_codes, pair_lut, list_ids, list_scores_t, token_scale,
        t_star, n_eq,
        cand_val, cand_idx, count_gt, count_eq,
        num_vectors, k,
        packed_head_stride, packed_row_stride, list_ids_head_stride,
        num_lists, token_scale_head_stride,
        PAIRS: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr, PACKED_U32: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_vectors
        garange = tl.arange(0, GROUPS)
        acc = _pair_lut_acc(
            packed_codes, pair_lut, list_ids, list_scores_t, token_scale,
            head_id, offsets, mask,
            packed_head_stride, packed_row_stride, list_ids_head_stride,
            num_lists, token_scale_head_stride,
            PAIRS, GROUPS, BLOCK, HAS_TOKEN_SCALE, PACKED_U32,
        )
        key = _fp16_sort_key(acc)
        ts = tl.load(t_star + head_id * GROUPS + garange).to(tl.int32)
        ne = tl.load(n_eq + head_id * GROUPS + garange).to(tl.int32)
        is_gt = mask[:, None] & (key > ts[None, :])
        is_eq = mask[:, None] & (key == ts[None, :])
        z = key * 0
        slots_gt = tl.atomic_add(
            count_gt + (head_id * GROUPS + garange)[None, :] + z,
            tl.where(is_gt, 1, 0).to(tl.int32),
        )
        slots_eq = tl.atomic_add(
            count_eq + (head_id * GROUPS + garange)[None, :] + z,
            tl.where(is_eq, 1, 0).to(tl.int32),
        )
        take_gt = is_gt & (slots_gt < k)
        take_eq = is_eq & (slots_eq < ne)
        base = (head_id * GROUPS + garange) * k
        tl.store(cand_val + base[None, :] + slots_gt, acc, mask=take_gt)
        tl.store(
            cand_idx + base[None, :] + slots_gt,
            offsets[:, None].to(tl.int32),
            mask=take_gt,
        )
        eq_slot = k - 1 - slots_eq
        tl.store(cand_val + base[None, :] + eq_slot, acc, mask=take_eq)
        tl.store(
            cand_idx + base[None, :] + eq_slot,
            offsets[:, None].to(tl.int32),
            mask=take_eq,
        )

    @triton.jit
    def _list_mass_csr_kernel(
        packed_codes, pair_lut, list_ids, list_scores_t, token_scale,
        inv_indices, inv_offsets, row_max, list_sum,
        packed_head_stride, packed_row_stride, list_ids_head_stride,
        inv_idx_head_stride, inv_off_head_stride,
        num_lists, token_scale_head_stride,
        PAIRS: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr,
        LISTS: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr, PACKED_U32: tl.constexpr,
    ):
        """Per-list pair-LUT mass via CSR (no atomics). Grid ``(L, H)``."""
        list_id = tl.program_id(0)
        head_id = tl.program_id(1)
        lo = tl.load(inv_offsets + head_id * inv_off_head_stride + list_id)
        hi = tl.load(inv_offsets + head_id * inv_off_head_stride + list_id + 1)
        ntok = hi - lo
        garange = tl.arange(0, GROUPS)
        log2e = 1.4426950408889634
        rmax = tl.load(row_max + head_id * GROUPS + garange).to(tl.float32)
        acc_sum = tl.zeros((GROUPS,), dtype=tl.float32)
        lut_base = pair_lut + head_id * (PAIRS * 256 * GROUPS)
        for start in range(0, ntok, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < ntok
            tok = tl.load(
                inv_indices + head_id * inv_idx_head_stride + lo + offs,
                mask=mask, other=0, cache_modifier=".cg",
            ).to(tl.int32)
            acc = tl.zeros((BLOCK, GROUPS), dtype=tl.float32)
            if PACKED_U32:
                word = tl.load(
                    packed_codes + head_id * packed_head_stride + tok * packed_row_stride,
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
                        packed_codes + head_id * packed_head_stride + tok * packed_row_stride + pair,
                        mask=mask, other=0, cache_modifier=".cg",
                    ).to(tl.int32)
                    acc += tl.load(
                        lut_base + pair * (256 * GROUPS) + byte[:, None] * GROUPS + garange[None, :],
                        cache_modifier=".ca",
                    )
            lid = tl.load(
                list_ids + head_id * list_ids_head_stride + tok,
                mask=mask, other=0, cache_modifier=".cg",
            ).to(tl.int32)
            acc += tl.load(
                list_scores_t + head_id * (num_lists * GROUPS) + lid[:, None] * GROUPS + garange[None, :],
                cache_modifier=".ca",
            ).to(tl.float32)
            if HAS_TOKEN_SCALE:
                scale = tl.load(
                    token_scale + head_id * token_scale_head_stride + tok,
                    mask=mask, other=1.0, cache_modifier=".cg",
                ).to(tl.float32)
                acc = acc * scale[:, None]
            mass = tl.where(mask[:, None], tl.math.exp2((acc - rmax[None, :]) * log2e), 0.0)
            acc_sum += tl.sum(mass, axis=0)
        tl.store(list_sum + (head_id * GROUPS + garange) * LISTS + list_id, acc_sum)

    _pass_a_kernel = _pass_a_hist_kernel
    _pass_b_collect = _pass_b_collect_kernel
    _pass_b_lo = _pass_b_lo_hist_kernel
    _pass_c_collect = _pass_c_collect_kernel
    _list_mass_csr = _list_mass_csr_kernel
else:
    _pass_a_kernel = None
    _pass_b_collect = None
    _pass_b_lo = None
    _pass_c_collect = None
    _list_mass_csr = None


def _prep_codes(packed_codes, num_subspaces):
    packed_u32 = (
        num_subspaces == 8
        and packed_codes.shape[-1] == 4
        and packed_codes.stride(-1) == 1
        and packed_codes.stride(1) == packed_codes.shape[-1]
    )
    codes = packed_codes
    hs, rs = packed_codes.stride(0), packed_codes.stride(1)
    if packed_u32:
        codes = packed_codes.view(torch.int32)
        hs, rs = codes.stride(0), codes.stride(1)
    return codes, hs, rs, packed_u32


def _radix_cap(num_vectors: int, k: int) -> int:
    # 128K measured n_cand_max≈4k; oversized cap makes aten topk dominate.
    default = min(num_vectors, int(os.environ.get("PQ_HSA_RADIX_CAP", "8192")))
    return min(num_vectors, max(default, k))


_WS: dict = {}


def _workspace(device, heads, groups, num_blocks, cap, k, variant: str):
    key = (str(device), heads, groups, num_blocks, cap, k, variant)
    ws = _WS.get(key)
    if ws is None:
        ws = {
            "hist": torch.zeros(heads, groups, 256, device=device, dtype=torch.int32),
            "block_max": torch.empty(heads * groups, num_blocks, device=device, dtype=torch.float32),
            "block_expsum": torch.empty(heads * groups, num_blocks, device=device, dtype=torch.float32),
            "list_sum": torch.zeros(heads, groups, 512, device=device, dtype=torch.float32),
            "overflow": torch.zeros(heads, device=device, dtype=torch.int32),
            "count": torch.zeros(heads, groups, device=device, dtype=torch.int32),
            "cand_val": torch.full((heads, groups, cap), float("-inf"), device=device),
            "cand_idx": torch.zeros(heads, groups, cap, device=device, dtype=torch.int32),
            "hist_lo": torch.zeros(heads, groups, 256, device=device, dtype=torch.int32),
            "count_gt": torch.zeros(heads, groups, device=device, dtype=torch.int32),
            "count_eq": torch.zeros(heads, groups, device=device, dtype=torch.int32),
            "full_val": torch.full((heads, groups, k), float("-inf"), device=device),
            "full_idx": torch.zeros(heads, groups, k, device=device, dtype=torch.int32),
        }
        _WS[key] = ws
    else:
        ws["hist"].zero_()
        ws["list_sum"].zero_()
        ws["overflow"].zero_()
        ws["count"].zero_()
        ws["cand_val"].fill_(float("-inf"))
        if variant == "full":
            ws["hist_lo"].zero_()
            ws["count_gt"].zero_()
            ws["count_eq"].zero_()
            ws["full_val"].fill_(float("-inf"))
    return ws


def fused_radix_select_h20(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    list_ids: torch.Tensor,
    list_scores: torch.Tensor,
    *,
    num_subspaces: int,
    k: int,
    token_scale: torch.Tensor | None = None,
    variant: str | None = None,
    inv_offsets: torch.Tensor | None = None,
    inv_indices: torch.Tensor | None = None,
) -> dict:
    """Exact retrieval top-k + hybrid stats. Does not allocate ``[H,G,N]`` scores."""
    if not is_fused_radix_available():
        raise RuntimeError("fused radix-select requires Triton")
    heads, num_vectors, _ = packed_codes.shape
    groups = int(lut.shape[1])
    num_lists = int(list_scores.shape[-1])
    k = min(max(int(k), 1), num_vectors)
    variant = variant or radix_variant()
    if groups & (groups - 1) != 0 or num_subspaces % 2 != 0:
        raise ValueError("radix-select needs power-of-2 G and even M")
    if num_lists > 512:
        raise ValueError("list mass supports L<=512")

    codes, hs, rs, packed_u32 = _prep_codes(packed_codes, num_subspaces)
    pair_lut, list_scores_t = _build_pair_tables(lut, list_scores)
    pairs = num_subspaces // 2
    block, nwarps, nstages = _h20_pair_heuristics(num_vectors, fused=False)
    num_blocks = triton.cdiv(num_vectors, block)
    device = packed_codes.device
    _ts = token_scale if token_scale is not None else packed_codes
    has_ts = token_scale is not None
    cap = _radix_cap(num_vectors, k)
    use_csr = (
        inv_offsets is not None
        and inv_indices is not None
        and _list_mass_csr is not None
    )

    ws = _workspace(device, heads, groups, num_blocks, cap, k, variant)
    hist = ws["hist"]
    block_max = ws["block_max"]
    block_expsum = ws["block_expsum"]
    grid = (num_blocks, heads)
    common = dict(
        PAIRS=pairs, GROUPS=groups, BLOCK=block,
        HAS_TOKEN_SCALE=has_ts, PACKED_U32=packed_u32,
        num_warps=nwarps, num_stages=nstages,
    )
    with torch.cuda.device(device):
        _pass_a_kernel[grid](
            codes, pair_lut, list_ids, list_scores_t, _ts,
            hist.reshape(-1), block_max, block_expsum,
            num_vectors, num_blocks,
            hs, rs, list_ids.stride(0), num_lists,
            token_scale.stride(0) if has_ts else 0,
            **common,
        )

    bm = block_max.view(heads, groups, num_blocks)
    be = block_expsum.view(heads, groups, num_blocks)
    row_max = bm.amax(dim=-1)
    ret_exp = (be * torch.exp(bm - row_max.unsqueeze(-1))).sum(-1)
    b_star, c_above = find_bin_from_hist(hist, k)

    list_sum = ws["list_sum"]
    overflow = ws["overflow"]
    do_atomic_mass = not use_csr

    def _csr_mass():
        with torch.cuda.device(device):
            _list_mass_csr[(num_lists, heads)](
                codes, pair_lut, list_ids, list_scores_t, _ts,
                inv_indices, inv_offsets, row_max.reshape(-1), list_sum.reshape(-1),
                hs, rs, list_ids.stride(0),
                inv_indices.stride(0), inv_offsets.stride(0),
                num_lists, token_scale.stride(0) if has_ts else 0,
                LISTS=512, **common,
            )

    if variant == "full":
        hist_lo = ws["hist_lo"]
        with torch.cuda.device(device):
            _pass_b_lo[grid](
                codes, pair_lut, list_ids, list_scores_t, _ts,
                b_star.reshape(-1), hist_lo.reshape(-1), row_max.reshape(-1),
                list_sum.reshape(-1),
                num_vectors,
                hs, rs, list_ids.stride(0), num_lists,
                token_scale.stride(0) if has_ts else 0,
                LISTS=512, DO_LIST_MASS=do_atomic_mass, **common,
            )
        need = (k - c_above).clamp(min=0)
        lo_star, lo_above = find_bin_from_hist(hist_lo, need)
        n_eq = (need - lo_above).clamp(min=0)
        t_star = (b_star << 8) + lo_star
        count_gt = ws["count_gt"]
        count_eq = ws["count_eq"]
        topk_val = ws["full_val"]
        topk_idx = ws["full_idx"]
        with torch.cuda.device(device):
            _pass_c_collect[grid](
                codes, pair_lut, list_ids, list_scores_t, _ts,
                t_star.reshape(-1), n_eq.reshape(-1),
                topk_val.reshape(-1), topk_idx.reshape(-1),
                count_gt.reshape(-1), count_eq.reshape(-1),
                num_vectors, k,
                hs, rs, list_ids.stride(0), num_lists,
                token_scale.stride(0) if has_ts else 0,
                **common,
            )
        if use_csr:
            _csr_mass()
        topk_val, ord_ = torch.sort(topk_val, dim=-1, descending=True)
        topk_idx = torch.gather(topk_idx.to(torch.int64), 2, ord_)
        return {
            "topk_val": topk_val,
            "topk_idx": topk_idx.clamp(0, num_vectors - 1),
            "row_max": row_max,
            "list_mass": list_sum[:, :, :num_lists],
            "ret_exp_sum": ret_exp,
            "b_star": b_star,
            "n_cand": count_gt + n_eq,
            "overflow": overflow,
            "variant": "full",
            "c_above": c_above,
            "t_star": t_star,
            "n_eq": n_eq,
        }

    cand_val = ws["cand_val"]
    cand_idx = ws["cand_idx"]
    count = ws["count"]
    with torch.cuda.device(device):
        _pass_b_collect[grid](
            codes, pair_lut, list_ids, list_scores_t, _ts,
            b_star.reshape(-1), row_max.reshape(-1),
            cand_val.reshape(-1), cand_idx.reshape(-1),
            count.reshape(-1), overflow, list_sum.reshape(-1),
            num_vectors, cap,
            hs, rs, list_ids.stride(0), num_lists,
            token_scale.stride(0) if has_ts else 0,
            LISTS=512, DO_LIST_MASS=do_atomic_mass, **common,
        )
    if use_csr:
        _csr_mass()
    topk_val, loc = torch.topk(cand_val, k=k, dim=-1, sorted=False)
    topk_idx = torch.gather(cand_idx.to(torch.int64), 2, loc).clamp(0, num_vectors - 1)
    n_expect = c_above + hist.gather(-1, b_star.long().unsqueeze(-1)).squeeze(-1)
    return {
        "topk_val": topk_val,
        "topk_idx": topk_idx,
        "row_max": row_max,
        "list_mass": list_sum[:, :, :num_lists],
        "ret_exp_sum": ret_exp,
        "b_star": b_star,
        "n_cand": count,
        "n_expect": n_expect,
        "overflow": overflow,
        "variant": "bin_aten",
        "c_above": c_above,
        "cap": cap,
    }


def fused_radix_decode_h20(
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
    inv_indices: torch.Tensor | None = None,
) -> Optional[torch.Tensor]:
    """Scan+radix-select+list-mass then epilogue. Graph-safe (no ``.item()``)."""
    if not is_fused_radix_available():
        return None
    if shared_base_k is None or shared_base_v is None or int(shared_base_cap) <= 0:
        return None
    try:
        from pq_hsa.kernels.fused_decode_h20 import fused_hybrid_epilogue_h20
    except Exception:
        return None
    sel = fused_radix_select_h20(
        packed_codes, lut, list_ids, list_scores,
        num_subspaces=num_subspaces, k=k, token_scale=token_scale,
        inv_offsets=inv_offsets, inv_indices=inv_indices,
    )
    idx = sel["topk_idx"]
    val = sel["topk_val"]
    idx_sorted, order = torch.sort(idx, dim=-1)
    val_sorted = torch.gather(val, 2, order)
    return fused_hybrid_epilogue_h20(
        q_hg, full_k, full_v, mask,
        idx_sorted, val_sorted,
        retrieval_global, shared_base_k, shared_base_v, None,
        int(shared_base_cap), list_ids, sel["list_mass"], value_centroids,
        sel["row_max"], sel["ret_exp_sum"], scale,
    )
