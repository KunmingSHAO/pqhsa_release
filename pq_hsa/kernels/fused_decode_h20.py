"""Fused PQ-HSA decode epilogue for Hopper / H20.

The LUT scan already writes ``[H,G,N]`` scores (needed for exact top-k).
Everything after that in ``_cg_forward_static`` was a pile of aten
topk / gather / GEMM / softmax / scatter ops. This module folds the
post-scan hybrid into one Triton launch:

  exact gather + exact logits + full-region logits + row-max +
  hybrid softmax + centroid background + output.

Top-k itself stays ``torch.topk`` (exact, no ``_approx_topk``). Paper knobs
unchanged. Not bit-identical to the aten graph (fp16 vs fp32 mixes); passkey
is the quality gate.
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


def is_fused_decode_available() -> bool:
    return triton is not None and tl is not None and _epilogue_kernel is not None


if triton is not None and tl is not None:

    @triton.jit
    def _epilogue_kernel(
        q_ptr, full_k_ptr, full_v_ptr, mask_ptr,
        topk_idx_ptr, topk_approx_ptr,
        ret_global_ptr,
        sb_k_ptr, sb_v_ptr,
        list_ids_ptr,
        bg_mass_ptr, centroids_ptr,
        ret_row_max_ptr, ret_exp_sum_ptr,
        out_ptr,
        H, G, D, N, K, F, L, CAP,
        stride_q_h, stride_q_g,
        stride_fk_h, stride_fv_h,
        stride_mask_h, stride_mask_g,
        stride_tk_h, stride_tk_g,
        stride_ta_h, stride_ta_g,
        stride_rg_h,
        stride_sbk_h, stride_sbv_h,
        stride_lid_h,
        stride_bg_h, stride_bg_g,
        stride_c_h,
        stride_out_h, stride_out_g,
        SCALE,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        h = tl.program_id(0)
        g = tl.program_id(1)
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        log2e = 1.4426950408889634

        q = tl.load(
            q_ptr + h * stride_q_h + g * stride_q_g + d_offs,
            mask=d_mask, other=0.0,
        ).to(tl.float32)

        # ---- full region ----
        full_max = -float("inf")
        full_lse = 0.0
        full_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for f0 in range(0, F, BLOCK_F):
            f_offs = f0 + tl.arange(0, BLOCK_F)
            f_mask = f_offs < F
            k = tl.load(
                full_k_ptr + h * stride_fk_h + f_offs[:, None] * D + d_offs[None, :],
                mask=f_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            logits = tl.sum(k * q[None, :], axis=1) * SCALE
            add = tl.load(
                mask_ptr + h * stride_mask_h + g * stride_mask_g + f_offs,
                mask=f_mask, other=-float("inf"),
            ).to(tl.float32)
            logits = tl.where(f_mask, logits + add, -float("inf"))
            tile_max = tl.max(logits, axis=0)
            new_max = tl.maximum(full_max, tile_max)
            scale_old = tl.math.exp2((full_max - new_max) * log2e)
            exps = tl.where(f_mask, tl.math.exp2((logits - new_max) * log2e), 0.0)
            v = tl.load(
                full_v_ptr + h * stride_fv_h + f_offs[:, None] * D + d_offs[None, :],
                mask=f_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            full_acc = full_acc * scale_old + tl.sum(exps[:, None] * v, axis=0)
            full_lse = full_lse * scale_old + tl.sum(exps, axis=0)
            full_max = new_max

        # ---- exact gather + logits ----
        exact_max = -float("inf")
        exact_lse = 0.0
        exact_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        old_exact_lse = 0.0
        old_exact_max = -float("inf")
        for t0 in range(0, K, BLOCK_K):
            t_offs = t0 + tl.arange(0, BLOCK_K)
            t_mask = t_offs < K
            local = tl.load(
                topk_idx_ptr + h * stride_tk_h + g * stride_tk_g + t_offs,
                mask=t_mask, other=0,
            ).to(tl.int64)
            approx = tl.load(
                topk_approx_ptr + h * stride_ta_h + g * stride_ta_g + t_offs,
                mask=t_mask, other=-float("inf"),
            ).to(tl.float32)
            glob = tl.load(
                ret_global_ptr + h * stride_rg_h + local,
                mask=t_mask, other=0,
            ).to(tl.int64)
            flat = glob + h * CAP
            ek = tl.load(
                sb_k_ptr + flat[:, None] * D + d_offs[None, :],
                mask=t_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            ev = tl.load(
                sb_v_ptr + flat[:, None] * D + d_offs[None, :],
                mask=t_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            elogits = tl.sum(ek * q[None, :], axis=1) * SCALE
            elogits = tl.where(t_mask, elogits, -float("inf"))
            approx = tl.where(t_mask, approx, -float("inf"))

            tile_max = tl.max(elogits, axis=0)
            new_max = tl.maximum(exact_max, tile_max)
            scale_old = tl.math.exp2((exact_max - new_max) * log2e)
            exps = tl.where(t_mask, tl.math.exp2((elogits - new_max) * log2e), 0.0)
            exact_acc = exact_acc * scale_old + tl.sum(exps[:, None] * ev, axis=0)
            exact_lse = exact_lse * scale_old + tl.sum(exps, axis=0)
            exact_max = new_max

            a_max = tl.max(approx, axis=0)
            new_om = tl.maximum(old_exact_max, a_max)
            old_exact_lse = old_exact_lse * tl.math.exp2((old_exact_max - new_om) * log2e) + tl.sum(
                tl.where(t_mask, tl.math.exp2((approx - new_om) * log2e), 0.0),
                axis=0,
            )
            old_exact_max = new_om

            # Subtract old approx mass from the matching IVF list (hybrid).
            lids = tl.load(
                list_ids_ptr + h * stride_lid_h + local,
                mask=t_mask, other=0,
            ).to(tl.int32)
            # Defer list correction to a second small loop after global row_max.

        ret_max = tl.load(ret_row_max_ptr + h * G + g).to(tl.float32)
        ret_sum = tl.load(ret_exp_sum_ptr + h * G + g).to(tl.float32)

        row_max = tl.maximum(full_max, ret_max)
        row_max = tl.maximum(row_max, exact_max)
        row_max = tl.maximum(row_max, old_exact_max)

        full_sum = full_lse * tl.math.exp2((full_max - row_max) * log2e)
        exact_sum = exact_lse * tl.math.exp2((exact_max - row_max) * log2e)
        old_sum = old_exact_lse * tl.math.exp2((old_exact_max - row_max) * log2e)
        ret_sum_adj = ret_sum * tl.math.exp2((ret_max - row_max) * log2e)
        denom = full_sum + ret_sum_adj - old_sum + exact_sum
        denom = tl.maximum(denom, 1e-16)

        full_out = full_acc * tl.math.exp2((full_max - row_max) * log2e) / denom
        exact_out = exact_acc * tl.math.exp2((exact_max - row_max) * log2e) / denom

        # Background: load list masses, subtract old exact, weight centroids.
        # L=512 is constexpr-friendly via a Python loop over 64-wide tiles.
        bg_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for l0 in range(0, L, 64):
            l_offs = l0 + tl.arange(0, 64)
            l_mask = l_offs < L
            mass = tl.load(
                bg_mass_ptr + h * stride_bg_h + g * stride_bg_g + l_offs,
                mask=l_mask, other=0.0,
            ).to(tl.float32)
            # mass is already exp(score - retrieval_row_max) from the host reduce
            # or list_exp_sums. Rescale to row_max.
            mass = mass * tl.math.exp2((ret_max - row_max) * log2e)
            cents = tl.load(
                centroids_ptr + h * stride_c_h + l_offs[:, None] * D + d_offs[None, :],
                mask=l_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            bg_acc += tl.sum(mass[:, None] * cents, axis=0)

        # Correct background for exact tokens (subtract old approx mass).
        for t0 in range(0, K, BLOCK_K):
            t_offs = t0 + tl.arange(0, BLOCK_K)
            t_mask = t_offs < K
            local = tl.load(
                topk_idx_ptr + h * stride_tk_h + g * stride_tk_g + t_offs,
                mask=t_mask, other=0,
            ).to(tl.int64)
            approx = tl.load(
                topk_approx_ptr + h * stride_ta_h + g * stride_ta_g + t_offs,
                mask=t_mask, other=-float("inf"),
            ).to(tl.float32)
            lids = tl.load(
                list_ids_ptr + h * stride_lid_h + local,
                mask=t_mask, other=0,
            ).to(tl.int32)
            old_m = tl.where(t_mask, tl.math.exp2((approx - row_max) * log2e), 0.0)
            # Subtract old_m * centroid[lid] from bg_acc (same as scatter_add -old).
            cents = tl.load(
                centroids_ptr + h * stride_c_h + lids[:, None] * D + d_offs[None, :],
                mask=t_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            bg_acc -= tl.sum(old_m[:, None] * cents, axis=0)

        bg_out = bg_acc / denom
        out = (full_out + exact_out + bg_out).to(tl.float32)
        tl.store(
            out_ptr + h * stride_out_h + g * stride_out_g + d_offs,
            out,
            mask=d_mask,
        )

    _epilogue_kernel = _epilogue_kernel
else:
    _epilogue_kernel = None


def fused_hybrid_epilogue_h20(
    q_hg: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    mask: torch.Tensor,
    exact_local: torch.Tensor,
    exact_approx_logits: torch.Tensor,
    retrieval_global: torch.Tensor,
    shared_base_k: torch.Tensor,
    shared_base_v: torch.Tensor,
    head_offsets: torch.Tensor,
    shared_base_cap: int,
    list_ids: torch.Tensor,
    background_mass_f32: torch.Tensor,
    value_centroids: torch.Tensor,
    retrieval_row_max: torch.Tensor,
    retrieval_exp_sum: torch.Tensor,
    scale: float,
) -> Optional[torch.Tensor]:
    """Return context ``[1, H*G, 1, D]`` or None to fall back."""
    if not is_fused_decode_available():
        return None
    if shared_base_k is None or shared_base_v is None or int(shared_base_cap) <= 0:
        return None
    H, G, D = int(q_hg.shape[0]), int(q_hg.shape[1]), int(q_hg.shape[2])
    N = int(retrieval_global.shape[-1])
    K = int(exact_local.shape[-1])
    F = int(full_k.shape[1])
    L = int(value_centroids.shape[1])
    if D > 128 or G < 1 or H < 1 or K < 1:
        return None
    # head_offsets is unused in the kernel: flat index = global + h*CAP
    # because shared_base is [H, CAP, D] with contiguous heads.
    q_c = q_hg.contiguous()
    fk = full_k.contiguous()
    fv = full_v.contiguous()
    mk = mask.contiguous()
    tk = exact_local.contiguous()
    ta = exact_approx_logits.contiguous()
    rg = retrieval_global.contiguous()
    sbk = shared_base_k.reshape(H * shared_base_cap, D).contiguous()
    sbv = shared_base_v.reshape(H * shared_base_cap, D).contiguous()
    lids = list_ids.contiguous()
    bg = background_mass_f32.contiguous()
    cents = value_centroids.contiguous()
    rmax = retrieval_row_max.contiguous()
    rsum = retrieval_exp_sum.contiguous()
    out = torch.empty(H, G, D, device=q_hg.device, dtype=torch.float32)
    BLOCK_D = 128
    BLOCK_K = 64
    BLOCK_F = 64
    grid = (H, G)
    _epilogue_kernel[grid](
        q_c, fk, fv, mk,
        tk, ta, rg, sbk, sbv, lids, bg, cents, rmax, rsum, out,
        H, G, D, N, K, F, L, int(shared_base_cap),
        q_c.stride(0), q_c.stride(1),
        fk.stride(0), fv.stride(0),
        mk.stride(0), mk.stride(1),
        tk.stride(0), tk.stride(1),
        ta.stride(0), ta.stride(1),
        rg.stride(0),
        sbk.stride(0), sbv.stride(0),
        lids.stride(0),
        bg.stride(0), bg.stride(1),
        cents.stride(0),
        out.stride(0), out.stride(1),
        float(scale),
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
        BLOCK_F=BLOCK_F,
        num_warps=4,
        num_stages=2,
    )
    return out.to(dtype=q_hg.dtype).reshape(1, H * G, 1, D)


def fused_decode_enabled() -> bool:
    # Default OFF. fused epilogue was a 29.1 vs 26.9 regression.
    # Set PQ_HSA_FUSED_DECODE=1 to restore the old path.
    return os.environ.get("PQ_HSA_FUSED_DECODE", "0") == "1"
