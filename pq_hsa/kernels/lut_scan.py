from __future__ import annotations

from dataclasses import dataclass

import torch

from pq_hsa.kernels.device import ALLOWED_KERNEL_BACKENDS
from pq_hsa.kernels.triton_lut_scan import (
    block_topk_packed_4bit_lut_triton,
    final_topk_merge_triton,
    is_triton_available,
    score_packed_4bit_lut_batched_triton,
    score_packed_4bit_lut_multihead_list_bias_triton,
    score_packed_4bit_lut_triton,
)


@dataclass(slots=True)
class KernelTopKResult:
    indices: torch.Tensor
    scores: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_scores: torch.Tensor


def score_packed_4bit_lut(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    *,
    num_subspaces: int,
    block_size: int | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Score packed 4-bit PQ codes from a query LUT without full unpacking.

    This is the PyTorch fallback for the fused LUT-scan kernel path. It consumes
    the same uint8 nibble layout expected by future CUDA/Triton kernels:

    - ``packed_codes``: ``[N, ceil(M / 2)]`` uint8, low nibble then high nibble.
    - ``lut``: ``[M, 16]`` for one query or ``[B, M, 16]`` for batched queries.

    Returns ``[N]`` for one query or ``[B, N]`` for batched queries.
    """

    _validate_common_inputs(packed_codes, num_subspaces)
    if backend not in ALLOWED_KERNEL_BACKENDS:
        raise ValueError("backend must be 'auto', 'torch', 'triton', or 'h20'")

    # Single-query / batched scan still uses the Ampere Triton kernels. ``h20``
    # is the multihead decode path; treat it as Triton here so IVFPQ search
    # keeps working when kernel_backend=h20.
    should_use_single_query_triton = (
        lut.ndim == 2
        and (
            backend in {"triton", "h20"}
            or (
                backend == "auto"
                and is_triton_available()
                and packed_codes.is_cuda
                and lut.is_cuda
            )
        )
    )
    if should_use_single_query_triton:
        return score_packed_4bit_lut_triton(
            packed_codes,
            lut,
            num_subspaces=num_subspaces,
            block_size=256 if block_size is None else block_size,
        )

    should_use_batched_triton = (
        lut.ndim == 3
        and packed_codes.is_cuda
        and lut.is_cuda
        and (
            backend in {"triton", "h20"}
            or (
                backend == "auto"
                and is_triton_available()
            )
        )
    )
    if should_use_batched_triton:
        return score_packed_4bit_lut_batched_triton(
            packed_codes,
            lut,
            num_subspaces=num_subspaces,
            block_size=256 if block_size is None else block_size,
        )

    if lut.ndim == 2:
        _validate_lut_shape(lut, num_subspaces)
        scores = torch.zeros(packed_codes.shape[0], device=lut.device, dtype=lut.dtype)
        for subspace in range(num_subspaces):
            byte = packed_codes[:, subspace // 2]
            codes = _nibble(byte, high=(subspace % 2 == 1)).to(torch.long)
            scores += lut[subspace].gather(0, codes)
        return scores

    if lut.ndim == 3:
        _validate_lut_shape(lut, num_subspaces)
        scores = torch.zeros(
            lut.shape[0],
            packed_codes.shape[0],
            device=lut.device,
            dtype=lut.dtype,
        )
        for subspace in range(num_subspaces):
            byte = packed_codes[:, subspace // 2]
            codes = _nibble(byte, high=(subspace % 2 == 1)).to(torch.long)
            gather_index = codes.unsqueeze(0).expand(lut.shape[0], -1)
            scores += lut[:, subspace, :].gather(1, gather_index)
        return scores

    raise ValueError(f"lut must be [M, 16] or [B, M, 16], got {tuple(lut.shape)}")


def topk_packed_4bit_lut(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    *,
    num_subspaces: int,
    topk: int,
    candidate_budget: int | None = None,
    score_bias: torch.Tensor | None = None,
    score_scale: torch.Tensor | None = None,
    block_size: int | None = None,
    backend: str = "auto",
) -> KernelTopKResult:
    """Score packed 4-bit codes and select top-k with optional candidate budget.

    This is the kernel contract for future fused LUT scan + top-k. The current
    implementation is a PyTorch fallback but keeps all top-k semantics in one
    place:

    - score all input rows from packed codes and LUT;
    - add an optional per-row bias such as IVF centroid dot query;
    - multiply by an optional per-row scale such as query/key norm correction;
    - optionally preselect candidates per block;
    - optionally keep only the highest ``candidate_budget`` rows;
    - return the final top-k within that budget.

    Returned indices are local to the input ``packed_codes`` rows.
    """

    if topk <= 0:
        raise ValueError("topk must be positive")
    if lut.ndim != 2:
        raise ValueError(f"topk_packed_4bit_lut expects one-query LUT [M, 16], got {tuple(lut.shape)}")

    _validate_common_inputs(packed_codes, num_subspaces)
    if backend not in ALLOWED_KERNEL_BACKENDS:
        raise ValueError("backend must be 'auto', 'torch', 'triton', or 'h20'")

    budget_for_block = packed_codes.shape[0] if candidate_budget is None else candidate_budget
    should_use_triton_block = (
        block_size is not None
        and (
            backend in {"triton", "h20"}
            or (
                backend == "auto"
                and is_triton_available()
                and packed_codes.is_cuda
                and lut.is_cuda
            )
        )
    )
    if should_use_triton_block:
        candidate_indices, candidate_scores = block_topk_packed_4bit_lut_triton(
            packed_codes,
            lut,
            num_subspaces=num_subspaces,
            candidate_budget=max(topk, budget_for_block),
            score_bias=score_bias,
            score_scale=score_scale,
            block_size=block_size,
        )
        valid_candidates = (candidate_indices < packed_codes.shape[0]) & torch.isfinite(candidate_scores)
        candidate_indices = candidate_indices[valid_candidates]
        candidate_scores = candidate_scores[valid_candidates]
        try:
            top_indices, top_scores, budget_indices, budget_scores = final_topk_merge_triton(
                candidate_indices,
                candidate_scores,
                topk=topk,
                candidate_budget=candidate_budget,
            )
            return KernelTopKResult(
                indices=top_indices,
                scores=top_scores,
                candidate_indices=budget_indices,
                candidate_scores=budget_scores,
            )
        except ValueError:
            pass
        return _finalize_candidate_topk(
            candidate_indices,
            candidate_scores,
            topk=topk,
            candidate_budget=candidate_budget,
        )

    scores = score_packed_4bit_lut(
        packed_codes,
        lut,
        num_subspaces=num_subspaces,
        backend=backend,
    )
    if score_bias is not None:
        if score_bias.shape != scores.shape:
            raise ValueError(f"score_bias must have shape {tuple(scores.shape)}, got {tuple(score_bias.shape)}")
        scores = scores + score_bias
    if score_scale is not None:
        if score_scale.shape != scores.shape:
            raise ValueError(
                f"score_scale must have shape {tuple(scores.shape)}, got {tuple(score_scale.shape)}"
            )
        scores = scores * score_scale

    return _select_topk(scores, topk=topk, candidate_budget=candidate_budget, block_size=block_size)


def _finalize_candidate_topk(
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    *,
    topk: int,
    candidate_budget: int | None,
) -> KernelTopKResult:
    if candidate_indices.ndim != 1 or candidate_scores.ndim != 1:
        raise ValueError("candidate_indices and candidate_scores must be rank-1")
    if candidate_indices.shape != candidate_scores.shape:
        raise ValueError("candidate_indices and candidate_scores must have matching shapes")
    if candidate_scores.numel() == 0:
        raise ValueError("cannot select top-k from zero candidates")

    budget = candidate_scores.numel() if candidate_budget is None else candidate_budget
    budget = min(max(budget, topk), candidate_scores.numel())
    if budget < candidate_scores.numel():
        budget_scores, budget_order = torch.topk(candidate_scores, k=budget)
        candidate_indices = candidate_indices[budget_order]
        candidate_scores = budget_scores

    k = min(topk, candidate_scores.numel())
    top_scores, top_order = torch.topk(candidate_scores, k=k)
    top_indices = candidate_indices[top_order]
    return KernelTopKResult(
        indices=top_indices,
        scores=top_scores,
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
    )


def _select_topk(
    scores: torch.Tensor,
    *,
    topk: int,
    candidate_budget: int | None,
    block_size: int | None,
) -> KernelTopKResult:
    if scores.ndim != 1:
        raise ValueError(f"scores must be [N], got {tuple(scores.shape)}")
    if scores.numel() == 0:
        raise ValueError("cannot select top-k from zero scores")
    if block_size is not None and block_size <= 0:
        raise ValueError("block_size must be positive when provided")

    candidate_indices = torch.arange(scores.shape[0], device=scores.device, dtype=torch.long)
    candidate_scores = scores
    budget = candidate_scores.numel() if candidate_budget is None else candidate_budget
    budget = min(max(budget, topk), candidate_scores.numel())

    if block_size is not None and block_size < candidate_scores.numel():
        block_indices = []
        block_scores = []
        for start in range(0, candidate_scores.numel(), block_size):
            end = min(start + block_size, candidate_scores.numel())
            current_scores = candidate_scores[start:end]
            keep = min(budget, current_scores.numel())
            current_top_scores, current_order = torch.topk(current_scores, k=keep)
            block_indices.append(candidate_indices[start:end][current_order])
            block_scores.append(current_top_scores)
        candidate_indices = torch.cat(block_indices, dim=0)
        candidate_scores = torch.cat(block_scores, dim=0)

    if budget < candidate_scores.numel():
        budget_scores, budget_order = torch.topk(candidate_scores, k=budget)
        candidate_indices = candidate_indices[budget_order]
        candidate_scores = budget_scores

    k = min(topk, candidate_scores.numel())
    top_scores, top_order = torch.topk(candidate_scores, k=k)
    top_indices = candidate_indices[top_order]

    return KernelTopKResult(
        indices=top_indices,
        scores=top_scores,
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
    )


def _nibble(byte: torch.Tensor, *, high: bool) -> torch.Tensor:
    if high:
        return byte >> 4
    return byte & 0x0F


def _validate_common_inputs(packed_codes: torch.Tensor, num_subspaces: int) -> None:
    if packed_codes.ndim != 2:
        raise ValueError(f"packed_codes must be [N, ceil(M/2)], got {tuple(packed_codes.shape)}")
    if packed_codes.dtype != torch.uint8:
        raise ValueError("packed_codes must use torch.uint8 storage")
    if num_subspaces <= 0:
        raise ValueError("num_subspaces must be positive")
    expected_width = (num_subspaces + 1) // 2
    if packed_codes.shape[1] != expected_width:
        raise ValueError(
            f"packed width must be {expected_width} for {num_subspaces} subspaces, "
            f"got {packed_codes.shape[1]}"
        )


def _validate_lut_shape(lut: torch.Tensor, num_subspaces: int) -> None:
    expected = (num_subspaces, 16)
    actual = tuple(lut.shape[-2:])
    if actual != expected:
        raise ValueError(f"lut trailing shape must be {expected}, got {actual}")


def score_packed_4bit_lut_multihead_list_bias(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    list_ids: torch.Tensor,
    list_scores: torch.Tensor,
    *,
    num_subspaces: int,
    block_size: int | None = None,
    num_warps: int | None = None,
    token_scale: torch.Tensor | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Dispatch multihead packed LUT scan. ``h20`` is Hopper-only; Ampere stays on Triton."""
    from pq_hsa.kernels.device import resolve_lut_backend

    resolved = resolve_lut_backend(backend, packed_codes.device)
    kwargs = dict(
        packed_codes=packed_codes,
        lut=lut,
        list_ids=list_ids,
        list_scores=list_scores,
        num_subspaces=num_subspaces,
        token_scale=token_scale,
    )
    if resolved == "h20":
        from pq_hsa.kernels.triton_lut_scan_h20 import (
            score_packed_4bit_lut_multihead_list_bias_h20,
        )

        return score_packed_4bit_lut_multihead_list_bias_h20(
            **kwargs,
            block_size=block_size,
            num_warps=num_warps,
        )
    if resolved == "triton":
        return score_packed_4bit_lut_multihead_list_bias_triton(
            **kwargs,
            block_size=256 if block_size is None else block_size,
            num_warps=num_warps,
        )
    raise ValueError(
        f"multihead LUT scan has no torch implementation; got backend={backend!r} "
        f"(resolved={resolved!r})"
    )
