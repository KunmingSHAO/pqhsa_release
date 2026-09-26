from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised when Triton is not installed.
    triton = None
    tl = None


def is_triton_available() -> bool:
    return triton is not None and tl is not None


def _require_same_cuda_device(*tensors: torch.Tensor | None) -> torch.device:
    devices = {tensor.device for tensor in tensors if tensor is not None and tensor.is_cuda}
    if len(devices) != 1:
        raise ValueError(f"Triton tensors must be on one CUDA device, got {sorted(map(str, devices))}")
    return next(iter(devices))


def score_packed_4bit_lut_triton(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    *,
    num_subspaces: int,
    block_size: int = 256,
) -> torch.Tensor:
    """Triton implementation for packed 4-bit LUT scan.

    This handles the single-query ``lut [M, 16]`` case used by decode-time
    retrieval. Batched-query LUTs continue to use the PyTorch fallback.
    """

    if not is_triton_available():
        raise RuntimeError("Triton is not available")
    if not packed_codes.is_cuda or not lut.is_cuda:
        raise ValueError("Triton LUT scan requires CUDA tensors")
    if packed_codes.ndim != 2:
        raise ValueError(f"packed_codes must be [N, ceil(M/2)], got {tuple(packed_codes.shape)}")
    if packed_codes.dtype != torch.uint8:
        raise ValueError("packed_codes must use torch.uint8 storage")
    if lut.ndim != 2:
        raise ValueError(f"Triton LUT scan expects one-query LUT [M, 16], got {tuple(lut.shape)}")
    if lut.shape != (num_subspaces, 16):
        raise ValueError(f"lut must have shape ({num_subspaces}, 16), got {tuple(lut.shape)}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    device = _require_same_cuda_device(packed_codes, lut)
    output = torch.empty(packed_codes.shape[0], device=lut.device, dtype=lut.dtype)
    grid = (triton.cdiv(packed_codes.shape[0], block_size),)
    assert _score_packed_4bit_lut_kernel is not None
    with torch.cuda.device(device):
        _score_packed_4bit_lut_kernel[grid](
            packed_codes,
            lut,
            output,
            packed_codes.shape[0],
            packed_codes.stride(0),
            NUM_SUBSPACES=num_subspaces,
            BLOCK_SIZE=block_size,
        )
    return output


def score_packed_4bit_lut_batched_triton(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    *,
    num_subspaces: int,
    block_size: int = 256,
) -> torch.Tensor:
    """Triton implementation for batched packed 4-bit LUT scan.

    ``packed_codes`` is shared across all queries and ``lut`` is ``[B, M, 16]``.
    The output is ``[B, N]`` approximate QK logits.
    """

    if not is_triton_available():
        raise RuntimeError("Triton is not available")
    if not packed_codes.is_cuda or not lut.is_cuda:
        raise ValueError("Triton batched LUT scan requires CUDA tensors")
    if packed_codes.ndim != 2:
        raise ValueError(f"packed_codes must be [N, ceil(M/2)], got {tuple(packed_codes.shape)}")
    if packed_codes.dtype != torch.uint8:
        raise ValueError("packed_codes must use torch.uint8 storage")
    if lut.ndim != 3:
        raise ValueError(f"Triton batched LUT scan expects LUT [B, M, 16], got {tuple(lut.shape)}")
    if lut.shape[1:] != (num_subspaces, 16):
        raise ValueError(f"lut trailing shape must be ({num_subspaces}, 16), got {tuple(lut.shape)}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    device = _require_same_cuda_device(packed_codes, lut)
    output = torch.empty(
        lut.shape[0],
        packed_codes.shape[0],
        device=lut.device,
        dtype=lut.dtype,
    )
    grid = (triton.cdiv(packed_codes.shape[0], block_size), lut.shape[0])
    assert _score_packed_4bit_lut_batched_kernel is not None
    with torch.cuda.device(device):
        _score_packed_4bit_lut_batched_kernel[grid](
            packed_codes,
            lut,
            output,
            packed_codes.shape[0],
            packed_codes.stride(0),
            lut.stride(0),
            lut.stride(1),
            lut.stride(2),
            output.stride(0),
            NUM_SUBSPACES=num_subspaces,
            BLOCK_SIZE=block_size,
        )
    return output


def score_packed_4bit_lut_batched_list_bias_triton(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    list_ids: torch.Tensor,
    list_scores: torch.Tensor,
    *,
    num_subspaces: int,
    block_size: int = 256,
) -> torch.Tensor:
    """Batched packed-code LUT scan fused with per-IVF-list score bias.

    ``list_scores`` is ``query @ coarse_centroids.T`` with shape
    ``[num_queries, num_lists]``. Each output element adds
    ``list_scores[query_id, list_ids[token_id]]`` to the PQ residual score.
    """

    if not is_triton_available():
        raise RuntimeError("Triton is not available")
    if not packed_codes.is_cuda or not lut.is_cuda or not list_ids.is_cuda or not list_scores.is_cuda:
        raise ValueError("Triton batched LUT scan with list bias requires CUDA tensors")
    if packed_codes.ndim != 2:
        raise ValueError(f"packed_codes must be [N, ceil(M/2)], got {tuple(packed_codes.shape)}")
    if packed_codes.dtype != torch.uint8:
        raise ValueError("packed_codes must use torch.uint8 storage")
    if lut.ndim != 3:
        raise ValueError(f"Triton batched LUT scan expects LUT [B, M, 16], got {tuple(lut.shape)}")
    if lut.shape[1:] != (num_subspaces, 16):
        raise ValueError(f"lut trailing shape must be ({num_subspaces}, 16), got {tuple(lut.shape)}")
    if list_ids.shape != (packed_codes.shape[0],):
        raise ValueError(
            f"list_ids must have shape ({packed_codes.shape[0]},), got {tuple(list_ids.shape)}"
        )
    if list_scores.ndim != 2 or list_scores.shape[0] != lut.shape[0]:
        raise ValueError(
            "list_scores must be [B, num_lists] with the same B as lut; "
            f"got {tuple(list_scores.shape)} for B={lut.shape[0]}"
        )
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    device = _require_same_cuda_device(packed_codes, lut, list_ids, list_scores)
    output = torch.empty(
        lut.shape[0],
        packed_codes.shape[0],
        device=lut.device,
        dtype=lut.dtype,
    )
    grid = (triton.cdiv(packed_codes.shape[0], block_size), lut.shape[0])
    assert _score_packed_4bit_lut_batched_list_bias_kernel is not None
    with torch.cuda.device(device):
        _score_packed_4bit_lut_batched_list_bias_kernel[grid](
            packed_codes,
            lut,
            list_ids,
            list_scores,
            output,
            packed_codes.shape[0],
            packed_codes.stride(0),
            lut.stride(0),
            lut.stride(1),
            lut.stride(2),
            list_scores.stride(0),
            list_scores.stride(1),
            output.stride(0),
            NUM_SUBSPACES=num_subspaces,
            BLOCK_SIZE=block_size,
        )
    return output


def score_packed_4bit_lut_multihead_list_bias_triton(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    list_ids: torch.Tensor,
    list_scores: torch.Tensor,
    *,
    num_subspaces: int,
    block_size: int = 256,
    num_warps: int | None = None,
    token_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score all heads' retrieval zones in a single Triton launch.

    Shapes (H heads, N retrieval tokens per head, G query groups per head):
      packed_codes [H, N, ceil(M/2)] uint8
      lut          [H, G, M, 16]
      list_ids     [H, N]
      list_scores  [H, G, num_lists]  == query @ coarse_centroids.T per head
      token_scale  [H, N] fp16, optional — per-token multiplicative scale
                   applied after the list-bias add (used for direction_normalize:
                   token_scale[h,n] = key_norm[h,n], so the final logit equals
                   scale * (q. k_i) when q is unnormalized).
    Output [H, G, N] approximate QK logits.

    This collapses the per-head Python loop of
    ``score_packed_4bit_lut_batched_list_bias_triton`` (one launch per head)
    into one launch over the (head, group) grid, which is the dominant
    launch-count win for launch-bound decode.

    ``num_warps`` is passed directly to the Triton kernel launcher. When None
    the function auto-selects: 4 warps for block_size <= 512, 8 warps for
    block_size in {1024, 2048}. Tuning at 256K showed bs=512/nw=4 saves ~24µs
    for M=8, and bs=2048/nw=8 is marginally faster for M=2.
    """

    if not is_triton_available():
        raise RuntimeError("Triton is not available")
    if not (packed_codes.is_cuda and lut.is_cuda and list_ids.is_cuda and list_scores.is_cuda):
        raise ValueError("Triton multihead LUT scan requires CUDA tensors")
    if token_scale is not None and not token_scale.is_cuda:
        raise ValueError("token_scale must be a CUDA tensor when provided")
    if packed_codes.ndim != 3:
        raise ValueError(f"packed_codes must be [H, N, ceil(M/2)], got {tuple(packed_codes.shape)}")
    if packed_codes.dtype != torch.uint8:
        raise ValueError("packed_codes must use torch.uint8 storage")
    if lut.ndim != 4:
        raise ValueError(f"lut must be [H, G, M, 16], got {tuple(lut.shape)}")
    heads, num_vectors, _ = packed_codes.shape
    if lut.shape[0] != heads:
        raise ValueError(f"lut head dim {lut.shape[0]} != packed head dim {heads}")
    if lut.shape[2:] != (num_subspaces, 16):
        raise ValueError(f"lut trailing shape must be ({num_subspaces}, 16), got {tuple(lut.shape)}")
    groups = lut.shape[1]
    if list_ids.shape != (heads, num_vectors):
        raise ValueError(f"list_ids must be [{heads}, {num_vectors}], got {tuple(list_ids.shape)}")
    if list_scores.ndim != 3 or list_scores.shape[:2] != (heads, groups):
        raise ValueError(
            f"list_scores must be [{heads}, {groups}, num_lists], got {tuple(list_scores.shape)}"
        )
    if token_scale is not None and token_scale.shape != (heads, num_vectors):
        raise ValueError(
            f"token_scale must have shape ({heads}, {num_vectors}), got {tuple(token_scale.shape)}"
        )
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    # Auto-select num_warps based on block_size if not provided.
    # Tuning at 256K (H=8, G=4, N=262144) showed:
    #   bs=256-512 → 4 warps optimal
    #   bs=1024    → 2 warps marginally best (M=2) / 4 warps for M=8 → default 4
    #   bs=2048    → 8 warps best
    if num_warps is None:
        if block_size <= 512:
            num_warps = 4
        elif block_size <= 1024:
            num_warps = 4
        else:
            num_warps = 8

    device = _require_same_cuda_device(packed_codes, lut, list_ids, list_scores, token_scale)
    output = torch.empty(heads, groups, num_vectors, device=lut.device, dtype=lut.dtype)
    grid = (triton.cdiv(num_vectors, block_size), heads * groups)
    assert _score_packed_4bit_lut_multihead_list_bias_kernel is not None
    # When token_scale is None, pass a dummy pointer (packed_codes) gated off
    # by the HAS_TOKEN_SCALE constexpr — the pointer is never dereferenced.
    _ts = token_scale if token_scale is not None else packed_codes
    with torch.cuda.device(device):
        _score_packed_4bit_lut_multihead_list_bias_kernel[grid](
            packed_codes,
            lut,
            list_ids,
            list_scores,
            _ts,
            output,
            num_vectors,
            groups,
            packed_codes.stride(0),
            packed_codes.stride(1),
            lut.stride(0),
            lut.stride(1),
            lut.stride(2),
            lut.stride(3),
            list_ids.stride(0),
            list_scores.stride(0),
            list_scores.stride(1),
            list_scores.stride(2),
            0 if token_scale is None else token_scale.stride(0),
            output.stride(0),
            output.stride(1),
            NUM_SUBSPACES=num_subspaces,
            BLOCK_SIZE=block_size,
            HAS_TOKEN_SCALE=token_scale is not None,
            num_warps=num_warps,
        )
    return output


def list_exp_sums_triton(
    logits: torch.Tensor,
    row_max: torch.Tensor,
    list_offsets: torch.Tensor,
    list_indices: torch.Tensor,
    *,
    num_lists: int,
    logit_scale: float = 1.0,
    block_size: int = 128,
) -> torch.Tensor:
    """Sum ``exp(logits * logit_scale - row_max)`` per IVF list for each query row."""

    if not is_triton_available():
        raise RuntimeError("Triton is not available")
    if (
        not logits.is_cuda
        or not row_max.is_cuda
        or not list_offsets.is_cuda
        or not list_indices.is_cuda
    ):
        raise ValueError("Triton list exp sums require CUDA tensors")
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B, N], got {tuple(logits.shape)}")
    if row_max.shape != (logits.shape[0],):
        raise ValueError(f"row_max must have shape ({logits.shape[0]},), got {tuple(row_max.shape)}")
    if list_offsets.shape != (num_lists + 1,):
        raise ValueError(
            f"list_offsets must have shape ({num_lists + 1},), got {tuple(list_offsets.shape)}"
        )
    if list_indices.ndim != 1:
        raise ValueError(f"list_indices must be rank-1, got {tuple(list_indices.shape)}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    device = _require_same_cuda_device(logits, row_max, list_offsets, list_indices)
    output = torch.empty(logits.shape[0], num_lists, device=logits.device, dtype=logits.dtype)
    grid = (num_lists, logits.shape[0])
    assert _list_exp_sums_kernel is not None
    with torch.cuda.device(device):
        _list_exp_sums_kernel[grid](
            logits,
            row_max,
            list_offsets,
            list_indices,
            output,
            logits.shape[1],
            logits.stride(0),
            output.stride(0),
            LOGIT_SCALE=float(logit_scale),
            BLOCK_SIZE=block_size,
        )
    return output


def list_stats_sorted_multihead_triton(
    logits: torch.Tensor,
    list_offsets: torch.Tensor,
    *,
    num_lists: int,
    block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-list (local_max, expsum_relative_to_local_max) for list-sorted multihead logits.

    ``logits`` is ``[H, G, N]`` fp16 with tokens grouped by IVF list per head
    (list-sorted order), so each list is a contiguous segment addressed by
    per-head CSR ``list_offsets [H, num_lists + 1]``.

    Returns two float32 tensors:
      list_max    [H, G, num_lists]  -- per-list max value (fp32)
      list_expsum [H, G, num_lists]  -- sum(exp(x - local_max)) per list (fp32)

    Empty lists return (−inf, 0.0). The caller folds these into the global
    denominator as:
      background_mass[h,g,l] = list_expsum[h,g,l] * exp(list_max[h,g,l] - row_max[h,g])
    which equals sum(exp(x - row_max)) and avoids a second O(N) pass over logits.
    """

    if not is_triton_available():
        raise RuntimeError("Triton is not available")
    if not logits.is_cuda or not list_offsets.is_cuda:
        raise ValueError("Triton list stats require CUDA tensors")
    if logits.ndim != 3:
        raise ValueError(f"logits must be [H, G, N], got {tuple(logits.shape)}")
    heads, groups, _ = logits.shape
    if list_offsets.shape != (heads, num_lists + 1):
        raise ValueError(
            f"list_offsets must have shape ({heads}, {num_lists + 1}), "
            f"got {tuple(list_offsets.shape)}"
        )
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    device = _require_same_cuda_device(logits, list_offsets)
    logits = logits.contiguous()
    list_offsets = list_offsets.contiguous()
    list_max = torch.empty(heads, groups, num_lists, device=logits.device, dtype=torch.float32)
    list_expsum = torch.empty(heads, groups, num_lists, device=logits.device, dtype=torch.float32)
    grid = (num_lists, heads * groups)
    assert _list_stats_sorted_multihead_kernel is not None
    with torch.cuda.device(device):
        _list_stats_sorted_multihead_kernel[grid](
            logits,
            list_offsets,
            list_max,
            list_expsum,
            logits.shape[-1],
            groups,
            num_lists + 1,
            num_lists,
            BLOCK_SIZE=block_size,
        )
    return list_max, list_expsum


def list_exp_sums_sorted_multihead_triton(
    logits: torch.Tensor,
    row_max: torch.Tensor,
    list_offsets: torch.Tensor,
    *,
    num_lists: int,
    block_size: int = 256,
    lists_per_prog: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Per-list ``sum(exp(logits - row_max))`` for list-sorted multihead logits.

    ``logits`` is ``[H, G, N]`` with tokens grouped by IVF list per head (the
    inverted-list order), so each list is a contiguous segment addressed by the
    per-head CSR ``list_offsets [H, num_lists + 1]``. ``row_max`` is ``[H, G]``.
    Returns float32 ``[H, G, num_lists]``.

    When ``lists_per_prog`` is None the function auto-selects between the
    simple kernel and the grid-coarsened variant based on ``num_lists``.
    For large grids (num_lists >= 512) the coarsened kernel (lpp=8, nw=2)
    yields ~33µs speedup on an Ampere-class GPU at 256K context.
    """

    if not is_triton_available():
        raise RuntimeError("Triton is not available")
    if not logits.is_cuda or not row_max.is_cuda or not list_offsets.is_cuda:
        raise ValueError("Triton list exp sums require CUDA tensors")
    if logits.ndim != 3:
        raise ValueError(f"logits must be [H, G, N], got {tuple(logits.shape)}")
    heads, groups, _ = logits.shape
    if row_max.shape != (heads, groups):
        raise ValueError(
            f"row_max must have shape ({heads}, {groups}), got {tuple(row_max.shape)}"
        )
    if list_offsets.shape != (heads, num_lists + 1):
        raise ValueError(
            f"list_offsets must have shape ({heads}, {num_lists + 1}), "
            f"got {tuple(list_offsets.shape)}"
        )
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    device = _require_same_cuda_device(logits, row_max, list_offsets)
    logits = logits.contiguous()
    row_max = row_max.contiguous()
    list_offsets = list_offsets.contiguous()
    output = torch.empty(heads, groups, num_lists, device=logits.device, dtype=torch.float32)

    # Auto-select coarse kernel for large grids: sweep showed lpp=8, nw=2, bs=256
    # gives ~49µs vs ~83µs for the simple kernel at L=2048, H*G=32.
    _use_coarse = (lists_per_prog is None and num_lists >= 512) or (
        lists_per_prog is not None and lists_per_prog > 1
    )
    _lpp = lists_per_prog if lists_per_prog is not None else 8
    _nw = num_warps if num_warps is not None else (2 if _use_coarse else 4)

    with torch.cuda.device(device):
        if _use_coarse:
            assert _list_exp_sums_sorted_multihead_coarse_kernel is not None
            n_progs = triton.cdiv(num_lists, _lpp)
            grid = (n_progs, heads * groups)
            _list_exp_sums_sorted_multihead_coarse_kernel[grid](
                logits,
                row_max,
                list_offsets,
                output,
                logits.shape[-1],
                groups,
                num_lists + 1,
                num_lists,
                num_lists,
                BLOCK_SIZE=block_size,
                LISTS_PER_PROG=_lpp,
                num_warps=_nw,
            )
        else:
            assert _list_exp_sums_sorted_multihead_kernel is not None
            grid = (num_lists, heads * groups)
            _list_exp_sums_sorted_multihead_kernel[grid](
                logits,
                row_max,
                list_offsets,
                output,
                logits.shape[-1],
                groups,
                num_lists + 1,
                num_lists,
                BLOCK_SIZE=block_size,
                num_warps=_nw,
            )
    return output


def block_topk_logits_multihead_triton(
    logits: torch.Tensor,
    *,
    block_k: int = 64,
    block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block top-BLOCK_K candidates from multihead sorted logits.

    ``logits`` is ``[H, G, N]`` fp16 (list-sorted). Each (HG, block) program
    emits its block's top-``block_k`` (score fp16, index int32) pairs.

    Returns:
      cand_scores  [H*G, num_blocks * block_k] fp16
      cand_indices [H*G, num_blocks * block_k] int32
    """
    if not is_triton_available():
        raise RuntimeError("Triton is not available")
    if not logits.is_cuda:
        raise ValueError("Triton block top-k requires CUDA tensor")
    if logits.ndim != 3:
        raise ValueError(f"logits must be [H, G, N], got {tuple(logits.shape)}")
    heads, groups, num_vectors = logits.shape
    HG = heads * groups
    if block_size <= 0 or block_k <= 0:
        raise ValueError("block_size and block_k must be positive")

    device = _require_same_cuda_device(logits)
    logits = logits.contiguous()
    kernel_block_size = triton.next_power_of_2(block_size)
    block_keep = min(block_k, kernel_block_size)
    num_blocks = triton.cdiv(num_vectors, kernel_block_size)
    cand_scores = torch.empty(HG, num_blocks * block_keep, device=device, dtype=logits.dtype)
    cand_indices = torch.empty(HG, num_blocks * block_keep, device=device, dtype=torch.int32)
    grid = (num_blocks, HG)
    assert _block_topk_logits_multihead_kernel is not None
    with torch.cuda.device(device):
        _block_topk_logits_multihead_kernel[grid](
            logits,
            cand_scores,
            cand_indices,
            num_vectors,
            groups,
            logits.stride(0),
            logits.stride(1),
            cand_scores.stride(0),
            BLOCK_SIZE=kernel_block_size,
            BLOCK_KEEP=block_keep,
        )
    return cand_scores, cand_indices


def batched_topk_merge_triton(
    cand_scores: torch.Tensor,
    cand_indices: torch.Tensor,
    *,
    topk: int,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched 2-level merge: reduce [HG, C] candidates to [HG, topk] top indices.

    Uses a hierarchical chunked merge so each Triton program stays within
    ``chunk_size`` candidates. Output indices are int32 (cast to long at call site).
    """
    if not is_triton_available():
        raise RuntimeError("Triton is not available")
    if not cand_scores.is_cuda or not cand_indices.is_cuda:
        raise ValueError("Triton batched merge requires CUDA tensors")
    if cand_scores.ndim != 2 or cand_indices.ndim != 2:
        raise ValueError("cand_scores and cand_indices must be rank-2")
    if cand_scores.shape != cand_indices.shape:
        raise ValueError("cand_scores and cand_indices must have matching shapes")
    HG, C = cand_scores.shape
    if topk <= 0:
        raise ValueError("topk must be positive")
    topk = min(topk, C)

    device = _require_same_cuda_device(cand_scores, cand_indices)

    current_scores = cand_scores
    current_indices = cand_indices

    # Iteratively merge in chunks until the candidate count fits in one kernel.
    while current_scores.shape[1] > chunk_size:
        C_cur = current_scores.shape[1]
        num_chunks = triton.cdiv(C_cur, chunk_size)
        next_C = num_chunks * topk
        next_scores = torch.empty(HG, next_C, device=device, dtype=current_scores.dtype)
        next_indices = torch.empty(HG, next_C, device=device, dtype=torch.int32)
        block_size = triton.next_power_of_2(min(chunk_size, C_cur))
        grid = (num_chunks, HG)
        assert _batched_topk_merge_kernel is not None
        with torch.cuda.device(device):
            _batched_topk_merge_kernel[grid](
                current_scores,
                current_indices,
                next_scores,
                next_indices,
                C_cur,
                block_size,
                TOPK=topk,
                BLOCK_SIZE=block_size,
            )
        current_scores = next_scores
        current_indices = next_indices

    # Final single-pass merge.
    C_cur = current_scores.shape[1]
    topk_final = min(topk, C_cur)
    block_size = triton.next_power_of_2(C_cur)
    out_scores = torch.empty(HG, topk_final, device=device, dtype=current_scores.dtype)
    out_indices = torch.empty(HG, topk_final, device=device, dtype=torch.int32)
    grid_final = (1, HG)
    assert _batched_topk_merge_kernel is not None
    with torch.cuda.device(device):
        _batched_topk_merge_kernel[grid_final](
            current_scores,
            current_indices,
            out_scores,
            out_indices,
            C_cur,
            0,  # chunk_offset unused when num_chunks==1 (grid dim 0 == 0)
            TOPK=topk_final,
            BLOCK_SIZE=block_size,
        )
    return out_scores, out_indices


def block_topk_packed_4bit_lut_triton(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    *,
    num_subspaces: int,
    candidate_budget: int,
    score_bias: torch.Tensor | None = None,
    score_scale: torch.Tensor | None = None,
    block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-block candidates from packed 4-bit LUT scan.

    The returned indices are local to ``packed_codes``. A caller should still
    merge these candidates to enforce the final global candidate budget and
    top-k. Keeping ``candidate_budget`` entries per block preserves exact global
    top-k after the merge: any item in the global top-B is also in its block's
    local top-B.
    """

    if not is_triton_available():
        raise RuntimeError("Triton is not available")
    if not packed_codes.is_cuda or not lut.is_cuda:
        raise ValueError("Triton block top-k requires CUDA tensors")
    if score_bias is not None and not score_bias.is_cuda:
        raise ValueError("score_bias must be a CUDA tensor when provided")
    if score_scale is not None and not score_scale.is_cuda:
        raise ValueError("score_scale must be a CUDA tensor when provided")
    if packed_codes.ndim != 2:
        raise ValueError(f"packed_codes must be [N, ceil(M/2)], got {tuple(packed_codes.shape)}")
    if packed_codes.dtype != torch.uint8:
        raise ValueError("packed_codes must use torch.uint8 storage")
    if lut.ndim != 2:
        raise ValueError(f"Triton block top-k expects one-query LUT [M, 16], got {tuple(lut.shape)}")
    if lut.shape != (num_subspaces, 16):
        raise ValueError(f"lut must have shape ({num_subspaces}, 16), got {tuple(lut.shape)}")
    if candidate_budget <= 0:
        raise ValueError("candidate_budget must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if score_bias is not None and score_bias.shape != (packed_codes.shape[0],):
        raise ValueError(
            f"score_bias must have shape ({packed_codes.shape[0]},), got {tuple(score_bias.shape)}"
        )
    if score_scale is not None and score_scale.shape != (packed_codes.shape[0],):
        raise ValueError(
            f"score_scale must have shape ({packed_codes.shape[0]},), got {tuple(score_scale.shape)}"
        )

    device = _require_same_cuda_device(packed_codes, lut, score_bias, score_scale)
    # Triton reductions are most efficient and robust with power-of-two blocks.
    kernel_block_size = triton.next_power_of_2(block_size)
    block_keep = min(candidate_budget, kernel_block_size)
    num_blocks = triton.cdiv(packed_codes.shape[0], kernel_block_size)
    candidate_scores = torch.empty(
        num_blocks * block_keep,
        device=lut.device,
        dtype=lut.dtype,
    )
    candidate_indices = torch.empty(
        num_blocks * block_keep,
        device=lut.device,
        dtype=torch.long,
    )
    bias = score_bias if score_bias is not None else packed_codes
    scale = score_scale if score_scale is not None else packed_codes
    assert _block_topk_packed_4bit_lut_kernel is not None
    with torch.cuda.device(device):
        _block_topk_packed_4bit_lut_kernel[(num_blocks,)](
            packed_codes,
            lut,
            bias,
            scale,
            candidate_indices,
            candidate_scores,
            packed_codes.shape[0],
            packed_codes.stride(0),
            NUM_SUBSPACES=num_subspaces,
            BLOCK_SIZE=kernel_block_size,
            BLOCK_KEEP=block_keep,
            HAS_BIAS=score_bias is not None,
            HAS_SCALE=score_scale is not None,
        )
    return candidate_indices, candidate_scores


def final_topk_merge_triton(
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    *,
    topk: int,
    candidate_budget: int | None = None,
    max_merge_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge block candidates into global budget/top-k using one Triton program."""

    if not is_triton_available():
        raise RuntimeError("Triton is not available")
    if not candidate_indices.is_cuda or not candidate_scores.is_cuda:
        raise ValueError("Triton final merge requires CUDA tensors")
    if candidate_indices.ndim != 1 or candidate_scores.ndim != 1:
        raise ValueError("candidate_indices and candidate_scores must be rank-1")
    if candidate_indices.shape != candidate_scores.shape:
        raise ValueError("candidate_indices and candidate_scores must have matching shapes")
    if candidate_scores.numel() == 0:
        raise ValueError("cannot select top-k from zero candidates")
    if topk <= 0:
        raise ValueError("topk must be positive")

    budget = candidate_scores.numel() if candidate_budget is None else candidate_budget
    budget = min(max(budget, topk), candidate_scores.numel())
    if candidate_scores.numel() > max_merge_size:
        if candidate_budget is None:
            return _hierarchical_topk_only_merge_triton(
                candidate_indices,
                candidate_scores,
                topk=topk,
                max_merge_size=max_merge_size,
            )
        if budget >= max_merge_size:
            raise ValueError(
                "Triton hierarchical final merge requires candidate_budget "
                f"smaller than max_merge_size={max_merge_size}; got budget={budget}"
            )
        return _hierarchical_final_topk_merge_triton(
            candidate_indices,
            candidate_scores,
            topk=topk,
            budget=budget,
            max_merge_size=max_merge_size,
        )

    return _single_final_topk_merge_triton(
        candidate_indices,
        candidate_scores,
        topk=topk,
        budget=budget,
    )


def _hierarchical_final_topk_merge_triton(
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    *,
    topk: int,
    budget: int,
    max_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_indices = candidate_indices
    current_scores = candidate_scores

    while current_scores.numel() > max_merge_size:
        merged_indices = []
        merged_scores = []
        for start in range(0, current_scores.numel(), max_merge_size):
            end = min(start + max_merge_size, current_scores.numel())
            chunk_budget = min(budget, end - start)
            _, _, chunk_indices, chunk_scores = _single_final_topk_merge_triton(
                current_indices[start:end],
                current_scores[start:end],
                topk=min(topk, chunk_budget),
                budget=chunk_budget,
            )
            merged_indices.append(chunk_indices)
            merged_scores.append(chunk_scores)

        next_indices = torch.cat(merged_indices, dim=0)
        next_scores = torch.cat(merged_scores, dim=0)
        if next_scores.numel() >= current_scores.numel():
            raise ValueError("Triton hierarchical final merge did not reduce candidate count")
        current_indices = next_indices
        current_scores = next_scores

    return _single_final_topk_merge_triton(
        current_indices,
        current_scores,
        topk=topk,
        budget=min(budget, current_scores.numel()),
    )


def _hierarchical_topk_only_merge_triton(
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    *,
    topk: int,
    max_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_indices = candidate_indices
    current_scores = candidate_scores
    keep = min(topk, current_scores.numel())

    while current_scores.numel() > max_merge_size:
        merged_indices = []
        merged_scores = []
        for start in range(0, current_scores.numel(), max_merge_size):
            end = min(start + max_merge_size, current_scores.numel())
            chunk_keep = min(keep, end - start)
            chunk_indices, chunk_scores, _, _ = _single_final_topk_merge_triton(
                current_indices[start:end],
                current_scores[start:end],
                topk=chunk_keep,
                budget=chunk_keep,
            )
            merged_indices.append(chunk_indices)
            merged_scores.append(chunk_scores)

        next_indices = torch.cat(merged_indices, dim=0)
        next_scores = torch.cat(merged_scores, dim=0)
        if next_scores.numel() >= current_scores.numel():
            raise ValueError("Triton hierarchical top-k merge did not reduce candidate count")
        current_indices = next_indices
        current_scores = next_scores

    top_indices, top_scores, _, _ = _single_final_topk_merge_triton(
        current_indices,
        current_scores,
        topk=keep,
        budget=keep,
    )
    return top_indices, top_scores, candidate_indices, candidate_scores


def _single_final_topk_merge_triton(
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    *,
    topk: int,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    top_count = min(topk, budget)
    block_size = triton.next_power_of_2(candidate_scores.numel())
    budget_indices = torch.empty(budget, device=candidate_indices.device, dtype=torch.long)
    budget_scores = torch.empty(budget, device=candidate_scores.device, dtype=candidate_scores.dtype)
    top_indices = torch.empty(top_count, device=candidate_indices.device, dtype=torch.long)
    top_scores = torch.empty(top_count, device=candidate_scores.device, dtype=candidate_scores.dtype)

    device = _require_same_cuda_device(candidate_indices, candidate_scores)
    assert _final_topk_merge_kernel is not None
    with torch.cuda.device(device):
        _final_topk_merge_kernel[(1,)](
            candidate_indices,
            candidate_scores,
            budget_indices,
            budget_scores,
            top_indices,
            top_scores,
            candidate_scores.numel(),
            BLOCK_SIZE=block_size,
            BUDGET=budget,
            TOPK=top_count,
        )
    return top_indices, top_scores, budget_indices, budget_scores


if is_triton_available():

    @triton.jit
    def _score_packed_4bit_lut_kernel(
        packed_codes,
        lut,
        output,
        num_vectors: tl.constexpr,
        packed_stride: tl.constexpr,
        NUM_SUBSPACES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for subspace in tl.static_range(0, NUM_SUBSPACES):
            byte = tl.load(
                packed_codes + offsets * packed_stride + (subspace // 2),
                mask=mask,
                other=0,
            ).to(tl.uint32)
            if subspace % 2 == 0:
                code = byte & 0x0F
            else:
                code = byte >> 4
            value = tl.load(lut + subspace * 16 + code, mask=mask, other=0.0)
            acc += value

        tl.store(output + offsets, acc, mask=mask)


    @triton.jit
    def _score_packed_4bit_lut_batched_kernel(
        packed_codes,
        lut,
        output,
        num_vectors: tl.constexpr,
        packed_stride: tl.constexpr,
        lut_batch_stride: tl.constexpr,
        lut_subspace_stride: tl.constexpr,
        lut_code_stride: tl.constexpr,
        output_batch_stride: tl.constexpr,
        NUM_SUBSPACES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        query_id = tl.program_id(1)
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for subspace in tl.static_range(0, NUM_SUBSPACES):
            byte = tl.load(
                packed_codes + offsets * packed_stride + (subspace // 2),
                mask=mask,
                other=0,
            ).to(tl.uint32)
            if subspace % 2 == 0:
                code = byte & 0x0F
            else:
                code = byte >> 4
            value = tl.load(
                lut
                + query_id * lut_batch_stride
                + subspace * lut_subspace_stride
                + code * lut_code_stride,
                mask=mask,
                other=0.0,
            )
            acc += value

        tl.store(output + query_id * output_batch_stride + offsets, acc, mask=mask)


    @triton.jit
    def _score_packed_4bit_lut_batched_list_bias_kernel(
        packed_codes,
        lut,
        list_ids,
        list_scores,
        output,
        num_vectors: tl.constexpr,
        packed_stride: tl.constexpr,
        lut_batch_stride: tl.constexpr,
        lut_subspace_stride: tl.constexpr,
        lut_code_stride: tl.constexpr,
        list_scores_batch_stride: tl.constexpr,
        list_scores_list_stride: tl.constexpr,
        output_batch_stride: tl.constexpr,
        NUM_SUBSPACES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        query_id = tl.program_id(1)
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for subspace in tl.static_range(0, NUM_SUBSPACES):
            byte = tl.load(
                packed_codes + offsets * packed_stride + (subspace // 2),
                mask=mask,
                other=0,
            ).to(tl.uint32)
            if subspace % 2 == 0:
                code = byte & 0x0F
            else:
                code = byte >> 4
            value = tl.load(
                lut
                + query_id * lut_batch_stride
                + subspace * lut_subspace_stride
                + code * lut_code_stride,
                mask=mask,
                other=0.0,
            )
            acc += value

        list_id = tl.load(list_ids + offsets, mask=mask, other=0).to(tl.int64)
        acc += tl.load(
            list_scores
            + query_id * list_scores_batch_stride
            + list_id * list_scores_list_stride,
            mask=mask,
            other=0.0,
        )
        tl.store(output + query_id * output_batch_stride + offsets, acc, mask=mask)


    @triton.jit
    def _score_packed_4bit_lut_multihead_list_bias_kernel(
        packed_codes,      # [H, N, W] uint8
        lut,               # [H, G, M, 16]
        list_ids,          # [H, N] int
        list_scores,       # [H, G, num_lists]
        token_scale,       # [H, N] fp16 (or dummy when HAS_TOKEN_SCALE=False)
        output,            # [H, G, N]
        num_vectors: tl.constexpr,   # N
        groups: tl.constexpr,        # G queries per head
        packed_head_stride: tl.constexpr,
        packed_row_stride: tl.constexpr,
        lut_head_stride: tl.constexpr,
        lut_group_stride: tl.constexpr,
        lut_subspace_stride: tl.constexpr,
        lut_code_stride: tl.constexpr,
        list_ids_head_stride: tl.constexpr,
        list_scores_head_stride: tl.constexpr,
        list_scores_group_stride: tl.constexpr,
        list_scores_list_stride: tl.constexpr,
        token_scale_head_stride: tl.constexpr,
        output_head_stride: tl.constexpr,
        output_group_stride: tl.constexpr,
        NUM_SUBSPACES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr,
    ):
        # program_id(1) enumerates (head, group) pairs so all H*G query rows are
        # scored in a single launch. Each head reads only its own code slab.
        hg = tl.program_id(1)
        head_id = hg // groups
        group_id = hg % groups
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        packed_base = head_id * packed_head_stride
        lut_base = head_id * lut_head_stride + group_id * lut_group_stride
        for subspace in tl.static_range(0, NUM_SUBSPACES):
            byte = tl.load(
                packed_codes + packed_base + offsets * packed_row_stride + (subspace // 2),
                mask=mask,
                other=0,
            ).to(tl.uint32)
            if subspace % 2 == 0:
                code = byte & 0x0F
            else:
                code = byte >> 4
            value = tl.load(
                lut + lut_base + subspace * lut_subspace_stride + code * lut_code_stride,
                mask=mask,
                other=0.0,
            )
            acc += value

        list_id = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask,
            other=0,
        ).to(tl.int64)
        acc += tl.load(
            list_scores
            + head_id * list_scores_head_stride
            + group_id * list_scores_group_stride
            + list_id * list_scores_list_stride,
            mask=mask,
            other=0.0,
        )
        if HAS_TOKEN_SCALE:
            acc = acc * tl.load(
                token_scale + head_id * token_scale_head_stride + offsets,
                mask=mask,
                other=1.0,
            ).to(tl.float32)
        tl.store(
            output + head_id * output_head_stride + group_id * output_group_stride + offsets,
            acc,
            mask=mask,
        )


    @triton.jit
    def _list_exp_sums_kernel(
        logits,
        row_max,
        list_offsets,
        list_indices,
        output,
        num_vectors: tl.constexpr,
        logits_batch_stride: tl.constexpr,
        output_batch_stride: tl.constexpr,
        LOGIT_SCALE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        list_id = tl.program_id(0)
        query_id = tl.program_id(1)
        start = tl.load(list_offsets + list_id)
        end = tl.load(list_offsets + list_id + 1)
        lanes = tl.arange(0, BLOCK_SIZE)
        row_shift = tl.load(row_max + query_id).to(tl.float32)
        total = tl.zeros((), dtype=tl.float32)
        cursor = start

        while cursor < end:
            positions = cursor + lanes
            mask = positions < end
            token_ids = tl.load(list_indices + positions, mask=mask, other=0).to(tl.int64)
            valid = mask & (token_ids < num_vectors)
            values = tl.load(
                logits + query_id * logits_batch_stride + token_ids,
                mask=valid,
                other=-float("inf"),
            ).to(tl.float32)
            total += tl.sum(tl.exp(values * LOGIT_SCALE - row_shift), axis=0)
            cursor += BLOCK_SIZE

        tl.store(output + query_id * output_batch_stride + list_id, total)


    @triton.jit
    def _list_exp_sums_sorted_multihead_kernel(
        logits,
        row_max,
        list_offsets,
        output,
        logits_row_stride: tl.constexpr,
        groups: tl.constexpr,
        offsets_row_stride: tl.constexpr,
        output_row_stride: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        list_id = tl.program_id(0)
        row_id = tl.program_id(1)  # h * groups + g
        head_id = row_id // groups
        offset_base = list_offsets + head_id * offsets_row_stride + list_id
        start = tl.load(offset_base)
        end = tl.load(offset_base + 1)
        row_shift = tl.load(row_max + row_id).to(tl.float32)
        lanes = tl.arange(0, BLOCK_SIZE)
        total = tl.zeros((), dtype=tl.float32)
        cursor = start

        while cursor < end:
            positions = cursor + lanes
            mask = positions < end
            values = tl.load(
                logits + row_id * logits_row_stride + positions,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            total += tl.sum(tl.exp(values - row_shift), axis=0)
            cursor += BLOCK_SIZE

        tl.store(output + row_id * output_row_stride + list_id, total)

    @triton.jit
    def _list_exp_sums_sorted_multihead_coarse_kernel(
        logits,
        row_max,
        list_offsets,
        output,
        logits_row_stride: tl.constexpr,
        groups: tl.constexpr,
        offsets_row_stride: tl.constexpr,
        output_row_stride: tl.constexpr,
        num_lists: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        LISTS_PER_PROG: tl.constexpr,
    ):
        """Grid-coarsened list_exp_sums: each program handles LISTS_PER_PROG lists.

        Reduces the grid from (num_lists, H*G) to (ceil(num_lists/LPP), H*G),
        cutting kernel-launch overhead on large-list-count configs (e.g. L=2048).
        At 256K context / L=2048 / H*G=32, lpp=8/bs=256/nw=2 saves ~33µs vs the
        simple kernel with its default settings.
        """
        base_list_id = tl.program_id(0) * LISTS_PER_PROG
        row_id = tl.program_id(1)  # h * groups + g
        head_id = row_id // groups
        lanes = tl.arange(0, BLOCK_SIZE)
        row_shift = tl.load(row_max + row_id).to(tl.float32)

        for k in tl.static_range(0, LISTS_PER_PROG):
            list_id = base_list_id + k
            if list_id < num_lists:
                offset_base = list_offsets + head_id * offsets_row_stride + list_id
                start = tl.load(offset_base)
                end = tl.load(offset_base + 1)
                total = tl.zeros((), dtype=tl.float32)
                cursor = start
                while cursor < end:
                    positions = cursor + lanes
                    mask = positions < end
                    values = tl.load(
                        logits + row_id * logits_row_stride + positions,
                        mask=mask,
                        other=-float("inf"),
                    ).to(tl.float32)
                    total += tl.sum(tl.exp(values - row_shift), axis=0)
                    cursor += BLOCK_SIZE
                tl.store(output + row_id * output_row_stride + list_id, total)

    @triton.jit
    def _list_stats_sorted_multihead_kernel(
        logits,           # [H, G, N] fp16
        list_offsets,     # [H, L+1] int64 CSR
        out_max,          # [H, G, L] fp32
        out_expsum,       # [H, G, L] fp32
        logits_row_stride: tl.constexpr,   # N  (stride from [h*G+g] row to next)
        groups: tl.constexpr,              # G
        offsets_row_stride: tl.constexpr,  # L+1
        output_row_stride: tl.constexpr,   # L
        BLOCK_SIZE: tl.constexpr,
    ):
        """Per-list (local_max, sum_exp_relative_to_local_max) in a single pass.

        grid = (num_lists, H*G).  Each program handles one (list_id, row_id).
        Uses the online Welford-style two-variable update to compute running max
        and expsum simultaneously in a single read over the list segment:
          new_max = max(old_max, block_max)
          total = total * exp(old_max - new_max) + block_expsum_rel_new_max

        This has the same memory bandwidth as a single-pass scan and is
        numerically stable for arbitrarily long lists.
        Empty lists emit (-inf, 0.0).
        """
        list_id = tl.program_id(0)
        row_id = tl.program_id(1)  # h * G + g
        head_id = row_id // groups
        offset_base = list_offsets + head_id * offsets_row_stride + list_id
        start = tl.load(offset_base)
        end = tl.load(offset_base + 1)
        lanes = tl.arange(0, BLOCK_SIZE)

        # Single-pass online (max, expsum) accumulation.
        running_max = tl.full((), -float("inf"), dtype=tl.float32)
        running_sum = tl.zeros((), dtype=tl.float32)
        cursor = start

        while cursor < end:
            positions = cursor + lanes
            mask = positions < end
            values = tl.load(
                logits + row_id * logits_row_stride + positions,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            # Per-block max and expsum relative to block_max
            block_max = tl.max(
                tl.where(mask, values, tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)),
                axis=0,
            )
            block_sum = tl.sum(
                tl.where(mask, tl.exp(values - block_max), tl.zeros((BLOCK_SIZE,), dtype=tl.float32)),
                axis=0,
            )
            # Merge (running_max, running_sum) with (block_max, block_sum)
            new_max = tl.maximum(running_max, block_max)
            running_sum = running_sum * tl.exp(running_max - new_max) + block_sum * tl.exp(block_max - new_max)
            running_max = new_max
            cursor += BLOCK_SIZE

        out_base = row_id * output_row_stride + list_id
        tl.store(out_max + out_base, running_max)
        tl.store(out_expsum + out_base, running_sum)

    @triton.jit
    def _block_topk_logits_multihead_kernel(
        logits,          # [H, G, N] fp16 -- contiguous
        cand_scores,     # [H*G, num_blocks * BLOCK_KEEP] fp16
        cand_indices,    # [H*G, num_blocks * BLOCK_KEEP] int32
        num_vectors: tl.constexpr,   # N
        groups: tl.constexpr,        # G
        logits_head_stride: tl.constexpr,   # G*N
        logits_group_stride: tl.constexpr,  # N
        cand_row_stride: tl.constexpr,      # num_blocks * BLOCK_KEEP
        BLOCK_SIZE: tl.constexpr,
        BLOCK_KEEP: tl.constexpr,
    ):
        """Per-block top-BLOCK_KEEP from multihead sorted logits [H, G, N].

        grid = (num_blocks, H*G). Each program emits its block's top-BLOCK_KEEP
        (score, local-index) pairs using the sequential max+mask loop.
        """
        block_id = tl.program_id(0)
        row_id = tl.program_id(1)  # h*G + g
        head_id = row_id // groups
        group_id = row_id % groups
        lanes = tl.arange(0, BLOCK_SIZE)
        offsets = block_id * BLOCK_SIZE + lanes
        mask = offsets < num_vectors
        scores = tl.load(
            logits + head_id * logits_head_stride + group_id * logits_group_stride + offsets,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        scores = tl.where(mask, scores, tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32))

        out_base = row_id * cand_row_stride + block_id * BLOCK_KEEP
        for keep_idx in tl.static_range(0, BLOCK_KEEP):
            best_score = tl.max(scores, axis=0)
            best_lane = tl.min(tl.where(scores == best_score, lanes, BLOCK_SIZE), axis=0)
            tl.store(cand_scores + out_base + keep_idx, best_score.to(tl.float16))
            tl.store(cand_indices + out_base + keep_idx, (block_id * BLOCK_SIZE + best_lane).to(tl.int32))
            scores = tl.where(lanes == best_lane, tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32), scores)

    @triton.jit
    def _batched_topk_merge_kernel(
        in_scores,      # [HG, C] fp16
        in_indices,     # [HG, C] int32
        out_scores,     # [HG, num_chunks * TOPK] fp16
        out_indices,    # [HG, num_chunks * TOPK] int32
        C: tl.constexpr,            # total candidates per row
        chunk_stride: tl.constexpr, # BLOCK_SIZE (size of one chunk for addressing)
        TOPK: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Batched chunk-wise top-TOPK merge.

        grid = (num_chunks, HG). Each program reads one chunk of size BLOCK_SIZE
        from in_scores/in_indices and emits the top-TOPK via sequential max+mask.
        When num_chunks == 1 (final merge), chunk_stride is 0 and program always
        reads from offset 0.
        """
        chunk_id = tl.program_id(0)
        row_id = tl.program_id(1)
        lanes = tl.arange(0, BLOCK_SIZE)
        # Compute the actual start of this chunk in the row.
        # When chunk_stride == 0 (single-program final merge), chunk start is 0.
        chunk_start = chunk_id * BLOCK_SIZE
        positions = chunk_start + lanes
        mask = positions < C
        scores = tl.load(
            in_scores + row_id * C + positions,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        indices = tl.load(
            in_indices + row_id * C + positions,
            mask=mask,
            other=0,
        )
        scores = tl.where(mask, scores, tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32))

        out_base = row_id * (tl.num_programs(0) * TOPK) + chunk_id * TOPK
        for rank in tl.static_range(0, TOPK):
            best_score = tl.max(scores, axis=0)
            best_lane = tl.min(tl.where(scores == best_score, lanes, BLOCK_SIZE), axis=0)
            best_index = tl.load(in_indices + row_id * C + chunk_start + best_lane)
            tl.store(out_scores + out_base + rank, best_score.to(tl.float16))
            tl.store(out_indices + out_base + rank, best_index)
            scores = tl.where(lanes == best_lane, tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32), scores)

    @triton.jit
    def _block_topk_packed_4bit_lut_kernel(
        packed_codes,
        lut,
        score_bias,
        score_scale,
        candidate_indices,
        candidate_scores,
        num_vectors: tl.constexpr,
        packed_stride: tl.constexpr,
        NUM_SUBSPACES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_KEEP: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_SCALE: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        lanes = tl.arange(0, BLOCK_SIZE)
        offsets = block_id * BLOCK_SIZE + lanes
        mask = offsets < num_vectors
        scores = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for subspace in tl.static_range(0, NUM_SUBSPACES):
            byte = tl.load(
                packed_codes + offsets * packed_stride + (subspace // 2),
                mask=mask,
                other=0,
            ).to(tl.uint32)
            if subspace % 2 == 0:
                code = byte & 0x0F
            else:
                code = byte >> 4
            value = tl.load(lut + subspace * 16 + code, mask=mask, other=0.0)
            acc += value

        if HAS_BIAS:
            acc += tl.load(score_bias + offsets, mask=mask, other=0.0)
        if HAS_SCALE:
            acc *= tl.load(score_scale + offsets, mask=mask, other=0.0)
        scores = tl.where(mask, acc, scores)

        for keep_idx in tl.static_range(0, BLOCK_KEEP):
            best_score = tl.max(scores, axis=0)
            best_lane = tl.min(tl.where(scores == best_score, lanes, BLOCK_SIZE), axis=0)
            out_pos = block_id * BLOCK_KEEP + keep_idx
            tl.store(candidate_scores + out_pos, best_score)
            tl.store(candidate_indices + out_pos, block_id * BLOCK_SIZE + best_lane)
            scores = tl.where(lanes == best_lane, -float("inf"), scores)


    @triton.jit
    def _final_topk_merge_kernel(
        candidate_indices,
        candidate_scores,
        budget_indices,
        budget_scores,
        top_indices,
        top_scores,
        num_candidates: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BUDGET: tl.constexpr,
        TOPK: tl.constexpr,
    ):
        lanes = tl.arange(0, BLOCK_SIZE)
        mask = lanes < num_candidates
        scores = tl.load(candidate_scores + lanes, mask=mask, other=-float("inf"))

        for rank in tl.static_range(0, BUDGET):
            best_score = tl.max(scores, axis=0)
            best_lane = tl.min(tl.where(scores == best_score, lanes, BLOCK_SIZE), axis=0)
            best_index = tl.load(candidate_indices + best_lane)
            tl.store(budget_scores + rank, best_score)
            tl.store(budget_indices + rank, best_index)
            if rank < TOPK:
                tl.store(top_scores + rank, best_score)
                tl.store(top_indices + rank, best_index)
            scores = tl.where(lanes == best_lane, -float("inf"), scores)

else:
    _score_packed_4bit_lut_kernel = None
    _score_packed_4bit_lut_batched_kernel = None
    _score_packed_4bit_lut_batched_list_bias_kernel = None
    _score_packed_4bit_lut_multihead_list_bias_kernel = None
    _list_exp_sums_kernel = None
    _list_exp_sums_sorted_multihead_kernel = None
    _list_exp_sums_sorted_multihead_coarse_kernel = None
    _list_stats_sorted_multihead_kernel = None
    _block_topk_logits_multihead_kernel = None
    _batched_topk_merge_kernel = None
    _block_topk_packed_4bit_lut_kernel = None
    _final_topk_merge_kernel = None
