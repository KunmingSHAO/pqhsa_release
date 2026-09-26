"""Hopper / H20 LUT scan, shaped by specialized Triton operator libraries.

Score contract is identical to ``score_packed_4bit_lut_multihead_list_bias_triton``:
approx_logit = scale * (q · reconstruct(k)). Ampere keeps ``kernel_backend=triton``.

Sources (patterns, not copied kernels):

* Triton tutorial 06-fused-attention / FlashAttention-2: GQA reuses K across
  query groups; online softmax uses ``exp2``; tile loop keeps ``(m_i, l_i)``.
* Liger-Kernel softmax: ``cache_modifier=".ca"`` on reused tables, ``".cg"`` /
  ``".cs"`` on streaming codes/stores; one-tile vs multi-tile heuristics.
* FlagGems softmax: fp32 exp accumulate, CTA heuristics over N.
* Unsloth ``calculate_settings``: warp count grows with block size.
* bitsandbytes Triton 4-bit: one packed word → nibble extract, not per-subspace
  byte loads.

A 16-wide compare-sum gather was tried and rejected (~20× slower): the 16-entry
LUT already hits L1, so ``tl.load(lut + code)`` stays. One program per (head,
group) also wastes HBM: GQA groups share the same codes, so the H20 path loads
codes once per head tile and scores every group (FlashAttn K-reuse).
"""

from __future__ import annotations

import os

import torch

from pq_hsa.kernels.triton_lut_scan import _require_same_cuda_device, is_triton_available

# Receipt: pair-kernel path launch counter {"<path>/G<g>": n} (diagnostic).
_PAIR_PATH_HITS: dict = {}

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

_h20_kernel = None
_h20_fused_kernel = None
_h20_gqa_kernel = None
_h20_gqa_fused_kernel = None
_h20_pair_kernel = None
_h20_pair_fused_kernel = None
_h20_pair_fused_padg_kernel = None
_h20_pair_fused_sg_kernel = None
_h20_pair_fused_vanyg_kernel = None
_h20_pair_fused_gchunk_kernel = None
_h20_pair_build_anyg_kernel = None

# FlashAttn-2 / Triton 06: exp2(x * log2(e)) == exp(x), cheaper on Hopper.
_LOG2E = 1.4426950408889634


def _h20_heuristics(num_vectors: int) -> tuple[int, int, int]:
    """Block / warps / stages. Unsloth+Liger size table, FlagGems-style split on N."""
    if num_vectors >= 65536:
        return 2048, 8, 4
    if num_vectors >= 16384:
        return 1024, 8, 3
    return 512, 4, 2


def exact_block_merge_topk(
    scores: torch.Tensor,
    k: int,
    *,
    block: int | None = None,
    sorted: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact top-k via per-block top-k + merge. Does not use ``_approx_topk``.

    Global top-k is a subset of the union of per-block top-k, so the index set
    matches ``torch.topk(scores, k, dim=-1)`` (ties may differ in order).
    ``PQ_HSA_BLOCK_TOPK=0`` keeps the caller on aten::topk; this helper is the
    first step. Full scan-internal (no ``[H,G,N]`` materialization) still
    needs per-list exp-sum in the scan kernel (P1).
    """
    if scores.ndim != 3:
        raise ValueError("exact_block_merge_topk expects [H, G, N]")
    heads, groups, num_vectors = scores.shape
    k = min(max(int(k), 0), num_vectors)
    if k == 0:
        empty_v = scores.new_empty(heads, groups, 0)
        empty_i = torch.empty(heads, groups, 0, device=scores.device, dtype=torch.long)
        return empty_v, empty_i
    if block is None:
        block = int(os.environ.get("PQ_HSA_BLOCK_TOPK_BLOCK", "4096"))
    if num_vectors <= block or k >= block:
        return torch.topk(scores, k=k, dim=2, sorted=sorted)
    n_blocks = (num_vectors + block - 1) // block
    pad = n_blocks * block - num_vectors
    tiled = torch.nn.functional.pad(scores, (0, pad), value=float("-inf")) if pad else scores
    tiled = tiled.view(heads, groups, n_blocks, block)
    block_val, block_idx = torch.topk(tiled, k=k, dim=-1, sorted=False)
    block_idx = block_idx + torch.arange(
        n_blocks, device=scores.device, dtype=block_idx.dtype
    ).view(1, 1, n_blocks, 1) * block
    cand_val = block_val.reshape(heads, groups, n_blocks * k)
    cand_idx = block_idx.reshape(heads, groups, n_blocks * k)
    merge_val, merge_loc = torch.topk(cand_val, k=k, dim=-1, sorted=sorted)
    merge_idx = torch.gather(cand_idx, 2, merge_loc)
    if pad:
        merge_idx = merge_idx.clamp(max=num_vectors - 1)
    return merge_val, merge_idx


def _h20_pair_heuristics(num_vectors: int, *, fused: bool = False) -> tuple[int, int, int]:
    """Block / warps / stages for the pair-LUT kernels.

    Swept on H20 (H=8, G=4, M=8, L=512, N=16K..256K, CUDA-graph timing):
    the scan kernel is pure gather/stream and wants maximal thread-level
    parallelism (bs=256, nw=8 -> 1 token/thread); the fused-stats kernel pays
    for cross-warp reductions per group, so fewer warps win (bs=512, nw=2).
    """
    if fused:
        if num_vectors >= 32768:
            return 512, 2, 2
        return 256, 2, 2
    if num_vectors >= 32768:
        return 256, 8, 2
    return 256, 4, 2


def _build_pair_tables(
    lut: torch.Tensor,
    list_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse per-subspace 16-entry LUTs into per-byte 256-entry pair LUTs.

    ``lut`` is ``[H, G, M, 16]`` with even M. Packed byte ``p`` stores subspace
    ``2p`` in the low nibble and ``2p+1`` in the high nibble, so
    ``pair[h, p, byte, g] = lut[h, g, 2p, byte & 15] + lut[h, g, 2p+1, byte >> 4]``.
    The group dim goes last so a single gather per (token, pair) fetches all G
    values as one contiguous read (8 bytes for G=4 fp16).

    ``list_scores [H, G, L]`` is transposed to ``[H, L, G]`` for the same
    reason. Both tables are tiny (H*P*256*G and H*L*G elements) and stay L1/L2
    resident. A single small Triton launch builds both; the cost is repaid by
    cutting per-token gathers from M*G + G to M/2 + 1.
    """
    heads, groups, num_subspaces, _ = lut.shape
    pairs = num_subspaces // 2
    num_lists = list_scores.shape[2]
    # Table dtype follows the LUT: fp16 keeps the per-(token, pair) gather at
    # 8 bytes for G=4; fp32 inputs keep full precision (pair sums are computed
    # in fp32 either way).
    pair_lut = torch.empty(heads, pairs, 256, groups, device=lut.device, dtype=lut.dtype)
    list_scores_t = torch.empty(heads, num_lists, groups, device=lut.device, dtype=lut.dtype)
    list_chunks = triton.cdiv(num_lists, 256)
    grid = (heads, pairs + list_chunks)
    _h20_pair_build_kernel[grid](
        lut,
        list_scores,
        pair_lut,
        list_scores_t,
        num_lists,
        lut.stride(0),
        lut.stride(1),
        lut.stride(2),
        lut.stride(3),
        list_scores.stride(0),
        list_scores.stride(1),
        list_scores.stride(2),
        PAIRS=pairs,
        GROUPS=groups,
        num_warps=2,
    )
    return pair_lut, list_scores_t



def _build_pair_tables_anyg(
    lut: torch.Tensor,
    list_scores: torch.Tensor,
    pad_to: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same tables as :func:`_build_pair_tables` for ANY group count.

    Output layout is compact (``[H, P, 256, G]`` / ``[H, L, G]`` with the real
    ``G`` as the last stride), so the scalar-G scan kernel indexes ``... * G + g``
    without any padding.  Internally the build kernel iterates a power-of-two
    ``GROUPS_PAD`` lane range and masks lanes ``>= G``; for power-of-two ``G``
    the produced tables are bit-identical to ``_build_pair_tables``.
    """
    heads, groups, num_subspaces, _ = lut.shape
    pairs = num_subspaces // 2
    num_lists = list_scores.shape[2]
    gp = 1
    while gp < groups:
        gp *= 2
    # Pad_to=gp writes stride-gp tables with lanes >= G zero-filled in the
    # same single launch (replaces the two host-side F.pad launches of the
    # PQ_HSA_PAIR_PAD_GROUPS=1 path); otherwise tables are compact (stride G).
    out_g = int(pad_to) if pad_to else groups
    pair_lut = torch.empty(heads, pairs, 256, out_g, device=lut.device, dtype=lut.dtype)
    list_scores_t = torch.empty(heads, num_lists, out_g, device=lut.device, dtype=lut.dtype)
    list_chunks = triton.cdiv(num_lists, 256)
    grid = (heads, pairs + list_chunks)
    _h20_pair_build_anyg_kernel[grid](
        lut,
        list_scores,
        pair_lut,
        list_scores_t,
        num_lists,
        lut.stride(0),
        lut.stride(1),
        lut.stride(2),
        lut.stride(3),
        list_scores.stride(0),
        list_scores.stride(1),
        list_scores.stride(2),
        PAIRS=pairs,
        GROUPS=groups,
        GROUPS_PAD=gp,
        OUT_G=out_g,
        num_warps=2,
    )
    return pair_lut, list_scores_t


def score_packed_4bit_lut_multihead_list_bias_h20(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    list_ids: torch.Tensor,
    list_scores: torch.Tensor,
    *,
    num_subspaces: int,
    block_size: int | None = None,
    num_warps: int | None = None,
    token_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """H20 multihead packed LUT scan. Shapes match the Ampere Triton launcher."""
    if not is_triton_available() or _h20_kernel is None:
        raise RuntimeError("H20 Triton LUT scan requires Triton")
    if not (packed_codes.is_cuda and lut.is_cuda and list_ids.is_cuda and list_scores.is_cuda):
        raise ValueError("H20 LUT scan requires CUDA tensors")
    if token_scale is not None and not token_scale.is_cuda:
        raise ValueError("token_scale must be a CUDA tensor when provided")
    if packed_codes.ndim != 3 or packed_codes.dtype != torch.uint8:
        raise ValueError("packed_codes must be [H, N, ceil(M/2)] uint8")
    if lut.ndim != 4:
        raise ValueError(f"lut must be [H, G, M, 16], got {tuple(lut.shape)}")

    heads, num_vectors, packed_width = packed_codes.shape
    if lut.shape[0] != heads or lut.shape[2:] != (num_subspaces, 16):
        raise ValueError("lut shape must be [H, G, M, 16] matching packed heads/M")
    groups = lut.shape[1]
    if list_ids.shape != (heads, num_vectors):
        raise ValueError(f"list_ids must be [{heads}, {num_vectors}]")
    if list_scores.ndim != 3 or list_scores.shape[:2] != (heads, groups):
        raise ValueError("list_scores must be [H, G, num_lists]")
    if token_scale is not None and token_scale.shape != (heads, num_vectors):
        raise ValueError("token_scale must be [H, N]")

    if block_size is not None and block_size <= 0:
        raise ValueError("block_size must be positive")

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

    device = _require_same_cuda_device(packed_codes, lut, list_ids, list_scores, token_scale)
    output = torch.empty(heads, groups, num_vectors, device=lut.device, dtype=lut.dtype)
    _ts = token_scale if token_scale is not None else packed_codes

    # Pair-LUT fast path: one 256-entry gather covers two subspaces and all G
    # groups at once (group-major table), cutting gathers per token from
    # M*G to M/2. Requires even M and byte-contiguous codes.
    use_pair = (
        _h20_pair_kernel is not None
        and num_subspaces % 2 == 0
        and num_subspaces >= 2
        and groups & (groups - 1) == 0  # tl.arange needs a power-of-2 group dim
        and (packed_u32 or packed_codes.stride(-1) == 1)
    )
    if use_pair:
        if block_size is None or num_warps is None:
            p_block, p_warps, p_stages = _h20_pair_heuristics(num_vectors)
            block_size = block_size or p_block
            num_warps = num_warps or p_warps
            num_stages = p_stages
        else:
            num_stages = 2
        pair_lut, list_scores_t = _build_pair_tables(lut, list_scores)
        grid = (triton.cdiv(num_vectors, block_size), heads)
        with torch.cuda.device(device):
            _h20_pair_kernel[grid](
                codes_launch,
                pair_lut,
                list_ids,
                list_scores_t,
                _ts,
                output,
                num_vectors,
                packed_head_stride,
                packed_row_stride,
                list_ids.stride(0),
                list_scores_t.shape[1],
                0 if token_scale is None else token_scale.stride(0),
                output.stride(0),
                output.stride(1),
                PAIRS=num_subspaces // 2,
                GROUPS=groups,
                BLOCK_SIZE=block_size,
                HAS_TOKEN_SCALE=token_scale is not None,
                PACKED_U32=packed_u32,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        return output

    if block_size is None or num_warps is None:
        h_block, h_warps, h_stages = _h20_heuristics(num_vectors)
        if block_size is None:
            block_size = h_block
        if num_warps is None:
            num_warps = h_warps
        num_stages = h_stages
    else:
        num_stages = 3
    use_gqa = groups >= 2 and _h20_gqa_kernel is not None
    grid = (
        (triton.cdiv(num_vectors, block_size), heads)
        if use_gqa
        else (triton.cdiv(num_vectors, block_size), heads * groups)
    )
    kernel = _h20_gqa_kernel if use_gqa else _h20_kernel
    with torch.cuda.device(device):
        kernel[grid](
            codes_launch,
            lut,
            list_ids,
            list_scores,
            _ts,
            output,
            num_vectors,
            groups,
            packed_head_stride,
            packed_row_stride,
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
            PACKED_U32=packed_u32,
            GROUPS=groups,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return output


def score_packed_4bit_lut_multihead_fused_stats_h20(
    packed_codes: torch.Tensor,
    lut: torch.Tensor,
    list_ids: torch.Tensor,
    list_scores: torch.Tensor,
    *,
    num_subspaces: int,
    block_size: int | None = None,
    num_warps: int | None = None,
    token_scale: torch.Tensor | None = None,
    pair_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scan plus per-block max / exp-sum. Scores stay exact; stats avoid a second O(N) pass.

    ``pair_tables`` supplies an already-built ``(pair_lut, list_scores_t)``
    pair, in which case ``lut`` / ``list_scores`` are only used for their shapes
    (and may be ``None``).

    Returns ``(scores [H,G,N], block_max [H,G,B] fp32, block_expsum [H,G,B] fp32)``.
    ``block_expsum`` is ``sum(exp(x - block_max))`` inside each token tile.
    Fold to a row with::

        row_max = block_max.amax(-1)
        exp_sum = (block_expsum * exp(block_max - row_max[..., None])).sum(-1)
    """
    if not is_triton_available() or _h20_fused_kernel is None:
        raise RuntimeError("H20 fused LUT scan requires Triton")
    if pair_tables is None and (packed_codes.ndim != 3 or lut.ndim != 4):
        raise ValueError("fused H20 scan requires packed [H,N,W] and lut [H,G,M,16]")
    heads, num_vectors, packed_width = packed_codes.shape
    if pair_tables is not None:
        groups = pair_tables[0].shape[3]
        _pt_dtype = pair_tables[0].dtype
    else:
        groups = lut.shape[1]
        _pt_dtype = lut.dtype
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
    if pair_tables is not None:
        device = _require_same_cuda_device(
            packed_codes, pair_tables[0], list_ids, pair_tables[1], token_scale)
    else:
        device = _require_same_cuda_device(packed_codes, lut, list_ids, list_scores, token_scale)
    _ts = token_scale if token_scale is not None else packed_codes

    use_pair = (
        _h20_pair_fused_kernel is not None
        and num_subspaces % 2 == 0
        and num_subspaces >= 2
        and groups & (groups - 1) == 0  # tl.arange needs a power-of-2 group dim
        and (packed_u32 or packed_codes.stride(-1) == 1)
    )
    if pair_tables is not None and not use_pair:
        raise ValueError("pair_tables given but the pair kernel is unavailable")
    # (opt-in, default OFF): non-power-of-2 GQA groups (G=5 on Qwen2.5-14B)
    # otherwise fall off the pair-LUT fast path onto the per-subspace gqa kernel
    # (measured 3x slower per KV head).  With PQ_HSA_PAIR_PAD_GROUPS=1 the LUT /
    # list-score tables are zero-padded to the next power of two, the pair kernel
    # runs at GROUPS=Gp, and the padded rows are sliced off.  Same fp32
    # accumulation of the same fp16 LUT entries as the G=4/8 production path.
    _pad_groups = 0
    if (
        not use_pair
        and pair_tables is None
        and _h20_pair_fused_kernel is not None
        and num_subspaces % 2 == 0
        and num_subspaces >= 2
        and (packed_u32 or packed_codes.stride(-1) == 1)
        and os.environ.get("PQ_HSA_PAIR_PAD_GROUPS", "0") == "1"
        and groups > 1
    ):
        _gp = 1
        while _gp < groups:
            _gp *= 2
        _pad_groups = _gp - groups
        if _pad_groups > 0:
            # (opt-in, PQ_HSA_PAIR_PADG_BUILD=1): let the single any-G build
            # launch emit the stride-gp zero-padded tables directly instead of two
            # host-side F.pad launches on lut / list_scores.
            _padg_build = (
                _h20_pair_build_anyg_kernel is not None
                and os.environ.get("PQ_HSA_PAIR_PADG_BUILD", "0") == "1"
            )
            if not _padg_build:
                lut = torch.nn.functional.pad(lut, (0, 0, 0, 0, 0, _pad_groups))
                list_scores = torch.nn.functional.pad(list_scores, (0, 0, 0, _pad_groups))
            groups = _gp
            use_pair = True
        else:
            _padg_build = False
    else:
        _padg_build = False
    # (opt-in, default OFF): scalar-G pair kernel.  The group dimension is
    # walked with tl.static_range(G) instead of a tl.arange(G) lane vector, so
    # ANY G (5, 16, ...) takes the pair-LUT fast path at cost ~ G_real, with the
    # same per-element fp32 accumulation order as the vector pair kernel
    # (pair_0..pair_{P-1}, list score, token scale).  Power-of-two G keeps the
    # vector kernel unless PQ_HSA_PAIR_SCALAR_G_FORCE=1 (equivalence testing).
    # (opt-in, default OFF): vector-lane any-G pair kernel.  Lanes span the
    # next power of two (GROUPS_PAD) but the tables are COMPACT (stride = real G),
    # lanes >= G are masked on load/store.  Same per-(token,pair) 2^k-wide gather
    # as the production vector kernel, no padded tables, no extra launches.
    _use_vg = False
    if (
        _h20_pair_fused_vanyg_kernel is not None
        and _pad_groups == 0
        and num_subspaces % 2 == 0
        and num_subspaces >= 2
        and (packed_u32 or packed_codes.stride(-1) == 1)
        and os.environ.get("PQ_HSA_PAIR_VEC_ANYG", "0") == "1"
        and groups >= 1
        and (not use_pair or os.environ.get("PQ_HSA_PAIR_VEC_ANYG_FORCE", "0") == "1")
    ):
        _use_vg = True
        use_pair = True
    # (opt-in, PQ_HSA_PAIR_GCHUNK=<c>, c in {4,8}): vector kernel over lane
    # chunks; for G > c with G % c == 0 (G=16 -> two 8-lane chunks).  Same tables
    # (stride G) and arithmetic as the vector kernel.
    _gchunk = 0
    try:
        _gc_env = int(os.environ.get("PQ_HSA_PAIR_GCHUNK", "0"))
    except ValueError:
        _gc_env = 0
    if (
        _h20_pair_fused_gchunk_kernel is not None
        and use_pair
        and _pad_groups == 0
        and not _use_vg
        and _gc_env in (4, 8)
        and groups > _gc_env
        and groups % _gc_env == 0
    ):
        _gchunk = _gc_env
    _use_sg = False
    if (
        not _use_vg
        and _gchunk == 0
        and _h20_pair_fused_sg_kernel is not None
        and _pad_groups == 0
        and num_subspaces % 2 == 0
        and num_subspaces >= 2
        and (packed_u32 or packed_codes.stride(-1) == 1)
        and os.environ.get("PQ_HSA_PAIR_SCALAR_G", "0") == "1"
        and groups >= 1
        and (not use_pair or os.environ.get("PQ_HSA_PAIR_SCALAR_G_FORCE", "0") == "1")
    ):
        _use_sg = True
        use_pair = True
    if use_pair:
        if block_size is None or num_warps is None:
            p_block, p_warps, p_stages = _h20_pair_heuristics(num_vectors, fused=True)
            block_size = block_size or p_block
            num_warps = num_warps or p_warps
            num_stages = p_stages
        else:
            num_stages = 2
        num_blocks = triton.cdiv(num_vectors, block_size)
        _dev = packed_codes.device
        _g_out = groups - _pad_groups
        block_max = torch.empty(heads * _g_out, num_blocks, device=_dev, dtype=torch.float32)
        block_expsum = torch.empty_like(block_max)
        output = torch.empty(heads, _g_out, num_vectors, device=_dev, dtype=_pt_dtype)
        if pair_tables is not None:
            pair_lut, list_scores_t = pair_tables
            if (_use_sg or _use_vg) and int(pair_lut.shape[3]) != int(groups):
                raise ValueError("any-G pair kernels need compact pair tables (last dim == G)")
        elif _use_sg or _use_vg:
            pair_lut, list_scores_t = _build_pair_tables_anyg(lut, list_scores)
        elif _padg_build:
            pair_lut, list_scores_t = _build_pair_tables_anyg(lut, list_scores, pad_to=groups)
        else:
            pair_lut, list_scores_t = _build_pair_tables(lut, list_scores)
        grid = (num_blocks, heads)
        _launch_groups = groups
        if _gchunk:
            _pk = _h20_pair_fused_gchunk_kernel
            _pk_extra = {"GCHUNK": _gchunk}
            _path_name = f"gchunk{_gchunk}"
        elif _use_vg:
            _gp = 1
            while _gp < groups:
                _gp *= 2
            _launch_groups = _gp
            _pk = _h20_pair_fused_vanyg_kernel
            _pk_extra = {"GROUPS_REAL": groups}
            _path_name = "vec_anyg"
        elif _use_sg:
            _pk = _h20_pair_fused_sg_kernel
            _pk_extra = {}
            _path_name = "scalar_g"
        else:
            _pk = _h20_pair_fused_padg_kernel if _pad_groups > 0 else _h20_pair_fused_kernel
            _pk_extra = {"GROUPS_REAL": _g_out} if _pad_groups > 0 else {}
            _path_name = ("padg_build" if _padg_build else "padg") if _pad_groups > 0 else "pair_vec"
        # Receipt (diagnostic only, no numeric effect): which pair-kernel
        # path was launched, keyed by path/G.  Under CUDA-graph replay this counts
        # capture-time launches, same semantics as _cuda_attend_only_hits.
        _k = f"{_path_name}/G{groups - _pad_groups}"
        _PAIR_PATH_HITS[_k] = _PAIR_PATH_HITS.get(_k, 0) + 1
        with torch.cuda.device(device):
            _pk[grid](
                codes_launch,
                pair_lut,
                list_ids,
                list_scores_t,
                _ts,
                output,
                block_max,
                block_expsum,
                num_vectors,
                num_blocks,
                packed_head_stride,
                packed_row_stride,
                list_ids.stride(0),
                list_scores_t.shape[1],
                0 if token_scale is None else token_scale.stride(0),
                output.stride(0),
                output.stride(1),
                PAIRS=num_subspaces // 2,
                GROUPS=_launch_groups,
                BLOCK_SIZE=block_size,
                HAS_TOKEN_SCALE=token_scale is not None,
                PACKED_U32=packed_u32,
                num_warps=num_warps,
                num_stages=num_stages,
                **_pk_extra,
            )
        return (
            output,
            block_max.view(heads, _g_out, num_blocks),
            block_expsum.view(heads, _g_out, num_blocks),
        )

    if block_size is None or num_warps is None:
        h_block, h_warps, h_stages = _h20_heuristics(num_vectors)
        if block_size is None:
            block_size = h_block
        if num_warps is None:
            num_warps = h_warps
        num_stages = h_stages
    else:
        num_stages = 3
    num_blocks = triton.cdiv(num_vectors, block_size)
    block_max = torch.empty(heads * groups, num_blocks, device=lut.device, dtype=torch.float32)
    block_expsum = torch.empty_like(block_max)
    output = torch.empty(heads, groups, num_vectors, device=lut.device, dtype=lut.dtype)
    use_gqa = groups >= 2 and _h20_gqa_fused_kernel is not None
    grid = (num_blocks, heads) if use_gqa else (num_blocks, heads * groups)
    kernel = _h20_gqa_fused_kernel if use_gqa else _h20_fused_kernel
    with torch.cuda.device(device):
        kernel[grid](
            codes_launch,
            lut,
            list_ids,
            list_scores,
            _ts,
            output,
            block_max,
            block_expsum,
            num_vectors,
            groups,
            num_blocks,
            packed_head_stride,
            packed_row_stride,
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
            PACKED_U32=packed_u32,
            GROUPS=groups,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return (
        output,
        block_max.view(heads, groups, num_blocks),
        block_expsum.view(heads, groups, num_blocks),
    )


def reduce_block_online_softmax(
    block_max: torch.Tensor,
    block_expsum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold per-block max / exp-sum into ``(row_max, exp_sum)`` over tokens."""
    row_max = block_max.amax(dim=-1)
    exp_sum = (block_expsum * torch.exp(block_max - row_max.unsqueeze(-1))).sum(dim=-1)
    return row_max, exp_sum


if is_triton_available():

    @triton.jit
    def _score_packed_4bit_lut_multihead_h20_kernel(
        packed_codes,
        lut,
        list_ids,
        list_scores,
        token_scale,
        output,
        num_vectors: tl.constexpr,
        groups: tl.constexpr,
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
        PACKED_U32: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        hg = tl.program_id(1)
        head_id = hg // groups
        group_id = hg % groups
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        packed_base = head_id * packed_head_stride
        lut_base = head_id * lut_head_stride + group_id * lut_group_stride

        word = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)
        if PACKED_U32:
            # Little-endian uint32 over 4 packed bytes: nibble 0 is the low 4 bits.
            word = tl.load(
                packed_codes + packed_base + offsets * packed_row_stride,
                mask=mask,
                other=0,
                cache_modifier=".cg",
            ).to(tl.uint32)

        for subspace in tl.static_range(0, NUM_SUBSPACES):
            if PACKED_U32:
                code = (word >> (4 * subspace)) & 0x0F
            else:
                byte = tl.load(
                    packed_codes + packed_base + offsets * packed_row_stride + (subspace // 2),
                    mask=mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.uint32)
                if subspace % 2 == 0:
                    code = byte & 0x0F
                else:
                    code = byte >> 4
            # Liger .ca: keep the tiny LUT in L1 across the token tile.
            acc += tl.load(
                lut + lut_base + subspace * lut_subspace_stride + code * lut_code_stride,
                mask=mask,
                other=0.0,
                cache_modifier=".ca",
            )

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
        ).to(tl.float32)
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
            cache_modifier=".cs",
        )

    _h20_kernel = _score_packed_4bit_lut_multihead_h20_kernel

    @triton.jit
    def _score_packed_4bit_lut_multihead_h20_fused_kernel(
        packed_codes,
        lut,
        list_ids,
        list_scores,
        token_scale,
        output,
        block_max,
        block_expsum,
        num_vectors: tl.constexpr,
        groups: tl.constexpr,
        num_blocks: tl.constexpr,
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
        PACKED_U32: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        hg = tl.program_id(1)
        head_id = hg // groups
        group_id = hg % groups
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        packed_base = head_id * packed_head_stride
        lut_base = head_id * lut_head_stride + group_id * lut_group_stride

        word = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)
        if PACKED_U32:
            word = tl.load(
                packed_codes + packed_base + offsets * packed_row_stride,
                mask=mask,
                other=0,
            ).to(tl.uint32)

        for subspace in tl.static_range(0, NUM_SUBSPACES):
            if PACKED_U32:
                code = (word >> (4 * subspace)) & 0x0F
            else:
                byte = tl.load(
                    packed_codes + packed_base + offsets * packed_row_stride + (subspace // 2),
                    mask=mask,
                    other=0,
                ).to(tl.uint32)
                if subspace % 2 == 0:
                    code = byte & 0x0F
                else:
                    code = byte >> 4
            acc += tl.load(
                lut + lut_base + subspace * lut_subspace_stride + code * lut_code_stride,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            )

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
        ).to(tl.float32)
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
        neg_inf = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)
        scores = tl.where(mask, acc, neg_inf)
        bmax = tl.max(scores, axis=0)
        bexp = tl.sum(tl.where(mask, tl.exp(acc - bmax), 0.0), axis=0)
        tl.store(block_max + hg * num_blocks + block_id, bmax)
        tl.store(block_expsum + hg * num_blocks + block_id, bexp)

    _h20_fused_kernel = _score_packed_4bit_lut_multihead_h20_fused_kernel

    @triton.jit
    def _score_packed_4bit_lut_multihead_h20_gqa_kernel(
        packed_codes,
        lut,
        list_ids,
        list_scores,
        token_scale,
        output,
        num_vectors: tl.constexpr,
        groups: tl.constexpr,
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
        PACKED_U32: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        # FlashAttn / FlashDecoding GQA: one code tile per KV head, score all groups.
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        packed_base = head_id * packed_head_stride

        word = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)
        if PACKED_U32:
            word = tl.load(
                packed_codes + packed_base + offsets * packed_row_stride,
                mask=mask,
                other=0,
                cache_modifier=".cg",
            ).to(tl.uint32)

        list_id = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask,
            other=0,
            cache_modifier=".cg",
        ).to(tl.int64)
        scale = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
        if HAS_TOKEN_SCALE:
            scale = tl.load(
                token_scale + head_id * token_scale_head_stride + offsets,
                mask=mask,
                other=1.0,
                cache_modifier=".cg",
            ).to(tl.float32)

        for group_id in tl.static_range(0, GROUPS):
            acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            lut_base = head_id * lut_head_stride + group_id * lut_group_stride
            for subspace in tl.static_range(0, NUM_SUBSPACES):
                if PACKED_U32:
                    code = (word >> (4 * subspace)) & 0x0F
                else:
                    byte = tl.load(
                        packed_codes + packed_base + offsets * packed_row_stride + (subspace // 2),
                        mask=mask,
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.uint32)
                    if subspace % 2 == 0:
                        code = byte & 0x0F
                    else:
                        code = byte >> 4
                acc += tl.load(
                    lut + lut_base + subspace * lut_subspace_stride + code * lut_code_stride,
                    mask=mask,
                    other=0.0,
                    cache_modifier=".ca",
                )
            acc += tl.load(
                list_scores
                + head_id * list_scores_head_stride
                + group_id * list_scores_group_stride
                + list_id * list_scores_list_stride,
                mask=mask,
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.float32)
            if HAS_TOKEN_SCALE:
                acc = acc * scale
            tl.store(
                output + head_id * output_head_stride + group_id * output_group_stride + offsets,
                acc,
                mask=mask,
                cache_modifier=".cs",
            )

    @triton.jit
    def _score_packed_4bit_lut_multihead_h20_gqa_fused_kernel(
        packed_codes,
        lut,
        list_ids,
        list_scores,
        token_scale,
        output,
        block_max,
        block_expsum,
        num_vectors: tl.constexpr,
        groups: tl.constexpr,
        num_blocks: tl.constexpr,
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
        PACKED_U32: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        packed_base = head_id * packed_head_stride
        log2e = 1.4426950408889634

        word = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)
        if PACKED_U32:
            word = tl.load(
                packed_codes + packed_base + offsets * packed_row_stride,
                mask=mask,
                other=0,
                cache_modifier=".cg",
            ).to(tl.uint32)

        list_id = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask,
            other=0,
            cache_modifier=".cg",
        ).to(tl.int64)
        scale = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
        if HAS_TOKEN_SCALE:
            scale = tl.load(
                token_scale + head_id * token_scale_head_stride + offsets,
                mask=mask,
                other=1.0,
                cache_modifier=".cg",
            ).to(tl.float32)

        for group_id in tl.static_range(0, GROUPS):
            acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            lut_base = head_id * lut_head_stride + group_id * lut_group_stride
            for subspace in tl.static_range(0, NUM_SUBSPACES):
                if PACKED_U32:
                    code = (word >> (4 * subspace)) & 0x0F
                else:
                    byte = tl.load(
                        packed_codes + packed_base + offsets * packed_row_stride + (subspace // 2),
                        mask=mask,
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.uint32)
                    if subspace % 2 == 0:
                        code = byte & 0x0F
                    else:
                        code = byte >> 4
                acc += tl.load(
                    lut + lut_base + subspace * lut_subspace_stride + code * lut_code_stride,
                    mask=mask,
                    other=0.0,
                    cache_modifier=".ca",
                )
            acc += tl.load(
                list_scores
                + head_id * list_scores_head_stride
                + group_id * list_scores_group_stride
                + list_id * list_scores_list_stride,
                mask=mask,
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.float32)
            if HAS_TOKEN_SCALE:
                acc = acc * scale
            tl.store(
                output + head_id * output_head_stride + group_id * output_group_stride + offsets,
                acc,
                mask=mask,
                cache_modifier=".cs",
            )
            # FlagGems / FlashAttn online tile stats in fp32; exp2 from tutorial 06.
            neg_inf = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)
            scores = tl.where(mask, acc, neg_inf)
            bmax = tl.max(scores, axis=0)
            bexp = tl.sum(tl.where(mask, tl.math.exp2((acc - bmax) * log2e), 0.0), axis=0)
            row = head_id * GROUPS + group_id
            tl.store(block_max + row * num_blocks + block_id, bmax)
            tl.store(block_expsum + row * num_blocks + block_id, bexp)

    _h20_gqa_kernel = _score_packed_4bit_lut_multihead_h20_gqa_kernel
    _h20_gqa_fused_kernel = _score_packed_4bit_lut_multihead_h20_gqa_fused_kernel

    @triton.jit
    def _score_packed_4bit_lut_multihead_h20_pair_kernel(
        packed_codes,      # [H, N, W] uint8 (int32 view when PACKED_U32)
        pair_lut,          # [H, PAIRS, 256, GROUPS] contiguous
        list_ids,          # [H, N]
        list_scores_t,     # [H, num_lists, GROUPS] contiguous
        token_scale,       # [H, N] (dummy when HAS_TOKEN_SCALE=False)
        output,            # [H, G, N]
        num_vectors,
        packed_head_stride,
        packed_row_stride,
        list_ids_head_stride,
        num_lists,
        token_scale_head_stride,
        output_head_stride,
        output_group_stride,
        PAIRS: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr,
        PACKED_U32: tl.constexpr,
    ):
        """Pair-LUT GQA scan: 2 subspaces + all G groups per gather.

        Each packed byte indexes a fused 256-entry table whose group dim is
        innermost, so one gather returns G contiguous fp16 values (a single
        8-byte L1 transaction for G=4). Gathers per token drop from M*G
        (nibble kernel) to M/2, which is what the latency-bound scan needs.
        All gathers are unmasked: out-of-range lanes read index 0 (valid) and
        are dropped by the masked store.
        """
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        garange = tl.arange(0, GROUPS)
        mask2 = mask[:, None]

        acc = tl.zeros((BLOCK_SIZE, GROUPS), dtype=tl.float32)
        lut_base = pair_lut + head_id * (PAIRS * 256 * GROUPS)
        if PACKED_U32:
            word = tl.load(
                packed_codes + head_id * packed_head_stride + offsets * packed_row_stride,
                mask=mask,
                other=0,
                cache_modifier=".cg",
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
                    mask=mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.int32)
                acc += tl.load(
                    lut_base + pair * (256 * GROUPS) + byte[:, None] * GROUPS + garange[None, :],
                    cache_modifier=".ca",
                )

        list_id = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask,
            other=0,
            cache_modifier=".cg",
        ).to(tl.int32)
        acc += tl.load(
            list_scores_t + head_id * (num_lists * GROUPS) + list_id[:, None] * GROUPS + garange[None, :],
            cache_modifier=".ca",
        ).to(tl.float32)
        if HAS_TOKEN_SCALE:
            scale = tl.load(
                token_scale + head_id * token_scale_head_stride + offsets,
                mask=mask,
                other=1.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            acc = acc * scale[:, None]
        tl.store(
            output + head_id * output_head_stride + garange[None, :] * output_group_stride + offsets[:, None],
            acc,
            mask=mask2,
            cache_modifier=".cs",
        )

    @triton.jit
    def _score_packed_4bit_lut_multihead_h20_pair_fused_kernel(
        packed_codes,
        pair_lut,
        list_ids,
        list_scores_t,
        token_scale,
        output,
        block_max,
        block_expsum,
        num_vectors,
        num_blocks,
        packed_head_stride,
        packed_row_stride,
        list_ids_head_stride,
        num_lists,
        token_scale_head_stride,
        output_head_stride,
        output_group_stride,
        PAIRS: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr,
        PACKED_U32: tl.constexpr,
    ):
        """Pair-LUT scan plus per-(block, group) max / exp-sum in the same pass."""
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        garange = tl.arange(0, GROUPS)
        mask2 = mask[:, None]
        log2e = 1.4426950408889634

        acc = tl.zeros((BLOCK_SIZE, GROUPS), dtype=tl.float32)
        lut_base = pair_lut + head_id * (PAIRS * 256 * GROUPS)
        if PACKED_U32:
            word = tl.load(
                packed_codes + head_id * packed_head_stride + offsets * packed_row_stride,
                mask=mask,
                other=0,
                cache_modifier=".cg",
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
                    mask=mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.int32)
                acc += tl.load(
                    lut_base + pair * (256 * GROUPS) + byte[:, None] * GROUPS + garange[None, :],
                    cache_modifier=".ca",
                )

        list_id = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask,
            other=0,
            cache_modifier=".cg",
        ).to(tl.int32)
        acc += tl.load(
            list_scores_t + head_id * (num_lists * GROUPS) + list_id[:, None] * GROUPS + garange[None, :],
            cache_modifier=".ca",
        ).to(tl.float32)
        if HAS_TOKEN_SCALE:
            scale = tl.load(
                token_scale + head_id * token_scale_head_stride + offsets,
                mask=mask,
                other=1.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            acc = acc * scale[:, None]
        tl.store(
            output + head_id * output_head_stride + garange[None, :] * output_group_stride + offsets[:, None],
            acc,
            mask=mask2,
            cache_modifier=".cs",
        )
        neg_inf = tl.full((BLOCK_SIZE, GROUPS), -float("inf"), dtype=tl.float32)
        scores = tl.where(mask2, acc, neg_inf)
        bmax = tl.max(scores, axis=0)  # [GROUPS]
        bexp = tl.sum(
            tl.where(mask2, tl.math.exp2((acc - bmax[None, :]) * log2e), 0.0),
            axis=0,
        )
        rows = head_id * GROUPS + garange
        tl.store(block_max + rows * num_blocks + block_id, bmax)
        tl.store(block_expsum + rows * num_blocks + block_id, bexp)

    @triton.jit
    def _score_packed_4bit_lut_multihead_h20_pair_fused_padg_kernel(
        packed_codes,
        pair_lut,
        list_ids,
        list_scores_t,
        token_scale,
        output,
        block_max,
        block_expsum,
        num_vectors,
        num_blocks,
        packed_head_stride,
        packed_row_stride,
        list_ids_head_stride,
        num_lists,
        token_scale_head_stride,
        output_head_stride,
        output_group_stride,
        PAIRS: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_REAL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr,
        PACKED_U32: tl.constexpr,
    ):
        """(opt-in, PQ_HSA_PAIR_PAD_GROUPS=1): identical to the pair-fused kernel
        above except the pair tables are padded to GROUPS (a power of two, needed by
        tl.arange) while only GROUPS_REAL group rows are stored, so a non-power-of-2
        GQA group (G=5) takes the pair fast path without any host-side pad/slice
        copies.  Separate kernel so the default path's codegen is untouched."""
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        garange = tl.arange(0, GROUPS)
        gmask = garange < GROUPS_REAL
        mask2 = mask[:, None]
        mask_out = mask2 & gmask[None, :]
        log2e = 1.4426950408889634

        acc = tl.zeros((BLOCK_SIZE, GROUPS), dtype=tl.float32)
        lut_base = pair_lut + head_id * (PAIRS * 256 * GROUPS)
        if PACKED_U32:
            word = tl.load(
                packed_codes + head_id * packed_head_stride + offsets * packed_row_stride,
                mask=mask,
                other=0,
                cache_modifier=".cg",
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
                    mask=mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.int32)
                acc += tl.load(
                    lut_base + pair * (256 * GROUPS) + byte[:, None] * GROUPS + garange[None, :],
                    cache_modifier=".ca",
                )

        list_id = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask,
            other=0,
            cache_modifier=".cg",
        ).to(tl.int32)
        acc += tl.load(
            list_scores_t + head_id * (num_lists * GROUPS) + list_id[:, None] * GROUPS + garange[None, :],
            cache_modifier=".ca",
        ).to(tl.float32)
        if HAS_TOKEN_SCALE:
            scale = tl.load(
                token_scale + head_id * token_scale_head_stride + offsets,
                mask=mask,
                other=1.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            acc = acc * scale[:, None]
        tl.store(
            output + head_id * output_head_stride + garange[None, :] * output_group_stride + offsets[:, None],
            acc,
            mask=mask_out,
            cache_modifier=".cs",
        )
        neg_inf = tl.full((BLOCK_SIZE, GROUPS), -float("inf"), dtype=tl.float32)
        scores = tl.where(mask2, acc, neg_inf)
        bmax = tl.max(scores, axis=0)  # [GROUPS]
        bexp = tl.sum(
            tl.where(mask2, tl.math.exp2((acc - bmax[None, :]) * log2e), 0.0),
            axis=0,
        )
        rows = head_id * GROUPS_REAL + garange
        tl.store(block_max + rows * num_blocks + block_id, bmax, mask=gmask)
        tl.store(block_expsum + rows * num_blocks + block_id, bexp, mask=gmask)

    @triton.jit
    def _score_packed_4bit_lut_multihead_h20_pair_fused_sg_kernel(
        packed_codes,
        pair_lut,
        list_ids,
        list_scores_t,
        token_scale,
        output,
        block_max,
        block_expsum,
        num_vectors,
        num_blocks,
        packed_head_stride,
        packed_row_stride,
        list_ids_head_stride,
        num_lists,
        token_scale_head_stride,
        output_head_stride,
        output_group_stride,
        PAIRS: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr,
        PACKED_U32: tl.constexpr,
    ):
        """(opt-in, PQ_HSA_PAIR_SCALAR_G=1): pair-LUT scan + block stats with
        the GQA group walked as a compile-time scalar loop (any G, no power-of-two
        requirement, no padding).  Tables are compact ``[H,P,256,G]`` / ``[H,L,G]``.
        Per element the fp32 accumulation order equals the vector pair kernel:
        pair_0 .. pair_{P-1}, then list score, then token scale.  Separate kernel
        so the default path's codegen is untouched."""
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        log2e = 1.4426950408889634
        lut_base = pair_lut + head_id * (PAIRS * 256 * GROUPS)
        if PACKED_U32:
            word = tl.load(
                packed_codes + head_id * packed_head_stride + offsets * packed_row_stride,
                mask=mask,
                other=0,
                cache_modifier=".cg",
            ).to(tl.uint32)
        list_id = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask,
            other=0,
            cache_modifier=".cg",
        ).to(tl.int32)
        if HAS_TOKEN_SCALE:
            scale = tl.load(
                token_scale + head_id * token_scale_head_stride + offsets,
                mask=mask,
                other=1.0,
                cache_modifier=".cg",
            ).to(tl.float32)
        for g in tl.static_range(0, GROUPS):
            acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            for pair in tl.static_range(0, PAIRS):
                if PACKED_U32:
                    byte = ((word >> (8 * pair)) & 0xFF).to(tl.int32)
                else:
                    byte = tl.load(
                        packed_codes + head_id * packed_head_stride + offsets * packed_row_stride + pair,
                        mask=mask,
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.int32)
                acc += tl.load(
                    lut_base + pair * (256 * GROUPS) + byte * GROUPS + g,
                    cache_modifier=".ca",
                )
            acc += tl.load(
                list_scores_t + head_id * (num_lists * GROUPS) + list_id * GROUPS + g,
                cache_modifier=".ca",
            ).to(tl.float32)
            if HAS_TOKEN_SCALE:
                acc = acc * scale
            tl.store(
                output + head_id * output_head_stride + g * output_group_stride + offsets,
                acc,
                mask=mask,
                cache_modifier=".cs",
            )
            neg_inf = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)
            scores = tl.where(mask, acc, neg_inf)
            bmax = tl.max(scores, axis=0)
            bexp = tl.sum(tl.where(mask, tl.math.exp2((acc - bmax) * log2e), 0.0), axis=0)
            row = head_id * GROUPS + g
            tl.store(block_max + row * num_blocks + block_id, bmax)
            tl.store(block_expsum + row * num_blocks + block_id, bexp)

    @triton.jit
    def _score_packed_4bit_lut_multihead_h20_pair_fused_vanyg_kernel(
        packed_codes,
        pair_lut,
        list_ids,
        list_scores_t,
        token_scale,
        output,
        block_max,
        block_expsum,
        num_vectors,
        num_blocks,
        packed_head_stride,
        packed_row_stride,
        list_ids_head_stride,
        num_lists,
        token_scale_head_stride,
        output_head_stride,
        output_group_stride,
        PAIRS: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_REAL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr,
        PACKED_U32: tl.constexpr,
    ):
        """(opt-in, PQ_HSA_PAIR_VEC_ANYG=1): the production vector pair kernel
        with GROUPS (= next power of two) lanes over COMPACT tables of stride
        GROUPS_REAL; lanes >= GROUPS_REAL are masked (other=0) on every load and
        on the stores.  For power-of-two G (GROUPS == GROUPS_REAL) the masks are
        all-true and the arithmetic is the vector kernel's."""
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        garange = tl.arange(0, GROUPS)
        gmask = garange < GROUPS_REAL
        mask2 = mask[:, None]
        mask_out = mask2 & gmask[None, :]
        gmask2 = gmask[None, :] & (offsets[:, None] >= 0)
        log2e = 1.4426950408889634

        acc = tl.zeros((BLOCK_SIZE, GROUPS), dtype=tl.float32)
        lut_base = pair_lut + head_id * (PAIRS * 256 * GROUPS_REAL)
        if PACKED_U32:
            word = tl.load(
                packed_codes + head_id * packed_head_stride + offsets * packed_row_stride,
                mask=mask,
                other=0,
                cache_modifier=".cg",
            ).to(tl.uint32)
            for pair in tl.static_range(0, PAIRS):
                byte = ((word >> (8 * pair)) & 0xFF).to(tl.int32)
                acc += tl.load(
                    lut_base + pair * (256 * GROUPS_REAL) + byte[:, None] * GROUPS_REAL + garange[None, :],
                    mask=gmask2,
                    other=0.0,
                    cache_modifier=".ca",
                )
        else:
            for pair in tl.static_range(0, PAIRS):
                byte = tl.load(
                    packed_codes + head_id * packed_head_stride + offsets * packed_row_stride + pair,
                    mask=mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.int32)
                acc += tl.load(
                    lut_base + pair * (256 * GROUPS_REAL) + byte[:, None] * GROUPS_REAL + garange[None, :],
                    mask=gmask2,
                    other=0.0,
                    cache_modifier=".ca",
                )

        list_id = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask,
            other=0,
            cache_modifier=".cg",
        ).to(tl.int32)
        acc += tl.load(
            list_scores_t + head_id * (num_lists * GROUPS_REAL) + list_id[:, None] * GROUPS_REAL + garange[None, :],
            mask=gmask2,
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.float32)
        if HAS_TOKEN_SCALE:
            scale = tl.load(
                token_scale + head_id * token_scale_head_stride + offsets,
                mask=mask,
                other=1.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            acc = acc * scale[:, None]
        tl.store(
            output + head_id * output_head_stride + garange[None, :] * output_group_stride + offsets[:, None],
            acc,
            mask=mask_out,
            cache_modifier=".cs",
        )
        neg_inf = tl.full((BLOCK_SIZE, GROUPS), -float("inf"), dtype=tl.float32)
        scores = tl.where(mask2, acc, neg_inf)
        bmax = tl.max(scores, axis=0)  # [GROUPS]
        bexp = tl.sum(
            tl.where(mask2, tl.math.exp2((acc - bmax[None, :]) * log2e), 0.0),
            axis=0,
        )
        rows = head_id * GROUPS_REAL + garange
        tl.store(block_max + rows * num_blocks + block_id, bmax, mask=gmask)
        tl.store(block_expsum + rows * num_blocks + block_id, bexp, mask=gmask)

    @triton.jit
    def _h20_pair_build_anyg_kernel_impl(
        lut,            # [H, G, M, 16]
        list_scores,    # [H, G, L]
        pair_out,       # [H, PAIRS, 256, GROUPS]  (compact, GROUPS = real G)
        ls_out,         # [H, L, GROUPS]
        num_lists,
        lut_head_stride,
        lut_group_stride,
        lut_subspace_stride,
        lut_code_stride,
        ls_head_stride,
        ls_group_stride,
        ls_list_stride,
        PAIRS: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_PAD: tl.constexpr,
        OUT_G: tl.constexpr,
    ):
        """Any-G variant of ``_h20_pair_build_kernel_impl``.  Lanes iterate a
        power-of-two ``GROUPS_PAD`` range and are masked at ``GROUPS``; the output
        tables are compact with stride ``GROUPS``.  For power-of-two G this is the
        original kernel with an all-true mask (bit-identical tables)."""
        head_id = tl.program_id(0)
        job = tl.program_id(1)
        garange = tl.arange(0, GROUPS_PAD)
        gmask = garange < GROUPS
        # store mask: compact output keeps lanes < GROUPS; padded output (OUT_G > GROUPS)
        # also writes the zero lanes GROUPS..OUT_G-1 (loads there are masked to 0).
        smask = garange < OUT_G
        idx = tl.arange(0, 256)
        if job < PAIRS:
            pair = job
            lo = idx & 0x0F
            hi = idx >> 4
            base = lut + head_id * lut_head_stride + garange[None, :] * lut_group_stride
            lo_vals = tl.load(
                base + (2 * pair) * lut_subspace_stride + lo[:, None] * lut_code_stride,
                mask=gmask[None, :],
                other=0.0,
            ).to(tl.float32)
            hi_vals = tl.load(
                base + (2 * pair + 1) * lut_subspace_stride + hi[:, None] * lut_code_stride,
                mask=gmask[None, :],
                other=0.0,
            ).to(tl.float32)
            out_base = pair_out + head_id * (PAIRS * 256 * OUT_G) + pair * (256 * OUT_G)
            tl.store(
                out_base + idx[:, None] * OUT_G + garange[None, :],
                lo_vals + hi_vals,
                mask=smask[None, :],
            )
        else:
            chunk = job - PAIRS
            rows = chunk * 256 + idx
            row_mask = rows < num_lists
            vals = tl.load(
                list_scores
                + head_id * ls_head_stride
                + garange[None, :] * ls_group_stride
                + rows[:, None] * ls_list_stride,
                mask=row_mask[:, None] & gmask[None, :],
                other=0.0,
            ).to(tl.float32)
            tl.store(
                ls_out + head_id * (num_lists * OUT_G) + rows[:, None] * OUT_G + garange[None, :],
                vals,
                mask=row_mask[:, None] & smask[None, :],
            )

    @triton.jit
    def _score_packed_4bit_lut_multihead_h20_pair_fused_gchunk_kernel(
        packed_codes,
        pair_lut,
        list_ids,
        list_scores_t,
        token_scale,
        output,
        block_max,
        block_expsum,
        num_vectors,
        num_blocks,
        packed_head_stride,
        packed_row_stride,
        list_ids_head_stride,
        num_lists,
        token_scale_head_stride,
        output_head_stride,
        output_group_stride,
        PAIRS: tl.constexpr,
        GROUPS: tl.constexpr,
        GCHUNK: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_TOKEN_SCALE: tl.constexpr,
        PACKED_U32: tl.constexpr,
    ):
        """(opt-in, PQ_HSA_PAIR_GCHUNK=<c>): the vector pair kernel applied
        to GROUPS/GCHUNK lane-chunks of the group dim (tables stride GROUPS).  Keeps
        the per-(token,pair) contiguous gather and an acc of [BLOCK, GCHUNK] so large
        G (16) does not blow registers the way a single [BLOCK, 16] accumulator does.
        Per-element arithmetic order == vector kernel; block stats per chunk column."""
        block_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_vectors
        mask2 = mask[:, None]
        log2e = 1.4426950408889634
        lut_base = pair_lut + head_id * (PAIRS * 256 * GROUPS)
        if PACKED_U32:
            word = tl.load(
                packed_codes + head_id * packed_head_stride + offsets * packed_row_stride,
                mask=mask,
                other=0,
                cache_modifier=".cg",
            ).to(tl.uint32)
        list_id = tl.load(
            list_ids + head_id * list_ids_head_stride + offsets,
            mask=mask,
            other=0,
            cache_modifier=".cg",
        ).to(tl.int32)
        if HAS_TOKEN_SCALE:
            scale = tl.load(
                token_scale + head_id * token_scale_head_stride + offsets,
                mask=mask,
                other=1.0,
                cache_modifier=".cg",
            ).to(tl.float32)
        for c in tl.static_range(0, GROUPS // GCHUNK):
            garange = c * GCHUNK + tl.arange(0, GCHUNK)
            acc = tl.zeros((BLOCK_SIZE, GCHUNK), dtype=tl.float32)
            for pair in tl.static_range(0, PAIRS):
                if PACKED_U32:
                    byte = ((word >> (8 * pair)) & 0xFF).to(tl.int32)
                else:
                    byte = tl.load(
                        packed_codes + head_id * packed_head_stride + offsets * packed_row_stride + pair,
                        mask=mask,
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.int32)
                acc += tl.load(
                    lut_base + pair * (256 * GROUPS) + byte[:, None] * GROUPS + garange[None, :],
                    cache_modifier=".ca",
                )
            acc += tl.load(
                list_scores_t + head_id * (num_lists * GROUPS) + list_id[:, None] * GROUPS + garange[None, :],
                cache_modifier=".ca",
            ).to(tl.float32)
            if HAS_TOKEN_SCALE:
                acc = acc * scale[:, None]
            tl.store(
                output + head_id * output_head_stride + garange[None, :] * output_group_stride + offsets[:, None],
                acc,
                mask=mask2,
                cache_modifier=".cs",
            )
            neg_inf = tl.full((BLOCK_SIZE, GCHUNK), -float("inf"), dtype=tl.float32)
            scores = tl.where(mask2, acc, neg_inf)
            bmax = tl.max(scores, axis=0)
            bexp = tl.sum(
                tl.where(mask2, tl.math.exp2((acc - bmax[None, :]) * log2e), 0.0),
                axis=0,
            )
            rows = head_id * GROUPS + garange
            tl.store(block_max + rows * num_blocks + block_id, bmax)
            tl.store(block_expsum + rows * num_blocks + block_id, bexp)

    @triton.jit
    def _h20_pair_build_kernel_impl(
        lut,            # [H, G, M, 16]
        list_scores,    # [H, G, L]
        pair_out,       # [H, PAIRS, 256, GROUPS] fp32
        ls_out,         # [H, L, GROUPS] fp32
        num_lists,
        lut_head_stride,
        lut_group_stride,
        lut_subspace_stride,
        lut_code_stride,
        ls_head_stride,
        ls_group_stride,
        ls_list_stride,
        PAIRS: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        """Build the fused pair LUT and transposed list-score table in one launch.

        grid = (H, PAIRS + ceil(L/256)). Programs with pid1 < PAIRS emit one
        256xG pair-LUT tile; the rest transpose 256-list chunks of list_scores.
        """
        head_id = tl.program_id(0)
        job = tl.program_id(1)
        garange = tl.arange(0, GROUPS)
        idx = tl.arange(0, 256)
        if job < PAIRS:
            pair = job
            lo = idx & 0x0F
            hi = idx >> 4
            base = lut + head_id * lut_head_stride + garange[None, :] * lut_group_stride
            lo_vals = tl.load(
                base + (2 * pair) * lut_subspace_stride + lo[:, None] * lut_code_stride
            ).to(tl.float32)
            hi_vals = tl.load(
                base + (2 * pair + 1) * lut_subspace_stride + hi[:, None] * lut_code_stride
            ).to(tl.float32)
            out_base = pair_out + head_id * (PAIRS * 256 * GROUPS) + pair * (256 * GROUPS)
            tl.store(out_base + idx[:, None] * GROUPS + garange[None, :], lo_vals + hi_vals)
        else:
            chunk = job - PAIRS
            rows = chunk * 256 + idx
            row_mask = rows < num_lists
            vals = tl.load(
                list_scores
                + head_id * ls_head_stride
                + garange[None, :] * ls_group_stride
                + rows[:, None] * ls_list_stride,
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            tl.store(
                ls_out + head_id * (num_lists * GROUPS) + rows[:, None] * GROUPS + garange[None, :],
                vals,
                mask=row_mask[:, None],
            )

    _h20_pair_kernel = _score_packed_4bit_lut_multihead_h20_pair_kernel
    _h20_pair_fused_kernel = _score_packed_4bit_lut_multihead_h20_pair_fused_kernel
    _h20_pair_fused_padg_kernel = _score_packed_4bit_lut_multihead_h20_pair_fused_padg_kernel
    _h20_pair_build_kernel = _h20_pair_build_kernel_impl
    _h20_pair_fused_sg_kernel = _score_packed_4bit_lut_multihead_h20_pair_fused_sg_kernel
    _h20_pair_fused_vanyg_kernel = _score_packed_4bit_lut_multihead_h20_pair_fused_vanyg_kernel
    _h20_pair_fused_gchunk_kernel = _score_packed_4bit_lut_multihead_h20_pair_fused_gchunk_kernel
    _h20_pair_build_anyg_kernel = _h20_pair_build_anyg_kernel_impl
else:
    _h20_kernel = None
    _h20_fused_kernel = None
    _h20_gqa_kernel = None
    _h20_gqa_fused_kernel = None
    _h20_pair_kernel = None
    _h20_pair_fused_kernel = None
    _h20_pair_build_kernel = None
