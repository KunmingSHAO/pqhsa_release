"""Gather vLLM V1 FLASH_ATTN paged KV into contiguous [1, H_kv, S, D] tensors.

FLASH_ATTN V1 cache layout is ``[2, num_blocks, block_size, kv_heads, dim]`` with a
dense ``block_table [num_reqs, max_blocks]`` and ``seq_lens [num_reqs]``, which is a
simpler (and cheaper to gather) addressing scheme than the FlashInfer indptr form.
"""

from __future__ import annotations

import torch


def flashattn_seq_len(attn_metadata, seq_idx: int = 0) -> int:
    """Sequence length for one request, without touching device memory.

    ``seq_lens`` lives on the GPU, so ``seq_lens[i].item()`` forces a device sync. On the
    decode path that sync runs once per layer per token (32x per token for Llama-8B) and
    serialises host against device, which measured as ~100 ms/tok of pipeline stall at
    128K. ``max_seq_len`` is derived from the runner's host-side numpy ``seq_lens_np``, so
    for a single-request decode batch it is the same number for free.
    """
    seq_lens = attn_metadata.seq_lens
    if seq_idx == 0 and int(seq_lens.shape[0]) == 1:
        return int(attn_metadata.max_seq_len)
    return int(seq_lens[seq_idx].item())


# ---------------------------------------------------------------------------
# VLLM >= 0.10 FLASH_ATTN layout. The per-layer tensor handed to
# ``FlashAttentionImpl.forward`` is no longer ``[2, num_blocks, block_size,
# kv_heads, dim]``; vLLM 0.29 stores K and V concatenated in the last dim of a
# 4-D ``[num_blocks, kv_heads, block_size, 2*dim]`` page and the backend itself
# does ``kv_cache.transpose(1, 2).split(head_size, dim=-1)``. ``split_kv_cache``
# returns views ``(key_cache, value_cache)`` shaped ``[num_blocks, block_size,
# kv_heads, dim]`` for BOTH layouts, so the gathers below stay byte-identical
# on 0.8.5.post1 (there it is exactly ``kv_cache[0]`` / ``kv_cache[1]``).
# ---------------------------------------------------------------------------


def is_new_kv_layout(kv_cache: torch.Tensor) -> bool:
    return kv_cache.dim() == 4


def split_kv_cache(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_cache.dim() == 5:
        return kv_cache[0], kv_cache[1]
    if kv_cache.dim() == 4:
        head_size = int(kv_cache.shape[-1]) // 2
        k, v = kv_cache.transpose(1, 2).split(head_size, dim=-1)
        return k, v
    raise ValueError(f"unsupported FLASH_ATTN kv_cache shape {tuple(kv_cache.shape)}")


def kv_cache_dims(kv_cache: torch.Tensor) -> tuple[int, int, int]:
    """(block_size, kv_heads, head_dim) for either layout."""
    k, _ = split_kv_cache(kv_cache)
    return int(k.shape[1]), int(k.shape[2]), int(k.shape[3])


def gather_flashattn_kv(
    kv_cache: torch.Tensor,
    attn_metadata,
    seq_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return contiguous K/V in PQ-HSA layout ``[1, kv_heads, seq, dim]``."""
    seq_len = flashattn_seq_len(attn_metadata, seq_idx)
    k_cache, v_cache = split_kv_cache(kv_cache)
    block_size = int(k_cache.shape[1])
    kv_heads = int(k_cache.shape[2])
    head_dim = int(k_cache.shape[3])
    if seq_len <= 0:
        empty = kv_cache.new_empty(1, kv_heads, 0, head_dim)
        return empty, empty, 0

    n_blocks = (seq_len + block_size - 1) // block_size
    blocks = attn_metadata.block_table[seq_idx, :n_blocks].long()

    keys = k_cache[blocks].reshape(-1, kv_heads, head_dim)[:seq_len]
    values = v_cache[blocks].reshape(-1, kv_heads, head_dim)[:seq_len]
    keys = keys.permute(1, 0, 2).unsqueeze(0).contiguous()
    values = values.permute(1, 0, 2).unsqueeze(0).contiguous()
    return keys, values, seq_len


# ---------------------------------------------------------------------------
# (PQ_HSA_PAGED_KV=1, opt-in): partial-range / scattered-index gathers
# straight out of vLLM's paged KV cache, so the sidecar does not need to keep
# its own resident raw K/V copy for these two read patterns. Both helpers are
# pure reads of vLLM's own kv_cache tensor (allocated once at engine init,
# see gpu_model_runner.initialize_kv_cache -> stable data_ptr for the whole
# engine lifetime) -- nothing here mutates kv_cache.
# ---------------------------------------------------------------------------


def gather_flashattn_kv_range(
    kv_cache: torch.Tensor,
    attn_metadata,
    seq_idx: int,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous token range [start, end) -> keys/values shaped [kv_heads, n, dim].

    Same physical bytes as ``gather_flashattn_kv`` restricted to a sub-range,
    used where the caller wants a small/bounded contiguous slice (sink
    window, local window, or the new-token span for a deferred index flush)
    instead of materializing the whole sequence.
    """
    n = int(end) - int(start)
    k_cache, v_cache = split_kv_cache(kv_cache)
    block_size = int(k_cache.shape[1])
    kv_heads = int(k_cache.shape[2])
    head_dim = int(k_cache.shape[3])
    if n <= 0:
        empty = kv_cache.new_empty(kv_heads, 0, head_dim)
        return empty, empty

    first_block = start // block_size
    last_block = (end - 1) // block_size
    blocks = attn_metadata.block_table[seq_idx, first_block : last_block + 1].long()
    keys = k_cache[blocks].reshape(-1, kv_heads, head_dim)
    values = v_cache[blocks].reshape(-1, kv_heads, head_dim)
    row0 = start - first_block * block_size
    keys = keys[row0 : row0 + n].permute(1, 0, 2).contiguous()
    values = values[row0 : row0 + n].permute(1, 0, 2).contiguous()
    return keys, values


def gather_flashattn_kv_at_indices(
    kv_cache: torch.Tensor,
    attn_metadata,
    seq_idx: int,
    token_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scattered per-head token gather -> keys/values shaped like
    ``token_indices`` + [dim].

    ``token_indices`` must be a LongTensor whose leading dimension is the KV
    head index (dim 0 == kv_heads), e.g. ``[H, G, topk]`` -- every entry is a
    token position in [0, seq_len) for that request; the physical block for
    a given token is the same across heads (one block layout per request),
    only the head slot inside the block row differs.
    """
    k_cache, v_cache = split_kv_cache(kv_cache)
    block_size = int(k_cache.shape[1])
    H = int(token_indices.shape[0])
    block_table = attn_metadata.block_table[seq_idx]  # [max_blocks]
    block_nums = torch.div(token_indices, block_size, rounding_mode="floor")
    rows = token_indices - block_nums * block_size
    block_ids = block_table[block_nums.reshape(-1)].reshape(block_nums.shape).long()
    head_ids = torch.arange(H, device=token_indices.device).view(
        H, *([1] * (token_indices.ndim - 1))
    ).expand_as(token_indices)
    keys = k_cache[block_ids, rows.long(), head_ids]
    values = v_cache[block_ids, rows.long(), head_ids]
    return keys, values
