"""Gather FlashInfer V1 paged KV into contiguous [1, H_kv, S, D] tensors."""

from __future__ import annotations

import torch


def flashinfer_seq_len(attn_metadata, seq_idx: int = 0) -> int:
    start = int(attn_metadata.paged_kv_indptr[seq_idx].item())
    end = int(attn_metadata.paged_kv_indptr[seq_idx + 1].item())
    n_pages = end - start
    if n_pages <= 0:
        return 0
    last = int(attn_metadata.paged_kv_last_page_len[seq_idx].item())
    return (n_pages - 1) * int(attn_metadata.page_size) + last


def gather_flashinfer_kv(
    kv_cache: torch.Tensor,
    attn_metadata,
    seq_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return contiguous K/V in PQ-HSA layout ``[1, kv_heads, seq, dim]``.

    FlashInfer V1 cache layout: ``[num_blocks, 2, block_size, kv_heads, dim]``.
    """
    start = int(attn_metadata.paged_kv_indptr[seq_idx].item())
    end = int(attn_metadata.paged_kv_indptr[seq_idx + 1].item())
    pages = attn_metadata.paged_kv_indices[start:end]
    n_pages = int(pages.numel())
    if n_pages == 0:
        empty = kv_cache.new_empty(1, kv_cache.shape[-2], 0, kv_cache.shape[-1])
        return empty, empty, 0
    last = int(attn_metadata.paged_kv_last_page_len[seq_idx].item())
    page_size = int(attn_metadata.page_size)
    seq_len = (n_pages - 1) * page_size + last
    k_pages = kv_cache[pages.long(), 0]
    v_pages = kv_cache[pages.long(), 1]
    kv_heads = k_pages.shape[-2]
    head_dim = k_pages.shape[-1]
    keys = k_pages.reshape(-1, kv_heads, head_dim)[:seq_len]
    values = v_pages.reshape(-1, kv_heads, head_dim)[:seq_len]
    keys = keys.permute(1, 0, 2).unsqueeze(0).contiguous()
    values = values.permute(1, 0, 2).unsqueeze(0).contiguous()
    return keys, values, seq_len
