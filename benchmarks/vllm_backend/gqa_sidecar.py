"""Thin GQA sidecar: one IVF-PQ stream per KV head (read-only pq_hsa calls)."""

from __future__ import annotations

import torch

from pq_hsa.attention.sparse_attention import IVFPQSparseAttention, SparseAttentionConfig
from pq_hsa.index.ivfpq import IVFPQConfig


class GQASidecar:
    """Share one IVF-PQ cache across each GQA group (paper ``share_gqa``)."""

    def __init__(
        self,
        index_config: IVFPQConfig,
        attention_config: SparseAttentionConfig,
        *,
        num_query_heads: int,
        num_kv_heads: int,
    ) -> None:
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("GQA requires num_query_heads % num_kv_heads == 0")
        self.index_config = index_config
        self.attention_config = attention_config
        self.num_query_heads = int(num_query_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.num_groups = self.num_query_heads // self.num_kv_heads
        self.streams: list[IVFPQSparseAttention] = []
        self.length = 0

    def build_cache(self, keys: torch.Tensor, values: torch.Tensor) -> "GQASidecar":
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys/values must be [1, kv_heads, seq, dim]")
        if keys.shape[0] != 1 or keys.shape[1] != self.num_kv_heads:
            raise ValueError(
                f"expected keys [1, {self.num_kv_heads}, S, D], got {tuple(keys.shape)}"
            )
        streams: list[IVFPQSparseAttention] = []
        for h in range(self.num_kv_heads):
            attn = IVFPQSparseAttention(self.index_config, self.attention_config)
            attn.build_cache(keys[0, h], values[0, h])
            streams.append(attn)
        self.streams = streams
        self.length = int(keys.shape[2])
        return self

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.ndim != 3 or values.ndim != 3:
            raise ValueError("append keys/values must be [1, kv_heads, dim]")
        for h, stream in enumerate(self.streams):
            stream.append(keys[0, h], values[0, h])
        self.length += 1

    def append_many(self, keys: torch.Tensor, values: torch.Tensor, *, flush_every: int = 0) -> int:
        """Passthrough for API symmetry with GQASidecarFast (per-token loop;
        this loop sidecar has no prefix checkpoint, so extend_prefix is N/A)."""
        if keys.ndim == 4:
            keys = keys[0]
            values = values[0]
        n = int(keys.shape[1])
        for i in range(n):
            self.append(keys[:, i].unsqueeze(0), values[:, i].unsqueeze(0))
        return 0

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """queries: ``[1, q_heads, dim]`` → context ``[1, q_heads, dim]``."""
        if queries.ndim != 3 or queries.shape[0] != 1:
            raise ValueError(f"queries must be [1, q_heads, dim], got {tuple(queries.shape)}")
        parts: list[torch.Tensor] = []
        g = self.num_groups
        for h, stream in enumerate(self.streams):
            q_h = queries[0, h * g : (h + 1) * g]
            outs = stream.forward_many(q_h)
            parts.append(torch.stack([item.output for item in outs], dim=0))
        return torch.cat(parts, dim=0).unsqueeze(0)
