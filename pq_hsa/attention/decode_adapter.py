from __future__ import annotations

from dataclasses import dataclass

import torch

from pq_hsa.attention.multihead import IVFPQMultiHeadAttention, MultiHeadAttentionOutput
from pq_hsa.attention.sparse_attention import SparseAttentionConfig
from pq_hsa.index.ivfpq import IVFPQConfig


@dataclass(slots=True)
class DecodeAttentionOutput:
    context: torch.Tensor
    per_query: tuple[MultiHeadAttentionOutput, ...]


class IVFPQDecodeAttentionAdapter:
    """Model-facing decode adapter for IVF-PQ sparse attention.

    The core attention implementation owns one cache per batch/head stream and
    works on one query per stream. This adapter accepts transformer-style state
    tensors and runs the core for each decode query position:

        keys/values: ``[batch, heads, seq, dim]``
        queries:     ``[batch, heads, query_len, dim]`` or ``[batch, heads, dim]``

    ``decode_step`` can append the current token K/V before attention, matching
    self-attention decode where the current query attends to the current key.
    """

    def __init__(self, index_config: IVFPQConfig, attention_config: SparseAttentionConfig):
        self.index_config = index_config
        self.attention_config = attention_config
        self.attention = IVFPQMultiHeadAttention(index_config, attention_config)

    @property
    def streams(self):
        return self.attention.streams

    @property
    def leading_shape(self) -> tuple[int, ...]:
        return self.attention.leading_shape

    def build_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> "IVFPQDecodeAttentionAdapter":
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must be [batch, heads, seq, dim]")
        self.attention.build_cache(keys, values)
        return self

    def state_dict(self) -> dict[str, object]:
        return {"attention": self.attention.state_dict()}

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, object],
        *,
        index_config: IVFPQConfig | None = None,
        attention_config: SparseAttentionConfig | None = None,
    ) -> "IVFPQDecodeAttentionAdapter":
        attention_state = _attention_state(state)
        attention = IVFPQMultiHeadAttention.from_state_dict(
            attention_state,
            index_config=index_config,
            attention_config=attention_config,
        )
        adapter = cls(attention.index_config, attention.attention_config)
        adapter.attention = attention
        return adapter

    def load_state_dict(self, state: dict[str, object]) -> "IVFPQDecodeAttentionAdapter":
        self.attention.load_state_dict(_attention_state(state))
        return self

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.attention.append(
            _squeeze_decode_token(keys, "keys"),
            _squeeze_decode_token(values, "values"),
        )

    def apply_pending_index_update(self) -> int:
        return self.attention.apply_pending_index_update()

    def forward(self, queries: torch.Tensor) -> DecodeAttentionOutput:
        query_states, had_query_len = _normalize_queries(queries)
        result = self.attention.forward_many(query_states)
        context = result.output
        if not had_query_len:
            context = context.squeeze(-2)
        return DecodeAttentionOutput(context=context, per_query=result.per_query)

    def decode_step(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        *,
        append_kv: bool = True,
    ) -> DecodeAttentionOutput:
        if append_kv:
            if keys is None or values is None:
                raise ValueError("keys and values are required when append_kv=True")
            self.append(keys, values)
        return self.forward(queries)


def _normalize_queries(queries: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if queries.ndim == 3:
        return queries.unsqueeze(-2), False
    if queries.ndim == 4:
        return queries, True
    raise ValueError("queries must be [batch, heads, dim] or [batch, heads, query_len, dim]")


def _squeeze_decode_token(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 4 and tensor.shape[-2] == 1:
        return tensor.squeeze(-2)
    raise ValueError(f"{name} must be [batch, heads, dim] or [batch, heads, 1, dim]")


def _attention_state(state: dict[str, object]) -> dict[str, object]:
    attention = state.get("attention")
    if not isinstance(attention, dict):
        raise ValueError("adapter state must contain an attention dict")
    return attention
