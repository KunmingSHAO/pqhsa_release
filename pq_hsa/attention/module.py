from __future__ import annotations

from dataclasses import dataclass

import torch

from pq_hsa.attention.decode_adapter import DecodeAttentionOutput, IVFPQDecodeAttentionAdapter
from pq_hsa.attention.sparse_attention import SparseAttentionConfig
from pq_hsa.index.ivfpq import IVFPQConfig


@dataclass(slots=True)
class ModuleAttentionOutput:
    context: torch.Tensor
    adapter_output: DecodeAttentionOutput


class IVFPQDecodeAttentionModule(torch.nn.Module):
    """Thin nn.Module wrapper for projected Q/K/V decode states.

    This module is intentionally projection-free: model code should pass
    already-projected query/key/value heads. It owns an
    ``IVFPQDecodeAttentionAdapter`` cache and provides a module-shaped boundary
    for replacing the attention kernel inside a model layer.

    Supported layouts:
    - ``"bhsd"``: ``[batch, heads, seq, dim]``
    - ``"bshd"``: ``[batch, seq, heads, dim]``
    """

    def __init__(
        self,
        index_config: IVFPQConfig,
        attention_config: SparseAttentionConfig,
        *,
        tensor_layout: str = "bhsd",
    ):
        super().__init__()
        if tensor_layout not in {"bhsd", "bshd"}:
            raise ValueError("tensor_layout must be 'bhsd' or 'bshd'")
        self.index_config = index_config
        self.attention_config = attention_config
        self.tensor_layout = tensor_layout
        self.adapter = IVFPQDecodeAttentionAdapter(index_config, attention_config)

    @property
    def streams(self):
        return self.adapter.streams

    def build_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> "IVFPQDecodeAttentionModule":
        keys = _sequence_to_bhsd(key_states, self.tensor_layout, "key_states")
        values = _sequence_to_bhsd(value_states, self.tensor_layout, "value_states")
        self.adapter.build_cache(keys, values)
        return self

    def cache_state_dict(self) -> dict[str, object]:
        return {
            "tensor_layout": self.tensor_layout,
            "adapter": self.adapter.state_dict(),
        }

    def load_cache_state_dict(self, state: dict[str, object]) -> "IVFPQDecodeAttentionModule":
        layout = state.get("tensor_layout")
        if layout != self.tensor_layout:
            raise ValueError("cache state tensor_layout does not match this module")
        adapter_state = state.get("adapter")
        if not isinstance(adapter_state, dict):
            raise ValueError("cache state must contain an adapter dict")
        self.adapter.load_state_dict(adapter_state)
        return self

    def apply_pending_index_update(self) -> int:
        return self.adapter.apply_pending_index_update()

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor | None = None,
        value_states: torch.Tensor | None = None,
        *,
        append_kv: bool = False,
    ) -> ModuleAttentionOutput:
        queries, had_query_len = _query_to_bhsd(query_states, self.tensor_layout)
        if append_kv:
            if key_states is None or value_states is None:
                raise ValueError("key_states and value_states are required when append_kv=True")
            keys = _decode_token_to_bhsd(key_states, self.tensor_layout, "key_states")
            values = _decode_token_to_bhsd(value_states, self.tensor_layout, "value_states")
            adapter_output = self.adapter.decode_step(queries, keys, values, append_kv=True)
        else:
            adapter_output = self.adapter.forward(queries)

        context = _context_from_bhsd(
            adapter_output.context,
            self.tensor_layout,
            had_query_len=had_query_len,
        )
        return ModuleAttentionOutput(context=context, adapter_output=adapter_output)

    def decode_step(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        append_kv: bool = True,
    ) -> ModuleAttentionOutput:
        queries, had_query_len = _query_to_bhsd(query_states, self.tensor_layout)
        keys = _decode_token_to_bhsd(key_states, self.tensor_layout, "key_states")
        values = _decode_token_to_bhsd(value_states, self.tensor_layout, "value_states")
        adapter_output = self.adapter.decode_step(
            queries,
            keys,
            values,
            append_kv=append_kv,
        )
        context = _context_from_bhsd(
            adapter_output.context,
            self.tensor_layout,
            had_query_len=had_query_len,
        )
        return ModuleAttentionOutput(context=context, adapter_output=adapter_output)


def _sequence_to_bhsd(tensor: torch.Tensor, layout: str, name: str) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be rank-4")
    if layout == "bhsd":
        return tensor
    return tensor.transpose(1, 2).contiguous()


def _query_to_bhsd(tensor: torch.Tensor, layout: str) -> tuple[torch.Tensor, bool]:
    if tensor.ndim == 3:
        return tensor, False
    if tensor.ndim != 4:
        raise ValueError("query_states must be rank-3 or rank-4")
    if layout == "bhsd":
        return tensor, True
    return tensor.transpose(1, 2).contiguous(), True


def _decode_token_to_bhsd(tensor: torch.Tensor, layout: str, name: str) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim != 4 or tensor.shape[-2 if layout == "bhsd" else 1] != 1:
        raise ValueError(f"{name} must be [B,H,D], [B,H,1,D], or [B,1,H,D]")
    if layout == "bhsd":
        return tensor
    return tensor.transpose(1, 2).contiguous()


def _context_from_bhsd(
    context: torch.Tensor,
    layout: str,
    *,
    had_query_len: bool,
) -> torch.Tensor:
    if context.ndim == 3:
        return context
    if layout == "bhsd":
        return context
    return context.transpose(1, 2).contiguous() if had_query_len else context
