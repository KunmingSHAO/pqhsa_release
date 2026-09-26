from __future__ import annotations

from dataclasses import dataclass

import torch

from pq_hsa.attention.module import IVFPQDecodeAttentionModule, ModuleAttentionOutput
from pq_hsa.attention.sparse_attention import SparseAttentionConfig
from pq_hsa.index.ivfpq import IVFPQConfig


@dataclass(slots=True)
class ProjectedAttentionOutput:
    hidden_states: torch.Tensor
    context: torch.Tensor
    attention: ModuleAttentionOutput
    key_states: torch.Tensor
    value_states: torch.Tensor


class IVFPQProjectedAttentionModule(torch.nn.Module):
    """Projection-aware wrapper for replacing a model attention layer.

    The wrapper owns existing projection modules and routes projected heads
    through ``IVFPQDecodeAttentionModule``. It intentionally avoids depending on
    Transformers internals; a model-specific patch only needs to pass its
    q/k/v/o projections and head counts.
    """

    def __init__(
        self,
        *,
        q_proj: torch.nn.Module,
        k_proj: torch.nn.Module,
        v_proj: torch.nn.Module,
        o_proj: torch.nn.Module,
        index_config: IVFPQConfig,
        attention_config: SparseAttentionConfig,
        num_heads: int,
        head_dim: int,
        num_key_value_heads: int | None = None,
        value_dim: int | None = None,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        num_key_value_heads = num_heads if num_key_value_heads is None else num_key_value_heads
        if num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if num_heads % num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads")
        value_dim = head_dim if value_dim is None else value_dim
        if value_dim <= 0:
            raise ValueError("value_dim must be positive")

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.attention = IVFPQDecodeAttentionModule(
            index_config,
            attention_config,
            tensor_layout="bshd",
        )

    @property
    def streams(self):
        return self.attention.streams

    def build_cache(self, hidden_states: torch.Tensor) -> "IVFPQProjectedAttentionModule":
        key_states, value_states = self.project_kv(hidden_states)
        self.attention.build_cache(key_states, value_states)
        return self

    def cache_state_dict(self) -> dict[str, object]:
        return {
            "num_heads": self.num_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "value_dim": self.value_dim,
            "attention": self.attention.cache_state_dict(),
        }

    def load_cache_state_dict(self, state: dict[str, object]) -> "IVFPQProjectedAttentionModule":
        expected = {
            "num_heads": self.num_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "value_dim": self.value_dim,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(f"cache state {name} does not match this projected module")
        attention_state = state.get("attention")
        if not isinstance(attention_state, dict):
            raise ValueError("cache state must contain an attention dict")
        self.attention.load_cache_state_dict(attention_state)
        return self

    def apply_pending_index_update(self) -> int:
        return self.attention.apply_pending_index_update()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        key_value_states: torch.Tensor | None = None,
        append_kv: bool = False,
    ) -> ProjectedAttentionOutput:
        query_states = self.project_query(hidden_states)
        kv_source = hidden_states if key_value_states is None else key_value_states
        key_states, value_states = self.project_kv(kv_source)

        if append_kv:
            attention = self.attention.decode_step(query_states, key_states, value_states)
        else:
            attention = self.attention(query_states)

        hidden = self.o_proj(_merge_heads(attention.context))
        return ProjectedAttentionOutput(
            hidden_states=hidden,
            context=attention.context,
            attention=attention,
            key_states=key_states,
            value_states=value_states,
        )

    def decode_step(
        self,
        hidden_states: torch.Tensor,
        *,
        key_value_states: torch.Tensor | None = None,
        append_kv: bool = True,
    ) -> ProjectedAttentionOutput:
        return self.forward(
            hidden_states,
            key_value_states=key_value_states,
            append_kv=append_kv,
        )

    def project_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _split_heads(
            self.q_proj(hidden_states),
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            name="query_states",
        )

    def project_kv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_states = _split_heads(
            self.k_proj(hidden_states),
            num_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            name="key_states",
        )
        value_states = _split_heads(
            self.v_proj(hidden_states),
            num_heads=self.num_key_value_heads,
            head_dim=self.value_dim,
            name="value_states",
        )
        if self.num_key_value_heads != self.num_heads:
            key_states = _repeat_kv_heads(key_states, self.num_heads // self.num_key_value_heads)
            value_states = _repeat_kv_heads(value_states, self.num_heads // self.num_key_value_heads)
        return key_states, value_states


def _split_heads(
    tensor: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    name: str,
) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"{name} projection output must be [batch, seq, heads * dim]")
    expected = num_heads * head_dim
    if tensor.shape[-1] != expected:
        raise ValueError(f"{name} last dim must be {expected}, got {tensor.shape[-1]}")
    return tensor.reshape(tensor.shape[0], tensor.shape[1], num_heads, head_dim)


def _merge_heads(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
        squeeze_query = True
    elif tensor.ndim == 4:
        squeeze_query = False
    else:
        raise ValueError(f"context must be [batch, heads, dim] or [batch, seq, heads, dim]")
    merged = tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3])
    return merged.squeeze(1) if squeeze_query else merged


def _repeat_kv_heads(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return tensor
    return tensor.repeat_interleave(repeats, dim=2)


def build_projected_attention_from_layer(
    layer: torch.nn.Module,
    *,
    index_config: IVFPQConfig,
    attention_config: SparseAttentionConfig,
    num_heads: int | None = None,
    num_key_value_heads: int | None = None,
    head_dim: int | None = None,
    value_dim: int | None = None,
    q_proj_name: str = "q_proj",
    k_proj_name: str = "k_proj",
    v_proj_name: str = "v_proj",
    o_proj_name: str = "o_proj",
) -> IVFPQProjectedAttentionModule:
    """Build an IVF-PQ projected-attention wrapper from a model layer.

    This is a dependency-free patch helper for HF/Llama-style layers. It reuses
    the layer's projection modules and infers head geometry from common
    attributes on the layer or ``layer.config`` when explicit values are not
    supplied.
    """

    q_proj = _get_projection(layer, q_proj_name)
    k_proj = _get_projection(layer, k_proj_name)
    v_proj = _get_projection(layer, v_proj_name)
    o_proj = _get_projection(layer, o_proj_name)

    inferred_heads = _first_int_attr(
        layer,
        ("num_heads", "num_attention_heads", "n_heads", "n_head"),
    )
    num_heads = inferred_heads if num_heads is None else num_heads
    if num_heads is None:
        raise ValueError("num_heads could not be inferred; pass num_heads explicitly")

    inferred_kv_heads = _first_int_attr(
        layer,
        ("num_key_value_heads", "num_kv_heads", "n_kv_heads"),
    )
    if num_key_value_heads is None:
        num_key_value_heads = num_heads if inferred_kv_heads is None else inferred_kv_heads

    if head_dim is None:
        head_dim = _first_int_attr(layer, ("head_dim",))
    if head_dim is None:
        q_out = _projection_out_features(q_proj, q_proj_name)
        if q_out % num_heads != 0:
            raise ValueError("q_proj output dim must be divisible by num_heads")
        head_dim = q_out // num_heads

    if value_dim is None:
        v_out = _projection_out_features(v_proj, v_proj_name)
        if v_out % num_key_value_heads != 0:
            raise ValueError("v_proj output dim must be divisible by num_key_value_heads")
        value_dim = v_out // num_key_value_heads

    return IVFPQProjectedAttentionModule(
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        o_proj=o_proj,
        index_config=index_config,
        attention_config=attention_config,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        value_dim=value_dim,
    )


def _get_projection(layer: torch.nn.Module, name: str) -> torch.nn.Module:
    projection = getattr(layer, name, None)
    if projection is None:
        raise ValueError(f"layer does not expose projection '{name}'")
    if not isinstance(projection, torch.nn.Module):
        raise ValueError(f"layer.{name} must be a torch.nn.Module")
    return projection


def _first_int_attr(layer: torch.nn.Module, names: tuple[str, ...]) -> int | None:
    for source in (layer, getattr(layer, "config", None)):
        if source is None:
            continue
        for name in names:
            value = getattr(source, name, None)
            if value is not None:
                return int(value)
    return None


def _projection_out_features(projection: torch.nn.Module, name: str) -> int:
    out_features = getattr(projection, "out_features", None)
    if out_features is None:
        raise ValueError(f"{name}.out_features is required for head-dim inference")
    return int(out_features)
