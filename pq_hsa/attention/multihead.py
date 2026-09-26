from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from pq_hsa.attention.sparse_attention import (
    AttentionOutput,
    IVFPQSparseAttention,
    SparseAttentionConfig,
)
from pq_hsa.attention.kv_cache import SparseKVCache
from pq_hsa.index.ivfpq import IVFPQConfig


@dataclass(slots=True)
class MultiHeadAttentionOutput:
    output: torch.Tensor
    per_stream: tuple[AttentionOutput, ...]
    leading_shape: tuple[int, ...]


@dataclass(slots=True)
class MultiQueryAttentionOutput:
    output: torch.Tensor
    per_query: tuple[MultiHeadAttentionOutput, ...]
    leading_shape: tuple[int, ...]


class IVFPQMultiHeadAttention:
    """Manage one IVF-PQ sparse-attention cache per batch/head stream.

    The single-stream attention core works on ``[seq, dim]`` K/V and one query.
    This wrapper accepts real KV-cache style tensors:

        keys:    ``[*leading, seq, key_dim]``
        values:  ``[*leading, seq, value_dim]``
        queries: ``[*leading, key_dim]``

    ``leading`` can be ``[heads]`` or ``[batch, heads]``. Each leading position
    owns an independent IVF-PQ index because attention heads have different key
    distributions and should not share code assignments.
    """

    def __init__(self, index_config: IVFPQConfig, attention_config: SparseAttentionConfig):
        self.index_config = index_config
        self.attention_config = attention_config
        self._streams: tuple[IVFPQSparseAttention, ...] = ()
        self._leading_shape: tuple[int, ...] | None = None
        self._key_dim: int | None = None
        self._value_dim: int | None = None

    @property
    def streams(self) -> tuple[IVFPQSparseAttention, ...]:
        return self._streams

    @property
    def leading_shape(self) -> tuple[int, ...]:
        if self._leading_shape is None:
            raise RuntimeError("build_cache must be called before reading leading_shape")
        return self._leading_shape

    def build_cache(self, keys: torch.Tensor, values: torch.Tensor) -> "IVFPQMultiHeadAttention":
        if keys.ndim < 3 or values.ndim < 3:
            raise ValueError("keys and values must be [*leading, seq, dim]")
        if keys.shape[:-2] != values.shape[:-2]:
            raise ValueError("keys and values must have matching leading dimensions")
        if keys.shape[-2] != values.shape[-2]:
            raise ValueError("keys and values must have the same sequence length")

        self._leading_shape = tuple(keys.shape[:-2])
        self._key_dim = keys.shape[-1]
        self._value_dim = values.shape[-1]

        flat_keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])
        flat_values = values.reshape(-1, values.shape[-2], values.shape[-1])
        streams = []
        for stream_keys, stream_values in zip(flat_keys, flat_values, strict=True):
            streams.append(
                IVFPQSparseAttention(self.index_config, self.attention_config).build_cache(
                    stream_keys,
                    stream_values,
                )
            )
        self._streams = tuple(streams)
        return self

    def state_dict(self) -> dict[str, object]:
        """Return a snapshot for all batch/head sparse-attention streams."""

        self._check_built()
        assert self._leading_shape is not None
        assert self._key_dim is not None
        assert self._value_dim is not None
        return {
            "index_config": asdict(self.index_config),
            "attention_config": asdict(self.attention_config),
            "leading_shape": self._leading_shape,
            "key_dim": self._key_dim,
            "value_dim": self._value_dim,
            "streams": tuple(stream.cache.state_dict() for stream in self._streams),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, object],
        *,
        index_config: IVFPQConfig | None = None,
        attention_config: SparseAttentionConfig | None = None,
    ) -> "IVFPQMultiHeadAttention":
        if index_config is None:
            index_config = _index_config_from_state(state)
        if attention_config is None:
            attention_config = _attention_config_from_state(state)
        attention = cls(index_config, attention_config)
        attention.load_state_dict(state)
        return attention

    def load_state_dict(self, state: dict[str, object]) -> "IVFPQMultiHeadAttention":
        """Load all stream cache states without rebuilding indexes."""

        state_index_config = _index_config_from_state(state)
        state_attention_config = _attention_config_from_state(state)
        if asdict(self.index_config) != asdict(state_index_config):
            raise ValueError("state index_config does not match this multi-head attention")
        if asdict(self.attention_config) != asdict(state_attention_config):
            raise ValueError("state attention_config does not match this multi-head attention")

        streams_state = state.get("streams")
        if not isinstance(streams_state, (tuple, list)):
            raise ValueError("state must contain a tuple/list of stream states")
        leading_shape = _shape_tuple(state.get("leading_shape"), "leading_shape")
        key_dim = int(state["key_dim"])
        value_dim = int(state["value_dim"])
        expected_streams = 1
        for size in leading_shape:
            expected_streams *= size
        if len(streams_state) != expected_streams:
            raise ValueError("stream state count does not match leading_shape")

        streams = []
        for stream_state in streams_state:
            if not isinstance(stream_state, dict):
                raise ValueError("each stream state must be a dict")
            stream = IVFPQSparseAttention(self.index_config, self.attention_config)
            stream.cache = SparseKVCache.from_state_dict(stream_state, index_config=self.index_config)
            streams.append(stream)

        self._leading_shape = leading_shape
        self._key_dim = key_dim
        self._value_dim = value_dim
        self._streams = tuple(streams)
        return self

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self._check_built()
        assert self._leading_shape is not None
        assert self._key_dim is not None
        assert self._value_dim is not None

        expected_key_shape = (*self._leading_shape, self._key_dim)
        expected_value_shape = (*self._leading_shape, self._value_dim)
        if tuple(keys.shape) != expected_key_shape:
            raise ValueError(f"keys must have shape {expected_key_shape}, got {tuple(keys.shape)}")
        if tuple(values.shape) != expected_value_shape:
            raise ValueError(
                f"values must have shape {expected_value_shape}, got {tuple(values.shape)}"
            )

        flat_keys = keys.reshape(-1, self._key_dim)
        flat_values = values.reshape(-1, self._value_dim)
        for stream, key, value in zip(self._streams, flat_keys, flat_values, strict=True):
            stream.append(key, value)

    def apply_pending_index_update(self) -> int:
        self._check_built()
        return sum(1 for stream in self._streams if stream.apply_pending_index_update())

    def forward(self, queries: torch.Tensor) -> MultiHeadAttentionOutput:
        self._check_built()
        assert self._leading_shape is not None
        assert self._key_dim is not None
        assert self._value_dim is not None

        expected_query_shape = (*self._leading_shape, self._key_dim)
        if tuple(queries.shape) != expected_query_shape:
            raise ValueError(
                f"queries must have shape {expected_query_shape}, got {tuple(queries.shape)}"
            )

        flat_queries = queries.reshape(-1, self._key_dim)
        per_stream = tuple(
            stream.forward(query)
            for stream, query in zip(self._streams, flat_queries, strict=True)
        )
        output = torch.stack([stream_output.output for stream_output in per_stream], dim=0)
        output = output.reshape(*self._leading_shape, self._value_dim)
        return MultiHeadAttentionOutput(
            output=output,
            per_stream=per_stream,
            leading_shape=self._leading_shape,
        )

    def forward_many(self, queries: torch.Tensor) -> MultiQueryAttentionOutput:
        self._check_built()
        assert self._leading_shape is not None
        assert self._key_dim is not None
        assert self._value_dim is not None

        expected_prefix = (*self._leading_shape,)
        if queries.ndim != len(expected_prefix) + 2:
            raise ValueError(
                "queries must have shape "
                f"{(*expected_prefix, 'query_len', self._key_dim)}, got {tuple(queries.shape)}"
            )
        if tuple(queries.shape[:-2]) != expected_prefix or queries.shape[-1] != self._key_dim:
            raise ValueError(
                "queries must have shape "
                f"{(*expected_prefix, 'query_len', self._key_dim)}, got {tuple(queries.shape)}"
            )

        query_len = queries.shape[-2]
        flat_queries = queries.reshape(-1, query_len, self._key_dim)
        per_stream_many = tuple(
            stream.forward_many(stream_queries)
            for stream, stream_queries in zip(self._streams, flat_queries, strict=True)
        )
        per_query = []
        query_outputs = []
        num_streams = len(self._streams)
        for position in range(query_len):
            current_per_stream = tuple(stream_outputs[position] for stream_outputs in per_stream_many)
            current_output = torch.stack(
                [stream_output.output for stream_output in current_per_stream],
                dim=0,
            ).reshape(*self._leading_shape, self._value_dim)
            query_outputs.append(current_output)
            per_query.append(
                MultiHeadAttentionOutput(
                    output=current_output,
                    per_stream=current_per_stream,
                    leading_shape=self._leading_shape,
                )
            )

        output = torch.stack(query_outputs, dim=len(self._leading_shape))
        assert output.shape == (*self._leading_shape, query_len, self._value_dim)
        assert all(len(item.per_stream) == num_streams for item in per_query)
        return MultiQueryAttentionOutput(
            output=output,
            per_query=tuple(per_query),
            leading_shape=self._leading_shape,
        )

    def _check_built(self) -> None:
        if not self._streams:
            raise RuntimeError("build_cache must be called before using IVFPQMultiHeadAttention")


def _index_config_from_state(state: dict[str, object]) -> IVFPQConfig:
    config = state.get("index_config")
    if not isinstance(config, dict):
        raise ValueError("state must contain an index_config dict")
    return IVFPQConfig(**config)


def _attention_config_from_state(state: dict[str, object]) -> SparseAttentionConfig:
    config = state.get("attention_config")
    if not isinstance(config, dict):
        raise ValueError("state must contain an attention_config dict")
    return SparseAttentionConfig(**config)


def _shape_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"state['{name}'] must be a tuple/list")
    return tuple(int(item) for item in value)
