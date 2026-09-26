"""Head-batched GQA sidecar: reuse the paper's own batched-heads decode adapter.

The original ``GQASidecar`` instantiates one ``IVFPQSparseAttention`` per KV head
and loops over them in Python. Profiling showed that costs ~922 CUDA launches per layer per token at only 21-25% GPU
occupancy, i.e. the sidecar was launch-bound, not compute-bound.

The paper harness already has the fix: ``GQAIVFPQDecodeAttentionAdapter`` in
``benchmarks/e2e_pq_param_sweep.py`` stacks all KV heads into ``[H, ...]`` tensors and
drives the H20 pair-LUT kernel with ``H>1`` (``_forward_many_batched_heads``), plus an
optional per-layer CUDA graph (``PQ_USE_CUDA_GRAPH=1``). This module exposes that
adapter behind the same tiny interface the vLLM wrap already uses, so nothing in
``pq_hsa/`` is touched.

``e2e_pq_param_sweep`` imports ``datasets`` at module scope only for its own dataset
loaders, which we never call; the vLLM environment need not have it. We stub the
module rather than requiring it there, to keep the vLLM/torch pins untouched.
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
import types

import torch


def _import_adapter_cls():
    if "datasets" not in sys.modules:
        stub = types.ModuleType("datasets")

        def _unavailable(*_args, **_kwargs):
            raise RuntimeError("datasets is not installed in the vLLM environment (not needed here)")

        stub.load_dataset = _unavailable  # type: ignore[attr-defined]
        # transformers probes optional deps with importlib.util.find_spec, which raises
        # on a module whose __spec__ is None. A real spec makes find_spec succeed while
        # importlib.metadata.version() still fails, so transformers marks it unavailable.
        stub.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
        sys.modules["datasets"] = stub

    from benchmarks.e2e_pq_param_sweep import GQAIVFPQDecodeAttentionAdapter

    return GQAIVFPQDecodeAttentionAdapter


class GQASidecarFast:
    """Same surface as ``GQASidecar`` but backed by the batched-heads adapter."""

    def __init__(
        self,
        index_config,
        attention_config,
        *,
        num_query_heads: int,
        num_kv_heads: int,
    ) -> None:
        cls = _import_adapter_cls()
        self.num_query_heads = int(num_query_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.adapter = cls(
            index_config,
            attention_config,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_kv_heads,
        )
        self.length = 0
        self._t83r_fast = None   # Bound fast-path, armed under PQ_HSA_LEAN_REPLAY

    @property
    def batched_heads_active(self) -> bool:
        return getattr(self.adapter, "_batched_heads", None) is not None

    @property
    def cuda_graph_active(self) -> bool:
        return getattr(self.adapter, "_cg", None) is not None

    @property
    def prefix_checkpoint_length(self) -> int | None:
        return getattr(self.adapter, "prefix_checkpoint_length", None)

    def build_cache(self, keys: torch.Tensor, values: torch.Tensor) -> "GQASidecarFast":
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys/values must be [1, kv_heads, seq, dim]")
        self.adapter.build_cache(keys, values)
        self.length = int(keys.shape[2])
        return self

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.ndim != 3 or values.ndim != 3:
            raise ValueError("append keys/values must be [1, kv_heads, dim]")
        self.adapter.append(keys, values)
        self.length += 1

    def lean_append(self, keys: torch.Tensor, values: torch.Tensor, q_src=None) -> bool:
        """Opt-in one-launch append (+ q staging). False -> use append()."""
        fn = getattr(self.adapter, "lean_append", None)
        if fn is None:
            return False
        if fn(keys, values, q_src):
            self.length += 1
            return True
        return False

    def set_paged_kv_context(self, kv_cache, attn_metadata, seq_idx: int = 0) -> None:
        """(PQ_HSA_PAGED_KV=1, opt-in): forward to the wrapped adapter."""
        self.adapter.set_paged_kv_context(kv_cache, attn_metadata, seq_idx)

    def checkpoint_prefix(self) -> None:
        self.adapter.checkpoint_prefix()

    def restore_prefix(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        kv_cache: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        seq_idx: int = 0,
    ) -> bool:
        # Forward the optional paged-KV refs so the wrapped adapter can
        # refresh the stale tail-block range (see its restore_prefix docstring
        # in e2e_pq_param_sweep.py). Both default to None -- unchanged
        # behavior for callers that don't pass them.
        restored = bool(
            self.adapter.restore_prefix(
                keys, values, kv_cache=kv_cache, attn_metadata=attn_metadata, seq_idx=seq_idx
            )
        )
        if restored:
            checkpoint_length = self.prefix_checkpoint_length
            assert checkpoint_length is not None
            self.length = int(checkpoint_length)
        return restored

    def append_many(self, keys: torch.Tensor, values: torch.Tensor, *, flush_every: int = 0) -> int:
        """Vectorized append of a contiguous token run ([1,H,n,D] or [H,n,D])."""
        n = int(keys.shape[-2])
        flushes = int(self.adapter.append_many(keys, values, flush_every=flush_every))
        self.length += n
        return flushes

    def extend_prefix(
        self,
        keys: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        *,
        kv_cache: torch.Tensor,
        attn_metadata: object,
        seq_idx: int = 0,
        new_length: int,
    ):
        """(opt-in PQ_HSA_PREFIX_EXTEND=1): forward to the wrapped adapter;
        on success the wrapper length follows the extended prefix."""
        fn = getattr(self.adapter, "extend_prefix", None)
        if fn is None:
            return None
        info = fn(
            keys, values, kv_cache=kv_cache, attn_metadata=attn_metadata,
            seq_idx=seq_idx, new_length=new_length,
        )
        if info is not None:
            self.length = int(new_length)
        return info

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """queries: ``[1, q_heads, dim]`` → context ``[1, q_heads, dim]``.

        Uses the adapter's graph-replay fast path when available: the
        returned tensor may alias the graph's static output buffer, so the
        vLLM wrap must copy it out before the next decode step (it does —
        ``_copy_ctx`` writes into vLLM's output buffer immediately).
        """
        # In steady decode the shape check and the getattr lookup are
        # re-done 32x per token for a value that cannot change; bind once.
        _f = self._t83r_fast
        if _f is not None:
            return _f(queries)
        if queries.ndim != 3 or queries.shape[0] != 1:
            raise ValueError(f"queries must be [1, q_heads, dim], got {tuple(queries.shape)}")
        fast = getattr(self.adapter, "forward_context_fast", None)
        if fast is not None:
            if os.environ.get("PQ_HSA_LEAN_REPLAY", "0") == "1":
                self._t83r_fast = fast
            return fast(queries)
        return self.adapter.forward(queries).context
