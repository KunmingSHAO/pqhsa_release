"""Tests for Feature 3: SinkLocalDecodeAdapter (StreamingLLM-style baseline).

Covers:
  - build_cache initializes correctly from a full KV tensor.
  - forward output matches a reference manual sink+local softmax computation.
  - append correctly updates the ring buffer.
  - decode_step (append_kv=True) is equivalent to append + forward.
  - GQA (groups > 1) output shape is correct.
  - Overflow of the local ring buffer is handled correctly.
  - SinkLocalDecodeAdapter.streams is an empty tuple (for _collect_adapter_memory).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.e2e_pq_param_sweep import SinkLocalDecodeAdapter


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _reference_sink_local_attn(
    query: torch.Tensor,   # [H, D]
    keys: torch.Tensor,    # [H, S, D]
    values: torch.Tensor,  # [H, S, Dv]
) -> torch.Tensor:
    """Exact scaled dot-product attention for one query token."""
    H, D = query.shape
    scale = 1.0 / math.sqrt(D)
    # logits: [H, S]
    logits = torch.einsum("hd,hsd->hs", query.float(), keys.float()) * scale
    weights = torch.softmax(logits, dim=-1)  # [H, S]
    ctx = torch.einsum("hs,hsd->hd", weights, values.float())  # [H, Dv]
    return ctx.to(query.dtype)


# ---------------------------------------------------------------------------
# Helper to build adapter + expected KV
# ---------------------------------------------------------------------------

def _make_adapter_and_kv(
    seq: int = 20,
    sink: int = 4,
    local: int = 8,
    H: int = 2,
    D: int = 16,
    Dv: int = 16,
    seed: int = 0,
) -> tuple[SinkLocalDecodeAdapter, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    keys = torch.randn(1, H, seq, D, dtype=torch.float32)
    values = torch.randn(1, H, seq, Dv, dtype=torch.float32)
    adapter = SinkLocalDecodeAdapter(
        sink_tokens=sink,
        local_window=local,
        num_query_heads=H,
        num_key_value_heads=H,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    adapter.build_cache(keys, values)
    return adapter, keys.squeeze(0), values.squeeze(0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_streams_is_empty():
    """SinkLocalDecodeAdapter.streams returns an empty tuple."""
    adapter, _, _ = _make_adapter_and_kv()
    assert adapter.streams == ()


def test_build_cache_stores_sink():
    """Sink tokens (first `sink` rows) are stored correctly."""
    sink = 4
    adapter, keys, values = _make_adapter_and_kv(sink=sink)
    assert adapter._sink_k is not None
    assert adapter._sink_k.shape == (2, sink, 16)
    torch.testing.assert_close(adapter._sink_k, keys[:, :sink, :])


def test_build_cache_ring_buffer_filled_from_tail():
    """Local ring buffer is filled from the tail of the prefill sequence."""
    seq, sink, local = 20, 4, 8
    adapter, keys, values = _make_adapter_and_kv(seq=seq, sink=sink, local=local)
    # Expected tail: tokens [12, 20) = last 8 tokens (since 20-8=12 > sink=4)
    expected_local = keys[:, 12:, :]
    assert adapter._local_filled == local
    # Reconstruct the ring in order (ring is full, starts at write_pos=0).
    local_k = adapter._local_k[:, :local, :]
    torch.testing.assert_close(local_k, expected_local)


def test_forward_matches_reference_sink_only():
    """When seq <= sink, all tokens are in sink, local is empty; output matches reference."""
    sink = 20
    local = 8
    seq = 10  # all tokens fall in sink
    torch.manual_seed(42)
    keys = torch.randn(1, 2, seq, 16, dtype=torch.float32)
    values = torch.randn(1, 2, seq, 16, dtype=torch.float32)
    adapter = SinkLocalDecodeAdapter(
        sink_tokens=sink, local_window=local,
        num_query_heads=2, num_key_value_heads=2,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    adapter.build_cache(keys, values)

    query = torch.randn(1, 2, 1, 16, dtype=torch.float32)
    out = adapter.forward(query)
    # Reference: all 10 tokens are in sink (no local for this short seq).
    ref = _reference_sink_local_attn(
        query.squeeze(0).squeeze(1),   # [2, 16]
        keys.squeeze(0),               # [2, 10, 16]
        values.squeeze(0),             # [2, 10, 16]
    )
    torch.testing.assert_close(out.context.squeeze(-2).squeeze(0), ref, atol=1e-5, rtol=1e-5)


def test_forward_matches_reference_sink_plus_local():
    """Output matches reference when both sink and local KV are present."""
    seq, sink, local = 30, 4, 8
    torch.manual_seed(7)
    keys = torch.randn(1, 2, seq, 16, dtype=torch.float32)
    values = torch.randn(1, 2, seq, 16, dtype=torch.float32)
    adapter = SinkLocalDecodeAdapter(
        sink_tokens=sink, local_window=local,
        num_query_heads=2, num_key_value_heads=2,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    adapter.build_cache(keys, values)

    query = torch.randn(1, 2, 1, 16, dtype=torch.float32)
    out = adapter.forward(query)

    # Reference: sink (0:4) + local (last 8 = 22:30).
    kv_k = torch.cat([keys.squeeze(0)[:, :sink, :], keys.squeeze(0)[:, seq - local:, :]], dim=1)
    kv_v = torch.cat([values.squeeze(0)[:, :sink, :], values.squeeze(0)[:, seq - local:, :]], dim=1)
    ref = _reference_sink_local_attn(query.squeeze(0).squeeze(1), kv_k, kv_v)
    torch.testing.assert_close(out.context.squeeze(-2).squeeze(0), ref, atol=1e-5, rtol=1e-5)


def test_append_updates_ring_buffer():
    """After one append, the new token appears in the local KV."""
    seq, sink, local = 20, 4, 8
    adapter, keys, values = _make_adapter_and_kv(seq=seq, sink=sink, local=local, seed=10)

    new_k = torch.randn(2, 16, dtype=torch.float32)  # [H, D]
    new_v = torch.randn(2, 16, dtype=torch.float32)
    adapter.append(new_k, new_v)

    # After one append (ring was full), write_pos should have advanced by 1.
    write_pos = adapter._local_write_pos
    # The slot just written is (write_pos - 1) % local_window.
    prev_pos = (write_pos - 1) % adapter.local_window
    torch.testing.assert_close(adapter._local_k[:, prev_pos, :], new_k)
    torch.testing.assert_close(adapter._local_v[:, prev_pos, :], new_v)


def test_decode_step_equivalent_to_append_plus_forward():
    """decode_step(append_kv=True) gives the same result as append + forward."""
    seq, sink, local = 20, 4, 8
    torch.manual_seed(11)
    keys = torch.randn(1, 2, seq, 16, dtype=torch.float32)
    values = torch.randn(1, 2, seq, 16, dtype=torch.float32)

    adapter_a = SinkLocalDecodeAdapter(
        sink_tokens=sink, local_window=local,
        num_query_heads=2, num_key_value_heads=2,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    adapter_b = SinkLocalDecodeAdapter(
        sink_tokens=sink, local_window=local,
        num_query_heads=2, num_key_value_heads=2,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    adapter_a.build_cache(keys.clone(), values.clone())
    adapter_b.build_cache(keys.clone(), values.clone())

    query = torch.randn(1, 2, 1, 16, dtype=torch.float32)
    new_k = torch.randn(1, 2, 16, dtype=torch.float32)
    new_v = torch.randn(1, 2, 16, dtype=torch.float32)

    adapter_a.append(new_k, new_v)
    out_a = adapter_a.forward(query)
    out_b = adapter_b.decode_step(query, new_k, new_v, append_kv=True)

    torch.testing.assert_close(out_a.context, out_b.context)


def test_gqa_output_shape():
    """GQA (groups > 1) produces output of shape [1, num_query_heads, 1, Dv]."""
    seq, sink, local = 20, 4, 8
    kv_heads = 2
    groups = 4
    q_heads = kv_heads * groups
    D = 16
    Dv = 16
    torch.manual_seed(12)
    keys = torch.randn(1, kv_heads, seq, D, dtype=torch.float32)
    values = torch.randn(1, kv_heads, seq, Dv, dtype=torch.float32)
    adapter = SinkLocalDecodeAdapter(
        sink_tokens=sink, local_window=local,
        num_query_heads=q_heads, num_key_value_heads=kv_heads,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    adapter.build_cache(keys, values)

    query = torch.randn(1, q_heads, 1, D, dtype=torch.float32)
    out = adapter.forward(query)
    assert out.context.shape == (1, q_heads, 1, Dv)


def test_ring_buffer_overflow():
    """After more appends than local_window, old tokens are evicted (ring overflows)."""
    seq, sink, local = 20, 4, 8
    adapter, _, _ = _make_adapter_and_kv(seq=seq, sink=sink, local=local, seed=13)
    # Append local+1 new tokens — the oldest local token should be evicted.
    sentinel_k = torch.ones(2, 16, dtype=torch.float32) * 999.0
    for i in range(local):
        k = torch.full((2, 16), float(i), dtype=torch.float32)
        v = torch.full((2, 16), float(i), dtype=torch.float32)
        adapter.append(k, v)
    # Now append sentinel.
    adapter.append(sentinel_k, torch.zeros(2, 16))
    # Ring is full; the local buffer should contain the most recent `local` tokens.
    # The sentinel was the (local+1)-th append, so it's in the ring.
    found = (adapter._local_k == 999.0).any().item()
    assert found, "Sentinel key should be present in the ring buffer"


def test_per_query_is_empty_tuple():
    """SinkLocalOutput.per_query returns empty tuple (for _update_attention_stats)."""
    adapter, _, _ = _make_adapter_and_kv()
    query = torch.randn(1, 2, 1, 16, dtype=torch.float32)
    out = adapter.forward(query)
    assert tuple(out.per_query) == ()


def test_forward_output_shape_3d_query():
    """forward accepts 3D query [batch, heads, D] and returns unsqueezed context."""
    adapter, _, _ = _make_adapter_and_kv()
    query_3d = torch.randn(1, 2, 16, dtype=torch.float32)
    out = adapter.forward(query_3d)
    # When squeeze is applied, context loses the query_len=1 dim.
    assert out.context.shape == (1, 2, 16)


def test_missing_keys_values_in_decode_step_raises():
    """decode_step with append_kv=True but no keys/values must raise ValueError."""
    adapter, _, _ = _make_adapter_and_kv()
    query = torch.randn(1, 2, 1, 16, dtype=torch.float32)
    with pytest.raises(ValueError, match="keys and values are required"):
        adapter.decode_step(query, append_kv=True)


def test_ring_buffer_wraparound_forward_matches_reference():
    """After ring buffer fills and wraps around, forward() matches manual reference attention.

    Scenario:
      - Prefill seq=20, sink=4, local=8 → ring is full with tokens [12, 20),
        write_pos=0.
      - Append 5 new tokens → write_pos=5, ring slots [0..4] overwritten with
        new tokens; slots [5..7] still hold old prefill tokens [12+5, 20)=[17,20).
      - Ordered local window (oldest→newest): slots [5..7] ++ slots [0..4]
        = prefill[17:20] ++ new tokens 0..4.
      - Reference: sink KV (prefill[0:4]) ++ ordered local → exact softmax.
    """
    seq, sink, local = 20, 4, 8
    n_new = 5  # append this many tokens after prefill; ring wraps at slot n_new
    H, D = 2, 16
    torch.manual_seed(99)

    keys_prefill = torch.randn(1, H, seq, D, dtype=torch.float32)
    vals_prefill = torch.randn(1, H, seq, D, dtype=torch.float32)

    adapter = SinkLocalDecodeAdapter(
        sink_tokens=sink, local_window=local,
        num_query_heads=H, num_key_value_heads=H,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    adapter.build_cache(keys_prefill.clone(), vals_prefill.clone())

    # Ring is full (write_pos=0). Append n_new tokens.
    new_keys = torch.randn(n_new, H, D, dtype=torch.float32)
    new_vals = torch.randn(n_new, H, D, dtype=torch.float32)
    for i in range(n_new):
        adapter.append(new_keys[i], new_vals[i])  # shape [H, D]

    # Sanity: ring is still full, write_pos == n_new.
    assert adapter._local_filled == local
    assert adapter._local_write_pos == n_new

    # Build reference ordered local KV (oldest→newest after wrap):
    #   slots [n_new .. local-1] hold old prefill tail starting at prefill[seq-local+n_new]
    #   slots [0 .. n_new-1] hold new_keys[0..n_new-1]
    prefill_flat_k = keys_prefill.squeeze(0)   # [H, seq, D]
    prefill_flat_v = vals_prefill.squeeze(0)

    # oldest part: prefill tokens that weren't overwritten
    tail_start = seq - local  # =12
    old_k = prefill_flat_k[:, tail_start + n_new:, :]  # [H, local-n_new, D]
    old_v = prefill_flat_v[:, tail_start + n_new:, :]

    # newest part: new appended tokens
    new_k_ref = new_keys.permute(1, 0, 2)  # [H, n_new, D]
    new_v_ref = new_vals.permute(1, 0, 2)

    local_k_ref = torch.cat([old_k, new_k_ref], dim=1)  # [H, local, D]
    local_v_ref = torch.cat([old_v, new_v_ref], dim=1)

    # Full KV for reference: sink ++ ordered local
    sink_k_ref = prefill_flat_k[:, :sink, :]
    sink_v_ref = prefill_flat_v[:, :sink, :]
    ref_k = torch.cat([sink_k_ref, local_k_ref], dim=1)  # [H, sink+local, D]
    ref_v = torch.cat([sink_v_ref, local_v_ref], dim=1)

    query = torch.randn(1, H, 1, D, dtype=torch.float32)
    out = adapter.forward(query)

    ref = _reference_sink_local_attn(
        query.squeeze(0).squeeze(1),  # [H, D]
        ref_k,                        # [H, S, D]
        ref_v,                        # [H, S, D]
    )
    torch.testing.assert_close(
        out.context.squeeze(-2).squeeze(0), ref, atol=1e-5, rtol=1e-5
    )
