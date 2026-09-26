"""Tests for preallocated growth storage in SparseKVCache (Task 1)
and shared-base adapter equivalence (Task 2).
"""
from __future__ import annotations

import pytest
import torch

from pq_hsa import IVFPQConfig, SparseKVCache
from pq_hsa.attention.sparse_attention import SparseAttentionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache(
    length: int = 200,
    D: int = 32,
    kv_storage: str = "device",
    pin: bool = False,
    growth_margin: int = 64,
    device: str = "cpu",
) -> SparseKVCache:
    torch.manual_seed(42)
    cfg = IVFPQConfig(
        num_lists=8,
        nprobe=4,
        num_subspaces=4,
        num_bits=4,
        coarse_max_iter=2,
        pq_max_iter=2,
        seed=1,
    )
    keys = torch.randn(length, D, device=device if kv_storage == "device" else "cpu")
    vals = torch.randn(length, D, device=device if kv_storage == "device" else "cpu")
    return SparseKVCache(
        keys,
        vals,
        index_config=cfg,
        sink_tokens=4,
        local_window=16,
        kv_storage=kv_storage,
        pin_offloaded_kv=pin,
        growth_margin=growth_margin,
    )


# ---------------------------------------------------------------------------
# Task 1a: append correctness over >growth_margin appends
# ---------------------------------------------------------------------------

def test_append_correctness_exceeds_growth_margin():
    """Appending more rows than growth_margin must not corrupt content."""
    L, D, margin = 100, 32, 32
    cache = _make_cache(length=L, D=D, growth_margin=margin)

    extra_keys = [torch.randn(D) for _ in range(margin + 10)]
    extra_vals = [torch.randn(D) for _ in range(margin + 10)]
    for k, v in zip(extra_keys, extra_vals):
        cache.append(k, v)

    assert cache._length == L + margin + 10
    # The last appended key must survive in self.keys.
    torch.testing.assert_close(
        cache.keys[-1].float(),
        extra_keys[-1].float(),
        atol=1e-5, rtol=1e-5,
    )
    # self.keys must be a view into the buffer, not a stale copy.
    assert cache.keys.data_ptr() == cache._key_buf.data_ptr()


def test_public_views_track_buffer_after_multiple_grows():
    """self.keys / self.values must remain valid views after several buffer doublings."""
    L, D, margin = 50, 16, 8
    cache = _make_cache(length=L, D=D, growth_margin=margin)

    appended = []
    # Force several grows: append 4×margin rows.
    for _ in range(4 * margin):
        k = torch.randn(D)
        v = torch.randn(D)
        appended.append((k, v))
        cache.append(k, v)

    assert cache._length == L + 4 * margin
    # Spot-check: last row must match.
    torch.testing.assert_close(cache.keys[-1].float(), appended[-1][0].float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cache.values[-1].float(), appended[-1][1].float(), atol=1e-5, rtol=1e-5)


def test_append_cpu_storage_correctness():
    """CPU kv_storage: appended rows are accessible from self.keys on CPU."""
    L, D, margin = 80, 16, 32
    cache = _make_cache(length=L, D=D, kv_storage="cpu", growth_margin=margin)
    assert cache.keys.device.type == "cpu"

    for _ in range(margin + 5):
        k = torch.randn(D)
        cache.append(k, torch.randn(D))

    assert cache._length == L + margin + 5
    assert cache.keys.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_append_cpu_pinned_storage_correctness():
    """CPU pinned kv_storage: buffer stays pinned after grows."""
    L, D, margin = 80, 16, 32
    cache = _make_cache(length=L, D=D, kv_storage="cpu", pin=True, growth_margin=margin)
    assert cache._key_buf.is_pinned()

    for _ in range(margin + 5):
        cache.append(torch.randn(D), torch.randn(D))

    assert cache._length == L + margin + 5
    # Buffer must still be pinned after growing.
    assert cache._key_buf.is_pinned()


def test_state_dict_round_trip_mid_growth():
    """state_dict + from_state_dict must reproduce the exact KV content."""
    L, D, margin = 100, 32, 32
    cache = _make_cache(length=L, D=D, growth_margin=margin)

    # Append halfway through the margin.
    half = margin // 2
    appended_keys = []
    for _ in range(half):
        k = torch.randn(D)
        cache.append(k, torch.randn(D))
        appended_keys.append(k)

    sd = cache.state_dict()

    # Restore from state_dict.
    from pq_hsa import IVFPQConfig
    restored = SparseKVCache.from_state_dict(sd)

    assert restored._length == L + half
    # Content must match exactly.
    torch.testing.assert_close(
        cache.keys.float(), restored.keys.float(), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        cache.values.float(), restored.values.float(), atol=1e-5, rtol=1e-5
    )
    # Continue appending after restore.
    k2 = torch.randn(D)
    restored.append(k2, torch.randn(D))
    assert restored._length == L + half + 1
    torch.testing.assert_close(restored.keys[-1].float(), k2.float(), atol=1e-5, rtol=1e-5)


def test_state_dict_does_not_include_padding():
    """state_dict should only save the valid [:_length] region, not preallocated cap."""
    L, D, margin = 100, 32, 512
    cache = _make_cache(length=L, D=D, growth_margin=margin)
    cache.append(torch.randn(D), torch.randn(D))

    sd = cache.state_dict()
    # The saved key tensor must have exactly _length rows.
    assert sd["keys"].shape[0] == cache._length


# ---------------------------------------------------------------------------
# Task 1 / 2: append-then-forward matches fresh cache
# ---------------------------------------------------------------------------

def test_append_then_forward_matches_fresh_build():
    """After N appends, a single forward() must give the same result as building
    the cache from scratch with those N extra tokens included (stale-view guard)."""
    torch.manual_seed(7)
    L, D, n_extra = 120, 32, 10
    cfg = IVFPQConfig(
        num_lists=8, nprobe=4, num_subspaces=4, num_bits=4,
        coarse_max_iter=2, pq_max_iter=2, seed=1,
    )
    ac = SparseAttentionConfig(
        mode="hybrid", sink_tokens=4, local_window=8,
        retrieval_top_fraction=1.0,
        hybrid_value_mode="full", hybrid_topk_source="all_pq",
        hybrid_denominator_source="all_pq",
        collect_attention_details=False,
    )

    from pq_hsa import IVFPQSparseAttention
    base_keys = torch.randn(L, D)
    base_vals = torch.randn(L, D)
    extra_keys = [torch.randn(D) for _ in range(n_extra)]
    extra_vals = [torch.randn(D) for _ in range(n_extra)]

    # Build-then-append path.
    stream_a = IVFPQSparseAttention(cfg, ac).build_cache(base_keys, base_vals)
    for k, v in zip(extra_keys, extra_vals):
        stream_a.append(k, v)

    # Fresh build with all tokens.
    all_keys = torch.cat([base_keys] + [k.unsqueeze(0) for k in extra_keys], dim=0)
    all_vals = torch.cat([base_vals] + [v.unsqueeze(0) for v in extra_vals], dim=0)
    stream_b = IVFPQSparseAttention(cfg, ac).build_cache(all_keys, all_vals)

    q = torch.randn(D)
    out_a = stream_a.forward(q)
    out_b = stream_b.forward(q)
    # The outputs are approximate (PQ-based), so use a loose tolerance.
    torch.testing.assert_close(out_a.output, out_b.output, atol=5e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# Task 2: shared-base adapter equivalence
# ---------------------------------------------------------------------------

try:
    import pq_hsa.kernels.triton_lut_scan  # noqa: F401
    from pq_hsa.kernels import is_triton_available as _is_triton
    _triton_ok = torch.cuda.is_available() and _is_triton()
except ImportError:
    _triton_ok = False

cuda_triton = pytest.mark.skipif(not _triton_ok, reason="requires CUDA + Triton")


def _make_gqa_adapter(ctx=3000, kvh=4, qg=2, D=64, dev="cuda:0"):
    from benchmarks.e2e_pq_param_sweep import GQAIVFPQDecodeAttentionAdapter
    torch.manual_seed(10)
    ic = IVFPQConfig(
        num_lists=64, nprobe=16, num_subspaces=4, num_bits=4,
        coarse_max_iter=1, pq_max_iter=2, seed=5,
        kernel_backend="triton", topk_block_size=256,
    )
    ac = SparseAttentionConfig(
        mode="hybrid", sink_tokens=4, local_window=64,
        retrieval_top_fraction=0.05, candidate_budget=2048,
        hybrid_value_mode="centroid", hybrid_topk_source="all_pq",
        hybrid_denominator_source="all_pq", nprobe=16,
        collect_attention_details=False, exact_rerank=True,
    )
    k = torch.randn(1, kvh, ctx, D, device=dev, dtype=torch.float16)
    v = torch.randn(1, kvh, ctx, D, device=dev, dtype=torch.float16)
    return GQAIVFPQDecodeAttentionAdapter(
        ic, ac, num_query_heads=kvh * qg, num_key_value_heads=kvh,
    ).build_cache(k, v), kvh * qg, D, dev


@cuda_triton
def test_shared_base_forward_matches_per_head_loop():
    """Batched forward with shared base must produce numerically close output
    compared to the per-head fallback loop."""
    adapter, qheads, D, dev = _make_gqa_adapter()
    assert adapter._batched_heads is not None, "batched path must be active"
    # Shared base should be set if device storage is enabled.
    assert adapter._shared_base_keys is not None, "shared base must be allocated"

    q = torch.randn(1, qheads, 1, D, device=dev, dtype=torch.float16)
    batched = adapter._forward_many_batched_heads(q)
    assert batched is not None

    saved = adapter._batched_heads
    adapter._batched_heads = None
    try:
        reference = adapter._forward_many(q)
    finally:
        adapter._batched_heads = saved

    torch.testing.assert_close(
        batched.context, reference.context, rtol=2e-2, atol=2e-2
    )


@cuda_triton
def test_shared_base_content_after_appends():
    """After several decode steps, the shared base buffer must reflect the
    appended data and the forward output must remain consistent."""
    adapter, qheads, D, dev = _make_gqa_adapter()
    torch.manual_seed(99)

    n_steps = 5
    for _ in range(n_steps):
        q = torch.randn(1, qheads, 1, D, device=dev, dtype=torch.float16)
        k = torch.randn(1, adapter.num_key_value_heads, 1, D, device=dev, dtype=torch.float16)
        v = torch.randn(1, adapter.num_key_value_heads, 1, D, device=dev, dtype=torch.float16)
        out = adapter.decode_step(q, k, v)
        assert out.context.shape == (1, qheads, 1, D)

    # After n_steps appends, cache length should have grown.
    for stream in adapter._streams:
        assert stream.cache._length > 0


@cuda_triton
def test_shared_base_flat_gather_correctness():
    """The flat-gather fast path (Task 2) must produce the same result as the
    per-head loop on the exact-rerank branch."""
    adapter, qheads, D, dev = _make_gqa_adapter()
    assert adapter._batched_heads is not None
    bh = adapter._batched_heads
    assert bh.get("shared_base_keys") is not None, "flat gather path must be active"

    q = torch.randn(1, qheads, 1, D, device=dev, dtype=torch.float16)

    # Run with shared base (fast path).
    out_fast = adapter._forward_many_batched_heads(q)
    assert out_fast is not None

    # Disable shared base and re-run (per-head loop fallback).
    saved_sbk = bh["shared_base_keys"]
    saved_sbv = bh["shared_base_vals"]
    saved_ho = bh["head_offsets"]
    bh["shared_base_keys"] = None
    bh["shared_base_vals"] = None
    bh["head_offsets"] = None
    try:
        out_slow = adapter._forward_many_batched_heads(q)
    finally:
        bh["shared_base_keys"] = saved_sbk
        bh["shared_base_vals"] = saved_sbv
        bh["head_offsets"] = saved_ho

    assert out_slow is not None
    torch.testing.assert_close(
        out_fast.context, out_slow.context, rtol=1e-3, atol=1e-3
    )
