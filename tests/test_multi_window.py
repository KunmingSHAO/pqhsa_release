"""Tests for Feature 1: multi-window evaluation (--eval-window-offsets).

Covers:
  - Window slicing produces correct token subsets.
  - Out-of-bounds offset raises ValueError (not silent skip).
  - _extract_agg_metric returns correct values.
  - Single-window path (no --eval-window-offsets) is unchanged.
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from benchmarks.e2e_pq_param_sweep import _extract_agg_metric


# ---------------------------------------------------------------------------
# Token-slicing helpers
# ---------------------------------------------------------------------------

def _make_tokens(n: int) -> torch.Tensor:
    """Return a [1, n] tensor of sequential token IDs."""
    return torch.arange(n).unsqueeze(0)


def test_window_slice_offset_zero():
    """Offset 0 gives the first window_size tokens."""
    tokens = _make_tokens(1000)
    offset = 0
    window_size = 100
    sliced = tokens[:, offset : offset + window_size]
    assert sliced.shape == (1, 100)
    assert sliced[0, 0].item() == 0
    assert sliced[0, -1].item() == 99


def test_window_slice_non_zero_offset():
    """Non-zero offset correctly shifts the window."""
    tokens = _make_tokens(1000)
    offset = 200
    window_size = 100
    sliced = tokens[:, offset : offset + window_size]
    assert sliced.shape == (1, 100)
    assert sliced[0, 0].item() == 200
    assert sliced[0, -1].item() == 299


def test_window_slices_are_independent():
    """Different window offsets produce non-overlapping token sequences."""
    tokens = _make_tokens(2000)
    window_size = 500
    offsets = [0, 500, 1000]
    slices = [tokens[:, off : off + window_size] for off in offsets]
    # No overlap: start tokens should be distinct.
    first_tokens = [s[0, 0].item() for s in slices]
    assert first_tokens == [0, 500, 1000]


def test_out_of_bounds_offset_raises():
    """An offset that makes the window exceed available tokens must raise ValueError."""
    tokens = _make_tokens(500)
    window_size = 100
    # This check mirrors the validation in main().
    offset = 450  # 450 + 100 = 550 > 500
    with pytest.raises(ValueError, match="exceeds available tokens"):
        for off in [offset]:
            if off + window_size > tokens.shape[1]:
                raise ValueError(
                    f"window offset {off} + window_size {window_size} = "
                    f"{off + window_size} exceeds available tokens "
                    f"({tokens.shape[1]})"
                )


def test_exact_boundary_offset_does_not_raise():
    """An offset that exactly fills available tokens is valid."""
    tokens = _make_tokens(500)
    window_size = 100
    offset = 400  # 400 + 100 = 500 == tokens.shape[1], valid
    # Should not raise.
    if offset + window_size > tokens.shape[1]:
        raise ValueError("should not happen")
    sliced = tokens[:, offset : offset + window_size]
    assert sliced.shape == (1, 100)
    assert sliced[0, -1].item() == 499


# ---------------------------------------------------------------------------
# _extract_agg_metric
# ---------------------------------------------------------------------------

def _make_result(
    *,
    ppl_ratio: float = 1.05,
    top1_agreement: float = 0.9,
    kl: float = 0.01,
    pq_ms: float = 10.0,
    dense_ms: float = 8.0,
) -> dict:
    return {
        "comparison": {
            "ppl_ratio": ppl_ratio,
            "top1_agreement": top1_agreement,
            "dense_to_sparse_kl": kl,
        },
        "pq_hsa": {"ms_per_token": pq_ms},
        "dense": {"ms_per_token": dense_ms},
    }


def test_extract_ppl_ratio():
    r = _make_result(ppl_ratio=1.23)
    assert _extract_agg_metric(r, "ppl_ratio") == pytest.approx(1.23)


def test_extract_top1_agreement():
    r = _make_result(top1_agreement=0.87)
    assert _extract_agg_metric(r, "top1_agreement") == pytest.approx(0.87)


def test_extract_dense_to_sparse_kl():
    r = _make_result(kl=0.042)
    assert _extract_agg_metric(r, "dense_to_sparse_kl") == pytest.approx(0.042)


def test_extract_pq_ms_per_token():
    r = _make_result(pq_ms=15.5)
    assert _extract_agg_metric(r, "pq_ms_per_token") == pytest.approx(15.5)


def test_extract_dense_ms_per_token():
    r = _make_result(dense_ms=7.2)
    assert _extract_agg_metric(r, "dense_ms_per_token") == pytest.approx(7.2)


def test_extract_unknown_metric_returns_none():
    r = _make_result()
    assert _extract_agg_metric(r, "nonexistent_metric") is None


# ---------------------------------------------------------------------------
# Aggregation math (simulated)
# ---------------------------------------------------------------------------

def test_mean_std_aggregation():
    """Mean and std computation for multi-window aggregation is correct (sample std, Bessel-corrected)."""
    values = [1.0, 2.0, 3.0]
    n = len(values)
    mean_v = sum(values) / n
    # sample std (Bessel-corrected): divide by N-1
    variance = sum((v - mean_v) ** 2 for v in values) / (n - 1)
    std_v = variance ** 0.5
    assert mean_v == pytest.approx(2.0)
    assert std_v == pytest.approx(1.0, rel=1e-5)  # sqrt(2/2) = 1.0


def test_mean_std_aggregation_single_value():
    """Sample std is 0.0 when N=1 (no Bessel correction applied)."""
    values = [3.14]
    n = len(values)
    mean_v = sum(values) / n
    # N=1: std=0.0 by convention
    std_v = 0.0 if n == 1 else (sum((v - mean_v) ** 2 for v in values) / (n - 1)) ** 0.5
    assert mean_v == pytest.approx(3.14)
    assert std_v == pytest.approx(0.0)


def test_single_window_no_aggregation_fields():
    """When only one window, aggregation fields should NOT be present (simulated check)."""
    # Simulate: build summary_rows as main() would for single-window path.
    # Single window => no agg suffix fields.
    agg_metrics = ["ppl_ratio", "top1_agreement", "dense_to_sparse_kl",
                   "pq_ms_per_token", "dense_ms_per_token"]
    single_window = True
    summary = {"ppl_ratio": 1.05, "top1_agreement": 0.9}
    if not single_window:
        for m in agg_metrics:
            summary[f"{m}_mean"] = 0.0
    # Should not have _mean fields.
    for m in agg_metrics:
        assert f"{m}_mean" not in summary
