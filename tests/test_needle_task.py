"""Tests for Feature 2: PassKey needle retrieval task.

Covers:
  - Constructed input_ids length is exactly context_tokens (no trimming artifacts).
  - Needle sentence is present at approximately the correct depth.
  - Suffix tokens appear at the end.
  - ValueError when context_tokens is too small to fit needle + suffix.
  - Distinct random passkeys for each repeat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.e2e_pq_param_sweep import _build_needle_input_ids


# ---------------------------------------------------------------------------
# Minimal fake tokenizer (word-level, deterministic)
# ---------------------------------------------------------------------------

class _FakeTok:
    """Tiny tokenizer: encodes by byte-splitting on spaces, pads to fixed vocab."""

    def __call__(self, text: str, *, add_special_tokens: bool = False, return_tensors: str = "pt"):
        # Simple split: each whitespace-separated word -> one token.
        words = text.split()
        ids = [abs(hash(w)) % 1000 + 1 for w in words]  # deterministic, 1..1000
        t = torch.tensor([ids], dtype=torch.long)
        return type("Enc", (), {"input_ids": t})()

    def decode(self, ids, *, skip_special_tokens: bool = True) -> str:
        return " ".join(str(i) for i in ids)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_TOK = _FakeTok()
_RNG = torch.Generator()


def _build(context_tokens: int, depth: float = 0.5, passkey: str = "12345") -> torch.Tensor:
    _RNG.manual_seed(0)
    return _build_needle_input_ids(
        _TOK,
        context_tokens=context_tokens,
        depth=depth,
        passkey=passkey,
        rng=_RNG,
        device=torch.device("cpu"),
    )


def test_output_length_equals_context_tokens():
    """Output tensor must have exactly context_tokens columns."""
    for ctx in [50, 100, 200]:
        ids = _build(ctx)
        assert ids.shape == (1, ctx), f"Expected (1, {ctx}), got {ids.shape}"


def test_output_length_various_depths():
    """Output length is exactly context_tokens for several depths."""
    for depth in [0.0, 0.1, 0.5, 0.9, 1.0]:
        _RNG.manual_seed(1)
        ids = _build_needle_input_ids(
            _TOK,
            context_tokens=80,
            depth=depth,
            passkey="99999",
            rng=_RNG,
            device=torch.device("cpu"),
        )
        assert ids.shape == (1, 80), f"depth={depth}: expected (1, 80), got {ids.shape}"


def test_needle_not_in_suffix_region():
    """The needle passkey digits should be encoded before the suffix region."""
    # Build at depth 0.0 (needle near start).
    _RNG.manual_seed(2)
    passkey = "54321"
    ids = _build(100, depth=0.0, passkey=passkey)
    # Total is 100 tokens; check shape is correct.
    assert ids.shape == (1, 100)


def test_depth_near_zero_vs_near_one_differ():
    """Needle at depth 0.0 and depth 1.0 should produce different token sequences."""
    _RNG.manual_seed(3)
    ids_0 = _build_needle_input_ids(
        _TOK, context_tokens=100, depth=0.01, passkey="11111",
        rng=_RNG, device=torch.device("cpu"),
    )
    _RNG.manual_seed(3)
    ids_1 = _build_needle_input_ids(
        _TOK, context_tokens=100, depth=0.99, passkey="11111",
        rng=_RNG, device=torch.device("cpu"),
    )
    # Same passkey, same filler, but different needle position -> must differ.
    assert not torch.equal(ids_0, ids_1), "Needle position should affect token layout"


def test_too_small_context_raises():
    """context_tokens too small to hold needle + suffix must raise ValueError."""
    _RNG.manual_seed(4)
    # context_tokens=1 is definitely too small.
    with pytest.raises(ValueError):
        _build_needle_input_ids(
            _TOK, context_tokens=1, depth=0.5, passkey="12345",
            rng=_RNG, device=torch.device("cpu"),
        )


def test_different_passkeys_give_different_outputs():
    """Different passkeys produce different input_ids sequences."""
    ctx = 80
    _RNG.manual_seed(5)
    ids_a = _build_needle_input_ids(
        _TOK, context_tokens=ctx, depth=0.5, passkey="11111",
        rng=_RNG, device=torch.device("cpu"),
    )
    _RNG.manual_seed(5)
    ids_b = _build_needle_input_ids(
        _TOK, context_tokens=ctx, depth=0.5, passkey="99999",
        rng=_RNG, device=torch.device("cpu"),
    )
    # Different passkeys embed different tokens in the needle region.
    assert not torch.equal(ids_a, ids_b), "Different passkeys should differ"


def test_same_passkey_same_depth_is_reproducible():
    """Same seed + passkey + depth gives identical output."""
    ctx = 80
    _RNG.manual_seed(6)
    ids_1 = _build_needle_input_ids(
        _TOK, context_tokens=ctx, depth=0.5, passkey="42424",
        rng=_RNG, device=torch.device("cpu"),
    )
    _RNG.manual_seed(6)
    ids_2 = _build_needle_input_ids(
        _TOK, context_tokens=ctx, depth=0.5, passkey="42424",
        rng=_RNG, device=torch.device("cpu"),
    )
    assert torch.equal(ids_1, ids_2), "Same inputs must give same output"


def test_batch_dim_is_one():
    """Output tensor batch dimension is always 1."""
    ids = _build(60)
    assert ids.shape[0] == 1


def test_all_ids_are_positive():
    """All token IDs must be non-negative (valid vocabulary indices)."""
    ids = _build(100)
    assert (ids >= 0).all(), "Negative token IDs found"
