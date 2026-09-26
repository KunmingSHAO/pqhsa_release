"""Exact top-k over the retrieval scores via a tail histogram.

Why this is not "another top-k kernel" (earlier fused / radix / block variants
all lost to aten ``mbtopk`` at 82 us/layer): those rewrote the *selection* itself over all N.  Here we
exploit two things the pipeline already produces for free:

  * ``row_max[h,g]`` -- the pair-LUT scan kernel already folds it in (fused stats), and
  * the measured tail geometry at 128K: the k-th score sits only ~1.07 below ``row_max``
    while the full score span is ~5.5, i.e. the top-k live in a very thin slab.

So we histogram the scores into ``BINS`` bins of a slab ``[row_max - WMAX, row_max]``
that provably covers the whole range, read off the exact bin that contains the k-th
value, compact the (few thousand) tokens above that bin's lower edge, and run aten
``topk`` on the compacted array instead of on all N.

Exactness: every token with score >= thr is kept, and thr is chosen as the largest bin
edge whose above-count is still >= k, so the true top-k set is a subset of the
candidates.  aten ``topk`` on the candidates therefore returns the exact global top-k
(up to ties, exactly like ``torch.topk`` itself).  Two failure modes are checked and
reported: underflow (fewer than k candidates -- impossible while WMAX covers the range)
and overflow (more than CAP candidates).  ``select_tail_topk`` returns None on either,
and the caller falls back to ``torch.topk``.

Switch: PQ_HSA_SCAN_TOPK=1.  Default OFF.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def is_available() -> bool:
    return triton is not None and tl is not None


def scan_topk_enabled() -> bool:
    return os.environ.get("PQ_HSA_SCAN_TOPK", "0") == "1"


# Slab width in logit units.  The measured full score span at 128K is ~5.5, so 8.0
# makes underflow structurally impossible; BINS=512 then gives 0.0156/bin, about the
# fp16 ULP at these magnitudes.
def _wmax() -> float:
    return float(os.environ.get("PQ_HSA_SCAN_TOPK_WMAX", "8.0"))


def _bins() -> int:
    return int(os.environ.get("PQ_HSA_SCAN_TOPK_BINS", "512"))


def _cap() -> int:
    return int(os.environ.get("PQ_HSA_SCAN_TOPK_CAP", "8192"))


if is_available():

    @triton.jit
    def _tail_hist_kernel(
        scores, row_max, hist,
        N, scores_row_stride, n_tiles,
        INV_DELTA: tl.constexpr,
        WMAX: tl.constexpr,
        BINS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Histogram scores into BINS bins of [row_max - WMAX, row_max], per (h,g) row.

        Persistent CTAs: each program strides over many tiles and keeps the histogram in
        registers, so the (expensive) atomic flush to global memory happens ONCE per
        program instead of once per tile.  Flushing per tile is what made the first
        version slower than aten mbtopk: ~350 occupied bins x 2048 tiles = 0.7M atomics.

        Bin 0 doubles as the underflow / out-of-range bucket and is never chosen as a
        threshold, so masked lanes can safely land there.
        """
        prog = tl.program_id(0)
        row = tl.program_id(1)
        nprog = tl.num_programs(0)
        rm = tl.load(row_max + row).to(tl.float32)
        lo = rm - WMAX
        acc = tl.zeros((BINS,), dtype=tl.int32)
        base = scores + row * scores_row_stride
        for t in range(prog, n_tiles, nprog):
            offs = t * BLOCK + tl.arange(0, BLOCK)
            m = offs < N
            s = tl.load(base + offs, mask=m, other=0.0).to(tl.float32)
            # clamp in fp32 BEFORE the int cast: masked / -inf padded slots would
            # otherwise be an undefined fptosi
            bf = tl.minimum(tl.maximum((s - lo) * INV_DELTA, 1.0), float(BINS - 1))
            b = tl.where(m, bf.to(tl.int32), 0)
            acc += tl.histogram(b, BINS)
        bidx = tl.arange(0, BINS)
        tl.atomic_add(hist + row * BINS + bidx, acc, mask=acc > 0)

    @triton.jit
    def _tail_thresh_kernel(
        hist, row_max, thr_out, cnt_out, flag_out, counter, overflow, cand_val,
        K, CAP,
        DELTA: tl.constexpr,
        WMAX: tl.constexpr,
        BINS: tl.constexpr,
        CAP_P2: tl.constexpr,
    ):
        """Pick the tightest exact threshold, and reset every buffer the compaction
        pass needs (histogram, counters, candidate values).  Folding the resets in here
        removes four separate zero_/fill_ launches from the graph body."""
        row = tl.program_id(0)
        bidx = tl.arange(0, BINS)
        h = tl.load(hist + row * BINS + bidx).to(tl.int32)
        h = tl.where(bidx == 0, 0, h)                 # bin 0 is the underflow bucket
        # suffix sum: above[b] = sum_{j >= b} h[j], non-increasing in b
        above = tl.flip(tl.cumsum(tl.flip(h), axis=0), 0)
        ok = above >= K
        # the LARGEST bin index that still has K scores at or above its lower edge is
        # the tightest threshold that provably contains the whole top-k set
        b_star = tl.max(tl.where(ok, bidx, 1))
        cnt = tl.sum(tl.where(bidx == b_star, above, 0))
        n_all = tl.sum(tl.where(bidx == 1, above, 0))
        rm = tl.load(row_max + row).to(tl.float32)
        tl.store(thr_out + row, rm - WMAX + b_star.to(tl.float32) * DELTA)
        tl.store(cnt_out + row, cnt)
        # 1 = underflow (fewer than K in the slab), 2 = overflow (> CAP candidates)
        flag = tl.where(n_all < K, 1, 0) + tl.where(cnt > CAP, 2, 0)
        tl.store(flag_out + row, flag)
        # ---- reset for the compaction pass ----
        tl.store(hist + row * BINS + bidx, tl.zeros((BINS,), dtype=tl.int32))
        tl.store(counter + row, 0)
        tl.store(overflow + row, 0)
        cidx = tl.arange(0, CAP_P2)
        tl.store(cand_val + row * CAP + cidx,
                 tl.full((CAP_P2,), -float("inf"), dtype=tl.float32),
                 mask=cidx < CAP)

    @triton.jit
    def _tail_compact_kernel(
        scores, thr, counter, cand_val, cand_idx, overflow,
        N, scores_row_stride,
        CAP,
        BLOCK: tl.constexpr,
    ):
        """Append every token with score >= thr[row] into cand_* (order irrelevant --
        the aten topk that follows re-orders them anyway)."""
        blk = tl.program_id(0)
        row = tl.program_id(1)
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        s = tl.load(scores + row * scores_row_stride + offs, mask=m,
                    other=-float("inf")).to(tl.float32)
        t = tl.load(thr + row).to(tl.float32)
        sel = m & (s >= t)
        n_sel = tl.sum(sel.to(tl.int32), axis=0)
        base = tl.atomic_add(counter + row, n_sel)
        pos = base + tl.cumsum(sel.to(tl.int32), axis=0) - 1
        w = sel & (pos < CAP)
        tl.store(cand_val + row * CAP + pos, s, mask=w)
        tl.store(cand_idx + row * CAP + pos, offs, mask=w)
        drop = tl.sum((sel & (pos >= CAP)).to(tl.int32), axis=0)
        tl.atomic_add(overflow + row, drop, mask=drop > 0)


class _Workspace:
    """Static buffers, so the whole path is CUDA-graph capturable (no allocation churn)."""

    __slots__ = ("rows", "bins", "cap", "device", "hist", "thr", "cnt", "flag",
                 "counter", "cand_val", "cand_idx", "overflow")

    def __init__(self, rows, bins, cap, device, dtype):
        self.rows, self.bins, self.cap, self.device = rows, bins, cap, device
        i32 = dict(device=device, dtype=torch.int32)
        self.hist = torch.zeros(rows, bins, **i32)
        self.thr = torch.zeros(rows, device=device, dtype=torch.float32)
        self.cnt = torch.zeros(rows, **i32)
        self.flag = torch.zeros(rows, **i32)
        self.counter = torch.zeros(rows, **i32)
        self.overflow = torch.zeros(rows, **i32)
        self.cand_val = torch.zeros(rows, cap, device=device, dtype=torch.float32)
        self.cand_idx = torch.zeros(rows, cap, **i32)


_WS: Optional[_Workspace] = None


def _ws(rows, bins, cap, device, dtype) -> _Workspace:
    global _WS
    if (_WS is None or _WS.rows != rows or _WS.bins != bins or _WS.cap != cap
            or _WS.device != device):
        _WS = _Workspace(rows, bins, cap, device, dtype)
    return _WS


def select_tail_topk(
    scores: torch.Tensor,       # [H, G, N] (fp16 or fp32), contiguous over N
    row_max: torch.Tensor,      # [H, G] float
    k: int,
    *,
    check: bool = False,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Exact top-k of ``scores`` along dim 2.  Returns ``(values, indices)`` or None.

    ``values`` is ``scores.dtype`` and ``indices`` is int64, matching ``torch.topk``.
    ``check=True`` reads the underflow/overflow flags back to the host (a sync!), so
    it must only be used outside the steady decode loop -- e.g. during graph capture
    warmup, where the caller falls back to ``torch.topk`` if the gate trips.
    """
    if not is_available() or scores.ndim != 3 or scores.stride(2) != 1:
        return None
    H, G, N = scores.shape
    if k <= 0 or k > N:
        return None
    rows = H * G
    BINS, CAP, WMAX = _bins(), _cap(), _wmax()
    if k > CAP:
        return None
    delta = WMAX / BINS
    # NOTE: hist / counter / overflow / cand_val are reset by _tail_thresh_kernel of
    # the PREVIOUS call (and zero-initialised on allocation), so the steady-state path
    # issues no separate zero_/fill_ launches.
    ws = _ws(rows, BINS, CAP, scores.device, scores.dtype)

    BLOCK = 2048
    n_tiles = triton.cdiv(N, BLOCK)
    row_stride = scores.stride(1) if G > 1 else scores.stride(0)
    nprog = int(os.environ.get("PQ_HSA_SCAN_TOPK_NPROG", "8"))
    cap_p2 = 1 << (CAP - 1).bit_length()
    _tail_hist_kernel[(nprog, rows)](
        scores, row_max, ws.hist, N, row_stride, n_tiles,
        INV_DELTA=1.0 / delta, WMAX=WMAX, BINS=BINS, BLOCK=BLOCK,
        num_warps=8, num_stages=2,
    )
    _tail_thresh_kernel[(rows,)](
        ws.hist, row_max, ws.thr, ws.cnt, ws.flag, ws.counter, ws.overflow, ws.cand_val,
        k, CAP, DELTA=delta, WMAX=WMAX, BINS=BINS, CAP_P2=cap_p2, num_warps=8,
    )
    _tail_compact_kernel[(n_tiles, rows)](
        scores, ws.thr, ws.counter, ws.cand_val, ws.cand_idx, ws.overflow,
        N, row_stride, CAP, BLOCK=BLOCK, num_warps=8, num_stages=2,
    )
    if check:
        if int(ws.flag.max().item()) != 0 or int(ws.overflow.max().item()) != 0:
            return None
    cv = ws.cand_val.view(H, G, CAP)
    val, pos = torch.topk(cv, k=k, dim=2, sorted=False)
    idx = torch.gather(ws.cand_idx.view(H, G, CAP), 2, pos).to(torch.int64)
    return val.to(scores.dtype), idx


def last_stats() -> dict:
    """Debug/validation snapshot of the last call (host sync)."""
    if _WS is None:
        return {}
    return {
        "cand_count_max": int(_WS.cnt.max().item()),
        "cand_count_min": int(_WS.cnt.min().item()),
        "written_max": int(_WS.counter.max().item()),
        "flag_max": int(_WS.flag.max().item()),
        "overflow_max": int(_WS.overflow.max().item()),
        "cap": _WS.cap,
        "bins": _WS.bins,
    }
