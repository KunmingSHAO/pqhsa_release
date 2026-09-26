"""Sync-free CUDA-event phase log for decode breakdown.

Two APIs:

* ``mark(name)`` — consecutive marks. Elapsed GPU time between mark i and
  mark i+1 is attributed to name i. Use only around a tight GPU sequence
  (the batched-heads path) where the next mark is recorded immediately.
* ``span_start(name)`` / ``span_end(name)`` — paired events around one
  module. Gaps between spans are NOT attributed (they become python/launch
  residual against wall-clock).

Disabled unless ``enable()`` is called. Timing only; no numerical changes.
"""
from __future__ import annotations

from collections import defaultdict

import torch

_enabled = False
_names: list[str] = []
_events: list[torch.cuda.Event] = []
_open: dict[str, torch.cuda.Event] = {}
_spans: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)


def enable() -> None:
    global _enabled
    _enabled = True
    reset()


def disable() -> None:
    global _enabled
    _enabled = False


def reset() -> None:
    _names.clear()
    _events.clear()
    _open.clear()
    _spans.clear()


def is_enabled() -> bool:
    return _enabled


def mark(name: str) -> None:
    if not _enabled:
        return
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    _names.append(name)
    _events.append(ev)


def span_start(name: str) -> None:
    if not _enabled:
        return
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    _open[name] = ev


def span_end(name: str) -> None:
    if not _enabled:
        return
    start = _open.pop(name, None)
    if start is None:
        return
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    _spans[name].append((start, ev))


def summarize() -> dict[str, float]:
    """Return {phase: microseconds} from consecutive marks plus closed spans."""
    if _events or _spans:
        torch.cuda.synchronize()
    acc: dict[str, float] = defaultdict(float)
    if len(_events) >= 2:
        for i in range(len(_names) - 1):
            acc[_names[i]] += float(_events[i].elapsed_time(_events[i + 1])) * 1000.0
    for name, pairs in _spans.items():
        for start, end in pairs:
            acc[name] += float(start.elapsed_time(end)) * 1000.0
    return dict(acc)


def mark_count() -> int:
    return len(_names) + sum(len(v) for v in _spans.values())
