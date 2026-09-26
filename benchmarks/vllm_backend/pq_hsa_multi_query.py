"""Multi-query (speculative decoding / MTP) PQ-HSA decode for the vLLM wrap.

**Everything here is dead code unless ``PQ_HSA_MULTI_QUERY=1`` (default 0).**
``pq_hsa/`` is not touched; ``pq_hsa_decode_runtime`` (rt) keeps its
single-query path byte-for-byte and only gains two opt-in branches:

  * ``rt.should_pq_decode``: when ``max_query_len != 1`` and the flag is on,
    ask ``should_pq_decode_mq`` instead of rejecting;
  * ``rt.run_pq_decode_into_output``: when the flag is on (and the batch path
    is not claiming the step), route to ``run_pq_decode_mq``.

Problem
-------
With speculative decoding (vLLM V1 ngram / EAGLE / an MTP head) one decode
step of a single request carries ``T = 1 + k`` query tokens at consecutive
positions ``C .. C+k`` (``C = seq_len - T`` = committed context). Two things
differ from the steady one-token path:

  1. query ``j`` must attend causally to the prefix *and* to in-step tokens
     ``C .. C+j`` (including itself);
  2. after verification the scheduler keeps only ``1 + a`` of the ``T`` tokens
     (``a`` accepted drafts + the bonus token); the rejected tail
     ``C+1+a .. C+k`` must disappear from the sidecar before the next step.

Design (correctness first; no intra-step fusion)
------------------------------------------------
* **Causality by sequential replay.** For ``j in range(T)`` we do exactly what
  the one-token path does: ``append(K_j, V_j)`` then ``forward(q_j)``. Query j
  therefore sees the prefix plus tokens ``C..C+j`` with the *same* sink/local/
  retrieval semantics as steady decode. ``T`` graph replays per layer per step
  (the captured graph is single-query; we replay it ``T`` times instead of
  capturing per-``T`` buckets -- see "CUDA graph" below).
* **Rollback by prefix checkpoint.** Before the first in-step append we call
  ``checkpoint_prefix()`` (state at length ``C``) and remember
  ``mq_ck_len = C``, ``mq_uncommitted = T``. At the *next* step we read the new
  ``C'``: ``C' == C + T`` means everything was accepted (drop the checkpoint);
  ``C <= C' < C + T`` means ``C' - C`` tokens were kept -> ``restore_prefix``
  back to ``C`` (frozen body, or the ring-aware variant when
  ``PQ_HSA_PAGED_ATTEND=1`` + ``PQ_HSA_PAGED_RESTORE=1``) and re-``append`` the
  kept rows ``C .. C'-1`` straight from vLLM's paged KV
  (``gather_flashattn_kv_range``; those rows were written by
  ``reshape_and_cache_flash`` last step and are still valid). ``restore_prefix``
  reverts ``_length``/pending counters/index refs; appends never touch the
  index tensors unless a flush runs, so **no flush is allowed between the
  checkpoint and its resolution** -- flush accounting is deferred to
  resolution time and only counts accepted tokens.
* **k = 0 stays byte-identical.** A ``T == 1`` step delegates to
  ``rt.run_pq_decode`` (after resolving any outstanding checkpoint), so with the
  flag on but no speculation the numbers are the default path.
* **Prefill chunks are not speculative steps.** Both have ``max_query_len > 1``
  and ``num_actual_tokens == max_query_len``; we tell them apart through the
  runner hook: a chunk whose ``context_len < num_prompt_tokens`` is prefill and
  is rejected (dense serves it, exactly as today).

CUDA graph
----------
The per-layer sidecar graph is captured for one query. Variable ``T`` is
handled by replaying that graph ``T`` times after ``T`` sequential appends
(each append refreshes the graph's static inputs exactly as in steady decode).
Per-``T`` bucketed captures would save the ``T-1`` extra launches of the
non-attention prologue but need ``T`` sets of static buffers; not done here.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch

from benchmarks.vllm_backend import pq_hsa_decode_runtime as rt
from benchmarks.vllm_backend.paged_kv_fa import flashattn_seq_len, gather_flashattn_kv_range

STATS = rt.STATS

_MQ_MAX: int | None = None


def mq_max() -> int:
    global _MQ_MAX
    if _MQ_MAX is None:
        _MQ_MAX = int(os.environ.get("PQ_HSA_MQ_MAX", "8"))
    return _MQ_MAX


def _bump(key: str, n: int = 1) -> None:
    STATS[key] = int(STATS.get(key, 0)) + n


def _paged_ring_active() -> bool:
    return os.environ.get("PQ_HSA_PAGED_ATTEND", "0") == "1"


def _paged_restore_on() -> bool:
    return os.environ.get("PQ_HSA_PAGED_RESTORE", "0") == "1"


# --------------------------------------------------------------------------- #
# admission
# --------------------------------------------------------------------------- #
def _request0(attn_metadata):
    runner = getattr(attn_metadata, "_pq_runner", None)
    if runner is None:
        return None
    req_ids = getattr(getattr(runner, "input_batch", None), "req_ids", None)
    if not req_ids or req_ids[0] is None:
        return None
    return runner.requests.get(req_ids[0])


def should_pq_decode_mq(attn_metadata) -> bool:
    """Admission for a ``max_query_len > 1`` step. Caller already handled
    capture / no-metadata / cascade."""
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is None or int(seq_lens.shape[0]) != 1:
        return rt.reject("mq_num_reqs!=1")
    T = int(getattr(attn_metadata, "max_query_len", 0))
    if T < 1 or T > mq_max():
        return rt.reject("mq_query_len_out_of_range")
    if int(getattr(attn_metadata, "num_actual_tokens", 0)) != T:
        return rt.reject("mq_padded_or_mixed")
    seq_len = flashattn_seq_len(attn_metadata, 0)
    context_len = seq_len - T
    req = _request0(attn_metadata)
    if req is None:
        return rt.reject("mq_no_runner")
    if context_len < int(getattr(req, "num_prompt_tokens", 1 << 62)):
        return rt.reject("mq_prefill_chunk")
    if context_len < rt.min_pq_seq():
        return rt.reject("seq_too_short")
    if _paged_ring_active() and not _paged_restore_on():
        # ring-mode sidecar cannot checkpoint without the ring-aware restore
        return rt.reject("mq_needs_paged_restore")
    return True


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _row(t: torch.Tensor, j: int) -> torch.Tensor:
    """Token ``j`` of a ``[T,H,D]`` tensor as the ``[1,H,D]`` view the sidecar wants."""
    if t.ndim == 3:
        return t[j : j + 1]
    if t.ndim == 2:
        return t[j : j + 1].unsqueeze(0)
    raise ValueError(f"expected [T,H,D] or [T,D], got {tuple(t.shape)}")


def _paged_rows(kv_cache, attn_metadata, start: int, end: int):
    """Rows ``[start, end)`` of request 0 from vLLM's paged KV as a list of
    ``([1,H_kv,D], [1,H_kv,D])`` pairs (sidecar append layout)."""
    k, v = gather_flashattn_kv_range(kv_cache, attn_metadata, 0, start, end)  # [H_kv, n, D]
    out = []
    for i in range(k.shape[1]):
        out.append((k[:, i : i + 1, :].permute(1, 0, 2).contiguous(),
                    v[:, i : i + 1, :].permute(1, 0, 2).contiguous()))
    return out


def _append_rows(adapter, rows) -> None:
    for k1, v1 in rows:
        adapter.append(k1, v1)


def _restore_to_checkpoint(adapter, kv_cache, attn_metadata, ck_len: int) -> None:
    """Roll the sidecar back to its checkpoint at length ``ck_len``."""
    k1, v1 = _paged_rows(kv_cache, attn_metadata, ck_len - 1, ck_len)[0]
    inner = getattr(adapter, "adapter", adapter)
    fn = adapter.restore_prefix
    if (
        _paged_restore_on()
        and getattr(inner, "_shared_base_ring", False)
        and hasattr(inner, "restore_prefix_ring")
    ):
        fn = inner.restore_prefix_ring  # Ring-aware restore (frozen restore_prefix inside)
    ok = fn(k1, v1, kv_cache=kv_cache, attn_metadata=attn_metadata, seq_idx=0)
    if not ok:
        raise RuntimeError("speculative rollback failed (checkpoint missing)")
    if fn is not adapter.restore_prefix and hasattr(adapter, "length"):
        cl = getattr(adapter, "prefix_checkpoint_length", None)
        if cl is not None:
            adapter.length = int(cl)


def _account_appends_and_flush(st: dict[str, Any], adapter, n_accepted: int) -> None:
    """Deferred flush accounting for ``n_accepted`` committed tokens (mirrors the
    ``appends % interval == 0`` cadence of the one-token path, evaluated once)."""
    if n_accepted <= 0:
        return
    interval = rt.flush_interval()
    before = int(st["appends"])
    after = before + n_accepted
    st["appends"] = after
    if interval > 0 and (after // interval) > (before // interval):
        t2 = time.perf_counter()
        rt.flush_sidecar(adapter)
        STATS["pq_flush_s_total"] += time.perf_counter() - t2
        STATS["pq_flush_calls"] += 1
        _bump("mq_flush_calls")


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def resolve_outstanding(st: dict[str, Any], kv_cache, attn_metadata, context_len: int) -> None:
    """Commit or roll back the speculative appends of the previous step."""
    adapter = st.get("sidecar")
    unc = int(st.get("mq_uncommitted", 0))
    if adapter is None or unc <= 0:
        return
    ck = int(st["mq_ck_len"])
    if context_len == ck + unc:
        accepted = unc
        _bump("mq_steps_all_accepted")
    elif ck <= context_len < ck + unc:
        accepted = context_len - ck
        t0 = time.perf_counter()
        _restore_to_checkpoint(adapter, kv_cache, attn_metadata, ck)
        rows = _paged_rows(kv_cache, attn_metadata, ck, context_len)
        _append_rows(adapter, rows)
        STATS["mq_rollback_s_total"] = float(STATS.get("mq_rollback_s_total", 0.0)) + (
            time.perf_counter() - t0
        )
        _bump("mq_steps_rolled_back")
        _bump("mq_tokens_rejected", unc - accepted)
    else:
        # context moved outside [ck, ck+unc): not a speculative continuation.
        # Drop the sidecar so the caller rebuilds from the paged KV (exact).
        _bump("mq_unexpected_context")
        st["sidecar"] = None
        st["seq_len"] = None
        st["mq_uncommitted"] = 0
        return
    st["seq_len"] = context_len
    st["mq_uncommitted"] = 0
    _bump("mq_tokens_accepted", accepted)
    _account_appends_and_flush(st, adapter, accepted)


def _rebuild(impl, st: dict[str, Any], kv_cache, attn_metadata, context_len: int):
    """Build a fresh sidecar over the committed prefix ``[0, context_len)``."""
    t_g = time.perf_counter()
    k, v = gather_flashattn_kv_range(kv_cache, attn_metadata, 0, 0, context_len)  # [H_kv, C, D]
    keys = k.unsqueeze(0).contiguous()
    values = v.unsqueeze(0).contiguous()
    STATS["pq_gather_s_total"] += time.perf_counter() - t_g
    t0 = time.perf_counter()
    adapter = rt.sidecar_cls()(
        rt.paper_index_config(),
        rt.paper_attention_config(impl.scale),
        num_query_heads=impl.num_heads,
        num_kv_heads=impl.num_kv_heads,
    ).build_cache(keys, values)
    STATS["pq_build_s_total"] += time.perf_counter() - t0
    STATS["pq_build_calls"] += 1
    STATS["layers_built"] = int(STATS["layers_built"]) + 1
    st["sidecar"] = adapter
    st["prefix_key"] = rt.prefix_cache_key(attn_metadata, context_len)
    st["seq_len"] = context_len
    st["appends"] = 0
    st["mq_uncommitted"] = 0
    if hasattr(adapter, "set_paged_kv_context"):
        adapter.set_paged_kv_context(kv_cache, attn_metadata, 0)
    _bump("mq_rebuilds")
    return adapter


def run_pq_decode_mq(impl, layer, query, key, value, kv_cache, attn_metadata, output) -> torch.Tensor:
    """Single request, ``T = max_query_len`` consecutive query tokens."""
    st = rt.state_for(impl, layer)
    T = int(attn_metadata.max_query_len)
    seq_len = flashattn_seq_len(attn_metadata, 0)
    context_len = seq_len - T

    # 1. resolve the previous speculative step (commit / rollback)
    resolve_outstanding(st, kv_cache, attn_metadata, context_len)
    adapter = st.get("sidecar")

    # 2. make the sidecar cover exactly [0, context_len)
    if adapter is not None and st.get("seq_len") is not None:
        cur = int(st["seq_len"])
        if cur < context_len and (context_len - cur) <= mq_max():
            # accepted tokens the one-token path never saw (e.g. first step after
            # a speculative step handled elsewhere): catch up from paged KV
            _append_rows(adapter, _paged_rows(kv_cache, attn_metadata, cur, context_len))
            _account_appends_and_flush(st, adapter, context_len - cur)
            st["seq_len"] = context_len
            _bump("mq_catchup_tokens", context_len - cur)
        elif cur != context_len:
            adapter = None
    if adapter is None:
        adapter = _rebuild(impl, st, kv_cache, attn_metadata, context_len)
    if (_paged_ring_active() or os.environ.get("PQ_HSA_PAGED_KV", "0") == "1") and hasattr(
        adapter, "set_paged_kv_context"
    ):
        adapter.set_paged_kv_context(kv_cache, attn_metadata, 0)

    # 3. k = 0: default one-token path, byte-identical
    if T == 1:
        ctx = rt.run_pq_decode(impl, layer, query, key, value, kv_cache, attn_metadata)
        rt._span_start("copy_ctx")
        rt.copy_ctx(output, ctx, 1)
        rt._span_end("copy_ctx")
        return output

    # 4. speculative step: checkpoint, then sequential append + replay
    adapter.checkpoint_prefix()
    st["mq_ck_len"] = context_len
    t_f = time.perf_counter()
    for j in range(T):
        rt._span_start("append")
        adapter.append(_row(key, j), _row(value, j))
        rt._span_end("append")
        rt._span_start("replay")
        ctx = adapter.forward(_row(query, j))
        rt._span_end("replay")
        dst = output[j : j + 1]
        if ctx.shape != dst.shape:
            ctx = ctx.reshape(dst.shape)
        if ctx.dtype != dst.dtype:
            ctx = ctx.to(dtype=dst.dtype)
        rt._span_start("copy_ctx")
        dst.copy_(ctx)
        rt._span_end("copy_ctx")
    STATS["pq_forward_s_total"] += time.perf_counter() - t_f
    st["seq_len"] = context_len + T
    st["mq_uncommitted"] = T
    STATS["pq_decode_calls"] += 1
    _bump("mq_spec_steps")
    _bump("mq_spec_queries", T)
    hist = STATS.setdefault("mq_query_len_hist", {})
    hist[str(T)] = int(hist.get(str(T), 0)) + 1
    STATS["cuda_graph_active"] = getattr(adapter, "cuda_graph_active", None)
    return output
