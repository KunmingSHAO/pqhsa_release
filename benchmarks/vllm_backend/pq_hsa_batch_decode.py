"""Naive multi-request (batch>1) PQ-HSA decode for the vLLM integration.

**Everything in this module is dead code unless ``PQ_HSA_BATCH=1`` (default 0).**
``pq_hsa/`` is not touched, ``pq_hsa_decode_runtime`` keeps its single-request
path byte-for-byte, and the only hooks into it are

  * ``should_pq_decode``  -> ``should_pq_decode_batch`` when the flag is on,
  * ``run_pq_decode_into_output`` -> ``run_pq_decode_batch`` after the (already
    batch-correct) ``reshape_and_cache_flash``.

Design (deliberately naive; asks for correctness, not throughput):

  * **one sidecar index instance per (layer, request)**.  ``_LAYER_STATE[layer]``
    grows a ``"reqs"`` dict keyed by the vLLM request id
    (``runner.input_batch.req_ids[i]``), each value having exactly the same
    ``{"sidecar", "seq_len", "prefix_key", "appends"}`` shape the single-request
    path keeps in the layer record.  Keying on the request id (not the batch
    slot) survives ``InputBatch.condense`` reordering when a request finishes.
  * **one graph replay per request per layer**, in a Python ``for i in
    range(num_reqs)`` loop.  No intra-batch fusion: each request replays its own
    captured graph against its own static buffers.
  * **``copy_ctx`` writes back per request**: ``output[i:i+1].copy_(ctx_i)``,
    immediately after that request's ``forward`` (the adapter may hand back an
    alias of its graph's static output buffer).
  * **host-side ``seq_lens``**: read once per step from ``runner.seq_lens_np``
    (the numpy buffer vLLM fills before building the metadata), so the
    per-request ``seq_lens[i].item()`` device syncs never happen on the hot path.

Prefill / decode mixed batches
------------------------------
A vLLM V1 step can carry chunked-prefill tokens and decode tokens together.  The
sidecar only knows how to advance one token at a time, so the whole step is
rejected (``reject("mixed_prefill_decode")``) and vLLM's dense FlashAttention
serves every request in it.  That is exact, and it is self-healing: the skipped
token still lands in the paged KV cache via ``reshape_and_cache_flash``, so on
the next PQ step the affected request sees ``seq_len != prev + 1``, fails the
``sequential`` test, and is rebuilt from ``gather_flashattn_kv`` (or restored
from its prefix checkpoint).  Same for a request that is shorter than
``min_pq_seq()``: the step is rejected as a whole rather than mixing a PQ result
and a dense result into one output buffer.

 instrumentation
---------------------
``rt._span_start`` / ``rt._span_end`` pairs for ``append`` / ``replay`` /
``copy_ctx`` mirror the ones the single-request path in
``pq_hsa_decode_runtime.run_pq_decode`` already carries, so the attention
segment caliber (``replay_gpu + append + copy_ctx``, CUDA Event) is defined in
batch mode too.  They are timing only and are *no-ops* unless
``PQ_HSA_STEADY_EVENTS=1`` (``rt._events_on()``); ``replay_gpu`` itself is
emitted from inside the sidecar adapter and already worked.  In batch mode the
loop records one span per request per layer per step, so the sums fold as
``ms/step = Sigma / n_steps`` and ``ms/tok = ms/step / num_reqs``.
"""

from __future__ import annotations

from array import array
import hashlib
import os
import time
from typing import Any

import torch

from benchmarks.vllm_backend import pq_hsa_decode_runtime as rt
from benchmarks.vllm_backend.paged_kv_fa import gather_flashattn_kv

STATS = rt.STATS


# --------------------------------------------------------------------------- #
# metadata helpers
# --------------------------------------------------------------------------- #
def batch_seq_lens(attn_metadata) -> list[int]:
    """Per-request seq_len for this step, host-side, cached on the metadata.

    ``attn_metadata.seq_lens`` lives on the GPU; ``[i].item()`` would sync once
    per request per layer.  ``runner.seq_lens_np[:num_reqs]`` is the same numbers
    (``gpu_model_runner._prepare_inputs`` fills it right before building the
    metadata) for free.  Falls back to a single ``.tolist()`` sync if the runner
    hook is not installed.
    """
    cached = getattr(attn_metadata, "_pq_batch_seq_lens", None)
    if cached is not None:
        return cached
    n = int(attn_metadata.seq_lens.shape[0])
    lens: list[int] | None = None
    runner = getattr(attn_metadata, "_pq_runner", None)
    if runner is not None:
        np_lens = getattr(runner, "seq_lens_np", None)
        if np_lens is not None and len(np_lens) >= n:
            lens = [int(x) for x in np_lens[:n]]
    if lens is None:
        lens = [int(x) for x in attn_metadata.seq_lens[:n].tolist()]
    attn_metadata._pq_batch_seq_lens = lens
    return lens


def batch_req_keys(attn_metadata, num_reqs: int) -> list[Any]:
    """Per-slot state key.  vLLM request id when available, else the slot index.

    The slot index alone is wrong once a request finishes and ``condense`` moves
    the tail request into the freed slot, so ``should_pq_decode_batch`` refuses
    num_reqs>1 when the request ids are not reachable.
    """
    cached = getattr(attn_metadata, "_pq_batch_req_keys", None)
    if cached is not None:
        return cached
    keys: list[Any] = [("slot", i) for i in range(num_reqs)]
    runner = getattr(attn_metadata, "_pq_runner", None)
    if runner is not None:
        req_ids = getattr(getattr(runner, "input_batch", None), "req_ids", None)
        if req_ids is not None and len(req_ids) >= num_reqs:
            for i in range(num_reqs):
                rid = req_ids[i]
                if rid is not None:
                    keys[i] = rid
    attn_metadata._pq_batch_req_keys = keys
    return keys


def _req_ids_available(attn_metadata, num_reqs: int) -> bool:
    runner = getattr(attn_metadata, "_pq_runner", None)
    if runner is None:
        return False
    req_ids = getattr(getattr(runner, "input_batch", None), "req_ids", None)
    if req_ids is None or len(req_ids) < num_reqs:
        return False
    return all(req_ids[i] is not None for i in range(num_reqs))


def prefix_cache_key_at(attn_metadata, seq_len: int, idx: int) -> tuple[int, bytes] | None:
    """Per-request version of ``rt.prefix_cache_key`` (which hardcodes req 0)."""
    cache = getattr(attn_metadata, "_pq_batch_prefix_keys", None)
    if cache is None:
        cache = {}
        attn_metadata._pq_batch_prefix_keys = cache
    if idx in cache:
        return cache[idx]
    key = None
    runner = getattr(attn_metadata, "_pq_runner", None)
    if runner is not None:
        req_ids = getattr(getattr(runner, "input_batch", None), "req_ids", None)
        if req_ids is not None and len(req_ids) > idx and req_ids[idx] is not None:
            context_len = int(seq_len) - int(attn_metadata.max_query_len)
            request = runner.requests.get(req_ids[idx])
            prompt_ids = None if request is None else request.prompt_token_ids
            if prompt_ids is not None:
                prompt_bytes = array("I", prompt_ids).tobytes()
                digest = hashlib.blake2b(prompt_bytes, digest_size=16).digest()
                key = (context_len, digest)
    cache[idx] = key
    return key


def should_pq_decode_batch(attn_metadata) -> bool:
    """``PQ_HSA_BATCH=1`` admission test.  Caller already checked max_query_len==1."""
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is None:
        return rt.reject("no_seq_lens")
    num_reqs = int(seq_lens.shape[0])
    if num_reqs < 1:
        return rt.reject("num_reqs<1")
    # Pure-decode step only: one scheduled token per running request.  Anything
    # else is a chunked-prefill / mixed step -> dense (see module docstring).
    if int(getattr(attn_metadata, "num_actual_tokens", 0)) != num_reqs:
        return rt.reject("mixed_prefill_decode")
    if num_reqs > 1 and not _req_ids_available(attn_metadata, num_reqs):
        return rt.reject("no_req_ids")
    lens = batch_seq_lens(attn_metadata)
    floor = rt.min_pq_seq()
    for length in lens:
        if length < floor:
            return rt.reject("seq_too_short")
    return True


# --------------------------------------------------------------------------- #
# per-request tensor slicing / write-back
# --------------------------------------------------------------------------- #
def row(t: torch.Tensor, idx: int) -> torch.Tensor:
    """Request ``idx``'s single decode token, same layout ``as_decode_qkv`` gives.

    For a contiguous ``[T, H, D]`` this is a contiguous ``[1, H, D]`` view whose
    ``data_ptr`` is row ``idx`` -- which is what ``lean_append`` /
    ``pq_append_prep`` require (they read ``H*D`` elements straight off
    ``data_ptr``).  At ``idx == 0`` it is the identical view ``t[:1]`` returns.
    """
    if t.ndim == 3:
        return t[idx : idx + 1]
    if t.ndim == 2:
        return t[idx : idx + 1].unsqueeze(0)
    raise ValueError(f"expected [T,H,D] or [T,D], got {tuple(t.shape)}")


def copy_ctx_at(output: torch.Tensor, ctx: torch.Tensor, idx: int) -> None:
    """Write one request's context into vLLM's output buffer.

    Deliberately keeps the shape/dtype branches ``rt.copy_ctx`` has behind
    ``PQ_HSA_LEAN_COPYCTX``: the lean variant assumes the whole buffer is one
    request, which is exactly what stops being true here.
    """
    dst = output[idx : idx + 1]
    src = ctx
    if src.shape != dst.shape:
        src = src.reshape(dst.shape)
    if src.dtype != dst.dtype:
        src = src.to(dtype=dst.dtype)
    dst.copy_(src)


def new_req_state() -> dict[str, Any]:
    return {"sidecar": None, "seq_len": None, "prefix_key": None, "appends": 0}


# --------------------------------------------------------------------------- #
# one request
# --------------------------------------------------------------------------- #
def run_pq_decode_one(
    impl,
    layer,
    st: dict[str, Any],
    seq_idx: int,
    seq_len: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
) -> torch.Tensor:
    """``rt.run_pq_decode`` for request ``seq_idx`` of a decode batch.

    Mirrors ``rt.run_pq_decode`` operation for operation; the only differences
    are (a) the state record is passed in instead of looked up per layer,
    (b) ``seq_len`` comes from the host-side batch vector instead of
    ``flashattn_seq_len(md, 0)``, and (c) every ``t[:1]`` becomes
    ``row(t, seq_idx)``.  At ``seq_idx == 0`` with one request the two are the
    same tensors and the same kernel launches (checked with ``torch.equal``).
    """
    prev = st["seq_len"]
    adapter = st["sidecar"]
    sequential = adapter is not None and prev is not None and seq_len == int(prev) + 1
    q = row(query, seq_idx)
    if (
        (
            os.environ.get("PQ_HSA_PAGED_KV", "0") == "1"
            # Mirror rt.run_pq_decode's per-step paged-KV-context
            # refresh (pq_hsa_decode_runtime.py:408-431). Without this call
            # every batch-mode adapter's self._paged_kv_cache stays None
            # forever (set_paged_kv_context was never invoked anywhere on
            # this code path before), so under PQ_HSA_PAGED_ATTEND=1
            # the ring-buffer shrink in e2e_pq_param_sweep.py silently
            # degrades every "paged_kv_cache is not None" gate to False and
            # falls back to the legacy absolute-position reads of the now-tiny
            # _shared_base_keys/_vals ring buffer -- an out-of-bounds CUDA
            # gather (IndexKernel.cu device assert) the first time a real
            # (large) token index is used.
            or os.environ.get("PQ_HSA_PAGED_ATTEND", "0") == "1"
        )
        and adapter is not None
        and hasattr(adapter, "set_paged_kv_context")
    ):
        adapter.set_paged_kv_context(kv_cache, attn_metadata, seq_idx)
    if not sequential:
        pkey = prefix_cache_key_at(attn_metadata, seq_len, seq_idx)
        checkpoint_len = getattr(adapter, "prefix_checkpoint_length", None)
        can_restore = (
            adapter is not None
            and pkey is not None
            and pkey == st["prefix_key"]
            and checkpoint_len is not None
            and int(checkpoint_len) == seq_len
        )
        if can_restore:
            t_restore = time.perf_counter()
            # Mirror rt.run_pq_decode's restore_prefix call (see this
            # function's docstring) -- pass kv_cache/attn_metadata so the
            # stale tail-block refresh applies here too.
            restored = adapter.restore_prefix(
                row(key, seq_idx), row(value, seq_idx),
                kv_cache=kv_cache, attn_metadata=attn_metadata, seq_idx=seq_idx,
            )
            restore_s = time.perf_counter() - t_restore
            if not restored:
                raise RuntimeError("prefix sidecar checkpoint disappeared during restore")
            STATS["pq_restore_s_total"] += restore_s
            STATS["prefix_persist_hits"] += 1
            st["seq_len"] = seq_len
            st["appends"] = 0
            t_forward = time.perf_counter()
            ctx = adapter.forward(q)
            forward_s = time.perf_counter() - t_forward
            STATS["pq_forward_only_s_total"] += forward_s
            STATS["pq_forward_s_total"] += restore_s + forward_s
            STATS["cuda_graph_active"] = getattr(adapter, "cuda_graph_active", None)
            STATS["pq_decode_calls"] += 1
            return ctx

        STATS["prefix_persist_misses"] += 1
        if adapter is None:
            miss_reason = "no_adapter"
        elif pkey is None:
            miss_reason = "no_prefix_key"
        elif pkey != st["prefix_key"]:
            miss_reason = "prefix_key_mismatch"
        else:
            miss_reason = "checkpoint_length_mismatch"
        reasons = STATS["prefix_miss_reasons"]
        reasons[miss_reason] = reasons.get(miss_reason, 0) + 1
        t_g = time.perf_counter()
        keys, values, gathered = gather_flashattn_kv(kv_cache, attn_metadata, seq_idx)
        STATS["pq_gather_s_total"] += time.perf_counter() - t_g
        if gathered != seq_len:
            raise RuntimeError(f"gather seq {gathered} != metadata seq {seq_len} (req {seq_idx})")
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
        STATS["batched_heads_active"] = getattr(adapter, "batched_heads_active", None)
        st["sidecar"] = adapter
        st["prefix_key"] = pkey
        st["seq_len"] = seq_len
        st["appends"] = 0
        if hasattr(adapter, "set_paged_kv_context"):
            adapter.set_paged_kv_context(kv_cache, attn_metadata, seq_idx)
        t1 = time.perf_counter()
        ctx = adapter.forward(q)
        forward_s = time.perf_counter() - t1
        STATS["pq_forward_s_total"] += forward_s
        STATS["pq_forward_only_s_total"] += forward_s
        if hasattr(adapter, "checkpoint_prefix"):
            adapter.checkpoint_prefix()
        STATS["cuda_graph_active"] = getattr(adapter, "cuda_graph_active", None)
    else:
        t_append = time.perf_counter()
        rt._span_start("append")
        if rt._lean_append():
            if not adapter.lean_append(row(key, seq_idx), row(value, seq_idx), q):
                adapter.append(row(key, seq_idx), row(value, seq_idx))
        else:
            adapter.append(row(key, seq_idx), row(value, seq_idx))
        rt._span_end("append")
        append_s = time.perf_counter() - t_append
        t_forward = time.perf_counter()
        rt._span_start("replay")
        ctx = adapter.forward(q)
        rt._span_end("replay")
        forward_s = time.perf_counter() - t_forward
        STATS["pq_append_s_total"] += append_s
        STATS["pq_forward_only_s_total"] += forward_s
        STATS["pq_forward_s_total"] += append_s + forward_s
        STATS["pq_append_calls"] += 1
        st["seq_len"] = seq_len
        st["appends"] = int(st["appends"]) + 1
        interval = rt.flush_interval()
        if interval > 0 and st["appends"] % interval == 0:
            t2 = time.perf_counter()
            rt.flush_sidecar(adapter)
            STATS["pq_flush_s_total"] += time.perf_counter() - t2
            STATS["pq_flush_calls"] += 1
    STATS["pq_decode_calls"] += 1
    return ctx


# --------------------------------------------------------------------------- #
# the batch driver
# --------------------------------------------------------------------------- #
def layer_req_states(impl, layer) -> dict[Any, dict[str, Any]]:
    layer_st = rt.state_for(impl, layer)
    reqs = layer_st.get("reqs")
    if reqs is None:
        reqs = {}
        layer_st["reqs"] = reqs
    return reqs


def run_pq_decode_batch(
    impl,
    layer,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Loop the requests of one pure-decode step; write each context back in place.

    ``reshape_and_cache_flash`` has already run for the whole batch in
    ``rt.run_pq_decode_into_output`` (it is batch-correct as written).
    """
    lens = batch_seq_lens(attn_metadata)
    num_reqs = len(lens)
    req_keys = batch_req_keys(attn_metadata, num_reqs)
    reqs = layer_req_states(impl, layer)

    # Drop sidecars of requests that have left the batch, so the per-request
    # index instances (and their captured graphs) do not accumulate.
    if len(reqs) > num_reqs:
        live = set(req_keys)
        for stale in [k for k in reqs if k not in live]:
            reqs.pop(stale, None)

    for i in range(num_reqs):
        rk = req_keys[i]
        st = reqs.get(rk)
        if st is None:
            st = new_req_state()
            reqs[rk] = st
        ctx = run_pq_decode_one(
            impl, layer, st, i, lens[i], query, key, value, kv_cache, attn_metadata
        )
        t_copy = time.perf_counter()
        rt._span_start("copy_ctx")
        copy_ctx_at(output, ctx, i)
        rt._span_end("copy_ctx")
        STATS["pq_copy_ctx_s_total"] += time.perf_counter() - t_copy

    STATS["pq_batch_steps"] = int(STATS["pq_batch_steps"]) + 1
    STATS["pq_batch_reqs_total"] = int(STATS["pq_batch_reqs_total"]) + num_reqs
    if num_reqs > int(STATS["pq_batch_max_reqs"] or 0):
        STATS["pq_batch_max_reqs"] = num_reqs
    live_sidecars = sum(1 for s in reqs.values() if s["sidecar"] is not None)
    if live_sidecars > int(STATS["pq_batch_live_sidecars"] or 0):
        STATS["pq_batch_live_sidecars"] = live_sidecars
    return output


__all__ = [
    "batch_seq_lens",
    "batch_req_keys",
    "copy_ctx_at",
    "layer_req_states",
    "new_req_state",
    "prefix_cache_key_at",
    "row",
    "run_pq_decode_batch",
    "run_pq_decode_one",
    "should_pq_decode_batch",
]
