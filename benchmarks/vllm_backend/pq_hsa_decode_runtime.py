"""Shared PQ-HSA decode runtime for the wrap fallback and the native backend.

Persist / flush / sidecar live here once. Both
``pq_hsa_flashattn_wrap`` (PQ_HSA_NATIVE_BACKEND=0) and ``PQHSAImpl``
(PQ_HSA_NATIVE_BACKEND=1) call these functions. Sidecar state is keyed by
``getattr(layer, "layer_name", id(impl))``, not hung only on the impl object.
"""

from __future__ import annotations

from array import array
import hashlib
import os
import time
from typing import Any, Callable, Optional

import torch

from benchmarks.vllm_backend.paged_kv_fa import flashattn_seq_len, gather_flashattn_kv
from pq_hsa.attention.sparse_attention import SparseAttentionConfig
from pq_hsa.index.ivfpq import IVFPQConfig

STATS: dict[str, Any] = {
    "installed": False,
    "backend": None,
    "sidecar": None,
    "pq_decode_calls": 0,
    "pq_build_calls": 0,
    "pq_append_calls": 0,
    "pq_build_s_total": 0.0,
    "pq_forward_s_total": 0.0,
    "pq_forward_only_s_total": 0.0,
    "pq_append_s_total": 0.0,
    "pq_restore_s_total": 0.0,
    "pq_cache_write_s_total": 0.0,
    "pq_gather_s_total": 0.0,
    "pq_flush_s_total": 0.0,
    "pq_flush_calls": 0,
    "pq_copy_ctx_s_total": 0.0,
    "pq_wrap_other_s_total": 0.0,
    "dense_fa_s_total": 0.0,
    "dense_fa_calls": 0,
    "batched_heads_active": None,
    "cuda_graph_active": None,
    "dense_fallback_calls": 0,
    "last_error": None,
    "layers_built": 0,
    "prefix_persist_hits": 0,
    "prefix_persist_misses": 0,
    "prefix_miss_reasons": {},
    "prefix_key_debug": None,
    # (PQ_HSA_PREFIX_EXTEND=1 only; stay 0/empty on the default path).
    "prefix_extend_hits": 0,
    "prefix_extend_rejects": {},
    "pq_extend_s_total": 0.0,
    "pq_extend_delta_tokens_total": 0,
    "pq_extend_last": None,
    "reject": {},
    "fused_decode_calls": 0,
    "fused_full_calls": 0,
    "fused_radix_calls": 0,
    "cuda_fused_calls": 0,
    "cuda_attend_only_calls": 0,
    "pq_norecapture_flush": 0,
    "pq_recapture_flush": 0,
    "graph_cap_warmup_s": 0.0,
    "graph_cap_capture_s": 0.0,
    "graph_cap_validate_s": 0.0,
    # (PQ_HSA_BATCH=1 only; stay 0 on the default single-request path).
    "pq_batch_steps": 0,
    "pq_batch_reqs_total": 0,
    "pq_batch_max_reqs": 0,
    "pq_batch_live_sidecars": 0,
    "batch_mode": False,
}

_PRESERVE_ON_RESET = (
    "installed",
    "backend",
    "sidecar",
    "batched_heads_active",
    "cuda_graph_active",
    "batch_mode",
)

# layer_name (or impl id) -> sidecar state. Not cleared by reset_stats.
_LAYER_STATE: dict[Any, dict[str, Any]] = {}


def pq_hsa_runtime_stats() -> dict[str, Any]:
    return dict(STATS)


def reset_pq_hsa_runtime_stats() -> None:
    for k, v in list(STATS.items()):
        if k in _PRESERVE_ON_RESET:
            continue
        if isinstance(v, dict):
            STATS[k] = {}
        elif isinstance(v, float):
            STATS[k] = 0.0
        elif isinstance(v, (bool, int)):
            STATS[k] = 0
        else:
            STATS[k] = None


def reset_pq_hsa_layer_state() -> None:
    """Drop sidecar adapters (call when destroying a vLLM engine)."""
    _LAYER_STATE.clear()


def layer_state_key(impl, layer) -> Any:
    if layer is not None:
        name = getattr(layer, "layer_name", None)
        if name is not None:
            return name
    return id(impl)


def state_for(impl, layer) -> dict[str, Any]:
    key = layer_state_key(impl, layer)
    st = _LAYER_STATE.get(key)
    if st is None:
        st = {"sidecar": None, "seq_len": None, "prefix_key": None, "appends": 0}
        _LAYER_STATE[key] = st
    return st


def sidecar_cls():
    if os.environ.get("PQ_HSA_SIDECAR", "fast") == "loop":
        from benchmarks.vllm_backend.gqa_sidecar import GQASidecar

        return GQASidecar
    from benchmarks.vllm_backend.gqa_sidecar_fast import GQASidecarFast

    return GQASidecarFast


def paper_index_config() -> IVFPQConfig:
    return IVFPQConfig(
        num_lists=int(os.environ.get("PQ_HSA_NUM_LISTS", "512")),
        nprobe=int(os.environ.get("PQ_HSA_NPROBE", "64")),
        num_subspaces=int(os.environ.get("PQ_HSA_SUBSPACES", "8")),
        num_bits=int(os.environ.get("PQ_HSA_BITS", "4")),
        residual=True,
        coarse_max_iter=int(os.environ.get("PQ_HSA_COARSE_ITER", "1")),
        pq_max_iter=int(os.environ.get("PQ_HSA_PQ_ITER", "2")),
        pack_codes=True,
        topk_block_size=int(os.environ.get("PQ_HSA_TOPK_BLOCK", "1024")),
        kernel_backend=os.environ.get("PQ_HSA_KERNEL", "h20"),
        rotation="none",
        direction_normalize=os.environ.get("PQ_HSA_DIR_NORM", "1") == "1",
        seed=0,
    )


def paper_attention_config(scale: float) -> SparseAttentionConfig:
    return SparseAttentionConfig(
        sink_tokens=int(os.environ.get("PQ_HSA_SINK", "4")),
        local_window=int(os.environ.get("PQ_HSA_LOCAL", "128")),
        retrieval_topk=100,
        retrieval_top_fraction=float(os.environ.get("PQ_HSA_TOP_FRAC", "0.01")),
        nprobe=int(os.environ.get("PQ_HSA_NPROBE", "64")),
        candidate_budget=int(os.environ.get("PQ_HSA_CAND_BUDGET", "4096")),
        exact_rerank=True,
        scale=float(scale),
        mode="hybrid",
        hybrid_value_mode="centroid",
        hybrid_topk_source="all_pq",
        hybrid_denominator_source="all_pq",
        index_update_interval=int(os.environ.get("PQ_HSA_UPD_INTERVAL", "256")),
        index_update_strategy="deferred",
        kv_storage="device",
        collect_attention_details=False,
        profile_attention_components=False,
    )


def _host_sync_fix() -> bool:
    """Skip leftover decode-path host/CUDA queries. PQ_HSA_HOST_SYNC_FIX=0 restores them."""
    return os.environ.get("PQ_HSA_HOST_SYNC_FIX", "1") == "1"


def _lean_wrap() -> bool:
    """A: skip per-layer perf_counter / env / STATS wall clocks. Default OFF until gated."""
    return os.environ.get("PQ_HSA_LEAN_WRAP", "0") == "1"


def batch_enabled() -> bool:
    """Naive multi-request decode. Default OFF -- nothing below changes.

    Cached like the other decode-path env gates; the flag is read once per
    process because switching it mid-run would strand per-request sidecars.
    """
    global _BATCH_MODE
    if _BATCH_MODE is None:
        _BATCH_MODE = os.environ.get("PQ_HSA_BATCH", "0") == "1"
        STATS["batch_mode"] = _BATCH_MODE
    return _BATCH_MODE


_BATCH_MODE: bool | None = None


def _graph_out_direct() -> bool:
    """C: skip dtype/shape branches in copy_ctx when shapes already match."""
    return os.environ.get("PQ_HSA_GRAPH_OUT_DIRECT", "0") == "1"


_LEAN_APPEND: bool | None = None
# Read at import: the same-stack A/B sets the variant env before the engine (and
# therefore this module) is imported.  _lean_copy_ctx() re-reads on demand.
_LEAN_COPYCTX: bool = os.environ.get("PQ_HSA_LEAN_COPYCTX", "0") == "1"


def _lean_append() -> bool:
    """One-launch append (pq_append_prep) + deferred view reslices."""
    global _LEAN_APPEND
    if _LEAN_APPEND is None:
        _LEAN_APPEND = os.environ.get("PQ_HSA_LEAN_APPEND", "0") == "1"
    return _LEAN_APPEND


def _lean_copy_ctx() -> bool:
    """Copy the graph context straight into vLLM's buffer, no branches."""
    global _LEAN_COPYCTX
    _LEAN_COPYCTX = os.environ.get("PQ_HSA_LEAN_COPYCTX", "0") == "1"
    return _LEAN_COPYCTX


_ENV_INT_CACHE: dict[str, int] = {}
_ENABLE_CACHE: tuple[bool, str] | None = None
_FLUSH_CACHE: int | None = None


def _cached_env_int(name: str, default: str) -> int:
    if _host_sync_fix():
        hit = _ENV_INT_CACHE.get(name)
        if hit is None:
            hit = int(os.environ.get(name, default))
            _ENV_INT_CACHE[name] = hit
        return hit
    return int(os.environ.get(name, default))


def _events_on() -> bool:
    return os.environ.get("PQ_HSA_STEADY_EVENTS", "0") == "1"


def _span_start(name: str) -> None:
    if not _events_on():
        return
    from benchmarks.cuda_event_log import enable, is_enabled, span_start

    if not is_enabled():
        enable()
    span_start(name)


def _span_end(name: str) -> None:
    if not _events_on():
        return
    from benchmarks.cuda_event_log import span_end

    span_end(name)


def min_pq_seq() -> int:
    sink = _cached_env_int("PQ_HSA_SINK", "4")
    local = _cached_env_int("PQ_HSA_LOCAL", "128")
    return sink + local + 16


_MQ_MODE: bool | None = None


def mq_enabled() -> bool:
    """(PQ_HSA_MULTI_QUERY=1, opt-in): speculative / MTP multi-query decode."""
    global _MQ_MODE
    if _MQ_MODE is None:
        _MQ_MODE = os.environ.get("PQ_HSA_MULTI_QUERY", "0") == "1"
    return _MQ_MODE


def reject(reason: str) -> bool:
    STATS["reject"][reason] = STATS["reject"].get(reason, 0) + 1
    return False


def _is_capturing() -> bool:
    # Runs enforce_eager (no vLLM piecewise capture). The CUDA
    # driver query ran 32×/token. HOST_SYNC_FIX=0 restores the old check.
    if _host_sync_fix():
        return False
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def should_pq_decode(attn_metadata) -> bool:
    if _lean_wrap():
        cached = getattr(attn_metadata, "_pq_should_decode", None)
        if cached is not None:
            return bool(cached)
    if _is_capturing():
        return reject("stream_capturing")
    if attn_metadata is None:
        return reject("no_metadata")
    if getattr(attn_metadata, "use_cascade", False):
        return reject("cascade")
    if int(getattr(attn_metadata, "max_query_len", 0)) != 1:
        if mq_enabled():
            # Speculative / MTP step (T = 1 + k queries of one request).
            from benchmarks.vllm_backend.pq_hsa_multi_query import should_pq_decode_mq

            ok = should_pq_decode_mq(attn_metadata)
            if ok and _lean_wrap():
                attn_metadata._pq_should_decode = True
            return ok
        return reject("max_query_len!=1")
    if batch_enabled():
        # Allow num_reqs > 1. Everything after this line is the
        # default single-request admission test and is left untouched.
        from benchmarks.vllm_backend.pq_hsa_batch_decode import should_pq_decode_batch

        ok = should_pq_decode_batch(attn_metadata)
        if ok and _lean_wrap():
            attn_metadata._pq_should_decode = True
        return ok
    if int(getattr(attn_metadata, "num_actual_tokens", 0)) != 1:
        return reject("num_actual_tokens!=1")
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is None or int(seq_lens.shape[0]) != 1:
        return reject("num_reqs!=1")
    if flashattn_seq_len(attn_metadata, 0) < min_pq_seq():
        return reject("seq_too_short")
    if _lean_wrap() and attn_metadata is not None:
        attn_metadata._pq_should_decode = True
    return True


def flush_interval() -> int:
    # Same-stack A/B toggles this between 0 and Δ. LEAN_WRAP caches until
    # PQ_HSA_FLUSH_INTERVAL actually changes (string identity).
    if _lean_wrap():
        global _FLUSH_CACHE
        raw = os.environ.get("PQ_HSA_FLUSH_INTERVAL", "256")
        if _FLUSH_CACHE is None or _FLUSH_CACHE[0] != raw:
            _FLUSH_CACHE = (raw, int(raw))
        return _FLUSH_CACHE[1]
    return int(os.environ.get("PQ_HSA_FLUSH_INTERVAL", "256"))


def pq_enabled() -> bool:
    global _ENABLE_CACHE
    raw = os.environ.get("PQ_HSA_ENABLE", "0")
    if _ENABLE_CACHE is None or _ENABLE_CACHE[1] != raw:
        _ENABLE_CACHE = (raw == "1", raw)
    return _ENABLE_CACHE[0]


def flush_sidecar(adapter) -> None:
    inner = getattr(adapter, "adapter", None)
    if inner is not None and hasattr(inner, "apply_pending_index_update"):
        inner.apply_pending_index_update()
        return
    streams = getattr(adapter, "streams", None)
    if streams:
        for stream in streams:
            if hasattr(stream, "apply_pending_index_update"):
                stream.apply_pending_index_update()


def as_decode_qkv(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3:
        return t[:1]
    if t.ndim == 2:
        return t[:1].unsqueeze(0)
    raise ValueError(f"expected [T,H,D] or [T,D], got {tuple(t.shape)}")


def copy_ctx(output: torch.Tensor, ctx: torch.Tensor, num_actual: int) -> None:
    # Steady decode has num_actual == 1 and output.shape[0] == 1 (checked
    # by should_pq_decode), so the slice + the two shape/dtype branches + the
    # per-call os.environ lookup are all dead weight: ~7.0 us -> ~4.1 us of host
    # per layer per token. Byte-identical copy.
    if _LEAN_COPYCTX and output.shape == ctx.shape and output.dtype == ctx.dtype:
        output.copy_(ctx)
        return
    dst = output[:num_actual]
    if _graph_out_direct() and ctx.shape == dst.shape and ctx.dtype == dst.dtype:
        dst.copy_(ctx)
        return
    if ctx.shape != dst.shape:
        ctx = ctx.reshape(dst.shape)
    if ctx.dtype != dst.dtype:
        ctx = ctx.to(dtype=dst.dtype)
    dst.copy_(ctx)


def prefix_cache_key(attn_metadata, seq_len: int) -> tuple[int, bytes] | None:
    cached = getattr(attn_metadata, "_pq_prefix_cache_key", None)
    if cached is not None:
        return cached
    runner = getattr(attn_metadata, "_pq_runner", None)
    if runner is None:
        return None
    context_len = int(seq_len) - int(attn_metadata.max_query_len)
    req_ids = runner.input_batch.req_ids
    if not req_ids:
        return None
    request = runner.requests.get(req_ids[0])
    prompt_ids = None if request is None else request.prompt_token_ids
    if prompt_ids is None:
        return None
    prompt_bytes = array("I", prompt_ids).tobytes()
    digest = hashlib.blake2b(prompt_bytes, digest_size=16).digest()
    key = (context_len, digest)
    attn_metadata._pq_prefix_cache_key = key
    return key


def _prefix_extend_enabled() -> bool:
    """Opt-in: restore-and-extend a checkpointed prefix (see
    GQAIVFPQDecodeAttentionAdapter.extend_prefix). Default OFF."""
    return os.environ.get("PQ_HSA_PREFIX_EXTEND", "0") == "1"


def _prefix_extend_max_frac() -> float:
    try:
        return float(os.environ.get("PQ_HSA_PREFIX_EXTEND_MAX_FRAC", "0.5"))
    except ValueError:
        return 0.5


def prefix_extend_match(attn_metadata, old_key, checkpoint_len, seq_len: int):
    """Decide whether the current request's prompt strictly EXTENDS the
    checkpointed prompt.

    ``old_key`` is the (context_len, digest) key stored when the checkpoint was
    written (the checkpointed prompt P, |P| == old_key[0]); ``checkpoint_len``
    is the sidecar length at that time (|P| + 1, the first decode token row).
    Returns ``(ok, reason, new_key, delta_tokens)``. The digest of the new
    prompt's first |P| ids is cached on ``attn_metadata`` per |P|, so the 32
    per-layer calls of one decode step hash the prefix once.
    """
    if old_key is None or checkpoint_len is None:
        return False, "no_checkpoint", None, 0
    old_ctx, old_digest = int(old_key[0]), old_key[1]
    checkpoint_len = int(checkpoint_len)
    if checkpoint_len != old_ctx + 1:
        return False, "checkpoint_inconsistent", None, 0
    context_len = int(seq_len) - int(attn_metadata.max_query_len)
    if context_len <= old_ctx:
        return False, "not_longer", None, 0
    delta = int(seq_len) - checkpoint_len
    if delta <= 0:
        return False, "not_longer", None, 0
    if float(delta) / float(seq_len) > _prefix_extend_max_frac():
        return False, "delta_too_large", None, delta
    digests = getattr(attn_metadata, "_pq_prefix_ext_digests", None)
    if digests is None:
        digests = {}
        try:
            attn_metadata._pq_prefix_ext_digests = digests
        except Exception:
            pass
    d = digests.get(old_ctx)
    if d is None:
        runner = getattr(attn_metadata, "_pq_runner", None)
        if runner is None:
            return False, "no_runner", None, delta
        req_ids = runner.input_batch.req_ids
        if not req_ids:
            return False, "no_request", None, delta
        request = runner.requests.get(req_ids[0])
        prompt_ids = None if request is None else request.prompt_token_ids
        if prompt_ids is None or len(prompt_ids) != context_len:
            return False, "prompt_len_mismatch", None, delta
        d = hashlib.blake2b(array("I", prompt_ids[:old_ctx]).tobytes(), digest_size=16).digest()
        digests[old_ctx] = d
    if d != old_digest:
        return False, "prefix_digest_mismatch", None, delta
    new_key = prefix_cache_key(attn_metadata, seq_len)
    if new_key is None:
        return False, "no_prefix_key", None, delta
    return True, "ok", new_key, delta


def run_pq_decode(impl, layer, query, key, value, kv_cache, attn_metadata) -> torch.Tensor:
    """Build / append / persist / flush the sidecar and return context [1,H,D]."""
    st = state_for(impl, layer)
    seq_len = flashattn_seq_len(attn_metadata, 0)
    prev = st["seq_len"]
    adapter = st["sidecar"]
    sequential = adapter is not None and prev is not None and seq_len == int(prev) + 1
    q = as_decode_qkv(query)
    if (
        (
            os.environ.get("PQ_HSA_PAGED_KV", "0") == "1"
            # (PQ_HSA_PAGED_ATTEND=1, opt-in): the production CUDA
            # exact-attend epilogue's paged-KV variant (cuda_attend_only_paged,
            # called from e2e_pq_param_sweep._cg_forward_static) needs this
            # same per-step-refreshed context; reuse the plumbing rather
            # than adding a second copy of it.
            or os.environ.get("PQ_HSA_PAGED_ATTEND", "0") == "1"
        )
        and adapter is not None
        and hasattr(adapter, "set_paged_kv_context")
    ):
        # (PQ_HSA_PAGED_KV=1, opt-in): refresh the paged-KV reference
        # every step. Gated on the flags themselves (not just hasattr) so the
        # default path never holds a reference to vLLM's kv_cache tensor on
        # the adapter -- an earlier unconditional version of this call was
        # harmless for compute (never read when both flags are off) but
        # polluted the sidecar tensor census: walking the
        # adapter's reachable tensors picked up vLLM's own (pre-existing,
        # engine-lifetime) kv_cache through this attribute and mis-bucketed
        # its full size as sidecar memory.
        adapter.set_paged_kv_context(kv_cache, attn_metadata, 0)
    if not sequential:
        pkey = prefix_cache_key(attn_metadata, seq_len)
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
            _span_start("restore")
            # Pass kv_cache/attn_metadata so restore_prefix can refresh
            # the stale tail-block range of the checkpointed prefix from
            # vLLM's live paged KV (see its docstring in e2e_pq_param_sweep.py).
            # No gather-kernel or CUDA-graph call sites changed.
            _restore_fn = adapter.restore_prefix
            # The sidecar may be the GQASidecarFast wrapper; the ring flag and
            # the ring-aware restore live on the wrapped adapter.
            _inner = getattr(adapter, "adapter", adapter)
            if (
                os.environ.get("PQ_HSA_PAGED_RESTORE", "0") == "1"
                and getattr(_inner, "_shared_base_ring", False)
                and hasattr(_inner, "restore_prefix_ring")
            ):
                _restore_fn = _inner.restore_prefix_ring  # Ring-aware restore
            restored = _restore_fn(
                as_decode_qkv(key), as_decode_qkv(value),
                kv_cache=kv_cache, attn_metadata=attn_metadata, seq_idx=0,
            )
            _span_end("restore")
            restore_s = time.perf_counter() - t_restore
            if not restored:
                raise RuntimeError("prefix sidecar checkpoint disappeared during restore")
            STATS["pq_restore_s_total"] += restore_s
            STATS["prefix_persist_hits"] += 1
            st["seq_len"] = seq_len
            st["appends"] = 0
            t_forward = time.perf_counter()
            _span_start("replay")
            ctx = adapter.forward(q)
            _span_end("replay")
            forward_s = time.perf_counter() - t_forward
            STATS["pq_forward_only_s_total"] += forward_s
            STATS["pq_forward_s_total"] += restore_s + forward_s
            STATS["cuda_graph_active"] = getattr(adapter, "cuda_graph_active", None)
            STATS["pq_decode_calls"] += 1
            return ctx

        # --- (opt-in PQ_HSA_PREFIX_EXTEND=1): restore-and-extend ---------
        # The exact-match restore above missed. If the new prompt strictly
        # extends the checkpointed one, restore the old index and fold the
        # delta in through the flush path instead of rebuilding everything.
        if _prefix_extend_enabled() and adapter is not None and hasattr(adapter, "extend_prefix"):
            ok_ext, ext_reason, ext_key, ext_delta = prefix_extend_match(
                attn_metadata, st["prefix_key"], checkpoint_len, seq_len
            )
            if ok_ext:
                t_ext = time.perf_counter()
                _span_start("extend")
                ext_info = adapter.extend_prefix(
                    as_decode_qkv(key), as_decode_qkv(value),
                    kv_cache=kv_cache, attn_metadata=attn_metadata, seq_idx=0,
                    new_length=seq_len,
                )
                _span_end("extend")
                ext_s = time.perf_counter() - t_ext
                if ext_info is None:
                    ext_reason = "adapter_declined"
                else:
                    STATS["pq_extend_s_total"] += ext_s
                    STATS["prefix_extend_hits"] += 1
                    STATS["pq_extend_delta_tokens_total"] = int(
                        STATS["pq_extend_delta_tokens_total"]
                    ) + int(ext_delta)
                    ext_info = dict(ext_info)
                    ext_info["wall_s"] = ext_s
                    STATS["pq_extend_last"] = ext_info
                    st["prefix_key"] = ext_key
                    st["seq_len"] = seq_len
                    st["appends"] = 0
                    t_forward = time.perf_counter()
                    _span_start("replay")
                    ctx = adapter.forward(q)
                    _span_end("replay")
                    forward_s = time.perf_counter() - t_forward
                    STATS["pq_forward_only_s_total"] += forward_s
                    STATS["pq_forward_s_total"] += ext_s + forward_s
                    if hasattr(adapter, "checkpoint_prefix"):
                        adapter.checkpoint_prefix()  # re-arm for the next turn
                    STATS["cuda_graph_active"] = getattr(adapter, "cuda_graph_active", None)
                    STATS["pq_decode_calls"] += 1
                    return ctx
            rej = STATS["prefix_extend_rejects"]
            rej[ext_reason] = rej.get(ext_reason, 0) + 1

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
        if os.environ.get("PQ_HSA_DEBUG_PREFIX", "0") == "1":
            old_key = st["prefix_key"]
            STATS["prefix_key_debug"] = {
                "new_context": None if pkey is None else pkey[0],
                "new_digest": None if pkey is None else pkey[1].hex(),
                "old_context": None if old_key is None else old_key[0],
                "old_digest": None if old_key is None else old_key[1].hex(),
                "checkpoint_length": checkpoint_len,
                "seq_len": seq_len,
            }
        t_g = time.perf_counter()
        keys, values, gathered = gather_flashattn_kv(kv_cache, attn_metadata, 0)
        STATS["pq_gather_s_total"] += time.perf_counter() - t_g
        if gathered != seq_len:
            raise RuntimeError(f"gather seq {gathered} != metadata seq {seq_len}")
        t0 = time.perf_counter()
        _pre = st.pop("prebuilt", None)
        if _pre is not None and _pre.get("seq_len") != seq_len:
            STATS["pq_prebuilt_mismatch"] = int(STATS.get("pq_prebuilt_mismatch", 0)) + 1
            STATS["pq_prebuilt_mismatch_detail"] = (int(_pre.get("seq_len")), int(seq_len))
        if _pre is not None and _pre.get("seq_len") == seq_len:
            from pq_hsa.index.batched_build import publish_prebuilt

            publish_prebuilt(_pre)
            STATS["pq_prebuilt_consumed"] = int(STATS.get("pq_prebuilt_consumed", 0)) + 1
        adapter = sidecar_cls()(
            paper_index_config(),
            paper_attention_config(impl.scale),
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
            adapter.set_paged_kv_context(kv_cache, attn_metadata, 0)
        t1 = time.perf_counter()
        _span_start("replay")
        ctx = adapter.forward(q)
        _span_end("replay")
        forward_s = time.perf_counter() - t1
        STATS["pq_forward_s_total"] += forward_s
        STATS["pq_forward_only_s_total"] += forward_s
        if hasattr(adapter, "checkpoint_prefix"):
            adapter.checkpoint_prefix()
        STATS["cuda_graph_active"] = getattr(adapter, "cuda_graph_active", None)
    else:
        lean = _lean_wrap()
        if not lean:
            t_append = time.perf_counter()
        _span_start("append")
        if _lean_append():
            # `key`/`value`/`query` are contiguous [T,H,D] with T == 1 in
            # steady decode, so the kernel can read row 0 straight off data_ptr
            # -- no t[:1] slices, no reshape, no shape validation.
            if not adapter.lean_append(key, value, q):
                adapter.append(as_decode_qkv(key), as_decode_qkv(value))
        else:
            adapter.append(as_decode_qkv(key), as_decode_qkv(value))
        _span_end("append")
        append_s = 0.0 if lean else (time.perf_counter() - t_append)
        if not lean:
            t_forward = time.perf_counter()
        _span_start("replay")
        ctx = adapter.forward(q)
        _span_end("replay")
        forward_s = 0.0 if lean else (time.perf_counter() - t_forward)
        STATS["pq_append_s_total"] += append_s
        STATS["pq_forward_only_s_total"] += forward_s
        STATS["pq_forward_s_total"] += append_s + forward_s
        STATS["pq_append_calls"] += 1
        st["seq_len"] = seq_len
        st["appends"] = int(st["appends"]) + 1
        interval = flush_interval()
        if interval > 0 and st["appends"] % interval == 0:
            t2 = time.perf_counter()
            flush_sidecar(adapter)
            STATS["pq_flush_s_total"] += time.perf_counter() - t2
            STATS["pq_flush_calls"] += 1
    STATS["pq_decode_calls"] += 1
    return ctx


def run_pq_decode_into_output(
    impl,
    layer,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """reshape_and_cache_flash + sidecar decode + copy into vLLM's output buffer."""
    lean = _lean_wrap()
    if not lean:
        t_cache = time.perf_counter()
    _span_start("cache_write")
    if kv_cache.dim() == 5:
        key_cache, value_cache = kv_cache.unbind(0)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            impl.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
    else:
        # (vLLM >= 0.10): FlashAttentionBackend.forward_includes_kv_cache_update
        # is False -- vLLM wrote this step's K/V into its (4-D, K|V-interleaved)
        # page before calling forward, so there is nothing to write here.
        STATS["cache_write_skipped_new_layout"] = int(STATS.get("cache_write_skipped_new_layout") or 0) + 1
    _span_end("cache_write")
    if not lean:
        STATS["pq_cache_write_s_total"] += time.perf_counter() - t_cache
    if mq_enabled() and (
        int(getattr(attn_metadata, "max_query_len", 0)) != 1 or not batch_enabled()
    ):
        # Multi-query steps, plus the T==1 steps in between them (the
        # module resolves the previous speculative step, then delegates T==1
        # to run_pq_decode unchanged).
        from benchmarks.vllm_backend.pq_hsa_multi_query import run_pq_decode_mq

        return run_pq_decode_mq(
            impl, layer, query, key, value, kv_cache, attn_metadata, output
        )
    if batch_enabled():
        # Reshape_and_cache_flash above is already batch-correct (it
        # consumes the whole slot_mapping); the sidecar loop is not.
        from benchmarks.vllm_backend.pq_hsa_batch_decode import run_pq_decode_batch

        return run_pq_decode_batch(
            impl, layer, query, key, value, kv_cache, attn_metadata, output
        )
    # Should_pq_decode already requires num_actual_tokens == 1.
    # int() on a Python int is free; keep the getattr for the old path.
    if _host_sync_fix():
        num_actual = 1
    else:
        num_actual = int(attn_metadata.num_actual_tokens)
    ctx = run_pq_decode(impl, layer, query, key, value, kv_cache, attn_metadata)
    if not lean:
        t_copy = time.perf_counter()
    _span_start("copy_ctx")
    copy_ctx(output, ctx, num_actual)
    _span_end("copy_ctx")
    if not lean:
        STATS["pq_copy_ctx_s_total"] += time.perf_counter() - t_copy
    return output


def pq_or_dense_forward(
    impl,
    layer,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: Optional[torch.Tensor],
    dense_forward: Callable[[], torch.Tensor],
) -> torch.Tensor:
    if (not pq_enabled()) or not should_pq_decode(attn_metadata):
        _span_start("dense_fa")
        t_dense = time.perf_counter()
        out = dense_forward()
        STATS["dense_fa_s_total"] += time.perf_counter() - t_dense
        STATS["dense_fa_calls"] = int(STATS["dense_fa_calls"]) + 1
        _span_end("dense_fa")
        if pq_enabled():
            _maybe_prefill_prebuild(impl, layer, kv_cache, attn_metadata)
        return out
    allow_fallback = os.environ.get("PQ_HSA_FALLBACK", "0") == "1"
    try:
        assert output is not None
        return run_pq_decode_into_output(
            impl, layer, query, key, value, kv_cache, attn_metadata, output
        )
    except Exception as exc:
        STATS["last_error"] = f"{type(exc).__name__}: {exc}"
        STATS["dense_fallback_calls"] += 1
        if not allow_fallback:
            raise
        return dense_forward()



def _maybe_prefill_prebuild(impl, layer, kv_cache, attn_metadata) -> None:
    """(opt-in PQ_HSA_PREFILL_PREBUILD=1): on the LAST prefill forward of a
    layer, launch the batched codebook training on a side stream so it overlaps with
    the remaining layers' prefill.  Consumed by run_pq_decode at the first decode step."""
    try:
        from pq_hsa.index.batched_build import prefill_prebuild_enabled, prebuild_layer_async

        if not prefill_prebuild_enabled():
            return
        if attn_metadata is None or getattr(attn_metadata, "use_cascade", False):
            return
        mql = int(getattr(attn_metadata, "max_query_len", 0))
        if mql <= 1:
            return
        seq_lens = attn_metadata.seq_lens
        if int(seq_lens.shape[0]) != 1:
            return
        seq_len = int(attn_metadata.max_seq_len)
        runner = getattr(attn_metadata, "_pq_runner", None)
        if runner is None:
            return
        req_ids = runner.input_batch.req_ids
        if not req_ids:
            return
        request = runner.requests.get(req_ids[0])
        prompt_ids = None if request is None else request.prompt_token_ids
        if prompt_ids is None or seq_len != len(prompt_ids):
            return  # not the last prefill chunk of this prompt
        t0 = time.perf_counter()
        keys, _values, gathered = gather_flashattn_kv(kv_cache, attn_metadata, 0)
        if gathered != seq_len:
            return
        cfg = paper_index_config()
        acfg = paper_attention_config(impl.scale)
        rec = prebuild_layer_async(
            keys[0], cfg, sink_tokens=int(acfg.sink_tokens), local_window=int(acfg.local_window)
        )
        st = state_for(impl, layer)
        st["prebuilt"] = rec
        STATS["pq_prebuild_last_seq_len"] = int(rec["seq_len"])
        STATS["pq_prebuild_calls"] = int(STATS.get("pq_prebuild_calls", 0)) + 1
        STATS["pq_prebuild_launch_s_total"] = float(STATS.get("pq_prebuild_launch_s_total", 0.0)) + (
            time.perf_counter() - t0
        )
    except Exception as exc:  # never break the dense path
        STATS["last_prebuild_error"] = f"{type(exc).__name__}: {exc}"


def install_flashattn_metadata_runner_hook() -> None:
    """Wrap fallback: attach runner onto FA metadata for prefix persist."""
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder

    if getattr(FlashAttentionMetadataBuilder.build, "_pq_hsa_wrapped", False):
        return
    if not hasattr(FlashAttentionMetadataBuilder, "runner"):
        # On 0.8.5.post1 the builder is constructed with the runner
        # (instance attribute, so hasattr on the class is False there too);
        # the capture hook is a no-op unless the runner class lacks it.
        from benchmarks.vllm_backend.compat_v029 import install_runner_capture

        install_runner_capture()
    orig_build_metadata = FlashAttentionMetadataBuilder.build

    def build_metadata_with_runner(self, *args, **kwargs):
        metadata = orig_build_metadata(self, *args, **kwargs)
        runner = getattr(self, "runner", None)
        if runner is None:
            # (vLLM >= 0.10): the builder no longer holds the model runner;
            # compat_v029 captures it from GPUModelRunner.execute_model.
            from benchmarks.vllm_backend.compat_v029 import current_runner

            runner = current_runner()
        metadata._pq_runner = runner
        return metadata

    build_metadata_with_runner._pq_hsa_wrapped = True  # type: ignore[attr-defined]
    FlashAttentionMetadataBuilder.build = build_metadata_with_runner
