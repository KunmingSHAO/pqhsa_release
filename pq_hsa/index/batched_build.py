"""Batched (all KV heads x all PQ subspaces at once) codebook
training on top of Flash-KMeans kernels.

Opt-in only.  Nothing in this module runs unless the adapter is invoked with
``PQ_HSA_BATCHED_BUILD=1``.  The default
per-head ``IVFPQIndex.build`` path is untouched.

Why: the production build is 32 layers x 8 KV heads = 256
independent ``IVFPQIndex.build`` calls, each doing 1 coarse k-means
(N~130K, K=512, D=128, 1 iter) + 8 tiny subspace k-means (N~130K, K=16, D=16,
2 iters) sequentially -> ~2.3K k-means calls, each with its own init,
host syncs (torch.equal / .any() / .item()) and a final chunked re-assign.
The k-means themselves are small; launch + sync overhead dominates
(GPU busy 51.8%).  Flash-KMeans' kernels are batched over a leading
dim, so one layer's 8 heads (coarse) and 8x8=64 (head, subspace) problems
(PQ) can be trained in ONE call each, and the Python-side loop drops from
~17 k-means/head to 2 kernel-pairs/layer with zero host syncs.

Numerics: same K, same iteration count, same seeded init rule as
``kmeans_l2`` (randperm(N) with the per-call seed; the per-head default path
uses the same seed for every head so the sampled indices are identical
across heads, which is exactly what a batched gather with one index set
reproduces).  Differences vs the torch path are the FlashAssign / sorted
update reduction order (reconstruction MSE rel. diff 2.4e-7).
"""
from __future__ import annotations

import os

import torch


def batched_build_enabled() -> bool:
    return os.environ.get("PQ_HSA_BATCHED_BUILD", "0") == "1"


def _kernels():
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton  # type: ignore
    from flash_kmeans.centroid_update_triton import (  # type: ignore
        triton_centroid_update_sorted_euclid,
    )

    return euclid_assign_triton, triton_centroid_update_sorted_euclid


def _seeded_perm(n: int, k: int, seed: int, device: torch.device) -> torch.Tensor:
    """Same init rule as kmeans_l2: sample-without-replacement via randperm."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    if n >= k:
        return torch.randperm(n, generator=g, device=device)[:k]
    extra = torch.randint(0, n, (k - n,), generator=g, device=device)
    return torch.cat([torch.arange(n, device=device), extra], dim=0)


def _x_sq(x: torch.Tensor) -> torch.Tensor:
    B, N, _ = x.shape
    out = torch.empty((B, N), device=x.device, dtype=x.dtype)
    step = 1 << 20
    for i in range(0, N, step):
        out[:, i : i + step] = (x[:, i : i + step] ** 2).sum(dim=-1)
    return out


def _kmeans_batched(
    x: torch.Tensor,
    init: torch.Tensor,
    *,
    iters: int,
) -> torch.Tensor:
    """x [B,N,D], init [B,K,D] -> final centroids [B,K,D].  No host syncs."""
    assign, update = _kernels()
    xsq = _x_sq(x)
    centroids = init.contiguous()
    for _ in range(max(iters, 1)):
        ids = assign(x, centroids, xsq)
        centroids = update(x, ids, centroids)
    return centroids


@torch.no_grad()
def train_codebooks_batched(
    keys: torch.Tensor,
    config,
    *,
    return_assignments: bool = False,
):
    """keys [H, N, D] (retrieval-region keys of one layer's KV heads).

    Returns (coarse_centroids [H, K, D], pq_codebooks [H, M, C, d]) in
    ``keys.dtype`` (plus list_ids [H, N] int64 and codes [H, N, M] uint8 when
    ``return_assignments``; both computed with the same FlashAssign kernel
    against the final codebooks, replacing the per-head torch.cdist argmin), trained per head exactly like ``IVFPQIndex.build`` would
    (projection, coarse k-means, residual, per-subspace PQ k-means), but with
    all heads / subspaces batched into two Flash-KMeans calls.
    """
    if os.environ.get("PQ_HSA_BATCHED_RESEED", "0") == "1":
        # Opt-in: per-element empty-cluster re-seeding with the kmeans_l2 rule and
        # fp32 ||x||^2 for the assignment kernel. Unset -> the code below, unchanged.
        from pq_hsa.index.batched_reseed import train_codebooks_batched_reseed

        return train_codebooks_batched_reseed(keys, config, return_assignments=return_assignments)
    if keys.ndim != 3:
        raise ValueError(f"keys must be [H, N, D], got {tuple(keys.shape)}")
    if config.rotation != "none":
        raise NotImplementedError("batched build supports rotation='none' only")
    H, N, D = keys.shape
    M = config.num_subspaces
    C = 1 << config.num_bits
    if D % M != 0:
        raise ValueError(f"dim={D} must be divisible by num_subspaces={M}")
    d = D // M
    K = min(config.num_lists, N)
    dev = keys.device

    x = keys.contiguous()
    if config.direction_normalize:
        norms = torch.linalg.vector_norm(x.float(), dim=-1).to(x.dtype).clamp_min(config.eps)
        x = x / norms.unsqueeze(-1)

    # ---- coarse k-means (batched over heads) ----
    perm = _seeded_perm(N, K, config.seed, dev)
    init = x[:, perm, :]  # [H, K, D]
    coarse = _kmeans_batched(x, init, iters=config.coarse_max_iter)

    # ---- residuals against final coarse centroids (same as build(): assign,
    # then keys - centroid[list]) ----
    assign, _ = _kernels()
    ids = assign(x, coarse, _x_sq(x))  # [H, N]
    if config.residual:
        gathered = torch.gather(coarse, 1, ids.long().unsqueeze(-1).expand(-1, -1, D))
        resid = x - gathered
    else:
        resid = x

    # ---- PQ codebooks: batch (head, subspace) -> [H*M, N, d] ----
    sub = resid.reshape(H, N, M, d).permute(0, 2, 1, 3).reshape(H * M, N, d).contiguous()
    pq_seed = config.seed + 10_000
    init_pq = torch.empty(H * M, C, d, device=dev, dtype=sub.dtype)
    for m in range(M):
        pm = _seeded_perm(N, C, pq_seed + m, dev)
        init_pq[m::M] = sub[m::M][:, pm, :]
    cb = _kmeans_batched(sub, init_pq, iters=config.pq_max_iter)  # [H*M, C, d]
    codebooks = cb.reshape(H, M, C, d).to(keys.dtype)
    if not return_assignments:
        return coarse.to(keys.dtype), codebooks
    # ---- encode (batched over head x subspace) with the final codebooks ----
    codes = assign(sub, cb.to(sub.dtype).contiguous(), _x_sq(sub))  # [H*M, N]
    codes = codes.reshape(H, M, N).permute(0, 2, 1).contiguous().to(torch.uint8)  # [H, N, M]
    return coarse.to(keys.dtype), codebooks, ids.long(), codes


# ---------------------------------------------------------------------------
# Prefill-overlapped prebuild (opt-in, PQ_HSA_PREFILL_PREBUILD=1).
#
# The vLLM decode runtime calls ``prebuild_layer_async`` from the LAST prefill
# forward of a layer (after vLLM has written that layer's K/V into the paged
# cache) and runs the batched codebook training on a side CUDA stream, so it
# overlaps with the remaining layers' prefill compute.  At the first decode
# step the runtime publishes the result through ``PENDING_PREBUILT`` right
# before constructing the sidecar; the adapter hook consumes it instead of
# training synchronously.  Single-threaded, one layer at a time.
# ---------------------------------------------------------------------------
PENDING_PREBUILT: dict | None = None
_SIDE_STREAM: torch.cuda.Stream | None = None


def prefill_prebuild_enabled() -> bool:
    return (
        batched_build_enabled()
        and os.environ.get("PQ_HSA_PREFILL_PREBUILD", "0") == "1"
    )


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    global _SIDE_STREAM
    if _SIDE_STREAM is None or _SIDE_STREAM.device != device:
        _SIDE_STREAM = torch.cuda.Stream(device=device)
    return _SIDE_STREAM


@torch.no_grad()
def prebuild_layer_async(
    keys_hnd: torch.Tensor,
    config,
    *,
    sink_tokens: int,
    local_window: int,
) -> dict:
    """keys_hnd [H, N_total, D] (full prompt K of one layer, any layout that
    is readable on the side stream).  Returns a record with an event to wait
    on and the per-head (coarse, codebooks, list_ids, codes) tensors."""
    H, L, D = keys_hnd.shape
    # Anticipate the FIRST decode step: the sidecar is built there with
    # seq_len = L + 1 (the new token is inside the local window), so the
    # retrieval region is [sink, (L+1) - local_window) -- all prompt tokens.
    L_dec = L + 1
    sink_end = min(max(0, sink_tokens), L_dec)
    ret_end = max(sink_end, L_dec - local_window) if local_window > 0 else max(sink_end, L_dec)
    ret_end = min(ret_end, L)
    stream = _side_stream(keys_hnd.device)
    ready = torch.cuda.Event()
    # side stream must see the producer's writes (paged-KV write + our gather)
    stream.wait_stream(torch.cuda.current_stream(keys_hnd.device))
    with torch.cuda.stream(stream):
        keys_hnd.record_stream(stream)
        ret = keys_hnd[:, sink_end:ret_end, :].contiguous()
        coarse, cbs, lids, codes = train_codebooks_batched(ret, config, return_assignments=True)
        for t in (coarse, cbs, lids, codes):
            t.record_stream(torch.cuda.current_stream(keys_hnd.device))
        ready.record(stream)
    return {
        "seq_len": int(L_dec),
        "retrieval_len": int(ret_end - sink_end),
        "H": int(H),
        "event": ready,
        "tensors": (coarse, cbs, lids, codes),
    }


def publish_prebuilt(record: dict) -> None:
    """Make the side-stream result visible to the adapter hook on the current stream."""
    global PENDING_PREBUILT
    torch.cuda.current_stream().wait_event(record["event"])
    PENDING_PREBUILT = record


def take_prebuilt(H: int, retrieval_len: int):
    """Adapter side: pop the pending record if it matches this layer's shape."""
    global PENDING_PREBUILT
    rec = PENDING_PREBUILT
    PENDING_PREBUILT = None
    if rec is None:
        return None
    if rec["H"] != H or rec["retrieval_len"] != retrieval_len:
        return None
    return rec["tensors"]
