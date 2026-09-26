"""Empty-cluster re-seeding for the batched (all-heads-at-once) index build.

Runs by default whenever ``PQ_HSA_BATCHED_BUILD=1``;
``PQ_HSA_BATCHED_RESEED=0`` falls back to the plain batched k-means of
``batched_build._kmeans_batched``. The default per-head build
(``PQ_HSA_BATCHED_BUILD`` unset) never calls this module.

What it changes relative to ``batched_build._kmeans_batched``:

1. Empty-cluster re-seeding, per batch element, with the rule of the default
   per-head ``kmeans_l2`` (pq_hsa/index/kmeans.py):
     * trigger: count == 0 after the assignment of an iteration (the update
       then has nothing to average);
     * replacement: ``torch.randint(0, N, (n_empty,), generator=g)`` drawn
       from the element's own generator, written to the empty clusters in
       ascending cluster order;
     * seed handling: element b's generator is seeded with the seed the
       per-head call would use (``config.seed`` for the coarse k-means,
       ``config.seed + 10_000 + m`` for PQ subspace m) and has already
       consumed the init ``randperm`` / ``randint`` -- implemented by cloning
       the generator state right after the (shared) init draw, so each
       element continues the exact random stream its per-head counterpart
       would see, including across several re-seeding iterations.
   The early stop of ``kmeans_l2`` (assignments unchanged -> stop before the
   update) is reproduced per element on device (a ``done`` mask), and the
   iteration count follows ``range(max_iter)`` like ``kmeans_l2``.
   One host sync per k-means iteration is needed to learn how many clusters
   each element must re-seed (the randint size is data dependent); the
   counters in ``RESEED_STATS`` record how many were taken.

2. ||x||^2 is passed to the FlashAssign kernel in float32 (the dtype the
   kernel documents).  The kernel computes ``max(xsq + csq - 2 x.c, 0)`` and
   keeps the FIRST index on ties; with a bf16-rounded xsq (as
   ``batched_build._x_sq`` produces for bf16 keys) the rounding error of
   ||x||^2 (up to ~4e-3 for direction-normalised keys) exceeds the true
   distance inside tight key clusters, so several distances clamp to 0 and
   the lowest-numbered centroid wins -> a few huge lists and many empty ones.
   With fp32 xsq the per-row constant cancels in the argmin as intended.

Nothing else differs from ``train_codebooks_batched``: same init indices,
same residual / subspace layout, same kernels for assignment and update, same
return values.  On CPU (unit tests) a torch reference backend that performs
exactly ``kmeans_l2``'s arithmetic replaces the Triton kernels.
"""
from __future__ import annotations

import os

import torch

from pq_hsa.index.kmeans import _nearest_centroids_chunked

RESEED_STATS: dict[str, int] = {
    "calls": 0,             # train_codebooks_batched_reseed calls
    "kmeans_calls": 0,      # batched k-means problems (2 per call)
    "iterations": 0,        # k-means iterations executed (batched)
    "host_syncs": 0,        # one per iteration (re-seed sizes)
    "reseeded_slots": 0,    # (element, cluster) centroids replaced by a random vector
    "reseed_events": 0,     # (element, iteration) pairs that re-seeded >= 1 cluster
    "early_stops": 0,       # elements that hit kmeans_l2's assignment-unchanged stop
}
_ANNOUNCED = False


def batched_reseed_enabled() -> bool:
    return os.environ.get("PQ_HSA_BATCHED_RESEED", "1") != "0"


def seeded_init_indices(
    n: int, k: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """kmeans_l2's init draw.  Returns (row indices [k], generator state after the draw)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    if n >= k:
        idx = torch.randperm(n, generator=g, device=device)[:k]
    else:
        extra = torch.randint(0, n, (k - n,), generator=g, device=device)
        idx = torch.cat([torch.arange(n, device=device), extra], dim=0)
    return idx, g.get_state()


# ---------------------------------------------------------------------------
# Backends: assignment + update.  Both return (new_centroids, counts) from
# ``update``; empty clusters keep the old centroid (flash-kmeans rule) and are
# overwritten by the re-seed step, so the empty-cluster value never survives.
# ---------------------------------------------------------------------------
class TorchReferenceBackend:
    """Per-element loop with exactly kmeans_l2's arithmetic (CPU tests / reference)."""

    name = "torch_reference"

    def xsq(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def assign(self, x: torch.Tensor, c: torch.Tensor, xsq) -> torch.Tensor:
        return torch.stack([_nearest_centroids_chunked(x[b], c[b]) for b in range(x.shape[0])], 0)

    def update(self, x: torch.Tensor, ids: torch.Tensor, c: torch.Tensor, eps: float):
        B, N, D = x.shape
        K = c.shape[1]
        out = torch.empty(B, K, D, device=x.device, dtype=torch.float32)
        counts = torch.empty(B, K, device=x.device, dtype=torch.int64)
        for b in range(B):
            a = ids[b].long()
            new = torch.zeros(K, D, device=x.device, dtype=torch.float32)
            cnt = torch.bincount(a, minlength=K).to(torch.float32)
            new.index_add_(0, a, x[b].float())
            non_empty = cnt > 0
            new[non_empty] /= cnt[non_empty].unsqueeze(1).clamp_min(eps)
            new[~non_empty] = c[b][~non_empty].float()
            out[b] = new
            counts[b] = cnt.to(torch.int64)
        return out, counts


class FlashBackend:
    """flash-kmeans Triton kernels (the ones batched_build uses)."""

    name = "flash"

    def __init__(self, xsq_fp32: bool = True):
        from flash_kmeans.assign_euclid_triton import euclid_assign_triton  # type: ignore
        from flash_kmeans.centroid_update_triton import (  # type: ignore
            triton_centroid_update_sorted_euclid,
        )

        self._assign = euclid_assign_triton
        self._update = triton_centroid_update_sorted_euclid
        self.xsq_fp32 = xsq_fp32

    def xsq(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        dt = torch.float32 if self.xsq_fp32 else x.dtype
        out = torch.empty((B, N), device=x.device, dtype=dt)
        step = 1 << 20
        for i in range(0, N, step):
            blk = x[:, i : i + step]
            if self.xsq_fp32:
                blk = blk.float()
            out[:, i : i + step] = (blk * blk).sum(dim=-1)
        return out

    def assign(self, x: torch.Tensor, c: torch.Tensor, xsq: torch.Tensor) -> torch.Tensor:
        return self._assign(x, c.contiguous(), xsq)

    def update(self, x: torch.Tensor, ids: torch.Tensor, c: torch.Tensor, eps: float):
        del eps
        B, _, _ = x.shape
        K = c.shape[1]
        cnt = torch.zeros((B, K), device=x.device, dtype=torch.int32)
        new = self._update(x, ids, c, centroid_cnts=cnt)
        return new, cnt


def _default_backend(x: torch.Tensor, xsq_fp32: bool):
    return FlashBackend(xsq_fp32=xsq_fp32) if x.is_cuda else TorchReferenceBackend()


def kmeans_batched_reseed(
    x: torch.Tensor,
    init: torch.Tensor,
    *,
    max_iter: int,
    gen_states: list[torch.Tensor],
    backend,
    eps: float = 1e-12,
    reseed: bool = True,
    early_stop: bool = True,
    xsq: torch.Tensor | None = None,
) -> torch.Tensor:
    """x [B,N,D], init [B,K,D], gen_states[b] = element b's generator state after init.

    Per element this is kmeans_l2's loop (without the final re-assignment,
    which the caller does).  Returns centroids [B,K,D] in x.dtype.
    """
    B, N, D = x.shape
    K = init.shape[1]
    dev = x.device
    if len(gen_states) != B:
        raise ValueError(f"need {B} generator states, got {len(gen_states)}")
    RESEED_STATS["kmeans_calls"] += 1
    if xsq is None:
        xsq = backend.xsq(x)
    c = init.to(x.dtype).contiguous()
    prev = torch.zeros(B, N, dtype=torch.long, device=dev)  # kmeans_l2: assignments = zeros
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    gens: dict[int, torch.Generator] = {}
    last_done = 0
    for _ in range(max_iter):
        RESEED_STATS["iterations"] += 1
        ids = backend.assign(x, c, xsq)
        if early_stop:
            # kmeans_l2: `if torch.equal(assignments, next_assignments): break` (before the update)
            done = done | (ids == prev).all(dim=1)
            prev = ids
        new, counts = backend.update(x, ids, c, eps)
        if reseed:
            empty = (counts == 0) & ~done.unsqueeze(1)
            # the one host sync of this iteration: re-seed sizes (+ early-stop count)
            host = torch.cat([empty.sum(dim=1), done.sum().view(1)]).tolist()
            n_empty, n_done = host[:B], host[B]
            RESEED_STATS["host_syncs"] += 1
            for b, ne in enumerate(n_empty):
                if ne == 0:
                    continue
                g = gens.get(b)
                if g is None:
                    g = torch.Generator(device=dev)
                    g.set_state(gen_states[b])
                    gens[b] = g
                rep = torch.randint(0, N, (ne,), generator=g, device=dev)
                new[b][empty[b]] = x[b][rep].to(new.dtype)
                RESEED_STATS["reseeded_slots"] += int(ne)
                RESEED_STATS["reseed_events"] += 1
        new = new.to(x.dtype)
        c = torch.where(done.view(B, 1, 1), c, new) if early_stop else new
        if early_stop and reseed:
            last_done = n_done
    if early_stop and reseed and max_iter > 0:
        RESEED_STATS["early_stops"] += int(last_done)
    return c.contiguous()


@torch.no_grad()
def train_codebooks_batched_reseed(
    keys: torch.Tensor,
    config,
    *,
    return_assignments: bool = False,
    reseed: bool = True,
    xsq_fp32: bool = True,
    backend=None,
):
    """Drop-in for ``batched_build.train_codebooks_batched`` (same inputs/outputs).

    ``reseed`` / ``xsq_fp32`` exist for ablations only; the env gate uses the
    defaults (both on).
    """
    global _ANNOUNCED
    if keys.ndim != 3:
        raise ValueError(f"keys must be [H, N, D], got {tuple(keys.shape)}")
    if config.rotation != "none":
        raise NotImplementedError("batched build supports rotation='none' only")
    RESEED_STATS["calls"] += 1
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(
            f"[batched-reseed] batched build with per-element empty-cluster re-seeding "
            f"(reseed={reseed}, xsq_fp32={xsq_fp32}, device={keys.device})",
            flush=True,
        )
    H, N, D = keys.shape
    M = config.num_subspaces
    C = 1 << config.num_bits
    if D % M != 0:
        raise ValueError(f"dim={D} must be divisible by num_subspaces={M}")
    d = D // M
    K = min(config.num_lists, N)
    dev = keys.device
    be = backend if backend is not None else _default_backend(keys, xsq_fp32)

    x = keys.contiguous()
    if config.direction_normalize:
        norms = torch.linalg.vector_norm(x.float(), dim=-1).to(x.dtype).clamp_min(config.eps)
        x = x / norms.unsqueeze(-1)

    # ---- coarse k-means: every head uses config.seed (as IVFPQIndex.build does) ----
    perm, st = seeded_init_indices(N, K, config.seed, dev)
    init = x[:, perm, :]
    xsq = be.xsq(x)
    coarse = kmeans_batched_reseed(
        x, init, max_iter=config.coarse_max_iter, gen_states=[st] * H,
        backend=be, eps=config.eps, reseed=reseed, xsq=xsq,
    )
    ids = be.assign(x, coarse, xsq)  # kmeans_l2's final assignment
    del xsq
    if config.residual:
        gathered = torch.gather(coarse, 1, ids.long().unsqueeze(-1).expand(-1, -1, D))
        resid = x - gathered
    else:
        resid = x

    # ---- PQ: element b = h*M + m uses seed config.seed + 10_000 + m ----
    sub = resid.reshape(H, N, M, d).permute(0, 2, 1, 3).reshape(H * M, N, d).contiguous()
    pq_seed = config.seed + 10_000
    init_pq = torch.empty(H * M, C, d, device=dev, dtype=sub.dtype)
    states: list[torch.Tensor | None] = [None] * (H * M)
    for m in range(M):
        pm, stm = seeded_init_indices(N, C, pq_seed + m, dev)
        init_pq[m::M] = sub[m::M][:, pm, :]
        for h in range(H):
            states[h * M + m] = stm
    xsq_sub = be.xsq(sub)
    cb = kmeans_batched_reseed(
        sub, init_pq, max_iter=config.pq_max_iter, gen_states=states,  # type: ignore[arg-type]
        backend=be, eps=config.eps, reseed=reseed, xsq=xsq_sub,
    )
    codebooks = cb.reshape(H, M, C, d).to(keys.dtype)
    if not return_assignments:
        return coarse.to(keys.dtype), codebooks
    codes = be.assign(sub, cb.to(sub.dtype).contiguous(), xsq_sub)  # [H*M, N]
    codes = codes.reshape(H, M, N).permute(0, 2, 1).contiguous().to(torch.uint8)  # [H, N, M]
    return coarse.to(keys.dtype), codebooks, ids.long(), codes
