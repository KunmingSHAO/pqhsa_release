from __future__ import annotations

import os

import torch

# Opt-in Flash-KMeans backend (arXiv 2603.09229, svg-project/flash-kmeans).
# Gated entirely behind PQ_HSA_FLASH_KMEANS=1; when unset (default) this module's
# behavior is byte-for-byte identical to before this change -- no import is even
# attempted.
_FLASH_KMEANS_MODULE = None  # lazy-imported, cached after first successful import


def _flash_kmeans_enabled() -> bool:
    return os.environ.get("PQ_HSA_FLASH_KMEANS", "0") == "1"


def _load_flash_kmeans():
    global _FLASH_KMEANS_MODULE
    if _FLASH_KMEANS_MODULE is None:
        try:
            import flash_kmeans  # type: ignore
        except Exception as exc:  # pragma: no cover - opt-in path only
            raise RuntimeError(
                "PQ_HSA_FLASH_KMEANS=1 was set but `import flash_kmeans` failed. "
                "Install it in the environment "
                "or unset PQ_HSA_FLASH_KMEANS to use the default torch k-means."
            ) from exc
        _FLASH_KMEANS_MODULE = flash_kmeans
    return _FLASH_KMEANS_MODULE


def _init_centroids_seeded(
    vectors: torch.Tensor,
    num_clusters: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Same init rule as the default kmeans_l2 below: uniform sample-without-
    replacement when possible, so Flash-KMeans and the default path start from
    an identical initialization for a given seed."""

    num_vectors = vectors.shape[0]
    if num_vectors >= num_clusters:
        perm = torch.randperm(num_vectors, generator=generator, device=vectors.device)
        return vectors[perm[:num_clusters]].clone()
    extra = torch.randint(
        0,
        num_vectors,
        (num_clusters - num_vectors,),
        generator=generator,
        device=vectors.device,
    )
    return torch.cat([vectors, vectors[extra]], dim=0).clone()


def _kmeans_l2_flash(
    vectors: torch.Tensor,
    num_clusters: int,
    *,
    max_iter: int,
    seed: int,
    eps: float,
    initial_centroids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flash-KMeans backed replacement for kmeans_l2 (opt-in, see PQ_HSA_FLASH_KMEANS).

    Same K, same max_iter (tol=0.0 forces exactly max_iter iterations, matching
    the common case here where max_iter is 1-2 and the default path's early
    assignment-equality stop essentially never triggers), and the same seeded
    initial centroids as the default path -- so any quality delta measured in
    is attributable to the FlashAssign/update kernels, not to init.
    """

    del eps  # flash-kmeans has no empty-cluster epsilon knob; unused, kept for signature parity
    fk = _load_flash_kmeans()

    num_vectors, dim = vectors.shape
    generator = torch.Generator(device=vectors.device)
    generator.manual_seed(seed)

    if initial_centroids is not None:
        init = initial_centroids.to(device=vectors.device, dtype=vectors.dtype).clone()
    else:
        init = _init_centroids_seeded(vectors, num_clusters, generator)

    x_b = vectors.unsqueeze(0)
    init_b = init.unsqueeze(0)
    _cluster_ids_b, centroids_b, _niters = fk.batch_kmeans_Euclid(
        x_b,
        num_clusters,
        max_iters=max(max_iter, 1),
        tol=0.0,
        init_centroids=init_b,
        verbose=False,
    )
    centroids = centroids_b.squeeze(0).to(vectors.dtype)
    # flash-kmeans' own cluster_ids come from the E-step *preceding* the final
    # M-step (i.e. assigned against the second-to-last centroids, not the
    # returned ones). The default path below always returns assignments
    # freshly computed against its final centroids, so recompute here with
    # the same helper to keep (centroids, assignments) consistent and make
    # the two backends apples-to-apples for downstream code (e.g. IVF list
    # ids) that consumes both.
    assignments = _nearest_centroids_chunked(vectors, centroids)
    return centroids, assignments


def _nearest_centroids_chunked(
    vectors: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk: int = 16384,
) -> torch.Tensor:
    """Return argmin over centroids for each vector, without materializing the
    full [N, K] distance matrix.

    At 256K vectors x 2048 clusters, a single torch.cdist allocates an
    [N, K] float32 matrix (~2 GiB) that OOMs. Chunking over N keeps the peak
    at [chunk, K]. Uses ||v-c||^2 = ||v||^2 - 2 v.c + ||c||^2; argmin is
    invariant to the ||v||^2 term, so we minimize (||c||^2 - 2 v.c).
    """
    cent_f = centroids.float()
    cent_sq = (cent_f * cent_f).sum(dim=1)  # [K]
    out = torch.empty(vectors.shape[0], dtype=torch.long, device=vectors.device)
    for start in range(0, vectors.shape[0], chunk):
        end = min(vectors.shape[0], start + chunk)
        v = vectors[start:end].float()
        # scores[i, k] = ||c_k||^2 - 2 v_i. c_k  (smaller = nearer)
        scores = cent_sq.unsqueeze(0) - 2.0 * v.matmul(cent_f.t())
        out[start:end] = scores.argmin(dim=1)
    return out


def kmeans_l2(
    vectors: torch.Tensor,
    num_clusters: int,
    *,
    max_iter: int = 25,
    seed: int = 0,
    eps: float = 1e-12,
    initial_centroids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a small deterministic L2 k-means routine in torch.

    This is intentionally simple: it gives the prototype a dependency-free
    training path and can later be replaced by FAISS or a CUDA kernel.
    """

    if vectors.ndim != 2:
        raise ValueError(f"vectors must be [N, D], got {tuple(vectors.shape)}")
    if vectors.shape[0] == 0:
        raise ValueError("cannot train k-means with zero vectors")
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")

    if _flash_kmeans_enabled() and vectors.is_cuda:
        return _kmeans_l2_flash(
            vectors,
            num_clusters,
            max_iter=max_iter,
            seed=seed,
            eps=eps,
            initial_centroids=initial_centroids,
        )

    num_vectors, dim = vectors.shape
    generator = torch.Generator(device=vectors.device)
    generator.manual_seed(seed)

    if initial_centroids is not None:
        if initial_centroids.ndim != 2:
            raise ValueError(
                f"initial_centroids must be [K, D], got {tuple(initial_centroids.shape)}"
            )
        if tuple(initial_centroids.shape) != (num_clusters, dim):
            raise ValueError(
                "initial_centroids must have shape "
                f"({num_clusters}, {dim}), got {tuple(initial_centroids.shape)}"
            )
        centroids = initial_centroids.to(device=vectors.device, dtype=vectors.dtype).clone()
    elif num_vectors >= num_clusters:
        perm = torch.randperm(num_vectors, generator=generator, device=vectors.device)
        centroids = vectors[perm[:num_clusters]].clone()
    else:
        extra = torch.randint(
            0,
            num_vectors,
            (num_clusters - num_vectors,),
            generator=generator,
            device=vectors.device,
        )
        centroids = torch.cat([vectors, vectors[extra]], dim=0).clone()

    assignments = torch.zeros(num_vectors, dtype=torch.long, device=vectors.device)
    for _ in range(max_iter):
        next_assignments = _nearest_centroids_chunked(vectors, centroids)
        if torch.equal(assignments, next_assignments):
            assignments = next_assignments
            break
        assignments = next_assignments

        new_centroids = torch.zeros(
            num_clusters,
            dim,
            device=vectors.device,
            dtype=torch.float32,
        )
        counts = torch.bincount(assignments, minlength=num_clusters).to(torch.float32)
        new_centroids.index_add_(0, assignments, vectors.float())

        non_empty = counts > 0
        new_centroids[non_empty] /= counts[non_empty].unsqueeze(1).clamp_min(eps)

        if (~non_empty).any():
            replacement = torch.randint(
                0,
                num_vectors,
                ((~non_empty).sum().item(),),
                generator=generator,
                device=vectors.device,
            )
            new_centroids[~non_empty] = vectors[replacement].float()

        centroids = new_centroids.to(vectors.dtype)

    assignments = _nearest_centroids_chunked(vectors, centroids)
    return centroids, assignments
