from __future__ import annotations

from dataclasses import dataclass

import torch

from pq_hsa.index.kmeans import kmeans_l2


@dataclass(slots=True)
class ProductQuantizerConfig:
    """Configuration for product quantization of key vectors or residuals."""

    num_subspaces: int = 16
    num_bits: int = 8
    max_iter: int = 25
    seed: int = 0
    eps: float = 1e-12

    @property
    def num_codes(self) -> int:
        return 1 << self.num_bits


class ProductQuantizer:
    """Product quantizer whose LUT scores are approximate QK logits.

    For a query q and encoded vector codes[i, m], the lookup table
    LUT[m, c] = dot(q_m, codebook[m, c]) gives

        score_pq(q, i) = sum_m LUT[m, codes[i, m]]

    which is exactly q dot decode(codes[i]) and therefore can be used as an
    approximate attention logit before optional exact reranking.
    """

    def __init__(self, config: ProductQuantizerConfig):
        self.config = config
        self.codebooks: torch.Tensor | None = None
        self.dim: int | None = None
        self.subdim: int | None = None

    @property
    def is_trained(self) -> bool:
        return self.codebooks is not None

    def train(
        self,
        vectors: torch.Tensor,
        *,
        initial_codebooks: torch.Tensor | None = None,
    ) -> "ProductQuantizer":
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be [N, D], got {tuple(vectors.shape)}")
        num_vectors, dim = vectors.shape
        if num_vectors == 0:
            raise ValueError("cannot train PQ with zero vectors")
        if dim % self.config.num_subspaces != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_subspaces={self.config.num_subspaces}"
            )

        self.dim = dim
        self.subdim = dim // self.config.num_subspaces
        chunks = vectors.reshape(num_vectors, self.config.num_subspaces, self.subdim)

        if initial_codebooks is not None:
            expected_shape = (
                self.config.num_subspaces,
                self.config.num_codes,
                self.subdim,
            )
            if tuple(initial_codebooks.shape) != expected_shape:
                raise ValueError(
                    f"initial_codebooks must have shape {expected_shape}, "
                    f"got {tuple(initial_codebooks.shape)}"
                )
            initial_codebooks = initial_codebooks.to(device=vectors.device, dtype=vectors.dtype)

        codebooks = []
        for subspace in range(self.config.num_subspaces):
            initial_centroids = None
            if initial_codebooks is not None:
                initial_centroids = initial_codebooks[subspace]
            centroids, _ = kmeans_l2(
                chunks[:, subspace, :].contiguous(),
                self.config.num_codes,
                max_iter=self.config.max_iter,
                seed=self.config.seed + subspace,
                eps=self.config.eps,
                initial_centroids=initial_centroids,
            )
            codebooks.append(centroids)
        self.codebooks = torch.stack(codebooks, dim=0)
        return self

    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        self._check_trained()
        assert self.codebooks is not None
        assert self.subdim is not None

        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"vectors must be [N, {self.dim}], got {tuple(vectors.shape)}")

        num_vectors = vectors.shape[0]
        chunks = vectors.reshape(num_vectors, self.config.num_subspaces, self.subdim)
        codes = []
        for subspace in range(self.config.num_subspaces):
            distances = torch.cdist(
                chunks[:, subspace, :].float(),
                self.codebooks[subspace].float(),
                p=2,
            )
            codes.append(distances.argmin(dim=1))
        return torch.stack(codes, dim=1)

    def update_codebooks_ema(
        self,
        vectors: torch.Tensor,
        *,
        learning_rate: float,
    ) -> torch.Tensor:
        """Move assigned PQ codewords toward a mini-batch mean.

        Returns the mini-batch code assignments computed before the update.
        Existing codes must be re-encoded after this method because moving a
        codebook changes what stored code ids reconstruct to.
        """

        self._check_trained()
        assert self.codebooks is not None
        assert self.subdim is not None

        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"vectors must be [N, {self.dim}], got {tuple(vectors.shape)}")
        if vectors.shape[0] == 0:
            return torch.empty(
                0,
                self.config.num_subspaces,
                device=self.codebooks.device,
                dtype=torch.long,
            )

        num_vectors = vectors.shape[0]
        chunks = vectors.reshape(num_vectors, self.config.num_subspaces, self.subdim)
        assignments = []
        updated_codebooks = self.codebooks.clone()
        for subspace in range(self.config.num_subspaces):
            distances = torch.cdist(
                chunks[:, subspace, :].float(),
                self.codebooks[subspace].float(),
                p=2,
            )
            sub_assignments = distances.argmin(dim=1)
            assignments.append(sub_assignments)

            sums = torch.zeros_like(self.codebooks[subspace])
            counts = torch.bincount(
                sub_assignments,
                minlength=self.config.num_codes,
            ).to(vectors.dtype)
            sums.index_add_(0, sub_assignments, chunks[:, subspace, :])
            non_empty = counts > 0
            means = sums[non_empty] / counts[non_empty].unsqueeze(1).clamp_min(self.config.eps)
            updated_codebooks[subspace, non_empty] = (
                (1.0 - learning_rate) * self.codebooks[subspace, non_empty]
                + learning_rate * means
            )

        self.codebooks = updated_codebooks
        return torch.stack(assignments, dim=1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self._check_trained()
        assert self.codebooks is not None
        assert self.subdim is not None

        if codes.ndim != 2 or codes.shape[1] != self.config.num_subspaces:
            raise ValueError(
                f"codes must be [N, {self.config.num_subspaces}], got {tuple(codes.shape)}"
            )

        parts = []
        for subspace in range(self.config.num_subspaces):
            parts.append(self.codebooks[subspace, codes[:, subspace]])
        return torch.cat(parts, dim=1)

    def compute_lut(self, query: torch.Tensor) -> torch.Tensor:
        """Return query-codebook dot-product LUT.

        Shape:
        - query [D] -> LUT [M, K]
        - query [B, D] -> LUT [B, M, K]
        """

        self._check_trained()
        assert self.codebooks is not None
        assert self.subdim is not None

        if query.ndim == 1:
            if query.shape[0] != self.dim:
                raise ValueError(f"query must be [{self.dim}], got {tuple(query.shape)}")
            q = query.reshape(self.config.num_subspaces, self.subdim)
            return torch.bmm(self.codebooks, q.unsqueeze(2)).squeeze(2)

        if query.ndim == 2:
            if query.shape[1] != self.dim:
                raise ValueError(f"query must be [B, {self.dim}], got {tuple(query.shape)}")
            q = query.reshape(query.shape[0], self.config.num_subspaces, self.subdim)
            return torch.bmm(
                q.permute(1, 0, 2),
                self.codebooks.transpose(1, 2),
            ).permute(1, 0, 2)

        raise ValueError(f"query must be [D] or [B, D], got {tuple(query.shape)}")

    def score_from_lut(self, codes: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
        """Gather PQ scores from a LUT.

        Shape:
        - codes [N, M], LUT [M, K] -> scores [N]
        - codes [N, M], LUT [B, M, K] -> scores [B, N]
        """

        self._check_trained()
        if codes.ndim != 2 or codes.shape[1] != self.config.num_subspaces:
            raise ValueError(
                f"codes must be [N, {self.config.num_subspaces}], got {tuple(codes.shape)}"
            )

        if lut.ndim == 2:
            if lut.shape[:2] != (self.config.num_subspaces, self.config.num_codes):
                raise ValueError(f"unexpected LUT shape {tuple(lut.shape)}")
            return lut.gather(1, codes.T).sum(dim=0)

        if lut.ndim == 3:
            if lut.shape[1:] != (self.config.num_subspaces, self.config.num_codes):
                raise ValueError(f"unexpected LUT shape {tuple(lut.shape)}")
            index = codes.T.unsqueeze(0).expand(lut.shape[0], -1, -1)
            return lut.gather(2, index).sum(dim=1)

        raise ValueError(f"lut must be [M, K] or [B, M, K], got {tuple(lut.shape)}")

    def approximate_scores(self, query: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        return self.score_from_lut(codes, self.compute_lut(query))

    def _check_trained(self) -> None:
        if self.codebooks is None:
            raise RuntimeError("ProductQuantizer must be trained first")
