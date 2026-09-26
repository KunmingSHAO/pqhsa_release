from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import torch

from pq_hsa.index.kmeans import kmeans_l2
from pq_hsa.index.packing import pack_4bit_codes, unpack_4bit_codes
from pq_hsa.index.pq import ProductQuantizer, ProductQuantizerConfig
from pq_hsa.kernels import score_packed_4bit_lut
from pq_hsa.kernels.device import ALLOWED_KERNEL_BACKENDS, GPU_LUT_BACKENDS
from pq_hsa.kernels.triton_lut_scan import (
    is_triton_available,
    score_packed_4bit_lut_batched_list_bias_triton,
)


def compact_id_dtype(max_inclusive: int) -> torch.dtype:
    """Smallest unsigned dtype that can hold ``0 .. max_inclusive``.

    PyTorch cannot use uint16/uint32 as index tensors (bincount / advanced
    indexing / index_add). Callers must run :func:`as_index` at those sites.
    Triton kernels load the stored dtype and cast to int64 after the load.
    """

    if max_inclusive < 0:
        raise ValueError("max_inclusive must be non-negative")
    if max_inclusive <= 0xFFFF:
        return torch.uint16
    if max_inclusive <= 0xFFFFFFFF:
        return torch.uint32
    return torch.int64


def csr_index_dtype(num_vectors: int) -> torch.dtype:
    """CSR offsets/indices are used as PyTorch indices, so they stay signed.

    ``num_vectors`` is the last offset value. 128K tokens fit in int32.
    """

    if num_vectors < 0:
        raise ValueError("num_vectors must be non-negative")
    if num_vectors <= 0x7FFFFFFF:
        return torch.int32
    return torch.int64


def as_index(ids: torch.Tensor) -> torch.Tensor:
    """Promote compact ids to int64 for indexing / bincount / index_add.

    uint16/uint32 are not valid PyTorch index dtypes. Promoting to int64
    (the historical stored dtype) keeps atomic ``index_add_`` accumulation
    order identical to the pre-compression path.
    """

    if ids.dtype == torch.int64:
        return ids
    return ids.to(torch.int64)


def gather_ids(ids: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
    """Select stored list ids by token index.

    CPU cannot index into uint16/uint32 tensors (``index_cpu`` is missing).
    Promote first on CPU only so the GPU resident path stays a direct gather
    of the compact dtype.
    """

    if indices is None:
        return ids
    if ids.dtype in (torch.uint16, torch.uint32) and ids.device.type == "cpu":
        return as_index(ids)[indices]
    return ids[indices]


def store_ids(ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if ids.dtype == dtype:
        return ids
    return ids.to(dtype)


@dataclass(slots=True)
class IVFPQConfig:
    """Configuration for inverted-file product quantization."""

    num_lists: int = 256
    nprobe: int = 8
    num_subspaces: int = 16
    num_bits: int = 8
    residual: bool = True
    coarse_max_iter: int = 30
    pq_max_iter: int = 25
    pack_codes: bool = True
    topk_block_size: int | None = None
    kernel_backend: str = "auto"
    rotation: str = "none"
    direction_normalize: bool = False
    online_codebook_lr: float = 0.05
    seed: int = 0
    eps: float = 1e-12


@dataclass(slots=True)
class SearchResult:
    indices: torch.Tensor
    approx_scores: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_scores: torch.Tensor
    probed_lists: torch.Tensor


@dataclass(slots=True)
class IVFPQMemoryStats:
    coarse_centroid_bytes: int
    list_id_bytes: int
    inverted_list_bytes: int
    code_bytes: int
    pq_codebook_bytes: int
    rotation_matrix_bytes: int = 0
    key_norm_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return (
            self.coarse_centroid_bytes
            + self.list_id_bytes
            + self.inverted_list_bytes
            + self.code_bytes
            + self.pq_codebook_bytes
            + self.rotation_matrix_bytes
            + self.key_norm_bytes
        )


class IVFPQIndex:
    """IVF-PQ index for MIPS-style KV-cache retrieval.

    If residual=True, keys are represented as

        k ~= centroid[list_id] + pq_decode(residual_code)

    and the approximate attention score is

        q dot centroid[list_id] + sum_m LUT_m[code_m].

    This keeps the index score in the same units as the raw QK attention logit,
    making it directly reusable by sparse attention or exact reranking.
    """

    def __init__(self, config: IVFPQConfig, *, profile: bool = False):
        if config.topk_block_size is not None and config.topk_block_size <= 0:
            raise ValueError("topk_block_size must be positive when provided")
        if config.kernel_backend not in ALLOWED_KERNEL_BACKENDS:
            raise ValueError("kernel_backend must be 'auto', 'torch', 'triton', or 'h20'")
        if config.rotation not in {"none", "random_orthogonal"}:
            raise ValueError("rotation must be 'none' or 'random_orthogonal'")
        if not (0.0 < config.online_codebook_lr <= 1.0):
            raise ValueError("online_codebook_lr must be in (0, 1]")
        self.config = config
        self.coarse_centroids: torch.Tensor | None = None
        self.list_ids: torch.Tensor | None = None
        self.inverted_list_offsets: torch.Tensor | None = None
        self.inverted_list_indices: torch.Tensor | None = None
        self._codes: torch.Tensor | None = None
        self.packed_codes: torch.Tensor | None = None
        self.pq: ProductQuantizer | None = None
        self.rotation_matrix: torch.Tensor | None = None
        self.key_norms: torch.Tensor | None = None
        self.dim: int | None = None
        self.num_vectors: int = 0
        self.profile_enabled = bool(profile)
        self.profile_stats: dict[str, float] = {}

    @property
    def is_built(self) -> bool:
        return self._codes is not None or self.packed_codes is not None

    @property
    def inverted_lists(self) -> tuple[torch.Tensor, ...]:
        """Return per-list member views for inspection and tests."""

        self._ensure_inverted_lists()
        if self.inverted_list_offsets is None or self.inverted_list_indices is None:
            return ()
        offsets = self.inverted_list_offsets.tolist()
        return tuple(
            self.inverted_list_indices[offsets[list_id] : offsets[list_id + 1]]
            for list_id in range(len(offsets) - 1)
        )

    @inverted_lists.setter
    def inverted_lists(self, members: tuple[torch.Tensor, ...]) -> None:
        if len(members) == 0:
            device = self.list_ids.device if self.list_ids is not None else torch.device("cpu")
            self.inverted_list_offsets = torch.zeros(1, device=device, dtype=torch.long)
            self.inverted_list_indices = torch.empty(0, device=device, dtype=torch.long)
            return

        device = members[0].device
        lengths = torch.tensor(
            [member.numel() for member in members],
            device=device,
            dtype=torch.long,
        )
        offsets = torch.empty(len(members) + 1, device=device, dtype=torch.long)
        offsets[0] = 0
        offsets[1:] = torch.cumsum(lengths, dim=0)
        non_empty_members = [
            member.to(device=device, dtype=torch.long) for member in members if member.numel() > 0
        ]
        indices = (
            torch.cat(non_empty_members, dim=0)
            if non_empty_members
            else torch.empty(0, device=device, dtype=torch.long)
        )
        self.inverted_list_offsets = offsets
        self.inverted_list_indices = indices

    @property
    def stores_packed_codes(self) -> bool:
        return self.config.pack_codes and self.config.num_bits == 4

    def _list_id_dtype(self) -> torch.dtype:
        assert self.coarse_centroids is not None
        return compact_id_dtype(max(int(self.coarse_centroids.shape[0]) - 1, 0))

    def _pack_list_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return store_ids(ids, self._list_id_dtype())

    def _release_inverted_lists(self) -> None:
        self.inverted_list_offsets = None
        self.inverted_list_indices = None

    def _ensure_inverted_lists(self) -> None:
        if self.inverted_list_offsets is None or self.inverted_list_indices is None:
            self._rebuild_inverted_lists()

    @property
    def codes(self) -> torch.Tensor | None:
        if self._codes is not None:
            return self._codes
        if self.packed_codes is None:
            return None
        return unpack_4bit_codes(self.packed_codes, self.config.num_subspaces)

    def reset_profile_stats(self) -> None:
        self.profile_stats.clear()

    def _profile_start(self, device: torch.device | str) -> float | None:
        if not self.profile_enabled:
            return None
        current_device = torch.device(device)
        if current_device.type == "cuda":
            torch.cuda.synchronize(current_device)
        return time.perf_counter()

    def _profile_stop(
        self,
        name: str,
        start: float | None,
        device: torch.device | str,
    ) -> None:
        if start is None:
            return
        current_device = torch.device(device)
        if current_device.type == "cuda":
            torch.cuda.synchronize(current_device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.profile_stats[f"{name}_ms"] = self.profile_stats.get(f"{name}_ms", 0.0) + elapsed_ms
        self.profile_stats[f"{name}_calls"] = (
            self.profile_stats.get(f"{name}_calls", 0.0) + 1.0
        )

    def build(
        self,
        keys: torch.Tensor,
        *,
        initial_coarse_centroids: torch.Tensor | None = None,
        initial_pq_codebooks: torch.Tensor | None = None,
    ) -> "IVFPQIndex":
        total_start = self._profile_start(keys.device)
        if keys.ndim != 2:
            raise ValueError(f"keys must be [N, D], got {tuple(keys.shape)}")
        if keys.shape[0] == 0:
            raise ValueError("cannot build IVF-PQ index with zero keys")
        if keys.shape[1] % self.config.num_subspaces != 0:
            raise ValueError(
                f"dim={keys.shape[1]} must be divisible by num_subspaces={self.config.num_subspaces}"
            )

        self.dim = keys.shape[1]
        profile_start = self._profile_start(keys.device)
        self._ensure_rotation_matrix(keys)
        transformed_keys, key_norms = self._project_keys(keys)
        self.key_norms = key_norms
        self.num_vectors = keys.shape[0]
        self._profile_stop("index_build_project_keys", profile_start, keys.device)

        num_lists = min(self.config.num_lists, transformed_keys.shape[0])
        profile_start = self._profile_start(keys.device)
        initial_coarse_centroids = self._prepare_initial_coarse_centroids(
            transformed_keys,
            num_lists,
            initial_coarse_centroids,
        )
        self._profile_stop("index_build_prepare_coarse_init", profile_start, keys.device)
        profile_start = self._profile_start(keys.device)
        self.coarse_centroids, raw_list_ids = kmeans_l2(
            transformed_keys,
            num_lists,
            max_iter=self.config.coarse_max_iter,
            seed=self.config.seed,
            eps=self.config.eps,
            initial_centroids=initial_coarse_centroids,
        )
        self.list_ids = self._pack_list_ids(raw_list_ids)
        self._profile_stop("index_build_coarse_kmeans", profile_start, keys.device)
        # CSR is kept: the hybrid centroid path uses it for list_exp_sums_triton.
        # Dropping it falls back to scatter_add and is not bit-identical.
        profile_start = self._profile_start(keys.device)
        self._rebuild_inverted_lists()
        self._profile_stop("index_build_inverted_lists", profile_start, keys.device)

        profile_start = self._profile_start(keys.device)
        residual_input = self._vectors_to_encode(transformed_keys)
        self._profile_stop("index_build_residual_vectors", profile_start, keys.device)
        pq_config = ProductQuantizerConfig(
            num_subspaces=self.config.num_subspaces,
            num_bits=self.config.num_bits,
            max_iter=self.config.pq_max_iter,
            seed=self.config.seed + 10_000,
            eps=self.config.eps,
        )
        profile_start = self._profile_start(keys.device)
        self.pq = ProductQuantizer(pq_config).train(
            residual_input,
            initial_codebooks=initial_pq_codebooks,
        )
        self._profile_stop("index_build_pq_train", profile_start, keys.device)
        profile_start = self._profile_start(keys.device)
        codes = self.pq.encode(residual_input)
        self._profile_stop("index_build_pq_encode", profile_start, keys.device)
        profile_start = self._profile_start(keys.device)
        self._store_codes(codes)
        self._profile_stop("index_build_store_codes", profile_start, keys.device)
        self._profile_stop("index_build_total", total_start, keys.device)
        return self

    def build_with_codebooks(
        self,
        keys: torch.Tensor,
        *,
        coarse_centroids: torch.Tensor,
        pq_codebooks: torch.Tensor,
        list_ids: torch.Tensor | None = None,
        codes: torch.Tensor | None = None,
    ) -> "IVFPQIndex":
        """Build per-vector metadata using fixed shared IVF/PQ codebooks.

        ``list_ids`` / ``codes`` (opt-in batched build) may be
        supplied precomputed; when None (default) behaviour is unchanged.
        """

        total_start = self._profile_start(keys.device)
        if keys.ndim != 2:
            raise ValueError(f"keys must be [N, D], got {tuple(keys.shape)}")
        if keys.shape[0] == 0:
            raise ValueError("cannot build IVF-PQ index with zero keys")
        if keys.shape[1] % self.config.num_subspaces != 0:
            raise ValueError(
                f"dim={keys.shape[1]} must be divisible by num_subspaces={self.config.num_subspaces}"
            )
        if coarse_centroids.ndim != 2 or coarse_centroids.shape[1] != keys.shape[1]:
            raise ValueError(
                "coarse_centroids must be [num_lists, dim], got "
                f"{tuple(coarse_centroids.shape)} for dim={keys.shape[1]}"
            )
        expected_pq_shape = (
            self.config.num_subspaces,
            1 << self.config.num_bits,
            keys.shape[1] // self.config.num_subspaces,
        )
        if tuple(pq_codebooks.shape) != expected_pq_shape:
            raise ValueError(
                f"pq_codebooks must have shape {expected_pq_shape}, got {tuple(pq_codebooks.shape)}"
            )

        self.dim = keys.shape[1]
        profile_start = self._profile_start(keys.device)
        self._ensure_rotation_matrix(keys)
        transformed_keys, key_norms = self._project_keys(keys)
        self.key_norms = key_norms
        self.num_vectors = keys.shape[0]
        self._profile_stop("index_build_shared_project_keys", profile_start, keys.device)

        self.coarse_centroids = coarse_centroids.to(device=keys.device, dtype=keys.dtype)
        profile_start = self._profile_start(keys.device)
        if list_ids is not None:
            self.list_ids = self._pack_list_ids(list_ids.to(device=keys.device))
        else:
            self.list_ids = self._assign_lists_transformed(transformed_keys)
        self._profile_stop("index_build_shared_assign_lists", profile_start, keys.device)
        profile_start = self._profile_start(keys.device)
        self._rebuild_inverted_lists()
        self._profile_stop("index_build_shared_inverted_lists", profile_start, keys.device)

        pq_config = ProductQuantizerConfig(
            num_subspaces=self.config.num_subspaces,
            num_bits=self.config.num_bits,
            max_iter=self.config.pq_max_iter,
            seed=self.config.seed + 10_000,
            eps=self.config.eps,
        )
        self.pq = ProductQuantizer(pq_config)
        self.pq.codebooks = pq_codebooks.to(device=keys.device, dtype=keys.dtype)
        self.pq.dim = keys.shape[1]
        self.pq.subdim = expected_pq_shape[2]

        if codes is None:
            profile_start = self._profile_start(keys.device)
            residual_input = self._vectors_to_encode(transformed_keys)
            self._profile_stop("index_build_shared_residual_vectors", profile_start, keys.device)
            profile_start = self._profile_start(keys.device)
            codes = self.pq.encode(residual_input)
            self._profile_stop("index_build_shared_pq_encode", profile_start, keys.device)
        else:
            codes = codes.to(device=keys.device)
        profile_start = self._profile_start(keys.device)
        self._store_codes(codes)
        self._profile_stop("index_build_shared_store_codes", profile_start, keys.device)
        self._profile_stop("index_build_shared_total", total_start, keys.device)
        return self

    def rebuild(self, keys: torch.Tensor, *, warm_start: bool = False) -> "IVFPQIndex":
        """Rebuild the index over ``keys``, optionally initialized from current codebooks."""

        initial_coarse_centroids = self.coarse_centroids if warm_start else None
        initial_pq_codebooks = None
        if warm_start and self.pq is not None:
            initial_pq_codebooks = self.pq.codebooks
        return self.build(
            keys,
            initial_coarse_centroids=initial_coarse_centroids,
            initial_pq_codebooks=initial_pq_codebooks,
        )

    def state_dict(self) -> dict[str, object]:
        """Return a serializable snapshot of trained IVF-PQ metadata."""

        self._check_built()
        assert self.coarse_centroids is not None
        assert self.list_ids is not None
        assert self.pq is not None
        assert self.pq.codebooks is not None

        return {
            "config": asdict(self.config),
            "coarse_centroids": self.coarse_centroids.detach().clone(),
            "list_ids": self.list_ids.detach().clone(),
            "codes": None if self._codes is None else self._codes.detach().clone(),
            "packed_codes": None
            if self.packed_codes is None
            else self.packed_codes.detach().clone(),
            "pq_codebooks": self.pq.codebooks.detach().clone(),
            "rotation_matrix": None
            if self.rotation_matrix is None
            else self.rotation_matrix.detach().clone(),
            "key_norms": None if self.key_norms is None else self.key_norms.detach().clone(),
            "dim": self.dim,
            "num_vectors": self.num_vectors,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, object],
        *,
        config: IVFPQConfig | None = None,
    ) -> "IVFPQIndex":
        state_config = _config_from_state(state)
        index = cls(state_config if config is None else config)
        index.load_state_dict(state)
        return index

    def load_state_dict(self, state: dict[str, object]) -> "IVFPQIndex":
        """Load an IVF-PQ metadata snapshot into this index."""

        state_config = _config_from_state(state)
        if asdict(self.config) != asdict(state_config):
            raise ValueError("state config does not match this IVFPQIndex config")

        coarse_centroids = _required_tensor(state, "coarse_centroids")
        list_ids = _required_tensor(state, "list_ids").to(device=coarse_centroids.device)
        pq_codebooks = _required_tensor(state, "pq_codebooks").to(
            device=coarse_centroids.device,
            dtype=coarse_centroids.dtype,
        )
        codes = state.get("codes")
        packed_codes = state.get("packed_codes")
        rotation_matrix = state.get("rotation_matrix")
        key_norms = state.get("key_norms")
        dim = int(state["dim"])
        num_vectors = int(state["num_vectors"])

        if coarse_centroids.ndim != 2 or coarse_centroids.shape[1] != dim:
            raise ValueError("coarse_centroids shape does not match saved dim")
        if list_ids.ndim != 1 or list_ids.shape[0] != num_vectors:
            raise ValueError("list_ids shape does not match saved num_vectors")
        if pq_codebooks.ndim != 3 or pq_codebooks.shape[:2] != (
            self.config.num_subspaces,
            1 << self.config.num_bits,
        ):
            raise ValueError("pq_codebooks shape does not match index config")
        if dim != self.config.num_subspaces * pq_codebooks.shape[2]:
            raise ValueError("saved dim does not match PQ codebook subspace shape")

        self.coarse_centroids = coarse_centroids.detach().clone()
        self.list_ids = self._pack_list_ids(list_ids.detach().clone())
        self.dim = dim
        self.num_vectors = num_vectors

        self.pq = ProductQuantizer(
            ProductQuantizerConfig(
                num_subspaces=self.config.num_subspaces,
                num_bits=self.config.num_bits,
                max_iter=self.config.pq_max_iter,
                seed=self.config.seed + 10_000,
                eps=self.config.eps,
            )
        )
        self.pq.codebooks = pq_codebooks.detach().clone()
        self.pq.dim = dim
        self.pq.subdim = pq_codebooks.shape[2]
        self.rotation_matrix = _optional_rotation_matrix(
            rotation_matrix,
            dim=dim,
            device=coarse_centroids.device,
            dtype=coarse_centroids.dtype,
            rotation=self.config.rotation,
        )
        self.key_norms = _optional_key_norms(
            key_norms,
            num_vectors=num_vectors,
            device=coarse_centroids.device,
            dtype=coarse_centroids.dtype,
            direction_normalize=self.config.direction_normalize,
        )

        if packed_codes is not None:
            if not self.stores_packed_codes:
                raise ValueError("packed_codes state requires 4-bit packed index config")
            packed = packed_codes.to(device=coarse_centroids.device, dtype=torch.uint8)
            expected_width = (self.config.num_subspaces + 1) // 2
            if packed.shape != (num_vectors, expected_width):
                raise ValueError("packed_codes shape does not match saved metadata")
            self.packed_codes = packed.detach().clone()
            self._codes = None
        elif codes is not None:
            code_tensor = codes.to(device=coarse_centroids.device, dtype=torch.long)
            if code_tensor.shape != (num_vectors, self.config.num_subspaces):
                raise ValueError("codes shape does not match saved metadata")
            self._store_codes(code_tensor.detach().clone())
        else:
            raise ValueError("state must contain either codes or packed_codes")

        self._rebuild_inverted_lists()
        return self

    def memory_footprint(self) -> IVFPQMemoryStats:
        """Return byte counts for compressed retrieval metadata."""

        self._check_built()
        assert self.coarse_centroids is not None
        assert self.list_ids is not None
        assert self.pq is not None
        assert self.pq.codebooks is not None

        codes = self.packed_codes if self.packed_codes is not None else self._codes
        assert codes is not None
        return IVFPQMemoryStats(
            coarse_centroid_bytes=_tensor_nbytes(self.coarse_centroids),
            list_id_bytes=_tensor_nbytes(self.list_ids),
            inverted_list_bytes=(
                0
                if self.inverted_list_offsets is None or self.inverted_list_indices is None
                else _tensor_nbytes(self.inverted_list_offsets)
                + _tensor_nbytes(self.inverted_list_indices)
            ),
            code_bytes=_tensor_nbytes(codes),
            pq_codebook_bytes=_tensor_nbytes(self.pq.codebooks),
            rotation_matrix_bytes=(
                0 if self.rotation_matrix is None else _tensor_nbytes(self.rotation_matrix)
            ),
            key_norm_bytes=(0 if self.key_norms is None else _tensor_nbytes(self.key_norms)),
        )

    def add(self, keys: torch.Tensor) -> torch.Tensor:
        """Encode new vectors into the existing IVF-PQ codebooks.

        This is the online append path used by decode-time KV cache updates.
        It keeps the trained coarse/PQ codebooks fixed, assigns each new key to
        its nearest coarse centroid, encodes the residual with the existing PQ,
        and appends only the compact metadata. Re-training or moving codebooks
        should happen through a full rebuild, otherwise old residual codes would
        no longer correspond to their original centroids.

        Returns the assigned IVF list id for each added key.
        """

        self._check_built()
        assert self.list_ids is not None
        assert self.pq is not None

        if keys.ndim != 2 or keys.shape[1] != self.dim:
            raise ValueError(f"keys must be [N, {self.dim}], got {tuple(keys.shape)}")
        if keys.shape[0] == 0:
            return torch.empty(0, device=self.list_ids.device, dtype=self._list_id_dtype())

        total_start = self._profile_start(keys.device)
        profile_start = self._profile_start(keys.device)
        transformed_keys, key_norms = self._project_keys(keys)
        self._profile_stop("index_add_project_keys", profile_start, keys.device)
        profile_start = self._profile_start(keys.device)
        list_ids = self._assign_lists_transformed(transformed_keys)
        self._profile_stop("index_add_assign_lists", profile_start, keys.device)
        profile_start = self._profile_start(keys.device)
        encoded = self._vectors_to_encode_with_lists(transformed_keys, list_ids)
        codes = self.pq.encode(encoded)
        self._profile_stop("index_add_pq_encode", profile_start, keys.device)

        profile_start = self._profile_start(keys.device)
        packed_new = self._pack_list_ids(list_ids)
        self.list_ids = torch.cat([self.list_ids, packed_new], dim=0)
        self._append_inverted_lists(packed_new)
        if key_norms is not None:
            if self.key_norms is None:
                self.key_norms = key_norms
            else:
                self.key_norms = torch.cat([self.key_norms, key_norms], dim=0)
        self._append_codes(codes)
        self.num_vectors += keys.shape[0]
        self._profile_stop("index_add_metadata", profile_start, keys.device)
        self._profile_stop("index_add_total", total_start, keys.device)
        return packed_new

    def online_refresh(
        self,
        keys: torch.Tensor,
        *,
        update_keys: torch.Tensor | None = None,
        learning_rate: float | None = None,
    ) -> "IVFPQIndex":
        """Update IVF/PQ codebooks from a mini-batch and re-encode all keys.

        This is a lightweight online-kmeans-style refresh for decode drift. New
        vectors move the coarse centroids and PQ codewords with EMA; because the
        codebooks move, all retrieval keys are reassigned and re-encoded so LUT
        scores still equal dot(query, reconstructed_key).
        """

        self._check_built()
        assert self.pq is not None
        assert self.coarse_centroids is not None

        if keys.ndim != 2 or keys.shape[1] != self.dim:
            raise ValueError(f"keys must be [N, {self.dim}], got {tuple(keys.shape)}")
        if keys.shape[0] == 0:
            raise ValueError("cannot refresh IVF-PQ index with zero keys")
        if update_keys is None:
            update_keys = keys
        if update_keys.ndim != 2 or update_keys.shape[1] != self.dim:
            raise ValueError(
                f"update_keys must be [N, {self.dim}], got {tuple(update_keys.shape)}"
            )

        lr = self.config.online_codebook_lr if learning_rate is None else learning_rate
        if not (0.0 < lr <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")

        total_start = self._profile_start(keys.device)
        profile_start = self._profile_start(keys.device)
        transformed_keys, key_norms = self._project_keys(keys)
        transformed_update_keys, _ = self._project_keys(update_keys)
        self._profile_stop("index_online_project_keys", profile_start, keys.device)
        if update_keys.shape[0] > 0:
            profile_start = self._profile_start(keys.device)
            update_list_ids = self._assign_lists_transformed(transformed_update_keys)
            self._update_coarse_centroids_ema(transformed_update_keys, update_list_ids, lr)
            update_list_ids = self._assign_lists_transformed(transformed_update_keys)
            update_residuals = self._vectors_to_encode_with_lists(
                transformed_update_keys,
                update_list_ids,
            )
            self.pq.update_codebooks_ema(update_residuals, learning_rate=lr)
            self._profile_stop("index_online_update_codebooks", profile_start, keys.device)

        profile_start = self._profile_start(keys.device)
        self.list_ids = self._assign_lists_transformed(transformed_keys)
        self.key_norms = key_norms
        encoded = self._vectors_to_encode_with_lists(transformed_keys, self.list_ids)
        self._store_codes(self.pq.encode(encoded))
        self.num_vectors = keys.shape[0]
        self._rebuild_inverted_lists()
        self._profile_stop("index_online_reencode", profile_start, keys.device)
        self._profile_stop("index_online_total", total_start, keys.device)
        return self

    def approximate_scores(
        self,
        query: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_built()
        assert self.pq is not None
        assert self.list_ids is not None
        assert self.coarse_centroids is not None

        self._validate_query(query)

        total_start = self._profile_start(query.device)
        profile_start = self._profile_start(query.device)
        transformed_query, query_norms = self._project_query(query)
        self._profile_stop("index_query_project", profile_start, query.device)
        scores = self._approximate_scores_transformed(transformed_query, query_norms, indices)
        self._profile_stop("index_approximate_scores_total", total_start, query.device)
        return scores

    def reconstruct(self, indices: torch.Tensor | None = None) -> torch.Tensor:
        self._check_built()
        assert self.pq is not None
        assert self.list_ids is not None
        assert self.coarse_centroids is not None

        codes = self._codes_for_scoring(indices)
        vectors = self.pq.decode(codes)
        if self.config.residual:
            list_ids = gather_ids(self.list_ids, indices)
            vectors = vectors + self.coarse_centroids[as_index(list_ids)]
        vectors = self._scale_reconstructed_vectors(vectors, indices)
        return self._inverse_transform_vectors(vectors)

    def search(
        self,
        query: torch.Tensor,
        *,
        topk: int,
        nprobe: int | None = None,
        candidate_budget: int | None = None,
    ) -> SearchResult:
        total_start = self._profile_start(query.device)
        self._check_built()
        assert self.list_ids is not None

        if topk <= 0:
            raise ValueError("topk must be positive")
        self._validate_single_query(query)
        profile_start = self._profile_start(query.device)
        transformed_query, query_norm = self._project_query(query)
        self._profile_stop("index_search_project_query", profile_start, query.device)
        profile_start = self._profile_start(query.device)
        probed_lists = self._select_lists_transformed(transformed_query, nprobe=nprobe)
        self._profile_stop("index_search_select_lists", profile_start, query.device)
        profile_start = self._profile_start(query.device)
        candidate_indices = self._candidate_indices_for_lists(probed_lists)
        self._profile_stop("index_search_collect_candidates", profile_start, query.device)

        profile_start = self._profile_start(query.device)
        top_indices, top_scores, candidate_indices, candidate_scores = self._search_candidate_topk(
            query=transformed_query,
            query_norm=query_norm,
            candidate_indices=candidate_indices,
            topk=topk,
            candidate_budget=candidate_budget,
        )
        self._profile_stop("index_search_candidate_topk", profile_start, query.device)
        self._profile_stop("index_search_total", total_start, query.device)

        return SearchResult(
            indices=top_indices,
            approx_scores=top_scores,
            candidate_indices=candidate_indices,
            candidate_scores=candidate_scores,
            probed_lists=probed_lists,
        )

    def search_many(
        self,
        queries: torch.Tensor,
        *,
        topk: int,
        nprobe: int | None = None,
        candidate_budget: int | None = None,
    ) -> tuple[SearchResult, ...]:
        self._check_built()
        assert self.list_ids is not None

        if topk <= 0:
            raise ValueError("topk must be positive")
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"queries must be [B, {self.dim}], got {tuple(queries.shape)}")

        total_start = self._profile_start(queries.device)
        profile_start = self._profile_start(queries.device)
        transformed_queries, query_norms = self._project_query(queries)
        self._profile_stop("index_search_many_project_queries", profile_start, queries.device)
        profile_start = self._profile_start(queries.device)
        probed_lists = self._select_lists_many_transformed(transformed_queries, nprobe=nprobe)
        self._profile_stop("index_search_many_select_lists", profile_start, queries.device)
        profile_start = self._profile_start(queries.device)
        batched_results = self._search_many_candidate_topk_batched(
            queries=transformed_queries,
            query_norms=query_norms,
            probed_lists=probed_lists,
            topk=topk,
            candidate_budget=candidate_budget,
        )
        self._profile_stop("index_search_many_candidate_topk", profile_start, queries.device)
        if batched_results is not None:
            self._profile_stop("index_search_many_total", total_start, queries.device)
            return batched_results

        results = tuple(
            self.search(query, topk=topk, nprobe=nprobe, candidate_budget=candidate_budget)
            for query in queries
        )
        self._profile_stop("index_search_many_total", total_start, queries.device)
        return results

    def select_lists(self, query: torch.Tensor, *, nprobe: int | None = None) -> torch.Tensor:
        self._check_built()
        assert self.coarse_centroids is not None

        self._validate_single_query(query)
        transformed_query, _ = self._project_query(query)
        return self._select_lists_transformed(transformed_query, nprobe=nprobe)

    def _select_lists_transformed(
        self,
        query: torch.Tensor,
        *,
        nprobe: int | None = None,
    ) -> torch.Tensor:
        assert self.coarse_centroids is not None

        probes = self.config.nprobe if nprobe is None else nprobe
        probes = min(max(probes, 1), self.coarse_centroids.shape[0])
        coarse_scores = self.coarse_centroids.matmul(query)
        return torch.topk(coarse_scores, k=probes).indices

    def _select_lists_many_transformed(
        self,
        queries: torch.Tensor,
        *,
        nprobe: int | None = None,
    ) -> torch.Tensor:
        assert self.coarse_centroids is not None

        probes = self.config.nprobe if nprobe is None else nprobe
        probes = min(max(probes, 1), self.coarse_centroids.shape[0])
        coarse_scores = queries.matmul(self.coarse_centroids.T)
        return torch.topk(coarse_scores, k=probes, dim=1).indices

    def _rebuild_inverted_lists(self) -> None:
        assert self.coarse_centroids is not None
        assert self.list_ids is not None

        num_lists = self.coarse_centroids.shape[0]
        indexable = as_index(self.list_ids)
        counts = torch.bincount(indexable, minlength=num_lists)
        # CSR stays int64. The hybrid centroid path feeds these tensors to
        # list_exp_sums_triton; an int32 resident copy made the second
        # forward_many diverge (max |Δ| ~ 20) vs the historical int64 layout.
        offsets = torch.empty(num_lists + 1, device=self.list_ids.device, dtype=torch.int64)
        offsets[0] = 0
        offsets[1:] = torch.cumsum(counts.to(torch.int64), dim=0)
        if self.list_ids.numel() == 0:
            indices = torch.empty(0, device=self.list_ids.device, dtype=torch.int64)
        else:
            indices = torch.argsort(indexable, stable=True)
        self.inverted_list_offsets = offsets
        self.inverted_list_indices = indices

    def _append_inverted_lists(self, list_ids: torch.Tensor) -> None:
        assert self.coarse_centroids is not None
        assert self.list_ids is not None

        num_lists = self.coarse_centroids.shape[0]
        if (
            self.inverted_list_offsets is None
            or self.inverted_list_indices is None
            or self.inverted_list_offsets.shape[0] != num_lists + 1
            or self.inverted_list_indices.shape[0] != self.num_vectors
        ):
            self._rebuild_inverted_lists()
            return

        if list_ids.numel() == 0:
            return

        device = self.inverted_list_offsets.device
        list_ids = as_index(list_ids.to(device=device))
        old_offsets = self.inverted_list_offsets
        old_indices = self.inverted_list_indices
        old_counts = old_offsets[1:] - old_offsets[:-1]
        new_counts = torch.bincount(list_ids, minlength=num_lists).to(torch.int64)
        total_counts = old_counts + new_counts

        new_offsets = torch.empty(old_offsets.shape[0], device=device, dtype=torch.int64)
        new_offsets[0] = 0
        new_offsets[1:] = torch.cumsum(total_counts, dim=0)
        new_indices = torch.empty(
            self.num_vectors + list_ids.numel(),
            device=device,
            dtype=torch.int64,
        )

        if old_indices.numel() > 0:
            old_member_list_ids = as_index(
                gather_ids(self.list_ids[: self.num_vectors].to(device=device), old_indices)
            )
            old_rank_base = torch.repeat_interleave(old_offsets[:-1], old_counts)
            old_ranks = (
                torch.arange(old_indices.numel(), device=device, dtype=torch.long)
                - old_rank_base
            )
            old_dest = new_offsets[old_member_list_ids] + old_ranks
            new_indices[old_dest] = old_indices

        new_order = torch.argsort(list_ids, stable=True)
        sorted_new_lists = list_ids[new_order]
        sorted_new_indices = torch.arange(
            self.num_vectors,
            self.num_vectors + list_ids.numel(),
            device=device,
            dtype=torch.long,
        )[new_order]
        new_rank_base = torch.repeat_interleave(
            torch.cumsum(new_counts, dim=0) - new_counts,
            new_counts,
        )
        new_ranks = (
            torch.arange(list_ids.numel(), device=device, dtype=torch.long)
            - new_rank_base
        )
        new_dest = new_offsets[sorted_new_lists] + old_counts[sorted_new_lists] + new_ranks
        new_indices[new_dest] = sorted_new_indices

        self.inverted_list_offsets = new_offsets
        self.inverted_list_indices = new_indices

    def _candidate_indices_for_lists(self, probed_lists: torch.Tensor) -> torch.Tensor:
        assert self.list_ids is not None

        if self.inverted_list_offsets is None or self.inverted_list_indices is None:
            self._rebuild_inverted_lists()
        assert self.inverted_list_offsets is not None
        assert self.inverted_list_indices is not None

        probed_lists = probed_lists.to(
            device=self.inverted_list_offsets.device,
            dtype=torch.long,
        ).flatten()
        valid = (probed_lists >= 0) & (probed_lists < self.inverted_list_offsets.shape[0] - 1)
        probed_lists = probed_lists[valid]
        if probed_lists.numel() == 0:
            return self._full_candidate_indices()

        starts = self.inverted_list_offsets[probed_lists]
        lengths = self.inverted_list_offsets[probed_lists + 1] - starts
        repeated_starts = torch.repeat_interleave(starts, lengths)
        if repeated_starts.numel() == 0:
            return self._full_candidate_indices()

        repeated_prefix = torch.repeat_interleave(torch.cumsum(lengths, dim=0) - lengths, lengths)
        positions = torch.arange(
            repeated_starts.shape[0],
            device=repeated_starts.device,
            dtype=torch.long,
        )
        source_positions = repeated_starts + positions - repeated_prefix
        return self.inverted_list_indices[source_positions]

    def _full_candidate_indices(self) -> torch.Tensor:
        assert self.list_ids is not None
        return torch.arange(
            self.num_vectors,
            device=self.list_ids.device,
            dtype=torch.long,
        )

    def assign_lists(self, keys: torch.Tensor) -> torch.Tensor:
        self._check_built()
        assert self.coarse_centroids is not None

        if keys.ndim != 2 or keys.shape[1] != self.dim:
            raise ValueError(f"keys must be [N, {self.dim}], got {tuple(keys.shape)}")
        transformed_keys, _ = self._project_keys(keys)
        return self._assign_lists_transformed(transformed_keys)

    def _assign_lists_transformed(self, keys: torch.Tensor) -> torch.Tensor:
        assert self.coarse_centroids is not None

        distances = torch.cdist(keys.float(), self.coarse_centroids.float(), p=2)
        return self._pack_list_ids(distances.argmin(dim=1))

    def _vectors_to_encode(self, keys: torch.Tensor) -> torch.Tensor:
        assert self.coarse_centroids is not None
        assert self.list_ids is not None
        return self._vectors_to_encode_with_lists(keys, self.list_ids)

    def _vectors_to_encode_with_lists(self, keys: torch.Tensor, list_ids: torch.Tensor) -> torch.Tensor:
        assert self.coarse_centroids is not None
        if not self.config.residual:
            return keys
        return keys - self.coarse_centroids[as_index(list_ids)]

    def _ensure_rotation_matrix(self, reference: torch.Tensor) -> None:
        assert self.dim is not None
        if self.config.rotation == "none":
            self.rotation_matrix = None
            return

        if self.rotation_matrix is not None:
            if self.rotation_matrix.shape != (self.dim, self.dim):
                raise ValueError("rotation_matrix shape does not match index dimension")
            self.rotation_matrix = self.rotation_matrix.to(
                device=reference.device,
                dtype=reference.dtype,
            )
            return

        self.rotation_matrix = _random_orthogonal_matrix(
            self.dim,
            device=reference.device,
            dtype=reference.dtype,
            seed=self.config.seed + 20_000,
        )

    def _transform_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        if self.rotation_matrix is None:
            return vectors
        rotation = self.rotation_matrix.to(device=vectors.device, dtype=vectors.dtype)
        return vectors.matmul(rotation)

    def _inverse_transform_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        if self.rotation_matrix is None:
            return vectors
        rotation = self.rotation_matrix.to(device=vectors.device, dtype=vectors.dtype)
        return vectors.matmul(rotation.T)

    def _project_keys(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        transformed = self._transform_vectors(keys)
        if not self.config.direction_normalize:
            return transformed, None

        norms = torch.linalg.vector_norm(transformed.float(), dim=-1).to(transformed.dtype)
        norms = norms.clamp_min(self.config.eps)
        return transformed / norms.unsqueeze(-1), norms

    def _project_query(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        transformed = self._transform_vectors(query)
        if not self.config.direction_normalize:
            return transformed, None

        norms = torch.linalg.vector_norm(transformed.float(), dim=-1).to(transformed.dtype)
        norms = norms.clamp_min(self.config.eps)
        if query.ndim == 1:
            return transformed / norms, norms
        return transformed / norms.unsqueeze(-1), norms

    def _scale_reconstructed_vectors(
        self,
        vectors: torch.Tensor,
        indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.config.direction_normalize:
            return vectors
        assert self.key_norms is not None

        key_norms = self.key_norms if indices is None else self.key_norms[indices]
        return vectors * key_norms.unsqueeze(-1)

    def _approximate_scores_transformed(
        self,
        query: torch.Tensor,
        query_norms: torch.Tensor | None,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.list_ids is not None

        list_ids = gather_ids(self.list_ids, indices)
        if self._can_use_fused_batched_list_bias_scores(query):
            scores = self._pq_scores_with_batched_list_bias(query, list_ids, indices)
        else:
            scores = self._pq_scores(query, indices)
            if self.config.residual:
                scores = scores + self._coarse_score_bias(query, list_ids)
        profile_start = self._profile_start(query.device)
        scaled_scores = self._scale_projected_scores(scores, query_norms, indices)
        self._profile_stop("index_score_scale", profile_start, query.device)
        return scaled_scores

    def _scale_projected_scores(
        self,
        scores: torch.Tensor,
        query_norms: torch.Tensor | None,
        indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.config.direction_normalize:
            return scores
        assert self.key_norms is not None
        assert query_norms is not None

        key_norms = self.key_norms if indices is None else self.key_norms[indices]
        if query_norms.ndim == 0:
            return scores * query_norms * key_norms
        return scores * query_norms.unsqueeze(-1) * key_norms.unsqueeze(0)

    def _prepare_initial_coarse_centroids(
        self,
        keys: torch.Tensor,
        num_lists: int,
        initial_centroids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if initial_centroids is None:
            return None
        if initial_centroids.ndim != 2:
            raise ValueError(
                f"initial_coarse_centroids must be [K, D], got {tuple(initial_centroids.shape)}"
            )
        if initial_centroids.shape[1] != keys.shape[1]:
            raise ValueError(
                f"initial_coarse_centroids dim must be {keys.shape[1]}, "
                f"got {initial_centroids.shape[1]}"
            )

        centroids = initial_centroids.to(device=keys.device, dtype=keys.dtype)
        if centroids.shape[0] == num_lists:
            return centroids.clone()
        if centroids.shape[0] > num_lists:
            return centroids[:num_lists].clone()

        missing = num_lists - centroids.shape[0]
        generator = torch.Generator(device=keys.device)
        generator.manual_seed(self.config.seed)
        if keys.shape[0] >= missing:
            indices = torch.randperm(keys.shape[0], generator=generator, device=keys.device)[:missing]
        else:
            indices = torch.randint(
                0,
                keys.shape[0],
                (missing,),
                generator=generator,
                device=keys.device,
            )
        return torch.cat([centroids, keys[indices]], dim=0).clone()

    def _update_coarse_centroids_ema(
        self,
        keys: torch.Tensor,
        list_ids: torch.Tensor,
        learning_rate: float,
    ) -> None:
        assert self.coarse_centroids is not None

        num_lists = self.coarse_centroids.shape[0]
        list_ids = as_index(list_ids)
        sums = torch.zeros_like(self.coarse_centroids)
        counts = torch.bincount(list_ids, minlength=num_lists).to(keys.dtype)
        sums.index_add_(0, list_ids, keys)
        non_empty = counts > 0
        means = sums[non_empty] / counts[non_empty].unsqueeze(1).clamp_min(self.config.eps)
        self.coarse_centroids[non_empty] = (
            (1.0 - learning_rate) * self.coarse_centroids[non_empty]
            + learning_rate * means
        )

    def _coarse_score_bias(self, query: torch.Tensor, list_ids: torch.Tensor) -> torch.Tensor:
        assert self.coarse_centroids is not None
        profile_start = self._profile_start(query.device)
        list_ids = as_index(list_ids)
        if query.ndim == 1:
            list_scores = self.coarse_centroids.matmul(query)
            result = list_scores[list_ids]
        else:
            list_scores = query.matmul(self.coarse_centroids.T)
            result = list_scores[:, list_ids]
        self._profile_stop("index_coarse_score_bias", profile_start, query.device)
        return result

    def _validate_query(self, query: torch.Tensor) -> None:
        if query.ndim == 1 and query.shape[0] == self.dim:
            return
        if query.ndim == 2 and query.shape[1] == self.dim:
            return
        raise ValueError(f"query must be [{self.dim}] or [B, {self.dim}], got {tuple(query.shape)}")

    def _validate_single_query(self, query: torch.Tensor) -> None:
        if query.ndim != 1 or query.shape[0] != self.dim:
            raise ValueError(f"query must be [{self.dim}], got {tuple(query.shape)}")

    def _check_built(self) -> None:
        if not self.is_built:
            raise RuntimeError("IVFPQIndex must be built first")

    def _store_codes(self, codes: torch.Tensor) -> None:
        if self.stores_packed_codes:
            self._codes = None
            self.packed_codes = pack_4bit_codes(codes)
        else:
            self._codes = codes
            self.packed_codes = None

    def _append_codes(self, codes: torch.Tensor) -> None:
        if self.stores_packed_codes:
            if self.packed_codes is None:
                self.packed_codes = pack_4bit_codes(codes)
            else:
                self.packed_codes = torch.cat([self.packed_codes, pack_4bit_codes(codes)], dim=0)
            self._codes = None
            return

        if self._codes is None:
            self._codes = codes
        else:
            self._codes = torch.cat([self._codes, codes], dim=0)
        self.packed_codes = None

    def _codes_for_scoring(self, indices: torch.Tensor | None = None) -> torch.Tensor:
        if self._codes is not None:
            return self._codes if indices is None else self._codes[indices]
        if self.packed_codes is None:
            raise RuntimeError("IVFPQIndex must be built first")
        packed = self.packed_codes if indices is None else self.packed_codes[indices]
        return unpack_4bit_codes(packed, self.config.num_subspaces)

    def _pq_scores(self, query: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        assert self.pq is not None
        if self.stores_packed_codes and self.packed_codes is not None:
            packed = self.packed_codes if indices is None else self.packed_codes[indices]
            profile_start = self._profile_start(query.device)
            lut = self.pq.compute_lut(query)
            self._profile_stop("index_pq_compute_lut", profile_start, query.device)
            profile_start = self._profile_start(query.device)
            scores = score_packed_4bit_lut(
                packed,
                lut,
                num_subspaces=self.config.num_subspaces,
                block_size=self.config.topk_block_size,
                backend=self.config.kernel_backend,
            )
            self._profile_stop("index_pq_lut_scan", profile_start, query.device)
            return scores

        codes = self._codes_for_scoring(indices)
        profile_start = self._profile_start(query.device)
        lut = self.pq.compute_lut(query)
        self._profile_stop("index_pq_compute_lut", profile_start, query.device)
        profile_start = self._profile_start(query.device)
        scores = self.pq.score_from_lut(codes, lut)
        self._profile_stop("index_pq_lut_scan", profile_start, query.device)
        return scores

    def _can_use_fused_batched_list_bias_scores(self, query: torch.Tensor) -> bool:
        return (
            self.config.residual
            and query.ndim == 2
            and self.stores_packed_codes
            and self.packed_codes is not None
            and self.config.kernel_backend in GPU_LUT_BACKENDS
            and is_triton_available()
            and query.is_cuda
            and self.packed_codes.is_cuda
        )

    def _pq_scores_with_batched_list_bias(
        self,
        query: torch.Tensor,
        list_ids: torch.Tensor,
        indices: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.pq is not None
        assert self.coarse_centroids is not None
        assert self.packed_codes is not None

        packed = self.packed_codes if indices is None else self.packed_codes[indices]
        profile_start = self._profile_start(query.device)
        list_scores = query.matmul(self.coarse_centroids.T)
        self._profile_stop("index_coarse_list_scores", profile_start, query.device)
        profile_start = self._profile_start(query.device)
        lut = self.pq.compute_lut(query)
        self._profile_stop("index_pq_compute_lut", profile_start, query.device)
        profile_start = self._profile_start(query.device)
        scores = score_packed_4bit_lut_batched_list_bias_triton(
            packed,
            lut,
            list_ids,
            list_scores,
            num_subspaces=self.config.num_subspaces,
            block_size=(
                256 if self.config.topk_block_size is None else self.config.topk_block_size
            ),
        )
        self._profile_stop("index_pq_lut_scan", profile_start, query.device)
        return scores

    def _search_candidate_topk(
        self,
        *,
        query: torch.Tensor,
        query_norm: torch.Tensor | None,
        candidate_indices: torch.Tensor,
        topk: int,
        candidate_budget: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.pq is not None
        assert self.list_ids is not None
        assert self.coarse_centroids is not None

        if self.stores_packed_codes and self.packed_codes is not None:
            candidate_list_ids = as_index(gather_ids(self.list_ids, candidate_indices))
            score_bias = None
            if self.config.residual:
                score_bias = self.coarse_centroids.matmul(query)[candidate_list_ids]
            score_scale = None
            if self.config.direction_normalize:
                assert self.key_norms is not None
                assert query_norm is not None
                score_scale = self.key_norms[candidate_indices] * query_norm

            candidate_scores = score_packed_4bit_lut(
                self.packed_codes[candidate_indices],
                self.pq.compute_lut(query),
                num_subspaces=self.config.num_subspaces,
                block_size=self.config.topk_block_size,
                backend=self.config.kernel_backend,
            )
            if score_bias is not None:
                candidate_scores = candidate_scores + score_bias
            if score_scale is not None:
                candidate_scores = candidate_scores * score_scale

            budget = candidate_indices.numel() if candidate_budget is None else candidate_budget
            budget = min(max(budget, topk), candidate_indices.numel())
            if budget < candidate_indices.numel():
                candidate_scores, budget_order = torch.topk(candidate_scores, k=budget)
                candidate_indices = candidate_indices[budget_order]

            k = min(topk, candidate_indices.numel())
            top_scores, top_order = torch.topk(candidate_scores, k=k)
            top_indices = candidate_indices[top_order]
            return top_indices, top_scores, candidate_indices, candidate_scores

        candidate_scores = self._approximate_scores_transformed(
            query,
            query_norm,
            candidate_indices,
        )
        budget = candidate_indices.numel() if candidate_budget is None else candidate_budget
        budget = min(max(budget, topk), candidate_indices.numel())

        if budget < candidate_indices.numel():
            budget_scores, budget_order = torch.topk(candidate_scores, k=budget)
            candidate_indices = candidate_indices[budget_order]
            candidate_scores = budget_scores

        k = min(topk, candidate_indices.numel())
        top_scores, top_order = torch.topk(candidate_scores, k=k)
        top_indices = candidate_indices[top_order]
        return top_indices, top_scores, candidate_indices, candidate_scores

    def _search_many_candidate_topk_batched(
        self,
        *,
        queries: torch.Tensor,
        query_norms: torch.Tensor | None,
        probed_lists: torch.Tensor,
        topk: int,
        candidate_budget: int | None,
    ) -> tuple[SearchResult, ...] | None:
        assert self.list_ids is not None

        if queries.shape[0] == 0:
            return ()
        if queries.ndim != 2:
            return None

        profile_start = self._profile_start(queries.device)
        per_query_candidates = [
            self._candidate_indices_for_lists(current_probed)
            for current_probed in probed_lists
        ]
        self._profile_stop("index_search_many_collect_candidates", profile_start, queries.device)
        if not per_query_candidates:
            return ()

        candidate_counts = [int(indices.numel()) for indices in per_query_candidates]
        if any(count == 0 for count in candidate_counts):
            return None

        all_candidate_indices = torch.cat(per_query_candidates, dim=0)
        candidate_list_ids = gather_ids(self.list_ids, all_candidate_indices)
        if self.stores_packed_codes and self.packed_codes is not None:
            assert self.pq is not None
            # Candidate sets are much smaller than full all-PQ scans. The torch
            # packed-gather path avoids first-use Triton compilation inside
            # decode, which dominates short E2E candidate runs.
            profile_start = self._profile_start(queries.device)
            lut = self.pq.compute_lut(queries)
            self._profile_stop("index_pq_compute_lut", profile_start, queries.device)
            profile_start = self._profile_start(queries.device)
            all_candidate_scores = score_packed_4bit_lut(
                self.packed_codes[all_candidate_indices],
                lut,
                num_subspaces=self.config.num_subspaces,
                backend="torch",
            )
            self._profile_stop("index_pq_lut_scan", profile_start, queries.device)
        else:
            all_candidate_scores = self._pq_scores(queries, all_candidate_indices)
        if self.config.residual:
            profile_start = self._profile_start(queries.device)
            all_candidate_scores = all_candidate_scores + self._coarse_score_bias(
                queries,
                candidate_list_ids,
            )
            self._profile_stop("index_search_many_residual_bias", profile_start, queries.device)
        profile_start = self._profile_start(queries.device)
        all_candidate_scores = self._scale_projected_scores(
            all_candidate_scores,
            query_norms,
            all_candidate_indices,
        )
        self._profile_stop("index_score_scale", profile_start, queries.device)
        if all_candidate_scores.shape != (queries.shape[0], all_candidate_indices.numel()):
            raise RuntimeError(
                "batched candidate scores must be "
                f"[{queries.shape[0]}, {all_candidate_indices.numel()}], "
                f"got {tuple(all_candidate_scores.shape)}"
            )

        results = []
        start = 0
        profile_start = self._profile_start(queries.device)
        for row, (current_probed, candidate_indices, count) in enumerate(
            zip(probed_lists, per_query_candidates, candidate_counts, strict=True)
        ):
            end = start + count
            candidate_scores = all_candidate_scores[row, start:end]
            start = end

            budget = count if candidate_budget is None else candidate_budget
            budget = min(max(budget, topk), count)
            if budget < count:
                budget_scores, budget_order = torch.topk(candidate_scores, k=budget)
                candidate_indices = candidate_indices[budget_order]
                candidate_scores = budget_scores

            k = min(topk, candidate_indices.numel())
            top_scores, top_order = torch.topk(candidate_scores, k=k)
            top_indices = candidate_indices[top_order]
            results.append(
                SearchResult(
                    indices=top_indices,
                    approx_scores=top_scores,
                    candidate_indices=candidate_indices,
                    candidate_scores=candidate_scores,
                    probed_lists=current_probed,
                )
            )
        self._profile_stop("index_search_many_result_topk_pack", profile_start, queries.device)
        return tuple(results)


def _config_from_state(state: dict[str, object]) -> IVFPQConfig:
    config_state = state.get("config")
    if not isinstance(config_state, dict):
        raise ValueError("state must contain an IVFPQConfig dict under 'config'")
    return IVFPQConfig(**config_state)


def _required_tensor(state: dict[str, object], name: str) -> torch.Tensor:
    value = state.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"state must contain tensor '{name}'")
    return value


def _optional_rotation_matrix(
    value: object,
    *,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    rotation: str,
) -> torch.Tensor | None:
    if rotation == "none":
        if value is not None:
            raise ValueError("rotation_matrix state requires rotation='random_orthogonal'")
        return None

    if not isinstance(value, torch.Tensor):
        raise ValueError("rotation_matrix state must be a tensor for random_orthogonal rotation")
    if value.shape != (dim, dim):
        raise ValueError("rotation_matrix shape does not match saved dim")
    return value.to(device=device, dtype=dtype).detach().clone()


def _optional_key_norms(
    value: object,
    *,
    num_vectors: int,
    device: torch.device,
    dtype: torch.dtype,
    direction_normalize: bool,
) -> torch.Tensor | None:
    if not direction_normalize:
        if value is not None:
            raise ValueError("key_norms state requires direction_normalize=True")
        return None

    if not isinstance(value, torch.Tensor):
        raise ValueError("key_norms state must be a tensor for direction-normalized indexes")
    if value.shape != (num_vectors,):
        raise ValueError("key_norms shape does not match saved num_vectors")
    return value.to(device=device, dtype=dtype).detach().clone()


def _random_orthogonal_matrix(
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    sample = torch.randn(dim, dim, generator=generator, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(sample)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q.to(dtype=dtype)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
