from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import time

import torch

from pq_hsa.index.ivfpq import IVFPQConfig, IVFPQIndex, IVFPQMemoryStats, as_index


def _offload_fast_enabled() -> bool:
    """Opt-in gate.

    The fast offload gather path (pinned staging buffers, a dedicated async
    copy stream + event, and an incrementally-maintained sink+local GPU
    mirror) only activates when this env var is set. Unset (the default), a
    ``SparseKVCache`` behaves byte-for-byte like the previous implementation
    -- no code on the default numerical path changes.
    """
    return os.environ.get("PQ_HSA_OFFLOAD_FAST", "0") == "1"


@dataclass(slots=True)
class KVCacheRegions:
    sink: torch.Tensor
    retrieval: torch.Tensor
    local: torch.Tensor


@dataclass(slots=True)
class FetchedKV:
    indices: torch.Tensor
    keys: torch.Tensor | None
    values: torch.Tensor | None


@dataclass(slots=True)
class PrefetchedKV:
    fetched: FetchedKV
    stream: torch.cuda.Stream | None
    device: torch.device | None

    def wait(self) -> FetchedKV:
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        return self.fetched


@dataclass(slots=True)
class KVCacheMemoryStats:
    key_bytes: int
    value_bytes: int
    index_bytes: int
    retrieval_full_key_bytes: int
    compressed_key_metadata_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.key_bytes + self.value_bytes + self.index_bytes

    @property
    def key_compression_ratio(self) -> float:
        if self.retrieval_full_key_bytes == 0:
            return 1.0
        return self.compressed_key_metadata_bytes / self.retrieval_full_key_bytes


class SparseKVCache:
    """Full KV store plus an IVF-PQ index over the retrieval region."""

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        index_config: IVFPQConfig,
        sink_tokens: int = 0,
        local_window: int = 0,
        index_update_interval: int = 1,
        index_update_strategy: str = "incremental",
        codebook_refresh_interval: int | None = None,
        kv_storage: str = "device",
        pin_offloaded_kv: bool = False,
        profile: bool = False,
        shared_coarse_centroids: torch.Tensor | None = None,
        shared_pq_codebooks: torch.Tensor | None = None,
        _precomputed_list_ids: torch.Tensor | None = None,
        _precomputed_codes: torch.Tensor | None = None,
        growth_margin: int = 4096,
        _key_buf: torch.Tensor | None = None,
        _val_buf: torch.Tensor | None = None,
    ):
        if keys.ndim != 2 or values.ndim != 2:
            raise ValueError("keys and values must both be rank-2 tensors")
        if keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values must have the same sequence length")
        if sink_tokens < 0 or local_window < 0:
            raise ValueError("sink_tokens and local_window must be non-negative")
        if index_update_interval <= 0:
            raise ValueError("index_update_interval must be positive")
        if index_update_strategy not in {"incremental", "online", "rebuild", "deferred"}:
            raise ValueError(
                "index_update_strategy must be 'incremental', 'online', 'rebuild', or 'deferred'"
            )
        if codebook_refresh_interval is not None and codebook_refresh_interval <= 0:
            raise ValueError("codebook_refresh_interval must be positive when provided")
        if kv_storage not in {"device", "cpu"}:
            raise ValueError("kv_storage must be 'device' or 'cpu'")

        self.index_device = keys.device
        self.kv_storage = kv_storage
        self.pin_offloaded_kv = pin_offloaded_kv
        self.profile_enabled = bool(profile)
        self.profile_stats: dict[str, float] = {}
        self.shared_coarse_centroids = shared_coarse_centroids
        self.shared_pq_codebooks = shared_pq_codebooks
        # (opt-in batched build): consumed by the FIRST _build_index
        # only, then cleared so deferred rebuilds behave exactly as before.
        self._precomputed_list_ids = _precomputed_list_ids
        self._precomputed_codes = _precomputed_codes
        self.index_config = index_config
        self.sink_tokens = sink_tokens
        self.local_window = local_window
        self.index_update_interval = index_update_interval
        self.index_update_strategy = index_update_strategy
        self.codebook_refresh_interval = codebook_refresh_interval
        self.growth_margin = int(growth_margin)
        initial_len = keys.shape[0]
        self._length = initial_len
        self._key_shape = tuple(keys.shape[1:])
        self._value_shape = tuple(values.shape[1:])
        self._indexed_length = initial_len
        self._pending_update_tokens = 0
        self._deferred_index_update_pending = False
        self._tokens_since_codebook_refresh = 0
        self.index_rebuilds = 0
        self.index_adds = 0
        self.online_codebook_updates = 0
        self.codebook_refreshes = 0
        self.deferred_index_updates = 0
        self.reset_fetch_stats()

        # Preallocated capacity buffer: avoids O(N) torch.cat on every append.
        # _key_buf / _val_buf are the backing stores; self.keys / self.values are
        # always public views into buf[:_length] so all consumers keep working.
        if _key_buf is not None and _val_buf is not None:
            # Externally-provided buffer (shared-base path in the adapter).
            assert _key_buf.shape[0] >= initial_len
            assert _val_buf.shape[0] >= initial_len
            self._key_buf = _key_buf
            self._val_buf = _val_buf
            self._buf_capacity = _key_buf.shape[0]
            # Copy initial data into the provided buffer.
            self._key_buf[:initial_len].copy_(self._prepare_full_tensor(keys))
            self._val_buf[:initial_len].copy_(self._prepare_full_tensor(values))
        else:
            cap = initial_len + self.growth_margin
            raw_k = self._prepare_full_tensor(keys)
            raw_v = self._prepare_full_tensor(values)
            self._key_buf = self._alloc_buf(cap, raw_k)
            self._val_buf = self._alloc_buf(cap, raw_v)
            self._buf_capacity = cap
            self._key_buf[:initial_len].copy_(raw_k)
            self._val_buf[:initial_len].copy_(raw_v)

        # Public views into valid portion of the buffer.
        self.keys = self._key_buf[:self._length]
        self.values = self._val_buf[:self._length]

        # Cached region tensors — rebuilt only when the region boundaries change.
        self._cached_sink_end: int = -1
        self._cached_retrieval_end: int = -1
        self._cached_length: int = -1
        self._cached_regions: KVCacheRegions | None = None

        self.regions = self._make_regions(initial_len, self.index_device, self._indexed_length)
        # (opt-in, PQ_HSA_OFFLOAD_FAST=1): see _offload_fast_enabled().
        # Must be set up before _build_index() below, since that calls
        # fetch_keys() -> _fast_eligible().
        self._fast_offload = (
            self.kv_storage == "cpu"
            and self.pin_offloaded_kv
            and _offload_fast_enabled()
        )
        if self._fast_offload:
            self._fast_init_state()

        self.index: IVFPQIndex | None = None
        self.retrieval_value_centroids: torch.Tensor | None = None
        self._retrieval_value_sums: torch.Tensor | None = None
        self._retrieval_value_counts: torch.Tensor | None = None
        self._build_index()

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append one generated token to the exact local/update-buffer region.

        New decode tokens remain full precision in the local/update-buffer path.
        Once ``index_update_interval`` tokens have accumulated, tokens that are
        old enough to leave the recent window are encoded into the IVF-PQ index.
        The default path keeps existing codebooks fixed and appends compact
        codes. ``index_update_strategy="online"`` moves codebooks with a
        mini-batch EMA update and re-encodes the retrieval region.
        ``index_update_strategy="deferred"`` records that the update is ready
        and keeps pending tokens in the exact local path until
        ``apply_pending_index_update()`` runs the online refresh. If
        ``codebook_refresh_interval`` is set, enough newly indexed tokens
        trigger a warm-start coarse/PQ retrain and full re-encode.
        """

        if tuple(key.shape) != self._key_shape:
            raise ValueError(f"key shape must be {self._key_shape}")
        if tuple(value.shape) != self._value_shape:
            raise ValueError(f"value shape must be {self._value_shape}")
        self._append_to_buf(key, value)
        self._length += 1
        self._pending_update_tokens += 1

        if self._pending_update_tokens >= self.index_update_interval:
            self.refresh_index()
        else:
            self.regions = self._make_regions(
                self._length,
                self.index_device,
                self._indexed_length,
            )

    def _notify_external_append(self) -> None:
        """Update bookkeeping after the caller has already written the token row
        into the backing buffer externally (e.g. via a shared multi-head write).

        This is identical to the bookkeeping tail of ``append()`` but skips the
        ``_append_to_buf`` call and the shape validation, because the data is
        already in place. The public ``keys``/``values`` views are resliced here.
        """
        self._length += 1
        # Reslice public views to expose the new row.
        self.keys = self._key_buf[: self._length]
        self.values = self._val_buf[: self._length]
        self._pending_update_tokens += 1

        if self._pending_update_tokens >= self.index_update_interval:
            self.refresh_index()
        else:
            self.regions = self._make_regions(
                self._length,
                self.index_device,
                self._indexed_length,
            )

    @property
    def pending_index_update(self) -> bool:
        return self._deferred_index_update_pending

    @property
    def pending_update_tokens(self) -> int:
        return self._pending_update_tokens

    def refresh_index(self, *, force: bool = False) -> None:
        """Move old-enough pending tokens into retrieval metadata."""

        if self.index_update_strategy == "deferred" and not force:
            self._mark_deferred_index_update()
            return
        self._refresh_index_now()

    def apply_pending_index_update(self) -> bool:
        """Apply a deferred online index update, if one has been scheduled."""

        if not self._deferred_index_update_pending:
            return False
        self._refresh_index_now()
        self._deferred_index_update_pending = False
        return True

    def _refresh_index_now(self) -> None:
        if self.index_update_strategy == "rebuild" or self.index is None:
            self.rebuild_index()
            return

        old_retrieval_len = self.regions.retrieval.numel()
        self._indexed_length = self._length
        self._pending_update_tokens = 0
        self.regions = self._make_regions(
            self._length,
            self.index_device,
            self._indexed_length,
        )
        new_retrieval = self.regions.retrieval[old_retrieval_len:]
        if new_retrieval.numel() == 0:
            return

        next_refresh_count = self._tokens_since_codebook_refresh + new_retrieval.numel()
        if (
            self.codebook_refresh_interval is not None
            and next_refresh_count >= self.codebook_refresh_interval
        ):
            self.rebuild_index(warm_start=True, count_codebook_refresh=True)
            return

        if self.index_update_strategy in {"online", "deferred"}:
            self._online_refresh_index(
                new_retrieval_len=new_retrieval.numel(),
                next_refresh_count=next_refresh_count,
            )
            return

        list_ids = self.index.add(self.fetch_keys(new_retrieval, device=self.index_device))
        self._append_retrieval_value_summaries(
            list_ids,
            self.fetch_values(new_retrieval, device=self.index_device),
        )
        self._tokens_since_codebook_refresh = next_refresh_count
        self.index_adds += 1

    def rebuild_index(
        self,
        *,
        warm_start: bool = False,
        count_codebook_refresh: bool = False,
    ) -> None:
        """Retrain IVF-PQ metadata using all tokens old enough for retrieval."""

        self._indexed_length = self._length
        self._pending_update_tokens = 0
        self._deferred_index_update_pending = False
        self.regions = self._make_regions(
            self._length,
            self.index_device,
            self._indexed_length,
        )
        self._build_index(warm_start=warm_start)
        self._tokens_since_codebook_refresh = 0
        if count_codebook_refresh and self.index is not None:
            self.codebook_refreshes += 1

    def state_dict(self) -> dict[str, object]:
        """Return a snapshot of full K/V storage plus compressed retrieval metadata."""

        return {
            "index_config": asdict(self.index_config),
            "metadata": {
                "sink_tokens": self.sink_tokens,
                "local_window": self.local_window,
                "index_update_interval": self.index_update_interval,
                "index_update_strategy": self.index_update_strategy,
                "codebook_refresh_interval": self.codebook_refresh_interval,
                "kv_storage": self.kv_storage,
                "pin_offloaded_kv": self.pin_offloaded_kv,
                "indexed_length": self._indexed_length,
                "pending_update_tokens": self._pending_update_tokens,
                "deferred_index_update_pending": self._deferred_index_update_pending,
                "tokens_since_codebook_refresh": self._tokens_since_codebook_refresh,
                "index_rebuilds": self.index_rebuilds,
                "index_adds": self.index_adds,
                "online_codebook_updates": self.online_codebook_updates,
                "codebook_refreshes": self.codebook_refreshes,
                "deferred_index_updates": self.deferred_index_updates,
            },
            "keys": self._key_buf[:self._length].detach().clone(),
            "values": self._val_buf[:self._length].detach().clone(),
            "index": None if self.index is None else self.index.state_dict(),
            "retrieval_value_centroids": _clone_optional_tensor(self.retrieval_value_centroids),
            "retrieval_value_sums": _clone_optional_tensor(self._retrieval_value_sums),
            "retrieval_value_counts": _clone_optional_tensor(self._retrieval_value_counts),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, object],
        *,
        index_config: IVFPQConfig | None = None,
    ) -> "SparseKVCache":
        """Restore a cache snapshot without retraining IVF-PQ metadata."""

        metadata = state.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("cache state must contain a metadata dict")
        keys = _required_tensor(state, "keys").detach().clone()
        values = _required_tensor(state, "values").detach().clone()
        if keys.ndim != 2 or values.ndim != 2:
            raise ValueError("saved keys and values must be rank-2 tensors")
        if keys.shape[0] != values.shape[0]:
            raise ValueError("saved keys and values must have the same sequence length")

        if index_config is None:
            index_config_state = state.get("index_config")
            if not isinstance(index_config_state, dict):
                raise ValueError("cache state must contain an index_config dict")
            index_config = IVFPQConfig(**index_config_state)

        index_state = state.get("index")
        index = None
        if index_state is not None:
            if not isinstance(index_state, dict):
                raise ValueError("cache index state must be a dict or None")
            index = IVFPQIndex.from_state_dict(index_state, config=index_config)

        obj = cls.__new__(cls)
        obj.index_config = index_config
        obj.kv_storage = _metadata_str(metadata, "kv_storage")
        obj.pin_offloaded_kv = bool(metadata["pin_offloaded_kv"])
        obj.profile_enabled = False
        obj.profile_stats = {}
        obj.shared_coarse_centroids = None
        obj.shared_pq_codebooks = None
        obj.index_device = _infer_index_device(index, keys)
        obj.sink_tokens = int(metadata["sink_tokens"])
        obj.local_window = int(metadata["local_window"])
        obj.index_update_interval = int(metadata["index_update_interval"])
        obj.index_update_strategy = _metadata_str(metadata, "index_update_strategy")
        obj.codebook_refresh_interval = metadata["codebook_refresh_interval"]
        if obj.codebook_refresh_interval is not None:
            obj.codebook_refresh_interval = int(obj.codebook_refresh_interval)
        obj.growth_margin = 4096
        obj._length = keys.shape[0]
        obj._key_shape = tuple(keys.shape[1:])
        obj._value_shape = tuple(values.shape[1:])
        obj._indexed_length = int(metadata["indexed_length"])
        obj._pending_update_tokens = int(metadata["pending_update_tokens"])
        obj._deferred_index_update_pending = bool(
            metadata.get("deferred_index_update_pending", False)
        )
        obj._tokens_since_codebook_refresh = int(metadata["tokens_since_codebook_refresh"])
        obj.index_rebuilds = int(metadata["index_rebuilds"])
        obj.index_adds = int(metadata["index_adds"])
        obj.online_codebook_updates = int(metadata["online_codebook_updates"])
        obj.codebook_refreshes = int(metadata["codebook_refreshes"])
        obj.deferred_index_updates = int(metadata.get("deferred_index_updates", 0))
        obj.reset_fetch_stats()

        # Restore: build a preallocated buffer from the saved valid [:length] data.
        pinned_k = _pin_loaded_tensor(keys, obj.kv_storage, obj.pin_offloaded_kv)
        pinned_v = _pin_loaded_tensor(values, obj.kv_storage, obj.pin_offloaded_kv)
        cap = obj._length + obj.growth_margin
        obj._key_buf = obj._alloc_buf(cap, pinned_k)
        obj._val_buf = obj._alloc_buf(cap, pinned_v)
        obj._buf_capacity = cap
        obj._key_buf[:obj._length].copy_(pinned_k)
        obj._val_buf[:obj._length].copy_(pinned_v)
        obj.keys = obj._key_buf[:obj._length]
        obj.values = obj._val_buf[:obj._length]

        obj._cached_sink_end = -1
        obj._cached_retrieval_end = -1
        obj._cached_length = -1
        obj._cached_regions = None
        obj.regions = obj._make_regions(obj._length, obj.index_device, obj._indexed_length)
        obj.index = index
        obj.retrieval_value_centroids = _clone_optional_tensor(
            state.get("retrieval_value_centroids")
        )
        obj._retrieval_value_sums = _clone_optional_tensor(state.get("retrieval_value_sums"))
        obj._retrieval_value_counts = _clone_optional_tensor(state.get("retrieval_value_counts"))
        # Restored caches take the previous path unconditionally.
        # (The harness never restores through from_state_dict; it
        # rebuilds via __init__, which does opt into the fast path.)
        obj._fast_offload = False
        return obj

    def memory_footprint(self) -> KVCacheMemoryStats:
        """Return byte counts for full K/V storage plus IVF-PQ metadata.

        Reported key_bytes / value_bytes cover only the valid [:_length] region
        (the public self.keys / self.values views), not the preallocated capacity.
        """
        index_stats = None if self.index is None else self.index.memory_footprint()
        index_bytes = 0 if index_stats is None else index_stats.total_bytes
        retrieval_full_key_bytes = self.regions.retrieval.numel() * self._row_nbytes(self.keys)
        return KVCacheMemoryStats(
            key_bytes=_tensor_nbytes(self.keys),
            value_bytes=_tensor_nbytes(self.values),
            index_bytes=index_bytes,
            retrieval_full_key_bytes=retrieval_full_key_bytes,
            compressed_key_metadata_bytes=_compressed_key_metadata_bytes(index_stats),
        )

    def retrieval_global_indices(self, local_indices: torch.Tensor) -> torch.Tensor:
        local_indices = local_indices.to(device=self.index_device, dtype=torch.long)
        if local_indices.numel() == 0:
            return local_indices.clone()
        retrieval_start = min(self.sink_tokens, self._length)
        return local_indices + retrieval_start

    def fetch_keys(self, indices: torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
        target_device = self.index_device if device is None else torch.device(device)
        profile_start = self._profile_start(target_device)
        self.key_fetches += 1
        self.fetched_key_rows += indices.numel()
        if self._fast_eligible(target_device):
            keys, _ = self._fast_gather(indices, include_keys=True, include_values=False)
            self._profile_stop("cache_fetch_keys", profile_start, target_device)
            return keys
        fetched = self._fetch_full_tensor(self.keys, indices, device=target_device)
        self._profile_stop("cache_fetch_keys", profile_start, target_device)
        return fetched

    def fetch_values(self, indices: torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
        target_device = self.index_device if device is None else torch.device(device)
        profile_start = self._profile_start(target_device)
        self.value_fetches += 1
        self.fetched_value_rows += indices.numel()
        if self._fast_eligible(target_device):
            _, values = self._fast_gather(indices, include_keys=False, include_values=True)
            self._profile_stop("cache_fetch_values", profile_start, target_device)
            return values
        fetched = self._fetch_full_tensor(self.values, indices, device=target_device)
        self._profile_stop("cache_fetch_values", profile_start, target_device)
        return fetched

    def fetch_kv(
        self,
        indices: torch.Tensor,
        *,
        device: torch.device | None = None,
        include_keys: bool = True,
        include_values: bool = True,
    ) -> FetchedKV:
        if not include_keys and not include_values:
            raise ValueError("fetch_kv must request keys, values, or both")

        total_start = self._profile_start(device if device is not None else self.index_device)
        self.kv_fetches += 1
        target_device = self.index_device if device is None else torch.device(device)

        if self._fast_eligible(target_device):
            # One batched pinned-buffer gather for keys+values combined
            # instead of two separate CPU fancy-index lookups.
            if include_keys:
                self.key_fetches += 1
                self.fetched_key_rows += indices.numel()
            if include_values:
                self.value_fetches += 1
                self.fetched_value_rows += indices.numel()
            keys, values = self._fast_gather(
                indices, include_keys=include_keys, include_values=include_values
            )
            self._profile_stop("cache_fetch_kv_total", total_start, target_device)
            return FetchedKV(indices=indices, keys=keys, values=values)

        storage_index = indices.to(device=self.keys.device, dtype=torch.long)
        keys = None
        values = None
        if include_keys:
            profile_start = self._profile_start(target_device)
            self.key_fetches += 1
            self.fetched_key_rows += indices.numel()
            keys = self._move_fetched_tensor(self.keys[storage_index], target_device)
            self._profile_stop("cache_fetch_kv_keys", profile_start, target_device)
        if include_values:
            profile_start = self._profile_start(target_device)
            self.value_fetches += 1
            self.fetched_value_rows += indices.numel()
            value_index = storage_index
            if self.values.device != self.keys.device:
                value_index = indices.to(device=self.values.device, dtype=torch.long)
            values = self._move_fetched_tensor(self.values[value_index], target_device)
            self._profile_stop("cache_fetch_kv_values", profile_start, target_device)
        self._profile_stop("cache_fetch_kv_total", total_start, target_device)
        return FetchedKV(indices=indices, keys=keys, values=values)

    def prefetch_kv(
        self,
        indices: torch.Tensor,
        *,
        device: torch.device | None = None,
        include_keys: bool = True,
        include_values: bool = True,
    ) -> PrefetchedKV:
        target_device = self.index_device if device is None else torch.device(device)
        if not self._should_async_prefetch(target_device):
            return PrefetchedKV(
                fetched=self.fetch_kv(
                    indices,
                    device=target_device,
                    include_keys=include_keys,
                    include_values=include_values,
                ),
                stream=None,
                device=target_device,
            )

        self.kv_prefetches += 1
        stream = torch.cuda.Stream(device=target_device)
        with torch.cuda.stream(stream):
            fetched = self.fetch_kv(
                indices,
                device=target_device,
                include_keys=include_keys,
                include_values=include_values,
            )
        return PrefetchedKV(fetched=fetched, stream=stream, device=target_device)

    def reset_fetch_stats(self) -> None:
        self.kv_prefetches = 0
        self.kv_fetches = 0
        self.key_fetches = 0
        self.value_fetches = 0
        self.fetched_key_rows = 0
        self.fetched_value_rows = 0

    def reset_profile_stats(self) -> None:
        self.profile_stats.clear()
        if self.index is not None:
            self.index.reset_profile_stats()

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

    def _build_index(self, *, warm_start: bool = False) -> None:
        if self.regions.retrieval.numel() == 0:
            self.index = None
            self.retrieval_value_centroids = None
            self._retrieval_value_sums = None
            self._retrieval_value_counts = None
            return
        profile_start = self._profile_start(self.index_device)
        retrieval_keys = self.fetch_keys(self.regions.retrieval, device=self.index_device)
        if warm_start and self.index is not None:
            self.index = self.index.rebuild(retrieval_keys, warm_start=True)
        elif self.shared_coarse_centroids is not None and self.shared_pq_codebooks is not None:
            self.index = IVFPQIndex(
                self.index_config,
                profile=self.profile_enabled,
            ).build_with_codebooks(
                retrieval_keys,
                coarse_centroids=self.shared_coarse_centroids,
                pq_codebooks=self.shared_pq_codebooks,
                list_ids=self._precomputed_list_ids,
                codes=self._precomputed_codes,
            )
            self._precomputed_list_ids = None
            self._precomputed_codes = None
        else:
            self.index = IVFPQIndex(
                self.index_config,
                profile=self.profile_enabled,
            ).build(retrieval_keys)
        self.retrieval_value_centroids = self._build_retrieval_value_summaries()
        self.index_rebuilds += 1
        self._profile_stop("cache_build_index_total", profile_start, self.index_device)

    def _online_refresh_index(self, *, new_retrieval_len: int, next_refresh_count: int) -> None:
        assert self.index is not None

        profile_start = self._profile_start(self.index_device)
        retrieval_keys = self.fetch_keys(self.regions.retrieval, device=self.index_device)
        update_keys = retrieval_keys[-new_retrieval_len:]
        self.index.online_refresh(retrieval_keys, update_keys=update_keys)
        self.retrieval_value_centroids = self._build_retrieval_value_summaries()
        self._tokens_since_codebook_refresh = next_refresh_count
        self.online_codebook_updates += 1
        self._profile_stop("cache_online_refresh_total", profile_start, self.index_device)

    def _mark_deferred_index_update(self) -> None:
        if not self._deferred_index_update_pending:
            self.deferred_index_updates += 1
        self._deferred_index_update_pending = True
        self.regions = self._make_regions(
            self._length,
            self.index_device,
            self._indexed_length,
        )

    def _build_retrieval_value_summaries(self) -> torch.Tensor:
        assert self.index is not None
        assert self.index.list_ids is not None
        assert self.index.coarse_centroids is not None

        profile_start = self._profile_start(self.index_device)
        list_ids = as_index(self.index.list_ids)
        num_lists = self.index.coarse_centroids.shape[0]
        retrieval_values = self.fetch_values(self.regions.retrieval, device=self.index_device)
        if os.environ.get("PQ_HSA_BATCHED_BUILD", "0") == "1" and retrieval_values.is_cuda:
            # (opt-in): accumulate in fp32 (native atomics) instead of
            # fp16 (CAS-emulated atomics, ~10x slower at 130K rows); result cast
            # back to the value dtype.  Rounding differs from the fp16 running
            # sum at the ULP level only.
            sums32 = torch.zeros(
                num_lists, retrieval_values.shape[1], device=retrieval_values.device, dtype=torch.float32
            )
            sums32.index_add_(0, list_ids, retrieval_values.float())
            self._retrieval_value_sums = sums32.to(retrieval_values.dtype)
        else:
            self._retrieval_value_sums = torch.zeros(
                num_lists,
                retrieval_values.shape[1],
                device=retrieval_values.device,
                dtype=retrieval_values.dtype,
            )
            self._retrieval_value_sums.index_add_(0, list_ids, retrieval_values)
        self._retrieval_value_counts = torch.bincount(
            list_ids,
            minlength=num_lists,
        ).to(retrieval_values.dtype)
        centroids = self._value_centroids_from_summaries()
        # Running sum/count are only needed for incremental add. After build
        # (and after deferred/online rebuild) they are recoverable from the
        # retrieval values + list_ids, so they are dropped from the resident
        # footprint. Paper decode uses hybrid_value_mode=centroid which reads
        # retrieval_value_centroids only.
        self._retrieval_value_sums = None
        self._retrieval_value_counts = None
        self._profile_stop("cache_build_value_centroids", profile_start, self.index_device)
        return centroids

    def _append_retrieval_value_summaries(
        self,
        list_ids: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        assert self.index is not None
        assert self.index.coarse_centroids is not None
        if self._retrieval_value_sums is None or self._retrieval_value_counts is None:
            # Rebuild running stats from the full retrieval region (already
            # includes the newly added tokens), then keep them for the next add.
            retrieval_values = self.fetch_values(self.regions.retrieval, device=self.index_device)
            index_ids = as_index(self.index.list_ids)
            num_lists = self.index.coarse_centroids.shape[0]
            self._retrieval_value_sums = torch.zeros(
                num_lists,
                retrieval_values.shape[1],
                device=retrieval_values.device,
                dtype=retrieval_values.dtype,
            )
            self._retrieval_value_counts = torch.bincount(
                index_ids,
                minlength=num_lists,
            ).to(retrieval_values.dtype)
            self._retrieval_value_sums.index_add_(0, index_ids, retrieval_values)
            self.retrieval_value_centroids = self._value_centroids_from_summaries()
            return

        num_lists = self.index.coarse_centroids.shape[0]
        list_ids = as_index(list_ids)
        self._retrieval_value_sums.index_add_(0, list_ids, values)
        self._retrieval_value_counts += torch.bincount(
            list_ids,
            minlength=num_lists,
        ).to(values.dtype)
        self.retrieval_value_centroids = self._value_centroids_from_summaries()

    def _value_centroids_from_summaries(self) -> torch.Tensor:
        assert self._retrieval_value_sums is not None
        assert self._retrieval_value_counts is not None

        centroids = torch.zeros_like(self._retrieval_value_sums)
        non_empty = self._retrieval_value_counts > 0
        centroids[non_empty] = (
            self._retrieval_value_sums[non_empty]
            / self._retrieval_value_counts[non_empty].unsqueeze(1)
        )
        return centroids

    def _prepare_full_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.kv_storage == "device":
            return tensor
        stored = tensor.detach().to("cpu")
        if self.pin_offloaded_kv and torch.cuda.is_available():
            stored = stored.pin_memory()
        return stored

    def _alloc_buf(self, capacity: int, like: torch.Tensor) -> torch.Tensor:
        """Allocate a preallocated backing buffer matching dtype/device of *like*."""
        buf = torch.empty(
            (capacity, *like.shape[1:]),
            dtype=like.dtype,
            device=like.device,
        )
        if self.kv_storage == "cpu" and self.pin_offloaded_kv and torch.cuda.is_available():
            buf = buf.pin_memory()
        return buf

    def _grow_buf(self) -> None:
        """Double the backing buffer capacity (amortised O(1) append cost)."""
        new_cap = max(self._buf_capacity * 2, self._buf_capacity + self.growth_margin)
        new_k = self._alloc_buf(new_cap, self._key_buf)
        new_v = self._alloc_buf(new_cap, self._val_buf)
        new_k[:self._length].copy_(self._key_buf[:self._length])
        new_v[:self._length].copy_(self._val_buf[:self._length])
        self._key_buf = new_k
        self._val_buf = new_v
        self._buf_capacity = new_cap

    def _append_to_buf(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write one new row in-place into the preallocated buffer, growing if needed."""
        if self._length >= self._buf_capacity:
            self._grow_buf()
        # Prepare to the right storage location/dtype.
        raw_k = self._prepare_full_tensor(key)
        raw_v = self._prepare_full_tensor(value)
        self._key_buf[self._length].copy_(raw_k)
        self._val_buf[self._length].copy_(raw_v)
        # Re-slice public views to expose the new row.
        self.keys = self._key_buf[:self._length + 1]
        self.values = self._val_buf[:self._length + 1]

    def _append_full_tensor(self, existing: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """Legacy helper kept for backward compatibility; no longer used internally."""
        appended = self._prepare_full_tensor(tensor.unsqueeze(0))
        combined = torch.cat([existing, appended], dim=0)
        if self.kv_storage == "cpu" and self.pin_offloaded_kv and torch.cuda.is_available():
            combined = combined.pin_memory()
        return combined

    def _fetch_full_tensor(
        self,
        tensor: torch.Tensor,
        indices: torch.Tensor,
        *,
        device: torch.device | None,
    ) -> torch.Tensor:
        index = indices.to(device=tensor.device, dtype=torch.long)
        fetched = tensor[index]
        target_device = self.index_device if device is None else torch.device(device)
        return self._move_fetched_tensor(fetched, target_device)

    def _move_fetched_tensor(self, fetched: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        if fetched.device == target_device:
            return fetched
        return fetched.to(target_device, non_blocking=self.pin_offloaded_kv)

    def _row_nbytes(self, tensor: torch.Tensor) -> int:
        if tensor.ndim <= 1:
            return tensor.element_size()
        row_items = 1
        for size in tensor.shape[1:]:
            row_items *= size
        return row_items * tensor.element_size()

    def _should_async_prefetch(self, target_device: torch.device) -> bool:
        return (
            self.kv_storage == "cpu"
            and self.pin_offloaded_kv
            and target_device.type == "cuda"
            and torch.cuda.is_available()
        )

    # ------------------------------------------------------------------
    # Fast offload gather (opt-in, PQ_HSA_OFFLOAD_FAST=1)
    #
    # Bottleneck this replaces: the default
    # kv_storage="cpu" gather does `self.keys[storage_index]` -- CPU fancy
    # indexing that allocates a *fresh, unpinned* tensor every call -- then
    # `.to(target_device, non_blocking=True)`. non_blocking on an unpinned
    # source silently degrades to a blocking H2D copy, and this happens once
    # per layer per decode step with no batching across layers and no reused
    # buffers.
    #
    # Fast path fixes, all gated behind this flag so the default path is
    # untouched:
    #   (a) indices are D2H'd once per call (not per token) into a pinned
    #       staging buffer;
    #   (b) the CPU-side gather is a single `torch.index_select(..., out=...)`
    #       into a reused pinned buffer -- no python token loop, no
    #       per-call `pin_memory()` allocation;
    #   (c) the H2D copy runs on a dedicated copy stream with double-buffered
    #       pinned staging (2 slots) so a call's CPU-side index_select can
    #       proceed while the *previous* call's H2D copy is still draining,
    #       and the compute stream only pays a `wait_event` (not a device-wide
    #       synchronize) before touching the result;
    #   (d) sink + local are kept resident on GPU via `fetch_resident_kv`,
    #       maintained incrementally (steady state = a 1-row fast-gather for
    #       the newest local token) so they never round-trip through the CPU
    #       buffer on the decode hot path; only the PQ-retrieval-selected
    #       exact indices actually pay the CPU gather.
    #
    # Correctness invariant: every tensor `_fast_gather` returns is a *fresh*
    # allocation (never a view into the reused pinned/staging buffers), so
    # callers may hold onto it indefinitely -- e.g. `fetch_resident_kv`
    # concatenates it into `_resident_local_key/val`, which are themselves
    # independent buffers, not aliases of the double-buffered scratch.

    _FAST_NUM_SLOTS = 2

    def _fast_eligible(self, target_device: torch.device) -> bool:
        return self._fast_offload and target_device == self.index_device

    def _fast_init_state(self) -> None:
        # Separate, off-by-default opt-in (profiling showed that
        # torch's intra-op thread pool re-spinning up for every small CPU
        # gather op is part of the observed CPU-time-vs-wall-time blowup).
        # A process-wide `torch.set_num_threads` call has effects beyond this
        # cache instance (e.g. it would also affect an already-running dense
        # arm's CPU-side ops in the same process), so it is intentionally
        # gated behind its own env var rather than being implied by
        # PQ_HSA_OFFLOAD_FAST=1.
        threads_env = os.environ.get("PQ_HSA_OFFLOAD_FAST_THREADS", "")
        if threads_env:
            try:
                n_threads = int(threads_env)
            except ValueError:
                n_threads = 0
            if n_threads > 0 and torch.get_num_threads() != n_threads:
                torch.set_num_threads(n_threads)

        is_cuda = self.index_device.type == "cuda"
        self._fast_copy_stream = torch.cuda.Stream(device=self.index_device) if is_cuda else None
        n = self._FAST_NUM_SLOTS
        self._fast_capacity = [0] * n
        self._fast_idx_pin: list[torch.Tensor | None] = [None] * n
        self._fast_key_pin: list[torch.Tensor | None] = [None] * n
        self._fast_val_pin: list[torch.Tensor | None] = [None] * n
        # Event recorded when slot `i`'s H2D copy was *issued* (not waited on
        # host-side); the next call reusing slot `i` must sync on it first so
        # it never overwrites a pinned buffer the copy engine is still
        # reading from.
        self._fast_pending_event: list["torch.cuda.Event | None"] = [None] * n
        self._fast_slot = 0

        # Resident sink+local mirror (requirement (d)).
        self._resident_sink_key: torch.Tensor | None = None
        self._resident_sink_val: torch.Tensor | None = None
        self._resident_sink_end_cached = -1
        self._resident_local_key: torch.Tensor | None = None
        self._resident_local_val: torch.Tensor | None = None
        self._resident_local_cap = 0
        self._resident_local_len = 0
        self._resident_local_start = -1  # global position of resident-local row 0

        self.fast_offload_stats: dict[str, int] = {
            "fast_gather_calls": 0,
            "fast_gather_rows": 0,
            "resident_refresh_calls": 0,
            "resident_delta_rows": 0,
            "resident_rebuild_events": 0,
        }

    def _fast_ensure_capacity(self, slot: int, n: int) -> None:
        if n <= self._fast_capacity[slot]:
            return
        new_cap = max(n, self._fast_capacity[slot] * 2 if self._fast_capacity[slot] else 256)
        is_cuda = self.index_device.type == "cuda"
        idx_buf = torch.empty(new_cap, dtype=torch.long)
        key_buf = torch.empty((new_cap, *self._key_shape), dtype=self._key_buf.dtype)
        val_buf = torch.empty((new_cap, *self._value_shape), dtype=self._val_buf.dtype)
        if is_cuda:
            idx_buf = idx_buf.pin_memory()
            key_buf = key_buf.pin_memory()
            val_buf = val_buf.pin_memory()
        self._fast_idx_pin[slot] = idx_buf
        self._fast_key_pin[slot] = key_buf
        self._fast_val_pin[slot] = val_buf
        self._fast_capacity[slot] = new_cap

    def _fast_gather(
        self,
        indices: torch.Tensor,
        *,
        include_keys: bool,
        include_values: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Batched pinned-buffer gather + async H2D. Always returns fresh,
        independently-owned GPU (or CPU, in a non-CUDA self-test) tensors --
        never a view into the reused staging buffers -- so callers may retain
        the result across later `_fast_gather` calls without aliasing risk.
        """
        n = int(indices.numel())
        self.fast_offload_stats["fast_gather_calls"] += 1
        self.fast_offload_stats["fast_gather_rows"] += n
        key_out = (
            torch.empty((n, *self._key_shape), dtype=self._key_buf.dtype, device=self.index_device)
            if include_keys
            else None
        )
        val_out = (
            torch.empty((n, *self._value_shape), dtype=self._val_buf.dtype, device=self.index_device)
            if include_values
            else None
        )
        if n == 0:
            return key_out, val_out

        is_cuda = self.index_device.type == "cuda"
        slot = self._fast_slot
        self._fast_slot = (slot + 1) % self._FAST_NUM_SLOTS
        pending = self._fast_pending_event[slot]
        if pending is not None:
            pending.synchronize()
            self._fast_pending_event[slot] = None

        self._fast_ensure_capacity(slot, n)
        idx_pin = self._fast_idx_pin[slot][:n]
        indices_long = indices if indices.dtype == torch.long else indices.to(torch.long)

        if is_cuda:
            with torch.cuda.stream(self._fast_copy_stream):
                idx_pin.copy_(indices_long, non_blocking=True)
                idx_event = torch.cuda.Event(enable_timing=False)
                idx_event.record(self._fast_copy_stream)
            # Indices must be resolved on host before the CPU-side
            # index_select below -- this is a small (few KB) transfer, unlike
            # the K/V payload, so a scoped host wait here is cheap.
            idx_event.synchronize()
        else:
            idx_pin.copy_(indices_long.to(device="cpu"))

        cpu_keys = self._key_buf[: self._length]
        cpu_vals = self._val_buf[: self._length]
        if include_keys:
            torch.index_select(cpu_keys, 0, idx_pin, out=self._fast_key_pin[slot][:n])
        if include_values:
            torch.index_select(cpu_vals, 0, idx_pin, out=self._fast_val_pin[slot][:n])

        if is_cuda:
            with torch.cuda.stream(self._fast_copy_stream):
                if include_keys:
                    key_out.copy_(self._fast_key_pin[slot][:n], non_blocking=True)
                if include_values:
                    val_out.copy_(self._fast_val_pin[slot][:n], non_blocking=True)
                copy_event = torch.cuda.Event(enable_timing=False)
                copy_event.record(self._fast_copy_stream)
            # GPU-side-only dependency: the compute stream stalls at this
            # point until the copy lands, but the host keeps running (e.g. to
            # prep the *next* layer's gather) and any kernels already queued
            # on the compute stream before this point are unaffected.
            torch.cuda.current_stream(self.index_device).wait_event(copy_event)
            self._fast_pending_event[slot] = copy_event
        else:
            if include_keys:
                key_out.copy_(self._fast_key_pin[slot][:n])
            if include_values:
                val_out.copy_(self._fast_val_pin[slot][:n])

        return key_out, val_out

    def _resident_ensure_local_capacity(self, n: int) -> None:
        if self._resident_local_key is not None and n <= self._resident_local_cap:
            return
        base = self.local_window if self.local_window > 0 else n
        new_cap = max(n, base + 64, self._resident_local_cap * 2)
        new_k = torch.empty((new_cap, *self._key_shape), dtype=self._key_buf.dtype, device=self.index_device)
        new_v = torch.empty((new_cap, *self._value_shape), dtype=self._val_buf.dtype, device=self.index_device)
        if self._resident_local_key is not None and self._resident_local_len > 0:
            new_k[: self._resident_local_len].copy_(self._resident_local_key[: self._resident_local_len])
            new_v[: self._resident_local_len].copy_(self._resident_local_val[: self._resident_local_len])
        self._resident_local_key = new_k
        self._resident_local_val = new_v
        self._resident_local_cap = new_cap

    def _fast_refresh_resident(self) -> None:
        stats = self.fast_offload_stats
        stats["resident_refresh_calls"] += 1

        sink_end = min(self.sink_tokens, self._length)
        if sink_end != self._resident_sink_end_cached:
            if sink_end == 0:
                self._resident_sink_key = None
                self._resident_sink_val = None
            else:
                k, v = self._fast_gather(self.regions.sink, include_keys=True, include_values=True)
                self._resident_sink_key, self._resident_sink_val = k, v
            self._resident_sink_end_cached = sink_end

        local = self.regions.local
        new_len = int(local.numel())
        retrieval_end = self._length - new_len
        if retrieval_end != self._resident_local_start:
            # Local-window start boundary moved (an index refresh just
            # rotated some local tokens into retrieval) -- rebuild from
            # scratch. Infrequent: once per index_update_interval tokens.
            k, v = self._fast_gather(local, include_keys=True, include_values=True)
            self._resident_ensure_local_capacity(new_len)
            if new_len > 0:
                self._resident_local_key[:new_len].copy_(k)
                self._resident_local_val[:new_len].copy_(v)
            self._resident_local_len = new_len
            self._resident_local_start = retrieval_end
            stats["resident_rebuild_events"] += 1
            return

        old_len = self._resident_local_len
        delta = new_len - old_len
        if delta > 0:
            # Steady state: only the newest row(s) since the last refresh
            # need fetching -- normally exactly 1.
            k, v = self._fast_gather(local[old_len:new_len], include_keys=True, include_values=True)
            self._resident_ensure_local_capacity(new_len)
            self._resident_local_key[old_len:new_len].copy_(k)
            self._resident_local_val[old_len:new_len].copy_(v)
            self._resident_local_len = new_len
            stats["resident_delta_rows"] += delta
        elif delta < 0:
            # Window shrank without the start moving -- not expected given
            # _make_regions' monotone growth, but stay correct if it happens.
            k, v = self._fast_gather(local, include_keys=True, include_values=True)
            self._resident_ensure_local_capacity(new_len)
            if new_len > 0:
                self._resident_local_key[:new_len].copy_(k)
                self._resident_local_val[:new_len].copy_(v)
            self._resident_local_len = new_len
            stats["resident_rebuild_events"] += 1

    def fetch_resident_kv(
        self,
        *,
        device: torch.device | None = None,
        include_values: bool = True,
    ) -> FetchedKV:
        """(d): sink+local kept resident on GPU via an incrementally
        maintained mirror. Row order matches
        `cat([regions.sink, regions.local])` exactly (both are disjoint,
        strictly increasing arange ranges, so this is also what
        `_unique_preserve_order(cat([regions.sink, regions.local]))` -- i.e.
        the previous `full_indices` -- produces).
        """
        assert self._fast_offload
        target_device = self.index_device if device is None else torch.device(device)
        self._fast_refresh_resident()

        indices = torch.cat([self.regions.sink, self.regions.local], dim=0)
        local_len = self._resident_local_len
        local_key = self._resident_local_key[:local_len] if local_len > 0 else None
        local_val = self._resident_local_val[:local_len] if local_len > 0 else None

        if self._resident_sink_key is None:
            keys = local_key
            values = local_val if include_values else None
        elif local_key is None:
            keys = self._resident_sink_key
            values = self._resident_sink_val if include_values else None
        else:
            keys = torch.cat([self._resident_sink_key, local_key], dim=0)
            values = torch.cat([self._resident_sink_val, local_val], dim=0) if include_values else None

        if keys is None:
            keys = torch.empty((0, *self._key_shape), dtype=self._key_buf.dtype, device=self.index_device)
        if include_values and values is None:
            values = torch.empty((0, *self._value_shape), dtype=self._val_buf.dtype, device=self.index_device)

        if target_device != self.index_device:
            keys = keys.to(target_device)
            values = values.to(target_device) if values is not None else None

        return FetchedKV(indices=indices, keys=keys, values=values)

    def _make_regions(
        self,
        length: int,
        device: torch.device,
        indexed_length: int,
    ) -> KVCacheRegions:
        sink_end = min(self.sink_tokens, length)
        indexed_length = min(indexed_length, length)
        if self.local_window > 0:
            retrieval_end = max(sink_end, indexed_length - self.local_window)
        else:
            retrieval_end = max(sink_end, indexed_length)

        # Fast path: reuse cached region tensors when boundaries haven't changed.
        if (
            self._cached_regions is not None
            and self._cached_sink_end == sink_end
            and self._cached_retrieval_end == retrieval_end
            and self._cached_length == length
        ):
            return self._cached_regions

        # Partial rebuild: reuse unchanged tensors, rebuild only what changed.
        cached = self._cached_regions
        if cached is not None and self._cached_sink_end == sink_end:
            sink = cached.sink
        else:
            sink = torch.arange(0, sink_end, device=device, dtype=torch.long)

        if cached is not None and self._cached_sink_end == sink_end and self._cached_retrieval_end == retrieval_end:
            retrieval = cached.retrieval
        else:
            retrieval = torch.arange(sink_end, retrieval_end, device=device, dtype=torch.long)

        local = torch.arange(retrieval_end, length, device=device, dtype=torch.long)

        regions = KVCacheRegions(sink=sink, retrieval=retrieval, local=local)
        self._cached_sink_end = sink_end
        self._cached_retrieval_end = retrieval_end
        self._cached_length = length
        self._cached_regions = regions
        return regions


def _clone_optional_tensor(value: object) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError("optional tensor state values must be tensors or None")
    return value.detach().clone()


def _required_tensor(state: dict[str, object], name: str) -> torch.Tensor:
    value = state.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"cache state must contain tensor '{name}'")
    return value


def _metadata_str(metadata: dict[str, object], name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str):
        raise ValueError(f"metadata['{name}'] must be a string")
    if name == "kv_storage" and value not in {"device", "cpu"}:
        raise ValueError("metadata['kv_storage'] must be 'device' or 'cpu'")
    if name == "index_update_strategy" and value not in {
        "incremental",
        "online",
        "rebuild",
        "deferred",
    }:
        raise ValueError(
            "metadata['index_update_strategy'] must be 'incremental', "
            "'online', 'rebuild', or 'deferred'"
        )
    return value


def _infer_index_device(index: IVFPQIndex | None, keys: torch.Tensor) -> torch.device:
    if index is not None and index.coarse_centroids is not None:
        return index.coarse_centroids.device
    return keys.device


def _pin_loaded_tensor(tensor: torch.Tensor, kv_storage: str, pin_offloaded_kv: bool) -> torch.Tensor:
    if (
        kv_storage == "cpu"
        and pin_offloaded_kv
        and tensor.device.type == "cpu"
        and torch.cuda.is_available()
        and not tensor.is_pinned()
    ):
        return tensor.pin_memory()
    return tensor


def _compressed_key_metadata_bytes(index_stats: IVFPQMemoryStats | None) -> int:
    if index_stats is None:
        return 0
    return index_stats.total_bytes


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
