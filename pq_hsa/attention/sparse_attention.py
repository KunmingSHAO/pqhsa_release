from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
import time

import torch

from pq_hsa.attention.kv_cache import FetchedKV, PrefetchedKV, SparseKVCache
from pq_hsa.index.ivfpq import IVFPQConfig, SearchResult, as_index, gather_ids
from pq_hsa.kernels.triton_lut_scan import is_triton_available, list_exp_sums_triton


@dataclass(slots=True)
class SparseAttentionConfig:
    sink_tokens: int = 0
    local_window: int = 0
    retrieval_topk: int = 100
    retrieval_top_fraction: float | None = None
    retrieval_top_p: float | None = None
    retrieval_top_p_scope: str = "retrieval"
    nprobe: int | None = None
    candidate_budget: int | None = None
    exact_rerank: bool = True
    scale: float | None = None
    mode: str = "sparse"
    hybrid_value_mode: str = "denominator"
    hybrid_topk_source: str = "all_pq"
    hybrid_denominator_source: str = "all_pq"
    index_update_interval: int = 1
    index_update_strategy: str = "incremental"
    codebook_refresh_interval: int | None = None
    kv_storage: str = "device"
    pin_offloaded_kv: bool = False
    prefetch_full_kv: bool = False
    collect_attention_details: bool = True
    profile_attention_components: bool = False


@dataclass(slots=True)
class AttentionOutput:
    output: torch.Tensor
    indices: torch.Tensor
    logits: torch.Tensor
    weights: torch.Tensor
    approx_retrieval_logits: torch.Tensor
    exact_retrieval_logits: torch.Tensor | None
    search_result: SearchResult | None
    denominator_indices: torch.Tensor | None = None
    denominator_logits: torch.Tensor | None = None
    denominator_weights: torch.Tensor | None = None
    background_weight_mass: torch.Tensor | None = None
    background_output: torch.Tensor | None = None
    selected_count: int | None = None
    denominator_count: int | None = None
    approx_retrieval_count: int | None = None
    exact_retrieval_count: int | None = None
    candidate_count: int | None = None
    probed_list_count: int | None = None


def dense_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    scale: float | None = None,
) -> AttentionOutput:
    if query.ndim != 1:
        raise ValueError("dense_attention currently expects a single query [D]")
    if keys.ndim != 2 or values.ndim != 2:
        raise ValueError("keys and values must be rank-2 tensors")
    if keys.shape[0] != values.shape[0] or keys.shape[1] != query.shape[0]:
        raise ValueError("incompatible query, key, and value shapes")

    scale_value = (1.0 / sqrt(query.shape[-1])) if scale is None else scale
    logits = keys.matmul(query) * scale_value
    weights = torch.softmax(logits, dim=-1)
    output = weights.matmul(values)
    indices = torch.arange(keys.shape[0], device=keys.device, dtype=torch.long)
    return AttentionOutput(
        output=output,
        indices=indices,
        logits=logits,
        weights=weights,
        approx_retrieval_logits=torch.empty(0, device=query.device, dtype=query.dtype),
        exact_retrieval_logits=None,
        search_result=None,
        denominator_indices=indices,
        denominator_logits=logits,
        denominator_weights=weights,
    )


class IVFPQSparseAttention:
    """Sparse attention using IVF-PQ approximate logits plus optional exact rerank."""

    def __init__(self, index_config: IVFPQConfig, attention_config: SparseAttentionConfig):
        self.index_config = index_config
        self.attention_config = attention_config
        self.profile_stats: dict[str, float] = {}
        self.cache: SparseKVCache | None = None
        if attention_config.mode not in {"sparse", "hybrid"}:
            raise ValueError("mode must be 'sparse' or 'hybrid'")
        if attention_config.hybrid_value_mode not in {"denominator", "centroid", "full"}:
            raise ValueError("hybrid_value_mode must be 'denominator', 'centroid', or 'full'")
        if attention_config.hybrid_topk_source not in {"all_pq", "ivf_candidates"}:
            raise ValueError("hybrid_topk_source must be 'all_pq' or 'ivf_candidates'")
        if attention_config.hybrid_denominator_source not in {"all_pq", "ivf_candidates"}:
            raise ValueError(
                "hybrid_denominator_source must be 'all_pq' or 'ivf_candidates'"
            )
        if (
            attention_config.hybrid_denominator_source == "ivf_candidates"
            and attention_config.hybrid_topk_source != "ivf_candidates"
        ):
            raise ValueError(
                "hybrid_denominator_source='ivf_candidates' requires "
                "hybrid_topk_source='ivf_candidates'"
            )
        if attention_config.retrieval_top_p_scope not in {"retrieval", "denominator"}:
            raise ValueError("retrieval_top_p_scope must be 'retrieval' or 'denominator'")
        if (
            attention_config.retrieval_top_p_scope == "denominator"
            and attention_config.mode != "hybrid"
        ):
            raise ValueError("retrieval_top_p_scope='denominator' requires mode='hybrid'")
        if attention_config.index_update_strategy not in {
            "incremental",
            "online",
            "rebuild",
            "deferred",
        }:
            raise ValueError(
                "index_update_strategy must be 'incremental', 'online', 'rebuild', or 'deferred'"
            )
        if attention_config.kv_storage not in {"device", "cpu"}:
            raise ValueError("kv_storage must be 'device' or 'cpu'")
        if (
            attention_config.codebook_refresh_interval is not None
            and attention_config.codebook_refresh_interval <= 0
        ):
            raise ValueError("codebook_refresh_interval must be positive when provided")
        if attention_config.retrieval_top_fraction is not None and not (
            0.0 < attention_config.retrieval_top_fraction <= 1.0
        ):
            raise ValueError("retrieval_top_fraction must be in (0, 1]")
        if attention_config.retrieval_top_p is not None and not (
            0.0 < attention_config.retrieval_top_p <= 1.0
        ):
            raise ValueError("retrieval_top_p must be in (0, 1]")
        if (
            attention_config.retrieval_top_fraction is not None
            and attention_config.retrieval_top_p is not None
        ):
            raise ValueError("retrieval_top_fraction and retrieval_top_p are mutually exclusive")
        if (
            attention_config.hybrid_denominator_source == "ivf_candidates"
            and attention_config.retrieval_top_p is not None
        ):
            raise ValueError(
                "hybrid_denominator_source='ivf_candidates' does not support retrieval_top_p"
            )

    def _profile_start(self, device: torch.device) -> float | None:
        if not self.attention_config.profile_attention_components:
            return None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _profile_stop(
        self,
        name: str,
        start: float | None,
        device: torch.device,
    ) -> None:
        if start is None:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.profile_stats[f"{name}_ms"] = self.profile_stats.get(f"{name}_ms", 0.0) + elapsed_ms
        self.profile_stats[f"{name}_calls"] = self.profile_stats.get(f"{name}_calls", 0.0) + 1.0

    def build_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        shared_coarse_centroids: torch.Tensor | None = None,
        shared_pq_codebooks: torch.Tensor | None = None,
        _key_buf: torch.Tensor | None = None,
        _val_buf: torch.Tensor | None = None,
        _precomputed_list_ids: torch.Tensor | None = None,
        _precomputed_codes: torch.Tensor | None = None,
    ) -> "IVFPQSparseAttention":
        self.cache = SparseKVCache(
            keys,
            values,
            index_config=self.index_config,
            sink_tokens=self.attention_config.sink_tokens,
            local_window=self.attention_config.local_window,
            index_update_interval=self.attention_config.index_update_interval,
            index_update_strategy=self.attention_config.index_update_strategy,
            codebook_refresh_interval=self.attention_config.codebook_refresh_interval,
            kv_storage=self.attention_config.kv_storage,
            pin_offloaded_kv=self.attention_config.pin_offloaded_kv,
            profile=self.attention_config.profile_attention_components,
            shared_coarse_centroids=shared_coarse_centroids,
            shared_pq_codebooks=shared_pq_codebooks,
            _key_buf=_key_buf,
            _val_buf=_val_buf,
            _precomputed_list_ids=_precomputed_list_ids,
            _precomputed_codes=_precomputed_codes,
        )
        return self

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self._check_cache()
        assert self.cache is not None
        self.cache.append(key, value)

    def apply_pending_index_update(self) -> bool:
        self._check_cache()
        assert self.cache is not None
        return self.cache.apply_pending_index_update()

    def forward(self, query: torch.Tensor) -> AttentionOutput:
        self._check_cache()
        assert self.cache is not None

        if query.ndim != 1:
            raise ValueError("IVFPQSparseAttention currently expects a single query [D]")

        scale_value = (
            (1.0 / sqrt(query.shape[-1]))
            if self.attention_config.scale is None
            else self.attention_config.scale
        )
        if self.attention_config.mode == "hybrid":
            return self._forward_hybrid(query, scale_value)
        return self._forward_sparse(query, scale_value)

    def forward_many(self, queries: torch.Tensor) -> tuple[AttentionOutput, ...]:
        """Run multiple queries against the same cache.

        The exact K/V materialization can differ per query, so this returns one
        ``AttentionOutput`` per query. When the current mode needs full
        retrieval-zone PQ scores, those scores are computed once as
        ``[num_queries, retrieval_tokens]`` and reused by the per-query path.
        """

        self._check_cache()
        assert self.cache is not None

        if queries.ndim != 2:
            raise ValueError("forward_many expects queries [Q, D]")
        if tuple(queries.shape[1:]) != self.cache._key_shape:
            raise ValueError(
                f"forward_many query dim must be {self.cache._key_shape}, "
                f"got {tuple(queries.shape[1:])}"
            )
        scale_value = (
            (1.0 / sqrt(queries.shape[-1]))
            if self.attention_config.scale is None
            else self.attention_config.scale
        )
        profile_start = self._profile_start(queries.device)
        precomputed = self._batched_retrieval_scores_for_queries(queries)
        self._profile_stop("batched_retrieval_scores", profile_start, queries.device)
        skip_search_results = (
            self.attention_config.mode == "hybrid"
            and self.attention_config.retrieval_top_p is None
            and self.attention_config.hybrid_topk_source == "all_pq"
            and self.attention_config.hybrid_denominator_source == "all_pq"
            and not self.attention_config.collect_attention_details
            and self.cache.index is not None
            and self.cache.regions.retrieval.numel() > 0
        )
        precomputed_search = None
        if not skip_search_results:
            profile_start = self._profile_start(queries.device)
            precomputed_search = self._batched_search_results_for_queries(
                queries,
                precomputed_approx_raw_all=precomputed,
                scale_value=scale_value,
            )
            self._profile_stop("batched_search_results", profile_start, queries.device)
        elif self.attention_config.profile_attention_components:
            self.profile_stats["batched_search_results_ms"] = self.profile_stats.get(
                "batched_search_results_ms",
                0.0,
            )
            self.profile_stats["batched_search_results_calls"] = self.profile_stats.get(
                "batched_search_results_calls",
                0.0,
            )
        if self._can_share_sparse_forward_many_fetch():
            return self._forward_many_sparse_shared_kv(
                queries,
                scale_value,
                precomputed_search_results=precomputed_search,
            )
        if self._can_share_hybrid_forward_many_fetch():
            return self._forward_many_hybrid_shared_kv(
                queries,
                scale_value,
                precomputed_approx_raw_all=precomputed,
                precomputed_search_results=precomputed_search,
            )

        outputs = []
        for row, query in enumerate(queries):
            approx_raw_all = None if precomputed is None else precomputed[row]
            search_result = None if precomputed_search is None else precomputed_search[row]
            if self.attention_config.mode == "hybrid":
                outputs.append(
                    self._forward_hybrid(
                        query,
                        scale_value,
                        precomputed_approx_raw_all=approx_raw_all,
                        precomputed_search_result=search_result,
                    )
                )
            else:
                outputs.append(
                    self._forward_sparse(
                        query,
                        scale_value,
                        precomputed_approx_raw_all=approx_raw_all,
                        precomputed_search_result=search_result,
                    )
                )
        return tuple(outputs)

    def _forward_many_hybrid_shared_kv(
        self,
        queries: torch.Tensor,
        scale_value: float,
        *,
        precomputed_approx_raw_all: torch.Tensor | None,
        precomputed_search_results: tuple[SearchResult, ...] | None,
    ) -> tuple[AttentionOutput, ...]:
        assert self.cache is not None
        assert self.cache.index is not None

        if queries.shape[0] == 0:
            return ()
        retrieval_len = self.cache.regions.retrieval.numel()
        uses_candidate_denominator = self._uses_ivf_candidate_denominator()
        if precomputed_approx_raw_all is None and not uses_candidate_denominator:
            profile_start = self._profile_start(queries.device)
            precomputed_approx_raw_all = self.cache.index.approximate_scores(queries)
            self._profile_stop("approx_scores", profile_start, queries.device)
        if (
            precomputed_approx_raw_all is not None
            and precomputed_approx_raw_all.shape != (queries.shape[0], retrieval_len)
        ):
            raise ValueError(
                "precomputed_approx_raw_all must be "
                f"[{queries.shape[0]}, {retrieval_len}], got {tuple(precomputed_approx_raw_all.shape)}"
            )

        # Cache regions are constructed as disjoint contiguous ranges.
        full_indices = torch.cat([self.cache.regions.sink, self.cache.regions.local], dim=0)
        if (
            not uses_candidate_denominator
            and self.attention_config.hybrid_topk_source == "all_pq"
            and precomputed_approx_raw_all is not None
            and (
                precomputed_search_results is not None
                or not self.attention_config.collect_attention_details
            )
        ):
            return self._forward_many_hybrid_all_pq_shared_kv_fast(
                queries=queries,
                scale_value=scale_value,
                full_indices=full_indices,
                precomputed_approx_raw_all=precomputed_approx_raw_all,
                precomputed_search_results=precomputed_search_results,
            )
        per_query_search = []
        per_query_exact_local = []
        per_query_exact_global = []
        per_query_materialized = []
        per_query_approx_logits = []
        per_query_denominator_local = []
        per_query_denominator_logits = []
        profile_start = self._profile_start(queries.device)
        for row, query in enumerate(queries):
            approx_raw_all = None if precomputed_approx_raw_all is None else precomputed_approx_raw_all[row]
            approx_logits_all = (
                torch.empty(0, device=query.device, dtype=query.dtype)
                if approx_raw_all is None
                else approx_raw_all * scale_value
            )
            topk = self._resolve_retrieval_topk(
                retrieval_len,
                approx_logits=None if approx_logits_all.numel() == 0 else approx_logits_all,
            )
            search_result = None
            if topk > 0:
                current_precomputed_search = (
                    None if precomputed_search_results is None else precomputed_search_results[row]
                )
                if (
                    uses_candidate_denominator
                    and approx_raw_all is None
                ):
                    if current_precomputed_search is not None:
                        search_result = current_precomputed_search
                    else:
                        search_result = self.cache.index.search(
                            query,
                            topk=topk,
                            nprobe=self.attention_config.nprobe,
                            candidate_budget=self.attention_config.candidate_budget,
                        )
                else:
                    assert approx_raw_all is not None
                    search_result = self._select_hybrid_exact_tokens(
                        query=query,
                        topk=topk,
                        approx_raw_all=approx_raw_all,
                        approx_logits_all=approx_logits_all,
                        precomputed_search_result=current_precomputed_search,
                    )
                exact_local = search_result.indices
                exact_global = self.cache.retrieval_global_indices(exact_local)
            else:
                exact_local = torch.empty(0, device=query.device, dtype=torch.long)
                exact_global = torch.empty(0, device=query.device, dtype=torch.long)

            if uses_candidate_denominator and search_result is not None:
                denominator_local = search_result.candidate_indices.to(
                    device=query.device,
                    dtype=torch.long,
                )
                denominator_logits = search_result.candidate_scores.to(query.device) * scale_value
                approx_logits_for_output = denominator_logits
            else:
                denominator_local = None
                denominator_logits = approx_logits_all
                approx_logits_for_output = approx_logits_all

            per_query_search.append(search_result)
            per_query_exact_local.append(exact_local)
            per_query_exact_global.append(exact_global)
            # Sink/local regions are disjoint from retrieval, and top-k indices
            # are unique, so this concat is already a unique materialization set.
            per_query_materialized.append(torch.cat([full_indices, exact_global], dim=0))
            per_query_approx_logits.append(approx_logits_for_output)
            per_query_denominator_local.append(denominator_local)
            per_query_denominator_logits.append(denominator_logits)
        self._profile_stop("select_tokens", profile_start, queries.device)

        profile_start = self._profile_start(queries.device)
        union_exact_global = _unique_preserve_order(torch.cat(per_query_exact_global, dim=0))
        union_materialized = torch.cat([full_indices, union_exact_global], dim=0)
        self._profile_stop("materialize_union", profile_start, queries.device)
        profile_start = self._profile_start(queries.device)
        fetched = self.cache.fetch_kv(union_materialized, device=queries.device)
        self._profile_stop("fetch_kv", profile_start, queries.device)
        assert fetched.keys is not None
        assert fetched.values is not None

        exact_counts = {int(indices.numel()) for indices in per_query_exact_local}
        candidate_denominator_counts = {
            int(indices.numel())
            for indices in per_query_denominator_local
            if indices is not None
        }
        if (
            uses_candidate_denominator
            and len(exact_counts) == 1
            and len(candidate_denominator_counts) == 1
            and not self.attention_config.collect_attention_details
        ):
            candidate_count = next(iter(candidate_denominator_counts))
            denominator_local_tensor = torch.stack(
                [
                    indices
                    for indices in per_query_denominator_local
                    if indices is not None
                ],
                dim=0,
            )
            denominator_logits_tensor = torch.stack(per_query_denominator_logits, dim=0)
            exact_positions_tensor = torch.stack(
                [
                    _positions(denominator_local, exact_local)
                    for denominator_local, exact_local in zip(
                        per_query_denominator_local,
                        per_query_exact_local,
                        strict=True,
                    )
                    if denominator_local is not None
                ],
                dim=0,
            )
            approx_exact_logits_tensor = (
                denominator_logits_tensor.gather(1, exact_positions_tensor)
                if exact_positions_tensor.numel() > 0
                else None
            )
            return self._forward_many_hybrid_shared_kv_vectorized(
                queries=queries,
                scale_value=scale_value,
                full_indices=full_indices,
                union_materialized=union_materialized,
                fetched=fetched,
                denominator_indices=None,
                per_query_search=per_query_search,
                per_query_exact_local=None,
                per_query_exact_global=None,
                per_query_approx_logits=None,
                exact_local_tensor=exact_positions_tensor,
                exact_global_tensor=torch.stack(per_query_exact_global, dim=0),
                approx_logits_all_tensor=denominator_logits_tensor,
                approx_exact_logits_tensor=approx_exact_logits_tensor,
                candidate_count_override=candidate_count,
                probed_list_count_override=self.attention_config.nprobe,
                denominator_count_override=int(full_indices.numel() + candidate_count),
                denominator_local_tensor=denominator_local_tensor,
            )
        if len(exact_counts) == 1 and not uses_candidate_denominator:
            denominator_indices = torch.cat([full_indices, self.cache.regions.retrieval], dim=0)
            return self._forward_many_hybrid_shared_kv_vectorized(
                queries=queries,
                scale_value=scale_value,
                full_indices=full_indices,
                union_materialized=union_materialized,
                fetched=fetched,
                denominator_indices=denominator_indices,
                per_query_search=per_query_search,
                per_query_exact_local=per_query_exact_local,
                per_query_exact_global=per_query_exact_global,
                per_query_approx_logits=per_query_approx_logits,
            )

        outputs = []
        for (
            query,
            search_result,
            exact_local,
            exact_global,
            exact_materialized,
            approx_logits_all,
            denominator_local,
            denominator_retrieval_logits,
        ) in zip(
            queries,
            per_query_search,
            per_query_exact_local,
            per_query_exact_global,
            per_query_materialized,
            per_query_approx_logits,
            per_query_denominator_local,
            per_query_denominator_logits,
            strict=True,
        ):
            materialized_positions_in_union = _positions(union_materialized, exact_materialized)
            materialized_keys = fetched.keys[materialized_positions_in_union]
            materialized_values = fetched.values[materialized_positions_in_union]
            materialized_key_logits = materialized_keys.matmul(query) * scale_value

            full_pos_in_materialized = _positions(exact_materialized, full_indices)
            full_logits = materialized_key_logits[full_pos_in_materialized]
            exact_pos_in_materialized = _positions(exact_materialized, exact_global)
            exact_retrieval_logits = None
            if self.attention_config.exact_rerank and exact_pos_in_materialized.numel() > 0:
                exact_retrieval_logits = materialized_key_logits[exact_pos_in_materialized]

            hybrid_retrieval_logits = denominator_retrieval_logits.clone()
            if exact_retrieval_logits is not None and exact_retrieval_logits.numel() > 0:
                if denominator_local is None:
                    denominator_local = torch.arange(
                        retrieval_len,
                        device=query.device,
                        dtype=torch.long,
                    )
                exact_positions_in_denominator = _positions(denominator_local, exact_local)
                hybrid_retrieval_logits[exact_positions_in_denominator] = exact_retrieval_logits

            if denominator_local is None:
                denominator_local = torch.arange(
                    retrieval_len,
                    device=query.device,
                    dtype=torch.long,
                )
            denominator_retrieval_indices = self.cache.retrieval_global_indices(denominator_local)
            denominator_indices = torch.cat([full_indices, denominator_retrieval_indices], dim=0)
            denominator_logits = torch.cat([full_logits, hybrid_retrieval_logits], dim=0)
            denominator_weights = torch.softmax(denominator_logits, dim=-1)
            materialized_positions = _positions(denominator_indices, exact_materialized)
            materialized_weights = denominator_weights[materialized_positions]
            materialized_logits = denominator_logits[materialized_positions]

            if self.attention_config.hybrid_value_mode == "centroid":
                output, background_weight_mass, background_output = self._hybrid_centroid_output(
                    full_indices=full_indices,
                    exact_local_indices=exact_local,
                    exact_global_indices=exact_global,
                    exact_materialized=exact_materialized,
                    materialized_values=materialized_values,
                    denominator_weights=denominator_weights,
                    retrieval_denominator_local_indices=denominator_local,
                )
            else:
                output = materialized_weights.matmul(materialized_values)
                background_weight_mass = None
                background_output = None

            outputs.append(
                AttentionOutput(
                    output=output,
                    indices=exact_materialized,
                    logits=materialized_logits,
                    weights=materialized_weights,
                    approx_retrieval_logits=approx_logits_all,
                    exact_retrieval_logits=exact_retrieval_logits,
                    search_result=search_result,
                    denominator_indices=denominator_indices,
                    denominator_logits=denominator_logits,
                    denominator_weights=denominator_weights,
                    background_weight_mass=background_weight_mass,
                    background_output=background_output,
                )
            )
        return tuple(outputs)

    def _forward_many_hybrid_all_pq_shared_kv_fast(
        self,
        *,
        queries: torch.Tensor,
        scale_value: float,
        full_indices: torch.Tensor,
        precomputed_approx_raw_all: torch.Tensor,
        precomputed_search_results: tuple[SearchResult, ...] | None,
    ) -> tuple[AttentionOutput, ...]:
        assert self.cache is not None
        assert self.cache.index is not None

        profile_start = self._profile_start(queries.device)
        approx_logits_all = precomputed_approx_raw_all * scale_value
        if precomputed_search_results is None:
            retrieval_len = precomputed_approx_raw_all.shape[1]
            topk = self._resolve_retrieval_topk(retrieval_len)
            if topk > 0:
                need_approx_topk_values = (
                    not self.attention_config.exact_rerank
                    or
                    self.attention_config.hybrid_value_mode == "centroid"
                    and not self.attention_config.collect_attention_details
                )
                sort_topk_values = (
                    self.attention_config.hybrid_value_mode == "centroid"
                    and not self.attention_config.collect_attention_details
                )
                exact_approx_logits, exact_local = torch.topk(
                    approx_logits_all,
                    k=topk,
                    dim=1,
                    sorted=sort_topk_values,
                )
                if not need_approx_topk_values:
                    exact_approx_logits = None
            else:
                exact_local = torch.empty(
                    queries.shape[0],
                    0,
                    device=queries.device,
                    dtype=torch.long,
                )
                exact_approx_logits = torch.empty(
                    queries.shape[0],
                    0,
                    device=queries.device,
                    dtype=approx_logits_all.dtype,
                )
            per_query_search = [None for _ in range(queries.shape[0])]
            candidate_count_override = int(retrieval_len)
            probed_list_count_override = 0
        else:
            per_query_search = list(precomputed_search_results)
            exact_local = torch.stack([result.indices for result in per_query_search], dim=0)
            exact_approx_logits = None
            candidate_count_override = None
            probed_list_count_override = None
        exact_global = self.cache.retrieval_global_indices(exact_local)
        self._profile_stop("select_tokens", profile_start, queries.device)

        profile_start = self._profile_start(queries.device)
        exact_rows_are_batched = not self.attention_config.collect_attention_details
        if exact_rows_are_batched:
            materialized_exact = exact_global.reshape(-1)
            union_materialized = torch.cat([full_indices, materialized_exact], dim=0)
        else:
            union_exact_global = torch.unique(exact_global.reshape(-1), sorted=True)
            union_materialized = torch.cat([full_indices, union_exact_global], dim=0)
        self._profile_stop("materialize_union", profile_start, queries.device)

        profile_start = self._profile_start(queries.device)
        skip_exact_key_fetch = (
            not self.attention_config.exact_rerank
            and not self.attention_config.collect_attention_details
            and self.attention_config.hybrid_value_mode == "denominator"
            and exact_rows_are_batched
        )
        if skip_exact_key_fetch:
            fetched_full = self.cache.fetch_kv(full_indices, device=queries.device)
            assert fetched_full.keys is not None
            assert fetched_full.values is not None
            if materialized_exact.numel() > 0:
                exact_values = self.cache.fetch_values(materialized_exact, device=queries.device)
                fetched_values = torch.cat([fetched_full.values, exact_values], dim=0)
            else:
                fetched_values = fetched_full.values
            fetched = FetchedKV(
                indices=union_materialized,
                keys=fetched_full.keys,
                values=fetched_values,
            )
        else:
            fetched = self.cache.fetch_kv(union_materialized, device=queries.device)
        self._profile_stop("fetch_kv", profile_start, queries.device)
        assert fetched.keys is not None
        assert fetched.values is not None

        denominator_count = int(full_indices.numel() + precomputed_approx_raw_all.shape[1])
        denominator_indices = (
            None
            if not self.attention_config.collect_attention_details
            else torch.cat([full_indices, self.cache.regions.retrieval], dim=0)
        )
        return self._forward_many_hybrid_shared_kv_vectorized(
            queries=queries,
            scale_value=scale_value,
            full_indices=full_indices,
            union_materialized=union_materialized,
            fetched=fetched,
            denominator_indices=denominator_indices,
            per_query_search=per_query_search,
            per_query_exact_local=None,
            per_query_exact_global=None,
            per_query_approx_logits=None,
            exact_local_tensor=exact_local,
            exact_global_tensor=exact_global,
            approx_logits_all_tensor=approx_logits_all,
            approx_exact_logits_tensor=exact_approx_logits,
            candidate_count_override=candidate_count_override,
            probed_list_count_override=probed_list_count_override,
            denominator_count_override=denominator_count,
            union_exact_global_sorted=not exact_rows_are_batched,
            exact_rows_are_batched=exact_rows_are_batched,
        )

    def _forward_many_hybrid_shared_kv_vectorized(
        self,
        *,
        queries: torch.Tensor,
        scale_value: float,
        full_indices: torch.Tensor,
        union_materialized: torch.Tensor,
        fetched: FetchedKV,
        denominator_indices: torch.Tensor | None,
        per_query_search: list[SearchResult | None],
        per_query_exact_local: list[torch.Tensor] | None,
        per_query_exact_global: list[torch.Tensor] | None,
        per_query_approx_logits: list[torch.Tensor] | None,
        exact_local_tensor: torch.Tensor | None = None,
        exact_global_tensor: torch.Tensor | None = None,
        approx_logits_all_tensor: torch.Tensor | None = None,
        approx_exact_logits_tensor: torch.Tensor | None = None,
        candidate_count_override: int | None = None,
        probed_list_count_override: int | None = None,
        denominator_count_override: int | None = None,
        denominator_local_tensor: torch.Tensor | None = None,
        union_exact_global_sorted: bool = False,
        exact_rows_are_batched: bool = False,
    ) -> tuple[AttentionOutput, ...]:
        assert self.cache is not None
        assert self.cache.index is not None
        assert fetched.keys is not None
        assert fetched.values is not None

        query_count = queries.shape[0]
        full_count = full_indices.numel()
        profile_start = self._profile_start(queries.device)
        if approx_logits_all_tensor is None:
            if per_query_approx_logits is None:
                raise RuntimeError("per_query_approx_logits or approx_logits_all_tensor is required")
            else:
                approx_logits_all = torch.stack(per_query_approx_logits, dim=0)
        else:
            approx_logits_all = approx_logits_all_tensor
        approx_score_shape = approx_logits_all.shape
        approx_score_dtype = approx_logits_all.dtype
        if exact_local_tensor is None:
            if per_query_exact_local is None:
                raise RuntimeError("per_query_exact_local or exact_local_tensor is required")
            exact_count = per_query_exact_local[0].numel() if per_query_exact_local else 0
        else:
            exact_count = exact_local_tensor.shape[1]

        # Fast paths build union_materialized with the full region as a prefix.
        full_keys = fetched.keys[:full_count]
        full_values = fetched.values[:full_count]
        full_logits = queries.matmul(full_keys.T) * scale_value
        self._profile_stop("vectorized_full_logits", profile_start, queries.device)

        if exact_count > 0:
            profile_start = self._profile_start(queries.device)
            if exact_local_tensor is None:
                assert per_query_exact_local is not None
                assert per_query_exact_global is not None
                exact_local = torch.stack(per_query_exact_local, dim=0)
                exact_global = torch.stack(per_query_exact_global, dim=0)
            else:
                exact_local = exact_local_tensor
                if exact_global_tensor is None:
                    raise RuntimeError("exact_global_tensor is required with exact_local_tensor")
                exact_global = exact_global_tensor
            if exact_rows_are_batched:
                exact_values = fetched.values[full_count:].reshape(
                    query_count,
                    exact_count,
                    fetched.values.shape[-1],
                )
                if self.attention_config.exact_rerank:
                    exact_keys = fetched.keys[full_count:].reshape(
                        query_count,
                        exact_count,
                        fetched.keys.shape[-1],
                    )
            else:
                union_exact_global = union_materialized[full_count:]
                exact_flat = exact_global.reshape(-1)
                if union_exact_global_sorted:
                    exact_positions = torch.searchsorted(union_exact_global, exact_flat) + full_count
                else:
                    exact_positions = _positions(union_exact_global, exact_flat) + full_count
                exact_positions = exact_positions.reshape(query_count, exact_count)
                exact_values = fetched.values[exact_positions]
                if self.attention_config.exact_rerank:
                    exact_keys = fetched.keys[exact_positions]
            if self.attention_config.exact_rerank:
                exact_retrieval_logits = torch.bmm(
                    exact_keys,
                    queries.unsqueeze(-1),
                ).squeeze(-1) * scale_value
            elif approx_exact_logits_tensor is None:
                exact_retrieval_logits = approx_logits_all.gather(1, exact_local)
            else:
                exact_retrieval_logits = approx_exact_logits_tensor
            self._profile_stop("vectorized_exact_logits", profile_start, queries.device)
        else:
            profile_start = self._profile_start(queries.device)
            exact_local = torch.empty(
                query_count,
                0,
                device=queries.device,
                dtype=torch.long,
            )
            exact_global = torch.empty(
                query_count,
                0,
                device=queries.device,
                dtype=torch.long,
            )
            exact_values = torch.empty(
                query_count,
                0,
                fetched.values.shape[-1],
                device=queries.device,
                dtype=fetched.values.dtype,
            )
            exact_retrieval_logits = torch.empty(
                query_count,
                0,
                device=queries.device,
                dtype=approx_score_dtype,
            )
            self._profile_stop("vectorized_exact_logits", profile_start, queries.device)

        if (
            self.attention_config.hybrid_value_mode == "centroid"
            and not self.attention_config.collect_attention_details
        ):
            profile_start = self._profile_start(queries.device)
            if self.cache.retrieval_value_centroids is None:
                raise RuntimeError("hybrid_value_mode='centroid' requires retrieval value centroids")
            row_max = None
            if full_count > 0:
                row_max = full_logits.max(dim=1).values
            if approx_score_shape[1] > 0:
                if (
                    approx_exact_logits_tensor is not None
                    and approx_exact_logits_tensor.shape[1] > 0
                ):
                    # torch.topk(..., sorted=True) returns the approximate
                    # retrieval maximum in the first selected column.
                    approx_row_max = approx_exact_logits_tensor[:, 0]
                else:
                    approx_row_max = approx_logits_all.max(dim=1).values
                row_max = approx_row_max if row_max is None else torch.maximum(row_max, approx_row_max)
            if exact_count > 0:
                exact_row_max = exact_retrieval_logits.max(dim=1).values
                row_max = exact_row_max if row_max is None else torch.maximum(row_max, exact_row_max)
            if row_max is None:
                raise RuntimeError("hybrid centroid path requires at least one denominator token")

            full_exp = torch.exp(full_logits - row_max[:, None])
            full_exp_sum = full_exp.sum(dim=1)
            use_list_exp_sums = (
                denominator_local_tensor is None
                and
                self.index_config.kernel_backend in {"auto", "triton", "h20"}
                and is_triton_available()
                and approx_logits_all.is_cuda
                and self.cache.index.inverted_list_offsets is not None
                and self.cache.index.inverted_list_indices is not None
            )
            denominator_list_ids = None
            if use_list_exp_sums:
                background_mass_exp = list_exp_sums_triton(
                    approx_logits_all,
                    row_max,
                    self.cache.index.inverted_list_offsets,
                    self.cache.index.inverted_list_indices,
                    num_lists=self.cache.retrieval_value_centroids.shape[0],
                )
                retrieval_exp_sum = background_mass_exp.sum(dim=1)
            else:
                retrieval_exp = torch.exp(approx_logits_all - row_max[:, None])
                retrieval_exp_sum = retrieval_exp.sum(dim=1)
            if exact_count > 0:
                if use_list_exp_sums:
                    if approx_exact_logits_tensor is None:
                        approx_exact_logits = approx_logits_all.gather(1, exact_local)
                    else:
                        approx_exact_logits = approx_exact_logits_tensor
                    old_exact_exp = torch.exp(approx_exact_logits - row_max[:, None])
                else:
                    old_exact_exp = retrieval_exp.gather(1, exact_local)
                exact_exp = torch.exp(exact_retrieval_logits - row_max[:, None])
                exact_exp_sum = exact_exp.sum(dim=1)
            else:
                old_exact_exp = torch.empty(
                    query_count,
                    0,
                    device=queries.device,
                    dtype=approx_score_dtype,
                )
                exact_exp = torch.empty(
                    query_count,
                    0,
                    device=queries.device,
                    dtype=approx_score_dtype,
                )
                exact_exp_sum = torch.zeros(
                    query_count,
                    device=queries.device,
                    dtype=approx_score_dtype,
                )
            denom = full_exp_sum + retrieval_exp_sum - old_exact_exp.sum(dim=1) + exact_exp_sum
            denom = denom.clamp_min(torch.finfo(denom.dtype).tiny)

            full_output = (full_exp / denom[:, None]).matmul(full_values)
            if exact_count > 0:
                exact_weights = exact_exp / denom[:, None]
                exact_output = torch.bmm(exact_weights.unsqueeze(1), exact_values).squeeze(1)
            else:
                exact_output = torch.zeros(
                    query_count,
                    fetched.values.shape[-1],
                    device=queries.device,
                    dtype=fetched.values.dtype,
                )

            list_ids = self.cache.index.list_ids.to(device=queries.device, dtype=torch.long)
            if denominator_local_tensor is not None:
                denominator_list_ids = list_ids[denominator_local_tensor]
            if not use_list_exp_sums:
                num_lists = self.cache.retrieval_value_centroids.shape[0]
                background_mass_exp = torch.zeros(
                    query_count,
                    num_lists,
                    device=queries.device,
                    dtype=approx_score_dtype,
                )
                background_mass_exp.scatter_add_(
                    1,
                    (
                        denominator_list_ids
                        if denominator_list_ids is not None
                        else list_ids.unsqueeze(0).expand(query_count, -1)
                    ),
                    retrieval_exp,
                )
            if exact_count > 0:
                exact_list_ids = (
                    denominator_list_ids.gather(1, exact_local)
                    if denominator_list_ids is not None
                    else list_ids[exact_local]
                )
                background_mass_exp.scatter_add_(1, exact_list_ids, -old_exact_exp)
            background_weight_mass = background_mass_exp / denom[:, None]
            centroids = self.cache.retrieval_value_centroids.to(device=queries.device)
            background_output = background_weight_mass.matmul(centroids)
            output = full_output + exact_output + background_output
            self._profile_stop("vectorized_no_details_centroid", profile_start, queries.device)

            profile_start = self._profile_start(queries.device)
            empty_long = torch.empty(0, device=queries.device, dtype=torch.long)
            empty_score = torch.empty(0, device=queries.device, dtype=approx_score_dtype)
            if denominator_count_override is None:
                if denominator_indices is None:
                    raise RuntimeError("denominator_indices or denominator_count_override is required")
                denominator_count = int(denominator_indices.numel())
            else:
                denominator_count = denominator_count_override
            approx_count = int(approx_score_shape[1])
            outputs = []
            for row in range(query_count):
                search_result = per_query_search[row]
                if candidate_count_override is None:
                    candidate_count = (
                        0 if search_result is None else int(search_result.candidate_indices.numel())
                    )
                else:
                    candidate_count = candidate_count_override
                if probed_list_count_override is None:
                    probed_list_count = (
                        0 if search_result is None else int(search_result.probed_lists.numel())
                    )
                else:
                    probed_list_count = probed_list_count_override
                outputs.append(
                    AttentionOutput(
                        output=output[row],
                        indices=empty_long,
                        logits=empty_score,
                        weights=empty_score,
                        approx_retrieval_logits=empty_score,
                        exact_retrieval_logits=None,
                        search_result=None,
                        denominator_indices=None,
                        denominator_logits=None,
                        denominator_weights=None,
                        selected_count=int(full_count + exact_count),
                        denominator_count=denominator_count,
                        approx_retrieval_count=approx_count,
                        exact_retrieval_count=int(exact_count),
                        candidate_count=candidate_count,
                        probed_list_count=probed_list_count,
                    )
                )
            self._profile_stop("vectorized_output_pack", profile_start, queries.device)
            return tuple(outputs)

        profile_start = self._profile_start(queries.device)
        hybrid_retrieval_logits = approx_logits_all.clone()
        if exact_count > 0:
            hybrid_retrieval_logits.scatter_(1, exact_local, exact_retrieval_logits)

        denominator_logits = torch.cat([full_logits, hybrid_retrieval_logits], dim=1)
        denominator_weights = torch.softmax(denominator_logits, dim=-1)
        full_weights = denominator_weights[:, :full_count]
        retrieval_weights = denominator_weights[:, full_count:]
        self._profile_stop("vectorized_softmax", profile_start, queries.device)

        profile_start = self._profile_start(queries.device)
        full_output = full_weights.matmul(full_values)
        if exact_count > 0:
            exact_weights = retrieval_weights.gather(1, exact_local)
            exact_output = torch.bmm(exact_weights.unsqueeze(1), exact_values).squeeze(1)
        else:
            exact_weights = torch.empty(
                query_count,
                0,
                device=queries.device,
                dtype=denominator_weights.dtype,
            )
            exact_output = torch.zeros(
                query_count,
                fetched.values.shape[-1],
                device=queries.device,
                dtype=fetched.values.dtype,
            )
        self._profile_stop("vectorized_exact_output", profile_start, queries.device)

        background_weight_mass = None
        background_output = None
        if self.attention_config.hybrid_value_mode == "centroid":
            profile_start = self._profile_start(queries.device)
            if self.cache.retrieval_value_centroids is None:
                raise RuntimeError("hybrid_value_mode='centroid' requires retrieval value centroids")
            retrieval_background_weights = retrieval_weights.clone()
            if exact_count > 0:
                retrieval_background_weights.scatter_(1, exact_local, 0)
            list_ids = self.cache.index.list_ids.to(device=queries.device, dtype=torch.long)
            num_lists = self.cache.retrieval_value_centroids.shape[0]
            background_weight_mass = torch.zeros(
                query_count,
                num_lists,
                device=queries.device,
                dtype=denominator_weights.dtype,
            )
            background_weight_mass.scatter_add_(
                1,
                list_ids.unsqueeze(0).expand(query_count, -1),
                retrieval_background_weights,
            )
            centroids = self.cache.retrieval_value_centroids.to(device=queries.device)
            background_output = background_weight_mass.matmul(centroids)
            output = full_output + exact_output + background_output
            self._profile_stop("vectorized_centroid", profile_start, queries.device)
        else:
            output = full_output + exact_output

        if not self.attention_config.collect_attention_details:
            profile_start = self._profile_start(queries.device)
            empty_long = torch.empty(0, device=queries.device, dtype=torch.long)
            empty_score = torch.empty(0, device=queries.device, dtype=approx_logits_all.dtype)
            if denominator_count_override is None:
                if denominator_indices is None:
                    raise RuntimeError("denominator_indices or denominator_count_override is required")
                denominator_count = int(denominator_indices.numel())
            else:
                denominator_count = denominator_count_override
            approx_count = int(approx_logits_all.shape[1])
            outputs = []
            for row in range(query_count):
                search_result = per_query_search[row]
                if candidate_count_override is None:
                    candidate_count = (
                        0 if search_result is None else int(search_result.candidate_indices.numel())
                    )
                else:
                    candidate_count = candidate_count_override
                if probed_list_count_override is None:
                    probed_list_count = (
                        0 if search_result is None else int(search_result.probed_lists.numel())
                    )
                else:
                    probed_list_count = probed_list_count_override
                outputs.append(
                    AttentionOutput(
                        output=output[row],
                        indices=empty_long,
                        logits=empty_score,
                        weights=empty_score,
                        approx_retrieval_logits=empty_score,
                        exact_retrieval_logits=None,
                        search_result=None,
                        denominator_indices=None,
                        denominator_logits=None,
                        denominator_weights=None,
                        selected_count=int(full_count + exact_count),
                        denominator_count=denominator_count,
                        approx_retrieval_count=approx_count,
                        exact_retrieval_count=int(exact_count),
                        candidate_count=candidate_count,
                        probed_list_count=probed_list_count,
                    )
                )
            self._profile_stop("vectorized_output_pack", profile_start, queries.device)
            return tuple(outputs)

        profile_start = self._profile_start(queries.device)
        if denominator_indices is None:
            raise RuntimeError("collect_attention_details requires denominator_indices")
        outputs = []
        for row in range(query_count):
            current_exact_global = exact_global[row]
            current_indices = torch.cat([full_indices, current_exact_global], dim=0)
            current_logits = torch.cat([full_logits[row], exact_retrieval_logits[row]], dim=0)
            current_weights = torch.cat([full_weights[row], exact_weights[row]], dim=0)
            current_exact_logits = (
                None
                if exact_count == 0 or not self.attention_config.exact_rerank
                else exact_retrieval_logits[row]
            )
            outputs.append(
                AttentionOutput(
                    output=output[row],
                    indices=current_indices,
                    logits=current_logits,
                    weights=current_weights,
                    approx_retrieval_logits=approx_logits_all[row],
                    exact_retrieval_logits=current_exact_logits,
                    search_result=per_query_search[row],
                    denominator_indices=denominator_indices,
                    denominator_logits=denominator_logits[row],
                    denominator_weights=denominator_weights[row],
                    background_weight_mass=(
                        None if background_weight_mass is None else background_weight_mass[row]
                    ),
                    background_output=(
                        None if background_output is None else background_output[row]
                    ),
                )
            )
        self._profile_stop("vectorized_output_pack", profile_start, queries.device)
        return tuple(outputs)

    def _forward_many_sparse_shared_kv(
        self,
        queries: torch.Tensor,
        scale_value: float,
        *,
        precomputed_search_results: tuple[SearchResult, ...] | None,
    ) -> tuple[AttentionOutput, ...]:
        assert self.cache is not None

        if queries.shape[0] == 0:
            return ()
        search_results: tuple[SearchResult | None, ...]
        if precomputed_search_results is None:
            search_results = tuple(None for _ in range(queries.shape[0]))
        else:
            if len(precomputed_search_results) != queries.shape[0]:
                raise ValueError("precomputed_search_results must match query count")
            search_results = precomputed_search_results

        per_query_indices = []
        per_query_retrieval_global = []
        per_query_approx_logits = []
        for query, search_result in zip(queries, search_results, strict=True):
            selected_parts = [self.cache.regions.sink, self.cache.regions.local]
            approx_retrieval_logits = torch.empty(0, device=query.device, dtype=query.dtype)
            retrieval_global = torch.empty(0, device=query.device, dtype=torch.long)
            if search_result is not None:
                retrieval_global = self.cache.retrieval_global_indices(search_result.indices)
                selected_parts.append(retrieval_global)
                approx_retrieval_logits = search_result.approx_scores * scale_value
            per_query_indices.append(_unique_preserve_order(torch.cat(selected_parts, dim=0)))
            per_query_retrieval_global.append(retrieval_global)
            per_query_approx_logits.append(approx_retrieval_logits)

        union_indices = _unique_preserve_order(torch.cat(per_query_indices, dim=0))
        fetched = self.cache.fetch_kv(union_indices, device=queries.device)
        assert fetched.keys is not None
        assert fetched.values is not None

        outputs = []
        for query, indices, retrieval_global, approx_retrieval_logits, search_result in zip(
            queries,
            per_query_indices,
            per_query_retrieval_global,
            per_query_approx_logits,
            search_results,
            strict=True,
        ):
            positions = _positions(union_indices, indices)
            keys = fetched.keys[positions]
            values = fetched.values[positions]
            logits = keys.matmul(query) * scale_value
            exact_retrieval_logits = None

            if search_result is not None and approx_retrieval_logits.numel() > 0:
                retrieval_pos = _positions(indices, retrieval_global)
                if self.attention_config.exact_rerank:
                    exact_retrieval_logits = logits[retrieval_pos]
                else:
                    logits[retrieval_pos] = approx_retrieval_logits

            weights = torch.softmax(logits, dim=-1)
            outputs.append(
                AttentionOutput(
                    output=weights.matmul(values),
                    indices=indices,
                    logits=logits,
                    weights=weights,
                    approx_retrieval_logits=approx_retrieval_logits,
                    exact_retrieval_logits=exact_retrieval_logits,
                    search_result=search_result,
                    denominator_indices=indices,
                    denominator_logits=logits,
                    denominator_weights=weights,
                )
            )
        return tuple(outputs)

    def _forward_sparse(
        self,
        query: torch.Tensor,
        scale_value: float,
        *,
        precomputed_approx_raw_all: torch.Tensor | None = None,
        precomputed_search_result: SearchResult | None = None,
    ) -> AttentionOutput:
        assert self.cache is not None
        selected_parts = [self.cache.regions.sink, self.cache.regions.local]
        search_result = None
        approx_retrieval_logits = torch.empty(0, device=query.device, dtype=query.dtype)
        exact_retrieval_logits = None

        if self.cache.index is not None and self._should_retrieve():
            retrieval_len = self.cache.regions.retrieval.numel()
            if self.attention_config.retrieval_top_p is None:
                topk = self._resolve_retrieval_topk(retrieval_len)
                if topk > 0:
                    if precomputed_search_result is not None:
                        search_result = precomputed_search_result
                    else:
                        search_result = self.cache.index.search(
                            query,
                            topk=topk,
                            nprobe=self.attention_config.nprobe,
                            candidate_budget=self.attention_config.candidate_budget,
                        )
            else:
                approx_raw_all = self._retrieval_scores(
                    query,
                    precomputed_approx_raw_all=precomputed_approx_raw_all,
                )
                approx_logits_all = approx_raw_all * scale_value
                topk = self._resolve_retrieval_topk(
                    approx_raw_all.numel(),
                    approx_logits=approx_logits_all,
                )
                search_result = self._select_all_pq_tokens(
                    approx_raw_all=approx_raw_all,
                    approx_logits_all=approx_logits_all,
                    topk=topk,
                )
            if search_result is not None:
                retrieval_global = self.cache.retrieval_global_indices(search_result.indices)
                selected_parts.append(retrieval_global)
                approx_retrieval_logits = search_result.approx_scores * scale_value

        indices = _unique_preserve_order(torch.cat(selected_parts, dim=0))
        fetched = self.cache.fetch_kv(indices, device=query.device)
        assert fetched.keys is not None
        assert fetched.values is not None
        logits = fetched.keys.matmul(query) * scale_value

        if (
            search_result is not None
            and not self.attention_config.exact_rerank
            and approx_retrieval_logits.numel() > 0
        ):
            retrieval_global = self.cache.retrieval_global_indices(search_result.indices)
            retrieval_pos = _positions(indices, retrieval_global)
            logits[retrieval_pos] = approx_retrieval_logits
        elif search_result is not None and approx_retrieval_logits.numel() > 0:
            retrieval_global = self.cache.retrieval_global_indices(search_result.indices)
            retrieval_pos = _positions(indices, retrieval_global)
            exact_retrieval_logits = logits[retrieval_pos]

        weights = torch.softmax(logits, dim=-1)
        output = weights.matmul(fetched.values)
        return AttentionOutput(
            output=output,
            indices=indices,
            logits=logits,
            weights=weights,
            approx_retrieval_logits=approx_retrieval_logits,
            exact_retrieval_logits=exact_retrieval_logits,
            search_result=search_result,
            denominator_indices=indices,
            denominator_logits=logits,
            denominator_weights=weights,
        )

    def _forward_hybrid(
        self,
        query: torch.Tensor,
        scale_value: float,
        *,
        precomputed_approx_raw_all: torch.Tensor | None = None,
        precomputed_search_result: SearchResult | None = None,
    ) -> AttentionOutput:
        assert self.cache is not None

        full_regions = [self.cache.regions.sink, self.cache.regions.local]
        full_indices = _unique_preserve_order(torch.cat(full_regions, dim=0))

        search_result = None
        exact_retrieval_logits = None
        approx_retrieval_logits = torch.empty(0, device=query.device, dtype=query.dtype)
        full_prefetch = None
        full_kv = None

        if self.cache.index is None or self.cache.regions.retrieval.numel() == 0:
            full_kv = self.cache.fetch_kv(full_indices, device=query.device)
            assert full_kv.keys is not None
            assert full_kv.values is not None
            full_logits = full_kv.keys.matmul(query) * scale_value
            weights = torch.softmax(full_logits, dim=-1)
            output = weights.matmul(full_kv.values)
            return AttentionOutput(
                output=output,
                indices=full_indices,
                logits=full_logits,
                weights=weights,
                approx_retrieval_logits=approx_retrieval_logits,
                exact_retrieval_logits=exact_retrieval_logits,
                search_result=search_result,
                denominator_indices=full_indices,
                denominator_logits=full_logits,
                denominator_weights=weights,
            )

        if full_indices.numel() > 0 and getattr(self.cache, "_fast_offload", False):
            # (opt-in, PQ_HSA_OFFLOAD_FAST=1): sink+local come from the
            # cache's incrementally-maintained GPU-resident mirror instead of
            # a CPU round trip -- this forces _fetch_materialized_kv's
            # already-existing split path (full_kv is not None) below, so
            # only the genuinely new PQ-retrieval indices pay the CPU gather.
            full_kv = self.cache.fetch_resident_kv(
                device=query.device,
                include_values=self.attention_config.hybrid_value_mode != "full",
            )
        elif self.attention_config.prefetch_full_kv and full_indices.numel() > 0:
            full_prefetch = self.cache.prefetch_kv(
                full_indices,
                device=query.device,
                include_keys=True,
                include_values=self.attention_config.hybrid_value_mode != "full",
            )

        retrieval_len = self.cache.regions.retrieval.numel()
        uses_candidate_denominator = self._uses_ivf_candidate_denominator()
        approx_raw_all = None
        if not uses_candidate_denominator or precomputed_approx_raw_all is not None:
            approx_raw_all = self._retrieval_scores(
                query,
                precomputed_approx_raw_all=precomputed_approx_raw_all,
            )
            approx_retrieval_logits = approx_raw_all * scale_value
        prefix_logits = None
        if (
            full_kv is None
            and self.attention_config.retrieval_top_p is not None
            and self.attention_config.retrieval_top_p_scope == "denominator"
        ):
            full_kv = self._fetch_full_kv(
                full_indices=full_indices,
                query_device=query.device,
                full_prefetch=full_prefetch,
            )
            full_prefetch = None
            assert full_kv.keys is not None
            prefix_logits = full_kv.keys.matmul(query) * scale_value
        topk = self._resolve_retrieval_topk(
            retrieval_len,
            approx_logits=approx_retrieval_logits,
            prefix_logits=prefix_logits,
        )

        if topk > 0:
            if (
                uses_candidate_denominator
                and approx_raw_all is None
            ):
                if precomputed_search_result is not None:
                    search_result = precomputed_search_result
                else:
                    assert self.cache.index is not None
                    search_result = self.cache.index.search(
                        query,
                        topk=topk,
                        nprobe=self.attention_config.nprobe,
                        candidate_budget=self.attention_config.candidate_budget,
                    )
            else:
                assert approx_raw_all is not None
                search_result = self._select_hybrid_exact_tokens(
                    query=query,
                    topk=topk,
                    approx_raw_all=approx_raw_all,
                    approx_logits_all=approx_retrieval_logits,
                    precomputed_search_result=precomputed_search_result,
                )
            exact_local_indices = search_result.indices
            exact_global_indices = self.cache.retrieval_global_indices(exact_local_indices)
        else:
            exact_local_indices = torch.empty(0, device=query.device, dtype=torch.long)
            exact_global_indices = torch.empty(0, device=query.device, dtype=torch.long)

        exact_materialized = _unique_preserve_order(torch.cat([full_indices, exact_global_indices], dim=0))
        materialized_kv = self._fetch_materialized_kv(
            full_indices=full_indices,
            exact_global_indices=exact_global_indices,
            exact_materialized=exact_materialized,
            query_device=query.device,
            full_kv=full_kv,
            full_prefetch=full_prefetch,
        )
        assert materialized_kv.keys is not None
        materialized_key_logits = materialized_kv.keys.matmul(query) * scale_value
        full_pos_in_materialized = _positions(exact_materialized, full_indices)
        full_logits = materialized_key_logits[full_pos_in_materialized]
        exact_pos_in_materialized = _positions(exact_materialized, exact_global_indices)
        if self.attention_config.exact_rerank and exact_pos_in_materialized.numel() > 0:
            exact_retrieval_logits = materialized_key_logits[exact_pos_in_materialized]

        if uses_candidate_denominator and search_result is not None:
            denominator_local_indices = search_result.candidate_indices.to(
                device=query.device,
                dtype=torch.long,
            )
            hybrid_retrieval_logits = search_result.candidate_scores.to(query.device) * scale_value
            approx_retrieval_logits = hybrid_retrieval_logits
        else:
            denominator_local_indices = torch.arange(
                retrieval_len,
                device=query.device,
                dtype=torch.long,
            )
            hybrid_retrieval_logits = approx_retrieval_logits.clone()
        if exact_retrieval_logits is not None and exact_retrieval_logits.numel() > 0:
            exact_positions_in_denominator = _positions(
                denominator_local_indices,
                exact_local_indices,
            )
            hybrid_retrieval_logits[exact_positions_in_denominator] = exact_retrieval_logits

        denominator_retrieval_indices = self.cache.retrieval_global_indices(
            denominator_local_indices
        )
        denominator_indices = torch.cat([full_indices, denominator_retrieval_indices], dim=0)
        denominator_logits = torch.cat([full_logits, hybrid_retrieval_logits], dim=0)
        denominator_weights = torch.softmax(denominator_logits, dim=-1)

        materialized_positions = _positions(denominator_indices, exact_materialized)
        materialized_weights = denominator_weights[materialized_positions]
        materialized_logits = denominator_logits[materialized_positions]

        if self.attention_config.hybrid_value_mode == "full":
            denominator_values = self.cache.fetch_values(denominator_indices, device=query.device)
            output = denominator_weights.matmul(denominator_values)
            output_indices = denominator_indices
            output_logits = denominator_logits
            output_weights = denominator_weights
            background_weight_mass = None
            background_output = None
        elif self.attention_config.hybrid_value_mode == "centroid":
            output, background_weight_mass, background_output = self._hybrid_centroid_output(
                full_indices=full_indices,
                exact_local_indices=exact_local_indices,
                exact_global_indices=exact_global_indices,
                exact_materialized=exact_materialized,
                materialized_values=materialized_kv.values,
                denominator_weights=denominator_weights,
                retrieval_denominator_local_indices=denominator_local_indices,
            )
            output_indices = exact_materialized
            output_logits = materialized_logits
            output_weights = materialized_weights
        else:
            assert materialized_kv.values is not None
            output = materialized_weights.matmul(
                materialized_kv.values
            )
            output_indices = exact_materialized
            output_logits = materialized_logits
            output_weights = materialized_weights
            background_weight_mass = None
            background_output = None

        return AttentionOutput(
            output=output,
            indices=output_indices,
            logits=output_logits,
            weights=output_weights,
            approx_retrieval_logits=approx_retrieval_logits,
            exact_retrieval_logits=exact_retrieval_logits,
            search_result=search_result,
            denominator_indices=denominator_indices,
            denominator_logits=denominator_logits,
            denominator_weights=denominator_weights,
            background_weight_mass=background_weight_mass,
            background_output=background_output,
        )

    def _select_hybrid_exact_tokens(
        self,
        *,
        query: torch.Tensor,
        topk: int,
        approx_raw_all: torch.Tensor,
        approx_logits_all: torch.Tensor,
        precomputed_search_result: SearchResult | None = None,
    ) -> SearchResult:
        assert self.cache is not None
        assert self.cache.index is not None

        if self.attention_config.hybrid_topk_source == "ivf_candidates":
            if precomputed_search_result is not None:
                return precomputed_search_result
            return self._select_ivf_candidate_tokens_from_scores(
                query=query,
                approx_raw_all=approx_raw_all,
                topk=topk,
            )

        return self._select_all_pq_tokens(
            approx_raw_all=approx_raw_all,
            approx_logits_all=approx_logits_all,
            topk=topk,
        )

    def _fetch_materialized_kv(
        self,
        *,
        full_indices: torch.Tensor,
        exact_global_indices: torch.Tensor,
        exact_materialized: torch.Tensor,
        query_device: torch.device,
        full_kv: FetchedKV | None,
        full_prefetch: PrefetchedKV | None,
    ) -> FetchedKV:
        assert self.cache is not None
        include_values = self.attention_config.hybrid_value_mode != "full"
        if full_kv is None and full_prefetch is None:
            return self.cache.fetch_kv(
                exact_materialized,
                device=query_device,
                include_keys=True,
                include_values=include_values,
            )

        if full_kv is None:
            full_kv = full_prefetch.wait()
        exact_only = _ordered_difference(exact_global_indices, full_indices)
        if exact_only.numel() == 0:
            return full_kv

        exact_kv = self.cache.fetch_kv(
            exact_only,
            device=query_device,
            include_keys=True,
            include_values=include_values,
        )
        keys = None
        values = None
        if full_kv.keys is not None and exact_kv.keys is not None:
            keys = torch.cat([full_kv.keys, exact_kv.keys], dim=0)
        if full_kv.values is not None and exact_kv.values is not None:
            values = torch.cat([full_kv.values, exact_kv.values], dim=0)
        return FetchedKV(indices=exact_materialized, keys=keys, values=values)

    def _fetch_full_kv(
        self,
        *,
        full_indices: torch.Tensor,
        query_device: torch.device,
        full_prefetch: PrefetchedKV | None,
    ) -> FetchedKV:
        assert self.cache is not None
        if full_prefetch is not None:
            return full_prefetch.wait()
        return self.cache.fetch_kv(
            full_indices,
            device=query_device,
            include_keys=True,
            include_values=self.attention_config.hybrid_value_mode != "full",
        )

    def _hybrid_centroid_output(
        self,
        *,
        full_indices: torch.Tensor,
        exact_local_indices: torch.Tensor,
        exact_global_indices: torch.Tensor,
        exact_materialized: torch.Tensor,
        materialized_values: torch.Tensor | None,
        denominator_weights: torch.Tensor,
        retrieval_denominator_local_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.cache is not None
        assert self.cache.index is not None
        assert self.cache.index.list_ids is not None
        if self.cache.retrieval_value_centroids is None:
            raise RuntimeError("hybrid_value_mode='centroid' requires retrieval value centroids")

        full_count = full_indices.numel()
        full_weights = denominator_weights[:full_count]
        retrieval_weights = denominator_weights[full_count:].clone()
        if retrieval_denominator_local_indices is None:
            retrieval_denominator_local_indices = torch.arange(
                retrieval_weights.numel(),
                device=denominator_weights.device,
                dtype=torch.long,
            )

        if materialized_values is None:
            raise RuntimeError("hybrid centroid output requires materialized values")
        full_pos = _positions(exact_materialized, full_indices)
        exact_pos = _positions(exact_materialized, exact_global_indices)
        full_values = materialized_values[full_pos]
        full_output = full_weights.matmul(full_values)
        exact_output = torch.zeros_like(full_output)
        if exact_local_indices.numel() > 0:
            exact_values = materialized_values[exact_pos]
            exact_denominator_pos = _positions(
                retrieval_denominator_local_indices,
                exact_local_indices,
            )
            exact_output = retrieval_weights[exact_denominator_pos].matmul(
                exact_values
            )
            retrieval_weights[exact_denominator_pos] = 0

        num_lists = self.cache.retrieval_value_centroids.shape[0]
        background_weight_mass = torch.zeros(
            num_lists,
            device=denominator_weights.device,
            dtype=denominator_weights.dtype,
        )
        retrieval_list_ids = as_index(
            gather_ids(
                self.cache.index.list_ids,
                retrieval_denominator_local_indices.to(device=self.cache.index.list_ids.device),
            )
        ).to(device=denominator_weights.device)
        background_weight_mass.index_add_(0, retrieval_list_ids, retrieval_weights)
        background_output = background_weight_mass.matmul(self.cache.retrieval_value_centroids)
        return full_output + exact_output + background_output, background_weight_mass, background_output

    def _select_all_pq_tokens(
        self,
        *,
        approx_raw_all: torch.Tensor,
        approx_logits_all: torch.Tensor,
        topk: int,
    ) -> SearchResult:
        k = min(max(topk, 0), approx_raw_all.numel())
        if k == 0:
            exact_local_indices = torch.empty(0, device=approx_raw_all.device, dtype=torch.long)
            approx_top_scores = torch.empty(
                0,
                device=approx_raw_all.device,
                dtype=approx_raw_all.dtype,
            )
        else:
            exact_local_indices = torch.topk(approx_logits_all, k=k).indices
            approx_top_scores = approx_raw_all[exact_local_indices]
        return SearchResult(
            indices=exact_local_indices,
            approx_scores=approx_top_scores,
            candidate_indices=torch.arange(
                approx_raw_all.numel(),
                device=approx_raw_all.device,
                dtype=torch.long,
            ),
            candidate_scores=approx_raw_all,
            probed_lists=torch.empty(0, device=approx_raw_all.device, dtype=torch.long),
        )

    def _resolve_retrieval_topk(
        self,
        retrieval_len: int,
        *,
        approx_logits: torch.Tensor | None = None,
        prefix_logits: torch.Tensor | None = None,
    ) -> int:
        if retrieval_len == 0:
            return 0
        if self.attention_config.retrieval_top_p is not None:
            if approx_logits is None:
                raise RuntimeError("retrieval_top_p requires approximate logits")
            if self.attention_config.retrieval_top_p_scope == "denominator":
                if prefix_logits is None:
                    raise RuntimeError(
                        "retrieval_top_p_scope='denominator' requires full-region logits"
                    )
                return _top_p_count_with_prefix(
                    prefix_logits,
                    approx_logits,
                    self.attention_config.retrieval_top_p,
                )
            return _top_p_count(approx_logits, self.attention_config.retrieval_top_p)
        if self.attention_config.retrieval_top_fraction is not None:
            topk = ceil(retrieval_len * self.attention_config.retrieval_top_fraction)
        else:
            topk = self.attention_config.retrieval_topk
        return min(max(topk, 0), retrieval_len)

    def _should_retrieve(self) -> bool:
        return (
            self.attention_config.retrieval_top_p is not None
            or self.attention_config.retrieval_topk > 0
            or self.attention_config.retrieval_top_fraction is not None
        )

    def _can_share_sparse_forward_many_fetch(self) -> bool:
        return (
            self.attention_config.mode == "sparse"
            and self.attention_config.retrieval_top_p is None
        )

    def _can_share_hybrid_forward_many_fetch(self) -> bool:
        assert self.cache is not None
        return (
            self.attention_config.mode == "hybrid"
            and self.attention_config.retrieval_top_p is None
            and self.attention_config.hybrid_value_mode in {"denominator", "centroid"}
            and not self.attention_config.prefetch_full_kv
            and self.cache.index is not None
            and self.cache.regions.retrieval.numel() > 0
        )

    def _check_cache(self) -> None:
        if self.cache is None:
            raise RuntimeError("build_cache must be called before forward")

    def _batched_retrieval_scores_for_queries(self, queries: torch.Tensor) -> torch.Tensor | None:
        assert self.cache is not None
        if self.cache.index is None or self.cache.regions.retrieval.numel() == 0:
            return None
        if self._uses_ivf_candidate_denominator() and self.attention_config.retrieval_top_p is None:
            return None
        if self.attention_config.mode == "hybrid" or self.attention_config.retrieval_top_p is not None:
            return self.cache.index.approximate_scores(queries)
        return None

    def _uses_ivf_candidate_denominator(self) -> bool:
        return (
            self.attention_config.mode == "hybrid"
            and self.attention_config.hybrid_denominator_source == "ivf_candidates"
        )

    def _batched_search_results_for_queries(
        self,
        queries: torch.Tensor,
        *,
        precomputed_approx_raw_all: torch.Tensor | None = None,
        scale_value: float | None = None,
    ) -> tuple[SearchResult, ...] | None:
        assert self.cache is not None
        if self.cache.index is None or self.cache.regions.retrieval.numel() == 0:
            return None
        if self.attention_config.retrieval_top_p is not None:
            return None
        if (
            self.attention_config.mode == "hybrid"
            and self.attention_config.hybrid_topk_source not in {"ivf_candidates", "all_pq"}
        ):
            return None
        if not self._should_retrieve():
            return None

        topk = self._resolve_retrieval_topk(self.cache.regions.retrieval.numel())
        if topk <= 0:
            return None
        if precomputed_approx_raw_all is not None:
            if precomputed_approx_raw_all.shape != (
                queries.shape[0],
                self.cache.regions.retrieval.numel(),
            ):
                raise ValueError(
                    "precomputed_approx_raw_all must be "
                    f"[{queries.shape[0]}, {self.cache.regions.retrieval.numel()}], "
                    f"got {tuple(precomputed_approx_raw_all.shape)}"
                )
            if (
                self.attention_config.mode == "hybrid"
                and self.attention_config.hybrid_topk_source == "all_pq"
            ):
                if scale_value is None:
                    raise RuntimeError("scale_value is required for batched all_pq top-k")
                return self._select_all_pq_tokens_many_from_scores(
                    approx_raw_all=precomputed_approx_raw_all,
                    scale_value=scale_value,
                    topk=topk,
                )
            return tuple(
                self._select_ivf_candidate_tokens_from_scores(
                    query=query,
                    approx_raw_all=approx_raw_all,
                    topk=topk,
                )
                for query, approx_raw_all in zip(
                    queries,
                    precomputed_approx_raw_all,
                    strict=True,
                )
            )
        return self.cache.index.search_many(
            queries,
            topk=topk,
            nprobe=self.attention_config.nprobe,
            candidate_budget=self.attention_config.candidate_budget,
        )

    def _select_ivf_candidate_tokens_from_scores(
        self,
        *,
        query: torch.Tensor,
        approx_raw_all: torch.Tensor,
        topk: int,
    ) -> SearchResult:
        assert self.cache is not None
        assert self.cache.index is not None

        probed_lists = self.cache.index.select_lists(
            query,
            nprobe=self.attention_config.nprobe,
        )
        candidate_indices = self.cache.index._candidate_indices_for_lists(probed_lists)
        if candidate_indices.numel() == 0:
            candidate_indices = torch.arange(
                approx_raw_all.numel(),
                device=approx_raw_all.device,
                dtype=torch.long,
            )
        candidate_indices = candidate_indices.to(device=approx_raw_all.device, dtype=torch.long)
        candidate_scores = approx_raw_all[candidate_indices]

        budget = candidate_scores.numel()
        if self.attention_config.candidate_budget is not None:
            budget = min(
                max(self.attention_config.candidate_budget, topk),
                candidate_scores.numel(),
            )
        if budget < candidate_scores.numel():
            budget_scores, budget_order = torch.topk(candidate_scores, k=budget)
            candidate_indices = candidate_indices[budget_order]
            candidate_scores = budget_scores

        k = min(max(topk, 0), candidate_scores.numel())
        if k == 0:
            exact_local_indices = torch.empty(0, device=approx_raw_all.device, dtype=torch.long)
            approx_top_scores = torch.empty(0, device=approx_raw_all.device, dtype=approx_raw_all.dtype)
        else:
            approx_top_scores, top_order = torch.topk(candidate_scores, k=k)
            exact_local_indices = candidate_indices[top_order]
        return SearchResult(
            indices=exact_local_indices,
            approx_scores=approx_top_scores,
            candidate_indices=candidate_indices,
            candidate_scores=candidate_scores,
            probed_lists=probed_lists,
        )

    def _select_all_pq_tokens_many_from_scores(
        self,
        *,
        approx_raw_all: torch.Tensor,
        scale_value: float,
        topk: int,
    ) -> tuple[SearchResult, ...]:
        if approx_raw_all.ndim != 2:
            raise ValueError(f"approx_raw_all must be [Q, N], got {tuple(approx_raw_all.shape)}")
        query_count, retrieval_len = approx_raw_all.shape
        k = min(max(topk, 0), retrieval_len)
        candidate_indices = torch.arange(
            retrieval_len,
            device=approx_raw_all.device,
            dtype=torch.long,
        )
        probed_lists = torch.empty(0, device=approx_raw_all.device, dtype=torch.long)
        if k == 0:
            empty_indices = torch.empty(0, device=approx_raw_all.device, dtype=torch.long)
            empty_scores = torch.empty(0, device=approx_raw_all.device, dtype=approx_raw_all.dtype)
            return tuple(
                SearchResult(
                    indices=empty_indices,
                    approx_scores=empty_scores,
                    candidate_indices=candidate_indices,
                    candidate_scores=approx_raw_all[row],
                    probed_lists=probed_lists,
                )
                for row in range(query_count)
            )

        approx_logits_all = approx_raw_all * scale_value
        _, exact_local_indices = torch.topk(approx_logits_all, k=k, dim=1)
        approx_top_scores = approx_raw_all.gather(1, exact_local_indices)
        return tuple(
            SearchResult(
                indices=exact_local_indices[row],
                approx_scores=approx_top_scores[row],
                candidate_indices=candidate_indices,
                candidate_scores=approx_raw_all[row],
                probed_lists=probed_lists,
            )
            for row in range(query_count)
        )

    def _retrieval_scores(
        self,
        query: torch.Tensor,
        *,
        precomputed_approx_raw_all: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.cache is not None
        assert self.cache.index is not None
        if precomputed_approx_raw_all is None:
            return self.cache.index.approximate_scores(query)
        retrieval_len = self.cache.regions.retrieval.numel()
        if precomputed_approx_raw_all.ndim != 1 or precomputed_approx_raw_all.numel() != retrieval_len:
            raise ValueError(
                "precomputed_approx_raw_all must be "
                f"[{retrieval_len}], got {tuple(precomputed_approx_raw_all.shape)}"
            )
        return precomputed_approx_raw_all


def _unique_preserve_order(indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim != 1:
        raise ValueError(f"indices must be rank-1, got {tuple(indices.shape)}")
    if indices.numel() <= 1:
        return indices.clone()

    sorted_order = torch.argsort(indices, stable=True)
    sorted_values = indices[sorted_order]
    is_first = torch.ones(indices.numel(), device=indices.device, dtype=torch.bool)
    is_first[1:] = sorted_values[1:] != sorted_values[:-1]
    unique_values = sorted_values[is_first]
    first_positions = sorted_order[is_first]
    original_order = torch.argsort(first_positions, stable=True)
    return unique_values[original_order]


def _positions(haystack: torch.Tensor, needles: torch.Tensor) -> torch.Tensor:
    """Return positions of ``needles`` in a unique ``haystack``.

    Callers build both tensors from the same token-id sets, so every needle is
    expected to be present.
    """

    if haystack.ndim != 1 or needles.ndim != 1:
        raise ValueError("haystack and needles must be rank-1")
    if needles.numel() == 0:
        return torch.empty(0, device=haystack.device, dtype=torch.long)
    if haystack.numel() == 0:
        raise ValueError("cannot locate non-empty needles in an empty haystack")

    needles = needles.to(device=haystack.device, dtype=haystack.dtype)
    sorted_order = torch.argsort(haystack)
    sorted_values = haystack[sorted_order]
    locations = torch.searchsorted(sorted_values, needles)
    return sorted_order[locations].to(torch.long)


def _ordered_difference(indices: torch.Tensor, excluded: torch.Tensor) -> torch.Tensor:
    if indices.ndim != 1 or excluded.ndim != 1:
        raise ValueError("indices and excluded must be rank-1")
    if indices.numel() == 0 or excluded.numel() == 0:
        return indices.clone()

    excluded = excluded.to(device=indices.device, dtype=indices.dtype)
    sorted_excluded = torch.sort(excluded).values
    positions = torch.searchsorted(sorted_excluded, indices)
    in_range = positions < sorted_excluded.numel()
    clamped_positions = positions.clamp_max(sorted_excluded.numel() - 1)
    matched = in_range & (sorted_excluded[clamped_positions] == indices)
    return indices[~matched]


def _top_p_count(logits: torch.Tensor, top_p: float) -> int:
    if logits.ndim != 1:
        raise ValueError(f"logits must be [N], got {tuple(logits.shape)}")
    if logits.numel() == 0:
        return 0
    sorted_logits = torch.sort(logits, descending=True).values
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    target = torch.tensor(top_p, device=logits.device, dtype=cumulative.dtype)
    count = int(torch.searchsorted(cumulative, target, right=False).item()) + 1
    return min(max(count, 1), logits.numel())


def _top_p_count_with_prefix(
    prefix_logits: torch.Tensor,
    retrieval_logits: torch.Tensor,
    top_p: float,
) -> int:
    if prefix_logits.ndim != 1 or retrieval_logits.ndim != 1:
        raise ValueError("prefix_logits and retrieval_logits must be rank-1")
    if retrieval_logits.numel() == 0:
        return 0

    all_logits = torch.cat([prefix_logits, retrieval_logits], dim=0)
    weights = torch.softmax(all_logits, dim=-1)
    prefix_mass = weights[: prefix_logits.numel()].sum()
    target = torch.tensor(top_p, device=weights.device, dtype=weights.dtype)
    if prefix_mass >= target:
        return 0

    retrieval_weights = weights[prefix_logits.numel():]
    sorted_order = torch.sort(retrieval_logits, descending=True).indices
    cumulative = prefix_mass + retrieval_weights[sorted_order].cumsum(dim=-1)
    count = int(torch.searchsorted(cumulative, target, right=False).item()) + 1
    return min(max(count, 0), retrieval_logits.numel())
