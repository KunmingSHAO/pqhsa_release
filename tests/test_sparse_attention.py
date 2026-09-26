import pytest
import torch

from pq_hsa import (
    IVFPQConfig,
    IVFPQSparseAttention,
    SparseAttentionConfig,
    SparseKVCache,
    dense_attention,
)
from pq_hsa.attention.sparse_attention import (
    _ordered_difference,
    _positions,
    _unique_preserve_order,
)


def test_attention_index_set_helpers_preserve_order_and_device():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        indices = torch.tensor([5, 3, 5, 1, 3, 9], device=device)
        unique = _unique_preserve_order(indices)
        torch.testing.assert_close(unique, torch.tensor([5, 3, 1, 9], device=device))
        assert unique.device.type == torch.device(device).type

        haystack = torch.tensor([10, 4, 7, 2], device=device)
        needles = torch.tensor([7, 10, 2], device=device)
        positions = _positions(haystack, needles)
        torch.testing.assert_close(positions, torch.tensor([2, 0, 3], device=device))
        assert positions.device.type == torch.device(device).type

        difference = _ordered_difference(indices, torch.tensor([3, 8], device=device))
        torch.testing.assert_close(difference, torch.tensor([5, 5, 1, 9], device=device))
        assert difference.device.type == torch.device(device).type


def test_sparse_attention_exact_rerank_matches_dense_when_all_tokens_selected():
    torch.manual_seed(3)
    keys = torch.randn(96, 32)
    values = torch.randn(96, 24)
    query = torch.randn(32)

    dense = dense_attention(query, keys, values)
    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=8,
            nprobe=8,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=6,
            pq_max_iter=6,
            seed=4,
        ),
        SparseAttentionConfig(
            sink_tokens=0,
            local_window=0,
            retrieval_topk=keys.shape[0],
            nprobe=8,
            exact_rerank=True,
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)

    assert output.indices.numel() == keys.shape[0]
    torch.testing.assert_close(output.output, dense.output, atol=1e-5, rtol=1e-5)


def test_sparse_attention_with_random_rotation_matches_dense_when_all_tokens_selected():
    torch.manual_seed(56)
    keys = torch.randn(64, 32)
    values = torch.randn(64, 16)
    query = torch.randn(32)

    dense = dense_attention(query, keys, values)
    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=8,
            nprobe=8,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            rotation="random_orthogonal",
            direction_normalize=True,
            seed=57,
        ),
        SparseAttentionConfig(
            sink_tokens=2,
            local_window=4,
            retrieval_top_fraction=1.0,
            nprobe=8,
            mode="hybrid",
            hybrid_value_mode="full",
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)

    torch.testing.assert_close(output.output, dense.output, atol=1e-5, rtol=1e-5)


def test_sparse_attention_uses_pq_logits_when_exact_rerank_disabled():
    torch.manual_seed(4)
    keys = torch.randn(128, 32)
    values = torch.randn(128, 16)
    query = torch.randn(32)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=6,
            pq_max_iter=6,
            seed=5,
        ),
        SparseAttentionConfig(
            sink_tokens=4,
            local_window=8,
            retrieval_topk=10,
            nprobe=4,
            candidate_budget=32,
            exact_rerank=False,
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)

    assert output.output.shape == (values.shape[1],)
    assert output.approx_retrieval_logits.shape == (10,)
    assert output.exact_retrieval_logits is None

    retrieval_global = sparse.cache.retrieval_global_indices(output.search_result.indices)
    positions = []
    for idx in retrieval_global.tolist():
        positions.append((output.indices == idx).nonzero(as_tuple=False).item())
    torch.testing.assert_close(
        output.logits[torch.tensor(positions)],
        output.approx_retrieval_logits,
        atol=1e-5,
        rtol=1e-5,
    )


def test_sparse_attention_keeps_sink_local_and_retrieval_regions_unique():
    torch.manual_seed(5)
    keys = torch.randn(80, 32)
    values = torch.randn(80, 16)
    query = torch.randn(32)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=6,
            nprobe=3,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=5,
            pq_max_iter=5,
            seed=6,
        ),
        SparseAttentionConfig(
            sink_tokens=5,
            local_window=7,
            retrieval_topk=11,
            nprobe=3,
            candidate_budget=30,
            exact_rerank=True,
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)

    assert output.indices.unique().numel() == output.indices.numel()
    assert set(range(5)).issubset(set(output.indices.tolist()))
    assert set(range(73, 80)).issubset(set(output.indices.tolist()))
    assert output.exact_retrieval_logits is not None
    assert output.exact_retrieval_logits.shape == output.approx_retrieval_logits.shape


def test_sparse_attention_retrieval_top_fraction_controls_search_count():
    torch.manual_seed(18)
    keys = torch.randn(70, 24)
    values = torch.randn(70, 12)
    query = torch.randn(24)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=5,
            nprobe=3,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=19,
        ),
        SparseAttentionConfig(
            sink_tokens=3,
            local_window=7,
            retrieval_top_fraction=0.20,
            nprobe=3,
            exact_rerank=True,
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)
    retrieval_len = sparse.cache.regions.retrieval.numel()
    expected_topk = int((retrieval_len * 0.20) + 0.999999)

    assert output.search_result.indices.numel() == expected_topk
    assert output.approx_retrieval_logits.shape == (expected_topk,)
    assert output.exact_retrieval_logits.shape == (expected_topk,)


def test_hybrid_attention_uses_pq_background_and_exact_top_logits():
    torch.manual_seed(6)
    keys = torch.randn(72, 32)
    values = torch.randn(72, 16)
    query = torch.randn(32)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=6,
            nprobe=3,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=5,
            pq_max_iter=5,
            seed=7,
        ),
        SparseAttentionConfig(
            sink_tokens=4,
            local_window=8,
            retrieval_top_fraction=0.25,
            mode="hybrid",
            hybrid_value_mode="denominator",
            index_update_interval=4,
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)
    retrieval_len = sparse.cache.regions.retrieval.numel()
    expected_topk = int((retrieval_len * 0.25) + 0.999999)

    assert output.approx_retrieval_logits.shape == (retrieval_len,)
    assert output.exact_retrieval_logits.shape == (expected_topk,)
    assert output.search_result.probed_lists.numel() == 0
    assert output.search_result.candidate_indices.numel() == retrieval_len
    assert output.denominator_indices.shape[0] == (
        sparse.cache.regions.sink.numel()
        + sparse.cache.regions.local.numel()
        + retrieval_len
    )
    assert torch.isclose(output.denominator_weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert output.weights.sum() < 1.0
    torch.testing.assert_close(
        output.output,
        output.weights.matmul(values[output.indices]),
        atol=1e-6,
        rtol=1e-6,
    )

    selected_global = sparse.cache.retrieval_global_indices(output.search_result.indices)
    denom_positions = []
    for idx in selected_global.tolist():
        denom_positions.append((output.denominator_indices == idx).nonzero(as_tuple=False).item())
    torch.testing.assert_close(
        output.denominator_logits[torch.tensor(denom_positions)],
        output.exact_retrieval_logits,
        atol=1e-5,
        rtol=1e-5,
    )


def test_forward_many_matches_forward_and_batches_retrieval_scores(monkeypatch):
    torch.manual_seed(52)
    keys = torch.randn(80, 32)
    values = torch.randn(80, 16)
    queries = torch.randn(4, 32)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=53,
        ),
        SparseAttentionConfig(
            sink_tokens=4,
            local_window=8,
            retrieval_top_fraction=0.25,
            mode="hybrid",
            hybrid_value_mode="centroid",
        ),
    ).build_cache(keys, values)

    expected = tuple(sparse.forward(query) for query in queries)
    calls = []
    original_approximate_scores = sparse.cache.index.approximate_scores

    def counting_approximate_scores(query, indices=None):
        calls.append(tuple(query.shape))
        return original_approximate_scores(query, indices)

    monkeypatch.setattr(sparse.cache.index, "approximate_scores", counting_approximate_scores)

    actual = sparse.forward_many(queries)

    assert calls == [(queries.shape[0], queries.shape[1])]
    assert len(actual) == queries.shape[0]
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output.indices, expected_output.indices)
        torch.testing.assert_close(actual_output.logits, expected_output.logits, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_output.output, expected_output.output, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            actual_output.denominator_logits,
            expected_output.denominator_logits,
            atol=1e-6,
            rtol=1e-6,
        )


def test_forward_many_ivf_candidates_reuses_batched_scores(monkeypatch):
    torch.manual_seed(58)
    keys = torch.randn(80, 32)
    values = torch.randn(80, 16)
    queries = torch.randn(4, 32)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            topk_block_size=8,
            rotation="random_orthogonal",
            direction_normalize=True,
            seed=59,
        ),
        SparseAttentionConfig(
            sink_tokens=4,
            local_window=8,
            retrieval_top_fraction=0.25,
            mode="hybrid",
            hybrid_value_mode="centroid",
            hybrid_topk_source="ivf_candidates",
            candidate_budget=24,
        ),
    ).build_cache(keys, values)

    expected = tuple(sparse.forward(query) for query in queries)
    search_many_calls = []
    select_list_calls = []
    original_select_lists = sparse.cache.index.select_lists

    def forbidden_search_many(query_batch, **kwargs):
        search_many_calls.append((tuple(query_batch.shape), kwargs))
        raise AssertionError("forward_many should reuse precomputed PQ scores")

    def counting_select_lists(query, **kwargs):
        select_list_calls.append((tuple(query.shape), kwargs))
        return original_select_lists(query, **kwargs)

    monkeypatch.setattr(sparse.cache.index, "search_many", forbidden_search_many)
    monkeypatch.setattr(sparse.cache.index, "select_lists", counting_select_lists)

    actual = sparse.forward_many(queries)

    assert search_many_calls == []
    assert len(select_list_calls) == queries.shape[0]
    assert all(call[0] == (queries.shape[1],) for call in select_list_calls)
    assert all(call[1]["nprobe"] is None for call in select_list_calls)
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output.indices, expected_output.indices)
        torch.testing.assert_close(actual_output.logits, expected_output.logits, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_output.output, expected_output.output, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_output.search_result.indices, expected_output.search_result.indices)
        assert actual_output.search_result.candidate_indices.numel() <= 24


def test_forward_many_can_skip_attention_details_but_keep_counts():
    torch.manual_seed(66)
    keys = torch.randn(80, 32)
    values = torch.randn(80, 16)
    queries = torch.randn(4, 32)
    index_config = IVFPQConfig(
        num_lists=8,
        nprobe=4,
        num_subspaces=4,
        num_bits=4,
        coarse_max_iter=4,
        pq_max_iter=4,
        seed=67,
    )
    detail_config = SparseAttentionConfig(
        sink_tokens=4,
        local_window=8,
        retrieval_top_fraction=0.25,
        mode="hybrid",
        hybrid_value_mode="centroid",
        hybrid_topk_source="ivf_candidates",
        candidate_budget=24,
    )
    lean_config = SparseAttentionConfig(
        sink_tokens=4,
        local_window=8,
        retrieval_top_fraction=0.25,
        mode="hybrid",
        hybrid_value_mode="centroid",
        hybrid_topk_source="ivf_candidates",
        candidate_budget=24,
        collect_attention_details=False,
    )
    detailed = IVFPQSparseAttention(index_config, detail_config).build_cache(keys, values)
    lean = IVFPQSparseAttention(index_config, lean_config).build_cache(keys, values)

    expected = detailed.forward_many(queries)
    actual = lean.forward_many(queries)

    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output.output, expected_output.output, atol=1e-6, rtol=1e-6)
        assert actual_output.indices.numel() == 0
        assert actual_output.denominator_indices is None
        assert actual_output.search_result is None
        assert actual_output.selected_count == expected_output.indices.numel()
        assert actual_output.denominator_count == expected_output.denominator_indices.numel()


def test_forward_many_all_pq_no_details_matches_detailed_output():
    torch.manual_seed(68)
    keys = torch.randn(96, 32)
    values = torch.randn(96, 16)
    queries = torch.randn(4, 32)
    index_config = IVFPQConfig(
        num_lists=8,
        nprobe=4,
        num_subspaces=4,
        num_bits=4,
        coarse_max_iter=4,
        pq_max_iter=4,
        seed=69,
    )
    detail_config = SparseAttentionConfig(
        sink_tokens=4,
        local_window=8,
        retrieval_top_fraction=0.25,
        mode="hybrid",
        hybrid_value_mode="centroid",
        hybrid_topk_source="all_pq",
        hybrid_denominator_source="all_pq",
    )
    lean_config = SparseAttentionConfig(
        sink_tokens=4,
        local_window=8,
        retrieval_top_fraction=0.25,
        mode="hybrid",
        hybrid_value_mode="centroid",
        hybrid_topk_source="all_pq",
        hybrid_denominator_source="all_pq",
        collect_attention_details=False,
    )
    detailed = IVFPQSparseAttention(index_config, detail_config).build_cache(keys, values)
    lean = IVFPQSparseAttention(index_config, lean_config).build_cache(keys, values)

    expected = detailed.forward_many(queries)
    actual = lean.forward_many(queries)

    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output.output, expected_output.output, atol=1e-6, rtol=1e-6)
        assert actual_output.indices.numel() == 0
        assert actual_output.logits.numel() == 0
        assert actual_output.denominator_indices is None
        assert actual_output.search_result is None
        assert actual_output.selected_count == expected_output.indices.numel()
        assert actual_output.denominator_count == expected_output.denominator_indices.numel()
        assert actual_output.approx_retrieval_count == expected_output.approx_retrieval_logits.numel()
        assert actual_output.exact_retrieval_count == expected_output.exact_retrieval_logits.numel()
        assert actual_output.approx_retrieval_count == expected_output.approx_retrieval_logits.numel()
        assert actual_output.exact_retrieval_count == expected_output.exact_retrieval_logits.numel()
        assert actual_output.candidate_count == expected_output.search_result.candidate_indices.numel()
        assert actual_output.probed_list_count == expected_output.search_result.probed_lists.numel()


def test_forward_many_no_exact_rerank_skips_exact_key_fetch():
    torch.manual_seed(70)
    keys = torch.randn(96, 32)
    values = torch.randn(96, 16)
    queries = torch.randn(4, 32)
    index_config = IVFPQConfig(
        num_lists=8,
        nprobe=4,
        num_subspaces=4,
        num_bits=4,
        coarse_max_iter=4,
        pq_max_iter=4,
        seed=71,
    )
    detail_config = SparseAttentionConfig(
        sink_tokens=4,
        local_window=8,
        retrieval_top_fraction=0.25,
        mode="hybrid",
        hybrid_value_mode="denominator",
        hybrid_topk_source="all_pq",
        hybrid_denominator_source="all_pq",
        exact_rerank=False,
    )
    lean_config = SparseAttentionConfig(
        sink_tokens=4,
        local_window=8,
        retrieval_top_fraction=0.25,
        mode="hybrid",
        hybrid_value_mode="denominator",
        hybrid_topk_source="all_pq",
        hybrid_denominator_source="all_pq",
        exact_rerank=False,
        collect_attention_details=False,
    )
    detailed = IVFPQSparseAttention(index_config, detail_config).build_cache(keys, values)
    lean = IVFPQSparseAttention(index_config, lean_config).build_cache(keys, values)

    expected = detailed.forward_many(queries)
    lean.cache.reset_fetch_stats()
    actual = lean.forward_many(queries)

    full_count = lean.cache.regions.sink.numel() + lean.cache.regions.local.numel()
    retrieval_len = lean.cache.regions.retrieval.numel()
    expected_topk = int((retrieval_len * 0.25) + 0.999999)
    assert lean.cache.fetched_key_rows == full_count
    assert lean.cache.fetched_value_rows == full_count + queries.shape[0] * expected_topk
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output.output, expected_output.output, atol=1e-5, rtol=1e-5)
        assert expected_output.exact_retrieval_logits is None
        assert actual_output.exact_retrieval_logits is None
        assert actual_output.indices.numel() == 0
        assert actual_output.denominator_indices is None
        assert actual_output.selected_count == expected_output.indices.numel()
        assert actual_output.denominator_count == expected_output.denominator_indices.numel()
        assert actual_output.exact_retrieval_count == expected_topk


def test_forward_many_sparse_shares_materialized_kv_fetch(monkeypatch):
    torch.manual_seed(60)
    keys = torch.randn(96, 32)
    values = torch.randn(96, 16)
    queries = torch.randn(4, 32)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            topk_block_size=8,
            rotation="random_orthogonal",
            direction_normalize=True,
            seed=61,
        ),
        SparseAttentionConfig(
            sink_tokens=4,
            local_window=8,
            retrieval_top_fraction=0.25,
            mode="sparse",
            candidate_budget=24,
            kv_storage="cpu",
        ),
    ).build_cache(keys, values)

    expected = tuple(sparse.forward(query) for query in queries)
    sparse.cache.reset_fetch_stats()
    calls = []
    original_search_many = sparse.cache.index.search_many

    def counting_search_many(query_batch, **kwargs):
        calls.append((tuple(query_batch.shape), kwargs))
        return original_search_many(query_batch, **kwargs)

    monkeypatch.setattr(sparse.cache.index, "search_many", counting_search_many)

    actual = sparse.forward_many(queries)

    union_count = torch.unique(torch.cat([output.indices for output in actual], dim=0)).numel()
    assert len(calls) == 1
    assert sparse.cache.kv_fetches == 1
    assert sparse.cache.fetched_key_rows == union_count
    assert sparse.cache.fetched_value_rows == union_count
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output.indices, expected_output.indices)
        torch.testing.assert_close(actual_output.logits, expected_output.logits, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_output.output, expected_output.output, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            actual_output.exact_retrieval_logits,
            expected_output.exact_retrieval_logits,
            atol=1e-6,
            rtol=1e-6,
        )


def test_forward_many_hybrid_shares_materialized_kv_fetch():
    torch.manual_seed(64)
    keys = torch.randn(96, 32)
    values = torch.randn(96, 16)
    queries = torch.randn(4, 32)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            topk_block_size=8,
            rotation="random_orthogonal",
            direction_normalize=True,
            seed=65,
        ),
        SparseAttentionConfig(
            sink_tokens=4,
            local_window=8,
            retrieval_top_fraction=0.25,
            mode="hybrid",
            hybrid_value_mode="centroid",
            candidate_budget=24,
            kv_storage="cpu",
        ),
    ).build_cache(keys, values)

    expected = tuple(sparse.forward(query) for query in queries)
    sparse.cache.reset_fetch_stats()

    actual = sparse.forward_many(queries)

    union_count = torch.unique(torch.cat([output.indices for output in actual], dim=0)).numel()
    assert sparse.cache.kv_fetches == 1
    assert sparse.cache.fetched_key_rows == union_count
    assert sparse.cache.fetched_value_rows == union_count
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output.indices, expected_output.indices)
        torch.testing.assert_close(actual_output.logits, expected_output.logits, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_output.output, expected_output.output, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            actual_output.denominator_logits,
            expected_output.denominator_logits,
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            actual_output.background_weight_mass,
            expected_output.background_weight_mass,
            atol=1e-6,
            rtol=1e-6,
        )


def test_hybrid_attention_top_p_selects_minimum_approx_softmax_mass():
    torch.manual_seed(19)
    keys = torch.randn(84, 32)
    values = torch.randn(84, 16)
    query = torch.randn(32)
    top_p = 0.65

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=7,
            nprobe=4,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=5,
            pq_max_iter=5,
            seed=20,
        ),
        SparseAttentionConfig(
            sink_tokens=4,
            local_window=8,
            retrieval_top_p=top_p,
            mode="hybrid",
            hybrid_value_mode="denominator",
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)
    sorted_logits, sorted_indices = torch.sort(output.approx_retrieval_logits, descending=True)
    cumulative_mass = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    expected_topk = int(torch.searchsorted(cumulative_mass, torch.tensor(top_p)).item()) + 1

    assert output.search_result.indices.numel() == expected_topk
    torch.testing.assert_close(output.search_result.indices, sorted_indices[:expected_topk])
    assert cumulative_mass[expected_topk - 1] >= top_p
    if expected_topk > 1:
        assert cumulative_mass[expected_topk - 2] < top_p


def test_hybrid_attention_top_p_denominator_scope_counts_full_token_mass():
    torch.manual_seed(50)
    dim = 16
    keys = torch.randn(48, dim) * 0.05
    values = torch.randn(48, 8)
    query = torch.zeros(dim)
    query[0] = 1.0
    keys[:2] = 0
    keys[:2, 0] = 20
    keys[-4:] = 0
    keys[-4:, 0] = 20
    top_p = 0.80

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=5,
            nprobe=3,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=51,
        ),
        SparseAttentionConfig(
            sink_tokens=2,
            local_window=4,
            retrieval_top_p=top_p,
            retrieval_top_p_scope="denominator",
            mode="hybrid",
            hybrid_value_mode="denominator",
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)
    full_count = sparse.cache.regions.sink.numel() + sparse.cache.regions.local.numel()

    assert output.search_result is None
    assert output.exact_retrieval_logits is None
    torch.testing.assert_close(
        output.indices,
        torch.cat([sparse.cache.regions.sink, sparse.cache.regions.local]),
    )
    assert output.denominator_weights[:full_count].sum() >= top_p
    assert output.weights.sum() >= top_p


def test_retrieval_top_fraction_and_top_p_are_mutually_exclusive():
    try:
        IVFPQSparseAttention(
            IVFPQConfig(num_lists=4, num_subspaces=4, num_bits=2),
            SparseAttentionConfig(retrieval_top_fraction=0.1, retrieval_top_p=0.9),
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected mutually exclusive retrieval policy to be rejected")


def test_denominator_top_p_scope_requires_hybrid_mode():
    try:
        IVFPQSparseAttention(
            IVFPQConfig(num_lists=4, num_subspaces=4, num_bits=2),
            SparseAttentionConfig(retrieval_top_p=0.8, retrieval_top_p_scope="denominator"),
        )
    except ValueError as exc:
        assert "requires mode='hybrid'" in str(exc)
    else:
        raise AssertionError("expected denominator top-p scope to require hybrid mode")


def test_hybrid_centroid_value_mode_adds_approximate_background_values():
    torch.manual_seed(8)
    keys = torch.randn(88, 32)
    values = torch.randn(88, 12)
    query = torch.randn(32)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=7,
            nprobe=4,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=5,
            pq_max_iter=5,
            seed=9,
        ),
        SparseAttentionConfig(
            sink_tokens=3,
            local_window=9,
            retrieval_top_fraction=0.10,
            mode="hybrid",
            hybrid_value_mode="centroid",
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)

    assert output.search_result is not None
    assert output.background_weight_mass is not None
    assert output.background_output is not None
    assert sparse.cache.retrieval_value_centroids is not None
    assert torch.isclose(output.denominator_weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert output.weights.sum() < 1.0

    full_indices = torch.cat([sparse.cache.regions.sink, sparse.cache.regions.local], dim=0)
    full_count = full_indices.numel()
    full_weights = output.denominator_weights[:full_count]
    retrieval_weights = output.denominator_weights[full_count:].clone()

    selected_local = output.search_result.indices
    selected_global = sparse.cache.retrieval_global_indices(selected_local)
    selected_weights = retrieval_weights[selected_local]
    retrieval_weights[selected_local] = 0

    expected_mass = torch.zeros_like(output.background_weight_mass)
    from pq_hsa.index.ivfpq import as_index

    expected_mass.index_add_(0, as_index(sparse.cache.index.list_ids), retrieval_weights)
    expected_background = expected_mass.matmul(sparse.cache.retrieval_value_centroids)
    expected = (
        full_weights.matmul(values[full_indices])
        + selected_weights.matmul(values[selected_global])
        + expected_background
    )

    torch.testing.assert_close(output.background_weight_mass, expected_mass, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output.background_output, expected_background, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output.output, expected, atol=1e-6, rtol=1e-6)


def test_hybrid_full_value_mode_uses_all_denominator_values():
    torch.manual_seed(9)
    keys = torch.randn(64, 32)
    values = torch.randn(64, 10)
    query = torch.randn(32)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=6,
            nprobe=3,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=5,
            pq_max_iter=5,
            seed=10,
        ),
        SparseAttentionConfig(
            sink_tokens=2,
            local_window=6,
            retrieval_top_fraction=0.20,
            mode="hybrid",
            hybrid_value_mode="full",
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)

    assert output.background_weight_mass is None
    assert output.background_output is None
    torch.testing.assert_close(output.indices, output.denominator_indices)
    torch.testing.assert_close(output.logits, output.denominator_logits)
    torch.testing.assert_close(output.weights, output.denominator_weights)
    assert torch.isclose(output.weights.sum(), torch.tensor(1.0), atol=1e-6)
    torch.testing.assert_close(
        output.output,
        output.denominator_weights.matmul(values[output.denominator_indices]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_hybrid_cpu_kv_storage_matches_dense_when_all_tokens_selected():
    torch.manual_seed(34)
    keys = torch.randn(48, 24)
    values = torch.randn(48, 12)
    query = torch.randn(24)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=6,
            nprobe=6,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=35,
        ),
        SparseAttentionConfig(
            retrieval_top_fraction=1.0,
            mode="hybrid",
            hybrid_value_mode="full",
            kv_storage="cpu",
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)
    expected = dense_attention(query, keys, values)

    assert sparse.cache.kv_storage == "cpu"
    assert sparse.cache.keys.device.type == "cpu"
    assert sparse.cache.values.device.type == "cpu"
    torch.testing.assert_close(output.output, expected.output, atol=1e-5, rtol=1e-5)


def test_hybrid_cpu_kv_storage_coalesces_materialized_fetches():
    torch.manual_seed(40)
    keys = torch.randn(72, 24)
    values = torch.randn(72, 12)
    query = torch.randn(24)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=6,
            nprobe=3,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=41,
        ),
        SparseAttentionConfig(
            sink_tokens=4,
            local_window=8,
            retrieval_top_fraction=0.25,
            mode="hybrid",
            hybrid_value_mode="denominator",
            kv_storage="cpu",
        ),
    ).build_cache(keys, values)

    sparse.cache.reset_fetch_stats()
    output = sparse.forward(query)

    assert sparse.cache.kv_fetches == 1
    assert sparse.cache.key_fetches == 1
    assert sparse.cache.value_fetches == 1
    assert sparse.cache.fetched_key_rows == output.indices.numel()
    assert sparse.cache.fetched_value_rows == output.indices.numel()
    assert torch.isfinite(output.output).all()


def test_hybrid_prefetch_full_kv_cpu_fallback_matches_regular_fetch():
    torch.manual_seed(42)
    keys = torch.randn(80, 24)
    values = torch.randn(80, 12)
    query = torch.randn(24)

    index_config = IVFPQConfig(
        num_lists=8,
        nprobe=4,
        num_subspaces=4,
        num_bits=3,
        coarse_max_iter=4,
        pq_max_iter=4,
        seed=43,
    )
    base_config = dict(
        sink_tokens=4,
        local_window=8,
        retrieval_top_fraction=0.25,
        mode="hybrid",
        hybrid_value_mode="denominator",
        kv_storage="cpu",
    )
    regular = IVFPQSparseAttention(
        index_config,
        SparseAttentionConfig(**base_config, prefetch_full_kv=False),
    ).build_cache(keys, values)
    prefetched = IVFPQSparseAttention(
        index_config,
        SparseAttentionConfig(**base_config, prefetch_full_kv=True),
    ).build_cache(keys, values)

    expected = regular.forward(query)
    prefetched.cache.reset_fetch_stats()
    output = prefetched.forward(query)

    assert prefetched.cache.kv_prefetches == 0
    assert prefetched.cache.kv_fetches >= 1
    torch.testing.assert_close(output.indices, expected.indices)
    torch.testing.assert_close(output.logits, expected.logits, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output.output, expected.output, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hybrid_cpu_kv_storage_fetches_selected_kv_to_cuda_query():
    torch.manual_seed(36)
    keys = torch.randn(40, 24, device="cuda")
    values = torch.randn(40, 12, device="cuda")
    query = torch.randn(24, device="cuda")

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=5,
            nprobe=5,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=37,
        ),
        SparseAttentionConfig(
            retrieval_top_fraction=1.0,
            mode="hybrid",
            hybrid_value_mode="full",
            kv_storage="cpu",
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)
    expected = dense_attention(query, keys, values)

    assert sparse.cache.keys.device.type == "cpu"
    assert sparse.cache.index.coarse_centroids.device.type == "cuda"
    assert output.output.device.type == "cuda"
    torch.testing.assert_close(output.output, expected.output, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hybrid_pinned_cpu_kv_prefetches_full_region_to_cuda_query():
    torch.manual_seed(44)
    keys = torch.randn(56, 24, device="cuda")
    values = torch.randn(56, 12, device="cuda")
    query = torch.randn(24, device="cuda")

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=7,
            nprobe=4,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=45,
        ),
        SparseAttentionConfig(
            sink_tokens=4,
            local_window=8,
            retrieval_top_fraction=0.25,
            mode="hybrid",
            hybrid_value_mode="denominator",
            kv_storage="cpu",
            pin_offloaded_kv=True,
            prefetch_full_kv=True,
        ),
    ).build_cache(keys, values)

    sparse.cache.reset_fetch_stats()
    output = sparse.forward(query)
    torch.cuda.synchronize()

    assert sparse.cache.keys.device.type == "cpu"
    assert sparse.cache.keys.is_pinned()
    assert sparse.cache.kv_prefetches == 1
    assert output.output.device.type == "cuda"
    assert torch.isfinite(output.output).all()


def test_hybrid_topk_source_ivf_candidates_limits_exact_loads_to_probed_lists():
    torch.manual_seed(14)
    keys = torch.randn(96, 32)
    values = torch.randn(96, 16)
    query = torch.randn(32)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=12,
            nprobe=1,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=5,
            pq_max_iter=5,
            seed=15,
        ),
        SparseAttentionConfig(
            sink_tokens=3,
            local_window=7,
            retrieval_topk=8,
            nprobe=1,
            candidate_budget=10,
            mode="hybrid",
            hybrid_value_mode="centroid",
            hybrid_topk_source="ivf_candidates",
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)
    retrieval_len = sparse.cache.regions.retrieval.numel()

    assert output.search_result is not None
    assert output.search_result.probed_lists.numel() == 1
    assert output.search_result.candidate_indices.numel() <= 10
    assert output.approx_retrieval_logits.shape == (retrieval_len,)
    assert output.denominator_indices.shape[0] == (
        sparse.cache.regions.sink.numel()
        + sparse.cache.regions.local.numel()
        + retrieval_len
    )
    assert output.exact_retrieval_logits.shape == output.search_result.indices.shape

    from pq_hsa.index.ivfpq import as_index, gather_ids

    probed = set(output.search_result.probed_lists.tolist())
    candidate_lists = as_index(gather_ids(sparse.cache.index.list_ids, output.search_result.candidate_indices)).tolist()
    selected_lists = as_index(gather_ids(sparse.cache.index.list_ids, output.search_result.indices)).tolist()
    assert set(candidate_lists).issubset(probed)
    assert set(selected_lists).issubset(probed)


def test_hybrid_denominator_source_ivf_candidates_limits_softmax_denominator():
    torch.manual_seed(62)
    keys = torch.randn(96, 32)
    values = torch.randn(96, 16)
    query = torch.randn(32)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=12,
            nprobe=1,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=5,
            pq_max_iter=5,
            seed=63,
        ),
        SparseAttentionConfig(
            sink_tokens=3,
            local_window=7,
            retrieval_topk=8,
            nprobe=1,
            candidate_budget=10,
            mode="hybrid",
            hybrid_value_mode="centroid",
            hybrid_topk_source="ivf_candidates",
            hybrid_denominator_source="ivf_candidates",
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)
    assert output.search_result is not None

    full_count = sparse.cache.regions.sink.numel() + sparse.cache.regions.local.numel()
    candidate_count = output.search_result.candidate_indices.numel()
    assert output.approx_retrieval_logits.shape == (candidate_count,)
    assert output.denominator_indices.shape[0] == full_count + candidate_count

    expected_retrieval_denominator = sparse.cache.retrieval_global_indices(
        output.search_result.candidate_indices
    )
    torch.testing.assert_close(
        output.denominator_indices[full_count:],
        expected_retrieval_denominator,
    )


def test_forward_many_candidate_denominator_skips_full_retrieval_scores(monkeypatch):
    torch.manual_seed(64)
    keys = torch.randn(96, 32)
    values = torch.randn(96, 16)
    queries = torch.randn(4, 32)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=8,
            nprobe=2,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=65,
        ),
        SparseAttentionConfig(
            sink_tokens=4,
            local_window=8,
            retrieval_top_fraction=0.25,
            mode="hybrid",
            hybrid_value_mode="centroid",
            hybrid_topk_source="ivf_candidates",
            hybrid_denominator_source="ivf_candidates",
            candidate_budget=24,
        ),
    ).build_cache(keys, values)

    def forbidden_approximate_scores(*args, **kwargs):
        raise AssertionError("candidate denominator should not full-scan retrieval scores")

    search_many_calls = []
    original_search_many = sparse.cache.index.search_many

    def counting_search_many(query_batch, **kwargs):
        search_many_calls.append((tuple(query_batch.shape), kwargs))
        return original_search_many(query_batch, **kwargs)

    monkeypatch.setattr(sparse.cache.index, "approximate_scores", forbidden_approximate_scores)
    monkeypatch.setattr(sparse.cache.index, "search_many", counting_search_many)

    outputs = sparse.forward_many(queries)

    assert len(search_many_calls) == 1
    assert search_many_calls[0][0] == tuple(queries.shape)
    for output in outputs:
        assert output.search_result is not None
        assert output.approx_retrieval_logits.numel() == output.search_result.candidate_indices.numel()


def test_append_buffers_recent_tokens_until_index_update_interval():
    torch.manual_seed(7)
    keys = torch.randn(32, 16)
    values = torch.randn(32, 8)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=4,
            nprobe=2,
            num_subspaces=4,
            num_bits=2,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=8,
        ),
        SparseAttentionConfig(
            sink_tokens=2,
            local_window=4,
            retrieval_topk=4,
            mode="hybrid",
            index_update_interval=3,
        ),
    ).build_cache(keys, values)

    initial_retrieval_len = sparse.cache.regions.retrieval.numel()
    initial_rebuilds = sparse.cache.index_rebuilds
    initial_adds = sparse.cache.index_adds

    sparse.append(torch.randn(16), torch.randn(8))
    sparse.append(torch.randn(16), torch.randn(8))

    assert sparse.cache.regions.retrieval.numel() == initial_retrieval_len
    assert sparse.cache.regions.local.numel() == 6
    assert sparse.cache.index_rebuilds == initial_rebuilds
    assert sparse.cache.index_adds == initial_adds

    sparse.append(torch.randn(16), torch.randn(8))

    assert sparse.cache.regions.retrieval.numel() == initial_retrieval_len + 3
    assert sparse.cache.regions.local.numel() == 4
    assert sparse.cache.index_rebuilds == initial_rebuilds
    assert sparse.cache.index_adds == initial_adds + 1
    assert sparse.cache.index.num_vectors == sparse.cache.regions.retrieval.numel()


def test_append_can_force_full_rebuild_update_strategy():
    torch.manual_seed(10)
    keys = torch.randn(28, 16)
    values = torch.randn(28, 8)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=4,
            nprobe=2,
            num_subspaces=4,
            num_bits=2,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=11,
        ),
        SparseAttentionConfig(
            sink_tokens=1,
            local_window=3,
            retrieval_topk=4,
            mode="hybrid",
            index_update_interval=2,
            index_update_strategy="rebuild",
        ),
    ).build_cache(keys, values)

    initial_rebuilds = sparse.cache.index_rebuilds
    initial_adds = sparse.cache.index_adds

    sparse.append(torch.randn(16), torch.randn(8))
    sparse.append(torch.randn(16), torch.randn(8))

    assert sparse.cache.index_rebuilds == initial_rebuilds + 1
    assert sparse.cache.index_adds == initial_adds
    assert sparse.cache.index.num_vectors == sparse.cache.regions.retrieval.numel()


def test_append_can_periodically_warm_rebuild_codebooks():
    torch.manual_seed(17)
    keys = torch.randn(28, 16)
    values = torch.randn(28, 8)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=4,
            nprobe=2,
            num_subspaces=4,
            num_bits=2,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=18,
        ),
        SparseAttentionConfig(
            sink_tokens=1,
            local_window=3,
            retrieval_topk=4,
            mode="hybrid",
            hybrid_value_mode="centroid",
            index_update_interval=2,
            codebook_refresh_interval=2,
        ),
    ).build_cache(keys, values)

    initial_rebuilds = sparse.cache.index_rebuilds
    initial_adds = sparse.cache.index_adds
    initial_refreshes = sparse.cache.codebook_refreshes

    sparse.append(torch.randn(16), torch.randn(8))
    sparse.append(torch.randn(16), torch.randn(8))

    assert sparse.cache.index_rebuilds == initial_rebuilds + 1
    assert sparse.cache.index_adds == initial_adds
    assert sparse.cache.codebook_refreshes == initial_refreshes + 1
    assert sparse.cache.index.num_vectors == sparse.cache.regions.retrieval.numel()
    assert (
        sparse.cache.retrieval_value_centroids.shape[0]
        == sparse.cache.index.coarse_centroids.shape[0]
    )

    output = sparse.forward(torch.randn(16))
    assert torch.isfinite(output.output).all()


def test_append_can_update_codebooks_with_online_strategy():
    torch.manual_seed(46)
    keys = torch.randn(30, 16)
    values = torch.randn(30, 8)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=4,
            nprobe=2,
            num_subspaces=4,
            num_bits=2,
            coarse_max_iter=4,
            pq_max_iter=4,
            online_codebook_lr=0.5,
            seed=47,
        ),
        SparseAttentionConfig(
            sink_tokens=1,
            local_window=3,
            retrieval_topk=4,
            mode="hybrid",
            hybrid_value_mode="centroid",
            index_update_interval=2,
            index_update_strategy="online",
        ),
    ).build_cache(keys, values)

    initial_rebuilds = sparse.cache.index_rebuilds
    initial_adds = sparse.cache.index_adds
    initial_online_updates = sparse.cache.online_codebook_updates
    old_coarse = sparse.cache.index.coarse_centroids.clone()
    old_codebooks = sparse.cache.index.pq.codebooks.clone()

    sparse.append(torch.randn(16) + 3.0, torch.randn(8))
    sparse.append(torch.randn(16) + 3.0, torch.randn(8))

    assert sparse.cache.index_rebuilds == initial_rebuilds
    assert sparse.cache.index_adds == initial_adds
    assert sparse.cache.online_codebook_updates == initial_online_updates + 1
    assert sparse.cache.index.num_vectors == sparse.cache.regions.retrieval.numel()
    assert sparse.cache.retrieval_value_centroids.shape[0] == sparse.cache.index.coarse_centroids.shape[0]
    assert not torch.allclose(sparse.cache.index.coarse_centroids, old_coarse)
    assert not torch.allclose(sparse.cache.index.pq.codebooks, old_codebooks)

    output = sparse.forward(torch.randn(16))
    assert torch.isfinite(output.output).all()


def test_append_can_defer_online_index_update_until_explicit_apply():
    torch.manual_seed(50)
    keys = torch.randn(30, 16)
    values = torch.randn(30, 8)
    query = torch.randn(16)

    index_config = IVFPQConfig(
        num_lists=4,
        nprobe=2,
        num_subspaces=4,
        num_bits=2,
        coarse_max_iter=4,
        pq_max_iter=4,
        online_codebook_lr=0.5,
        seed=51,
    )
    attention_config = SparseAttentionConfig(
        sink_tokens=1,
        local_window=3,
        retrieval_topk=4,
        mode="hybrid",
        hybrid_value_mode="centroid",
        index_update_interval=2,
        index_update_strategy="deferred",
    )
    sparse = IVFPQSparseAttention(index_config, attention_config).build_cache(keys, values)

    initial_retrieval_len = sparse.cache.regions.retrieval.numel()
    initial_rebuilds = sparse.cache.index_rebuilds
    initial_adds = sparse.cache.index_adds
    initial_online_updates = sparse.cache.online_codebook_updates
    initial_deferred_updates = sparse.cache.deferred_index_updates

    sparse.append(torch.randn(16) + 3.0, torch.randn(8))
    sparse.append(torch.randn(16) + 3.0, torch.randn(8))

    assert sparse.cache.pending_index_update
    assert sparse.cache.pending_update_tokens == 2
    assert sparse.cache.deferred_index_updates == initial_deferred_updates + 1
    assert sparse.cache.regions.retrieval.numel() == initial_retrieval_len
    assert sparse.cache.regions.local.numel() == 5
    assert sparse.cache.index_rebuilds == initial_rebuilds
    assert sparse.cache.index_adds == initial_adds
    assert sparse.cache.online_codebook_updates == initial_online_updates
    assert torch.isfinite(sparse.forward(query).output).all()

    assert sparse.apply_pending_index_update()
    assert not sparse.cache.pending_index_update
    assert sparse.cache.pending_update_tokens == 0
    assert sparse.cache.regions.retrieval.numel() == initial_retrieval_len + 2
    assert sparse.cache.regions.local.numel() == 3
    assert sparse.cache.index_rebuilds == initial_rebuilds
    assert sparse.cache.index_adds == initial_adds
    assert sparse.cache.online_codebook_updates == initial_online_updates + 1
    assert sparse.cache.index.num_vectors == sparse.cache.regions.retrieval.numel()
    assert not sparse.apply_pending_index_update()


def test_sparse_kv_cache_state_dict_round_trip_preserves_deferred_update_state():
    torch.manual_seed(52)
    keys = torch.randn(30, 16)
    values = torch.randn(30, 8)

    index_config = IVFPQConfig(
        num_lists=4,
        nprobe=2,
        num_subspaces=4,
        num_bits=2,
        coarse_max_iter=4,
        pq_max_iter=4,
        online_codebook_lr=0.5,
        seed=53,
    )
    attention_config = SparseAttentionConfig(
        sink_tokens=1,
        local_window=3,
        retrieval_topk=4,
        mode="hybrid",
        hybrid_value_mode="centroid",
        index_update_interval=2,
        index_update_strategy="deferred",
    )
    sparse = IVFPQSparseAttention(index_config, attention_config).build_cache(keys, values)
    sparse.append(torch.randn(16) + 2.0, torch.randn(8))
    sparse.append(torch.randn(16) + 2.0, torch.randn(8))

    restored_cache = SparseKVCache.from_state_dict(sparse.cache.state_dict())
    restored = IVFPQSparseAttention(index_config, attention_config)
    restored.cache = restored_cache

    assert restored.cache.pending_index_update
    assert restored.cache.pending_update_tokens == 2
    assert restored.cache.deferred_index_updates == sparse.cache.deferred_index_updates
    assert restored.cache.regions.retrieval.numel() == sparse.cache.regions.retrieval.numel()

    assert restored.apply_pending_index_update()
    assert not restored.cache.pending_index_update
    assert restored.cache.online_codebook_updates == sparse.cache.online_codebook_updates + 1
    assert restored.cache.index.num_vectors == restored.cache.regions.retrieval.numel()


def test_sparse_kv_cache_state_dict_round_trip_preserves_attention_output():
    torch.manual_seed(48)
    keys = torch.randn(36, 16)
    values = torch.randn(36, 8)
    query = torch.randn(16)

    index_config = IVFPQConfig(
        num_lists=4,
        nprobe=2,
        num_subspaces=4,
        num_bits=4,
        coarse_max_iter=4,
        pq_max_iter=4,
        topk_block_size=8,
        online_codebook_lr=0.25,
        seed=49,
    )
    attention_config = SparseAttentionConfig(
        sink_tokens=2,
        local_window=4,
        retrieval_top_fraction=0.25,
        mode="hybrid",
        hybrid_value_mode="centroid",
        index_update_interval=2,
        index_update_strategy="online",
        kv_storage="cpu",
    )
    sparse = IVFPQSparseAttention(index_config, attention_config).build_cache(keys, values)
    sparse.append(torch.randn(16) + 2.0, torch.randn(8))
    sparse.append(torch.randn(16) + 2.0, torch.randn(8))

    expected = sparse.forward(query)
    restored_cache = SparseKVCache.from_state_dict(sparse.cache.state_dict())
    restored = IVFPQSparseAttention(index_config, attention_config)
    restored.cache = restored_cache
    actual = restored.forward(query)

    assert restored.cache.keys.device.type == "cpu"
    assert restored.cache.index.num_vectors == sparse.cache.index.num_vectors
    assert restored.cache.online_codebook_updates == sparse.cache.online_codebook_updates
    torch.testing.assert_close(actual.indices, expected.indices)
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.output, expected.output)
    torch.testing.assert_close(actual.denominator_logits, expected.denominator_logits)


def test_append_with_cpu_kv_storage_updates_index_from_offloaded_fetch():
    torch.manual_seed(38)
    keys = torch.randn(28, 16)
    values = torch.randn(28, 8)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=4,
            nprobe=2,
            num_subspaces=4,
            num_bits=2,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=39,
        ),
        SparseAttentionConfig(
            sink_tokens=1,
            local_window=3,
            retrieval_topk=4,
            mode="hybrid",
            index_update_interval=2,
            kv_storage="cpu",
        ),
    ).build_cache(keys, values)

    initial_rebuilds = sparse.cache.index_rebuilds
    initial_adds = sparse.cache.index_adds

    sparse.append(torch.randn(16), torch.randn(8))
    sparse.append(torch.randn(16), torch.randn(8))

    assert sparse.cache.keys.device.type == "cpu"
    assert sparse.cache.values.device.type == "cpu"
    assert sparse.cache.index_rebuilds == initial_rebuilds
    assert sparse.cache.index_adds == initial_adds + 1
    assert sparse.cache.index.num_vectors == sparse.cache.regions.retrieval.numel()

    output = sparse.forward(torch.randn(16))
    assert torch.isfinite(output.output).all()
