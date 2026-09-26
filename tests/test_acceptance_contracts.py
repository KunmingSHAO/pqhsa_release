import torch

from pq_hsa import IVFPQConfig, IVFPQSparseAttention, SparseAttentionConfig


def test_hybrid_attention_three_source_logit_contract():
    torch.manual_seed(80)
    keys = torch.randn(72, 32)
    values = torch.randn(72, 16)
    query = torch.randn(32)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=6,
            nprobe=3,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=81,
        ),
        SparseAttentionConfig(
            sink_tokens=4,
            local_window=8,
            retrieval_top_fraction=0.25,
            mode="hybrid",
            hybrid_value_mode="denominator",
            hybrid_topk_source="all_pq",
        ),
    ).build_cache(keys, values)

    output = sparse.forward(query)
    cache = sparse.cache
    assert cache is not None
    assert output.search_result is not None
    assert output.denominator_indices is not None
    assert output.denominator_logits is not None
    assert output.exact_retrieval_logits is not None

    scale = sparse.attention_config.scale or (1.0 / (query.shape[-1] ** 0.5))
    full_indices = torch.cat([cache.regions.sink, cache.regions.local], dim=0)
    full_count = full_indices.numel()
    retrieval_len = cache.regions.retrieval.numel()

    torch.testing.assert_close(
        output.denominator_indices,
        torch.cat([full_indices, cache.regions.retrieval], dim=0),
    )
    torch.testing.assert_close(
        output.denominator_logits[:full_count],
        keys[full_indices].matmul(query) * scale,
        atol=1e-5,
        rtol=1e-5,
    )

    selected_local = output.search_result.indices
    selected_global = cache.retrieval_global_indices(selected_local)
    retrieval_logits = output.denominator_logits[full_count:]
    expected_retrieval_logits = output.approx_retrieval_logits.clone()
    expected_retrieval_logits[selected_local] = keys[selected_global].matmul(query) * scale

    assert output.approx_retrieval_logits.shape == (retrieval_len,)
    torch.testing.assert_close(
        retrieval_logits,
        expected_retrieval_logits,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        output.exact_retrieval_logits,
        keys[selected_global].matmul(query) * scale,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.isclose(output.denominator_weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_recent_tokens_remain_exact_until_index_update_interval():
    torch.manual_seed(82)
    keys = torch.randn(40, 24)
    values = torch.randn(40, 12)
    query = torch.randn(24)

    sparse = IVFPQSparseAttention(
        IVFPQConfig(
            num_lists=5,
            nprobe=3,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=3,
            pq_max_iter=3,
            seed=83,
        ),
        SparseAttentionConfig(
            sink_tokens=2,
            local_window=4,
            retrieval_top_fraction=0.25,
            mode="hybrid",
            hybrid_value_mode="denominator",
            index_update_interval=3,
        ),
    ).build_cache(keys, values)

    cache = sparse.cache
    assert cache is not None
    initial_indexed = cache._indexed_length
    initial_vectors = 0 if cache.index is None else cache.index.num_vectors

    sparse.append(torch.randn(24), torch.randn(12))
    sparse.append(torch.randn(24), torch.randn(12))

    assert cache._indexed_length == initial_indexed
    assert cache.pending_update_tokens == 2
    assert cache.index is not None
    assert cache.index.num_vectors == initial_vectors
    assert cache.regions.local.eq(keys.shape[0]).any().item()
    assert cache.regions.local.eq(keys.shape[0] + 1).any().item()

    output = sparse.forward(query)
    assert output.denominator_indices is not None
    assert output.denominator_indices.eq(keys.shape[0]).any().item()
    assert output.denominator_indices.eq(keys.shape[0] + 1).any().item()

    sparse.append(torch.randn(24), torch.randn(12))

    assert cache.pending_update_tokens == 0
    assert cache._indexed_length == keys.shape[0] + 3
    assert cache.index.num_vectors == cache.regions.retrieval.numel()
