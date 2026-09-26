import torch

from pq_hsa import (
    IVFPQConfig,
    IVFPQDecodeAttentionModule,
    SparseAttentionConfig,
    dense_attention,
)


def test_decode_attention_module_bhsd_matches_dense_when_all_tokens_are_exact():
    torch.manual_seed(42)
    keys = torch.randn(2, 2, 20, 12)
    values = torch.randn(2, 2, 20, 7)
    queries = torch.randn(2, 2, 3, 12)

    module = IVFPQDecodeAttentionModule(
        IVFPQConfig(
            num_lists=4,
            nprobe=4,
            num_subspaces=3,
            num_bits=2,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=43,
        ),
        SparseAttentionConfig(
            retrieval_top_fraction=1.0,
            mode="hybrid",
            hybrid_value_mode="full",
        ),
    ).build_cache(keys, values)

    output = module(queries)
    expected = torch.empty_like(output.context)
    for batch in range(keys.shape[0]):
        for head in range(keys.shape[1]):
            for position in range(queries.shape[2]):
                expected[batch, head, position] = dense_attention(
                    queries[batch, head, position],
                    keys[batch, head],
                    values[batch, head],
                ).output

    assert output.context.shape == (2, 2, 3, 7)
    assert len(output.adapter_output.per_query) == 3
    torch.testing.assert_close(output.context, expected, atol=1e-5, rtol=1e-5)


def test_decode_attention_module_bshd_decode_step_preserves_layout_and_appends():
    torch.manual_seed(44)
    keys = torch.randn(1, 18, 2, 8)
    values = torch.randn(1, 18, 2, 5)
    query = torch.randn(1, 1, 2, 8)
    new_key = torch.randn(1, 1, 2, 8)
    new_value = torch.randn(1, 1, 2, 5)

    module = IVFPQDecodeAttentionModule(
        IVFPQConfig(
            num_lists=4,
            nprobe=4,
            num_subspaces=2,
            num_bits=2,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=45,
        ),
        SparseAttentionConfig(
            local_window=3,
            retrieval_top_fraction=1.0,
            mode="hybrid",
            hybrid_value_mode="full",
            index_update_interval=2,
            kv_storage="cpu",
        ),
        tensor_layout="bshd",
    ).build_cache(keys, values)

    output = module.decode_step(query, new_key, new_value)
    expected_keys = torch.cat([keys, new_key], dim=1).transpose(1, 2)
    expected_values = torch.cat([values, new_value], dim=1).transpose(1, 2)
    expected = torch.stack(
        [
            dense_attention(
                query[0, 0, head],
                expected_keys[0, head],
                expected_values[0, head],
            ).output
            for head in range(keys.shape[2])
        ],
        dim=0,
    ).unsqueeze(0).unsqueeze(1)

    assert output.context.shape == (1, 1, 2, 5)
    assert module.streams[0].cache.keys.device.type == "cpu"
    torch.testing.assert_close(output.context, expected, atol=1e-5, rtol=1e-5)

    second = module.decode_step(
        torch.randn(1, 2, 8),
        torch.randn(1, 2, 8),
        torch.randn(1, 2, 5),
    )
    assert second.context.shape == (1, 2, 5)
    assert module.streams[0].cache.index_adds == 1


def test_decode_attention_module_cache_state_round_trip_preserves_outputs():
    torch.manual_seed(54)
    keys = torch.randn(1, 18, 2, 8)
    values = torch.randn(1, 18, 2, 5)
    query = torch.randn(1, 2, 8)
    index_config = IVFPQConfig(
        num_lists=4,
        nprobe=2,
        num_subspaces=2,
        num_bits=4,
        coarse_max_iter=4,
        pq_max_iter=4,
        topk_block_size=8,
        seed=55,
    )
    attention_config = SparseAttentionConfig(
        sink_tokens=1,
        local_window=3,
        retrieval_top_fraction=0.25,
        mode="hybrid",
        hybrid_value_mode="centroid",
        index_update_interval=2,
        kv_storage="cpu",
    )
    module = IVFPQDecodeAttentionModule(
        index_config,
        attention_config,
        tensor_layout="bshd",
    ).build_cache(keys, values)
    module.decode_step(torch.randn(1, 1, 2, 8), torch.randn(1, 1, 2, 8), torch.randn(1, 1, 2, 5))
    module.decode_step(torch.randn(1, 1, 2, 8), torch.randn(1, 1, 2, 8), torch.randn(1, 1, 2, 5))

    expected = module(query)
    restored = IVFPQDecodeAttentionModule(
        index_config,
        attention_config,
        tensor_layout="bshd",
    ).load_cache_state_dict(module.cache_state_dict())
    actual = restored(query)

    assert restored.streams[0].cache.keys.device.type == "cpu"
    assert restored.streams[0].cache.index_adds == module.streams[0].cache.index_adds
    torch.testing.assert_close(actual.context, expected.context)
