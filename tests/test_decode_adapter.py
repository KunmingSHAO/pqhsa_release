import torch

from pq_hsa import (
    IVFPQConfig,
    IVFPQDecodeAttentionAdapter,
    SparseAttentionConfig,
    dense_attention,
)


def test_decode_adapter_matches_dense_when_all_tokens_are_exact():
    torch.manual_seed(21)
    keys = torch.randn(2, 3, 24, 12)
    values = torch.randn(2, 3, 24, 6)
    queries = torch.randn(2, 3, 2, 12)

    adapter = IVFPQDecodeAttentionAdapter(
        IVFPQConfig(
            num_lists=4,
            nprobe=4,
            num_subspaces=3,
            num_bits=2,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=22,
        ),
        SparseAttentionConfig(
            retrieval_top_fraction=1.0,
            mode="hybrid",
            hybrid_value_mode="full",
        ),
    ).build_cache(keys, values)

    output = adapter.forward(queries)
    expected = torch.empty_like(output.context)
    for batch in range(keys.shape[0]):
        for head in range(keys.shape[1]):
            for position in range(queries.shape[2]):
                expected[batch, head, position] = dense_attention(
                    queries[batch, head, position],
                    keys[batch, head],
                    values[batch, head],
                ).output

    assert output.context.shape == (2, 3, 2, 6)
    assert len(output.per_query) == 2
    torch.testing.assert_close(output.context, expected, atol=1e-5, rtol=1e-5)


def test_decode_adapter_forward_uses_multihead_forward_many(monkeypatch):
    torch.manual_seed(25)
    keys = torch.randn(1, 2, 18, 8)
    values = torch.randn(1, 2, 18, 5)
    queries = torch.randn(1, 2, 3, 8)

    adapter = IVFPQDecodeAttentionAdapter(
        IVFPQConfig(
            num_lists=4,
            nprobe=2,
            num_subspaces=2,
            num_bits=4,
            coarse_max_iter=3,
            pq_max_iter=3,
            seed=26,
        ),
        SparseAttentionConfig(
            sink_tokens=2,
            local_window=4,
            retrieval_top_fraction=0.25,
            mode="hybrid",
            hybrid_value_mode="centroid",
        ),
    ).build_cache(keys, values)

    calls = []
    original_forward_many = adapter.attention.forward_many

    def counting_forward_many(query_states):
        calls.append(tuple(query_states.shape))
        return original_forward_many(query_states)

    monkeypatch.setattr(adapter.attention, "forward_many", counting_forward_many)

    output = adapter.forward(queries)

    assert calls == [tuple(queries.shape)]
    assert output.context.shape == (1, 2, 3, 5)
    assert len(output.per_query) == queries.shape[-2]


def test_decode_adapter_appends_current_kv_before_attention():
    torch.manual_seed(23)
    keys = torch.randn(1, 2, 12, 8)
    values = torch.randn(1, 2, 12, 5)
    query = torch.randn(1, 2, 1, 8)
    new_key = torch.randn(1, 2, 1, 8)
    new_value = torch.randn(1, 2, 1, 5)

    adapter = IVFPQDecodeAttentionAdapter(
        IVFPQConfig(
            num_lists=4,
            nprobe=4,
            num_subspaces=2,
            num_bits=2,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=24,
        ),
        SparseAttentionConfig(
            local_window=3,
            retrieval_top_fraction=1.0,
            mode="hybrid",
            hybrid_value_mode="full",
            index_update_interval=2,
        ),
    ).build_cache(keys, values)

    output = adapter.decode_step(query, new_key, new_value)
    expected_keys = torch.cat([keys, new_key], dim=2)
    expected_values = torch.cat([values, new_value], dim=2)
    expected = torch.stack(
        [
            dense_attention(
                query[0, head, 0],
                expected_keys[0, head],
                expected_values[0, head],
            ).output
            for head in range(keys.shape[1])
        ],
        dim=0,
    ).unsqueeze(0).unsqueeze(2)

    assert output.context.shape == (1, 2, 1, 5)
    torch.testing.assert_close(output.context, expected, atol=1e-5, rtol=1e-5)
    assert adapter.streams[0].cache.keys.shape[0] == keys.shape[2] + 1

    second_output = adapter.decode_step(
        torch.randn(1, 2, 8),
        torch.randn(1, 2, 8),
        torch.randn(1, 2, 5),
    )

    assert second_output.context.shape == (1, 2, 5)
    assert adapter.streams[0].cache.keys.shape[0] == keys.shape[2] + 2
    assert adapter.streams[0].cache.index_adds == 1
