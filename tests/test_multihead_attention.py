import torch

from pq_hsa import (
    IVFPQConfig,
    IVFPQMultiHeadAttention,
    SparseAttentionConfig,
    dense_attention,
)


def test_multihead_hybrid_full_matches_dense_when_all_tokens_selected():
    torch.manual_seed(11)
    keys = torch.randn(2, 2, 32, 16)
    values = torch.randn(2, 2, 32, 8)
    queries = torch.randn(2, 2, 16)

    attention = IVFPQMultiHeadAttention(
        IVFPQConfig(
            num_lists=4,
            nprobe=4,
            num_subspaces=4,
            num_bits=2,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=12,
        ),
        SparseAttentionConfig(
            retrieval_top_fraction=1.0,
            mode="hybrid",
            hybrid_value_mode="full",
        ),
    ).build_cache(keys, values)

    output = attention.forward(queries)
    expected = torch.stack(
        [
            dense_attention(queries[b, h], keys[b, h], values[b, h]).output
            for b in range(keys.shape[0])
            for h in range(keys.shape[1])
        ],
        dim=0,
    ).reshape_as(output.output)

    assert output.output.shape == (2, 2, 8)
    assert output.leading_shape == (2, 2)
    assert len(output.per_stream) == 4
    torch.testing.assert_close(output.output, expected, atol=1e-5, rtol=1e-5)


def test_multihead_append_updates_each_stream_cache():
    torch.manual_seed(12)
    keys = torch.randn(2, 2, 16, 12)
    values = torch.randn(2, 2, 16, 6)

    attention = IVFPQMultiHeadAttention(
        IVFPQConfig(
            num_lists=4,
            nprobe=2,
            num_subspaces=3,
            num_bits=2,
            coarse_max_iter=3,
            pq_max_iter=3,
            seed=13,
        ),
        SparseAttentionConfig(
            sink_tokens=1,
            local_window=3,
            retrieval_topk=4,
            mode="hybrid",
            hybrid_value_mode="centroid",
            index_update_interval=2,
        ),
    ).build_cache(keys, values)

    initial_retrieval_len = attention.streams[0].cache.regions.retrieval.numel()
    initial_rebuilds = attention.streams[0].cache.index_rebuilds
    initial_adds = attention.streams[0].cache.index_adds

    attention.append(torch.randn(2, 2, 12), torch.randn(2, 2, 6))
    for stream in attention.streams:
        assert stream.cache.regions.retrieval.numel() == initial_retrieval_len
        assert stream.cache.regions.local.numel() == 4
        assert stream.cache.index_rebuilds == initial_rebuilds
        assert stream.cache.index_adds == initial_adds

    attention.append(torch.randn(2, 2, 12), torch.randn(2, 2, 6))
    for stream in attention.streams:
        assert stream.cache.regions.retrieval.numel() == initial_retrieval_len + 2
        assert stream.cache.regions.local.numel() == 3
        assert stream.cache.index_rebuilds == initial_rebuilds
        assert stream.cache.index_adds == initial_adds + 1


def test_multihead_can_apply_deferred_index_updates_for_all_streams():
    torch.manual_seed(17)
    keys = torch.randn(2, 2, 16, 12)
    values = torch.randn(2, 2, 16, 6)

    attention = IVFPQMultiHeadAttention(
        IVFPQConfig(
            num_lists=4,
            nprobe=2,
            num_subspaces=3,
            num_bits=2,
            coarse_max_iter=3,
            pq_max_iter=3,
            online_codebook_lr=0.5,
            seed=18,
        ),
        SparseAttentionConfig(
            sink_tokens=1,
            local_window=3,
            retrieval_topk=4,
            mode="hybrid",
            hybrid_value_mode="centroid",
            index_update_interval=2,
            index_update_strategy="deferred",
        ),
    ).build_cache(keys, values)

    attention.append(torch.randn(2, 2, 12), torch.randn(2, 2, 6))
    attention.append(torch.randn(2, 2, 12), torch.randn(2, 2, 6))

    assert all(stream.cache.pending_index_update for stream in attention.streams)
    assert all(stream.cache.online_codebook_updates == 0 for stream in attention.streams)

    assert attention.apply_pending_index_update() == 4
    assert all(not stream.cache.pending_index_update for stream in attention.streams)
    assert all(stream.cache.online_codebook_updates == 1 for stream in attention.streams)
    assert attention.apply_pending_index_update() == 0


def test_multihead_state_dict_round_trip_preserves_outputs():
    torch.manual_seed(14)
    keys = torch.randn(2, 2, 20, 12)
    values = torch.randn(2, 2, 20, 6)
    queries = torch.randn(2, 2, 12)

    attention = IVFPQMultiHeadAttention(
        IVFPQConfig(
            num_lists=4,
            nprobe=2,
            num_subspaces=3,
            num_bits=4,
            coarse_max_iter=3,
            pq_max_iter=3,
            topk_block_size=8,
            seed=15,
        ),
        SparseAttentionConfig(
            sink_tokens=1,
            local_window=3,
            retrieval_top_fraction=0.25,
            mode="hybrid",
            hybrid_value_mode="centroid",
            index_update_interval=2,
            kv_storage="cpu",
        ),
    ).build_cache(keys, values)
    attention.append(torch.randn(2, 2, 12), torch.randn(2, 2, 6))
    attention.append(torch.randn(2, 2, 12), torch.randn(2, 2, 6))

    expected = attention.forward(queries)
    restored = IVFPQMultiHeadAttention.from_state_dict(attention.state_dict())
    actual = restored.forward(queries)

    assert restored.leading_shape == attention.leading_shape
    assert restored.streams[0].cache.keys.device.type == "cpu"
    assert restored.streams[0].cache.index_adds == attention.streams[0].cache.index_adds
    torch.testing.assert_close(actual.output, expected.output)
    for actual_stream, expected_stream in zip(actual.per_stream, expected.per_stream, strict=True):
        torch.testing.assert_close(actual_stream.indices, expected_stream.indices)
        torch.testing.assert_close(actual_stream.logits, expected_stream.logits)


def test_multihead_forward_many_matches_per_position_forward():
    torch.manual_seed(16)
    keys = torch.randn(2, 2, 28, 12)
    values = torch.randn(2, 2, 28, 6)
    queries = torch.randn(2, 2, 3, 12)

    attention = IVFPQMultiHeadAttention(
        IVFPQConfig(
            num_lists=4,
            nprobe=2,
            num_subspaces=3,
            num_bits=4,
            coarse_max_iter=3,
            pq_max_iter=3,
            topk_block_size=8,
            seed=17,
        ),
        SparseAttentionConfig(
            sink_tokens=2,
            local_window=4,
            retrieval_top_fraction=0.25,
            mode="hybrid",
            hybrid_value_mode="centroid",
        ),
    ).build_cache(keys, values)

    expected = tuple(attention.forward(queries[..., position, :]) for position in range(queries.shape[-2]))
    actual = attention.forward_many(queries)

    assert actual.output.shape == (2, 2, 3, 6)
    assert len(actual.per_query) == queries.shape[-2]
    for position, expected_output in enumerate(expected):
        torch.testing.assert_close(actual.output[..., position, :], expected_output.output)
        torch.testing.assert_close(actual.per_query[position].output, expected_output.output)
        for actual_stream, expected_stream in zip(
            actual.per_query[position].per_stream,
            expected_output.per_stream,
            strict=True,
        ):
            torch.testing.assert_close(actual_stream.output, expected_stream.output)
            torch.testing.assert_close(actual_stream.denominator_logits, expected_stream.denominator_logits)
