import torch

from pq_hsa import (
    IVFPQConfig,
    IVFPQProjectedAttentionModule,
    SparseAttentionConfig,
    build_projected_attention_from_layer,
    dense_attention,
)


def test_projected_attention_module_matches_dense_when_all_tokens_are_exact():
    torch.manual_seed(46)
    hidden = torch.randn(2, 18, 16)
    query_hidden = torch.randn(2, 3, 16)
    num_heads = 2
    head_dim = 6
    value_dim = 5

    q_proj = torch.nn.Linear(16, num_heads * head_dim, bias=False)
    k_proj = torch.nn.Linear(16, num_heads * head_dim, bias=False)
    v_proj = torch.nn.Linear(16, num_heads * value_dim, bias=False)
    o_proj = torch.nn.Linear(num_heads * value_dim, 14, bias=False)

    module = IVFPQProjectedAttentionModule(
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        o_proj=o_proj,
        index_config=IVFPQConfig(
            num_lists=4,
            nprobe=4,
            num_subspaces=3,
            num_bits=2,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=47,
        ),
        attention_config=SparseAttentionConfig(
            retrieval_top_fraction=1.0,
            mode="hybrid",
            hybrid_value_mode="full",
        ),
        num_heads=num_heads,
        head_dim=head_dim,
        value_dim=value_dim,
    ).build_cache(hidden)

    output = module(query_hidden)

    queries = module.project_query(query_hidden)
    keys, values = module.project_kv(hidden)
    dense_context = torch.empty_like(output.context)
    for batch in range(hidden.shape[0]):
        for position in range(query_hidden.shape[1]):
            for head in range(num_heads):
                dense_context[batch, position, head] = dense_attention(
                    queries[batch, position, head],
                    keys[batch, :, head],
                    values[batch, :, head],
                ).output
    expected_hidden = o_proj(dense_context.reshape(2, 3, num_heads * value_dim))

    assert output.context.shape == (2, 3, num_heads, value_dim)
    assert output.hidden_states.shape == (2, 3, 14)
    torch.testing.assert_close(output.context, dense_context, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(output.hidden_states, expected_hidden, atol=1e-5, rtol=1e-5)


def test_projected_attention_module_decode_step_appends_projected_current_kv():
    torch.manual_seed(48)
    hidden = torch.randn(1, 14, 12)
    current = torch.randn(1, 1, 12)
    num_heads = 3
    head_dim = 4

    q_proj = torch.nn.Linear(12, num_heads * head_dim, bias=False)
    k_proj = torch.nn.Linear(12, num_heads * head_dim, bias=False)
    v_proj = torch.nn.Linear(12, num_heads * head_dim, bias=False)
    o_proj = torch.nn.Linear(num_heads * head_dim, 12, bias=False)

    module = IVFPQProjectedAttentionModule(
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        o_proj=o_proj,
        index_config=IVFPQConfig(
            num_lists=4,
            nprobe=4,
            num_subspaces=2,
            num_bits=2,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=49,
        ),
        attention_config=SparseAttentionConfig(
            local_window=3,
            retrieval_top_fraction=1.0,
            mode="hybrid",
            hybrid_value_mode="full",
            index_update_interval=2,
            kv_storage="cpu",
        ),
        num_heads=num_heads,
        head_dim=head_dim,
    ).build_cache(hidden)

    output = module.decode_step(current)
    all_hidden = torch.cat([hidden, current], dim=1)
    queries = module.project_query(current)
    keys, values = module.project_kv(all_hidden)
    dense_context = torch.empty_like(output.context)
    for head in range(num_heads):
        dense_context[0, 0, head] = dense_attention(
            queries[0, 0, head],
            keys[0, :, head],
            values[0, :, head],
        ).output

    assert module.streams[0].cache.keys.device.type == "cpu"
    assert module.streams[0].cache.keys.shape[0] == hidden.shape[1] + 1
    torch.testing.assert_close(output.context, dense_context, atol=1e-5, rtol=1e-5)

    second = module.decode_step(torch.randn(1, 1, 12))
    assert second.hidden_states.shape == (1, 1, 12)
    assert module.streams[0].cache.index_adds == 1


def test_projected_attention_module_cache_state_round_trip_preserves_outputs():
    torch.manual_seed(56)
    hidden = torch.randn(1, 16, 12)
    query_hidden = torch.randn(1, 2, 12)
    num_heads = 3
    head_dim = 4
    index_config = IVFPQConfig(
        num_lists=4,
        nprobe=2,
        num_subspaces=2,
        num_bits=4,
        coarse_max_iter=4,
        pq_max_iter=4,
        topk_block_size=8,
        seed=57,
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
    module = IVFPQProjectedAttentionModule(
        q_proj=torch.nn.Linear(12, num_heads * head_dim, bias=False),
        k_proj=torch.nn.Linear(12, num_heads * head_dim, bias=False),
        v_proj=torch.nn.Linear(12, num_heads * head_dim, bias=False),
        o_proj=torch.nn.Linear(num_heads * head_dim, 12, bias=False),
        index_config=index_config,
        attention_config=attention_config,
        num_heads=num_heads,
        head_dim=head_dim,
    ).build_cache(hidden)
    module.decode_step(torch.randn(1, 1, 12))
    module.decode_step(torch.randn(1, 1, 12))

    expected = module(query_hidden)
    restored = IVFPQProjectedAttentionModule(
        q_proj=torch.nn.Linear(12, num_heads * head_dim, bias=False),
        k_proj=torch.nn.Linear(12, num_heads * head_dim, bias=False),
        v_proj=torch.nn.Linear(12, num_heads * head_dim, bias=False),
        o_proj=torch.nn.Linear(num_heads * head_dim, 12, bias=False),
        index_config=index_config,
        attention_config=attention_config,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    restored.load_state_dict(module.state_dict())
    restored.load_cache_state_dict(module.cache_state_dict())
    actual = restored(query_hidden)

    assert "q_proj.weight" in module.state_dict()
    assert restored.streams[0].cache.keys.device.type == "cpu"
    assert restored.streams[0].cache.index_adds == module.streams[0].cache.index_adds
    torch.testing.assert_close(actual.context, expected.context)
    torch.testing.assert_close(actual.hidden_states, expected.hidden_states)


def test_projected_attention_module_repeats_gqa_key_value_heads():
    torch.manual_seed(50)
    hidden = torch.randn(1, 10, 8)
    num_heads = 4
    num_kv_heads = 2
    head_dim = 3

    module = IVFPQProjectedAttentionModule(
        q_proj=torch.nn.Linear(8, num_heads * head_dim, bias=False),
        k_proj=torch.nn.Linear(8, num_kv_heads * head_dim, bias=False),
        v_proj=torch.nn.Linear(8, num_kv_heads * head_dim, bias=False),
        o_proj=torch.nn.Linear(num_heads * head_dim, 8, bias=False),
        index_config=IVFPQConfig(
            num_lists=2,
            nprobe=2,
            num_subspaces=1,
            num_bits=2,
            coarse_max_iter=2,
            pq_max_iter=2,
            seed=51,
        ),
        attention_config=SparseAttentionConfig(
            retrieval_top_fraction=1.0,
            mode="hybrid",
            hybrid_value_mode="full",
        ),
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
    )

    keys, values = module.project_kv(hidden)

    assert keys.shape == (1, 10, num_heads, head_dim)
    assert values.shape == (1, 10, num_heads, head_dim)
    torch.testing.assert_close(keys[:, :, 0], keys[:, :, 1])
    torch.testing.assert_close(keys[:, :, 2], keys[:, :, 3])


def test_build_projected_attention_from_layer_infers_config_and_reuses_projections():
    torch.manual_seed(52)

    class DummyConfig:
        num_attention_heads = 4
        num_key_value_heads = 2

    class DummyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = DummyConfig()
            self.q_proj = torch.nn.Linear(10, 12, bias=False)
            self.k_proj = torch.nn.Linear(10, 6, bias=False)
            self.v_proj = torch.nn.Linear(10, 6, bias=False)
            self.o_proj = torch.nn.Linear(12, 10, bias=False)

    layer = DummyLayer()
    hidden = torch.randn(1, 12, 10)
    query_hidden = torch.randn(1, 2, 10)

    module = build_projected_attention_from_layer(
        layer,
        index_config=IVFPQConfig(
            num_lists=4,
            nprobe=4,
            num_subspaces=1,
            num_bits=2,
            coarse_max_iter=3,
            pq_max_iter=3,
            seed=53,
        ),
        attention_config=SparseAttentionConfig(
            retrieval_top_fraction=1.0,
            mode="hybrid",
            hybrid_value_mode="full",
        ),
    ).build_cache(hidden)

    output = module(query_hidden)

    assert module.q_proj is layer.q_proj
    assert module.o_proj is layer.o_proj
    assert module.num_heads == 4
    assert module.num_key_value_heads == 2
    assert module.head_dim == 3
    assert output.hidden_states.shape == (1, 2, 10)
