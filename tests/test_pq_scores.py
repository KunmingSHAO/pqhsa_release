import torch

import pq_hsa.index.ivfpq as ivfpq_module
from pq_hsa.index.ivfpq import IVFPQConfig, IVFPQIndex, as_index, gather_ids
from pq_hsa.index.kmeans import kmeans_l2
from pq_hsa.index.packing import unpack_4bit_codes
from pq_hsa.index.pq import ProductQuantizer, ProductQuantizerConfig


def _assert_inverted_lists_cover_index(index: IVFPQIndex) -> None:
    assert index.coarse_centroids is not None
    assert index.list_ids is not None
    index._ensure_inverted_lists()
    assert index.inverted_list_offsets is not None
    assert index.inverted_list_indices is not None
    assert index.inverted_list_offsets.device == index.list_ids.device
    assert index.inverted_list_indices.device == index.list_ids.device
    assert index.inverted_list_offsets.shape == (index.coarse_centroids.shape[0] + 1,)
    assert index.inverted_list_offsets[0].item() == 0
    assert index.inverted_list_offsets[-1].item() == index.num_vectors
    assert len(index.inverted_lists) == index.coarse_centroids.shape[0]

    members = []
    for list_id, list_members in enumerate(index.inverted_lists):
        assert list_members.device == index.list_ids.device
        if list_members.numel() > 0:
            gathered = as_index(gather_ids(index.list_ids, list_members))
            torch.testing.assert_close(
                gathered,
                torch.full((list_members.numel(),), list_id, device=gathered.device, dtype=torch.int64),
            )
        members.append(list_members)

    covered = torch.cat(members, dim=0) if members else torch.empty(0, device=index.list_ids.device, dtype=torch.int64)
    torch.testing.assert_close(
        torch.sort(covered.to(torch.int64)).values,
        torch.arange(index.num_vectors, device=index.list_ids.device, dtype=torch.int64),
    )


def test_kmeans_l2_accepts_initial_centroids():
    vectors = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    initial_centroids = torch.tensor([[0.0, 0.0], [10.0, 0.0]])

    centroids, assignments = kmeans_l2(
        vectors,
        2,
        max_iter=0,
        initial_centroids=initial_centroids,
    )

    torch.testing.assert_close(centroids, initial_centroids)
    torch.testing.assert_close(assignments, torch.tensor([0, 0, 1]))


def test_pq_lut_score_matches_decoded_dot_product():
    torch.manual_seed(0)
    vectors = torch.randn(96, 32)
    query = torch.randn(32)

    pq = ProductQuantizer(
        ProductQuantizerConfig(num_subspaces=4, num_bits=3, max_iter=8, seed=1)
    ).train(vectors)
    codes = pq.encode(vectors)

    lut_scores = pq.approximate_scores(query, codes)
    decoded_scores = pq.decode(codes).matmul(query)

    torch.testing.assert_close(lut_scores, decoded_scores, atol=1e-5, rtol=1e-5)


def test_pq_compute_lut_matches_codebook_dot_product_for_single_and_batched_queries():
    torch.manual_seed(3)
    vectors = torch.randn(96, 32)
    query = torch.randn(32)
    queries = torch.randn(5, 32)

    pq = ProductQuantizer(
        ProductQuantizerConfig(num_subspaces=4, num_bits=3, max_iter=8, seed=4)
    ).train(vectors)
    assert pq.codebooks is not None
    assert pq.subdim is not None
    codes = pq.encode(vectors)

    query_chunks = query.reshape(pq.config.num_subspaces, pq.subdim)
    expected_lut = torch.stack(
        [
            query_chunks[subspace].matmul(pq.codebooks[subspace].T)
            for subspace in range(pq.config.num_subspaces)
        ],
        dim=0,
    )

    queries_chunks = queries.reshape(
        queries.shape[0],
        pq.config.num_subspaces,
        pq.subdim,
    )
    expected_batched_lut = torch.stack(
        [
            torch.stack(
                [
                    queries_chunks[batch, subspace].matmul(pq.codebooks[subspace].T)
                    for subspace in range(pq.config.num_subspaces)
                ],
                dim=0,
            )
            for batch in range(queries.shape[0])
        ],
        dim=0,
    )

    lut = pq.compute_lut(query)
    batched_lut = pq.compute_lut(queries)
    decoded = pq.decode(codes)

    assert lut.shape == (pq.config.num_subspaces, pq.config.num_codes)
    assert batched_lut.shape == (
        queries.shape[0],
        pq.config.num_subspaces,
        pq.config.num_codes,
    )
    torch.testing.assert_close(lut, expected_lut, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batched_lut, expected_batched_lut, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        pq.score_from_lut(codes, lut),
        decoded.matmul(query),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        pq.score_from_lut(codes, batched_lut),
        queries.matmul(decoded.T),
        atol=1e-5,
        rtol=1e-5,
    )


def test_pq_train_accepts_initial_codebooks():
    torch.manual_seed(20)
    vectors = torch.randn(64, 16)

    trained = ProductQuantizer(
        ProductQuantizerConfig(num_subspaces=4, num_bits=2, max_iter=4, seed=21)
    ).train(vectors)
    warm_started = ProductQuantizer(
        ProductQuantizerConfig(num_subspaces=4, num_bits=2, max_iter=0, seed=22)
    ).train(vectors, initial_codebooks=trained.codebooks)

    torch.testing.assert_close(warm_started.codebooks, trained.codebooks)


def test_ivfpq_score_matches_reconstructed_dot_product():
    torch.manual_seed(1)
    keys = torch.randn(128, 32)
    query = torch.randn(32)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=8,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=8,
            pq_max_iter=8,
            seed=2,
        )
    ).build(keys)

    indices = torch.arange(keys.shape[0])
    approx_scores = index.approximate_scores(query, indices)
    reconstructed_scores = index.reconstruct(indices).matmul(query)

    torch.testing.assert_close(approx_scores, reconstructed_scores, atol=1e-5, rtol=1e-5)


def test_ivfpq_build_with_shared_codebooks_matches_reconstructed_dot_product():
    torch.manual_seed(40)
    shared_training_keys = torch.randn(160, 32)
    stream_keys = torch.randn(80, 32)
    query = torch.randn(32)
    config = IVFPQConfig(
        num_lists=8,
        nprobe=4,
        num_subspaces=4,
        num_bits=3,
        coarse_max_iter=5,
        pq_max_iter=5,
        seed=41,
    )

    shared = IVFPQIndex(config).build(shared_training_keys)
    assert shared.coarse_centroids is not None
    assert shared.pq is not None
    assert shared.pq.codebooks is not None

    fixed = IVFPQIndex(config).build_with_codebooks(
        stream_keys,
        coarse_centroids=shared.coarse_centroids,
        pq_codebooks=shared.pq.codebooks,
    )

    assert fixed.coarse_centroids is not None
    assert fixed.pq is not None
    assert fixed.pq.codebooks is not None
    assert fixed.coarse_centroids.data_ptr() == shared.coarse_centroids.data_ptr()
    assert fixed.pq.codebooks.data_ptr() == shared.pq.codebooks.data_ptr()
    _assert_inverted_lists_cover_index(fixed)

    indices = torch.arange(stream_keys.shape[0])
    approx_scores = fixed.approximate_scores(query, indices)
    reconstructed_scores = fixed.reconstruct(indices).matmul(query)

    torch.testing.assert_close(approx_scores, reconstructed_scores, atol=1e-5, rtol=1e-5)


def test_ivfpq_random_orthogonal_rotation_preserves_inner_product_contract():
    torch.manual_seed(32)
    keys = torch.randn(96, 32)
    added = torch.randn(11, 32)
    query = torch.randn(32)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=5,
            pq_max_iter=5,
            rotation="random_orthogonal",
            seed=33,
        )
    ).build(keys)

    assert index.rotation_matrix is not None
    rotation = index.rotation_matrix
    torch.testing.assert_close(
        rotation.T.matmul(rotation),
        torch.eye(rotation.shape[0]),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        keys.matmul(query),
        keys.matmul(rotation).matmul(query.matmul(rotation)),
        atol=1e-4,
        rtol=1e-4,
    )

    index.add(added)
    all_indices = torch.arange(keys.shape[0] + added.shape[0])
    approx_scores = index.approximate_scores(query, all_indices)
    reconstructed_scores = index.reconstruct(all_indices).matmul(query)
    result = index.search(query, topk=7, candidate_budget=24)

    torch.testing.assert_close(approx_scores, reconstructed_scores, atol=1e-5, rtol=1e-5)
    assert result.indices.shape == (7,)
    assert torch.all(result.approx_scores[:-1] >= result.approx_scores[1:])


def test_ivfpq_random_orthogonal_rotation_state_round_trip_with_packed_codes():
    torch.manual_seed(34)
    keys = torch.randn(72, 40)
    queries = torch.randn(3, 40)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=5,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            rotation="random_orthogonal",
            topk_block_size=12,
            seed=35,
        )
    ).build(keys)

    stats = index.memory_footprint()
    assert index.stores_packed_codes
    assert index.rotation_matrix is not None
    assert stats.rotation_matrix_bytes == index.rotation_matrix.numel() * index.rotation_matrix.element_size()

    restored = IVFPQIndex.from_state_dict(index.state_dict())
    scores = index.approximate_scores(queries)
    restored_scores = restored.approximate_scores(queries)
    search = index.search(queries[0], topk=6, candidate_budget=18)
    restored_search = restored.search(queries[0], topk=6, candidate_budget=18)

    torch.testing.assert_close(restored.rotation_matrix, index.rotation_matrix)
    torch.testing.assert_close(restored_scores, scores)
    torch.testing.assert_close(restored_search.indices, search.indices)
    torch.testing.assert_close(restored_search.approx_scores, search.approx_scores)


def test_ivfpq_direction_normalization_scales_scores_back_to_qk_units():
    torch.manual_seed(36)
    scales = torch.linspace(0.25, 3.0, steps=84).unsqueeze(1)
    keys = torch.randn(84, 32) * scales
    added = torch.randn(9, 32) * 4.0
    queries = torch.randn(4, 32)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            rotation="random_orthogonal",
            direction_normalize=True,
            topk_block_size=12,
            seed=37,
        )
    ).build(keys)
    index.add(added)
    all_keys = torch.cat([keys, added], dim=0)

    assert index.key_norms is not None
    torch.testing.assert_close(
        index.key_norms,
        torch.linalg.vector_norm(all_keys, dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )

    all_indices = torch.arange(all_keys.shape[0])
    scores = index.approximate_scores(queries, all_indices)
    expected = queries.matmul(index.reconstruct(all_indices).T)
    search = index.search(queries[0], topk=8, candidate_budget=24)

    torch.testing.assert_close(scores, expected, atol=1e-5, rtol=1e-5)
    assert search.indices.shape == (8,)
    assert torch.all(search.approx_scores[:-1] >= search.approx_scores[1:])

    stats = index.memory_footprint()
    assert stats.key_norm_bytes == index.key_norms.numel() * index.key_norms.element_size()

    restored = IVFPQIndex.from_state_dict(index.state_dict())
    restored_scores = restored.approximate_scores(queries, all_indices)
    restored_search = restored.search(queries[0], topk=8, candidate_budget=24)

    torch.testing.assert_close(restored.key_norms, index.key_norms)
    torch.testing.assert_close(restored_scores, scores)
    torch.testing.assert_close(restored_search.indices, search.indices)
    torch.testing.assert_close(restored_search.approx_scores, search.approx_scores)


def test_ivfpq_batched_scores_match_reconstructed_dot_product():
    torch.manual_seed(28)
    keys = torch.randn(96, 32)
    queries = torch.randn(5, 32)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=5,
            pq_max_iter=5,
            seed=29,
        )
    ).build(keys)

    indices = torch.arange(7, 83, 3)
    approx_scores = index.approximate_scores(queries, indices)
    reconstructed_scores = queries.matmul(index.reconstruct(indices).T)

    assert approx_scores.shape == (queries.shape[0], indices.numel())
    torch.testing.assert_close(approx_scores, reconstructed_scores, atol=1e-5, rtol=1e-5)


def test_ivfpq_warm_start_rebuild_reuses_previous_codebooks():
    torch.manual_seed(22)
    initial = torch.randn(80, 24)
    added = torch.randn(8, 24)
    query = torch.randn(24)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=6,
            nprobe=3,
            num_subspaces=4,
            num_bits=2,
            coarse_max_iter=0,
            pq_max_iter=0,
            seed=23,
        )
    ).build(initial)

    old_coarse = index.coarse_centroids.clone()
    old_codebooks = index.pq.codebooks.clone()
    index.rebuild(torch.cat([initial, added], dim=0), warm_start=True)

    all_indices = torch.arange(initial.shape[0] + added.shape[0])
    approx_scores = index.approximate_scores(query, all_indices)
    reconstructed_scores = index.reconstruct(all_indices).matmul(query)

    torch.testing.assert_close(index.coarse_centroids, old_coarse)
    torch.testing.assert_close(index.pq.codebooks, old_codebooks)
    torch.testing.assert_close(approx_scores, reconstructed_scores, atol=1e-5, rtol=1e-5)


def test_ivfpq_search_returns_sorted_approximate_scores():
    torch.manual_seed(2)
    keys = torch.randn(160, 32)
    query = torch.randn(32)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=10,
            nprobe=5,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=6,
            pq_max_iter=6,
            seed=3,
        )
    ).build(keys)

    result = index.search(query, topk=12, candidate_budget=40)

    assert result.indices.shape == (12,)
    assert result.approx_scores.shape == (12,)
    assert result.candidate_indices.numel() <= 40
    assert torch.all(result.approx_scores[:-1] >= result.approx_scores[1:])


def test_ivfpq_packed_search_scores_all_candidates_without_block_merge(monkeypatch):
    torch.manual_seed(52)
    keys = torch.randn(96, 32)
    query = torch.randn(32)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            topk_block_size=16,
            seed=53,
        )
    ).build(keys)

    calls = {"count": 0}
    original_score = ivfpq_module.score_packed_4bit_lut

    def counting_score(*args, **kwargs):
        calls["count"] += 1
        return original_score(*args, **kwargs)

    monkeypatch.setattr(ivfpq_module, "score_packed_4bit_lut", counting_score)

    result = index.search(query, topk=7, candidate_budget=10_000)

    assert result.indices.shape == (7,)
    assert calls["count"] == 1
    assert result.candidate_indices.numel() <= keys.shape[0]
    assert result.candidate_scores.shape == result.candidate_indices.shape
    assert torch.all(result.approx_scores[:-1] >= result.approx_scores[1:])


def test_ivfpq_inverted_lists_drive_candidate_generation():
    torch.manual_seed(44)
    keys = torch.randn(128, 32)
    query = torch.randn(32)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=11,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=5,
            pq_max_iter=5,
            topk_block_size=16,
            seed=45,
        )
    ).build(keys)

    _assert_inverted_lists_cover_index(index)
    result = index.search(query, topk=9, candidate_budget=None)
    expected_candidates = torch.cat(
        [index.inverted_lists[int(list_id)] for list_id in result.probed_lists.tolist()],
        dim=0,
    )
    torch.testing.assert_close(
        torch.sort(result.candidate_indices).values,
        torch.sort(expected_candidates).values,
    )

    stats = index.memory_footprint()
    assert stats.inverted_list_bytes == (
        index.inverted_list_offsets.numel() * index.inverted_list_offsets.element_size()
        + index.inverted_list_indices.numel() * index.inverted_list_indices.element_size()
    )


def test_ivfpq_search_falls_back_to_all_tokens_when_probed_lists_are_empty():
    torch.manual_seed(46)
    keys = torch.randn(96, 24)
    query = torch.randn(24)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=6,
            nprobe=2,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=47,
        )
    ).build(keys)

    assert index.list_ids is not None
    empty_members = torch.empty(0, dtype=torch.long, device=index.list_ids.device)
    index.inverted_lists = tuple(empty_members for _ in index.inverted_lists)
    result = index.search(query, topk=7, candidate_budget=None)

    torch.testing.assert_close(
        result.candidate_indices,
        torch.arange(index.num_vectors, device=index.list_ids.device, dtype=torch.long),
    )


def test_ivfpq_search_many_matches_per_query_search():
    torch.manual_seed(42)
    keys = torch.randn(144, 32)
    queries = torch.randn(5, 32)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=9,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=5,
            pq_max_iter=5,
            topk_block_size=12,
            rotation="random_orthogonal",
            direction_normalize=True,
            seed=43,
        )
    ).build(keys)

    many = index.search_many(queries, topk=11, candidate_budget=29)
    expected = tuple(index.search(query, topk=11, candidate_budget=29) for query in queries)

    assert len(many) == queries.shape[0]
    for actual, expected_result in zip(many, expected, strict=True):
        torch.testing.assert_close(actual.indices, expected_result.indices)
        torch.testing.assert_close(actual.approx_scores, expected_result.approx_scores)
        torch.testing.assert_close(actual.candidate_indices, expected_result.candidate_indices)
        torch.testing.assert_close(actual.candidate_scores, expected_result.candidate_scores)
        torch.testing.assert_close(actual.probed_lists, expected_result.probed_lists)


def test_ivfpq_add_preserves_reconstructed_score_contract():
    torch.manual_seed(13)
    initial = torch.randn(96, 32)
    added = torch.randn(17, 32)
    query = torch.randn(32)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=6,
            pq_max_iter=6,
            seed=14,
        )
    ).build(initial)

    list_ids = index.add(added)
    all_indices = torch.arange(initial.shape[0] + added.shape[0])
    approx_scores = index.approximate_scores(query, all_indices)
    reconstructed_scores = index.reconstruct(all_indices).matmul(query)

    assert index.num_vectors == initial.shape[0] + added.shape[0]
    assert index.codes.shape == (initial.shape[0] + added.shape[0], 4)
    assert index.list_ids.shape == (initial.shape[0] + added.shape[0],)
    assert list_ids.shape == (added.shape[0],)
    _assert_inverted_lists_cover_index(index)
    torch.testing.assert_close(approx_scores, reconstructed_scores, atol=1e-5, rtol=1e-5)


def test_ivfpq_add_appends_inverted_lists_without_full_rebuild(monkeypatch):
    torch.manual_seed(48)
    initial = torch.randn(64, 24)
    added = torch.randn(13, 24)
    query = torch.randn(24)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=7,
            nprobe=3,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            seed=49,
        )
    ).build(initial)

    def fail_rebuild():
        raise AssertionError("add should append CSR membership without full rebuild")

    monkeypatch.setattr(index, "_rebuild_inverted_lists", fail_rebuild)
    index.add(added)
    result = index.search(query, topk=6, candidate_budget=18)

    assert index.num_vectors == initial.shape[0] + added.shape[0]
    assert result.indices.shape == (6,)
    _assert_inverted_lists_cover_index(index)


def test_ivfpq_online_refresh_updates_codebooks_and_preserves_score_contract():
    torch.manual_seed(24)
    initial = torch.randn(72, 24)
    added = torch.randn(10, 24) + 3.0
    query = torch.randn(24)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=6,
            nprobe=3,
            num_subspaces=4,
            num_bits=3,
            coarse_max_iter=5,
            pq_max_iter=5,
            online_codebook_lr=0.5,
            seed=25,
        )
    ).build(initial)

    old_coarse = index.coarse_centroids.clone()
    old_codebooks = index.pq.codebooks.clone()
    all_keys = torch.cat([initial, added], dim=0)
    index.online_refresh(all_keys, update_keys=added)

    all_indices = torch.arange(all_keys.shape[0])
    approx_scores = index.approximate_scores(query, all_indices)
    reconstructed_scores = index.reconstruct(all_indices).matmul(query)

    assert index.num_vectors == all_keys.shape[0]
    assert index.codes.shape == (all_keys.shape[0], 4)
    assert index.list_ids.shape == (all_keys.shape[0],)
    assert not torch.allclose(index.coarse_centroids, old_coarse)
    assert not torch.allclose(index.pq.codebooks, old_codebooks)
    _assert_inverted_lists_cover_index(index)
    torch.testing.assert_close(approx_scores, reconstructed_scores, atol=1e-5, rtol=1e-5)


def test_ivfpq_state_dict_round_trip_preserves_scores_and_search():
    torch.manual_seed(26)
    initial = torch.randn(80, 32)
    added = torch.randn(8, 32) + 2.0
    query = torch.randn(32)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=5,
            pq_max_iter=5,
            topk_block_size=16,
            online_codebook_lr=0.25,
            seed=27,
        )
    ).build(initial)
    all_keys = torch.cat([initial, added], dim=0)
    index.online_refresh(all_keys, update_keys=added)

    restored = IVFPQIndex.from_state_dict(index.state_dict())
    all_indices = torch.arange(all_keys.shape[0])
    original_scores = index.approximate_scores(query, all_indices)
    restored_scores = restored.approximate_scores(query, all_indices)
    original_search = index.search(query, topk=9, candidate_budget=20)
    restored_search = restored.search(query, topk=9, candidate_budget=20)

    assert restored.stores_packed_codes
    assert restored.packed_codes is not None
    _assert_inverted_lists_cover_index(restored)
    torch.testing.assert_close(restored.coarse_centroids, index.coarse_centroids)
    torch.testing.assert_close(restored.pq.codebooks, index.pq.codebooks)
    torch.testing.assert_close(restored_scores, original_scores)
    torch.testing.assert_close(restored_search.candidate_indices, original_search.candidate_indices)
    torch.testing.assert_close(restored_search.indices, original_search.indices)
    torch.testing.assert_close(restored_search.approx_scores, original_search.approx_scores)


def test_ivfpq_uses_packed_codes_for_4bit_storage():
    torch.manual_seed(15)
    initial = torch.randn(80, 40)
    added = torch.randn(9, 40)
    query = torch.randn(40)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=5,
            num_bits=4,
            coarse_max_iter=5,
            pq_max_iter=5,
            seed=16,
        )
    ).build(initial)

    assert index.stores_packed_codes
    assert index.packed_codes is not None
    assert index.packed_codes.dtype == torch.uint8
    assert index.packed_codes.shape == (initial.shape[0], 3)
    assert index._codes is None
    torch.testing.assert_close(index.codes, unpack_4bit_codes(index.packed_codes, 5))

    index.add(added)
    all_indices = torch.arange(initial.shape[0] + added.shape[0])
    approx_scores = index.approximate_scores(query, all_indices)
    reconstructed_scores = index.reconstruct(all_indices).matmul(query)

    assert index.packed_codes.shape == (initial.shape[0] + added.shape[0], 3)
    assert index.codes.shape == (initial.shape[0] + added.shape[0], 5)
    torch.testing.assert_close(approx_scores, reconstructed_scores, atol=1e-5, rtol=1e-5)


def test_ivfpq_packed_batched_scores_match_reconstructed_dot_product():
    torch.manual_seed(30)
    keys = torch.randn(72, 40)
    queries = torch.randn(4, 40)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=5,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            kernel_backend="triton",
            seed=31,
        )
    ).build(keys)

    scores = index.approximate_scores(queries)
    expected = queries.matmul(index.reconstruct().T)

    assert index.stores_packed_codes
    assert scores.shape == (queries.shape[0], keys.shape[0])
    torch.testing.assert_close(scores, expected, atol=1e-5, rtol=1e-5)


def test_ivfpq_rejects_unknown_kernel_backend():
    try:
        IVFPQIndex(IVFPQConfig(kernel_backend="unknown"))
    except ValueError as exc:
        assert "kernel_backend" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown kernel backend")
