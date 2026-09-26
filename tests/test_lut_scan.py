import pytest
import torch

import pq_hsa.index.ivfpq as ivfpq_module
import pq_hsa.kernels.lut_scan as lut_scan_module
from pq_hsa import (
    IVFPQConfig,
    IVFPQIndex,
    is_triton_available,
    list_exp_sums_triton,
    pack_4bit_codes,
    score_packed_4bit_lut,
    score_packed_4bit_lut_batched_list_bias_triton,
    score_packed_4bit_lut_batched_triton,
    topk_packed_4bit_lut,
)
from pq_hsa.kernels.triton_lut_scan import final_topk_merge_triton


def test_score_packed_4bit_lut_matches_dense_lut_gather():
    torch.manual_seed(17)
    codes = torch.randint(0, 16, (23, 5))
    packed = pack_4bit_codes(codes)
    lut = torch.randn(5, 16)

    scores = score_packed_4bit_lut(packed, lut, num_subspaces=5)
    expected = lut.gather(1, codes.T).sum(dim=0)

    torch.testing.assert_close(scores, expected)


def test_score_packed_4bit_lut_torch_backend_matches_dense_lut_gather():
    torch.manual_seed(25)
    codes = torch.randint(0, 16, (31, 4))
    packed = pack_4bit_codes(codes)
    lut = torch.randn(4, 16)

    scores = score_packed_4bit_lut(packed, lut, num_subspaces=4, backend="torch")
    expected = lut.gather(1, codes.T).sum(dim=0)

    torch.testing.assert_close(scores, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA and Triton are required",
)
def test_score_packed_4bit_lut_triton_backend_matches_torch_backend():
    torch.manual_seed(26)
    codes = torch.randint(0, 16, (257, 5), device="cuda")
    packed = pack_4bit_codes(codes)
    lut = torch.randn(5, 16, device="cuda")

    triton_scores = score_packed_4bit_lut(packed, lut, num_subspaces=5, backend="triton")
    torch_scores = score_packed_4bit_lut(packed, lut, num_subspaces=5, backend="torch")

    torch.testing.assert_close(triton_scores, torch_scores, atol=1e-5, rtol=1e-5)


def test_score_packed_4bit_lut_matches_batched_dense_lut_gather():
    torch.manual_seed(18)
    codes = torch.randint(0, 16, (19, 4))
    packed = pack_4bit_codes(codes)
    lut = torch.randn(3, 4, 16)

    scores = score_packed_4bit_lut(packed, lut, num_subspaces=4)
    gather_index = codes.T.unsqueeze(0).expand(lut.shape[0], -1, -1)
    expected = lut.gather(2, gather_index).sum(dim=1)

    torch.testing.assert_close(scores, expected)


def test_score_packed_4bit_lut_batched_lut_ignores_single_query_triton_backend():
    torch.manual_seed(36)
    codes = torch.randint(0, 16, (21, 4))
    packed = pack_4bit_codes(codes)
    lut = torch.randn(3, 4, 16)

    scores = score_packed_4bit_lut(packed, lut, num_subspaces=4, backend="triton")
    gather_index = codes.T.unsqueeze(0).expand(lut.shape[0], -1, -1)
    expected = lut.gather(2, gather_index).sum(dim=1)

    torch.testing.assert_close(scores, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA and Triton are required",
)
def test_score_packed_4bit_lut_batched_triton_backend_matches_torch_backend():
    torch.manual_seed(37)
    codes = torch.randint(0, 16, (259, 5), device="cuda")
    packed = pack_4bit_codes(codes)
    lut = torch.randn(5, 16, 3, device="cuda").permute(2, 0, 1)

    direct_scores = score_packed_4bit_lut_batched_triton(packed, lut, num_subspaces=5)
    auto_scores = score_packed_4bit_lut(packed, lut, num_subspaces=5, backend="auto")
    triton_scores = score_packed_4bit_lut(packed, lut, num_subspaces=5, backend="triton")
    torch_scores = score_packed_4bit_lut(packed, lut, num_subspaces=5, backend="torch")

    assert direct_scores.shape == (3, codes.shape[0])
    torch.testing.assert_close(direct_scores, torch_scores, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(auto_scores, torch_scores, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(triton_scores, torch_scores, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA and Triton are required",
)
def test_score_packed_4bit_lut_batched_list_bias_triton_matches_dense_gather():
    torch.manual_seed(40)
    codes = torch.randint(0, 16, (263, 5), device="cuda")
    packed = pack_4bit_codes(codes)
    lut = torch.randn(4, 5, 16, device="cuda")
    list_ids = torch.randint(0, 17, (codes.shape[0],), device="cuda")
    list_scores = torch.randn(lut.shape[0], 17, device="cuda")

    scores = score_packed_4bit_lut_batched_list_bias_triton(
        packed,
        lut,
        list_ids,
        list_scores,
        num_subspaces=5,
    )
    gather_index = codes.T.unsqueeze(0).expand(lut.shape[0], -1, -1)
    expected = lut.gather(2, gather_index).sum(dim=1)
    expected = expected + list_scores.gather(1, list_ids.unsqueeze(0).expand(lut.shape[0], -1))

    torch.testing.assert_close(scores, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA and Triton are required",
)
def test_list_exp_sums_triton_matches_scatter_add():
    torch.manual_seed(41)
    batch = 4
    num_vectors = 257
    num_lists = 19
    logits = torch.randn(batch, num_vectors, device="cuda", dtype=torch.float16)
    row_max = logits.max(dim=1).values
    list_ids = torch.randint(0, num_lists - 1, (num_vectors,), device="cuda")
    order = torch.argsort(list_ids, stable=True)
    sorted_list_ids = list_ids[order]
    counts = torch.bincount(sorted_list_ids, minlength=num_lists)
    offsets = torch.empty(num_lists + 1, device="cuda", dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)

    actual = list_exp_sums_triton(
        logits,
        row_max,
        offsets,
        order,
        num_lists=num_lists,
        block_size=32,
    )
    exp_values = torch.exp(logits - row_max[:, None])
    expected = torch.zeros(batch, num_lists, device="cuda", dtype=logits.dtype)
    expected.scatter_add_(1, list_ids.unsqueeze(0).expand(batch, -1), exp_values)

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


def test_ivfpq_packed_approximate_scores_use_packed_scan_without_full_unpack(monkeypatch):
    torch.manual_seed(19)
    keys = torch.randn(96, 32)
    query = torch.randn(32)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=5,
            pq_max_iter=5,
            seed=20,
        )
    ).build(keys)

    def fail_unpack(*_args, **_kwargs):
        raise AssertionError("approximate_scores should not unpack packed codes")

    monkeypatch.setattr(ivfpq_module, "unpack_4bit_codes", fail_unpack)

    scores = index.approximate_scores(query)

    assert scores.shape == (keys.shape[0],)


def test_topk_packed_4bit_lut_matches_dense_scores_with_budget_and_bias():
    torch.manual_seed(21)
    codes = torch.randint(0, 16, (29, 5))
    packed = pack_4bit_codes(codes)
    lut = torch.randn(5, 16)
    bias = torch.randn(29)

    result = topk_packed_4bit_lut(
        packed,
        lut,
        num_subspaces=5,
        topk=7,
        candidate_budget=13,
        score_bias=bias,
    )
    dense_scores = lut.gather(1, codes.T).sum(dim=0) + bias
    budget_scores, budget_indices = torch.topk(dense_scores, k=13)
    expected_scores, expected_order = torch.topk(budget_scores, k=7)
    expected_indices = budget_indices[expected_order]

    torch.testing.assert_close(result.indices, expected_indices)
    torch.testing.assert_close(result.scores, expected_scores)
    torch.testing.assert_close(result.candidate_indices, budget_indices)
    torch.testing.assert_close(result.candidate_scores, budget_scores)


def test_topk_packed_4bit_lut_matches_dense_scores_with_bias_and_scale():
    torch.manual_seed(37)
    codes = torch.randint(0, 16, (31, 5))
    packed = pack_4bit_codes(codes)
    lut = torch.randn(5, 16)
    bias = torch.randn(31)
    scale = torch.linspace(0.5, 2.0, steps=31)

    result = topk_packed_4bit_lut(
        packed,
        lut,
        num_subspaces=5,
        topk=7,
        candidate_budget=15,
        score_bias=bias,
        score_scale=scale,
    )
    dense_scores = (lut.gather(1, codes.T).sum(dim=0) + bias) * scale
    budget_scores, budget_indices = torch.topk(dense_scores, k=15)
    expected_scores, expected_order = torch.topk(budget_scores, k=7)
    expected_indices = budget_indices[expected_order]

    torch.testing.assert_close(result.indices, expected_indices)
    torch.testing.assert_close(result.scores, expected_scores)
    torch.testing.assert_close(result.candidate_indices, budget_indices)
    torch.testing.assert_close(result.candidate_scores, budget_scores)


def test_block_topk_packed_4bit_lut_matches_dense_scores():
    torch.manual_seed(24)
    codes = torch.randint(0, 16, (53, 6))
    packed = pack_4bit_codes(codes)
    lut = torch.randn(6, 16)
    bias = torch.randn(53)

    result = topk_packed_4bit_lut(
        packed,
        lut,
        num_subspaces=6,
        topk=8,
        candidate_budget=17,
        score_bias=bias,
        block_size=9,
    )
    dense_scores = lut.gather(1, codes.T).sum(dim=0) + bias
    budget_scores, budget_indices = torch.topk(dense_scores, k=17)
    expected_scores, expected_order = torch.topk(budget_scores, k=8)
    expected_indices = budget_indices[expected_order]

    torch.testing.assert_close(result.indices, expected_indices)
    torch.testing.assert_close(result.scores, expected_scores)
    torch.testing.assert_close(result.candidate_indices, budget_indices)
    torch.testing.assert_close(result.candidate_scores, budget_scores)


def test_block_topk_packed_4bit_lut_matches_dense_scores_with_scale():
    torch.manual_seed(38)
    codes = torch.randint(0, 16, (57, 6))
    packed = pack_4bit_codes(codes)
    lut = torch.randn(6, 16)
    bias = torch.randn(57)
    scale = torch.rand(57).clamp_min(0.1)

    result = topk_packed_4bit_lut(
        packed,
        lut,
        num_subspaces=6,
        topk=8,
        candidate_budget=19,
        score_bias=bias,
        score_scale=scale,
        block_size=11,
    )
    dense_scores = (lut.gather(1, codes.T).sum(dim=0) + bias) * scale
    budget_scores, budget_indices = torch.topk(dense_scores, k=19)
    expected_scores, expected_order = torch.topk(budget_scores, k=8)
    expected_indices = budget_indices[expected_order]

    torch.testing.assert_close(result.indices, expected_indices)
    torch.testing.assert_close(result.scores, expected_scores)
    torch.testing.assert_close(result.candidate_indices, budget_indices)
    torch.testing.assert_close(result.candidate_scores, budget_scores)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA and Triton are required",
)
def test_block_topk_packed_4bit_lut_triton_matches_dense_scores():
    torch.manual_seed(29)
    codes = torch.randint(0, 16, (71, 5), device="cuda")
    packed = pack_4bit_codes(codes)
    lut = torch.randn(5, 16, device="cuda")
    bias = torch.randn(71, device="cuda")

    result = topk_packed_4bit_lut(
        packed,
        lut,
        num_subspaces=5,
        topk=9,
        candidate_budget=21,
        score_bias=bias,
        block_size=17,
        backend="triton",
    )
    dense_scores = lut.gather(1, codes.T).sum(dim=0) + bias
    budget_scores, budget_indices = torch.topk(dense_scores, k=21)
    expected_scores, expected_order = torch.topk(budget_scores, k=9)
    expected_indices = budget_indices[expected_order]

    torch.testing.assert_close(result.indices, expected_indices)
    torch.testing.assert_close(result.scores, expected_scores, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.candidate_indices, budget_indices)
    torch.testing.assert_close(result.candidate_scores, budget_scores, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA and Triton are required",
)
def test_block_topk_packed_4bit_lut_triton_matches_dense_scores_with_scale():
    torch.manual_seed(39)
    codes = torch.randint(0, 16, (73, 5), device="cuda")
    packed = pack_4bit_codes(codes)
    lut = torch.randn(5, 16, device="cuda")
    bias = torch.randn(73, device="cuda")
    scale = torch.rand(73, device="cuda").clamp_min(0.1)

    result = topk_packed_4bit_lut(
        packed,
        lut,
        num_subspaces=5,
        topk=9,
        candidate_budget=23,
        score_bias=bias,
        score_scale=scale,
        block_size=17,
        backend="triton",
    )
    dense_scores = (lut.gather(1, codes.T).sum(dim=0) + bias) * scale
    budget_scores, budget_indices = torch.topk(dense_scores, k=23)
    expected_scores, expected_order = torch.topk(budget_scores, k=9)
    expected_indices = budget_indices[expected_order]

    torch.testing.assert_close(result.indices, expected_indices)
    torch.testing.assert_close(result.scores, expected_scores, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.candidate_indices, budget_indices)
    torch.testing.assert_close(result.candidate_scores, budget_scores, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA and Triton are required",
)
def test_final_topk_merge_triton_matches_dense_scores():
    torch.manual_seed(30)
    candidate_indices = torch.randperm(97, device="cuda")
    candidate_scores = torch.randn(97, device="cuda")

    top_indices, top_scores, budget_indices, budget_scores = final_topk_merge_triton(
        candidate_indices,
        candidate_scores,
        topk=11,
        candidate_budget=23,
    )
    expected_budget_scores, expected_budget_order = torch.topk(candidate_scores, k=23)
    expected_budget_indices = candidate_indices[expected_budget_order]
    expected_top_scores, expected_top_order = torch.topk(expected_budget_scores, k=11)
    expected_top_indices = expected_budget_indices[expected_top_order]

    torch.testing.assert_close(budget_indices, expected_budget_indices)
    torch.testing.assert_close(budget_scores, expected_budget_scores)
    torch.testing.assert_close(top_indices, expected_top_indices)
    torch.testing.assert_close(top_scores, expected_top_scores)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA and Triton are required",
)
def test_final_topk_merge_triton_hierarchical_matches_dense_scores():
    torch.manual_seed(32)
    candidate_indices = torch.randperm(5000, device="cuda")
    candidate_scores = torch.randn(5000, device="cuda")

    top_indices, top_scores, budget_indices, budget_scores = final_topk_merge_triton(
        candidate_indices,
        candidate_scores,
        topk=17,
        candidate_budget=97,
        max_merge_size=512,
    )
    expected_budget_scores, expected_budget_order = torch.topk(candidate_scores, k=97)
    expected_budget_indices = candidate_indices[expected_budget_order]
    expected_top_scores, expected_top_order = torch.topk(expected_budget_scores, k=17)
    expected_top_indices = expected_budget_indices[expected_top_order]

    torch.testing.assert_close(budget_indices, expected_budget_indices)
    torch.testing.assert_close(budget_scores, expected_budget_scores)
    torch.testing.assert_close(top_indices, expected_top_indices)
    torch.testing.assert_close(top_scores, expected_top_scores)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA and Triton are required",
)
def test_final_topk_merge_triton_hierarchical_no_budget_matches_dense_topk():
    torch.manual_seed(34)
    candidate_indices = torch.randperm(5000, device="cuda")
    candidate_scores = torch.randn(5000, device="cuda")

    top_indices, top_scores, budget_indices, budget_scores = final_topk_merge_triton(
        candidate_indices,
        candidate_scores,
        topk=19,
        candidate_budget=None,
        max_merge_size=512,
    )
    expected_top_scores, expected_top_order = torch.topk(candidate_scores, k=19)
    expected_top_indices = candidate_indices[expected_top_order]

    torch.testing.assert_close(top_indices, expected_top_indices)
    torch.testing.assert_close(top_scores, expected_top_scores)
    torch.testing.assert_close(budget_indices, candidate_indices)
    torch.testing.assert_close(budget_scores, candidate_scores)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA and Triton are required",
)
def test_topk_packed_4bit_lut_uses_triton_final_merge(monkeypatch):
    torch.manual_seed(31)
    codes = torch.randint(0, 16, (73, 5), device="cuda")
    packed = pack_4bit_codes(codes)
    lut = torch.randn(5, 16, device="cuda")
    calls = {"count": 0}
    original_merge = lut_scan_module.final_topk_merge_triton

    def counting_merge(*args, **kwargs):
        calls["count"] += 1
        return original_merge(*args, **kwargs)

    monkeypatch.setattr(lut_scan_module, "final_topk_merge_triton", counting_merge)

    result = topk_packed_4bit_lut(
        packed,
        lut,
        num_subspaces=5,
        topk=7,
        candidate_budget=19,
        block_size=13,
        backend="triton",
    )

    assert calls["count"] == 1
    assert result.indices.shape == (7,)
    assert result.candidate_indices.shape == (19,)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA and Triton are required",
)
def test_topk_packed_4bit_lut_large_triton_merge_avoids_torch_fallback(monkeypatch):
    torch.manual_seed(33)
    codes = torch.randint(0, 16, (9001, 4), device="cuda")
    packed = pack_4bit_codes(codes)
    lut = torch.randn(4, 16, device="cuda")

    def fail_fallback(*_args, **_kwargs):
        raise AssertionError("large Triton merge should not fall back to PyTorch")

    monkeypatch.setattr(lut_scan_module, "_finalize_candidate_topk", fail_fallback)

    result = topk_packed_4bit_lut(
        packed,
        lut,
        num_subspaces=4,
        topk=11,
        candidate_budget=64,
        block_size=32,
        backend="triton",
    )
    dense_scores = lut.gather(1, codes.T).sum(dim=0)
    budget_scores, budget_indices = torch.topk(dense_scores, k=64)
    expected_scores, expected_order = torch.topk(budget_scores, k=11)
    expected_indices = budget_indices[expected_order]

    torch.testing.assert_close(result.indices, expected_indices)
    torch.testing.assert_close(result.scores, expected_scores, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.candidate_indices, budget_indices)
    torch.testing.assert_close(result.candidate_scores, budget_scores, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA and Triton are required",
)
def test_topk_packed_4bit_lut_large_triton_no_budget_avoids_torch_fallback(monkeypatch):
    torch.manual_seed(35)
    codes = torch.randint(0, 16, (9001, 4), device="cuda")
    packed = pack_4bit_codes(codes)
    lut = torch.randn(4, 16, device="cuda")

    def fail_fallback(*_args, **_kwargs):
        raise AssertionError("large no-budget Triton merge should not fall back to PyTorch")

    monkeypatch.setattr(lut_scan_module, "_finalize_candidate_topk", fail_fallback)

    result = topk_packed_4bit_lut(
        packed,
        lut,
        num_subspaces=4,
        topk=11,
        candidate_budget=None,
        block_size=32,
        backend="triton",
    )
    dense_scores = lut.gather(1, codes.T).sum(dim=0)
    expected_scores, _ = torch.topk(dense_scores, k=11)

    torch.testing.assert_close(result.scores, expected_scores, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dense_scores[result.indices], result.scores, atol=1e-5, rtol=1e-5)
    assert result.indices.unique().numel() == result.indices.numel()
    assert result.candidate_indices.numel() == codes.shape[0]
    assert result.candidate_scores.numel() == codes.shape[0]


def test_ivfpq_search_uses_packed_score_backend(monkeypatch):
    torch.manual_seed(22)
    keys = torch.randn(128, 32)
    query = torch.randn(32)
    calls = {"count": 0}
    captured = {}
    original_score = ivfpq_module.score_packed_4bit_lut

    def counting_score(*args, **kwargs):
        calls["count"] += 1
        captured.update(kwargs)
        return original_score(*args, **kwargs)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=5,
            pq_max_iter=5,
            topk_block_size=11,
            seed=23,
        )
    ).build(keys)
    monkeypatch.setattr(ivfpq_module, "score_packed_4bit_lut", counting_score)

    result = index.search(query, topk=9, candidate_budget=17)

    assert calls["count"] == 1
    assert captured["block_size"] == 11
    assert result.indices.shape == (9,)
    assert result.candidate_indices.numel() <= 17
    assert torch.all(result.approx_scores[:-1] >= result.approx_scores[1:])


def test_ivfpq_direction_normalized_search_uses_packed_score_backend(monkeypatch):
    torch.manual_seed(40)
    keys = torch.randn(128, 32)
    query = torch.randn(32)
    calls = {"count": 0}
    captured = {}
    original_score = ivfpq_module.score_packed_4bit_lut

    def counting_score(*args, **kwargs):
        calls["count"] += 1
        captured.update(kwargs)
        return original_score(*args, **kwargs)

    index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=5,
            pq_max_iter=5,
            topk_block_size=11,
            rotation="random_orthogonal",
            direction_normalize=True,
            seed=41,
        )
    ).build(keys)
    monkeypatch.setattr(ivfpq_module, "score_packed_4bit_lut", counting_score)

    result = index.search(query, topk=9, candidate_budget=17)

    assert calls["count"] == 1
    assert captured["block_size"] == 11
    assert result.indices.shape == (9,)
    assert result.candidate_indices.numel() <= 17
    assert torch.all(result.approx_scores[:-1] >= result.approx_scores[1:])


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="CUDA and Triton are required",
)
def test_ivfpq_search_supports_triton_backend():
    torch.manual_seed(27)
    keys = torch.randn(128, 32, device="cuda")
    query = torch.randn(32, device="cuda")

    triton_index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            topk_block_size=16,
            kernel_backend="triton",
            seed=28,
        )
    ).build(keys)
    torch_index = IVFPQIndex(
        IVFPQConfig(
            num_lists=8,
            nprobe=4,
            num_subspaces=4,
            num_bits=4,
            coarse_max_iter=4,
            pq_max_iter=4,
            topk_block_size=16,
            kernel_backend="torch",
            seed=28,
        )
    ).build(keys)

    triton_result = triton_index.search(query, topk=10, candidate_budget=24)
    torch_result = torch_index.search(query, topk=10, candidate_budget=24)

    torch.testing.assert_close(triton_result.indices, torch_result.indices)
    torch.testing.assert_close(triton_result.approx_scores, torch_result.approx_scores, atol=1e-5, rtol=1e-5)
