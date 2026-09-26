import pytest
import torch

from pq_hsa import pack_4bit_codes, unpack_4bit_codes


def test_pack_unpack_4bit_codes_even_subspaces():
    codes = torch.tensor(
        [
            [0, 1, 2, 3],
            [15, 14, 13, 12],
        ],
        dtype=torch.long,
    )

    packed = pack_4bit_codes(codes)

    assert packed.dtype == torch.uint8
    assert packed.shape == (2, 2)
    torch.testing.assert_close(unpack_4bit_codes(packed, 4), codes)


def test_pack_unpack_4bit_codes_odd_subspaces():
    codes = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=torch.long,
    )

    packed = pack_4bit_codes(codes)

    assert packed.shape == (2, 2)
    assert torch.all((packed[:, -1] >> 4) == 0)
    torch.testing.assert_close(unpack_4bit_codes(packed, 3), codes)


def test_pack_4bit_codes_rejects_out_of_range_codes():
    codes = torch.tensor([[0, 16]], dtype=torch.long)

    with pytest.raises(ValueError, match="\\[0, 15\\]"):
        pack_4bit_codes(codes)
