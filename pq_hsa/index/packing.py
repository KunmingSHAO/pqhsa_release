from __future__ import annotations

import torch


def pack_4bit_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack ``[N, M]`` 4-bit PQ codes into uint8 nibbles.

    Even subspace codes use the low nibble; odd subspace codes use the high
    nibble. If ``M`` is odd, the final high nibble is padded with zero.
    """

    if codes.ndim != 2:
        raise ValueError(f"codes must be [N, M], got {tuple(codes.shape)}")
    if codes.numel() > 0 and (codes.min() < 0 or codes.max() > 15):
        raise ValueError("4-bit packing requires codes in [0, 15]")

    codes_u8 = codes.to(torch.uint8)
    num_vectors, num_subspaces = codes_u8.shape
    packed_width = (num_subspaces + 1) // 2
    packed = torch.zeros(
        num_vectors,
        packed_width,
        device=codes_u8.device,
        dtype=torch.uint8,
    )
    packed[:, torch.arange(0, num_subspaces, 2, device=codes_u8.device) // 2] = (
        codes_u8[:, 0::2] & 0x0F
    )
    if num_subspaces > 1:
        packed[:, torch.arange(1, num_subspaces, 2, device=codes_u8.device) // 2] |= (
            codes_u8[:, 1::2] << 4
        )
    return packed


def unpack_4bit_codes(packed: torch.Tensor, num_subspaces: int) -> torch.Tensor:
    """Unpack uint8 nibble storage back to ``[N, M]`` int64 PQ codes."""

    if packed.ndim != 2:
        raise ValueError(f"packed codes must be [N, ceil(M/2)], got {tuple(packed.shape)}")
    if num_subspaces <= 0:
        raise ValueError("num_subspaces must be positive")
    expected_width = (num_subspaces + 1) // 2
    if packed.shape[1] != expected_width:
        raise ValueError(
            f"packed width must be {expected_width} for {num_subspaces} subspaces, "
            f"got {packed.shape[1]}"
        )

    unpacked = torch.empty(
        packed.shape[0],
        num_subspaces,
        device=packed.device,
        dtype=torch.long,
    )
    unpacked[:, 0::2] = (packed[:, : ((num_subspaces + 1) // 2)] & 0x0F).to(torch.long)
    if num_subspaces > 1:
        unpacked[:, 1::2] = (packed[:, : (num_subspaces // 2)] >> 4).to(torch.long)
    return unpacked
