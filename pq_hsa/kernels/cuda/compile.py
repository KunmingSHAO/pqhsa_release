"""JIT-compile PQ-HSA CUDA extensions with a persistent /tmp cache."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

_DIR = Path(__file__).resolve().parent
_CACHE = Path(os.environ.get("PQ_HSA_CUDA_EXT_DIR", "/tmp/torch_extensions"))


def compile_extension(name: str, sources: Sequence[str], extra_cuda: list[str] | None = None):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    os.environ.setdefault("CUDA_HOME", os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    from torch.utils.cpp_extension import load

    srcs = [str(_DIR / s) if not os.path.isabs(s) else s for s in sources]
    build_dir = _CACHE / name
    build_dir.mkdir(parents=True, exist_ok=True)
    flags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
    ]
    if extra_cuda:
        flags.extend(extra_cuda)
    return load(
        name=name,
        sources=srcs,
        extra_cuda_cflags=flags,
        extra_cflags=["-O3"],
        build_directory=str(build_dir),
        verbose=os.environ.get("PQ_HSA_CUDA_VERBOSE", "0") == "1",
        with_cuda=True,
    )
