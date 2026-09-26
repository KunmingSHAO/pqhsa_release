"""PQ-HSA vLLM plugin configuration.

Single place that documents and applies the ``PQ_HSA_*`` env surface that
``benchmarks/vllm_backend`` already reads directly via ``os.environ.get`` at
call time. This module does **not** change any of
those call sites or their defaults -- it only decides *what gets written into
os.environ* before the first decode call, so the numeric/kernel code in
``pq_hsa/`` and ``benchmarks/vllm_backend`` is byte-for-byte unchanged.

Two layers:

* ``PQHSAConfig`` -- the small set of knobs a user is expected to tune
  (enable, budget ``p`` == retrieval top-fraction, flush interval, sink/local,
  sidecar backend, batch mode, native backend, GQA multi-group, paged KV).
* ``ADOPTED_KERNEL_ENV`` -- the frozen CUDA graph-body kernel configuration
  (19 switches) plus ``LONG_CONTEXT_ADDITIONS`` (GQA instances, paged KV).
  This is the optimized configuration used for the speed measurements; the
  plugin applies it with ``setdefault`` so an explicit env value always wins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

# Optimized CUDA graph-body configuration ("p3r" variant). 19 knobs, all
# opt-in switches with a safe fallback when unset (0).
ADOPTED_KERNEL_ENV: dict[str, str] = {
    "PQ_HSA_CUDA_ATTEND_ONLY": "1",
    "PQ_HSA_CUDA_ATTEND_SORT": "0",
    "PQ_HSA_CUDA_SPLITS": "8",
    "PQ_HSA_CUDA_PARTWARPS": "8",
    "PQ_HSA_CUDA_PARTUNROLL": "8",
    "PQ_HSA_FP16_LUT": "1",
    "PQ_HSA_CUDA_REDWARPS": "4",
    "PQ_HSA_CUDA_BGSPLITS": "8",
    "PQ_HSA_CUDA_BGWARPS": "8",
    "PQ_HSA_CUDA_BGUNROLL": "8",
    "PQ_HSA_CUDA_BGMM": "0",
    "PQ_HSA_LEAN_APPEND": "1",
    "PQ_HSA_LEAN_COPYCTX": "1",
    "PQ_HSA_PARSTREAM": "1",
    "PQ_HSA_FUSED_LUTPREP": "1",
    "PQ_HSA_FUSED_BLOCKRED": "1",
    "PQ_HSA_ATTEND_RAWIDX": "1",
    "PQ_HSA_MASKSKIP": "1",
    "PQ_HSA_LEAN_REPLAY": "1",
}

# GQA group counts other than the CUDA kernel's compile-time default (kG=4)
# silently fall back to aten unless this is set; long-context paged KV attend
# needs the paged variants. All three are opt-in / no-op when the shape
# doesn't match, so defaulting them on is safe for models with
# G in {4, 5, 8, 16} (see pq_hsa/kernels/cuda/pq_fused_h20.py _G_INSTANCES).
LONG_CONTEXT_ADDITIONS: dict[str, str] = {
    "PQ_HSA_CUDA_GQA_MULTI": "1",
    "PQ_HSA_PAGED_ATTEND": "1",
    "PQ_HSA_PAGED_RESTORE": "1",
}

# Correctness + performance baseline switches the optimized stack was measured
# on top of (minus the launch-time-only vars -- VLLM_ENABLE_V1_MULTIPROCESSING /
# VLLM_ATTENTION_BACKEND cannot be set from inside a general_plugins hook, see
# the top-level README).
BASE_STACK_ENV: dict[str, str] = {
    "PQ_HSA_SIDECAR": "fast",
    "PQ_USE_CUDA_GRAPH": "1",
    "PQ_CG_SENTINEL_INTERVAL": "0",
    "PQ_HSA_FUSED_DECODE": "0",
    "PQ_HSA_CUDA_FUSED": "0",
    "PQ_HSA_HOST_SYNC_FIX": "1",
    "PQ_HSA_INCREMENTAL_FLUSH": "1",
    "PQ_HSA_BLOCK_TOPK": "0",
    "PQ_HSA_FLUSH_NO_RECAPTURE": "0",
    "PQ_HSA_LEAN_WRAP": "0",
    "PQ_HSA_GRAPH_OUT_DIRECT": "0",
    "PQ_HSA_GRAPH_APPEND": "0",
    "PQ_HSA_FUSED_RADIX": "0",
    "PQ_HSA_FUSED_FULL": "0",
    "PQ_HSA_SCAN_TOPK": "0",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw == "1"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class PQHSAConfig:
    """The user-facing knob set. Mirrors ``benchmarks/vllm_backend`` defaults
    exactly (see ``pq_hsa_decode_runtime.paper_attention_config`` /
    ``paper_index_config``), except ``gqa_multi``/``paged_attend``/
    ``paged_restore`` which default True here (they default False in the
    bare benchmark scripts, which pin them explicitly per run).
    """

    # Master switches.
    enable: bool = True                 # PQ_HSA_ENABLE
    native_backend: bool = False        # PQ_HSA_NATIVE_BACKEND
    batch: bool = False                 # PQ_HSA_BATCH (num_reqs > 1)
    sidecar: str = "fast"               # PQ_HSA_SIDECAR (fast|loop)
    fallback: bool = False              # PQ_HSA_FALLBACK (silent dense fallback on error)

    # Budget "p" (retrieval top-fraction) and window.
    top_fraction: float = 0.01          # PQ_HSA_TOP_FRAC
    candidate_budget: int = 4096        # PQ_HSA_CAND_BUDGET
    sink_tokens: int = 4                # PQ_HSA_SINK
    local_window: int = 128             # PQ_HSA_LOCAL
    nprobe: int = 64                    # PQ_HSA_NPROBE

    # Index.
    num_lists: int = 512                # PQ_HSA_NUM_LISTS
    num_subspaces: int = 8              # PQ_HSA_SUBSPACES
    num_bits: int = 4                   # PQ_HSA_BITS

    # Deferred-index amortization.
    flush_interval: int = 256           # PQ_HSA_FLUSH_INTERVAL
    update_interval: int = 256          # PQ_HSA_UPD_INTERVAL

    # GQA / long-context (adopted defaults; see class docstring).
    gqa_multi: bool = True              # PQ_HSA_CUDA_GQA_MULTI
    paged_attend: bool = True           # PQ_HSA_PAGED_ATTEND
    paged_restore: bool = True          # PQ_HSA_PAGED_RESTORE

    # Whether to also stamp the frozen CUDA graph-body kernel configuration.
    apply_adopted_kernel_stack: bool = True

    extra_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "PQHSAConfig":
        # The PQ_HSA_BATCH=1 sidecar (pq_hsa_batch_decode.py::run_pq_decode_one)
        # calls set_paged_kv_context() at both call sites, mirroring the
        # single-request path, so paged_attend/paged_restore default on
        # unconditionally here too; batch=1 does not get a different default.
        batch = _env_bool("PQ_HSA_BATCH", False)
        return cls(
            enable=_env_bool("PQ_HSA_ENABLE", True),
            native_backend=_env_bool("PQ_HSA_NATIVE_BACKEND", False),
            batch=batch,
            sidecar=_env_str("PQ_HSA_SIDECAR", "fast"),
            fallback=_env_bool("PQ_HSA_FALLBACK", False),
            top_fraction=_env_float("PQ_HSA_TOP_FRAC", 0.01),
            candidate_budget=_env_int("PQ_HSA_CAND_BUDGET", 4096),
            sink_tokens=_env_int("PQ_HSA_SINK", 4),
            local_window=_env_int("PQ_HSA_LOCAL", 128),
            nprobe=_env_int("PQ_HSA_NPROBE", 64),
            num_lists=_env_int("PQ_HSA_NUM_LISTS", 512),
            num_subspaces=_env_int("PQ_HSA_SUBSPACES", 8),
            num_bits=_env_int("PQ_HSA_BITS", 4),
            flush_interval=_env_int("PQ_HSA_FLUSH_INTERVAL", 256),
            update_interval=_env_int("PQ_HSA_UPD_INTERVAL", 256),
            gqa_multi=_env_bool("PQ_HSA_CUDA_GQA_MULTI", True),
            paged_attend=_env_bool("PQ_HSA_PAGED_ATTEND", True),
            paged_restore=_env_bool("PQ_HSA_PAGED_RESTORE", True),
            apply_adopted_kernel_stack=_env_bool("PQ_HSA_ADOPT_KERNEL_STACK", True),
        )

    def as_env(self) -> dict[str, str]:
        """The env this config maps onto. Order-independent, setdefault-applied."""
        out: dict[str, str] = dict(BASE_STACK_ENV)
        if self.apply_adopted_kernel_stack:
            out.update(ADOPTED_KERNEL_ENV)
        out.update(
            {
                "PQ_HSA_ENABLE": "1" if self.enable else "0",
                "PQ_HSA_NATIVE_BACKEND": "1" if self.native_backend else "0",
                "PQ_HSA_BATCH": "1" if self.batch else "0",
                "PQ_HSA_SIDECAR": self.sidecar,
                "PQ_HSA_FALLBACK": "1" if self.fallback else "0",
                "PQ_HSA_TOP_FRAC": str(self.top_fraction),
                "PQ_HSA_CAND_BUDGET": str(self.candidate_budget),
                "PQ_HSA_SINK": str(self.sink_tokens),
                "PQ_HSA_LOCAL": str(self.local_window),
                "PQ_HSA_NPROBE": str(self.nprobe),
                "PQ_HSA_NUM_LISTS": str(self.num_lists),
                "PQ_HSA_SUBSPACES": str(self.num_subspaces),
                "PQ_HSA_BITS": str(self.num_bits),
                "PQ_HSA_FLUSH_INTERVAL": str(self.flush_interval),
                "PQ_HSA_UPD_INTERVAL": str(self.update_interval),
                "PQ_HSA_CUDA_GQA_MULTI": "1" if self.gqa_multi else "0",
                "PQ_HSA_PAGED_ATTEND": "1" if self.paged_attend else "0",
                "PQ_HSA_PAGED_RESTORE": "1" if self.paged_restore else "0",
            }
        )
        out.update(self.extra_env)
        return out

    def apply_env_defaults(self) -> dict[str, str]:
        """``os.environ.setdefault`` every key in :meth:`as_env`.

        An env var the user (or the launch script) already set always wins --
        this only fills gaps, it never overwrites. Returns the keys actually
        written (i.e. that were previously unset), for logging/stats.
        """
        written: dict[str, str] = {}
        for k, v in self.as_env().items():
            if k not in os.environ:
                os.environ[k] = v
                written[k] = v
        return written

    def asdict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


__all__ = [
    "PQHSAConfig",
    "ADOPTED_KERNEL_ENV",
    "LONG_CONTEXT_ADDITIONS",
    "BASE_STACK_ENV",
]
