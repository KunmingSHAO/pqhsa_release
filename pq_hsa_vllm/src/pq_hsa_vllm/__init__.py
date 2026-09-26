"""PQ-HSA vLLM plugin.

Packages ``benchmarks/vllm_backend`` as an installable, opt-in vLLM general
plugin (vLLM 0.8.5.post1 and 0.29.0). No vLLM source edits.

Quick start::

    pip install -e ./pq_hsa_vllm        # into your vLLM environment, from the repo root
    PQ_HSA_VLLM=1 python your_script_that_calls_LLM.py

See README.md for configuration (``PQHSAConfig`` / ``PQ_HSA_*`` env), the
TP>1 ``--worker-cls pq_hsa_vllm.PQHSAWorker`` path, and limits.

Imports are lazy (mirrors ``benchmarks/vllm_backend/__init__.py``) so
``import pq_hsa_vllm`` alone -- e.g. for the plugin-discovery CPU self-test --
never touches torch/vllm/CUDA.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "PQHSAConfig",
    "PQHSAWorker",
    "vllm_general_plugin",
    "install",
    "enabled",
]


def __getattr__(name: str) -> Any:
    if name == "PQHSAConfig":
        from pq_hsa_vllm.config import PQHSAConfig

        return PQHSAConfig
    if name == "PQHSAWorker":
        from pq_hsa_vllm.worker import PQHSAWorker

        return PQHSAWorker
    if name in {"vllm_general_plugin", "install", "enabled"}:
        from pq_hsa_vllm import plugin

        return getattr(plugin, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
