"""``vllm.general_plugins`` entry point + installer for PQ-HSA.

No vLLM source edits. Two ways to turn this on:

1. ``PQ_HSA_VLLM=1`` in the process env before ``LLM(...)`` / the vLLM
   server starts. vLLM auto-loads every registered ``vllm.general_plugins``
   entry point once per process (driver and, for TP>1, every worker
   subprocess -- see ``vllm.worker.worker_base.WorkerWrapperBase.init_worker``,
   which calls ``load_general_plugins()`` before resolving ``worker_cls``).
   :func:`vllm_general_plugin` is that entry point; it is a no-op unless
   ``PQ_HSA_VLLM=1`` is set, so merely having the package importable /
   discoverable changes nothing (CPU self-test in ``tests/``).
2. ``--worker-cls pq_hsa_vllm.PQHSAWorker`` (see ``worker.py``). Choosing the
   worker class is itself the opt-in, so ``PQHSAWorker.init_device`` installs
   unconditionally.

Both paths delegate the actual monkey-patch to
``benchmarks.vllm_backend``, which is the code this plugin packages --
nothing here reimplements the decode runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from pq_hsa_vllm.config import PQHSAConfig

_INSTALLED = False


def _repo_root() -> Path:
    """Directory containing both ``pq_hsa/`` and ``benchmarks/``.

    This package is ``pip install -e``'d from inside the repository
    (``pq_hsa_vllm/`` at its top level), so ``../..`` from this file's real
    location (editable installs keep the source path) is the repo root. Set
    ``PQ_HSA_REPO_ROOT`` to override (e.g. for a non-editable install). We
    do not vendor ``benchmarks/vllm_backend`` -- the plugin packages it, it
    does not fork it.
    """
    override = os.environ.get("PQ_HSA_REPO_ROOT")
    if override:
        return Path(override).resolve()
    # src/pq_hsa_vllm/plugin.py -> src/pq_hsa_vllm -> src -> pq_hsa_vllm -> REPO
    return Path(__file__).resolve().parents[3]


def _ensure_repo_importable() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def enabled() -> bool:
    return os.environ.get("PQ_HSA_VLLM", "0") == "1"


def install(rank: Optional[int] = None, *, force: bool = False, config: Optional[PQHSAConfig] = None) -> dict[str, Any]:
    """Apply :class:`PQHSAConfig` defaults and install the decode backend.

    Idempotent per process (subsequent calls are cheap no-ops once the
    backend forward is wrapped -- ``install_pq_hsa_flashattn_backend`` /
    ``install_pq_hsa_vllm_backend`` already guard on that).  Safe to call
    from multiple entry points (general plugin + worker_cls) in the same
    process.
    """
    global _INSTALLED
    if not force and not enabled():
        return {"installed": False, "reason": "PQ_HSA_VLLM!=1"}

    _ensure_repo_importable()

    cfg = config or PQHSAConfig.from_env()
    written = cfg.apply_env_defaults()

    from benchmarks.vllm_backend.worker_hook import install_in_process

    result = install_in_process(rank=rank)
    result["config"] = cfg.asdict()
    result["env_defaults_written"] = sorted(written)
    _INSTALLED = True
    return result


def vllm_general_plugin() -> None:
    """The ``vllm.general_plugins`` entry point (registered in pyproject.toml
    as ``pq_hsa_vllm = "pq_hsa_vllm.plugin:vllm_general_plugin"``).

    vLLM calls this once per process regardless of whether the user wants
    PQ-HSA -- see module docstring. Must be a no-op unless PQ_HSA_VLLM=1.
    """
    if not enabled():
        return
    install()


__all__ = ["vllm_general_plugin", "install", "enabled", "PQHSAConfig"]
