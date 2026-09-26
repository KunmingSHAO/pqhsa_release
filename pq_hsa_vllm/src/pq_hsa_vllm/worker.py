"""``PQHSAWorker`` -- explicit ``--worker-cls`` opt-in.

Use when you would rather not rely on the ``vllm.general_plugins`` /
``VLLM_PLUGINS`` auto-load path (e.g. TP>1 launchers that already pass
``worker_cls`` explicitly, or when you want PQ-HSA on regardless of
``PQ_HSA_VLLM``). Pass ``worker_cls="pq_hsa_vllm.PQHSAWorker"`` (V1) to
``LLM(...)`` / ``vllm serve``. Requires vLLM 0.8.5.post1's V1
``vllm.v1.worker.gpu_worker.Worker`` (see README "Limits").

Importing this module imports vLLM's V1 GPU worker, so it is only imported
when a caller actually asks for the class (via ``resolve_obj_by_qualname`` or
an explicit ``from pq_hsa_vllm.worker import PQHSAWorker``) -- see the lazy
``__getattr__`` in ``pq_hsa_vllm/__init__.py``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from vllm.v1.worker.gpu_worker import Worker

from pq_hsa_vllm.plugin import install

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput


class PQHSAWorker(Worker):
    """Installs PQ-HSA on every TP rank. Choosing this class is the opt-in:
    ``init_device`` installs unconditionally (``force=True``), independent of
    ``PQ_HSA_VLLM``. Set ``PQ_HSA_ENABLE=0`` explicitly to keep the backend
    registered but stay on the dense path (e.g. for an in-process A/B)."""

    def init_device(self):
        install(rank=self.rank, force=True)
        super().init_device()

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional["ModelRunnerOutput"]:
        from benchmarks.vllm_backend.worker_hook import dump_rank_stats

        try:
            return super().execute_model(scheduler_output)
        finally:
            try:
                dump_rank_stats(self.rank, reason="execute_model")
            except Exception:
                pass

    def pq_hsa_set_enable(self, on: bool) -> str:
        os.environ["PQ_HSA_ENABLE"] = "1" if on else "0"
        return os.environ["PQ_HSA_ENABLE"]

    def pq_hsa_runtime_stats(self) -> dict:
        from benchmarks.vllm_backend.pq_hsa_decode_runtime import pq_hsa_runtime_stats

        return {
            "rank": int(self.rank),
            "pid": os.getpid(),
            "pq_hsa_enable": os.environ.get("PQ_HSA_ENABLE", "0"),
            **pq_hsa_runtime_stats(),
        }


__all__ = ["PQHSAWorker"]
